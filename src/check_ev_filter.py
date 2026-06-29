# coding: utf-8
"""
的中率モデル × EV フィルター バックテスト
  - prob_gap = 1位確率 - 2位確率（モデル確信度）
  - ev       = P_calib × 単勝オッズ（期待収益率）
  - 2次元グリッドで 2323 OOS をバックテスト
  - 最良閾値を 2025 / 2026 で検証
"""
import sys, os, pickle
import numpy as np
import pandas as pd
from itertools import product

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from save_conditional_logit import prepare, segment_softmax, BASE_DIR, DATA_FILE
from save_v3 import add_computed_features

# ── モデルロード ────────────────────────────────────────────────────────────
with open(os.path.join(BASE_DIR, 'models', 'hitrate_model.pkl'), 'rb') as f:
    MODEL = pickle.load(f)

# ── データロード ────────────────────────────────────────────────────────────
print("データ読み込み中...", flush=True)
df = pd.read_parquet(DATA_FILE)
df['日付_num'] = pd.to_numeric(df['日付'], errors='coerce')
df['着順_num']  = pd.to_numeric(df['着順_num'], errors='coerce')
df = df.dropna(subset=['日付_num', '着順_num'])
df = df[df['着順_num'] < 99]
df['race_id'] = (df['日付_num'].astype(int).astype(str) + '_' +
                 df['開催'].astype(str).str.strip() + '_' + df['Ｒ'].astype(str).str.strip())
df = df[df['開催'].notna()].copy()
df['surface'] = df['距離'].astype(str).str.strip().str.extract(r'^([芝ダ])')[0].fillna('不明')
dm = pd.to_numeric(df['距離'].astype(str).str.extract(r'(\d+)')[0], errors='coerce')
df['クラス_rank'] = pd.to_numeric(df['クラス_rank'], errors='coerce')
df = add_computed_features(df)
if '今回_会場' in df.columns and '1走前_開催' in df.columns:
    df['輸送有無'] = (df['今回_会場'].astype(str) != df['1走前_開催'].astype(str).str[1]).astype(float)
    df.loc[df['1走前_開催'].isna(), '輸送有無'] = float('nan')
baba_map = {'良':0,'稍重':1,'重':2,'不良':3}
for col in df.columns:
    if '馬場状態' in col and col != '馬場状態':
        df[col] = df[col].map(baba_map)
df['dist_m'] = dm
df['単勝オッズ_num'] = pd.to_numeric(df['単勝オッズ'], errors='coerce')

s = df['surface']; r = df['クラス_rank']; d = df['dist_m']
SEG_MASKS = {
    'ダ長': (s=='ダ')&(d>1400) &(r!=1.0),
    'ダ短': (s=='ダ')&(d<=1400)&(r!=1.0),
    '芝短': (s=='芝')&(d<=1400)&(r!=1.0),
    '芝中': (s=='芝')&(d>1400) &(d<=2000)&(r!=1.0),
    '芝長': (s=='芝')&(d>2000) &(r!=1.0),
}

# ── 全セグメントで確率・EV を計算 ──────────────────────────────────────────
print("確率・EV 計算中...", flush=True)
all_rows = []

for seg in ['芝長','芝中','芝短','ダ長','ダ短']:
    pkg = MODEL[seg]
    feat_cols, scaler, coef, iso = pkg['feat_cols'], pkg['scaler'], pkg['coef'], pkg['isotonic']

    seg_df = df[SEG_MASKS[seg]].copy()
    if len(seg_df) == 0:
        continue

    # NaN指示変数
    for fc in feat_cols:
        if fc.endswith('_isnan'):
            base = fc[:-6]
            if base in seg_df.columns and fc not in seg_df.columns:
                seg_df[fc] = seg_df[base].isna().astype(float)

    try:
        X, _, gs, n, *_ = prepare(seg_df, feat_cols, scaler=scaler, top_idx=None, top_idx3=None)
    except Exception as e:
        print(f"  {seg} prepare error: {e}")
        continue

    ss = seg_df.sort_values('race_id').reset_index(drop=True)
    raw_prob = segment_softmax(X @ coef, gs, n)
    ss['prob_raw'] = raw_prob

    # isotonic calibration
    ss['prob_calib'] = iso.predict(raw_prob)

    # レース内ランク
    ss['rank_pred'] = ss.groupby('race_id')['prob_raw'].rank(ascending=False, method='first')

    # 各レースの1位・2位確率
    rank1 = ss[ss['rank_pred']==1].set_index('race_id')[['prob_raw','prob_calib','単勝オッズ_num','着順_num']]
    rank2 = ss[ss['rank_pred']==2].set_index('race_id')[['prob_raw']].rename(columns={'prob_raw':'prob2'})
    merged = rank1.join(rank2, how='left')
    merged['prob_gap'] = merged['prob_raw'] - merged['prob2'].fillna(0)
    merged['ev']       = merged['prob_calib'] * merged['単勝オッズ_num']
    merged['hit']      = (merged['着順_num'] == 1).astype(int)
    merged['seg']      = seg

    # 日付_num をつける（分割用）
    date_map = ss.drop_duplicates('race_id').set_index('race_id')['日付_num']
    merged['日付_num'] = merged.index.map(date_map)
    merged = merged.reset_index()

    all_rows.append(merged)

all_df = pd.concat(all_rows, ignore_index=True)
all_df = all_df.dropna(subset=['ev','prob_gap','着順_num','単勝オッズ_num'])
print(f"全レース: {len(all_df)}R  (EV・gap計算済み)")

# ── 期間分割 ─────────────────────────────────────────────────────────────────
oos2324 = all_df[(all_df['日付_num']>=230101)&(all_df['日付_num']<250101)]
oos2025 = all_df[(all_df['日付_num']>=250101)&(all_df['日付_num']<260101)]
oos2026 = all_df[all_df['日付_num']>=260101]

def eval_filter(data, gap_th, ev_th):
    """フィルタ後の 的中率・ROI・R数"""
    f = data[(data['prob_gap'] >= gap_th) & (data['ev'] >= ev_th)]
    nr = len(f)
    if nr == 0:
        return 0, 0.0, 0.0
    acc = f['hit'].mean()
    roi = (f[f['hit']==1]['単勝オッズ_num'].sum() - nr) / nr
    return nr, acc, roi

# ── グリッドサーチ（2323 OOS） ────────────────────────────────────────────────
GAP_THS = [0.00, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]
EV_THS  = [0.00, 0.50, 0.70, 0.80, 0.90, 1.00, 1.10, 1.20]

print(f"\n── 2323 OOS グリッドサーチ ({len(oos2324)}R) ──")
print(f"{'gap≥':>6} {'ev≥':>5} {'R数':>5} {'的中率':>8} {'ROI':>8}")
print("-"*38)

best_roi = -99; best_params = (0, 0)
results_grid = []

# フィルタなしベースライン
nr0, acc0, roi0 = eval_filter(oos2324, 0, 0)
print(f"  {'全買い':>6} {'---':>5} {nr0:>5} {acc0:>7.1%} {roi0:>+7.1%}  (ベースライン)")

for gap_th, ev_th in product(GAP_THS, EV_THS):
    if gap_th == 0 and ev_th == 0:
        continue
    nr, acc, roi = eval_filter(oos2324, gap_th, ev_th)
    if nr < 50:  # サンプル少なすぎはスキップ
        continue
    results_grid.append((gap_th, ev_th, nr, acc, roi))
    if roi > best_roi:
        best_roi = roi; best_params = (gap_th, ev_th)

# ROI上位10件表示
results_grid.sort(key=lambda x: -x[4])
for gap_th, ev_th, nr, acc, roi in results_grid[:15]:
    mark = ' ← best' if (gap_th, ev_th) == best_params else ''
    print(f"  {gap_th:>5.2f} {ev_th:>5.2f} {nr:>5} {acc:>7.1%} {roi:>+7.1%}{mark}")

# ── best パラメータで 2025・2026 検証 ────────────────────────────────────────
print(f"\n── ベスト閾値（2323最適）で検証 gap≥{best_params[0]:.2f} / ev≥{best_params[1]:.2f} ──")
for label, data in [('2323 OOS', oos2324), ('2025 OOS', oos2025), ('2026 OOS', oos2026)]:
    nr, acc, roi = eval_filter(data, *best_params)
    nr0b, acc0b, roi0b = eval_filter(data, 0, 0)
    print(f"  {label}: {nr}/{nr0b}R  的中率={acc:.1%}(全:{acc0b:.1%}) ROI={roi:+.1%}(全:{roi0b:+.1%})")

# ── セグメント別（ベストパラメータ） ─────────────────────────────────────────
print(f"\n── セグメント別（gap≥{best_params[0]:.2f} / ev≥{best_params[1]:.2f}） ──")
for seg in ['芝長','芝中','芝短','ダ長','ダ短']:
    seg_d = all_df[(all_df['seg']==seg)&(all_df['日付_num']>=230101)]
    nr, acc, roi = eval_filter(seg_d, *best_params)
    nr0b, acc0b, roi0b = eval_filter(seg_d, 0, 0)
    print(f"  {seg}: {nr}/{nr0b}R  的中率={acc:.1%}(全:{acc0b:.1%}) ROI={roi:+.1%}(全:{roi0b:+.1%})")

# ── EV 分布確認 ───────────────────────────────────────────────────────────────
print(f"\n── EV 分布（2323 OOS 1位候補馬） ──")
print(oos2324['ev'].describe().round(3).to_string())
print(f"\nEV>1.0 の割合: {(oos2324['ev']>1.0).mean():.1%}  ({(oos2324['ev']>1.0).sum()}R)")
print(f"EV>0.8 の割合: {(oos2324['ev']>0.8).mean():.1%}  ({(oos2324['ev']>0.8).sum()}R)")
