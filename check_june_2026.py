# coding: utf-8
"""
2026年6月6日・7日のレースに accuracy_model.pkl で予想し、的中率を確認する
"""
import sys, os, pickle
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from save_conditional_logit import prepare, segment_softmax, BASE_DIR, DATA_FILE
from save_v3 import add_computed_features

TARGET_DATES = [260606, 260607]

with open(os.path.join(BASE_DIR, 'models', 'accuracy_model.pkl'), 'rb') as f:
    MODEL = pickle.load(f)

# ── データロード ──────────────────────────────────────────────────────────────
print("データ読み込み中...", flush=True)
df = pd.read_parquet(DATA_FILE)
df['日付_num'] = pd.to_numeric(df['日付'], errors='coerce')
df['着順_num'] = pd.to_numeric(df['着順_num'], errors='coerce')
# 着順_numが0（未入力）も含める（出馬表データ対応）
df = df.dropna(subset=['日付_num'])
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
baba_map = {'良': 0, '稍重': 1, '重': 2, '不良': 3}
for col in df.columns:
    if '馬場状態' in col and col != '馬場状態':
        df[col] = df[col].map(baba_map)

# 距離数値を df に付与
df['dist_m'] = dm

# ── 対象日フィルタ ─────────────────────────────────────────────────────────────
df_target = df[df['日付_num'].isin(TARGET_DATES)].copy()
print(f"対象レース日: 2026-06-06, 2026-06-07  ({len(df_target)}頭 / {df_target['race_id'].nunique()}レース)")
print("※ 着順データ未入力のため予想のみ出力（結果照合は別途）")

s = df['surface']
r = df['クラス_rank']
s_t = df_target['surface']
r_t = df_target['クラス_rank']
dm_t = df_target['dist_m']

SEG_MASKS_TRAIN = {
    'ダ長': (s=='ダ') & (dm>1400) & (r!=1.0),
    'ダ短': (s=='ダ') & (dm<=1400) & (r!=1.0),
    '芝短': (s=='芝') & (dm<=1400) & (r!=1.0),
    '芝中': (s=='芝') & (dm>1400) & (dm<=2000) & (r!=1.0),
    '芝長': (s=='芝') & (dm>2000) & (r!=1.0),
}
SEG_MASKS_TARGET = {
    'ダ長': (s_t=='ダ') & (dm_t>1400) & (r_t!=1.0),
    'ダ短': (s_t=='ダ') & (dm_t<=1400) & (r_t!=1.0),
    '芝短': (s_t=='芝') & (dm_t<=1400) & (r_t!=1.0),
    '芝中': (s_t=='芝') & (dm_t>1400) & (dm_t<=2000) & (r_t!=1.0),
    '芝長': (s_t=='芝') & (dm_t>2000) & (r_t!=1.0),
}

FAV = {'ダ長': 0.3403, 'ダ短': 0.3490, '芝短': 0.2869, '芝中': 0.3321, '芝長': 0.3605}

print("\n" + "="*70)
print(f"{'セグメント':<6} {'R数':>5}")
print("="*70)

all_preds = []
total_races = 0

for seg in ['芝長', '芝中', '芝短', 'ダ長', 'ダ短']:
    pkg = MODEL[seg]
    feat_cols = pkg['feat_cols']
    scaler    = pkg['scaler']
    coef      = pkg['coef']
    isotonic  = pkg['isotonic']

    # 対象日のセグメント
    seg_t = df_target[SEG_MASKS_TARGET[seg]].copy()
    if len(seg_t) == 0:
        print(f"{seg:<6} {'(なし)':>5}")
        continue

    # NaN指示変数を対象データに生成（feat_colsに_isnanがある場合）
    for fc in feat_cols:
        if fc.endswith('_isnan'):
            base = fc[:-6]
            if base in seg_t.columns and fc not in seg_t.columns:
                seg_t[fc] = seg_t[base].isna().astype(float)

    try:
        X_p, _, gs_p, n_p, *_ = prepare(seg_t, feat_cols, scaler=scaler,
                                          top_idx=None, top_idx3=None)
        s_sorted = seg_t.sort_values('race_id').reset_index(drop=True)
        probs = segment_softmax(X_p @ coef, gs_p, n_p)
        s_sorted['prob'] = probs
        s_sorted['rank'] = s_sorted.groupby('race_id')['prob'].rank(
            ascending=False, method='first')

        # 各レースの予想1位馬
        pred1 = s_sorted[s_sorted['rank'] == 1].copy()
        nr = s_sorted['race_id'].nunique()
        print(f"{seg:<6} {nr:>5}R")

        # 詳細保存
        pred1 = pred1.copy()
        pred1['seg'] = seg
        name_col = '馬名S' if '馬名S' in pred1.columns else ('馬名' if '馬名' in pred1.columns else None)
        if name_col:
            all_preds.append(pred1[['seg','race_id', name_col,'prob','単勝オッズ']].rename(
                columns={name_col: '馬名'}).copy())

        total_races += nr
    except Exception as e:
        print(f"{seg:<6} ERROR: {e}")

print("="*70)
print(f"合計 {total_races}レース")

# ── レース別詳細 ──────────────────────────────────────────────────────────────
if all_preds:
    preds_all = pd.concat(all_preds, ignore_index=True)
    print("\n── レース別予想（的中率モデル 1位推奨馬） ──")
    print(f"{'日付_R':<20} {'セグ':<5} {'推奨馬':<16} {'単勝オッズ':>8} {'確率':>7}")
    print("-"*60)
    for _, row in preds_all.sort_values('race_id').iterrows():
        odds_str = f"{float(row['単勝オッズ']):.1f}倍" if pd.notna(row['単勝オッズ']) and row['単勝オッズ'] != 0 else "---"
        print(f"  {row['race_id']:<20} {row['seg']:<5} {str(row['馬名']):<16} {odds_str:>8}  {row['prob']:.3f}")
