# coding: utf-8
"""
複勝版 EVフィルタ バックテスト
  推定複勝EV = P_calib × 単勝オッズ × 0.32
  実績複勝EV = P_calib × 複勝配当/100  (バックテスト用)
  セグメント別閾値最適化 (2325合算ROIで選択 → 2026検証)
"""
import sys, os, pickle
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from save_conditional_logit import prepare, segment_softmax, BASE_DIR, DATA_FILE
from save_v3 import add_computed_features

with open(os.path.join(BASE_DIR, 'models', 'accuracy_fukusho_model.pkl'), 'rb') as f:
    MODEL = pickle.load(f)

print("データ読み込み中...", flush=True)
df = pd.read_parquet(DATA_FILE)
df['日付_num']       = pd.to_numeric(df['日付'], errors='coerce')
df['着順_num']       = pd.to_numeric(df['着順_num'], errors='coerce')
df['複勝配当_num']   = pd.to_numeric(df['複勝配当'], errors='coerce')
df['単勝オッズ_num'] = pd.to_numeric(df['単勝オッズ'], errors='coerce')
df = df.dropna(subset=['日付_num', '着順_num'])
df = df[df['着順_num'] < 99]
df['race_id'] = (df['日付_num'].astype(int).astype(str) + '_' +
                 df['開催'].astype(str).str.strip() + '_' + df['Ｒ'].astype(str).str.strip())
df = df[df['開催'].notna()].copy()
df['surface']    = df['距離'].astype(str).str.strip().str.extract(r'^([芝ダ])')[0].fillna('不明')
dm               = pd.to_numeric(df['距離'].astype(str).str.extract(r'(\d+)')[0], errors='coerce')
df['クラス_rank'] = pd.to_numeric(df['クラス_rank'], errors='coerce')
df = add_computed_features(df)
if '今回_会場' in df.columns and '1走前_開催' in df.columns:
    df['輸送有無'] = (df['今回_会場'].astype(str) != df['1走前_開催'].astype(str).str[1]).astype(float)
    df.loc[df['1走前_開催'].isna(), '輸送有無'] = float('nan')
baba_map = {'良': 0, '稍重': 1, '重': 2, '不良': 3}
for col in df.columns:
    if '馬場状態' in col and col != '馬場状態':
        df[col] = df[col].map(baba_map)
df['dist_m'] = dm

s = df['surface']; r = df['クラス_rank']; d = df['dist_m']
SEG_MASKS = {
    'ダ長': (s=='ダ')&(d>1400) &(r!=1.0),
    'ダ短': (s=='ダ')&(d<=1400)&(r!=1.0),
    '芝短': (s=='芝')&(d<=1400)&(r!=1.0),
    '芝中': (s=='芝')&(d>1400) &(d<=2000)&(r!=1.0),
    '芝長': (s=='芝')&(d>2000) &(r!=1.0),
}

print("確率・EV 計算中...", flush=True)
all_rows = []
for seg in ['芝長', '芝中', '芝短', 'ダ長', 'ダ短']:
    pkg = MODEL[seg]
    feat_cols, scaler, coef, iso = pkg['feat_cols'], pkg['scaler'], pkg['coef'], pkg['isotonic']
    seg_df = df[SEG_MASKS[seg]].copy()
    for fc in feat_cols:
        if fc.endswith('_isnan'):
            base = fc[:-6]
            if base in seg_df.columns and fc not in seg_df.columns:
                seg_df[fc] = seg_df[base].isna().astype(float)
    try:
        X, _, gs, n, *_ = prepare(seg_df, feat_cols, scaler=scaler, top_idx=None, top_idx3=None)
    except Exception as e:
        print(f"  {seg} skip: {e}"); continue

    ss = seg_df.sort_values('race_id').reset_index(drop=True)
    raw_prob = segment_softmax(X @ coef, gs, n)
    ss['prob_raw']   = raw_prob
    ss['prob_calib'] = iso.predict(raw_prob)
    ss['rank_pred']  = ss.groupby('race_id')['prob_raw'].rank(ascending=False, method='first')

    rank1 = ss[ss['rank_pred']==1].set_index('race_id')[[
        'prob_raw','prob_calib','単勝オッズ_num','複勝配当_num','着順_num']]
    rank2 = ss[ss['rank_pred']==2].set_index('race_id')[['prob_raw']].rename(columns={'prob_raw':'prob2'})
    merged = rank1.join(rank2, how='left')
    merged['prob_gap'] = merged['prob_raw'] - merged['prob2'].fillna(0)

    # 推定複勝EV（出走前に計算可能）
    merged['ev_est'] = merged['prob_calib'] * merged['単勝オッズ_num'] * 0.32
    # 実績複勝EV（バックテスト用 - 1着/2着/3着の場合のみ複勝配当あり）
    merged['fukusho_odds'] = merged['複勝配当_num'] / 100
    merged['ev_real'] = merged['prob_calib'] * merged['fukusho_odds']

    merged['hit_fukusho'] = (merged['着順_num'] <= 3).astype(int)
    merged['seg'] = seg

    date_map = ss.drop_duplicates('race_id').set_index('race_id')['日付_num']
    merged['日付_num'] = merged.index.map(date_map)
    merged = merged.reset_index()
    all_rows.append(merged)

all_df = pd.concat(all_rows, ignore_index=True)
all_df = all_df.dropna(subset=['ev_est', '着順_num', '単勝オッズ_num'])
print(f"全レース: {len(all_df)}R\n")

EV_THS = [0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70]
MIN_R = 50

def ef_fukusho(data, ev_th):
    """推定EVでフィルタし、実績複勝配当でROIを計算"""
    f = data[data['ev_est'] >= ev_th]   # 4着以下も含む（dropnaしない）
    nr = len(f)
    if nr == 0: return 0, 0., 0.
    acc = f['hit_fukusho'].mean()
    # 的中馬の実際の複勝払い戻し
    hits = f[f['hit_fukusho'] == 1]
    payout = hits['fukusho_odds'].fillna(0).sum()
    roi = (payout - nr) / nr
    return nr, acc, roi

def score_2325(d24, d25, ev_th):
    r24, a24, roi24 = ef_fukusho(d24, ev_th)
    r25, a25, roi25 = ef_fukusho(d25, ev_th)
    if r24 + r25 < MIN_R: return -99.
    roi24c = max(min(roi24, 0.30), -0.30)
    roi25c = max(min(roi25, 0.30), -0.30)
    return (roi24c * r24 + roi25c * r25) / (r24 + r25)

print('=' * 85)
print('複勝版 EVフィルタ最適化（推定EV=P×単勝×0.32  実績EV=P×複勝配当/100）')
print('=' * 85)

best_by_seg = {}
for seg in ['芝長', '芝中', '芝短', 'ダ長', 'ダ短']:
    seg_d = all_df[all_df['seg']==seg]
    d24 = seg_d[(seg_d['日付_num']>=230101)&(seg_d['日付_num']<250101)]
    d25 = seg_d[(seg_d['日付_num']>=250101)&(seg_d['日付_num']<260101)]
    d26 = seg_d[seg_d['日付_num']>=260101]

    _, a_b24, roi_b24 = ef_fukusho(d24, 0)
    _, a_b25, roi_b25 = ef_fukusho(d25, 0)
    _, a_b26, roi_b26 = ef_fukusho(d26, 0)
    print(f'\n[{seg}]  全買いベースライン: '
          f'2324={a_b24:.1%}/{roi_b24:+.1%}  2025={a_b25:.1%}/{roi_b25:+.1%}  2026={a_b26:.1%}/{roi_b26:+.1%}')
    print(f"  {'推定EV≥':>8} {'2325sc':>7} | {'2324R':>5} {'2324的中率':>9} {'2324ROI':>8} "
          f"| {'2025R':>5} {'2025ROI':>8} | {'2026R':>5} {'2026的中率':>9} {'2026ROI':>8}")
    print('  ' + '-' * 75)

    best_sc = score_2325(d24, d25, 0); best_ev = 0.0
    for ev_th in EV_THS[1:]:
        sc = score_2325(d24, d25, ev_th)
        if sc == -99.: continue
        r24, a24, roi24 = ef_fukusho(d24, ev_th)
        r25, a25, roi25 = ef_fukusho(d25, ev_th)
        r26, a26, roi26 = ef_fukusho(d26, ev_th)
        mark = ''
        if sc > best_sc: best_sc = sc; best_ev = ev_th; mark = ' ← best'
        print(f'  {ev_th:>8.2f} {sc:>+7.1%} | {r24:>5} {a24:>9.1%} {roi24:>+8.1%} '
              f'| {r25:>5} {roi25:>+8.1%} | {r26:>5} {a26:>9.1%} {roi26:>+8.1%}{mark}')
    best_by_seg[seg] = best_ev
    print(f'  → 最適推定EV閾値: {best_ev:.2f}  (2325スコア={best_sc:+.1%})')

print()
print('=' * 85)
print('最終サマリー: 複勝版 セグメント別最適閾値 × 全期間')
print('=' * 85)
print(f"{'セグメント':<6} {'EV閾値':>6} | {'2324R':>5} {'2324複勝%':>9} {'2324ROI':>8} "
      f"| {'2025R':>5} {'2025複勝%':>9} {'2025ROI':>8} "
      f"| {'2026R':>5} {'2026複勝%':>9} {'2026ROI':>8} | {'25+26ROI':>9}")
print('-' * 90)

agg = {'r25': 0, 'pay25': 0, 'r26': 0, 'pay26': 0}
for seg in ['芝長', '芝中', '芝短', 'ダ長', 'ダ短']:
    seg_d = all_df[all_df['seg']==seg]
    ev_th = best_by_seg[seg]
    d24 = seg_d[(seg_d['日付_num']>=230101)&(seg_d['日付_num']<250101)]
    d25 = seg_d[(seg_d['日付_num']>=250101)&(seg_d['日付_num']<260101)]
    d26 = seg_d[seg_d['日付_num']>=260101]
    r24, a24, roi24 = ef_fukusho(d24, ev_th)
    r25, a25, roi25 = ef_fukusho(d25, ev_th)
    r26, a26, roi26 = ef_fukusho(d26, ev_th)
    f25 = d25[d25['ev_est'] >= ev_th]
    f26 = d26[d26['ev_est'] >= ev_th]
    pay25 = f25[f25['hit_fukusho']==1]['fukusho_odds'].sum()
    pay26 = f26[f26['hit_fukusho']==1]['fukusho_odds'].sum()
    roi2526 = (pay25+pay26-r25-r26)/(r25+r26) if (r25+r26) > 0 else float('nan')
    agg['r25'] += r25; agg['pay25'] += pay25
    agg['r26'] += r26; agg['pay26'] += pay26
    print(f"{seg:<6} {ev_th:>6.2f} | {r24:>5} {a24:>9.1%} {roi24:>+8.1%} "
          f"| {r25:>5} {a25:>9.1%} {roi25:>+8.1%} "
          f"| {r26:>5} {a26:>9.1%} {roi26:>+8.1%} | {roi2526:>+9.1%}")

roi_all = (agg['pay25']+agg['pay26']-agg['r25']-agg['r26']) / (agg['r25']+agg['r26'])
print('-' * 90)
print(f"{'全体計':<13} | {'---':>5} {'---':>9} {'---':>8} "
      f"| {agg['r25']:>5} {'---':>9} {'---':>8} "
      f"| {agg['r26']:>5} {'---':>9} {'---':>8} | {roi_all:>+9.1%}")
print(f"\n全体 25+26合算 複勝ROI: {roi_all:+.1%}  ({agg['r25']+agg['r26']}R)")
