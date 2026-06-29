# -*- coding: utf-8 -*-
"""
val_selection_search.py
Val（2023-2024）で買い条件を探索し、OOS（2025+）で一発評価する。
実際の複勝配当・単勝配当を使用。カンニングなし。

使い方:
  python data/analysis/val_selection_search.py
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import itertools

VAL_PATH      = 'C:/horse_racing_ai/data/processed/val_predictions.parquet'
OOS_PATH      = 'C:/horse_racing_ai/data/processed/oos_predictions.parquet'
FEATURES_PATH = 'C:/horse_racing_ai/data/processed/all_venues_features.parquet'

val = pd.read_parquet(VAL_PATH)
oos = pd.read_parquet(OOS_PATH)

print(f'Val: {len(val):,}行  year={val["year"].min()}〜{val["year"].max()}')
print(f'OOS: {len(oos):,}行  year={oos["year"].min()}〜{oos["year"].max()}')

# ── 実際の複勝配当を結合 ──
feats = pd.read_parquet(FEATURES_PATH, columns=['日付', '馬名S', '複勝配当'])
feats['date_key'] = feats['日付'].astype(str).str.zfill(6)

def add_real_payouts(pred, feats):
    pred = pred.copy()
    pred['date_key'] = pred['race_id'].str[:6]
    merged = pred.merge(feats[['date_key','馬名S','複勝配当']],
                        on=['date_key','馬名S'], how='left')
    merged['fuku_payout'] = merged['複勝配当'].fillna(0) / 100
    return merged

val = add_real_payouts(val, feats)
oos = add_real_payouts(oos, feats)

print(f'複勝配当結合率 Val: {val["複勝配当"].notna().mean():.1%}  OOS: {oos["複勝配当"].notna().mean():.1%}')
print()

# ── ROI計算 ──
def roi_tan(sub):
    s = sub.dropna(subset=['単勝オッズ'])
    if len(s) < 10: return np.nan, 0
    return s[s['target_win'] == 1]['単勝オッズ'].sum() / len(s) - 1, len(s)

def roi_fuku_real(sub):
    if len(sub) < 10: return np.nan, 0
    return sub['fuku_payout'].sum() / len(sub) - 1, len(sub)

# ── 条件グリッド ──
TRACKS     = ['ダ', '芝', 'both']
HEAD_THRES = [13, 14, 15, 16]
EDGE_THRES = [0.00, 0.02, 0.05, 0.08, 0.10, 0.15]

SEP = '─' * 75

def apply_cond(df, track, heads, edge):
    sub = df[df['rank_edge'] == 1].copy()
    if track == 'ダ':
        sub = sub[sub['gk'].str.endswith('_ダ')]
    elif track == '芝':
        sub = sub[sub['gk'].str.endswith('_芝')]
    sub = sub[sub['頭数'] >= heads]
    sub = sub[sub['edge'] >= edge]
    return sub

print('=== Val（2023-2024）で条件探索 ===')
print(f'{"track":<6} {"heads≥":>6} {"edge≥":>6} | {"単勝ROI":>8} {"複勝ROI(実)":>11} {"N":>5}')
print(SEP)

results = []
for track, heads, edge in itertools.product(TRACKS, HEAD_THRES, EDGE_THRES):
    sub = apply_cond(val, track, heads, edge)
    r_tan, n   = roi_tan(sub)
    r_fuku, _  = roi_fuku_real(sub)
    if np.isnan(r_tan): continue
    results.append({
        'track': track, 'heads': heads, 'edge': edge,
        'roi_tan': r_tan, 'roi_fuku': r_fuku, 'n': n
    })

res_df = pd.DataFrame(results).sort_values('roi_tan', ascending=False)

for _, row in res_df.head(20).iterrows():
    print(f'{row["track"]:<6} {int(row["heads"]):>6} {row["edge"]:>6.2f} | '
          f'{row["roi_tan"]:>+8.1%} {row["roi_fuku"]:>+11.1%} {int(row["n"]):>5}件')

# ── Val最良条件を選ぶ（単勝ROI基準、N≥100件） ──
print()
print('=== 条件選択基準: Val 単勝ROI最大（N≥100件） ===')
valid = res_df[res_df['n'] >= 100]
if valid.empty:
    print('N≥100の条件なし。N≥50で再試行。')
    valid = res_df[res_df['n'] >= 50]

best = valid.iloc[0]
print(f'選択条件: track={best["track"]} heads≥{int(best["heads"])} edge≥{best["edge"]:.2f}')
print(f'Val ROI: 単勝={best["roi_tan"]:+.1%}  複勝(実)={best["roi_fuku"]:+.1%}  N={int(best["n"])}件')

# ── OOS一発評価（カンニングなし） ──
print()
print('=' * 75)
print('=== OOS（2025-2026）一発評価 ===')
print('=' * 75)
oos_sub = apply_cond(oos, best['track'], int(best['heads']), best['edge'])

r_tan_oos, n_tan = roi_tan(oos_sub)
r_fuku_oos, _    = roi_fuku_real(oos_sub)
n_oos_years      = len(oos['year'].unique())
weekly           = n_tan / max(n_oos_years * 52, 1)

print(f'条件: track={best["track"]}  頭数≥{int(best["heads"])}  edge≥{best["edge"]:.2f}')
print(f'件数: {n_tan}件  週平均: {weekly:.1f}件')
print(f'単勝 ROI: {r_tan_oos:+.1%}  的中率: {oos_sub["target_win"].mean():.1%}')
print(f'複勝 ROI(実): {r_fuku_oos:+.1%}  的中率: {(oos_sub["複勝配当"].notna()).mean():.1%}')
print()

print('年別 単勝ROI:')
for yr, g in oos_sub.groupby('year'):
    r, n = roi_tan(g)
    print(f'  20{yr}: {r:>+7.1%} ({n}件)')

print()
print('グループ別 単勝ROI:')
for gk_val, g in oos_sub.groupby('gk'):
    r, n = roi_tan(g)
    print(f'  {gk_val}: {r:>+7.1%} ({n}件)')

# ── 参考: Val上位5条件をすべてOOSで確認 ──
print()
print('=== 参考: Val上位5条件のOOS確認 ===')
print(f'{"Val条件":<30} | {"Val単勝":>8} | {"OOS単勝":>8} {"OOS複勝(実)":>11} {"N":>5}')
print(SEP)
for _, row in valid.head(5).iterrows():
    sub_o = apply_cond(oos, row['track'], int(row['heads']), row['edge'])
    r_t, n_o = roi_tan(sub_o)
    r_f, _   = roi_fuku_real(sub_o)
    label = f'{row["track"]} heads≥{int(row["heads"])} edge≥{row["edge"]:.2f}'
    print(f'{label:<30} | {row["roi_tan"]:>+8.1%} | {r_t:>+8.1%} {r_f:>+11.1%} {n_o:>5}件')
