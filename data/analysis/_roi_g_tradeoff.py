# coding: utf-8
"""G指標: 閾値別 件数×ROI トレードオフ + 年別安定性"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd

df = pd.read_parquet('data/processed/oos_predictions.parquet')

ranked = df.sort_values(['race_id', 'prob_win'], ascending=[True, False])
ranked['rank_prob'] = ranked.groupby('race_id')['prob_win'].rank(ascending=False, method='first')

top1 = ranked[ranked['rank_prob'] == 1].copy()
top2 = ranked[ranked['rank_prob'] == 2][['race_id','prob_win']].rename(columns={'prob_win':'prob2'})
top1 = top1.merge(top2, on='race_id', how='left')
top1['gap'] = top1['prob_win'] - top1['prob2']
top1['payout'] = top1['単勝オッズ'] * 100 * top1['target_win']

bins   = [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 1.0]
labels = ['~0.05','0.05~0.10','0.10~0.15','0.15~0.20','0.20~0.25','0.25~0.30','0.30~']
top1['gap_band'] = pd.cut(top1['gap'], bins=bins, labels=labels, right=False)
band_avg = top1.groupby('gap_band', observed=True)['単勝オッズ'].mean()
top1['band_avg_odds'] = top1['gap_band'].map(band_avg).astype(float)
top1 = top1.reset_index(drop=True)

BET = 100
FULL_YEARS = ['21', '22', '23', '24', '25']  # 26は途中

candidates = [
    ('現状',        0.15, 1.30),
    ('緩め①',      0.15, 1.20),
    ('緩め②',      0.15, 1.10),
    ('gap緩め①',   0.12, 1.30),
    ('gap緩め②',   0.12, 1.20),
    ('gap緩め③',   0.10, 1.30),
    ('全緩め',      0.12, 1.10),
]

print('=== 閾値別サマリー (全期間) ===')
print(f'{"設定":<12} {"gap":>5} {"倍率":>5} {"件数":>6} {"年平均":>6} {"的中率":>7} {"avg_odds":>9} {"ROI":>8}')
for name, gap_thr, mult in candidates:
    s = top1[(top1['gap'] >= gap_thr) & (top1['単勝オッズ'] >= top1['band_avg_odds'] * mult)]
    if len(s) == 0:
        continue
    n    = len(s)
    wins = s['target_win'].sum()
    ret  = s['payout'].sum()
    roi  = (ret - n*BET) / (n*BET) * 100
    wr   = wins / n * 100
    ao   = s['単勝オッズ'].mean()
    n_per_yr = n / 5.5
    print(f'{name:<12} {gap_thr:>5.2f} {mult:>5.1f} {n:>6} {n_per_yr:>6.0f} {wr:>7.1f}% {ao:>9.2f} {roi:>8.1f}%')

print()
# デバッグ: year列確認
s0 = top1[(top1['gap'] >= 0.15) & (top1['単勝オッズ'] >= top1['band_avg_odds'] * 1.3)]
print(f'[debug] year dtype={s0["year"].dtype}, unique={sorted(s0["year"].unique())}')
print()

# 年別ROIを横に並べる
print('=== 年別ROI 比較 ===')
header = f'{"設定":<12}' + ''.join(f'{y:>8}' for y in FULL_YEARS) + f'{"平均":>8} {"標準偏差":>8}'
print(header)
print('-' * len(header))

for name, gap_thr, mult in candidates:
    s = top1[(top1['gap'] >= gap_thr) & (top1['単勝オッズ'] >= top1['band_avg_odds'] * mult)]
    rois = []
    row = f'{name:<12}'
    for y in FULL_YEARS:
        sy = s[s['year'] == y]
        if len(sy) == 0:
            row += f'{"N/A":>8}'
            continue
        r = (sy['payout'].sum() - len(sy)*BET) / (len(sy)*BET) * 100
        rois.append(r)
        row += f'{r:>7.1f}%'
    avg_roi = sum(rois)/len(rois) if rois else 0
    std_roi = pd.Series(rois).std() if len(rois) > 1 else 0
    row += f'{avg_roi:>7.1f}% {std_roi:>8.1f}'
    print(row)

print()

# 勝ち年の数
print('=== 黒字年の数 (2021-2025) ===')
for name, gap_thr, mult in candidates:
    s = top1[(top1['gap'] >= gap_thr) & (top1['単勝オッズ'] >= top1['band_avg_odds'] * mult)]
    wins_yr = 0
    for y in FULL_YEARS:
        sy = s[s['year'] == y]
        if len(sy) == 0:
            continue
        r = (sy['payout'].sum() - len(sy)*BET) / (len(sy)*BET) * 100
        if r > 0:
            wins_yr += 1
    n = len(s)
    print(f'{name:<12} 黒字年: {wins_yr}/{len(FULL_YEARS)}年  年平均件数: {n/5.5:.0f}件')
