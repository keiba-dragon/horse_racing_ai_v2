# coding: utf-8
"""
実力1位・2位のgapに対して「平均より高いオッズ」のときだけ買う戦略
gap帯ごとの平均オッズをベースラインにして、それ以上なら市場が割安 → 買い
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd
import numpy as np

df = pd.read_parquet('data/processed/oos_predictions.parquet')

ranked = df.sort_values(['race_id', 'prob_win'], ascending=[True, False])
ranked['rank_prob'] = ranked.groupby('race_id')['prob_win'].rank(ascending=False, method='first')

top1 = ranked[ranked['rank_prob'] == 1].copy()
top2 = ranked[ranked['rank_prob'] == 2][['race_id', 'prob_win']].rename(columns={'prob_win': 'prob2'})
top1 = top1.merge(top2, on='race_id', how='left')
top1['gap'] = top1['prob_win'] - top1['prob2']
top1['payout'] = top1['単勝オッズ'] * 100 * top1['target_win']

BET = 100

# ── gap帯ごとの平均オッズ（ベースライン）──
bins   = [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 1.0]
labels = ['~0.05','0.05~0.10','0.10~0.15','0.15~0.20','0.20~0.25','0.25~0.30','0.30~']
top1['gap_band'] = pd.cut(top1['gap'], bins=bins, labels=labels, right=False)

band_avg = top1.groupby('gap_band', observed=True)['単勝オッズ'].mean().rename('band_avg_odds')
top1 = top1.join(band_avg, on='gap_band')

print('=== gap帯別 平均オッズ（ベースライン）===')
base = top1.groupby('gap_band', observed=True).agg(
    n=('target_win','count'),
    avg_odds=('単勝オッズ','mean'),
    wr=('target_win','mean'),
)
base['wr'] = base['wr'] * 100
print(base.to_string(float_format='%.3f'))
print()

# ── フィルタ: gap >= G かつ odds >= band_avg_odds * M ──
print('=== gap閾値 × オッズ倍率フィルタ ===')
print(f'{"gap>=":>6} {"倍率M":>6} {"件数":>6} {"割合":>6} {"的中率":>7} {"avg_odds":>9} {"ROI":>8}')

for gap_thr in [0.05, 0.10, 0.15, 0.20]:
    for mult in [1.0, 1.1, 1.2, 1.3, 1.5]:
        sub = top1[(top1['gap'] >= gap_thr) & (top1['単勝オッズ'] >= top1['band_avg_odds'] * mult)]
        if len(sub) < 30:
            continue
        wins = sub['target_win'].sum()
        ret  = sub['payout'].sum()
        bet  = len(sub) * BET
        roi  = (ret - bet) / bet * 100
        wr   = wins / len(sub) * 100
        ao   = sub['単勝オッズ'].mean()
        pct  = len(sub) / len(top1) * 100
        print(f'{gap_thr:>6.2f} {mult:>6.1f} {len(sub):>6} {pct:>5.1f}% {wr:>7.1f}% {ao:>9.2f} {roi:>8.1f}%')
    print()

# ── 逆: gap >= G かつ odds < band_avg（平均より人気＝過大評価）──
print('=== 参考: 平均より低いオッズ（市場が過大評価）===')
print(f'{"gap>=":>6} {"件数":>6} {"的中率":>7} {"avg_odds":>9} {"ROI":>8}')
for gap_thr in [0.05, 0.10, 0.15, 0.20]:
    sub = top1[(top1['gap'] >= gap_thr) & (top1['単勝オッズ'] < top1['band_avg_odds'])]
    if len(sub) < 30:
        continue
    wins = sub['target_win'].sum()
    ret  = sub['payout'].sum()
    bet  = len(sub) * BET
    roi  = (ret - bet) / bet * 100
    wr   = wins / len(sub) * 100
    ao   = sub['単勝オッズ'].mean()
    print(f'{gap_thr:>6.2f} {len(sub):>6} {wr:>7.1f}% {ao:>9.2f} {roi:>8.1f}%')
