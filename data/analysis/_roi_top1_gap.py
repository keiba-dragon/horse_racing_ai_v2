# coding: utf-8
"""実力1位の「抜け具合」（2位との差・比率）でフィルタしたROIスキャン"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd
import numpy as np

df = pd.read_parquet('data/processed/oos_predictions.parquet')

# 各レースの1位・2位prob
ranked = df.sort_values(['race_id', 'prob_win'], ascending=[True, False])
ranked['rank_prob'] = ranked.groupby('race_id')['prob_win'].rank(ascending=False, method='first')

top1 = ranked[ranked['rank_prob'] == 1].copy()
top2 = ranked[ranked['rank_prob'] == 2][['race_id', 'prob_win']].rename(columns={'prob_win': 'prob2'})

top1 = top1.merge(top2, on='race_id', how='left')
top1['gap']   = top1['prob_win'] - top1['prob2']          # 差
top1['ratio'] = top1['prob_win'] / top1['prob2'].clip(0.001)  # 比率
top1['payout'] = top1['単勝オッズ'] * 100 * top1['target_win']

BET = 100
N = len(top1)

print(f'全体: {N}件  gap平均={top1["gap"].mean():.4f}  ratio平均={top1["ratio"].mean():.2f}')
print()

# ── gap（差）スキャン ──
print('=== prob_win gap (1位-2位) >= X ===')
print(f'{"min_gap":>9} {"件数":>6} {"割合":>6} {"的中率":>7} {"avg_odds":>9} {"ROI":>8}')
for thr in [0.00, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20]:
    sub = top1[top1['gap'] >= thr]
    if len(sub) < 50:
        break
    wins = sub['target_win'].sum()
    ret  = sub['payout'].sum()
    bet  = len(sub) * BET
    roi  = (ret - bet) / bet * 100
    wr   = wins / len(sub) * 100
    ao   = sub['単勝オッズ'].mean()
    pct  = len(sub) / N * 100
    print(f'{thr:>9.2f} {len(sub):>6} {pct:>5.1f}% {wr:>7.1f}% {ao:>9.2f} {roi:>8.1f}%')

print()

# ── ratio（比率）スキャン ──
print('=== prob_win ratio (1位/2位) >= X ===')
print(f'{"min_ratio":>9} {"件数":>6} {"割合":>6} {"的中率":>7} {"avg_odds":>9} {"ROI":>8}')
for thr in [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0]:
    sub = top1[top1['ratio'] >= thr]
    if len(sub) < 50:
        break
    wins = sub['target_win'].sum()
    ret  = sub['payout'].sum()
    bet  = len(sub) * BET
    roi  = (ret - bet) / bet * 100
    wr   = wins / len(sub) * 100
    ao   = sub['単勝オッズ'].mean()
    pct  = len(sub) / N * 100
    print(f'{thr:>9.1f} {len(sub):>6} {pct:>5.1f}% {wr:>7.1f}% {ao:>9.2f} {roi:>8.1f}%')

print()

# ── gap帯別内訳 ──
print('=== gap帯別 内訳 ===')
bins = [0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 1.0]
labels = ['~0.02','0.02~0.04','0.04~0.06','0.06~0.08','0.08~0.10','0.10~0.15','0.15~']
top1['gap_band'] = pd.cut(top1['gap'], bins=bins, labels=labels, right=False)
gr = top1.groupby('gap_band', observed=True).agg(
    n=('target_win','count'), wins=('target_win','sum'), ret=('payout','sum'), avg_odds=('単勝オッズ','mean')
)
gr['bet'] = gr['n'] * BET
gr['roi'] = (gr['ret'] - gr['bet']) / gr['bet'] * 100
gr['wr']  = gr['wins'] / gr['n'] * 100
print(gr[['n','wins','wr','avg_odds','roi']].to_string(float_format='%.1f'))

print()

# ── gap + オッズ組み合わせ（有望そうな交差点） ──
print('=== gap >= 0.05 & オッズ帯別 ===')
sub = top1[top1['gap'] >= 0.05]
bins2 = [0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 99]
labels2 = ['~1.5','1.5~2','2~2.5','2.5~3','3~4','4~5','5~']
sub = sub.copy()
sub['ob'] = pd.cut(sub['単勝オッズ'], bins=bins2, labels=labels2, right=False)
gr2 = sub.groupby('ob', observed=True).agg(
    n=('target_win','count'), wins=('target_win','sum'), ret=('payout','sum')
)
gr2['bet'] = gr2['n'] * BET
gr2['roi'] = (gr2['ret'] - gr2['bet']) / gr2['bet'] * 100
gr2['wr']  = gr2['wins'] / gr2['n'] * 100
print(gr2[['n','wins','wr','roi']].to_string(float_format='%.1f'))
