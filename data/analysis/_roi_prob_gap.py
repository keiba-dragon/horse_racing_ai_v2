# coding: utf-8
"""実力評価（prob_win）1位と2位の差でフィルタしたROI分析"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd

df = pd.read_parquet('data/processed/oos_predictions.parquet')

ranked = df.sort_values(['race_id', 'prob_win'], ascending=[True, False])
ranked['rank_prob'] = ranked.groupby('race_id')['prob_win'].rank(ascending=False, method='first')

top1 = ranked[ranked['rank_prob'] == 1].copy()
top2 = ranked[ranked['rank_prob'] == 2][['race_id', 'prob_win']].rename(columns={'prob_win': 'prob2'})

top1 = top1.merge(top2, on='race_id', how='left')
top1['gap'] = top1['prob_win'] - top1['prob2']
top1['payout'] = top1['単勝オッズ'] * 100 * top1['target_win']

BET = 100

print(f'全体 {len(top1)}レース')
print(f'gap分布: 平均={top1["gap"].mean():.3f}  中央値={top1["gap"].median():.3f}  最大={top1["gap"].max():.3f}')
print()

# gap閾値スキャン（細かく）
print('=== 実力1位と2位のprobの差 >= X のみ賭ける ===')
print(f'{"gap閾値":>8} {"件数":>6} {"割合":>6} {"的中率":>7} {"avg_odds":>9} {"ROI":>8}  ブレイクイーブン必要的中率')
for thr in [0.00, 0.05, 0.10, 0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.35, 0.40]:
    sub = top1[top1['gap'] >= thr]
    if len(sub) < 30:
        break
    wins = sub['target_win'].sum()
    ret  = sub['payout'].sum()
    bet  = len(sub) * BET
    roi  = (ret - bet) / bet * 100
    wr   = wins / len(sub) * 100
    ao   = sub['単勝オッズ'].mean()
    pct  = len(sub) / len(top1) * 100
    be   = 1 / ao / 0.75 * 100  # ブレイクイーブン的中率（控除率25%考慮）
    print(f'{thr:>8.2f} {len(sub):>6} {pct:>5.1f}% {wr:>7.1f}% {ao:>9.2f} {roi:>8.1f}%  {be:.1f}%必要')

print()

# 的中したときのgapと外れたときのgapを比較
won = top1[top1['target_win'] == 1]
lost = top1[top1['target_win'] == 0]
print(f'=== 的中・不的中 別 gap平均 ===')
print(f'的中時 gap平均: {won["gap"].mean():.4f}  中央値: {won["gap"].median():.4f}')
print(f'外れ時 gap平均: {lost["gap"].mean():.4f}  中央値: {lost["gap"].median():.4f}')
print()

# gap四分位別
print('=== gapの四分位別 的中率・ROI ===')
top1['gap_q'] = pd.qcut(top1['gap'], q=4, labels=['Q1(小)','Q2','Q3','Q4(大)'])
gr = top1.groupby('gap_q', observed=True).agg(
    n=('target_win','count'),
    wins=('target_win','sum'),
    ret=('payout','sum'),
    avg_odds=('単勝オッズ','mean'),
    gap_min=('gap','min'),
    gap_max=('gap','max'),
)
gr['bet'] = gr['n'] * BET
gr['roi'] = (gr['ret'] - gr['bet']) / gr['bet'] * 100
gr['wr']  = gr['wins'] / gr['n'] * 100
print(gr[['n','wins','wr','avg_odds','gap_min','gap_max','roi']].to_string(float_format='%.3f'))
