# coding: utf-8
"""実力1位をオッズで絞った場合のROI・的中率スキャン"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd

df = pd.read_parquet('data/processed/oos_predictions.parquet')

top1 = df.loc[df.groupby('race_id')['prob_win'].idxmax()].copy()
top1['payout'] = top1['単勝オッズ'] * 100 * top1['target_win']

BET = 100
total = len(top1)

print(f'実力1位ベース: {total}件\n')

# ── 最低オッズ縛り（穴寄り）──
print('=== 最低オッズ縛り (オッズ >= X のみ賭ける) ===')
print(f'{"min_odds":>9} {"件数":>6} {"的中":>5} {"的中率":>7} {"avg_odds":>9} {"ROI":>8}')
for thr in [1.0, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]:
    sub = top1[top1['単勝オッズ'] >= thr]
    if len(sub) == 0:
        continue
    wins = sub['target_win'].sum()
    ret  = sub['payout'].sum()
    bet  = len(sub) * BET
    roi  = (ret - bet) / bet * 100
    wr   = wins / len(sub) * 100
    ao   = sub['単勝オッズ'].mean()
    print(f'{thr:>9.1f} {len(sub):>6} {int(wins):>5} {wr:>7.1f}% {ao:>9.2f} {roi:>8.1f}%')

print()

# ── 最大オッズ縛り（本命寄り）──
print('=== 最大オッズ縛り (オッズ <= X のみ賭ける) ===')
print(f'{"max_odds":>9} {"件数":>6} {"的中":>5} {"的中率":>7} {"avg_odds":>9} {"ROI":>8}')
for thr in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0]:
    sub = top1[top1['単勝オッズ'] <= thr]
    if len(sub) == 0:
        continue
    wins = sub['target_win'].sum()
    ret  = sub['payout'].sum()
    bet  = len(sub) * BET
    roi  = (ret - bet) / bet * 100
    wr   = wins / len(sub) * 100
    ao   = sub['単勝オッズ'].mean()
    print(f'{thr:>9.1f} {len(sub):>6} {int(wins):>5} {wr:>7.1f}% {ao:>9.2f} {roi:>8.1f}%')

print()

# ── エッジ追加フィルタ（実力1位 & edge >= X）──
print('=== 実力1位 & edge >= X ===')
print(f'{"min_edge":>9} {"件数":>6} {"的中":>5} {"的中率":>7} {"avg_odds":>9} {"ROI":>8}')
for thr in [0.00, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15]:
    sub = top1[top1['edge'] >= thr]
    if len(sub) == 0:
        continue
    wins = sub['target_win'].sum()
    ret  = sub['payout'].sum()
    bet  = len(sub) * BET
    roi  = (ret - bet) / bet * 100
    wr   = wins / len(sub) * 100
    ao   = sub['単勝オッズ'].mean()
    print(f'{thr:>9.2f} {len(sub):>6} {int(wins):>5} {wr:>7.1f}% {ao:>9.2f} {roi:>8.1f}%')

print()

# ── オッズ帯別内訳 ──
print('=== オッズ帯別 的中率・ROI ===')
bins = [0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0, 99]
labels = ['~1.5', '1.5~2', '2~2.5', '2.5~3', '3~4', '4~5', '5~7', '7~10', '10~']
top1['odds_band'] = pd.cut(top1['単勝オッズ'], bins=bins, labels=labels, right=False)
gr = top1.groupby('odds_band', observed=True).agg(
    n=('target_win','count'), wins=('target_win','sum'), ret=('payout','sum')
)
gr['bet'] = gr['n'] * BET
gr['roi'] = (gr['ret'] - gr['bet']) / gr['bet'] * 100
gr['wr']  = gr['wins'] / gr['n'] * 100
print(gr[['n','wins','wr','ret','roi']].to_string(float_format='%.1f'))
