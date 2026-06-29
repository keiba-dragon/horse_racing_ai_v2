# coding: utf-8
"""
実力1位（prob_win最大）の馬を単勝100円で買い続けた場合のROI。
OOS predictions parquet を使用。
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd

df = pd.read_parquet('data/processed/oos_predictions.parquet')

print(f'OOS期間: {df["日付_num"].min()} ~ {df["日付_num"].max()}')
print(f'レース数: {df["race_id"].nunique()}  馬数: {len(df)}')
print(f'年別レース数:')
print(df.groupby('year')['race_id'].nunique().to_string())
print()

# 各レースで実力1位（prob_win最大）を選択
top1 = df.loc[df.groupby('race_id')['prob_win'].idxmax()].copy()
top1['payout'] = top1['単勝オッズ'] * 100 * top1['target_win']

BET = 100

total_bet    = len(top1) * BET
total_return = top1['payout'].sum()
roi = (total_return - total_bet) / total_bet * 100

print(f'=== 実力1位 単勝 ROI (OOS全体) ===')
print(f'レース数  : {len(top1)}件')
print(f'的中数    : {int(top1["target_win"].sum())}件 ({top1["target_win"].mean()*100:.1f}%)')
print(f'平均オッズ: {top1["単勝オッズ"].mean():.2f}倍')
print(f'的中時平均オッズ: {top1.loc[top1["target_win"]==1, "単勝オッズ"].mean():.2f}倍')
print(f'投資      : {total_bet:,}円')
print(f'回収      : {total_return:,.0f}円')
print(f'ROI       : {roi:+.1f}%')
print()

# 年別
print('=== 年別 ===')
yr = top1.groupby('year').agg(
    races=('target_win', 'count'),
    wins=('target_win', 'sum'),
    payout=('payout', 'sum'),
    avg_odds=('単勝オッズ', 'mean'),
)
yr['bet'] = yr['races'] * BET
yr['roi'] = (yr['payout'] - yr['bet']) / yr['bet'] * 100
yr['win_rate'] = yr['wins'] / yr['races'] * 100
print(yr[['races','wins','win_rate','avg_odds','bet','payout','roi']].to_string(float_format='%.1f'))
print()

# 参考: エッジ1位との比較
top_edge = df.loc[df.groupby('race_id')['edge'].idxmax()].copy()
top_edge['payout'] = top_edge['単勝オッズ'] * 100 * top_edge['target_win']
tr_e = top_edge['payout'].sum()
roi_e = (tr_e - total_bet) / total_bet * 100
print(f'=== 参考: エッジ1位 ===')
print(f'的中: {int(top_edge["target_win"].sum())}件 ({top_edge["target_win"].mean()*100:.1f}%)')
print(f'ROI : {roi_e:+.1f}%  (回収{tr_e:,.0f}円)')
