# coding: utf-8
"""gap>=0.15 & odds>=band_avg*M 戦略の年別詳細"""
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

bins   = [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 1.0]
labels = ['~0.05','0.05~0.10','0.10~0.15','0.15~0.20','0.20~0.25','0.25~0.30','0.30~']
top1['gap_band'] = pd.cut(top1['gap'], bins=bins, labels=labels, right=False)

# band_avg_odds は全期間で計算（予測時点では不明なので要注意）
band_avg = top1.groupby('gap_band', observed=True)['単勝オッズ'].mean().rename('band_avg_odds')
top1 = top1.join(band_avg, on='gap_band')

# 倍率スキャン（M=1.2〜1.4）× gap閾値 0.15
print('=== 倍率細かくスキャン (gap>=0.15) ===')
print(f'{"倍率M":>6} {"件数":>6} {"的中率":>7} {"avg_odds":>9} {"ROI":>8}  オッズ上限確認')
for mult in [1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40, 1.45, 1.50]:
    sub = top1[(top1['gap'] >= 0.15) & (top1['単勝オッズ'] >= top1['band_avg_odds'] * mult)]
    if len(sub) < 20:
        break
    wins = sub['target_win'].sum()
    ret  = sub['payout'].sum()
    bet  = len(sub) * BET
    roi  = (ret - bet) / bet * 100
    wr   = wins / len(sub) * 100
    ao   = sub['単勝オッズ'].mean()
    mx   = sub['単勝オッズ'].max()
    pct95= sub['単勝オッズ'].quantile(0.95)
    print(f'{mult:>6.2f} {len(sub):>6} {wr:>7.1f}% {ao:>9.2f} {roi:>8.1f}%  max={mx:.0f}倍 95%ile={pct95:.1f}倍')

print()

# 年別（gap>=0.15 & odds>=1.3×avg）
print('=== 年別詳細 (gap>=0.15 & odds>=band_avg*1.3) ===')
sub = top1[(top1['gap'] >= 0.15) & (top1['単勝オッズ'] >= top1['band_avg_odds'] * 1.3)].copy()
yr = sub.groupby('year').agg(
    n=('target_win','count'),
    wins=('target_win','sum'),
    ret=('payout','sum'),
    avg_odds=('単勝オッズ','mean'),
)
yr['bet'] = yr['n'] * BET
yr['roi'] = (yr['ret'] - yr['bet']) / yr['bet'] * 100
yr['wr']  = yr['wins'] / yr['n'] * 100
print(yr[['n','wins','wr','avg_odds','bet','ret','roi']].to_string(float_format='%.1f'))
print(f'\n合計: {len(sub)}件 / 的中{int(sub["target_win"].sum())}件 / 投資{len(sub)*BET:,}円 / 回収{sub["payout"].sum():,.0f}円 / ROI {(sub["payout"].sum()-len(sub)*BET)/(len(sub)*BET)*100:+.1f}%')

print()

# オッズ上限も加える（高すぎるオッズを除外）
print('=== gap>=0.15 & odds>=avg*1.3 & オッズ上限フィルタ ===')
print(f'{"max_odds":>9} {"件数":>6} {"的中率":>7} {"avg_odds":>9} {"ROI":>8}')
for mx in [3, 4, 5, 6, 8, 10, 20, 99]:
    s = sub[sub['単勝オッズ'] <= mx]
    if len(s) < 10:
        continue
    w = s['target_win'].sum()
    r = s['payout'].sum()
    b = len(s) * BET
    print(f'{mx:>9} {len(s):>6} {w/len(s)*100:>7.1f}% {s["単勝オッズ"].mean():>9.2f} {(r-b)/b*100:>8.1f}%')

print()

# 参考: edge戦略との比較（edge>=0.02）
print('=== 参考: edge>=0.02 のオッズ分布 ===')
edge_top = df.loc[df.groupby('race_id')['edge'].idxmax()].copy()
edge_top['payout'] = edge_top['単勝オッズ'] * 100 * edge_top['target_win']
edge_f = edge_top[edge_top['edge'] >= 0.02]
print(f'件数:{len(edge_f)}  avg_odds:{edge_f["単勝オッズ"].mean():.2f}  max:{edge_f["単勝オッズ"].max():.0f}  95%ile:{edge_f["単勝オッズ"].quantile(0.95):.1f}')
print(f'ROI: {(edge_f["payout"].sum()-len(edge_f)*BET)/(len(edge_f)*BET)*100:+.1f}%')

# gap戦略のオッズ分布
print(f'\n=== gap>=0.15 & *1.3 のオッズ分布 ===')
print(f'件数:{len(sub)}  avg_odds:{sub["単勝オッズ"].mean():.2f}  max:{sub["単勝オッズ"].max():.0f}  95%ile:{sub["単勝オッズ"].quantile(0.95):.1f}')
