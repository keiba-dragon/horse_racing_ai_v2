# coding: utf-8
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd, numpy as np

df = pd.read_parquet('data/processed/all_venues_features.parquet')
df['日付_num'] = pd.to_numeric(df['日付'], errors='coerce')
df['着順_num'] = pd.to_numeric(df['着順_num'], errors='coerce')
df['odds_num'] = pd.to_numeric(df['単勝オッズ'], errors='coerce')
df['pop_num']  = pd.to_numeric(df['人気'], errors='coerce')
df = df.dropna(subset=['日付_num','着順_num','odds_num','pop_num'])
df = df[df['着順_num'] < 99]
df['race_id'] = (df['日付_num'].astype(int).astype(str) + '_' +
                 df['開催'].astype(str).str.strip() + '_' +
                 df['Ｒ'].astype(str).str.strip())
df['yr2'] = df['日付_num'].astype(str).str[:2].astype(int)

market_fav = df[df['pop_num'] == 1]

print('=== 市場1番人気 全買い ===')
for period, mask in [
    ('OOS 2023-2026', (df['日付_num'] >= 230101)),
    ('Val 2021-2022', (df['日付_num'] >= 210101) & (df['日付_num'] <= 221231)),
    ('Trn 2013-2020', (df['日付_num'] >= 130101) & (df['日付_num'] < 210101)),
]:
    sub = market_fav[mask[market_fav.index]]
    if len(sub) == 0:
        continue
    print(f'\n【{period}】')
    for yr in sorted(sub['yr2'].unique()):
        s   = sub[sub['yr2'] == yr]
        won = s['着順_num'] == 1
        r   = (s.loc[won,'odds_num']*100).sum()/(len(s)*100) - 1
        avg_o = s['odds_num'].mean()
        win_o = s.loc[won,'odds_num'].mean() if won.any() else 0
        print(f'  20{yr:02d}: {len(s):5,}R  win={won.mean():.3f}  '
              f'avg_odds={avg_o:.2f}  winner_odds={win_o:.2f}  ROI={r:+.3f}')
    won = sub['着順_num'] == 1
    r   = (sub.loc[won,'odds_num']*100).sum()/(len(sub)*100) - 1
    print(f'  合計: {len(sub):5,}R  win={won.mean():.3f}  ROI={r:+.3f}')
