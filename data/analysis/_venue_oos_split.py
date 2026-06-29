# coding: utf-8
"""会場別に適切なOOSカットオフを計算（学習:OOS = 75:25 目安）"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd
import numpy as np

df = pd.read_parquet('data/processed/all_venues_features.parquet')
df['venue'] = df['開催'].str.extract(r'\d+([^\d]+)')
df['日付_num'] = pd.to_numeric(df['日付'], errors='coerce')
df['year_full'] = 2000 + (df['日付_num'] // 10000)

# 主要10会場のみ
MAIN_VENUES = ['東', '中', '阪', '京', '名', '新', '小', '福', '札', '函']
df = df[df['venue'].isin(MAIN_VENUES)]

print('=== 会場別 年ごとレース数 ===')
yr_race = df.groupby(['venue', 'year_full']).apply(
    lambda g: g.groupby(['日付_num', '開催', 'Ｒ']).ngroups
).rename('races').reset_index()

# 75%カットオフ年を計算
print(f'\n{"会場":<5} {"総レース":>8} {"75%時点カットオフ":>16} {"OOS件数":>8} {"学習件数":>8}')
print('-' * 55)

venue_oos = {}
for v in MAIN_VENUES:
    sub = yr_race[yr_race['venue'] == v].sort_values('year_full')
    if len(sub) == 0:
        continue
    total = sub['races'].sum()
    cumsum = sub['races'].cumsum()
    # 75%地点の年
    cutoff_idx = (cumsum < total * 0.75).sum()
    if cutoff_idx >= len(sub):
        cutoff_idx = len(sub) - 1
    cutoff_year = int(sub.iloc[cutoff_idx]['year_full'])
    oos_races  = int(sub[sub['year_full'] >  cutoff_year]['races'].sum())
    train_races = int(sub[sub['year_full'] <= cutoff_year]['races'].sum())
    venue_oos[v] = cutoff_year + 1  # カットオフ翌年以降がOOS
    print(f'{v:<5} {total:>8,} {cutoff_year+1:>16} {oos_races:>8,} {train_races:>8,}')

print()
print('=== 参考: 固定カットオフ2021の場合 ===')
print(f'{"会場":<5} {"学習(~2020)":>10} {"OOS(2021+)":>10} {"比率":>8}')
for v in MAIN_VENUES:
    sub = yr_race[yr_race['venue'] == v]
    tr  = int(sub[sub['year_full'] <= 2020]['races'].sum())
    te  = int(sub[sub['year_full'] >= 2021]['races'].sum())
    if tr + te == 0:
        continue
    ratio = te / (tr + te) * 100
    print(f'{v:<5} {tr:>10,} {te:>10,} {ratio:>7.1f}%')

print()
print('=== 会場別年ごとのレース数（確認用）===')
pivot = yr_race.pivot(index='year_full', columns='venue', values='races').fillna(0).astype(int)
pivot = pivot[[c for c in MAIN_VENUES if c in pivot.columns]]
print(pivot.to_string())
