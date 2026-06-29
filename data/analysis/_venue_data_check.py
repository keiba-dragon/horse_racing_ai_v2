# coding: utf-8
"""会場別データ量・期間確認"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd

df = pd.read_parquet('data/processed/all_venues_features.parquet')
df['venue'] = df['開催'].str.extract(r'\d+([^\d]+)')
df['日付_num'] = pd.to_numeric(df['日付'], errors='coerce')

print(f'{"会場":<5} {"馬数":>8} {"レース数":>8} {"開始":>8} {"終了":>8} {"年数":>5}')
print('-' * 50)
for v, g in sorted(df.groupby('venue'), key=lambda x: -len(x[1])):
    races = g.groupby(['日付_num', '開催', 'Ｒ']).ngroups
    mn = int(g['日付_num'].min())
    mx = int(g['日付_num'].max())
    years = (mx // 10000 - mn // 10000 + (mx % 10000 >= mn % 10000))
    # 2桁年なので
    yr_start = 2000 + mn // 10000
    yr_end   = 2000 + mx // 10000
    n_yrs = yr_end - yr_start + 1
    print(f'{v:<5} {len(g):>8,} {races:>8,} {mn:>8} {mx:>8} {n_yrs:>5}年')
