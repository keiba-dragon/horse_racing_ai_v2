# coding: utf-8
"""前走系異常NaN 34頭の原因を特定"""
import pickle, sys, io
import pandas as pd, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

with open('data/raw/cache/20260530.cache.pkl', 'rb') as f:
    cache = pickle.load(f)
df = cache['result'].copy()
df.columns = df.columns.astype(object)

import pyarrow.parquet as pq
pq_df = pd.read_parquet('data/processed/all_venues_features.parquet',
                         columns=['馬名S', '日付', '着順_num', 'Ｒ', '開催', '前走頭数', '前走着差タイム'])
pq_df['日付_num'] = pd.to_numeric(pq_df['日付'], errors='coerce')
pq_df['uma'] = pq_df['馬名S'].astype(str).str.strip()

# careerカウント
career_count = pq_df.groupby('uma').size().rename('career')
df['_uma'] = df['馬名S'].astype(str).str.strip()
df = df.join(career_count, on='_uma')
df['career'] = df['career'].fillna(0).astype(int)

# 前走頭数が異常NaN (career>=1 なのにNaN)
if '前走頭数' in df.columns:
    mask = df['前走頭数'].isna() & (df['career'] >= 1)
    abnormal = df[mask][['馬名S', '開催', 'Ｒ', 'career', '_nan_count']].copy()
    print(f'前走頭数 異常NaN: {len(abnormal)}頭')
    print(abnormal.sort_values('career', ascending=False).to_string(index=False))

    print()
    # parquet内でのこれらの馬の記録を確認
    print('== parquet内の記録 ==')
    for h in abnormal['馬名S'].astype(str).str.strip().unique()[:10]:
        rows = pq_df[pq_df['uma'] == h].sort_values('日付_num')
        print(f'\n{h} ({len(rows)}走):')
        print(rows[['日付', '開催', 'Ｒ', '着順_num', '前走頭数', '前走着差タイム']].to_string(index=False))
else:
    print('前走頭数列なし')
    print('利用可能な前走系列:', [c for c in df.columns if '前走' in c][:10])
