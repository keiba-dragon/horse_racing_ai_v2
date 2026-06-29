# coding: utf-8
"""parquet未登録馬の原因調査"""
import pickle, sys, io
import pandas as pd, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# 今日のキャッシュ
with open('data/raw/cache/20260530.cache.pkl', 'rb') as f:
    cache = pickle.load(f)
df = cache['result'].copy()
df.columns = df.columns.astype(object)

# NaN数が多い馬 = parquet未登録馬
print('== NaN特徴量が多い馬 (未登録候補) ==')
nan_horses = df[df['_nan_count'] >= 100][
    ['馬名S', '開催', 'Ｒ', '_nan_count', '_nan_total', 'clogit_calib']
].copy()
print(f'_nan_count >= 100: {len(nan_horses)}頭')
print(nan_horses.sort_values('_nan_count', ascending=False).to_string(index=False))

# parquetに何が入っているか確認
print('\n== parquet内の馬名一覧サンプル ==')
import pyarrow.parquet as pq
schema = pq.read_schema('data/processed/all_venues_features.parquet')
cols = schema.names
has_uma = '馬名' in cols or '馬名S' in cols
print(f'parquet列数: {len(cols)}')
uma_col = '馬名' if '馬名' in cols else ('馬名S' if '馬名S' in cols else None)
print(f'馬名列: {uma_col}')

if uma_col:
    pq_df = pd.read_parquet('data/processed/all_venues_features.parquet',
                             columns=[uma_col, '日付', '開催', 'Ｒ'])
    pq_df['日付_num'] = pd.to_numeric(pq_df['日付'], errors='coerce')
    # 最新データ（2026年分）の馬名
    recent = pq_df[pq_df['日付_num'] >= 260101]
    print(f'parquet 2026年レコード数: {len(recent):,}')
    recent_horses = set(recent[uma_col].astype(str).str.strip().unique())
    print(f'parquet 2026年 ユニーク馬数: {len(recent_horses):,}')

    # 今日の馬がparquetにいるか
    today_horses = df['馬名S'].astype(str).str.strip().unique()
    print(f'\n今日の出走馬数: {len(today_horses)}頭')

    all_pq_horses = set(pq_df[uma_col].astype(str).str.strip().unique())
    missing = [h for h in today_horses if h not in all_pq_horses]
    present = [h for h in today_horses if h in all_pq_horses]
    print(f'parquetに存在: {len(present)}頭')
    print(f'parquetに不在: {len(missing)}頭')

    print(f'\n-- parquet不在の馬 --')
    for h in sorted(missing):
        row = df[df['馬名S'].str.strip() == h].iloc[0]
        nan_c = row.get('_nan_count', '?')
        print(f'  {h}  ({row["開催"]} {int(row["Ｒ"])}R)  NaN={nan_c}')

    # 存在する馬の最終出走日
    print(f'\n-- parquet在籍馬の最終データ日付分布 --')
    latest = pq_df.groupby(uma_col)['日付_num'].max()
    today_latest = latest[latest.index.isin(today_horses)]
    print(today_latest.value_counts().sort_index(ascending=False).head(10).to_string())

    # parquetの最新日付
    print(f'\nparquet全体の最新日付: {pq_df["日付_num"].max():.0f}')
    print(f'parquet全体の最古日付: {pq_df["日付_num"].min():.0f}')
