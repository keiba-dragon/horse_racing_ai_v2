# coding: utf-8
"""parquet未登録・NaN多数馬の原因を深掘り"""
import pickle, sys, io
import pandas as pd, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

with open('data/raw/cache/20260530.cache.pkl', 'rb') as f:
    cache = pickle.load(f)
df = cache['result'].copy()
df.columns = df.columns.astype(object)

import pyarrow.parquet as pq
pq_df = pd.read_parquet('data/processed/all_venues_features.parquet',
                         columns=['馬名S', '日付', '着順_num', 'Ｒ', '開催'])
pq_df['日付_num'] = pd.to_numeric(pq_df['日付'], errors='coerce')
pq_df['uma'] = pq_df['馬名S'].astype(str).str.strip()

# ── 原因1: parquet完全不在 (17頭) ──
print('=' * 60)
print('【原因1】parquet完全不在 → 一度もJRAで出走記録なし')
print('= 新馬デビュー or 外国馬転入の可能性')
print('=' * 60)
all_pq = set(pq_df['uma'].unique())
today = df['馬名S'].astype(str).str.strip()
missing17 = df[~today.isin(all_pq)][['馬名S','開催','Ｒ','_nan_count']].copy()
print(missing17.to_string(index=False))

# ── 原因2: parquetにあるがNaN多数 → 出走回数が少ない ──
print()
print('=' * 60)
print('【原因2】parquetにあるがNaN多数 → 出走回数が少ない')
print('=' * 60)

high_nan = df[df['_nan_count'] >= 100]
in_pq = high_nan[today[high_nan.index].isin(all_pq)]
print(f'NaN>=100 かつ parquet在籍: {len(in_pq)}頭')

# 各馬のparquet内での出走回数を確認
sample_horses = in_pq['馬名S'].astype(str).str.strip().unique()[:15]
print(f'\n{"馬名":<20} {"pq内出走回数":>8} {"最終出走日":>10} {"NaN数":>6}')
print('-' * 50)
for h in sample_horses:
    h_rows = pq_df[pq_df['uma'] == h]
    n_races = len(h_rows)
    last_date = int(h_rows['日付_num'].max()) if n_races > 0 else 0
    nan_c = int(df[df['馬名S'].str.strip() == h]['_nan_count'].values[0])
    print(f'{h:<20} {n_races:>8}走  {last_date:>10}  {nan_c:>6}')

# ── 原因3: parquet更新が5/24止まり → 最近の出走が反映されていない ──
print()
print('=' * 60)
print('【原因3】parquet最終更新日の確認')
print('=' * 60)
print(f'parquet最新日付: {int(pq_df["日付_num"].max())} (= 2026-05-24)')
print(f'今日の日付:      20260530 (= 2026-05-30)')
print(f'差分: 6日間のデータが未反映')
print()

# 5/25以降に出走した馬が今日も出走しているか
recent_pq = pq_df[pq_df['日付_num'] >= 260525]
if len(recent_pq) == 0:
    print('5/25以降のparquetレコード: なし (= parquetは5/24が最新)')
else:
    print(f'5/25以降: {len(recent_pq)}件')

# ── 原因4: NaN特徴量の種類 → どの特徴群が埋まらないのか ──
print()
print('=' * 60)
print('【原因4】NaN特徴量の種類 (NaN>=100の馬での頻出パターン)')
print('=' * 60)
from collections import Counter
cnt = Counter()
for row in df[df['_nan_count'] >= 100]['_nan_features'].dropna():
    if row:
        for f in row.split(','):
            f = f.strip()
            if f: cnt[f] += 1

# 特徴量名のプレフィックスでグループ化
prefix_cnt = Counter()
for feat, n in cnt.items():
    prefix = feat.split('_')[0] if '_' in feat else feat[:8]
    prefix_cnt[prefix] += n

print('特徴量グループ別NaN頻度 (上位20):')
for pref, n in sorted(prefix_cnt.items(), key=lambda x: -x[1])[:20]:
    print(f'  {pref:<20}: {n:>5}件')
