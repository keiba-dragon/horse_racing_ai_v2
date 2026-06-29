# coding: utf-8
import pandas as pd, sys
sys.stdout.reconfigure(encoding='utf-8')
pq = pd.read_parquet('data/processed/all_venues_features.parquet')
print('全列数:', len(pq.columns))
# 騎手を含む列
jky = [c for c in pq.columns if '騎手' in str(c)]
print('騎手系列:', jky[:5])
# 着順を含む列
rank = [c for c in pq.columns if '着順' in str(c)]
print('着順系列:', rank[:5])
# 日付
date = [c for c in pq.columns if '日付' in str(c)]
print('日付系列:', date[:3])
# 開催
kai = [c for c in pq.columns if '開催' in str(c)]
print('開催系列:', kai[:3])
