# coding: utf-8
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import save_lambdarank_pace as _slp
from save_lambdarank_pace import add_pace_features
import pandas as pd, numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_parquet(os.path.join(BASE_DIR, 'data', 'processed', 'all_venues_features.parquet'))
df['日付_num'] = pd.to_numeric(df['日付'], errors='coerce')
df['着順_num'] = pd.to_numeric(df['着順_num'], errors='coerce')
df = df.dropna(subset=['日付_num','着順_num'])
df = df[df['着順_num'] < 99]
df['race_id'] = (df['日付_num'].astype(int).astype(str) + '_' +
                 df['開催'].astype(str).str.strip() + '_' +
                 df['Ｒ'].astype(str).str.strip())
df = add_pace_features(df)

# 開催列と距離列の実際の値を確認
print('開催サンプル:', df['開催'].head(10).tolist())
print('距離サンプル:', df['距離'].head(10).tolist())

# 場名コードを抽出（フォーマット: "4東7" = 回+会場+日）
df['venue'] = df['開催'].astype(str).str.strip().str[1:-1]
print('venue サンプル:', df['venue'].head(10).tolist())
print('venue 値カウント:', df['venue'].value_counts().to_dict())

# 距離から表面抽出
df['surface'] = df['距離'].astype(str).str.strip().str.extract(r'^([芝ダ])')[0]
df = df[df['surface'].isin(['芝', 'ダ'])].copy()

# 期間別・会場別・表面別 レース数
df['race_id'] = (df['日付_num'].astype(int).astype(str) + '_' +
                 df['開催'].astype(str).str.strip() + '_' +
                 df['Ｒ'].astype(str).str.strip())

trn = df[(df['日付_num'] >= 130101) & (df['日付_num'] < 210101)]
oos = df[df['日付_num'] >= 230101]

print('\n=== 学習 2013-2020: 会場×表面別 レース数 ===')
trn_r = trn.groupby(['venue', 'surface'])['race_id'].nunique().unstack(fill_value=0)
print(trn_r.sort_values(trn_r.columns[0], ascending=False).to_string())

print('\n=== OOS 2023+: 会場×表面別 レース数 ===')
oos_r = oos.groupby(['venue', 'surface'])['race_id'].nunique().unstack(fill_value=0)
print(oos_r.sort_values(oos_r.columns[0], ascending=False).to_string())
