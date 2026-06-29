# coding: utf-8
"""前走系異常NaNの原因を career=1走 / career=2走 別に特定"""
import pickle, sys, io
import pandas as pd, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

with open('data/raw/cache/20260530.cache.pkl', 'rb') as f:
    cache = pickle.load(f)
df = cache['result'].copy()
df.columns = df.columns.astype(object)

pq_df = pd.read_parquet('data/processed/all_venues_features.parquet',
                         columns=['馬名S', '日付', '着順_num', 'Ｒ', '開催',
                                  '前走頭数', '前走着差タイム', '前走_surface', '距離変化_前走'])
pq_df['日付_num'] = pd.to_numeric(pq_df['日付'], errors='coerce')
pq_df['uma'] = pq_df['馬名S'].astype(str).str.strip()

career_count = pq_df.groupby('uma').size().rename('career')
df['_uma'] = df['馬名S'].astype(str).str.strip()
df = df.join(career_count, on='_uma')
df['career'] = df['career'].fillna(0).astype(int)

# 前走頭数 異常NaN の馬
if '前走頭数' not in df.columns:
    print('前走頭数 列なし')
    import sys; sys.exit()

mask_abn = df['前走頭数'].isna() & (df['career'] >= 1)
abn = df[mask_abn].copy()

# ── career=1走の馬 ──
print('='*60)
print('【career=1走の馬】')
print('parquetに1行しかない = デビュー戦記録のみ')
print('その1行の前走頭数 = NaN (デビュー戦に前走なし) → 正常NaN？')
print('='*60)
c1 = abn[abn['career'] == 1]
print(f'該当: {len(c1)}頭')

# 実際にparquetで確認
sample = c1['馬名S'].astype(str).str.strip().unique()[:5]
for h in sample:
    rows = pq_df[pq_df['uma'] == h][['日付', '開催', 'Ｒ', '着順_num', '前走頭数']].sort_values('日付')
    print(f'\n  {h}:')
    print(rows.to_string(index=False))
print()
print('→ 1走前のデータは「その1行自体」の race data として存在する')
print('  つまり 前走=デビュー戦 のデータを今日の予測に使えていない')

# ── career=2走の馬 ──
print()
print('='*60)
print('【career=2走の馬】')
print('parquetに2行ある = 2走目の行に前走データが入るべき')
print('='*60)
c2 = abn[abn['career'] == 2]
print(f'該当: {len(c2)}頭')

sample2 = c2['馬名S'].astype(str).str.strip().unique()[:5]
for h in sample2:
    rows = pq_df[pq_df['uma'] == h][['日付', '開催', 'Ｒ', '着順_num', '前走頭数', '前走着差タイム']].sort_values('日付')
    print(f'\n  {h}:')
    print(rows.to_string(index=False))
print()
print('→ 2行目(最新行)の前走頭数がNaNなら特徴量エンジニアリングのバグ')

# ── 特徴量エンジニアリングコードを確認 ──
print()
print('='*60)
print('【予測コードでの特徴量取得方法の確認】')
print('='*60)

# 06_predict_from_card.py で特徴量をどう取得しているか
# parquetから最新行を取得 → その行の前走系列を使う
# 今日の出走馬でcareer=2走の馬の「最新行」
for h in sample2[:3]:
    rows = pq_df[pq_df['uma'] == h].sort_values('日付_num')
    latest = rows.iloc[-1]
    print(f'  {h} 最新行: 日付={latest["日付"]}  前走頭数={latest["前走頭数"]}')
