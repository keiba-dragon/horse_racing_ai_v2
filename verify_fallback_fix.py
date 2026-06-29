# coding: utf-8
"""06_predict_from_card.pyのフォールバックパス修正を検証
career=1走の馬に前走特徴量が正しく入るか確認する"""
import pickle, sys, io, re
import pandas as pd, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

target_date_num = 20260530

pq_df = pd.read_parquet('data/processed/all_venues_features.parquet',
    columns=['馬名S','日付','着順_num','頭数','斤量','馬番','着差',
             '前走頭数','前走斤量','前走馬番','前走着差タイム',
             '前走上り3F','前走走破タイム_sec','前走馬体重'])
pq_df['日付_num'] = pd.to_numeric(pq_df['日付'], errors='coerce')
pq_df['uma'] = pq_df['馬名S'].astype(str).str.strip()

# フォールバックパスと同じ処理を再現
df_hist = pq_df[pq_df['日付_num'] < target_date_num]
df_latest = (df_hist.sort_values('日付_num')
             .groupby('uma', sort=False).last().reset_index())

# _mae_base_map 適用 (修正後)
_mae_base_map = {
    '前走着順_num': '着順_num',
    '前走馬体重':   '馬体重',
    '前走頭数':     '頭数',
    '前走斤量':     '斤量',
    '前走馬番':     '馬番',
}
for dst, src in _mae_base_map.items():
    if dst in df_latest.columns and src in df_latest.columns:
        df_latest[dst] = df_latest[dst].fillna(df_latest[src])

# 前走着差タイム補完
if '前走着差タイム' in df_latest.columns and '着差' in df_latest.columns:
    _cs = df_latest['着差'].astype(str).str.replace('----','NaN').pipe(pd.to_numeric, errors='coerce')
    df_latest['前走着差タイム'] = df_latest['前走着差タイム'].fillna(_cs)

# career=1走の馬を確認
career_count = pq_df.groupby('uma').size().rename('career')
df_latest = df_latest.join(career_count, on='uma')

c1 = df_latest[df_latest['career'] == 1]
sample = ['クロレ','ペイシャトサモア','ドンレミラピュセル','イクオクコウネン','アメリカンイズム']

print('== career=1走の馬のフォールバック後の前走特徴量 ==')
print(f'{"馬名":<20} {"前走頭数":>6} {"前走斤量":>6} {"前走馬番":>6} {"前走着差":>8}')
print('-' * 55)
for h in sample:
    row = df_latest[df_latest['uma'] == h]
    if len(row) == 0:
        print(f'  {h:<20} (未ヒット)')
        continue
    r = row.iloc[0]
    print(f'  {h:<20} {r["前走頭数"]:>6} {r["前走斤量"]:>6} {r["前走馬番"]:>6} {r["前走着差タイム"]:>8.1f}')

# 改善後の異常NaN数
print()
print('== 修正後の前走頭数 NaN状況 ==')
total = len(df_latest)
nan_after = df_latest['前走頭数'].isna().sum()
# career>=1なのにNaN
c1_nan = (df_latest['前走頭数'].isna() & (df_latest['career'] >= 1)).sum()
print(f'前走頭数 NaN: {nan_after}/{total} ({nan_after/total*100:.1f}%)')
print(f'  うち career>=1: {c1_nan}頭  (修正前: 34頭)')
print(f'  残り {c1_nan}頭の内訳:')
if c1_nan > 0:
    still_nan = df_latest[df_latest['前走頭数'].isna() & (df_latest['career'] >= 1)][['uma','career']].head(10)
    print(still_nan.to_string(index=False))
