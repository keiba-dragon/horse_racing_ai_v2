# coding: utf-8
"""fillna修正の効果をparquetデータで検証 (parquet再生成なしで確認)"""
import sys, io
import pandas as pd, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

pq_df = pd.read_parquet('data/processed/all_venues_features.parquet',
    columns=['馬名S', '日付', '着順_num', '頭数', '斤量', '馬番', '着差',
             '前走頭数', '1走前_頭数', '前走斤量', '1走前_斤量',
             '前走馬番', '1走前_馬番', '前走着差タイム', '1走前_前走着差タイム'])
pq_df['日付_num'] = pd.to_numeric(pq_df['日付'], errors='coerce')
pq_df['uma'] = pq_df['馬名S'].astype(str).str.strip()

# career=2走で前走頭数NaNの馬
career = pq_df.groupby('uma').size()
c2 = career[career == 2].index

target_horses = ['ウインアトム', 'マテンロウプリンス', 'ラルクアンレーヴ', 'ボッチョ', 'パピヨンヴェール']

print('=== 修正シミュレーション ===')
print('修正前 vs 修正後 (fillna適用)')
print()

def clean_time_diff(series):
    return series.astype(str).str.replace('----', 'NaN').pipe(pd.to_numeric, errors='coerce')

for h in target_horses:
    rows = pq_df[pq_df['uma'] == h].sort_values('日付_num').copy()
    if len(rows) < 2:
        continue
    latest = rows.iloc[-1]
    print(f'{h} (最新行 日付={latest["日付"]}):')
    print(f'  前走頭数  修正前={latest["前走頭数"]}  →  1走前_頭数={latest["1走前_頭数"]}  修正後={latest["前走頭数"] if pd.notna(latest["前走頭数"]) else latest["1走前_頭数"]}')
    print(f'  前走斤量  修正前={latest["前走斤量"]}  →  1走前_斤量={latest["1走前_斤量"]}  修正後={latest["前走斤量"] if pd.notna(latest["前走斤量"]) else latest["1走前_斤量"]}')
    print(f'  前走馬番  修正前={latest["前走馬番"]}  →  1走前_馬番={latest["1走前_馬番"]}  修正後={latest["前走馬番"] if pd.notna(latest["前走馬番"]) else latest["1走前_馬番"]}')

    # 前走着差タイム: 1走前の着差 shift(1)
    prev_row = rows.iloc[-2]
    prev_着差_sec = clean_time_diff(pd.Series([prev_row['着差']])).iloc[0]
    print(f'  前走着差タイム 修正前={latest["前走着差タイム"]}  →  1走前_着差={prev_着差_sec}  修正後={latest["前走着差タイム"] if pd.notna(latest["前走着差タイム"]) else prev_着差_sec}')
    print()

# 修正で埋まる馬の数を推定
print('=== 修正効果の推計 ===')
# 前走頭数がNaNで1走前_頭数が有効な行数
n_fix_tousuu = (pq_df['前走頭数'].isna() & pq_df['1走前_頭数'].notna()).sum()
n_fix_kinryou = (pq_df['前走斤量'].isna() & pq_df['1走前_斤量'].notna()).sum()
n_fix_umaban = (pq_df['前走馬番'].isna() & pq_df['1走前_馬番'].notna()).sum()

# 着差 → 前走着差タイム の補完可能数
pq_df['_着差_sec'] = clean_time_diff(pq_df['着差'])
pq_df['_1走前_着差'] = pq_df.groupby('uma', sort=False)['_着差_sec'].shift(1)
n_fix_chakusa = (pq_df['前走着差タイム'].isna() & pq_df['_1走前_着差'].notna()).sum()

total = len(pq_df)
print(f'parquet総行数: {total:,}')
print(f'前走頭数   補完可能: {n_fix_tousuu:,}行 ({n_fix_tousuu/total*100:.1f}%)')
print(f'前走斤量   補完可能: {n_fix_kinryou:,}行 ({n_fix_kinryou/total*100:.1f}%)')
print(f'前走馬番   補完可能: {n_fix_umaban:,}行 ({n_fix_umaban/total*100:.1f}%)')
print(f'前走着差タイム 補完可能: {n_fix_chakusa:,}行 ({n_fix_chakusa/total*100:.1f}%)')
