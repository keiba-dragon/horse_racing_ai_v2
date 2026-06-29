# -*- coding: utf-8 -*-
import sys, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
df = pd.read_parquet('data/processed/all_venues_features.parquet')

df['クラス_rank'] = pd.to_numeric(df['クラス_rank'], errors='coerce')
surface = df['距離'].astype(str).str.strip().str.extract(r'^([芝ダ])')[0].fillna('不明')
dist_m = pd.to_numeric(df['距離'].astype(str).str.extract(r'(\d+)')[0], errors='coerce')
seg = df[(surface == 'ダ') & (dist_m > 1400) & (df['クラス_rank'] != 1.0)].copy()
print(f'ダ長レコード数: {len(seg):,}')

new_cands = [
    '同距離帯_平均着順_近5走',
    '同馬場_平均着順_近5走',
    '芝ダ一致_平均着順_近5走',
    '距離延長時_平均着順_近5走',
    '距離短縮時_平均着順_近5走',
    '相手レベル_平均着順',
    '馬体重トレンド_近5走',
    '近3走_体重増減合計',
    '母父馬_勝率',
    '馬_r20_勝率',
    '馬コース_r20_勝率',
    '騎手コース距離_r100_勝率',
    '騎手距離_r100_勝率',
    '騎手馬場_r100_勝率',
    '調教師_r200_勝率',
    '近3走_勝率',
    '近5走_複勝率',
    'コース枠_r200_複勝率',
    '良馬場_平均着順_近5走',
]

print()
print(f"{'特徴量':<35} NaN率")
print('-'*50)
for c in new_cands:
    if c in seg.columns:
        nan_rate = seg[c].isna().mean()
        print(f'{c:<35} {nan_rate:.1%}')
    else:
        print(f'{c:<35} *** 存在しない ***')
