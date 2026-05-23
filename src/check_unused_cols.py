# coding: utf-8
import pandas as pd, numpy as np, sys, io, json, pickle
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, 'src')
from save_lambdarank_pace import add_pace_features, EXCLUDE, ODDS_REMOVE
from save_conditional_logit import add_new_features, NEW_FEATURE_COLS
import os
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
df = add_new_features(df)

num_cols  = df.select_dtypes(include='number').columns.tolist()
feat_cols = set(c for c in num_cols if c not in EXCLUDE and c not in ODDS_REMOVE)
print(f'現在の feat_cols: {len(feat_cols)}列')

# 全カラムから未使用で有望
keywords = ['枠', '頭数', '馬齢', '今回', '馬場', '月', '走前_馬場', '走前_距離', 'ペース', '番人気']
print('\n未使用列（有望キーワード）:')
for kw in keywords:
    matched = [c for c in df.columns if kw in c and c not in feat_cols]
    if matched:
        print(f'  {kw}: {matched[:6]}')

# 非数値だが派生で使えそうなもの
print('\n非数値列（派生特徴量の素材）:')
str_cols = df.select_dtypes(exclude='number').columns.tolist()
for kw in ['馬場', 'コース', '芝', 'ダ', 'クラス', '距離']:
    matched = [c for c in str_cols if kw in c]
    if matched:
        print(f'  {kw}: {matched[:6]}')
