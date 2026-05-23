# coding: utf-8
"""利用可能な列と現在の EXCLUDE/ODDS_REMOVE を確認"""
import sys, os
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import save_lambdarank_pace as _slp
from save_lambdarank_pace import add_pace_features, EXCLUDE, ODDS_REMOVE
from save_conditional_logit import add_new_features

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, 'data', 'processed', 'all_venues_features.parquet')

df = pd.read_parquet(DATA_FILE)
df['日付_num'] = pd.to_numeric(df['日付'], errors='coerce')
df['着順_num'] = pd.to_numeric(df['着順_num'], errors='coerce')
df = df.dropna(subset=['日付_num', '着順_num'])
df = df[df['着順_num'] < 99]
df['race_id'] = (df['日付_num'].astype(int).astype(str) + '_' +
                 df['開催'].astype(str).str.strip() + '_' +
                 df['Ｒ'].astype(str).str.strip())
df = add_pace_features(df)
df = add_new_features(df)

print('=== 全列 (型別) ===')
print(f'総列数: {len(df.columns)}')
print(f'数値列: {len(df.select_dtypes(include="number").columns)}')
print(f'オブジェクト列: {len(df.select_dtypes(include="object").columns)}')

print('\n=== EXCLUDE ===')
print(sorted(EXCLUDE))

print('\n=== ODDS_REMOVE ===')
print(sorted(ODDS_REMOVE))

num_cols = df.select_dtypes(include='number').columns.tolist()
feat_cols = [c for c in num_cols if c not in EXCLUDE and c not in ODDS_REMOVE]
print(f'\n=== 現在の特徴量列: {len(feat_cols)}列 ===')

print('\n=== 除外されているオブジェクト列（サンプル値付き） ===')
obj_cols = df.select_dtypes(include='object').columns.tolist()
race_context_keywords = ['馬場', '条件', 'クラス', 'グレード', 'ランク', '種別', '天候', '状態', '重']
for col in obj_cols:
    sample = df[col].dropna().head(5).tolist()
    is_context = any(kw in col for kw in race_context_keywords)
    marker = ' *** レース条件?' if is_context else ''
    print(f'  {col}: {sample}{marker}')

print('\n=== ODDS_REMOVE に含まれている有用そうな列 ===')
for col in sorted(ODDS_REMOVE):
    if col in df.columns:
        sample = df[col].dropna().head(3).tolist()
        print(f'  {col}: {sample}')
