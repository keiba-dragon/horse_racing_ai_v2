"""
週次データ更新スクリプト
使い方:
  1. JRA-VANアプリで結果CSVをダウンロード → data/raw/master/recent_all.csv に上書き保存
  2. python src/weekly_update.py

やること:
  - recent_all.csv の新規データ（master_all.csv 未収録分）を master_all.csv に追記
  - Parquetキャッシュを削除（次回の予測時に特徴量を再生成させる）
  - recent_all.csv はそのまま残す（次週上書きで自動クリア）
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import os

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR     = os.path.join(BASE_DIR, 'data', 'raw')
PROC_DIR    = os.path.join(BASE_DIR, 'data', 'processed')

MASTER_ALL  = os.path.join(RAW_DIR,  'master', 'master_all.csv')
RECENT_ALL  = os.path.join(RAW_DIR,  'master', 'recent_all.csv')
PARQUET     = os.path.join(PROC_DIR, 'all_venues_features.parquet')

def read_csv(path, **kwargs):
    try:
        return pd.read_csv(path, encoding='cp932', low_memory=False, **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding='utf-8', low_memory=False, **kwargs)

# ── 1. recent_all.csv の確認 ──────────────────────────────────────
if not os.path.exists(RECENT_ALL):
    print(f'ERROR: {RECENT_ALL} が見つかりません。')
    print('JRA-VANアプリで結果CSVをダウンロードして recent_all.csv として保存してください。')
    sys.exit(1)

print('recent_all.csv を読み込み中...')
df_recent = read_csv(RECENT_ALL)
df_recent['_date_num'] = pd.to_numeric(df_recent['日付'], errors='coerce')
df_recent = df_recent.dropna(subset=['_date_num'])

if df_recent.empty:
    print('ERROR: recent_all.csv にデータがありません。')
    sys.exit(1)

recent_min = int(df_recent['_date_num'].min())
recent_max = int(df_recent['_date_num'].max())
print(f'  recent_all: {len(df_recent):,}行  日付: {recent_min} 〜 {recent_max}')

# ── 2. master_all.csv の最終日付を取得（チャンク読みで軽量化）────
print('master_all.csv の最終日付を確認中...')
master_max_date = 0
master_rows = 0
for chunk in pd.read_csv(MASTER_ALL, encoding='cp932', low_memory=False,
                          chunksize=50000, usecols=['日付']):
    d = pd.to_numeric(chunk['日付'], errors='coerce').max()
    if d > master_max_date:
        master_max_date = int(d)
    master_rows += len(chunk)
print(f'  master_all: {master_rows:,}行  最終日付: {master_max_date}')

# ── 3. 新規データの抽出 ────────────────────────────────────────────
new_rows = df_recent[df_recent['_date_num'] > master_max_date].drop(columns=['_date_num'])

if new_rows.empty:
    print(f'\n追記するデータがありません（recent_all の全データが master_all 収録済み）。')
    print(f'  master_all 最終日付 {master_max_date} >= recent_all 最終日付 {recent_max}')
    sys.exit(0)

new_min = int(pd.to_numeric(new_rows['日付'], errors='coerce').min())
new_max = int(pd.to_numeric(new_rows['日付'], errors='coerce').max())
print(f'\n新規データ: {len(new_rows):,}行  日付: {new_min} 〜 {new_max}')

# ── 4. master_all.csv に追記 ──────────────────────────────────────
print('master_all.csv に追記中...')
new_rows.to_csv(MASTER_ALL, mode='a', header=False, index=False, encoding='cp932')
print(f'  追記完了: {master_rows:,} → {master_rows + len(new_rows):,}行')

# ── 5. Parquetキャッシュを削除（次回予測時に再生成） ──────────────
if os.path.exists(PARQUET):
    os.remove(PARQUET)
    print('Parquetキャッシュを削除しました（次回予測時に再生成されます）。')

# ── 6. 完了サマリー ────────────────────────────────────────────────
print()
print('=' * 50)
print('  週次更新 完了')
print('=' * 50)
print(f'  追記日付: {new_min} 〜 {new_max}')
print(f'  追記行数: {len(new_rows):,}行')
print(f'  master_all 合計: {master_rows + len(new_rows):,}行')
print()
print('次のステップ:')
print('  出馬表CSVをダウンロードして予測を実行してください。')
print('  python src/10_predict_submodel.py data/raw/出馬表形式XX月XX日.csv')
print('  ※ 初回は特徴量を再生成するため25〜30分かかります。')
