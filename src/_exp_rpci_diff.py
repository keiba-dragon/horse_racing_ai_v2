# coding: utf-8
"""
実験: 自馬_対_レース_RPCI差 特徴量を追加して再学習
- 近5走_RPCI平均 からレース内平均を引いた符号付き差分
- 正=このレースで相対的に後方待機型, 負=相対的に先行型
- リークなし・馬別・既存parquetから動的計算
- 保存先: models/exp_rpci_diff/  (本番に影響なし)
"""
import os, sys, subprocess
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ORIG_PARQUET = os.path.join(BASE_DIR, 'data', 'processed', 'all_venues_features.parquet')
TMP_PARQUET  = os.path.join(BASE_DIR, 'data', 'processed', '_exp_rpci_diff.parquet')
OUT_DIR      = os.path.join(BASE_DIR, 'models', 'exp_rpci_diff')
os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. 既存parquetを読み込み、新特徴量を追加 ──────────────────────
print('parquet読み込み中...')
df = pd.read_parquet(ORIG_PARQUET)

RPCI_COL = '近5走_RPCI平均'
if RPCI_COL not in df.columns:
    print(f'ERROR: {RPCI_COL} がparquetにありません')
    sys.exit(1)

# race_id 構築（save_conditional_logit.pyと同じ方法）
df['日付_num'] = pd.to_numeric(df['日付'], errors='coerce')
df['race_id'] = (
    df['日付_num'].astype(int).astype(str) + '_' +
    df['開催'].astype(str).str.strip() + '_' +
    df['Ｒ'].astype(str).str.strip()
)

# レース内RPCI平均（各レースの全馬平均）
race_rpci_mean = df.groupby('race_id')[RPCI_COL].transform('mean')
df['自馬_対_レース_RPCI差'] = df[RPCI_COL] - race_rpci_mean

# 追加状況確認
new_col = df['自馬_対_レース_RPCI差']
nan_pct = new_col.isna().mean()
print(f'自馬_対_レース_RPCI差: 非NaN={new_col.notna().sum():,}行  NaN率={nan_pct:.1%}')
print(f'  mean={new_col.mean():.4f}  std={new_col.std():.4f}  '
      f'min={new_col.min():.4f}  max={new_col.max():.4f}')

# race_id / 日付_num は一時列なので削除（save_conditional_logit内で再計算される）
df = df.drop(columns=['race_id', '日付_num'])

# ── 2. 一時parquetに保存 ──────────────────────────────────────────
print(f'\n一時parquet保存: {TMP_PARQUET}')
df.to_parquet(TMP_PARQUET, index=False)
print(f'行数: {len(df):,}  列数: {len(df.columns):,}')

# ── 3. save_conditional_logit.py を実験モードで実行 ───────────────
print(f'\n--- conditional_logit 学習 ---')
r = subprocess.run(
    [sys.executable,
     os.path.join(BASE_DIR, 'src', 'save_conditional_logit.py'),
     '--data-file', TMP_PARQUET,
     '--out-dir',   OUT_DIR],
    cwd=BASE_DIR,
    capture_output=False,
    text=True,
)
if r.returncode != 0:
    print('ERROR: save_conditional_logit failed')
    sys.exit(1)

# ── 4. save_final_model.py を実験モードで実行 ──────────────────────
print(f'\n--- final_model 学習 + OOS ROI ---')
r2 = subprocess.run(
    [sys.executable,
     os.path.join(BASE_DIR, 'src', 'save_final_model.py'),
     '--data-file',   TMP_PARQUET,
     '--clogit-dir',  OUT_DIR,
     '--out-dir',     OUT_DIR],
    cwd=BASE_DIR,
    capture_output=False,
    text=True,
)
if r2.returncode != 0:
    print('ERROR: save_final_model failed')
    sys.exit(1)

print(f'\n完了。モデル保存先: {OUT_DIR}')
