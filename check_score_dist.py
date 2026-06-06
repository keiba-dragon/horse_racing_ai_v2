# coding: utf-8
"""
check_score_dist.py - セグメント別スコア分散・タイ頻度チェック
特徴量が少ない場合のスコア均一化リスクを確認する
"""
import sys, os, pickle
import numpy as np
import pandas as pd

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from save_conditional_logit import prepare, segment_softmax, DATA_FILE
from save_v3 import add_computed_features

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'final_model.pkl')

with open(MODEL_PATH, 'rb') as f:
    pkg = pickle.load(f)

arts = pkg['artifacts']

# セグメント定義
def get_segment(df):
    df = df.copy()
    df['日付_num'] = pd.to_numeric(df['日付'], errors='coerce')
    df['着順_num'] = pd.to_numeric(df['着順_num'], errors='coerce')
    df = df.dropna(subset=['日付_num', '着順_num'])
    df = df[df['着順_num'] < 99]
    df['race_id'] = (df['日付_num'].astype(int).astype(str) + '_' +
                     df['開催'].astype(str).str.strip() + '_' +
                     df['Ｒ'].astype(str).str.strip())
    df['surface'] = (df['距離'].astype(str).str.strip()
                     .str.extract(r'^([芝ダ])')[0].fillna('不明'))
    dm = pd.to_numeric(df['距離'].astype(str).str.extract(r'(\d+)')[0], errors='coerce')
    df['dist_m'] = dm
    baba_map = {'良': 0, '稍重': 1, '重': 2, '不良': 3}
    for col in df.columns:
        if '馬場状態' in col:
            df[col] = df[col].map(baba_map)
    return df, dm

SEGS = {
    '芝短': lambda df, dm: (df['surface'] == '芝') & (dm <= 1400),
    '芝中': lambda df, dm: (df['surface'] == '芝') & (dm > 1400) & (dm <= 2000),
    '芝長': lambda df, dm: (df['surface'] == '芝') & (dm > 2000),
    'ダ短': lambda df, dm: (df['surface'] == 'ダ') & (dm <= 1400),
    'ダ':   lambda df, dm: (df['surface'] == 'ダ') & (dm > 1400),
}

print('データ読み込み中...')
raw = pd.read_parquet(DATA_FILE)
df, dm = get_segment(raw)
df = add_computed_features(df)
for col in df.select_dtypes(include='object').columns:
    if any(kw in col for kw in ['馬場状態']):
        baba_map = {'良': 0, '稍重': 1, '重': 2, '不良': 3}
        df[col] = df[col].map(baba_map)

oos = df[df['日付_num'] >= 250101].copy()  # 2025以降をOOSとして確認
dm_oos = oos['dist_m']

print(f'\nOOSサンプル (2025+): {len(oos):,}行  {oos["race_id"].nunique()}R\n')

print('=' * 70)
print(f'  {"セグメント":6s}  {"特徴数":4s}  {"レース数":6s}  {"NaN率最大":8s}  '
      f'{"完全NaN":6s}  {"スコア分散<0.001":10s}  {"タイ1位(%)":8s}')
print('-' * 70)

for seg_name, mask_fn in SEGS.items():
    if seg_name not in arts:
        print(f'  {seg_name}: artifactなし')
        continue
    art = arts[seg_name]
    feats = art['feat_cols']

    mask = mask_fn(oos, dm_oos)
    seg_df = oos[mask].copy()

    # 芝長は障害除外
    if seg_name == '芝長' and 'クラス_rank' in seg_df.columns:
        cr = pd.to_numeric(seg_df['クラス_rank'], errors='coerce')
        seg_df = seg_df[cr.notna()]

    # ダ短は新馬除外
    if seg_name == 'ダ短' and 'クラス_rank' in seg_df.columns:
        seg_df = seg_df[seg_df['クラス_rank'] != 1.0]

    n_races = seg_df['race_id'].nunique()
    if n_races == 0:
        continue

    # 全特徴量を数値変換（baba含む）
    baba_map_local = {'良': 0, '稍重': 1, '重': 2, '不良': 3}
    for col in seg_df.columns:
        if '馬場状態' in col:
            seg_df[col] = seg_df[col].map(baba_map_local)

    # NaN率
    nan_rates = {}
    for f in feats:
        if f in seg_df.columns:
            seg_df[f] = pd.to_numeric(seg_df[f], errors='coerce')
            nan_rates[f] = seg_df[f].isna().mean()
        else:
            nan_rates[f] = 1.0
    max_nan = max(nan_rates.values()) if nan_rates else 0

    # スコア計算
    seg_df = seg_df.dropna(subset=['race_id'])
    seg_sorted = seg_df.sort_values('race_id').reset_index(drop=True)
    valid_feats = [f for f in feats if f in seg_sorted.columns]
    try:
        X, _, gs, n, *_ = prepare(seg_sorted, valid_feats,
                                   scaler=art['scaler'], top_idx=None, top_idx3=None)
        lin = X @ art['coef']
        probs = segment_softmax(lin, gs, n)
        seg_sorted['_prob'] = probs
    except Exception as e:
        print(f'  {seg_name}: スコア計算エラー {e}')
        continue

    # レース内スコア分散
    race_var = seg_sorted.groupby('race_id')['_prob'].var()
    n_low_var = (race_var < 0.001).sum()
    frac_low = n_low_var / len(race_var)

    # 全馬NaN（スコアが均一になりやすい）= 分散 < 1e-10
    n_near_zero_var = (race_var < 1e-10).sum()

    # タイ1位の頻度
    seg_sorted['_rank'] = seg_sorted.groupby('race_id')['_prob'].rank(
        ascending=False, method='min')
    ties_at1 = seg_sorted[seg_sorted['_rank'] == 1].groupby('race_id').size()
    tie_races = (ties_at1 > 1).sum()
    tie_frac = tie_races / n_races

    print(f'  {seg_name:6s}  {len(feats):4d}個  {n_races:6d}R  '
          f'{max_nan*100:7.1f}%  {n_near_zero_var:5d}R  '
          f'{frac_low*100:8.1f}%  {tie_frac*100:7.1f}%')

    # NaN率詳細
    for f, nr in sorted(nan_rates.items(), key=lambda x: -x[1]):
        flag = ' ★高NaN' if nr > 0.3 else ''
        print(f'      {f}: NaN={nr*100:.1f}%{flag}')

print('=' * 70)
