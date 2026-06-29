# coding: utf-8
"""買い推奨 × NaN率 長期ROI分析 (roi_model.pkl ベース)"""
import os, sys, pickle
import pandas as pd
import numpy as np

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ── モデル読み込み ──────────────────────────────────────────
with open(os.path.join(base, 'models', 'roi_model.pkl'), 'rb') as f:
    m = pickle.load(f)

artifacts     = m['artifacts']
feat_cols     = m['feat_cols']
FACTOR_MAIDEN = m['factor_maiden']
FACTOR_OTHER  = m['factor_other']

# ── データ読み込み ──────────────────────────────────────────
feat = pd.read_parquet(os.path.join(base, 'data', 'processed', 'all_venues_features.parquet'))

# race_id / surface 構築（save_final_model.py と同じ）
feat['race_id'] = (feat['日付_num'].astype(int).astype(str) + '_' +
                   feat['開催'].astype(str).str.strip() + '_' +
                   feat['Ｒ'].astype(str).str.strip())
feat['surface'] = feat['距離'].astype(str).str.strip().str.extract(r'^([芝ダ])')[0].fillna('不明')
feat = feat[feat['surface'].isin(['芝', 'ダ'])].copy()

# ── OOS: 2023+ ──────────────────────────────────────────────
oos = feat[feat['日付_num'] >= 230101].sort_values('race_id').reset_index(drop=True)
print(f'OOS行数: {len(oos):,}  レース数: {oos["race_id"].nunique():,}')

# ── NaN率計算（予測前、impute前）──────────────────────────
use_cols = [c for c in feat_cols if c in oos.columns]
miss_cols = [c for c in feat_cols if c not in oos.columns]
print(f'特徴量: {len(feat_cols)}列 (parquetにある: {len(use_cols)}, ない: {len(miss_cols)})')

oos['nan_count'] = oos[use_cols].isna().sum(axis=1) + len(miss_cols)  # ない列はNaN扱い
oos['nan_pct']   = oos['nan_count'] / len(feat_cols)

# ── 予測 ───────────────────────────────────────────────────
def segment_softmax(lin, gs, n):
    out = np.zeros_like(lin, dtype=float)
    idx = 0
    for g in gs:
        seg = lin[idx:idx+g]
        seg = seg - seg.max()
        exp = np.exp(seg)
        out[idx:idx+g] = exp / exp.sum()
        idx += g
    return out

def predict_surface(df_s, art):
    s = df_s.sort_values('race_id').reset_index(drop=True)
    fc = art['feat_cols']
    # 不足列をNaNで補完してからfillna(0)
    for c in fc:
        if c not in s.columns:
            s[c] = np.nan
    Xraw = s[fc].fillna(0).values.astype(float)
    X    = art['scaler'].transform(Xraw)
    if art['poly2'] is not None and art['top_idx'] is not None:
        Xsub = X[:, art['top_idx']]
        Xi   = art['poly2'].transform(Xsub)[:, len(art['top_idx']):]
        Xi2  = art['inter_scaler2'].transform(Xi)
        X    = np.hstack([X, Xi2])
    gs  = s.groupby('race_id', sort=False).size().values
    lin = X @ art['coef']
    raw = segment_softmax(lin, gs, len(s))
    return art['isotonic'].predict(raw), s.index

calib_arr = np.zeros(len(oos))
for surf in ['芝', 'ダ']:
    mask   = (oos['surface'] == surf).values
    oos_s  = oos[mask].copy()
    if len(oos_s) == 0:
        continue
    calib, orig_idx = predict_surface(oos_s, artifacts[surf])
    # oos_s.indexはoos内の行番号
    oos_s_index = np.where(mask)[0]
    calib_arr[oos_s_index] = calib

oos['calib_prob']  = calib_arr
oos['market_prob'] = 1.0 / pd.to_numeric(oos['単勝オッズ'], errors='coerce').clip(lower=1.0)
factor_arr         = np.where(oos['クラス_rank'] == 2, FACTOR_MAIDEN, FACTOR_OTHER)
oos['score']       = oos['calib_prob'] - factor_arr * oos['market_prob']
oos['ev_score']    = oos['calib_prob'] - oos['market_prob'] * 0.80
oos['rank_final']  = oos.groupby('race_id')['score'].rank(ascending=False, method='first')
oos['gap'] = oos.groupby('race_id')['calib_prob'].transform(
    lambda x: (x.nlargest(2).iloc[0] - x.nlargest(2).iloc[1])
              if x.dropna().shape[0] >= 2 else 0.0
)

# ── 全体ROI確認 ─────────────────────────────────────────────
top1 = oos[oos['rank_final'] == 1]
w    = top1['着順_num'] == 1
total_roi = (pd.to_numeric(top1.loc[w, '単勝オッズ'], errors='coerce').sum() * 100) / (len(top1) * 100) - 1
print(f'\n全体 rank=1 ROI (2023+): {total_roi:+.3f}  ({len(top1)}件  win={w.mean():.3f})')

# ── 買い推奨フィルタ ─────────────────────────────────────────
buy = oos[
    (oos['rank_final'] == 1) &
    (oos['gap'] >= 0.15) &
    (oos['ev_score'] >= 0.0) &
    (oos['クラス_rank'] != 1)   # 新馬除外
].copy()
print(f'買い推奨候補: {len(buy)}件')

# ── NaN率別ROI ───────────────────────────────────────────────
def show_roi(df, label):
    n = len(df)
    if n < 10:
        print(f'  {label}: {n}件(少)')
        return
    hits = (df['着順_num'] == 1).sum()
    ret  = pd.to_numeric(df[df['着順_num'] == 1]['単勝オッズ'], errors='coerce').sum() * 100
    r    = ret / (n * 100) - 1
    mark = '★' if r > 0 else ''
    print(f'  {label:<28}: {n:5d}件  勝率{hits/n:.1%}  ROI {r:+.1%} {mark}')

print()
print('=' * 60)
print('★買い推奨 × NaN率別ROI (OOS 2023+)')
print('=' * 60)
show_roi(buy,                               '全体')
print()
for thr in [0.05, 0.10, 0.15, 0.20, 0.30]:
    show_roi(buy[buy['nan_pct'] < thr],  f'NaN<{thr:.0%}')
print()
for thr in [0.05, 0.10, 0.15, 0.20, 0.30]:
    show_roi(buy[buy['nan_pct'] >= thr], f'NaN≥{thr:.0%}')

print()
print('--- nan_pct分布(buy対象) ---')
print(buy['nan_pct'].describe())

# ── レース内最大NaN率（競合含む）でのROI分析 ─────────────────
print()
print('=' * 60)
print('★買い推奨 × レース内最大NaN率（競合全馬含む）')
print('=' * 60)

# 各レースの全馬NaN率の最大値（favとの競合も含む）
race_max_nan = oos.groupby('race_id')['nan_pct'].max().rename('race_max_nan')
buy2 = buy.join(race_max_nan, on='race_id')

show_roi(buy2,                                      '全体')
print()
for thr in [0.10, 0.20, 0.30, 0.40, 0.50]:
    show_roi(buy2[buy2['race_max_nan'] < thr],  f'レース最大NaN<{thr:.0%}')
print()
for thr in [0.10, 0.20, 0.30, 0.40, 0.50]:
    show_roi(buy2[buy2['race_max_nan'] >= thr], f'レース最大NaN≥{thr:.0%}')

# ── 人気馬（単勝1番人気）のNaN率でのROI分析 ─────────────────
print()
print('=' * 60)
print('★買い推奨 × 買い推奨外の高NaN馬に崩された割合')
print('=' * 60)

# 買い推奨レース一覧（race_id）
buy_races = set(buy['race_id'])

# 買い推奨レース内の全馬（ピック馬以外）
others = oos[(oos['race_id'].isin(buy_races)) & (oos['rank_final'] != 1)].copy()
# 実際の勝者（着順=1）
winners = others[others['着順_num'] == 1][['race_id', 'nan_pct', 'market_prob']].rename(
    columns={'nan_pct': 'winner_nan', 'market_prob': 'winner_mprob'})

buy_with_winner = buy.merge(winners, on='race_id', how='left')
# 負けたレース（ピック馬が1着でない）
lost = buy_with_winner[buy_with_winner['着順_num'] != 1].copy()

print(f'買い推奨負けレース: {len(lost)}件 / 全{len(buy)}件')
print()

for thr in [0.20, 0.30, 0.40, 0.50]:
    beaten_by_high_nan = lost[lost['winner_nan'] >= thr]
    pct = len(beaten_by_high_nan) / len(lost) * 100
    print(f'  負け中、勝者NaN≥{thr:.0%}: {len(beaten_by_high_nan):4d}件 ({pct:.1f}%)')

# 買い推奨ピックが負けた時の勝者NaN分布
print()
print('負けた時の勝者nan_pct分布:')
print(lost['winner_nan'].describe())

# 競合に高NaN馬がいたレース vs いないレースで、崩され率を比較
print()
print('--- 競合高NaN馬の有無 × 敗率 ---')
# 競合（rank!=1）の最大NaN
comp_max = (oos[(oos['race_id'].isin(buy_races)) & (oos['rank_final'] != 1)]
            .groupby('race_id')['nan_pct'].max()
            .rename('comp_max_nan'))
buy_cm = buy.join(comp_max, on='race_id')

for thr in [0.20, 0.30, 0.40, 0.50]:
    has_highnан = buy_cm[buy_cm['comp_max_nan'] >= thr]
    no_high = buy_cm[buy_cm['comp_max_nan'] < thr]
    if len(has_highnан) > 0 and len(no_high) > 0:
        win_h = (has_highnан['着順_num'] == 1).mean()
        win_n = (no_high['着順_num'] == 1).mean()
        roi_h_val = (pd.to_numeric(has_highnан[has_highnан['着順_num']==1]['単勝オッズ'], errors='coerce').sum()*100)/(len(has_highnан)*100)-1
        roi_n_val = (pd.to_numeric(no_high[no_high['着順_num']==1]['単勝オッズ'], errors='coerce').sum()*100)/(len(no_high)*100)-1
        print(f'  競合NaN≥{thr:.0%}あり: {len(has_highnан):4d}件  勝率{win_h:.1%}  ROI{roi_h_val:+.1%}')
        print(f'  競合NaN≥{thr:.0%}なし: {len(no_high):4d}件  勝率{win_n:.1%}  ROI{roi_n_val:+.1%}')
        print()

print()
print('=' * 60)
print('★買い推奨 × 人気馬(1番人気)のNaN率')
print('=' * 60)

# 各レースの1番人気のNaN率
fav_nan = (oos.sort_values('単勝オッズ')
              .groupby('race_id')
              .first()[['nan_pct']]
              .rename(columns={'nan_pct': 'fav_nan'}))
buy3 = buy.join(fav_nan, on='race_id')

show_roi(buy3,                                   '全体')
print()
for thr in [0.10, 0.20, 0.30, 0.40]:
    show_roi(buy3[buy3['fav_nan'] < thr],    f'1番人気NaN<{thr:.0%}')
print()
for thr in [0.10, 0.20, 0.30, 0.40]:
    show_roi(buy3[buy3['fav_nan'] >= thr],   f'1番人気NaN≥{thr:.0%}')
