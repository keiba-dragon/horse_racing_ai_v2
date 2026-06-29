# coding: utf-8
"""芝中距離（芝1401m-2000m）のベースライン確認 - 旧320特徴モデル（芝 artifact）"""
import sys, os, pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from save_conditional_logit import prepare, segment_softmax, BASE_DIR, DATA_FILE
from save_v3 import add_computed_features

MODEL_DIR = os.path.join(BASE_DIR, 'models')

def roi_from_top1(top1):
    won  = top1['着順_num'] == 1
    odds = pd.to_numeric(top1['単勝オッズ'], errors='coerce')
    if len(top1) == 0:
        return float('nan'), 0
    return (odds[won] * 100).sum() / (len(top1) * 100) - 1, len(top1)

def comb2526(r25, n25, r26, n26):
    if n25 + n26 == 0:
        return 0.0
    return (r25 * n25 + r26 * n26) / (n25 + n26)

print("モデル読み込み中...")
with open(os.path.join(MODEL_DIR, 'roi_model.pkl'), 'rb') as f:
    pkg = pickle.load(f)
art           = pkg['artifacts']['芝']
beta          = art['coef']
scaler        = art['scaler']
feats         = art['feat_cols']
top_idx       = art.get('top_idx', None)
top_idx3      = art.get('top_idx3', None)
poly2         = art.get('poly2', None)
inter_scaler2 = art.get('inter_scaler2', None)
poly3         = art.get('poly3', None)
inter_scaler3 = art.get('inter_scaler3', None)
print(f"芝 artifact: {len(feats)}基底特徴量, beta:{beta.shape[0]}次元 (poly交互作用込み)")

print("データ読み込み中...")
df = pd.read_parquet(DATA_FILE)
df['日付_num'] = pd.to_numeric(df['日付'], errors='coerce')
df['着順_num'] = pd.to_numeric(df['着順_num'], errors='coerce')
df = df.dropna(subset=['日付_num', '着順_num'])
df = df[df['着順_num'] < 99]
df['race_id'] = (df['日付_num'].astype(int).astype(str) + '_' +
                 df['開催'].astype(str).str.strip() + '_' +
                 df['Ｒ'].astype(str).str.strip())
df = df[df['開催'].notna()].copy()
df['surface'] = df['距離'].astype(str).str.strip().str.extract(r'^([芝ダ])')[0].fillna('不明')
dm = pd.to_numeric(df['距離'].astype(str).str.extract(r'(\d+)')[0], errors='coerce')

# 芝中距離フィルタ: 1401m-2000m
df_mid = df[(df['surface'] == '芝') & (dm > 1400) & (dm <= 2000)].copy()
df_mid['dist_m'] = dm[df_mid.index]
df_mid = add_computed_features(df_mid)

print(f"\n芝中距離 全体: {len(df_mid):,}行")
print(f"距離構成:")
for d, n in df_mid['dist_m'].value_counts().sort_index().items():
    print(f"  {int(d)}m: {n:,}行")

oos_2324 = df_mid[(df_mid['日付_num'] >= 230101) & (df_mid['日付_num'] < 250101)]
oos_2025 = df_mid[(df_mid['日付_num'] >= 250101) & (df_mid['日付_num'] < 260101)]
oos_2026 = df_mid[df_mid['日付_num'] >= 260101]
print(f"\nOOS: 2324:{oos_2324['race_id'].nunique()}R  "
      f"2025:{oos_2025['race_id'].nunique()}R  "
      f"2026:{oos_2026['race_id'].nunique()}R")

missing = [c for c in feats if c not in df_mid.columns]
print(f"\n欠けている特徴量: {len(missing)}個")
for c in missing:
    df_mid[c] = np.nan

oos = df_mid[df_mid['日付_num'] >= 230101].copy()
X_p, _, gs_p, n_p, *_ = prepare(oos, feats, scaler=scaler,
                                  poly2=poly2, inter_scaler2=inter_scaler2, top_idx=top_idx,
                                  poly3=poly3, inter_scaler3=inter_scaler3, top_idx3=top_idx3)
oos_s = oos.sort_values('race_id').reset_index(drop=True)
oos_s['prob'] = segment_softmax(X_p @ beta, gs_p, n_p)
oos_s['rank'] = oos_s.groupby('race_id')['prob'].rank(ascending=False, method='first')
top1 = oos_s[oos_s['rank'] == 1].copy()

print(f"\n{'='*55}")
print(f"  芝中距離(1401-2000m) ベースライン (旧芝artifact)")
print(f"{'='*55}")
print(f"  {'年度':<6} {'R数':>6} {'ROI':>10}")
print(f"  {'-'*30}")
year_data = {}
for yr in sorted(top1['日付_num'].floordiv(10000).unique()):
    s = top1[top1['日付_num'] // 10000 == yr]
    r, n = roi_from_top1(s)
    year_data[yr] = (r, n)
    print(f"  20{yr}  {n:>6,}  {r:>+9.2%}")
print(f"  {'-'*30}")
r25, n25 = year_data.get(25, (0, 1))
r26, n26 = year_data.get(26, (0, 1))
comb = comb2526(r25, n25, r26, n26)
print(f"  25+26合算: {comb:>+9.2%}  ({n25}+{n26}R)")

print(f"\n距離別 ROI (2324):")
top1_2324 = top1[(top1['日付_num'] >= 230101) & (top1['日付_num'] < 250101)].copy()
for d, grp in top1_2324.groupby('dist_m'):
    r, n = roi_from_top1(grp)
    print(f"  {int(d)}m: {r:+.2%}  ({n}R)")
