# coding: utf-8
"""芝短距離（芝≤1400m）のベースライン確認 - 旧320特徴モデル（芝 artifact）"""
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
art          = pkg['artifacts']['芝']
beta         = art['coef']
scaler       = art['scaler']
feats        = art['feat_cols']
top_idx      = art.get('top_idx', None)
top_idx3     = art.get('top_idx3', None)
poly2        = art.get('poly2', None)
inter_scaler2= art.get('inter_scaler2', None)
poly3        = art.get('poly3', None)
inter_scaler3= art.get('inter_scaler3', None)
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

# 芝短距離フィルタ
df_short = df[(df['surface'] == '芝') & (dm <= 1400)].copy()
df_short['dist_m'] = dm[df_short.index]
df_short = add_computed_features(df_short)

# 新馬の確認
print(f"\n新馬（クラス_rank==1.0）の割合:")
if 'クラス_rank' in df_short.columns:
    df_short['クラス_rank_num'] = pd.to_numeric(df_short['クラス_rank'], errors='coerce')
    n_shinba = (df_short['クラス_rank_num'] == 1.0).sum()
    print(f"  新馬: {n_shinba:,}行 / 全体: {len(df_short):,}行 ({n_shinba/len(df_short)*100:.1f}%)")

print(f"\n芝短距離 全体: {len(df_short):,}行")

# OOS のみ評価
oos = df_short[df_short['日付_num'] >= 230101].copy()
oos_2324 = df_short[(df_short['日付_num'] >= 230101) & (df_short['日付_num'] < 250101)]
oos_2025 = df_short[(df_short['日付_num'] >= 250101) & (df_short['日付_num'] < 260101)]
oos_2026 = df_short[df_short['日付_num'] >= 260101]
print(f"OOS: 2324:{len(oos_2324):,}  2025:{len(oos_2025):,}  2026:{len(oos_2026):,}")

# 欠けている特徴量をNaN列で補完（scalerが320特徴を期待するため）
missing = [c for c in feats if c not in oos.columns]
print(f"欠けている特徴量: {missing}")
for c in missing:
    oos[c] = np.nan
valid_p = feats  # 全320特徴を使用（欠損はNaN→prepare内でゼロ補完）
print(f"使用特徴量: {len(valid_p)}")
X_p, _, gs_p, n_p, *_ = prepare(oos, valid_p, scaler=scaler,
                                  poly2=poly2, inter_scaler2=inter_scaler2, top_idx=top_idx,
                                  poly3=poly3, inter_scaler3=inter_scaler3, top_idx3=top_idx3)
oos_s = oos.sort_values('race_id').reset_index(drop=True)
oos_s['prob'] = segment_softmax(X_p @ beta, gs_p, n_p)
oos_s['rank'] = oos_s.groupby('race_id')['prob'].rank(ascending=False, method='first')
top1 = oos_s[oos_s['rank'] == 1].copy()

print(f"\n{'='*50}")
print(f"  芝短距離 ベースライン (旧芝artifact, 新馬込み)")
print(f"{'='*50}")
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

# 新馬除外版も確認
print(f"\n{'='*50}")
print(f"  芝短距離 ベースライン (新馬除外)")
print(f"{'='*50}")
if 'クラス_rank_num' in df_short.columns:
    df_ex = df_short[df_short['クラス_rank_num'] != 1.0].copy()
    oos_ex = df_ex[df_ex['日付_num'] >= 230101].copy()
    for c in missing:
        oos_ex[c] = np.nan
    X_p2, _, gs_p2, n_p2, *_ = prepare(oos_ex, feats, scaler=scaler,
                                          poly2=poly2, inter_scaler2=inter_scaler2, top_idx=top_idx,
                                          poly3=poly3, inter_scaler3=inter_scaler3, top_idx3=top_idx3)
    oos_ex_s = oos_ex.sort_values('race_id').reset_index(drop=True)
    oos_ex_s['prob'] = segment_softmax(X_p2 @ beta, gs_p2, n_p2)
    oos_ex_s['rank'] = oos_ex_s.groupby('race_id')['prob'].rank(ascending=False, method='first')
    top1_ex = oos_ex_s[oos_ex_s['rank'] == 1].copy()
    print(f"  {'年度':<6} {'R数':>6} {'ROI':>10}")
    print(f"  {'-'*30}")
    year_data2 = {}
    for yr in sorted(top1_ex['日付_num'].floordiv(10000).unique()):
        s = top1_ex[top1_ex['日付_num'] // 10000 == yr]
        r, n = roi_from_top1(s)
        year_data2[yr] = (r, n)
        print(f"  20{yr}  {n:>6,}  {r:>+9.2%}")
    print(f"  {'-'*30}")
    r25b, n25b = year_data2.get(25, (0, 1))
    r26b, n26b = year_data2.get(26, (0, 1))
    comb2 = comb2526(r25b, n25b, r26b, n26b)
    print(f"  25+26合算: {comb2:>+9.2%}  ({n25b}+{n26b}R)")
