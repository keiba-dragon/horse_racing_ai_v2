# coding: utf-8
"""parquetの全テストデータにモデルを適用してS指標ROI分析を行う"""
import sys, io, re, json, pickle, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
import pandas as pd
import numpy as np

BASE      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE, 'models_2025')

def extract_venue(k):
    m = re.search(r'\d+([^\d]+)', str(k))
    return m.group(1) if m else str(k)

def get_distance_band(dist_str):
    m = re.search(r'\d+', str(dist_str))
    if not m: return None
    d = int(m.group())
    if d <= 1400:   return '短距離'
    elif d <= 1800: return 'マイル'
    elif d <= 2200: return '中距離'
    else:           return '長距離'

def get_class_group(r):
    try: r = int(float(r))
    except: return '3勝以上'
    if r == 1: return '新馬'
    elif r == 2: return '未勝利'
    elif r == 3: return '1勝'
    elif r == 4: return '2勝'
    else: return '3勝以上'

# ── 読み込み ──────────────────────────────────────────────
print("読み込み中...")
with open(os.path.join(MODEL_DIR, 'model_info.json'), encoding='utf-8') as f:
    cur_info = json.load(f)
with open(os.path.join(MODEL_DIR, 'submodel', 'submodel_info.json'), encoding='utf-8') as f:
    sub_info = json.load(f)

cur_features = cur_info['features']
sub_features = sub_info['features']
cur_models_meta = cur_info['models']
sub_models_meta = sub_info['models']

df = pd.read_parquet(os.path.join(BASE, 'data', 'processed', 'all_venues_features.parquet'))
dnum_col = '日付_num' if '日付_num' in df.columns else '日付'
df['_dnum'] = pd.to_numeric(df[dnum_col], errors='coerce')
df = df[df['_dnum'] >= 230715].reset_index(drop=True)
print(f"テストデータ: {len(df)}行")

# 特徴量を数値変換（StringDtype対策）
all_feats = list(set(cur_features + sub_features))
for col in all_feats:
    if col in df.columns:
        df[col] = df[col].astype(str).replace('nan','').replace('None','')
        df[col] = pd.to_numeric(df[col], errors='coerce')

# キー生成
df['会場']      = df['開催'].apply(extract_venue)
df['cur_key']   = df['会場'] + '_' + df['距離'].astype(str)
df['_dist_band'] = df['距離'].apply(get_distance_band)
mask = (df['芝・ダ'] == 'ダ') & (df['_dist_band'].isin(['中距離', '長距離']))
df.loc[mask, '_dist_band'] = '中長距離'
df['_cls_group'] = df['クラス_rank'].apply(get_class_group)
df['sub_key']   = df['芝・ダ'].astype(str) + '_' + df['_dist_band'].astype(str) + '_' + df['_cls_group'].astype(str)
df['race_key']  = df['_dnum'].astype(str) + '_' + df['開催'].astype(str) + '_' + df['Ｒ'].astype(str)

for col in ['cur_prob','sub_prob','cur_cs','sub_cs','cur_ri','sub_ri','cur_r','sub_r','_cur_score','_sub_score']:
    df[col] = np.nan

# ── curモデル適用 ─────────────────────────────────────────
print("curモデル適用中...")
cur_feats_avail = [c for c in cur_features if c in df.columns]
ok_cur = 0
for ck in df['cur_key'].dropna().unique():
    wf = os.path.join(MODEL_DIR, f'lgb_{ck}_win.pkl')
    if not os.path.exists(wf): continue
    idx = df[df['cur_key'] == ck].index
    with open(wf, 'rb') as f: wm = pickle.load(f)
    try:
        prob = wm.predict_proba(df.loc[idx, cur_feats_avail].values)[:, 1]
        df.loc[idx, 'cur_prob'] = prob
        st  = cur_models_meta.get(ck, {}).get('stats', {})
        w_m = st.get('win_mean', np.nanmean(prob))
        w_s = st.get('win_std',  np.nanstd(prob))
        df.loc[idx, 'cur_cs'] = 50 + 10 * (prob - w_m) / (w_s if w_s > 0 else 1)
        ok_cur += 1
    except Exception as e:
        pass

# ranker適用（cur）: cur_keyグループ内でスコア計算→race_key内でランク付け
for ck in df['cur_key'].dropna().unique():
    rf = os.path.join(MODEL_DIR, 'ranker', f'ranker_{ck}.pkl')
    if not os.path.exists(rf): continue
    idx = df[df['cur_key'] == ck].index
    if df.loc[idx, 'cur_prob'].isna().all(): continue
    with open(rf, 'rb') as f: rm = pickle.load(f)
    try:
        scores = rm.predict(df.loc[idx, cur_feats_avail].values)
        df.loc[idx, '_cur_score'] = scores
    except: pass

df['cur_r'] = df.groupby('race_key')['_cur_score'].rank(ascending=False, method='min')

# レース内偏差値（cur）
def add_ri(prob_col, ri_col):
    gm = df.groupby('race_key')[prob_col].transform('mean')
    gs = df.groupby('race_key')[prob_col].transform('std')
    df[ri_col] = 50 + 10 * (df[prob_col] - gm) / gs.clip(lower=1e-6)

add_ri('cur_prob', 'cur_ri')
print(f"  cur ok: {ok_cur}キー, prob非NaN: {df['cur_prob'].notna().sum()}, r非NaN: {df['cur_r'].notna().sum()}")

# ── subモデル適用 ─────────────────────────────────────────
print("subモデル適用中...")
sub_feats_avail = [c for c in sub_features if c in df.columns]
ok_sub = 0
for sk in df['sub_key'].dropna().unique():
    wf = os.path.join(MODEL_DIR, 'submodel', f'sub_{sk}_win.pkl')
    if not os.path.exists(wf): continue
    idx = df[df['sub_key'] == sk].index
    with open(wf, 'rb') as f: wm = pickle.load(f)
    try:
        prob = wm.predict_proba(df.loc[idx, sub_feats_avail].values)[:, 1]
        df.loc[idx, 'sub_prob'] = prob
        st  = sub_models_meta.get(sk, {}).get('stats', {})
        w_m = st.get('win_mean', np.nanmean(prob))
        w_s = st.get('win_std',  np.nanstd(prob))
        df.loc[idx, 'sub_cs'] = 50 + 10 * (prob - w_m) / (w_s if w_s > 0 else 1)
        ok_sub += 1
    except: pass

# ranker適用（sub）
for sk in df['sub_key'].dropna().unique():
    rf = os.path.join(MODEL_DIR, 'submodel_ranker', f'class_ranker_{sk}.pkl')
    if not os.path.exists(rf): continue
    idx = df[df['sub_key'] == sk].index
    if df.loc[idx, 'sub_prob'].isna().all(): continue
    with open(rf, 'rb') as f: rm = pickle.load(f)
    try:
        scores = rm.predict(df.loc[idx, sub_feats_avail].values)
        df.loc[idx, '_sub_score'] = scores
    except: pass

df['sub_r'] = df.groupby('race_key')['_sub_score'].rank(ascending=False, method='min')
add_ri('sub_prob', 'sub_ri')
print(f"  sub ok: {ok_sub}キー, prob非NaN: {df['sub_prob'].notna().sum()}, r非NaN: {df['sub_r'].notna().sum()}")

# ── S指標計算 ─────────────────────────────────────────────
prod_r = (df['cur_r'] * df['sub_r']).clip(lower=0.25)
df['S'] = df['sub_cs'] * df['sub_ri'] / prod_r
print(f"\nS指標: {df['S'].notna().sum()}頭 / {df[df['S'].notna()]['race_key'].nunique()}レース\n")

# ── 勝率・複勝率・相関分析（ROIはparquet配当が前走値なので除外） ──────
from scipy.stats import spearmanr
df2 = df.dropna(subset=['S']).copy()
df2['着_num'] = pd.to_numeric(df2['着順_num'], errors='coerce')
df2['odds']   = pd.to_numeric(df2['単勝オッズ'], errors='coerce')
df2 = df2.dropna(subset=['着_num', 'odds'])
n_races = df2['race_key'].nunique()

def show_row(subset, label, n_total):
    n = len(subset)
    if n < 10: return
    wr  = subset['着_num'].eq(1).mean()
    pr  = (subset['着_num'] <= 3).mean()
    ao  = subset['odds'].mean()
    cov = n / n_total
    print(f"  {label:<26}  {n:>6}  {cov:>6.1%}  {wr:>7.1%}  {pr:>7.1%}  {ao:>6.1f}倍")

# 1番人気ベースライン
ikk = df2.sort_values('odds').groupby('race_key').first().reset_index()
wr_ik = ikk['着_num'].eq(1).mean()
pr_ik = (ikk['着_num'] <= 3).mean()
corr_ik, _ = spearmanr(df2['単勝オッズ'], df2['着_num'])
corr_s,  _ = spearmanr(-df2['S'].fillna(-9999), df2['着_num'])
print(f"総データ: {len(df2)}頭 / {n_races}レース")
print(f"※ ROIはparquetに正確な配当がないため省略（2026年分の1026レースで別途確認済み）\n")
print(f"1番人気: 勝率{wr_ik:.1%} / 複勝率{pr_ik:.1%} / 相関{corr_ik:.3f}")
print(f"S全体(相関): {corr_s:.3f}\n")

# 各レースでS最高の1頭
top1 = df2.sort_values('S', ascending=False).drop_duplicates('race_key', keep='first')

print(f"  {'条件':<26}  {'N':>6}  {'カバー':>6}  {'勝率':>7}  {'複勝率':>7}  {'平均OD':>6}")
print("  " + "-" * 70)
for thr in [0, 1000, 2000, 3000, 4000]:
    show_row(top1[top1['S'] > thr], f"S>{thr}", n_races)

print()
for s_thr in [0, 2000]:
    for o_thr in [3.0, 4.0, 5.0, 6.0, 8.0]:
        sub = top1[(top1['S'] > s_thr) & (top1['odds'] > o_thr)]
        if len(sub) < 10: break
        show_row(sub, f"S>{s_thr} & オッズ>{o_thr:.0f}", n_races)
    print()
