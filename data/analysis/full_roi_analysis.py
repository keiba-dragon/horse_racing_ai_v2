# coding: utf-8
"""parquet × 配当マスターCSV で S指標ROI分析（2023-07〜2026-03）"""
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

# ── モデル読み込み ──
print("モデル読み込み中...")
with open(os.path.join(MODEL_DIR, 'model_info.json'), encoding='utf-8') as f:
    cur_info = json.load(f)
with open(os.path.join(MODEL_DIR, 'submodel', 'submodel_info.json'), encoding='utf-8') as f:
    sub_info = json.load(f)

cur_features     = cur_info['features']
sub_features     = sub_info['features']
cur_models_meta  = cur_info['models']
sub_models_meta  = sub_info['models']

# ── parquet 読み込み ──
print("parquet 読み込み中...")
df = pd.read_parquet(os.path.join(BASE, 'data', 'processed', 'all_venues_features.parquet'))
dnum_col = '日付_num' if '日付_num' in df.columns else '日付'
df['_dnum'] = pd.to_numeric(df[dnum_col], errors='coerce')
df = df[df['_dnum'] >= 230715].reset_index(drop=True)
print(f"テストデータ: {len(df)}行")

# 特徴量数値変換
all_feats = list(set(cur_features + sub_features))
for col in all_feats:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col].astype(str).replace({'nan':'','None':''}), errors='coerce')

# キー生成
df['会場']       = df['開催'].apply(extract_venue)
df['cur_key']    = df['会場'] + '_' + df['距離'].astype(str)
df['_dist_band'] = df['距離'].apply(get_distance_band)
mask = (df['芝・ダ'] == 'ダ') & (df['_dist_band'].isin(['中距離','長距離']))
df.loc[mask, '_dist_band'] = '中長距離'
df['_cls_group'] = df['クラス_rank'].apply(get_class_group)
df['sub_key']    = df['芝・ダ'].astype(str) + '_' + df['_dist_band'].astype(str) + '_' + df['_cls_group'].astype(str)
df['race_key']   = df['_dnum'].astype(str) + '_' + df['開催'].astype(str) + '_' + df['Ｒ'].astype(str)

for col in ['cur_prob','sub_prob','cur_cs','sub_cs','cur_ri','sub_ri',
            'cur_r','sub_r','_cur_score','_sub_score']:
    df[col] = np.nan

# ── curモデル適用 ──
print("curモデル適用中...")
cur_feats_avail = [c for c in cur_features if c in df.columns]
for ck in df['cur_key'].dropna().unique():
    wf = os.path.join(MODEL_DIR, f'lgb_{ck}_win.pkl')
    if not os.path.exists(wf): continue
    idx = df[df['cur_key'] == ck].index
    with open(wf,'rb') as f: wm = pickle.load(f)
    try:
        prob = wm.predict_proba(df.loc[idx, cur_feats_avail].values)[:,1]
        df.loc[idx,'cur_prob'] = prob
        st  = cur_models_meta.get(ck,{}).get('stats',{})
        w_m = st.get('win_mean', np.nanmean(prob))
        w_s = st.get('win_std',  np.nanstd(prob))
        df.loc[idx,'cur_cs'] = 50 + 10*(prob-w_m)/(w_s if w_s>0 else 1)
    except: pass

for ck in df['cur_key'].dropna().unique():
    rf = os.path.join(MODEL_DIR,'ranker',f'ranker_{ck}.pkl')
    if not os.path.exists(rf): continue
    idx = df[df['cur_key']==ck].index
    if df.loc[idx,'cur_prob'].isna().all(): continue
    with open(rf,'rb') as f: rm = pickle.load(f)
    try:
        df.loc[idx,'_cur_score'] = rm.predict(df.loc[idx,cur_feats_avail].values)
    except: pass

df['cur_r'] = df.groupby('race_key')['_cur_score'].rank(ascending=False,method='min')

def add_ri(prob_col, ri_col):
    gm = df.groupby('race_key')[prob_col].transform('mean')
    gs = df.groupby('race_key')[prob_col].transform('std')
    df[ri_col] = 50 + 10*(df[prob_col]-gm)/gs.clip(lower=1e-6)

add_ri('cur_prob','cur_ri')

# ── subモデル適用 ──
print("subモデル適用中...")
sub_feats_avail = [c for c in sub_features if c in df.columns]
for sk in df['sub_key'].dropna().unique():
    wf = os.path.join(MODEL_DIR,'submodel',f'sub_{sk}_win.pkl')
    if not os.path.exists(wf): continue
    idx = df[df['sub_key']==sk].index
    with open(wf,'rb') as f: wm = pickle.load(f)
    try:
        prob = wm.predict_proba(df.loc[idx,sub_feats_avail].values)[:,1]
        df.loc[idx,'sub_prob'] = prob
        st  = sub_models_meta.get(sk,{}).get('stats',{})
        w_m = st.get('win_mean', np.nanmean(prob))
        w_s = st.get('win_std',  np.nanstd(prob))
        df.loc[idx,'sub_cs'] = 50 + 10*(prob-w_m)/(w_s if w_s>0 else 1)
    except: pass

for sk in df['sub_key'].dropna().unique():
    rf = os.path.join(MODEL_DIR,'submodel_ranker',f'class_ranker_{sk}.pkl')
    if not os.path.exists(rf): continue
    idx = df[df['sub_key']==sk].index
    if df.loc[idx,'sub_prob'].isna().all(): continue
    with open(rf,'rb') as f: rm = pickle.load(f)
    try:
        df.loc[idx,'_sub_score'] = rm.predict(df.loc[idx,sub_feats_avail].values)
    except: pass

df['sub_r'] = df.groupby('race_key')['_sub_score'].rank(ascending=False,method='min')
add_ri('sub_prob','sub_ri')

# ── S指標計算 ──
prod_r = (df['cur_r'] * df['sub_r']).clip(lower=0.25)
df['S'] = df['sub_cs'] * df['sub_ri'] / prod_r
print(f"S指標: {df['S'].notna().sum()}頭 / {df[df['S'].notna()]['race_key'].nunique()}レース")

# ── 配当マスターCSV 読み込み ──
print("\n配当マスター読み込み中...")
pay_path = os.path.join(BASE,'data','raw','master','20260315_2013まで_配当.csv')
try:    pay = pd.read_csv(pay_path, encoding='cp932', low_memory=False)
except: pay = pd.read_csv(pay_path, encoding='utf-8', low_memory=False)

def zen_to_num(s):
    if pd.isna(s): return np.nan
    s = str(s).strip().translate(str.maketrans('０１２３４５６７８９', '0123456789'))
    m = re.search(r'\d+', s)
    return int(m.group()) if m else np.nan

pay['_dnum']  = pd.to_numeric(pay['日付'], errors='coerce')
pay['_tan']   = pd.to_numeric(pay['単勝配当'], errors='coerce')
pay['_fuku']  = pd.to_numeric(pay['複勝配当'], errors='coerce')
pay['_atch']  = pay['着順'].apply(zen_to_num)
pay['_ninki'] = pay['人気'].apply(zen_to_num) if '人気' in pay.columns else np.nan
pay = pay[pay['_dnum'] >= 230715].copy()
pay['_R']     = pd.to_numeric(pay['Ｒ'], errors='coerce')
print(f"配当データ（2023-07〜）: {len(pay)}行 / {pay['_dnum'].nunique()}日")

# レース単位で単勝配当を取得（1着馬から）
win_rows = pay[pay['_atch'] == 1].drop_duplicates(['_dnum','開催','_R'])
win_rows = win_rows.copy()
win_rows['_race_key'] = (win_rows['_dnum'].astype(str) + '_' +
                         win_rows['開催'].astype(str) + '_' +
                         win_rows['_R'].astype(str))
tan_map = win_rows.set_index('_race_key')['_tan'].to_dict()

# 各馬の複勝配当・着順マップ（馬名S+race_key → 値）
pay['_race_key'] = (pay['_dnum'].astype(str) + '_' +
                    pay['開催'].astype(str) + '_' +
                    pay['_R'].astype(str))
pay['_join_key'] = pay['_race_key'] + '_' + pay['馬名S'].astype(str)
fuku_map  = pay.set_index('_join_key')['_fuku'].to_dict()
atch_map  = pay.set_index('_join_key')['_atch'].to_dict()
ninki_map = pay.set_index('_join_key')['_ninki'].to_dict()

# ── parquet側のrace_keyを配当側と同形式に統一 ──
df['_pay_rk']  = (df['_dnum'].astype(str) + '_' +
                  df['開催'].astype(str) + '_' +
                  pd.to_numeric(df['Ｒ'], errors='coerce').astype(str))
df['_join_key'] = df['_pay_rk'] + '_' + df['馬名S'].astype(str)

# 着順・配当をマッチング
df['_atch']  = df['_join_key'].map(atch_map)
df['_fuku']  = df['_join_key'].map(fuku_map)
df['_tan']   = df['_pay_rk'].map(tan_map)
df['_ninki'] = df['_join_key'].map(ninki_map)

matched = df['_atch'].notna().sum()
print(f"着順マッチング: {matched}行 ({matched/len(df):.1%})")

# ── 分析用データ作成 ──
df2 = df[df['S'].notna() & df['_atch'].notna()].copy()
df2['odds']     = pd.to_numeric(df2['単勝オッズ'], errors='coerce')
df2['S_mean']   = df2.groupby('_pay_rk')['S'].transform('mean').clip(lower=1)
df2['S_pct']    = (df2['S'] - df2['S_mean']) / df2['S_mean'] * 100
df2['S_rank']   = df2.groupby('_pay_rk')['S'].rank(ascending=False, method='min')
df2['_n_qual']  = df2.groupby('_pay_rk')['S_pct'].transform(lambda x: (x > 200).sum())

# S1位・S2位でgap計算
def get_gap(g):
    g2 = g.sort_values('S', ascending=False).reset_index(drop=True)
    s1 = g2.iloc[0]['S'] if len(g2) >= 1 else np.nan
    s2 = g2.iloc[1]['S'] if len(g2) >= 2 else np.nan
    return pd.Series({
        'gap_ratio': s1/s2 if pd.notna(s2) and s2>0 else np.nan,
        'gap_pct':   (s1-s2)/s2*100 if pd.notna(s2) and s2>0 else np.nan,
    })

print("gap計算中（時間がかかります）...")
gap_df = df2.groupby('_pay_rk').apply(get_gap)
top1 = df2[df2['S_rank'] == 1].copy()
top1 = top1.merge(gap_df, left_on='_pay_rk', right_index=True, how='left')

n_races = df2['_pay_rk'].nunique()
print(f"\n分析対象: {len(df2)}頭 / {n_races}レース / S1位馬: {len(top1)}頭\n")

# 1番人気ベースライン
fav = df2.sort_values('_ninki').groupby('_pay_rk').first()
wr_fav = fav['_atch'].eq(1).mean()
pr_fav = (fav['_atch'] <= 3).mean()
sub_t  = fav.dropna(subset=['_tan'])
roi_t_fav = (sub_t[sub_t['_atch']==1]['_tan'].sum()/100 - len(sub_t)) / len(sub_t)
placed_fav = fav[fav['_atch'] <= 3].dropna(subset=['_fuku'])
roi_f_fav  = (placed_fav['_fuku'].sum()/100 - len(fav)) / len(fav)
print(f"1番人気: 勝率{wr_fav:.1%} / 複勝率{pr_fav:.1%} / 単勝ROI{roi_t_fav:+.1%} / 複勝ROI{roi_f_fav:+.1%}\n")

def show(sub, label):
    n = len(sub)
    if n < 10: return
    wr = sub['_atch'].eq(1).mean()
    pr = (sub['_atch'] <= 3).mean()
    sub_t2 = sub.dropna(subset=['_tan'])
    roi_t = (sub_t2[sub_t2['_atch']==1]['_tan'].sum()/100 - len(sub_t2)) / len(sub_t2) if len(sub_t2) > 0 else np.nan
    placed = sub[sub['_atch'] <= 3].dropna(subset=['_fuku'])
    roi_f  = (placed['_fuku'].sum()/100 - n) / n if n > 0 else np.nan
    ao = sub['odds'].mean() if 'odds' in sub.columns else np.nan
    rt = f"{roi_t:>+7.1%}" if pd.notna(roi_t) else "      -"
    rf = f"{roi_f:>+7.1%}" if pd.notna(roi_f) else "      -"
    aod = f"{ao:>6.1f}倍" if pd.notna(ao) else "     -"
    print(f"  {label:<30}  {n:>6}  {wr:>7.1%}  {pr:>7.1%}  {rt}  {rf}  {aod}")

hdr = f"  {'条件':<30}  {'N':>6}  {'勝率':>7}  {'複勝率':>7}  {'単勝ROI':>7}  {'複勝ROI':>7}  {'平均OD':>6}"
sep = "  " + "-" * 84

# ── gap_pct 閾値別 ──
print("── gap_pct（S1-S2の%差）閾値別 ──"); print(hdr); print(sep)
for thr in [0, 50, 100, 200, 300, 500, 1000]:
    show(top1[top1['gap_pct'] > thr], f"gap_pct>{thr}%")

print()
print("── gap_pct × OD>6 ──"); print(hdr); print(sep)
for thr in [0, 50, 100, 200, 300, 500]:
    show(top1[(top1['gap_pct'] > thr) & (top1['odds'] > 6)], f"gap>{thr}% & OD>6")

# ── gap_ratio 閾値別 ──
print()
print("── gap_ratio（S1/S2倍率）閾値別 ──"); print(hdr); print(sep)
for thr in [1.5, 2, 3, 5, 8, 10]:
    show(top1[top1['gap_ratio'] > thr], f"ratio>{thr}x")

print()
print("── gap_ratio × OD>6 ──"); print(hdr); print(sep)
for thr in [1.5, 2, 3, 5, 8]:
    show(top1[(top1['gap_ratio'] > thr) & (top1['odds'] > 6)], f"ratio>{thr}x & OD>6")

# ── S_pct>200% × 1頭/複数 ──
print()
print("── S_pct>200% × 1頭抜け vs 複数抜け ──"); print(hdr); print(sep)
one_q   = df2[(df2['S_pct'] > 200) & (df2['_n_qual'] == 1)]
multi_q = df2[(df2['S_pct'] > 200) & (df2['_n_qual'] >= 2)]
multi_t1= df2[(df2['S_pct'] > 200) & (df2['_n_qual'] >= 2) & (df2['S_rank'] == 1)]
show(one_q,                           "S_pct>200% 1頭のみ")
show(multi_q,                         "S_pct>200% 2頭以上（全員）")
show(multi_t1,                        "S_pct>200% 2頭以上のS1位")

print()
print("── S_pct>200% × OD>6 ──"); print(hdr); print(sep)
show(one_q[one_q['odds'] > 6],         "1頭抜け & OD>6")
show(multi_q[multi_q['odds'] > 6],     "2頭以上抜け & OD>6（全員）")
show(multi_t1[multi_t1['odds'] > 6],   "2頭以上のS1位 & OD>6")

# ── S基本 × OD ──
print()
print("── S1位 × OD閾値 ──"); print(hdr); print(sep)
show(top1,                             "S1位（全体）")
for od in [3, 4, 5, 6, 8, 10]:
    show(top1[top1['odds'] > od],      f"S1位 & OD>{od}")
