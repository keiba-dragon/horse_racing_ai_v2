# coding: utf-8
"""parquet(2023-07+) x 新result CSV で印別ROI分析"""
import sys, io, re, json, pickle, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
import pandas as pd, numpy as np

MODEL_DIR = 'models_2025'

def extract_venue(k):
    m = re.search(r'\d+([^\d]+)', str(k))
    return m.group(1) if m else str(k)

def get_distance_band(d):
    m = re.search(r'\d+', str(d))
    if not m: return None
    d = int(m.group())
    if d <= 1400: return '短距離'
    elif d <= 1800: return 'マイル'
    elif d <= 2200: return '中距離'
    else: return '長距離'

def get_class_group(r):
    try: r = int(float(r))
    except: return '3勝以上'
    if r == 1: return '新馬'
    elif r == 2: return '未勝利'
    elif r == 3: return '1勝'
    elif r == 4: return '2勝'
    else: return '3勝以上'

# モデル情報
with open(os.path.join(MODEL_DIR, 'model_info.json'), encoding='utf-8') as f:
    cur_info = json.load(f)
with open(os.path.join(MODEL_DIR, 'submodel', 'submodel_info.json'), encoding='utf-8') as f:
    sub_info = json.load(f)

cur_features = cur_info['features']
sub_features = sub_info['features']
cur_models_meta = cur_info['models']
sub_models_meta = sub_info['models']

# parquet読み込み（2023-07以降）
print("parquet読み込み中...")
df = pd.read_parquet('data/processed/all_venues_features.parquet')
dnum_col = '日付_num' if '日付_num' in df.columns else '日付'
df['_dnum'] = pd.to_numeric(df[dnum_col], errors='coerce')
df = df[df['_dnum'] >= 260101].reset_index(drop=True)
print(f"テストデータ: {len(df)}行 / {df['_dnum'].nunique()}日")

all_feats = list(set(cur_features + sub_features))
for col in all_feats:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col].astype(str).replace({'nan': '', 'None': ''}), errors='coerce')

df['会場']       = df['開催'].apply(extract_venue)
df['cur_key']    = df['会場'] + '_' + df['距離'].astype(str)
df['_dist_band'] = df['距離'].apply(get_distance_band)
mask = (df['芝・ダ'] == 'ダ') & (df['_dist_band'].isin(['中距離', '長距離']))
df.loc[mask, '_dist_band'] = '中長距離'
df['_cls_group'] = df['クラス_rank'].apply(get_class_group)
df['sub_key']    = df['芝・ダ'].astype(str) + '_' + df['_dist_band'].astype(str) + '_' + df['_cls_group'].astype(str)
df['race_key']   = df['_dnum'].astype(str) + '_' + df['開催'].astype(str) + '_' + df['Ｒ'].astype(str)

for col in ['cur_prob', 'sub_prob', 'cur_cs', 'sub_cs', 'cur_ri', 'sub_ri', 'cur_r', 'sub_r', '_cur_sc', '_sub_sc']:
    df[col] = np.nan

# curモデル
print("curモデル適用中...")
cur_feats_avail = [c for c in cur_features if c in df.columns]
for ck in df['cur_key'].dropna().unique():
    wf = os.path.join(MODEL_DIR, f'lgb_{ck}_win.pkl')
    if not os.path.exists(wf): continue
    idx = df[df['cur_key'] == ck].index
    with open(wf, 'rb') as f: wm = pickle.load(f)
    try:
        prob = wm.predict_proba(df.loc[idx, cur_feats_avail].values)[:, 1]
        df.loc[idx, 'cur_prob'] = prob
        st = cur_models_meta.get(ck, {}).get('stats', {})
        w_m = st.get('win_mean', np.nanmean(prob))
        w_s = st.get('win_std',  np.nanstd(prob))
        df.loc[idx, 'cur_cs'] = 50 + 10 * (prob - w_m) / (w_s if w_s > 0 else 1)
    except: pass

for ck in df['cur_key'].dropna().unique():
    rf = os.path.join(MODEL_DIR, 'ranker', f'ranker_{ck}.pkl')
    if not os.path.exists(rf): continue
    idx = df[df['cur_key'] == ck].index
    if df.loc[idx, 'cur_prob'].isna().all(): continue
    with open(rf, 'rb') as f: rm = pickle.load(f)
    try:
        df.loc[idx, '_cur_sc'] = rm.predict(df.loc[idx, cur_feats_avail].values)
    except: pass

df['cur_r'] = df.groupby('race_key')['_cur_sc'].rank(ascending=False, method='min')
gm = df.groupby('race_key')['cur_prob'].transform('mean')
gs_s = df.groupby('race_key')['cur_prob'].transform('std')
df['cur_ri'] = 50 + 10 * (df['cur_prob'] - gm) / gs_s.clip(lower=1e-6)
print(f"  cur: prob非NaN {df['cur_prob'].notna().sum()}")

# subモデル
print("subモデル適用中...")
sub_feats_avail = [c for c in sub_features if c in df.columns]
for sk in df['sub_key'].dropna().unique():
    wf = os.path.join(MODEL_DIR, 'submodel', f'sub_{sk}_win.pkl')
    if not os.path.exists(wf): continue
    idx = df[df['sub_key'] == sk].index
    with open(wf, 'rb') as f: wm = pickle.load(f)
    try:
        prob = wm.predict_proba(df.loc[idx, sub_feats_avail].values)[:, 1]
        df.loc[idx, 'sub_prob'] = prob
        st = sub_models_meta.get(sk, {}).get('stats', {})
        w_m = st.get('win_mean', np.nanmean(prob))
        w_s = st.get('win_std',  np.nanstd(prob))
        df.loc[idx, 'sub_cs'] = 50 + 10 * (prob - w_m) / (w_s if w_s > 0 else 1)
    except: pass

for sk in df['sub_key'].dropna().unique():
    rf = os.path.join(MODEL_DIR, 'submodel_ranker', f'class_ranker_{sk}.pkl')
    if not os.path.exists(rf): continue
    idx = df[df['sub_key'] == sk].index
    if df.loc[idx, 'sub_prob'].isna().all(): continue
    with open(rf, 'rb') as f: rm = pickle.load(f)
    try:
        df.loc[idx, '_sub_sc'] = rm.predict(df.loc[idx, sub_feats_avail].values)
    except: pass

df['sub_r'] = df.groupby('race_key')['_sub_sc'].rank(ascending=False, method='min')
gm = df.groupby('race_key')['sub_prob'].transform('mean')
gs_s = df.groupby('race_key')['sub_prob'].transform('std')
df['sub_ri'] = 50 + 10 * (df['sub_prob'] - gm) / gs_s.clip(lower=1e-6)
print(f"  sub: prob非NaN {df['sub_prob'].notna().sum()}")

# D値
prod_r = df['sub_r'].clip(lower=0.25)
df['D'] = df['sub_cs'] * df['sub_ri'] / prod_r
print(f"D非NaN: {df['D'].notna().sum()}頭 / {df[df['D'].notna()]['race_key'].nunique()}レース")

df['D_rank'] = df.groupby('race_key')['D'].rank(ascending=False, method='min')
df['D_mean'] = df.groupby('race_key')['D'].transform('mean').clip(lower=1)
df['D_pct']  = (df['D'] - df['D_mean']) / df['D_mean'] * 100
df['_n_qual']= df.groupby('race_key')['D_pct'].transform(lambda x: (x > 200).sum())

def calc_gap(g):
    s = g.sort_values('D', ascending=False)['D'].values
    return s[0] / s[1] if len(s) >= 2 and s[1] > 0 else np.nan

gap_s = df.groupby('race_key').apply(calc_gap)
df = df.join(gap_s.rename('gap_ratio'), on='race_key')

top1 = df[df['D_rank'] == 1].copy()

# result CSV結合
print("\nresult CSV読み込み中...")
with open('data/raw/2023年～の結果.csv', 'rb') as f:
    raw = f.read()
res = pd.read_csv(pd.io.common.BytesIO(raw), encoding='cp932')
res.columns = res.columns.str.strip()

def zen(s):
    if pd.isna(s): return np.nan
    s = str(s).strip().translate(str.maketrans('０１２３４５６７８９', '0123456789'))
    m = re.search(r'\d+', s)
    return int(m.group()) if m else np.nan

res['着_num'] = res['着順'].apply(zen)
res['_dnum']  = pd.to_numeric(res['日付'], errors='coerce').astype('Int64')
res['_venue'] = res['開催'].apply(extract_venue)
res['_R']     = pd.to_numeric(res['Ｒ'], errors='coerce')
res['_tan']   = pd.to_numeric(res['単勝配当'], errors='coerce')
res['_fuku']  = pd.to_numeric(res['複勝配当'], errors='coerce')

# 単勝配当をレース単位で付与（1着馬の配当）
tan_race = (res[res['着_num'] == 1]
            .groupby(['_dnum', '_venue', '_R'])['_tan']
            .first().reset_index()
            .rename(columns={'_tan': '_tan_race'}))
res = res.merge(tan_race, on=['_dnum', '_venue', '_R'], how='left')
res['_fuku_use'] = res['_fuku'].where(res['着_num'] <= 3)

print(f"result CSV: {len(res)}行, {res['_dnum'].nunique()}日")

# join
top1['_dnum_k']  = top1['_dnum'].astype(int)
top1['_venue_k'] = top1['会場'].astype(str)
top1['_R_k']     = pd.to_numeric(top1['Ｒ'], errors='coerce')
top1['単勝オッズ'] = pd.to_numeric(top1['単勝オッズ'], errors='coerce')

res_sub = res[['_dnum', '_venue', '_R', '馬名S', '着_num', '_tan_race', '_fuku_use']].copy()
res_sub['_dnum'] = res_sub['_dnum'].astype(int)

merged = top1.merge(res_sub,
    left_on=['_dnum_k', '_venue_k', '_R_k', '馬名S'],
    right_on=['_dnum', '_venue', '_R', '馬名S'], how='inner')

print(f"マッチ: {len(merged)}頭 / {merged['race_key'].nunique()}レース")

# 印ロジック
def tan_level(row):
    od, gap, dpct, nq = row['単勝オッズ'], row['gap_ratio'], row['D_pct'], row['_n_qual']
    is_ippon = (dpct > 200) and (nq == 1)
    if pd.notna(od) and od > 8 and is_ippon:                 return 3
    if pd.notna(od) and od > 6 and pd.notna(gap) and gap >= 3: return 2
    if pd.notna(od) and od > 6:                               return 1
    return 0

def fuku_level(row):
    od, gap = row['単勝オッズ'], row['gap_ratio']
    if pd.notna(od) and od > 6 and pd.notna(gap) and gap >= 3: return 3
    if pd.notna(od) and od > 5 and pd.notna(gap) and gap >= 3: return 2
    if pd.notna(od) and od > 5:                               return 1
    return 0

merged['tan_lv']  = merged.apply(tan_level, axis=1)
merged['fuku_lv'] = merged.apply(fuku_level, axis=1)

def show(sub, label):
    n = len(sub)
    if n < 5:
        print(f"  {label:<10} N={n}")
        return
    wr = sub['着_num'].eq(1).mean()
    pr = (sub['着_num'] <= 3).mean()
    ao = sub['単勝オッズ'].mean()
    sub_t = sub.dropna(subset=['_tan_race'])
    roi_t = (sub_t[sub_t['着_num'] == 1]['_tan_race'].sum() / 100 - len(sub_t)) / len(sub_t) if len(sub_t) > 0 else np.nan
    placed = sub[sub['着_num'] <= 3].dropna(subset=['_fuku_use'])
    roi_f  = (placed['_fuku_use'].sum() / 100 - n) / n if n > 0 else np.nan
    if pd.notna(ao) and ao > 1 and wr > 0:
        kelly = (wr * ao - 1) / (ao - 1)
    else:
        kelly = np.nan
    rt = f"{roi_t:>+6.1%}" if pd.notna(roi_t) else "     -"
    rf = f"{roi_f:>+6.1%}" if pd.notna(roi_f) else "     -"
    kt = f"{kelly:>+5.1%}" if pd.notna(kelly) else "    -"
    print(f"  {label:<10}  {n:>5}  {wr:>6.1%}  {pr:>6.1%}  {ao:>6.1f}倍  {rt}  {rf}  {kt}")

print(f"\n  {'印':<10}  {'N':>5}  {'勝率':>6}  {'複勝率':>6}  {'平均OD':>6}  {'単ROI':>6}  {'複ROI':>6}  {'Kelly':>5}")
print("  " + "-" * 74)
print("── 単勝印 ──")
for lv, lb in [(3, '◎単'), (2, '○単'), (1, '▲単')]:
    show(merged[merged['tan_lv'] == lv], lb)
show(merged[merged['tan_lv'] > 0], "単印合計")
print("\n── 複勝印 ──")
for lv, lb in [(3, '◎複'), (2, '○複'), (1, '▲複')]:
    show(merged[merged['fuku_lv'] == lv], lb)
show(merged[merged['fuku_lv'] > 0], "複印合計")
