# coding: utf-8
"""強さPT × ランカー順位 相関分析"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import pandas as pd, numpy as np, os, pickle, json, re, glob

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_dir = os.path.join(base_dir, 'models_2025')

def get_distance_band(dist):
    m = re.search(r'\d+', str(dist))
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
    return '3勝以上'

def _zen(s):
    if pd.isna(s): return np.nan
    s = str(s).strip().translate(str.maketrans('０１２３４５６７８９', '0123456789'))
    m = re.search(r'\d+', s)
    return int(m.group()) if m else np.nan

with open(os.path.join(model_dir, 'model_info.json'), encoding='utf-8') as f:
    cur_info = json.load(f)
cur_features = cur_info['features']; cur_models = cur_info['models']

with open(os.path.join(model_dir, 'submodel', 'submodel_info.json'), encoding='utf-8') as f:
    sub_info = json.load(f)
sub_features = sub_info['features']; sub_models = sub_info['models']

# ranker_info読み込み（別ファイル）
cur_rankers = {}; sub_rankers = {}
cur_ranker_features = []; sub_ranker_features = []
rinfo_path     = os.path.join(model_dir, 'ranker', 'ranker_info.json')
rinfo_sub_path = os.path.join(model_dir, 'submodel_ranker', 'class_ranker_info.json')
if os.path.exists(rinfo_path):
    with open(rinfo_path, encoding='utf-8') as f:
        rd = json.load(f)
        cur_rankers = rd.get('rankers', {})
        cur_ranker_features = rd.get('features', [])
if os.path.exists(rinfo_sub_path):
    with open(rinfo_sub_path, encoding='utf-8') as f:
        rd = json.load(f)
        sub_rankers = rd.get('rankers', {})
        sub_ranker_features = rd.get('features', [])
print(f"cur_rankers: {len(cur_rankers)}件, sub_rankers: {len(sub_rankers)}件")

import pyarrow.parquet as pq_mod
pq_path = os.path.join(base_dir, 'data', 'processed', 'all_venues_features.parquet')
all_feats = list(set(cur_features + sub_features))
avail = set(pq_mod.read_schema(pq_path).names)
load_cols = ['馬名S', '日付', '距離'] + [c for c in avail if c in set(all_feats)]
df_pq = pd.read_parquet(pq_path, columns=load_cols)
df_pq['日付_n'] = pd.to_numeric(df_pq['日付'], errors='coerce')
pq_latest = df_pq.sort_values('日付_n').groupby('馬名S', sort=False).last().reset_index()

cur_model_cache = {}; sub_model_cache = {}
cur_ranker_cache = {}; sub_ranker_cache = {}

def get_cur_model(ck):
    if ck not in cur_model_cache and ck in cur_models:
        p = os.path.join(model_dir, cur_models[ck]['win'])
        if os.path.exists(p):
            with open(p, 'rb') as f: m = pickle.load(f)
            cur_model_cache[ck] = (m, m.booster_.feature_name())
    return cur_model_cache.get(ck)

def get_sub_model(sk):
    if sk not in sub_model_cache and sk in sub_models:
        p = os.path.join(model_dir, 'submodel', sub_models[sk]['win'])
        if os.path.exists(p):
            with open(p, 'rb') as f: m = pickle.load(f)
            sub_model_cache[sk] = (m, m.booster_.feature_name())
    return sub_model_cache.get(sk)

def get_cur_ranker(ck):
    if ck not in cur_ranker_cache and ck in cur_rankers:
        p = os.path.join(model_dir, 'ranker', cur_rankers[ck])
        if os.path.exists(p):
            with open(p, 'rb') as f: cur_ranker_cache[ck] = pickle.load(f)
    return cur_ranker_cache.get(ck)

def get_sub_ranker(sk):
    if sk not in sub_ranker_cache and sk in sub_rankers:
        p = os.path.join(model_dir, 'submodel_ranker', sub_rankers[sk])
        if os.path.exists(p):
            with open(p, 'rb') as f: sub_ranker_cache[sk] = pickle.load(f)
    return sub_ranker_cache.get(sk)

def apply_ranker(ranker, sub, features):
    wf = features if features else (ranker.feature_name() if hasattr(ranker, 'feature_name') else [])
    for c in wf:
        if c not in sub.columns: sub[c] = np.nan
    scores = ranker.predict(sub[wf])
    return pd.Series(scores).rank(ascending=False).values

VENUE_MAP = {'中山':'中','東京':'東','阪神':'阪','中京':'名','京都':'京',
             '函館':'函','新潟':'新','小倉':'小','札幌':'札','福島':'福'}

result_csvs = sorted(glob.glob(os.path.join(base_dir, 'data', 'raw', 'results', '出馬表形式*結果*.csv')))
records = []

for rf in result_csvs:
    try: dfr = pd.read_csv(rf, encoding='cp932', low_memory=False)
    except: dfr = pd.read_csv(rf, encoding='utf-8', low_memory=False)
    if '日付S' not in dfr.columns: continue
    dfr['着_num']   = dfr['着'].apply(_zen)
    dfr['会場']     = dfr['場所'].astype(str).map(VENUE_MAP).fillna(dfr['場所'].astype(str))
    dfr['_surface'] = dfr['芝ダ'].astype(str).str.strip() if '芝ダ' in dfr.columns else 'ダ'
    dfr['cur_key']  = dfr['会場'] + '_' + dfr['_surface'] + dfr['距離'].astype(str)
    dfr['_dist_band'] = dfr['距離'].apply(get_distance_band)
    _mask = (dfr['_surface'] == 'ダ') & (dfr['_dist_band'].isin(['中距離', '長距離']))
    dfr.loc[_mask, '_dist_band'] = '中長距離'
    if 'クラス' in dfr.columns:
        _cls_map = {'新馬': 1, '未勝利': 2, '1勝': 3, '2勝': 4}
        dfr['クラス_rank'] = dfr['クラス'].apply(lambda v: next((r for k, r in _cls_map.items() if k in str(v)), 5))
    dfr['_cls_group'] = dfr['クラス_rank'].apply(get_class_group) if 'クラス_rank' in dfr.columns else '3勝以上'
    dfr['sub_key']  = dfr['_surface'] + '_' + dfr['_dist_band'] + '_' + dfr['_cls_group']
    dfr['性別_num'] = dfr['性別'].map({'牡': 0, '牝': 1, 'セ': 2}).astype(float)
    feat_cols = ['馬名S'] + [c for c in all_feats if c in pq_latest.columns]
    merged = dfr.merge(pq_latest[feat_cols], on='馬名S', how='left', suffixes=('', '_p'))
    for c in all_feats:
        if c in merged.columns: merged[c] = pd.to_numeric(merged[c], errors='coerce')

    for gk, idx in merged.groupby(['場所', 'Ｒ'], sort=False).groups.items():
        sub = merged.loc[idx].copy()
        ck = sub['cur_key'].iloc[0]; sk = sub['sub_key'].iloc[0]
        n = len(sub)

        cur_cs = np.full(n, np.nan); sub_cs = np.full(n, np.nan)
        cur_rk = np.full(n, np.nan); sub_rk = np.full(n, np.nan)

        cm = get_cur_model(ck)
        if cm:
            m, wf = cm
            for c in wf:
                if c not in sub.columns: sub[c] = np.nan
            prob = m.predict_proba(sub[wf])[:, 1]
            st = cur_models[ck].get('stats', {})
            wm = st.get('win_mean', prob.mean()); ws = st.get('win_std', prob.std())
            cur_cs = 50 + 10 * (prob - wm) / (ws if ws > 0 else 1)
            cr = get_cur_ranker(ck)
            if cr:
                cur_rk = apply_ranker(cr, sub.copy(), cur_ranker_features)

        sm = get_sub_model(sk)
        if sm:
            m, wf = sm
            for c in wf:
                if c not in sub.columns: sub[c] = np.nan
            prob = m.predict_proba(sub[wf])[:, 1]
            st = sub_models[sk].get('stats', {})
            wm = st.get('win_mean', prob.mean()); ws = st.get('win_std', prob.std())
            sub_cs = 50 + 10 * (prob - wm) / (ws if ws > 0 else 1)
            sr = get_sub_ranker(sk)
            if sr:
                sub_rk = apply_ranker(sr, sub.copy(), sub_ranker_features)

        cur_s = pd.Series(cur_cs, index=sub.index)
        sub_s = pd.Series(sub_cs, index=sub.index)
        both = ~(cur_s.isna() | sub_s.isna())
        pt = pd.Series(np.nan, index=sub.index)
        pt[both] = (cur_s[both] + sub_s[both]) / 2
        pt[~both] = cur_s.fillna(sub_s)[~both]

        for i, (row_idx, row) in enumerate(sub.iterrows()):
            records.append({
                '着_num':  row['着_num'],
                '強さPT':  pt.iloc[i],
                'cur_rk':  cur_rk[i],
                'sub_rk':  sub_rk[i],
                'n_horses': n,
            })

df = pd.DataFrame(records).dropna(subset=['着_num', '強さPT'])
df['has_ranker'] = ~(df['cur_rk'].isna() & df['sub_rk'].isna())
print(f'総頭数: {len(df)}  ランカーあり: {df["has_ranker"].sum()}')

dr = df.dropna(subset=['cur_rk', 'sub_rk']).copy()
dr['avg_rk'] = (dr['cur_rk'] + dr['sub_rk']) / 2

print()
print('=== 強さPT × ランカー順位 相関 ===')
print(f'PT × cur_rk 相関: {dr["強さPT"].corr(dr["cur_rk"]):.3f}')
print(f'PT × sub_rk 相関: {dr["強さPT"].corr(dr["sub_rk"]):.3f}')
print(f'PT × avg_rk 相関: {dr["強さPT"].corr(dr["avg_rk"]):.3f}')
print(f'cur_rk × sub_rk 相関: {dr["cur_rk"].corr(dr["sub_rk"]):.3f}')

print()
print('=== ランカー順位1位のPT分布 vs 非1位 ===')
for col, label in [('cur_rk', 'cur'), ('sub_rk', 'sub')]:
    r1  = dr[dr[col] == 1]['強さPT']
    r2  = dr[dr[col] == 2]['強さPT']
    r3p = dr[dr[col] >= 3]['強さPT']
    print(f'{label}ランカー1位: 平均{r1.mean():.1f} / 中央値{r1.median():.1f} (n={len(r1)})')
    print(f'{label}ランカー2位: 平均{r2.mean():.1f} / 中央値{r2.median():.1f} (n={len(r2)})')
    print(f'{label}ランカー3位以下: 平均{r3p.mean():.1f} / 中央値{r3p.median():.1f} (n={len(r3p)})')
    print()

print('=== 勝率: PT閾値 × ランカー1位かどうか ===')
print(f'{"PT閾値":>8}  {"ランカー1位":>10}  {"非1位":>10}  {"差":>6}')
for t in [50, 55, 57, 60, 65]:
    pt_ok = dr['強さPT'] >= t
    r1_cur = dr['cur_rk'] == 1
    r1_sub = dr['sub_rk'] == 1
    both_r1 = r1_cur & r1_sub
    g1 = dr[pt_ok & both_r1]
    g2 = dr[pt_ok & ~both_r1]
    if len(g1) == 0 or len(g2) == 0: continue
    wr1 = (g1['着_num'] == 1).mean()
    wr2 = (g2['着_num'] == 1).mean()
    pr1 = (g1['着_num'] <= 3).mean()
    pr2 = (g2['着_num'] <= 3).mean()
    print(f'PT{t}以上  両ランカー1位: 勝率{wr1:.1%} 3着内{pr1:.1%}(n={len(g1)})  '
          f'非1位: 勝率{wr2:.1%} 3着内{pr2:.1%}(n={len(g2)})')
