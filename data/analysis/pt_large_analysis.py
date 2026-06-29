# coding: utf-8
"""強さPT × ランカー 大規模分析（2025後半〜2026）"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import pandas as pd, numpy as np, os, pickle, json, re, glob, time

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

def _zen(s):
    if pd.isna(s): return np.nan
    s = str(s).strip().translate(str.maketrans('０１２３４５６７８９', '0123456789'))
    m = re.search(r'\d+', s)
    return int(m.group()) if m else np.nan

def get_class_group(r):
    try: r = int(float(r))
    except: return '3勝以上'
    if r == 1: return '新馬'
    elif r == 2: return '未勝利'
    elif r == 3: return '1勝'
    elif r == 4: return '2勝'
    return '3勝以上'

# モデル読み込み
with open(os.path.join(model_dir, 'model_info.json'), encoding='utf-8') as f:
    cur_info = json.load(f)
cur_features = cur_info['features']; cur_models = cur_info['models']

with open(os.path.join(model_dir, 'submodel', 'submodel_info.json'), encoding='utf-8') as f:
    sub_info = json.load(f)
sub_features = sub_info['features']; sub_models = sub_info['models']

cur_rankers = {}; sub_rankers = {}
cur_ranker_features = []; sub_ranker_features = []
rinfo_path     = os.path.join(model_dir, 'ranker', 'ranker_info.json')
rinfo_sub_path = os.path.join(model_dir, 'submodel_ranker', 'class_ranker_info.json')
if os.path.exists(rinfo_path):
    with open(rinfo_path, encoding='utf-8') as f:
        rd = json.load(f); cur_rankers = rd.get('rankers', {}); cur_ranker_features = rd.get('features', [])
if os.path.exists(rinfo_sub_path):
    with open(rinfo_sub_path, encoding='utf-8') as f:
        rd = json.load(f); sub_rankers = rd.get('rankers', {}); sub_ranker_features = rd.get('features', [])
print(f"cur_models:{len(cur_models)} cur_rankers:{len(cur_rankers)} sub_models:{len(sub_models)} sub_rankers:{len(sub_rankers)}")

# Parquet読み込み（2025/7以降）
import pyarrow.parquet as pq_mod
pq_path = os.path.join(base_dir, 'data', 'processed', 'all_venues_features.parquet')
all_feats = list(set(cur_features + sub_features + cur_ranker_features + sub_ranker_features))
avail = set(pq_mod.read_schema(pq_path).names)

load_cols = ['馬名S', '日付', '開催', 'Ｒ', '距離', '芝・ダ', '着順', '単勝配当', '人気',
             'クラス_rank', '頭数', '性別', '年齢'] + [c for c in avail if c in set(all_feats)]
load_cols = list(dict.fromkeys(load_cols))  # dedup

df_all = pd.read_parquet(pq_path, columns=load_cols)
df_all['日付_n'] = pd.to_numeric(df_all['日付'], errors='coerce')
df_all['着_num'] = df_all['着順'].apply(_zen)
df_all['_tan']   = pd.to_numeric(df_all['単勝配当'], errors='coerce')

# 2025/7以降
df_all = df_all[df_all['日付_n'] >= 250701].copy()
print(f"対象: {len(df_all)}行 / 日付範囲: {df_all['日付_n'].min()}〜{df_all['日付_n'].max()}")
print(f"着_num非NaN: {df_all['着_num'].notna().sum()}")

VENUE_MAP = {'中山':'中','東京':'東','阪神':'阪','中京':'名','京都':'京',
             '函館':'函','新潟':'新','小倉':'小','札幌':'札','福島':'福'}

df_all['会場']     = df_all['開催'].astype(str).str.extract(r'([^\d\s])')[0]
df_all['_surface'] = df_all['芝・ダ'].astype(str).str.strip()
# 距離から数字だけ抽出（例: '芝1800' → '1800'）
df_all['_dist_num'] = df_all['距離'].astype(str).apply(
    lambda x: re.search(r'\d+', x).group() if re.search(r'\d+', x) else '')
df_all['cur_key']  = df_all['会場'] + '_' + df_all['_surface'] + df_all['_dist_num']
df_all['_dist_band'] = df_all['_dist_num'].apply(get_distance_band)
_mask = (df_all['_surface'] == 'ダ') & (df_all['_dist_band'].isin(['中距離', '長距離']))
df_all.loc[_mask, '_dist_band'] = '中長距離'
df_all['_cls_group'] = df_all['クラス_rank'].apply(get_class_group)
df_all['sub_key']  = df_all['_surface'] + '_' + df_all['_dist_band'] + '_' + df_all['_cls_group']
df_all['性別_num'] = df_all['性別'].map({'牡': 0, '牝': 1, 'セ': 2}).astype(float)
for c in all_feats:
    if c in df_all.columns:
        df_all[c] = pd.to_numeric(df_all[c], errors='coerce')

# モデルキャッシュ
cur_model_cache = {}; sub_model_cache = {}; cur_ranker_cache = {}; sub_ranker_cache = {}

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

# レース別予測
t0 = time.time()
records = []
race_groups = list(df_all.groupby(['日付_n', '開催', 'Ｒ'], sort=True).groups.items())
print(f"総レース数: {len(race_groups)}")

for i, (gk, idx) in enumerate(race_groups):
    if i % 500 == 0:
        print(f"  {i}/{len(race_groups)} ... {time.time()-t0:.0f}s")
    sub = df_all.loc[idx].copy()
    if len(sub) < 2: continue
    ck = sub['cur_key'].iloc[0]; sk = sub['sub_key'].iloc[0]
    n  = len(sub)

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
            wf_r = cur_ranker_features if cur_ranker_features else (cr.feature_name() if hasattr(cr, 'feature_name') else wf)
            sc = sub.copy()
            for c in wf_r:
                if c not in sc.columns: sc[c] = np.nan
            scores = cr.predict(sc[wf_r])
            cur_rk = pd.Series(scores).rank(ascending=False).values

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
            wf_r = sub_ranker_features if sub_ranker_features else (sr.feature_name() if hasattr(sr, 'feature_name') else wf)
            sc = sub.copy()
            for c in wf_r:
                if c not in sc.columns: sc[c] = np.nan
            scores = sr.predict(sc[wf_r])
            sub_rk = pd.Series(scores).rank(ascending=False).values

    cur_s = pd.Series(cur_cs, index=sub.index)
    sub_s = pd.Series(sub_cs, index=sub.index)
    both  = ~(cur_s.isna() | sub_s.isna())
    pt    = pd.Series(np.nan, index=sub.index)
    pt[both]  = (cur_s[both] + sub_s[both]) / 2
    pt[~both] = cur_s.fillna(sub_s)[~both]
    race_mean = pt.mean()

    win_tan = sub[sub['着_num'] == 1]['_tan'].values
    win_tan_val = win_tan[0] if len(win_tan) > 0 else np.nan

    for j, (row_idx, row) in enumerate(sub.iterrows()):
        records.append({
            '日付_n':    gk[0],
            '着_num':    row['着_num'],
            '強さPT':    pt.iloc[j],
            'race_mean': race_mean,
            'cur_rk':    cur_rk[j],
            'sub_rk':    sub_rk[j],
            '_tan':      win_tan_val,
            'n_horses':  n,
        })

df = pd.DataFrame(records)
df = df.dropna(subset=['着_num', '強さPT'])
df['pt_diff'] = df['強さPT'] - df['race_mean']
df['place']   = df['着_num'] <= 3
print(f"\n完了: {len(df)}頭 / {df['日付_n'].nunique()}日 / {time.time()-t0:.0f}s")

BET = 100
print()
print('=' * 65)
print('【PT絶対値】勝率 / 3着内率')
print(f'{"PT閾値":>8}  {"頭数":>6}  {"勝率":>6}  {"3着内率":>8}')
print('=' * 65)
for t in [50, 55, 57, 58, 60, 62, 65, 70]:
    s = df[df['強さPT'] >= t]
    if len(s) == 0: continue
    print(f'  {t}以上:  {len(s):6d}頭  {(s["着_num"]==1).mean():5.1%}  {s["place"].mean():7.1%}')

print()
print('=' * 65)
print('【PT × 両ランカー1位】勝率 / 3着内率')
print(f'{"PT閾値":>8}  {"両R1位":>22}  {"非両R1位":>22}')
print('=' * 65)
dr = df.dropna(subset=['cur_rk', 'sub_rk'])
for t in [50, 55, 57, 60, 65]:
    pt_ok  = dr['強さPT'] >= t
    both_r1 = (dr['cur_rk'] == 1) & (dr['sub_rk'] == 1)
    g1 = dr[pt_ok & both_r1];  g2 = dr[pt_ok & ~both_r1]
    if len(g1) == 0: continue
    wr1 = (g1['着_num']==1).mean(); pr1 = g1['place'].mean()
    wr2 = (g2['着_num']==1).mean(); pr2 = g2['place'].mean()
    print(f'  PT{t}以上  両R1位: 勝率{wr1:.1%} 3着内{pr1:.1%}(n={len(g1):4d})  '
          f'非R1位: 勝率{wr2:.1%} 3着内{pr2:.1%}(n={len(g2):5d})')

print()
print('=' * 65)
print('【相関】')
dr2 = df.dropna(subset=['cur_rk', 'sub_rk'])
print(f'PT × cur_rk: {dr2["強さPT"].corr(dr2["cur_rk"]):.3f}')
print(f'PT × sub_rk: {dr2["強さPT"].corr(dr2["sub_rk"]):.3f}')
print(f'cur_rk × sub_rk: {dr2["cur_rk"].corr(dr2["sub_rk"]):.3f}')
