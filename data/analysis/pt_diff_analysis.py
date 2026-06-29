# coding: utf-8
"""強さPT レース内差分分析"""
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

import pyarrow.parquet as pq_mod
pq_path = os.path.join(base_dir, 'data', 'processed', 'all_venues_features.parquet')
all_feats = list(set(cur_features + sub_features))
avail = set(pq_mod.read_schema(pq_path).names)
load_cols = ['馬名S', '日付', '距離'] + [c for c in avail if c in set(all_feats)]
df_pq = pd.read_parquet(pq_path, columns=load_cols)
df_pq['日付_n'] = pd.to_numeric(df_pq['日付'], errors='coerce')
pq_latest = df_pq.sort_values('日付_n').groupby('馬名S', sort=False).last().reset_index()
print(f"Parquet読込完了: {len(pq_latest)}頭分")

cur_model_cache = {}; sub_model_cache = {}

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

VENUE_MAP = {'中山':'中','東京':'東','阪神':'阪','中京':'名','京都':'京',
             '函館':'函','新潟':'新','小倉':'小','札幌':'札','福島':'福'}

result_csvs = sorted(glob.glob(os.path.join(base_dir, 'data', 'raw', 'results', '出馬表形式*結果*.csv')))
all_records = []

for rf in result_csvs:
    fname = os.path.basename(rf)
    try: dfr = pd.read_csv(rf, encoding='cp932', low_memory=False)
    except: dfr = pd.read_csv(rf, encoding='utf-8', low_memory=False)
    if '日付S' not in dfr.columns: continue
    ds = str(dfr['日付S'].iloc[0]).replace('/', '.').split('.')
    dnum = (int(ds[0]) - 2000) * 10000 + int(ds[1]) * 100 + int(ds[2])

    dfr['着_num']  = dfr['着'].apply(_zen)
    dfr['_tan']    = pd.to_numeric(dfr['単勝'], errors='coerce')
    dfr['_odds']   = pd.to_numeric(dfr['単オッズ'], errors='coerce')
    dfr['会場']    = dfr['場所'].astype(str).map(VENUE_MAP).fillna(dfr['場所'].astype(str))
    dfr['_surface']= dfr['芝ダ'].astype(str).str.strip() if '芝ダ' in dfr.columns else 'ダ'
    dfr['cur_key'] = dfr['会場'] + '_' + dfr['_surface'] + dfr['距離'].astype(str)
    dfr['_dist_band'] = dfr['距離'].apply(get_distance_band)
    _mask = (dfr['_surface'] == 'ダ') & (dfr['_dist_band'].isin(['中距離', '長距離']))
    dfr.loc[_mask, '_dist_band'] = '中長距離'
    if 'クラス' in dfr.columns:
        _cls_map = {'新馬': 1, '未勝利': 2, '1勝': 3, '2勝': 4}
        dfr['クラス_rank'] = dfr['クラス'].apply(
            lambda v: next((r for k, r in _cls_map.items() if k in str(v)), 5))
    dfr['_cls_group'] = dfr['クラス_rank'].apply(get_class_group) if 'クラス_rank' in dfr.columns else '3勝以上'
    dfr['sub_key']  = dfr['_surface'] + '_' + dfr['_dist_band'] + '_' + dfr['_cls_group']
    dfr['性別_num'] = dfr['性別'].map({'牡': 0, '牝': 1, 'セ': 2}).astype(float)

    feat_cols = ['馬名S'] + [c for c in all_feats if c in pq_latest.columns]
    merged = dfr.merge(pq_latest[feat_cols], on='馬名S', how='left', suffixes=('', '_p'))
    for c in all_feats:
        if c in merged.columns: merged[c] = pd.to_numeric(merged[c], errors='coerce')
    rkey = merged['場所'].astype(str) + '_' + merged['Ｒ'].astype(str)
    merged['_race_key'] = rkey
    win_tan = merged[merged['着_num'] == 1].drop_duplicates('_race_key').set_index('_race_key')['_tan']
    merged['_tansho'] = merged['_race_key'].map(win_tan)

    for gk, idx in merged.groupby(['場所', 'Ｒ'], sort=False).groups.items():
        sub = merged.loc[idx].copy()
        ck = sub['cur_key'].iloc[0]; sk = sub['sub_key'].iloc[0]
        cur_cs = np.full(len(sub), np.nan); sub_cs = np.full(len(sub), np.nan)
        cm = get_cur_model(ck)
        if cm:
            m, wf = cm
            for c in wf:
                if c not in sub.columns: sub[c] = np.nan
            prob = m.predict_proba(sub[wf])[:, 1]
            st = cur_models[ck].get('stats', {})
            wm = st.get('win_mean', prob.mean()); ws = st.get('win_std', prob.std())
            cur_cs = 50 + 10 * (prob - wm) / (ws if ws > 0 else 1)
        sm = get_sub_model(sk)
        if sm:
            m, wf = sm
            for c in wf:
                if c not in sub.columns: sub[c] = np.nan
            prob = m.predict_proba(sub[wf])[:, 1]
            st = sub_models[sk].get('stats', {})
            wm = st.get('win_mean', prob.mean()); ws = st.get('win_std', prob.std())
            sub_cs = 50 + 10 * (prob - wm) / (ws if ws > 0 else 1)
        cur_s = pd.Series(cur_cs, index=sub.index)
        sub_s = pd.Series(sub_cs, index=sub.index)
        both = ~(cur_s.isna() | sub_s.isna())
        pt = pd.Series(np.nan, index=sub.index)
        pt[both] = (cur_s[both] + sub_s[both]) / 2
        pt[~both] = cur_s.fillna(sub_s)[~both]
        race_mean = pt.mean()
        for i, (row_idx, row) in enumerate(sub.iterrows()):
            all_records.append({
                '日付_num': dnum, '場所': str(row.get('場所', '')), 'Ｒ': row.get('Ｒ'),
                '馬名S': row.get('馬名S', ''), '着_num': row['着_num'],
                '_tansho': row['_tansho'], '_odds': row['_odds'],
                '強さPT': pt.iloc[i], 'race_mean': race_mean,
            })
    print(f"  {fname[:20]}... {dnum} 処理完了")

df = pd.DataFrame(all_records)
df = df.dropna(subset=['着_num', '強さPT', '_tansho'])
df['pt_diff'] = df['強さPT'] - df['race_mean']

print(f"\n総データ: {len(df)}頭 / {df['日付_num'].nunique()}日")
print(f"pt_diff 分布: mean={df['pt_diff'].mean():.2f} / std={df['pt_diff'].std():.1f} / max={df['pt_diff'].max():.1f}")

BET = 100
print()
print("=" * 60)
print("レース内PT差（馬のPT - レース平均PT）別 勝率・ROI")
print(f"{'PT差':>8}  {'頭数':>6}  {'勝率':>6}  {'平均オッズ':>8}  {'単勝ROI':>8}")
print("=" * 60)
for th in [0, 3, 5, 7, 8, 10, 12, 15]:
    mask = df['pt_diff'] >= th
    s = df[mask]
    if len(s) == 0: continue
    n = len(s); wins = int((s['着_num'] == 1).sum())
    wr = wins / n; avg_o = s['_odds'].mean()
    ret = s[s['着_num'] == 1]['_tansho'].sum() * BET / 100
    roi = ret / (n * BET) - 1.0
    marker = ' ◀' if roi > 0 else ''
    print(f"  +{th:>2}以上:  {n:5d}頭  {wr:5.1%}  {avg_o:7.1f}倍  {roi:+7.1%}{marker}")

print()
print("=" * 60)
print("各レースのPT差1位馬（差が最大の馬）の成績")
print("= PT差閾値別 =")
print("=" * 60)
for min_diff in [3, 5, 7, 10]:
    top1 = df.loc[df.groupby(['日付_num', '場所', 'Ｒ'])['pt_diff'].idxmax()]
    top1 = top1[top1['pt_diff'] >= min_diff]
    if len(top1) == 0: continue
    n = len(top1); wins = int((top1['着_num'] == 1).sum())
    ret = top1[top1['着_num'] == 1]['_tansho'].sum() * BET / 100
    roi = ret / (n * BET) - 1.0
    avg_o = top1['_odds'].mean()
    marker = ' ◀' if roi > 0 else ''
    print(f"  PT差{min_diff}以上のレース: {n}R  勝率 {wins/n:.1%}  平均{avg_o:.1f}倍  ROI {roi:+.1%}{marker}")

print()
print("=" * 60)
print("PT差 vs PT絶対値 組み合わせ（PT差5以上 かつ PT絶対値別）")
print("=" * 60)
for pt_abs in [55, 57, 60, 62, 65]:
    mask = (df['pt_diff'] >= 5) & (df['強さPT'] >= pt_abs)
    s = df[mask]
    if len(s) == 0: continue
    n = len(s); wins = int((s['着_num'] == 1).sum())
    ret = s[s['着_num'] == 1]['_tansho'].sum() * BET / 100
    roi = ret / (n * BET) - 1.0
    avg_o = s['_odds'].mean()
    marker = ' ◀' if roi > 0 else ''
    print(f"  差5以上 & PT{pt_abs}以上:  {n:4d}頭  勝率{wins/n:5.1%}  {avg_o:.1f}倍  ROI {roi:+.1%}{marker}")

print()
print("=" * 60)
print("3着内率（PT絶対値 / PT差 / PT差1位馬）")
print("=" * 60)
df['place'] = df['着_num'] <= 3
print("【PT絶対値別】")
print(f"{'PT閾値':>8}  {'頭数':>6}  {'勝率':>6}  {'3着内率':>8}")
for t in [50, 55, 57, 58, 60, 62, 65, 70]:
    s = df[df['強さPT'] >= t]
    if len(s) == 0: continue
    print(f"  {t}以上:  {len(s):5d}頭  {(s['着_num']==1).mean():5.1%}  {s['place'].mean():7.1%}")

print()
print("【PT差（レース平均との差）別】")
print(f"{'PT差':>8}  {'頭数':>6}  {'勝率':>6}  {'3着内率':>8}")
for t in [0, 3, 5, 7, 10, 15]:
    s = df[df['pt_diff'] >= t]
    if len(s) == 0: continue
    print(f"  +{t}以上:  {len(s):5d}頭  {(s['着_num']==1).mean():5.1%}  {s['place'].mean():7.1%}")

print()
print("【各レースPT差1位馬 閾値別】")
for min_diff in [0, 3, 5, 7, 10]:
    top1 = df.loc[df.groupby(['日付_num', '場所', 'Ｒ'])['pt_diff'].idxmax()]
    top1 = top1[top1['pt_diff'] >= min_diff]
    if len(top1) == 0: continue
    n = len(top1)
    print(f"  PT差{min_diff}以上の1位馬: {n:3d}R  勝率 {(top1['着_num']==1).mean():.1%}  3着内率 {top1['place'].mean():.1%}")
