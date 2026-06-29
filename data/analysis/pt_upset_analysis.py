# coding: utf-8
"""低PT馬が勝つときの傾向分析"""
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
records = []

for rf in result_csvs:
    try: dfr = pd.read_csv(rf, encoding='cp932', low_memory=False)
    except: dfr = pd.read_csv(rf, encoding='utf-8', low_memory=False)
    if '日付S' not in dfr.columns: continue
    dfr['着_num']   = dfr['着'].apply(_zen)
    dfr['_odds']    = pd.to_numeric(dfr['単オッズ'], errors='coerce')
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
        race_max_pt = pt.max()

        for i, (row_idx, row) in enumerate(sub.iterrows()):
            records.append({
                '着_num':      row['着_num'],
                '強さPT':      pt.iloc[i],
                'race_max_pt': race_max_pt,
                '_odds':       row['_odds'],
                '_surface':    row['_surface'],
                '_dist_band':  row['_dist_band'],
                '_cls_group':  row['_cls_group'],
                '場所':        str(row.get('場所', '')),
            })

df = pd.DataFrame(records).dropna(subset=['着_num', '強さPT', '_odds'])

# 低PT勝ち馬 vs 高PT勝ち馬
low_win  = df[(df['着_num'] == 1) & (df['強さPT'] < 47)]
high_win = df[(df['着_num'] == 1) & (df['強さPT'] >= 57)]

print(f'低PT勝ち馬（PT<47）: {len(low_win)}頭')
print(f'高PT勝ち馬（PT≥57）: {len(high_win)}頭')

print()
print('=' * 55)
print('オッズ分布比較')
print(f'{"":>12}  {"低PT勝ち":>10}  {"高PT勝ち":>10}')
print('=' * 55)
for label, lo, hi in [('~3倍',0,3),('3~5倍',3,5),('5~10倍',5,10),('10~20倍',10,20),('20倍~',20,999)]:
    lc = ((low_win['_odds'] >= lo) & (low_win['_odds'] < hi)).sum()
    hc = ((high_win['_odds'] >= lo) & (high_win['_odds'] < hi)).sum()
    lp = lc/len(low_win)*100; hp = hc/len(high_win)*100
    print(f'  {label:>8}:  {lc:3d}頭({lp:4.1f}%)  {hc:3d}頭({hp:4.1f}%)')
print(f'  平均オッズ:  {low_win["_odds"].mean():.1f}倍          {high_win["_odds"].mean():.1f}倍')

print()
print('=' * 55)
print('芝ダ')
for surf in ['芝', 'ダ']:
    lc = (low_win['_surface'] == surf).sum()
    hc = (high_win['_surface'] == surf).sum()
    print(f'  {surf}: 低PT {lc}頭({lc/len(low_win)*100:.1f}%)  高PT {hc}頭({hc/len(high_win)*100:.1f}%)')

print()
print('=' * 55)
print('距離帯')
for band in ['短距離', 'マイル', '中距離', '長距離']:
    lc = (low_win['_dist_band'] == band).sum()
    hc = (high_win['_dist_band'] == band).sum()
    print(f'  {band:>5}: 低PT {lc}頭({lc/len(low_win)*100:.1f}%)  高PT {hc}頭({hc/len(high_win)*100:.1f}%)')

print()
print('=' * 55)
print('クラス')
for cls in ['新馬', '未勝利', '1勝', '2勝', '3勝以上']:
    lc = (low_win['_cls_group'] == cls).sum()
    hc = (high_win['_cls_group'] == cls).sum()
    print(f'  {cls:>6}: 低PT {lc}頭({lc/len(low_win)*100:.1f}%)  高PT {hc}頭({hc/len(high_win)*100:.1f}%)')

print()
print('=' * 55)
print('レース内最高PTとの差（低PT勝ち馬がどれだけ差をつけられていたか）')
low_win2 = df[(df['着_num'] == 1) & (df['強さPT'] < 47)].copy()
low_win2['pt_gap'] = low_win2['race_max_pt'] - low_win2['強さPT']
print(f'  平均ギャップ: {low_win2["pt_gap"].mean():.1f}pt')
print(f'  中央値:       {low_win2["pt_gap"].median():.1f}pt')
print(f'  最大ギャップ: {low_win2["pt_gap"].max():.1f}pt（最も大番狂わせ）')
for label, lo, hi in [('5以下',0,5),('5~10',5,10),('10~15',10,15),('15~',15,999)]:
    cnt = ((low_win2['pt_gap'] >= lo) & (low_win2['pt_gap'] < hi)).sum()
    print(f'  ギャップ{label}: {cnt}頭({cnt/len(low_win2)*100:.1f}%)')
