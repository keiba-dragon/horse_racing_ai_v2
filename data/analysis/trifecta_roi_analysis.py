# coding: utf-8
"""三連単 ROI分析 — 各戦略の回収率シミュレーション"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import pandas as pd, numpy as np, os, pickle, json, re, glob, time
from itertools import permutations

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_dir = os.path.join(base_dir, 'models_2025')

# ── ユーティリティ ──
def _zen(s):
    if pd.isna(s): return np.nan
    s = str(s).strip().translate(str.maketrans('０１２３４５６７８９', '0123456789'))
    m = re.search(r'\d+', s)
    return int(m.group()) if m else np.nan

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

# ── モデル読み込み ──
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

# ── Parquetから最新特徴量（馬名S単位）を取得 ──
import pyarrow.parquet as pq_mod
pq_path = os.path.join(base_dir, 'data', 'processed', 'all_venues_features.parquet')
all_feats = list(set(cur_features + sub_features + cur_ranker_features + sub_ranker_features))
avail = set(pq_mod.read_schema(pq_path).names)
feat_load = ['馬名S', '日付'] + [c for c in avail if c in set(all_feats)]
feat_load = list(dict.fromkeys(feat_load))

df_pq = pd.read_parquet(pq_path, columns=feat_load)
df_pq['日付_n'] = pd.to_numeric(df_pq['日付'], errors='coerce')
pq_latest = df_pq.sort_values('日付_n').groupby('馬名S', sort=False).last().reset_index()
print(f"Parquet特徴量: {len(pq_latest)}頭分")

# ── 結果CSVを読み込み（三連単配当・着順・予測用特徴量） ──
VENUE_MAP = {'中山':'中','東京':'東','阪神':'阪','中京':'名','京都':'京',
             '函館':'函','新潟':'新','小倉':'小','札幌':'札','福島':'福'}

def parse_date(s):
    try:
        parts = str(s).replace('/', '.').split('.')
        y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
        return (y - 2000) * 10000 + m * 100 + d
    except: return np.nan

result_csvs = sorted(glob.glob(os.path.join(base_dir, 'data', 'raw', 'results', '出馬表形式*結果*.csv')))
print(f"結果CSV: {len(result_csvs)}件")

all_merged = []
for rf in result_csvs:
    try: dfr = pd.read_csv(rf, encoding='cp932', low_memory=False)
    except: dfr = pd.read_csv(rf, encoding='utf-8', low_memory=False)
    if '日付S' not in dfr.columns or '３連単配当' not in dfr.columns: continue
    dfr['日付_n']  = dfr['日付S'].apply(parse_date)
    dfr['会場']    = dfr['場所'].astype(str).map(VENUE_MAP).fillna(dfr['場所'].astype(str))
    dfr['着_num']  = dfr['着'].apply(_zen)
    dfr['_tan3']   = pd.to_numeric(dfr['３連単配当'], errors='coerce')
    dfr['_odds']   = pd.to_numeric(dfr.get('単オッズ', dfr.get('単勝オッズ', pd.Series(np.nan, index=dfr.index))), errors='coerce')
    # 芝ダ・距離・クラス
    dfr['_surface']  = dfr['芝ダ'].astype(str).str.strip() if '芝ダ' in dfr.columns else 'ダ'
    dfr['_dist_num'] = dfr['距離'].astype(str)
    dfr['cur_key']   = dfr['会場'] + '_' + dfr['_surface'] + dfr['_dist_num']
    dfr['_dist_band'] = dfr['距離'].apply(get_distance_band)
    _mask = (dfr['_surface'] == 'ダ') & (dfr['_dist_band'].isin(['中距離', '長距離']))
    dfr.loc[_mask, '_dist_band'] = '中長距離'
    if 'クラス' in dfr.columns:
        _cls_map = {'新馬': 1, '未勝利': 2, '1勝': 3, '2勝': 4}
        dfr['クラス_rank'] = dfr['クラス'].apply(lambda v: next((r for k,r in _cls_map.items() if k in str(v)), 5))
    dfr['_cls_group'] = dfr['クラス_rank'].apply(get_class_group) if 'クラス_rank' in dfr.columns else '3勝以上'
    dfr['sub_key']   = dfr['_surface'] + '_' + dfr['_dist_band'] + '_' + dfr['_cls_group']
    dfr['性別_num']  = dfr['性別'].map({'牡': 0, '牝': 1, 'セ': 2}).astype(float)
    # Parquet特徴量マージ
    feat_cols = ['馬名S'] + [c for c in all_feats if c in pq_latest.columns]
    merged = dfr.merge(pq_latest[feat_cols], on='馬名S', how='left', suffixes=('','_pq'))
    for c in all_feats:
        if c in merged.columns:
            merged[c] = pd.to_numeric(merged[c], errors='coerce')
    all_merged.append(merged)

if not all_merged:
    print("結果CSVが読み込めません")
    sys.exit(1)

df_all = pd.concat(all_merged, ignore_index=True)
print(f"結果CSV合計: {len(df_all)}頭 / {df_all['日付_n'].nunique()}日 / {df_all.groupby(['日付_n','会場','Ｒ']).ngroups}レース")

# ── モデルキャッシュ ──
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
    return sub_ranker_cache.get(ck)

# ── レース別予測 ──
print("予測計算中...")
t0 = time.time()
records = []
race_groups = list(df_all.groupby(['日付_n', '会場', 'Ｒ'], sort=True).groups.items())
print(f"総レース数: {len(race_groups)}")

for i, (gk, idx) in enumerate(race_groups):
    sub = df_all.loc[idx].copy()
    if len(sub) < 3: continue
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
            sc = sub.copy()
            for c in cur_ranker_features:
                if c not in sc.columns: sc[c] = np.nan
            scores = cr.predict(sc[cur_ranker_features])
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
            sc = sub.copy()
            for c in sub_ranker_features:
                if c not in sc.columns: sc[c] = np.nan
            scores = sr.predict(sc[sub_ranker_features])
            sub_rk = pd.Series(scores).rank(ascending=False).values

    cur_s = pd.Series(cur_cs, index=sub.index)
    sub_s = pd.Series(sub_cs, index=sub.index)
    both  = ~(cur_s.isna() | sub_s.isna())
    pt    = pd.Series(np.nan, index=sub.index)
    pt[both]  = (cur_s[both] + sub_s[both]) / 2
    pt[~both] = cur_s.fillna(sub_s)[~both]
    cur_diff = cur_s - cur_s.mean()
    sub_diff = sub_s - sub_s.mean()

    tan3_val = sub['_tan3'].dropna().iloc[0] if sub['_tan3'].notna().any() else np.nan

    for j, (row_idx, row) in enumerate(sub.iterrows()):
        records.append({
            '日付_n':     gk[0],
            '会場':       gk[1],
            'Ｒ':         int(float(gk[2])),
            '馬名S':      row['馬名S'],
            '着_num':     row['着_num'],
            '強さPT':     pt.iloc[j],
            'cur_rk':     cur_rk[j],
            'sub_rk':     sub_rk[j],
            'cur_diff':   cur_diff.iloc[j],
            'sub_diff':   sub_diff.iloc[j],
            '_odds':      row['_odds'],
            '三連単配当': tan3_val,
            'n_horses':   n,
        })

df = pd.DataFrame(records).dropna(subset=['着_num', '強さPT'])
df['both_r1']   = (df['cur_rk'] == 1) & (df['sub_rk'] == 1)
df['mask_geki'] = df['both_r1'] & (df['cur_diff'] >= 10) & (df['sub_diff'] >= 10)
df['mask_maru'] = df['both_r1'] & (df['sub_diff'] >= 10) & ~df['mask_geki']
df['mask_tri']  = (df['cur_rk'] <= 2) & (df['sub_rk'] <= 2) & ~df['both_r1'] & (df['sub_diff'] >= 10)

n_races = df.groupby(['日付_n','会場','Ｒ']).ngroups
n_with_tan3 = df[df['三連単配当'].notna()].groupby(['日付_n','会場','Ｒ']).ngroups
print(f"\n完了: {len(df)}頭 / {df['日付_n'].nunique()}日 / {n_races}レース / 三連単配当あり:{n_with_tan3}レース ({time.time()-t0:.0f}s)")

BET = 100

def simulate_strategy(df, name, selector_fn, n_combo_fn=None):
    """
    selector_fn(race_df) -> list of 馬名S (ordered: predicted 1st, 2nd, 3rd candidates)
    n_combo_fn(picks) -> list of (1着,2着,3着) tuples to bet on
    """
    total_bet = 0; total_return = 0; n_hit = 0; n_race = 0
    no_tan3 = 0

    for (日付, 会場, r), grp in df.groupby(['日付_n', '会場', 'Ｒ']):
        grp = grp.dropna(subset=['強さPT'])
        if len(grp) < 3: continue
        tan3 = grp['三連単配当'].iloc[0] if '三連単配当' in grp.columns else np.nan
        if pd.isna(tan3):
            no_tan3 += 1
            continue

        picks = selector_fn(grp)  # list of 馬名S
        if not picks or len(picks) < 3: continue

        combos = n_combo_fn(picks) if n_combo_fn else list(permutations(picks[:3], 3))
        n_combo = len(combos)
        if n_combo == 0: continue

        actual_1 = grp[grp['着_num'] == 1]['馬名S'].values
        actual_2 = grp[grp['着_num'] == 2]['馬名S'].values
        actual_3 = grp[grp['着_num'] == 3]['馬名S'].values
        if len(actual_1) == 0 or len(actual_2) == 0 or len(actual_3) == 0: continue

        a1, a2, a3 = actual_1[0], actual_2[0], actual_3[0]

        hit = any(c[0] == a1 and c[1] == a2 and c[2] == a3 for c in combos)
        total_bet    += BET * n_combo
        total_return += (tan3 if hit else 0)
        n_hit        += (1 if hit else 0)
        n_race       += 1

    roi = (total_return / total_bet - 1) * 100 if total_bet > 0 else np.nan
    hit_rate = n_hit / n_race * 100 if n_race > 0 else 0
    avg_combo = total_bet / BET / n_race if n_race > 0 else 0
    print(f'  {name:<40} レース数:{n_race:4d}  平均点数:{avg_combo:4.1f}  '
          f'的中率:{hit_rate:4.1f}%  ROI:{roi:+.1f}%  '
          f'(的中{n_hit}回 / 投資{total_bet//100}百円 / 回収{total_return//100}百円)')
    return {'name': name, 'n_race': n_race, 'n_hit': n_hit, 'roi': roi, 'avg_combo': avg_combo}

# ── 戦略定義 ──

def pt_top3_box(grp):
    return grp.nlargest(3, '強さPT')['馬名S'].tolist()

def pt_top4_box(grp):
    return grp.nlargest(4, '強さPT')['馬名S'].tolist()

def pt_top5_box(grp):
    return grp.nlargest(5, '強さPT')['馬名S'].tolist()

def geki_axis_pt23(grp):
    """激熱1着固定 × PT上位2・3着"""
    geki = grp[grp['mask_geki']]
    if len(geki) == 0: return []
    axis = geki.nlargest(1, '強さPT')['馬名S'].tolist()
    rest = grp[~grp['馬名S'].isin(axis)].nlargest(3, '強さPT')['馬名S'].tolist()
    return axis + rest  # axis[0]が軸

def bothr1_axis_pt23(grp):
    """both_r1かつPT最高 × PT上位3頭"""
    b1 = grp[grp['both_r1']]
    if len(b1) == 0: return []
    axis = b1.nlargest(1, '強さPT')['馬名S'].tolist()
    rest = grp[~grp['馬名S'].isin(axis)].nlargest(4, '強さPT')['馬名S'].tolist()
    return axis + rest

def geki_only_races(grp):
    """激熱存在レースのみ × PT上位3頭ボックス"""
    if grp['mask_geki'].sum() == 0: return []
    return grp.nlargest(3, '強さPT')['馬名S'].tolist()

def geki_exists_top4(grp):
    """激熱存在レースのみ × PT上位4頭ボックス"""
    if grp['mask_geki'].sum() == 0: return []
    return grp.nlargest(4, '強さPT')['馬名S'].tolist()

def high_pt_gap_top3(grp):
    """レース内PT差が大きい（PT1位-PT3位 >= 10）× PT上位3頭"""
    pts = grp.nlargest(3, '強さPT')['強さPT'].values
    if len(pts) < 3 or (pts[0] - pts[2]) < 10: return []
    return grp.nlargest(3, '強さPT')['馬名S'].tolist()

def high_pt_gap_top4(grp):
    """PT差>=10 × PT上位4頭ボックス"""
    pts = grp.nlargest(3, '強さPT')['強さPT'].values
    if len(pts) < 3 or (pts[0] - pts[2]) < 10: return []
    return grp.nlargest(4, '強さPT')['馬名S'].tolist()

def ranker_axis_combo(grp):
    """both_r1馬を1着軸 × PT上位3頭を2・3着"""
    b1 = grp[grp['both_r1']]
    if len(b1) == 0: return []
    axis = b1.nlargest(1, '強さPT')['馬名S'].tolist()
    rest = grp[~grp['馬名S'].isin(axis)].nlargest(3, '強さPT')['馬名S'].tolist()
    if len(rest) < 2: return []
    return axis + rest

def geki_1_maru_23(grp):
    """激熱1着固定 × (〇▲)2・3着"""
    geki = grp[grp['mask_geki']]
    if len(geki) == 0: return []
    axis = geki.nlargest(1, '強さPT')['馬名S'].tolist()
    rest_mask = grp['mask_maru'] | grp['mask_tri']
    rest = grp[rest_mask & ~grp['馬名S'].isin(axis)]
    if len(rest) < 2:
        rest = grp[~grp['馬名S'].isin(axis)].nlargest(3, '強さPT')
    return axis + rest['馬名S'].tolist()

def pt60_top3(grp):
    """PT60以上の馬が3頭以上いるレース × その上位3頭"""
    hi = grp[grp['強さPT'] >= 60]
    if len(hi) < 3: return []
    return hi.nlargest(3, '強さPT')['馬名S'].tolist()

def pt60_exists_box4(grp):
    """PT60以上が1頭以上 × PT上位4頭"""
    if (grp['強さPT'] >= 60).sum() == 0: return []
    return grp.nlargest(4, '強さPT')['馬名S'].tolist()

# ── フォーメーション定義 ──
def box3(picks):
    """ボックス(上位3頭の全順列 = 6点)"""
    return list(permutations(picks[:3], 3))

def box4(picks):
    """ボックス(上位4頭の全順列 = 24点)"""
    return list(permutations(picks[:4], 3))

def box5(picks):
    """ボックス(上位5頭の全順列 = 60点)"""
    return list(permutations(picks[:5], 3))

def form_1x2x3(picks):
    """1着固定フォーメーション: 軸(picks[0]) × 2着(picks[1:3]) × 3着(picks[1:4]) = 最大6点"""
    if len(picks) < 3: return []
    axis = picks[0]
    cands = picks[1:]
    combos = []
    for p2 in cands[:2]:
        for p3 in cands[:3]:
            if p2 != p3:
                combos.append((axis, p2, p3))
    return combos

def form_1x3x4(picks):
    """1着固定: 軸 × 2着3頭 × 3着4頭 = 最大12点"""
    if len(picks) < 4: return []
    axis = picks[0]; cands = picks[1:]
    combos = []
    for p2 in cands[:3]:
        for p3 in cands[:4]:
            if p2 != p3:
                combos.append((axis, p2, p3))
    return list(set(combos))

print()
print('=' * 90)
print('三連単 ROI シミュレーション（BET=100円/点）')
print('=' * 90)

results = []
strategies = [
    ('PT上位3頭ボックス(6点)',          pt_top3_box,          box3),
    ('PT上位4頭ボックス(24点)',         pt_top4_box,          box4),
    ('PT上位5頭ボックス(60点)',         pt_top5_box,          box5),
    ('激熱1着軸×PT上位3(6点フォーム)', geki_axis_pt23,       form_1x2x3),
    ('激熱1着軸×PT上位4(12点フォーム)',geki_axis_pt23,       form_1x3x4),
    ('激熱レース限定×PT上位3ボックス',  geki_only_races,      box3),
    ('激熱レース限定×PT上位4ボックス',  geki_exists_top4,     box4),
    ('both_r1軸×PT上位3フォーム',      bothr1_axis_pt23,     form_1x2x3),
    ('both_r1軸×PT上位4フォーム',      ranker_axis_combo,    form_1x3x4),
    ('PT差10以上レース×上位3ボックス',  high_pt_gap_top3,     box3),
    ('PT差10以上レース×上位4ボックス',  high_pt_gap_top4,     box4),
    ('激熱1軸×〇▲2着フォーム',         geki_1_maru_23,       form_1x2x3),
    ('PT60以上3頭レース×ボックス',      pt60_top3,            box3),
    ('PT60以上存在レース×上位4ボックス',pt60_exists_box4,     box4),
]

for name, sel_fn, combo_fn in strategies:
    r = simulate_strategy(df, name, sel_fn, combo_fn)
    results.append(r)

print()
print('=' * 90)
print('ROI+ の戦略まとめ')
print('=' * 90)
roi_plus = sorted([r for r in results if r['roi'] is not np.nan and r['roi'] > 0],
                  key=lambda x: x['roi'], reverse=True)
if roi_plus:
    for r in roi_plus:
        print(f'  ✅ {r["name"]:<40} ROI:{r["roi"]:+.1f}%  的中{r["n_hit"]}回/{r["n_race"]}レース  平均{r["avg_combo"]:.1f}点')
else:
    print('  ROI+の戦略なし → 条件を絞り込む必要あり')
