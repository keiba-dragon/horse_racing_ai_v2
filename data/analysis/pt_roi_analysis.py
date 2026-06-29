# coding: utf-8
"""強さPT別ROI分析"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import pandas as pd, numpy as np, os, pickle, json, re, glob

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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

# モデル読み込み
with open(os.path.join(model_dir, 'model_info.json'), encoding='utf-8') as f:
    cur_info = json.load(f)
cur_features = cur_info['features']
cur_models   = cur_info['models']
with open(os.path.join(model_dir, 'submodel', 'submodel_info.json'), encoding='utf-8') as f:
    sub_info = json.load(f)
sub_features = sub_info['features']
sub_models   = sub_info['models']

# Parquet最新特徴量
import pyarrow.parquet as pq_mod
pq_path = os.path.join(base_dir, 'data', 'processed', 'all_venues_features.parquet')
all_feats = list(set(cur_features + sub_features))
avail = set(pq_mod.read_schema(pq_path).names)
load_cols = ['馬名S', '日付', '距離'] + [c for c in avail if c in set(all_feats)]
df_pq = pd.read_parquet(pq_path, columns=load_cols)
df_pq['日付_n'] = pd.to_numeric(df_pq['日付'], errors='coerce')
pq_latest = df_pq.sort_values('日付_n').groupby('馬名S', sort=False).last().reset_index()
print(f"Parquet読込完了: {len(pq_latest)}頭分の最新特徴量")

VENUE_MAP = {'中山':'中','東京':'東','阪神':'阪','中京':'名','京都':'京',
             '函館':'函','新潟':'新','小倉':'小','札幌':'札','福島':'福'}

# モデルキャッシュ（初回読み込みのみ）
cur_model_cache = {}
sub_model_cache = {}

def get_cur_model(ck):
    if ck not in cur_model_cache and ck in cur_models:
        p = os.path.join(model_dir, cur_models[ck]['win'])
        if os.path.exists(p):
            with open(p, 'rb') as f:
                m = pickle.load(f)
            cur_model_cache[ck] = (m, m.booster_.feature_name())
    return cur_model_cache.get(ck)

def get_sub_model(sk):
    if sk not in sub_model_cache and sk in sub_models:
        p = os.path.join(model_dir, 'submodel', sub_models[sk]['win'])
        if os.path.exists(p):
            with open(p, 'rb') as f:
                m = pickle.load(f)
            sub_model_cache[sk] = (m, m.booster_.feature_name())
    return sub_model_cache.get(sk)

# ── cache.pkl インデックス作成（日付 → pkl）──────────────────
pkl_files = sorted(glob.glob(os.path.join(base_dir, 'data', 'raw', 'cache', '*.cache.pkl')))
pkl_by_date = {}
for pf in pkl_files:
    try:
        with open(pf, 'rb') as _f: _c = pickle.load(_f)
        _d = _c.get('target_date')
        if _d:
            # オッズありキャッシュを優先（ファイル名に「オッズあり」を含む）
            if _d not in pkl_by_date or 'オッズあり' in os.path.basename(pf):
                pkl_by_date[int(_d)] = pf
    except Exception as e:
        print(f"  pkl skip: {os.path.basename(pf)} ({e})")
print(f"cache.pkl: {len(pkl_by_date)}件 → dates: {sorted(pkl_by_date.keys())}")

# ── 結果CSV一覧（分析用 + 結果確認）──────────────────────────
result_csvs = sorted(glob.glob(os.path.join(base_dir, 'data', 'raw', 'cards', '出馬表形式*分析用*結果込み*.csv')))
result_csvs += sorted(glob.glob(os.path.join(base_dir, 'data', 'raw', 'results', '出馬表形式*結果*.csv')))

# 日付 → 結果CSVマップ（後から来たもので上書き＝結果確認CSV優先）
csv_by_date = {}
for rf in result_csvs:
    try:    _tmp = pd.read_csv(rf, encoding='cp932', nrows=1, low_memory=False)
    except:
        try: _tmp = pd.read_csv(rf, encoding='utf-8', nrows=1, low_memory=False)
        except: continue
    if '日付S' not in _tmp.columns: continue
    ds = str(_tmp['日付S'].iloc[0]).replace('/', '.').split('.')
    d  = (int(ds[0])-2000)*10000 + int(ds[1])*100 + int(ds[2])
    csv_by_date[d] = rf
print(f"結果確認CSV: {sorted(csv_by_date.keys())}")

all_records = []

for dnum in sorted(set(list(pkl_by_date.keys()) + list(csv_by_date.keys()))):
    # ── 結果CSV から着・配当・オッズを取得 ──
    if dnum not in csv_by_date:
        print(f"  skip(結果CSVなし): dnum={dnum}")
        continue
    rf = csv_by_date[dnum]
    fname = os.path.basename(rf)
    try:    dfr = pd.read_csv(rf, encoding='cp932', low_memory=False)
    except: dfr = pd.read_csv(rf, encoding='utf-8',  low_memory=False)

    dfr['着_num'] = dfr['着'].apply(_zen)
    dfr['_tan']   = pd.to_numeric(dfr['単勝'],    errors='coerce')
    dfr['_fuku']  = pd.to_numeric(dfr['複勝'],    errors='coerce')
    dfr['_odds']  = pd.to_numeric(dfr['単オッズ'], errors='coerce')
    dfr['場所_s'] = dfr['場所'].astype(str)
    dfr['Ｒ_s']   = dfr['Ｒ'].astype(str)
    rkey = dfr['場所_s'] + '_' + dfr['Ｒ_s']
    dfr['_race_key'] = rkey
    win_tan = dfr[dfr['着_num']==1].drop_duplicates('_race_key').set_index('_race_key')['_tan']
    win_fuku = dfr[dfr['着_num']<=3].groupby('_race_key')['_fuku'].mean()
    dfr['_tansho']  = dfr['_race_key'].map(win_tan)
    dfr['_fukusho'] = dfr['_race_key'].map(win_fuku)

    result_map = dfr.set_index('馬名S')[['着_num','_tansho','_fukusho','_odds']].to_dict('index')

    # ── pkl があれば予測スコアを流用、なければ再計算 ──
    PKL_SCORE_COLS = [
        'cur_コース偏差値', 'sub_コース偏差値',
        'cur_レース内偏差値', 'sub_レース内偏差値',
        'cur_偏差値の差', 'sub_偏差値の差',
        'cur_gap', 'sub_gap', 'combo_gap',
        'cur_prob_win', 'sub_prob_win',
        'cur_ランカー順位', 'sub_ランカー順位',
    ]
    if dnum in pkl_by_date:
        with open(pkl_by_date[dnum], 'rb') as _f: _c = pickle.load(_f)
        res = _c['result'].copy()
        if 'cur_コース偏差値' not in res.columns or 'sub_コース偏差値' not in res.columns:
            print(f"  skip(pkl列不足): dnum={dnum}")
            continue
        cur_s = pd.to_numeric(res['cur_コース偏差値'], errors='coerce')
        sub_s = pd.to_numeric(res['sub_コース偏差値'], errors='coerce')
        both  = ~(cur_s.isna() | sub_s.isna())
        pt = pd.Series(np.nan, index=res.index)
        pt[both]  = (cur_s[both] + sub_s[both]) / 2
        pt[~both] = cur_s.fillna(sub_s)[~both]

        for i, (_, row) in enumerate(res.iterrows()):
            horse = str(row.get('馬名S', ''))
            rinfo = result_map.get(horse, {})
            rec = {
                '日付_num': dnum,
                '日付_str': f"20{str(dnum)[:2]}/{str(dnum)[2:4]}/{str(dnum)[4:]}",
                '場所':     str(row.get('会場', row.get('開催', ''))),
                'Ｒ':       row.get('Ｒ'),
                '馬名S':    horse,
                '着_num':   rinfo.get('着_num', np.nan),
                '_tansho':  rinfo.get('_tansho', np.nan),
                '_fuku':    rinfo.get('_fukusho', np.nan),
                '_odds':    rinfo.get('_odds', pd.to_numeric(row.get('単勝オッズ', np.nan), errors='coerce')),
                '強さPT':   pt.iloc[i],
                'cur_cs':   cur_s.iloc[i],
                'sub_cs':   sub_s.iloc[i],
            }
            for col in PKL_SCORE_COLS:
                rec[col] = pd.to_numeric(row.get(col, np.nan), errors='coerce')
            all_records.append(rec)
        n_races = res.groupby('Ｒ').ngroups if 'Ｒ' in res.columns else '?'
        print(f"  [pkl] {fname[:20]}... dnum={dnum} {n_races}R")

    else:
        # pklなし → 分析用CSVから再計算
        dfr['会場']     = dfr['場所'].astype(str).map(VENUE_MAP).fillna(dfr['場所'].astype(str))
        dfr['_surface'] = dfr['芝ダ'].astype(str).str.strip() if '芝ダ' in dfr.columns else 'ダ'
        dfr['cur_key']  = dfr['会場'] + '_' + dfr['_surface'] + dfr['距離'].astype(str)
        dfr['_dist_band'] = dfr['距離'].apply(get_distance_band)
        _mask = (dfr['_surface']=='ダ') & (dfr['_dist_band'].isin(['中距離','長距離']))
        dfr.loc[_mask, '_dist_band'] = '中長距離'
        if 'クラス' in dfr.columns:
            _cls_map = {'新馬':1,'未勝利':2,'1勝':3,'2勝':4}
            dfr['クラス_rank'] = dfr['クラス'].apply(
                lambda v: next((r for k,r in _cls_map.items() if k in str(v)), 5))
        dfr['_cls_group'] = dfr['クラス_rank'].apply(get_class_group) if 'クラス_rank' in dfr.columns else '3勝以上'
        dfr['sub_key']   = dfr['_surface'] + '_' + dfr['_dist_band'] + '_' + dfr['_cls_group']
        dfr['性別_num']  = dfr['性別'].map({'牡':0,'牝':1,'セ':2}).astype(float)

        feat_cols = ['馬名S'] + [c for c in all_feats if c in pq_latest.columns]
        merged = dfr.merge(pq_latest[feat_cols], on='馬名S', how='left', suffixes=('','_p'))
        if '距離' in pq_latest.columns:
            merged['前距離'] = pq_latest.set_index('馬名S')['距離'].reindex(
                merged['馬名S'].values).apply(
                lambda x: float(re.search(r'\d+',str(x)).group())
                if re.search(r'\d+',str(x)) else np.nan).values
        for c in all_feats:
            if c in merged.columns:
                merged[c] = pd.to_numeric(merged[c], errors='coerce')

        merged['_tansho']  = merged['_race_key'].map(win_tan)
        merged['_fukusho'] = merged['_race_key'].map(win_fuku)

        for gk, idx in merged.groupby(['場所','Ｒ'], sort=False).groups.items():
            sub = merged.loc[idx].copy()
            ck = sub['cur_key'].iloc[0]; sk = sub['sub_key'].iloc[0]
            cur_cs = np.full(len(sub), np.nan)
            sub_cs = np.full(len(sub), np.nan)
            cm = get_cur_model(ck)
            if cm:
                m, wf = cm
                for c in wf:
                    if c not in sub.columns: sub[c] = np.nan
                prob = m.predict_proba(sub[wf])[:, 1]
                st = cur_models[ck].get('stats', {})
                wm = st.get('win_mean', prob.mean()); ws = st.get('win_std', prob.std())
                cur_cs = 50 + 10*(prob-wm)/(ws if ws>0 else 1)
            sm = get_sub_model(sk)
            if sm:
                m, wf = sm
                for c in wf:
                    if c not in sub.columns: sub[c] = np.nan
                prob = m.predict_proba(sub[wf])[:, 1]
                st = sub_models[sk].get('stats', {})
                wm = st.get('win_mean', prob.mean()); ws = st.get('win_std', prob.std())
                sub_cs = 50 + 10*(prob-wm)/(ws if ws>0 else 1)

            cur_s = pd.Series(cur_cs, index=sub.index)
            sub_s = pd.Series(sub_cs, index=sub.index)
            both  = ~(cur_s.isna() | sub_s.isna())
            pt = pd.Series(np.nan, index=sub.index)
            pt[both]  = (cur_s[both]+sub_s[both])/2
            pt[~both] = cur_s.fillna(sub_s)[~both]

            for i, (_, row) in enumerate(sub.iterrows()):
                all_records.append({
                    '日付_num': dnum,
                    '日付_str': f"20{str(dnum)[:2]}/{str(dnum)[2:4]}/{str(dnum)[4:]}",
                    '場所':     str(row.get('場所','')),
                    'Ｒ':       row.get('Ｒ'),
                    '馬名S':    row.get('馬名S',''),
                    '着_num':   row['着_num'],
                    '_tansho':  row['_tansho'],
                    '_fuku':    row.get('_fukusho', row.get('_fuku', np.nan)),
                    '_odds':    row['_odds'],
                    '強さPT':   pt.iloc[i],
                    'cur_cs':   cur_s.iloc[i],
                    'sub_cs':   sub_s.iloc[i],
                })

        n_races = merged.groupby(['場所','Ｒ']).ngroups
        print(f"  [csv] {fname[:20]}... dnum={dnum} {n_races}R")

df = pd.DataFrame(all_records)
df = df.dropna(subset=['着_num', '強さPT', '_tansho'])
print(f"\n総データ: {len(df)}頭 / {df['日付_num'].nunique()}日 / 勝ち馬あり")
print(f"強さPT 分布: min={df['強さPT'].min():.1f} / mean={df['強さPT'].mean():.1f} / max={df['強さPT'].max():.1f}")
print()

# ── レース内平均との差を計算 ──
race_key = df['日付_num'].astype(str) + '_' + df['場所'].astype(str) + '_' + df['Ｒ'].astype(str)
df['_race_key2'] = race_key
race_avg_pt = df.groupby('_race_key2')['強さPT'].transform('mean')
race_max_pt = df.groupby('_race_key2')['強さPT'].transform('max')
df['PT_vs_avg'] = df['強さPT'] - race_avg_pt
df['PT_vs_max'] = df['強さPT'] - race_max_pt  # 0=レース最高PT馬

# ── PT閾値別ROI分析 ──
print("=" * 65)
print(f"{'PT閾値':>8}  {'頭数':>6}  {'勝率':>6}  {'平均オッズ':>8}  {'単勝ROI':>8}  {'損益':>8}")
print("=" * 65)

thresholds = [50, 52, 54, 55, 56, 57, 58, 59, 60, 62, 65, 68, 70]
BET = 100

results_by_thresh = {}
for t in thresholds:
    mask  = df['強さPT'] >= t
    sub2  = df[mask]
    if len(sub2) == 0:
        continue
    n     = len(sub2)
    wins  = int(sub2['着_num'].eq(1).sum())
    wr    = wins / n
    avg_o = sub2['_odds'].mean()
    tb    = n * BET
    ret   = sub2[sub2['着_num'] == 1]['_tansho'].sum() * BET / 100
    roi   = ret / tb - 1.0
    pf    = int(ret - tb)
    results_by_thresh[t] = {'n': n, 'wins': wins, 'wr': wr, 'avg_odds': avg_o, 'roi': roi, 'pf': pf}
    marker = ' ◀' if roi > 0 else ''
    print(f"  {t}以上:  {n:5d}頭  {wr:5.1%}  {avg_o:7.1f}倍  {roi:+7.1%}  {pf:+8,}円{marker}")

print()

# ── 日別×PT閾値 ──
print("=" * 55)
print("日別収支（PT58以上 / PT60以上 / PT62以上）")
print("=" * 55)
print(f"{'日付':>12}  {'PT58':>12}  {'PT60':>12}  {'PT62':>12}")
for dnum, grp in df.groupby('日付_num'):
    date_str = f"20{str(int(dnum))[:2]}/{str(int(dnum))[2:4]}/{str(int(dnum))[4:]}"
    row_parts = [f"{date_str:>12}"]
    for t in [58, 60, 62]:
        sub3 = grp[grp['強さPT'] >= t]
        if len(sub3) == 0:
            row_parts.append(f"{'  -':>12}")
            continue
        tb  = len(sub3) * BET
        ret = sub3[sub3['着_num'] == 1]['_tansho'].sum() * BET / 100
        pf  = int(ret - tb)
        roi = ret / tb - 1.0
        s   = f"{pf:+,}円({roi:+.0%})"
        row_parts.append(f"{s:>12}")
    print("  ".join(row_parts))

print()

# ── 複勝も見る ──
df2 = pd.DataFrame(all_records)
df2 = df2.dropna(subset=['着_num', '強さPT', '_fuku'])
print("=" * 65)
print("複勝ROI（PT閾値別）")
print(f"{'PT閾値':>8}  {'頭数':>6}  {'複勝率':>6}  {'単勝ROI':>8}  {'複勝ROI':>8}")
print("=" * 65)
for t in [55, 57, 58, 59, 60, 62, 65]:
    mask  = df2['強さPT'] >= t
    sub4  = df2[mask]
    if len(sub4) == 0: continue
    n     = len(sub4)
    fuku_hits = int((sub4['着_num'] <= 3).sum())
    fuku_rate = fuku_hits / n
    # 単勝
    sub4b = df.copy(); sub4b = sub4b[sub4b['強さPT'] >= t]
    if len(sub4b) > 0:
        ret_t = sub4b[sub4b['着_num'] == 1]['_tansho'].sum() * BET / 100
        roi_t = ret_t / (len(sub4b) * BET) - 1.0
    else:
        roi_t = np.nan
    # 複勝
    ret_f = sub4[sub4['着_num'] <= 3]['_fuku'].sum() * BET / 100
    roi_f = ret_f / (n * BET) - 1.0
    marker = ' ◀' if roi_f > 0 else ''
    print(f"  {t}以上:  {n:5d}頭  {fuku_rate:5.1%}  {roi_t:+7.1%}  {roi_f:+7.1%}{marker}")

print()

# ── レース内平均との差別ROI ──
df2b = pd.DataFrame(all_records)
df2b = df2b.dropna(subset=['着_num', '強さPT'])
rk2 = df2b['日付_num'].astype(str) + '_' + df2b['場所'].astype(str) + '_' + df2b['Ｒ'].astype(str)
df2b['_rk'] = rk2
df2b['PT_vs_avg'] = df2b['強さPT'] - df2b.groupby('_rk')['強さPT'].transform('mean')
df2b['PT_vs_max'] = df2b['強さPT'] - df2b.groupby('_rk')['強さPT'].transform('max')

df_tan  = df2b.dropna(subset=['_tansho'])
df_fuku = df2b.dropna(subset=['_fuku'])

def _roi_row(st, sf, label):
    if len(st) == 0: return
    n     = len(st)
    wr    = st['着_num'].eq(1).mean()
    avg_o = st['_odds'].mean()
    ret_t = st[st['着_num']==1]['_tansho'].sum() * BET / 100
    roi_t = ret_t / (n * BET) - 1.0
    if len(sf) > 0:
        fuku_rate = (sf['着_num'] <= 3).mean()
        ret_f = sf[sf['着_num']<=3]['_fuku'].sum() * BET / 100
        roi_f = ret_f / (len(sf) * BET) - 1.0
        fuku_s = f"{fuku_rate:5.1%}  {roi_f:+7.1%}"
    else:
        fuku_s = f"{'  -':>6}  {'  -':>7}"
    mk_t = ' ◀' if roi_t > 0 else ''
    print(f"  {label}  {n:5d}頭  {wr:5.1%}  {avg_o:7.1f}倍  {roi_t:+7.1%}{mk_t}  {fuku_s}")

hdr = f"{'差の閾値':>12}  {'頭数':>6}  {'勝率':>6}  {'平均オッズ':>8}  {'単勝ROI':>8}    {'複勝率':>6}  {'複勝ROI':>8}"

print("=" * 75)
print("【レース内平均PTとの差】別 ROI")
print(hdr)
print("=" * 75)
for t in [0, 2, 4, 5, 6, 7, 8, 10, 12, 15]:
    mask_t = df_tan['PT_vs_avg'] >= t
    mask_f = df_fuku['PT_vs_avg'] >= t
    _roi_row(df_tan[mask_t], df_fuku[mask_f], f"avg+{t:2d}以上:")

print()
print("=" * 75)
print("【レース内最高PTとの差】別 ROI（0=そのレースの最高スコア馬）")
print(hdr)
print("=" * 75)
for t in [0, -1, -2, -3, -5, -8, -10]:
    mask_t = df_tan['PT_vs_max'] >= t
    mask_f = df_fuku['PT_vs_max'] >= t
    label = f"max{t:+d}以上:" if t != 0 else "max±0  :"
    _roi_row(df_tan[mask_t], df_fuku[mask_f], label)

print()
print("=" * 75)
print("【平均差 × 絶対PT】クロス集計（単勝ROI）")
print(f"{'':>14}", end='')
for pt_t in [50, 52, 55, 58, 60]:
    print(f"  PT{pt_t}以上", end='')
print()
for d in [0, 2, 4, 6, 8, 10]:
    print(f"  avg+{d:2d}以上:", end='')
    for pt_t in [50, 52, 55, 58, 60]:
        sub = df_tan[(df_tan['PT_vs_avg'] >= d) & (df_tan['強さPT'] >= pt_t)]
        if len(sub) < 10:
            print(f"  {'  -':>8}", end='')
        else:
            ret = sub[sub['着_num']==1]['_tansho'].sum() * BET / 100
            roi = ret / (len(sub) * BET) - 1.0
            print(f"  {roi:+7.1%}", end='')
    print()

print()

# ── 全指標「レース内1位選択」比較表 ────────────────────────
df_all = pd.DataFrame(all_records)
df_all = df_all.dropna(subset=['着_num'])

# レースキー（日付+場所+R）
df_all['_rk'] = (df_all['日付_num'].astype(str) + '_'
                 + df_all['場所'].astype(str) + '_'
                 + df_all['Ｒ'].astype(str))

# 高いほど良い指標（max選択）と低いほど良い指標（min選択）
METRICS_HIGH = [
    ('強さPT',          '強さPT'),
    ('cur_コース偏差値', 'cur_cs'),
    ('sub_コース偏差値', 'sub_cs'),
    ('cur_レース内偏差値','cur_レース内偏差値'),
    ('sub_レース内偏差値','sub_レース内偏差値'),
    ('cur_偏差値の差',   'cur_偏差値の差'),
    ('sub_偏差値の差',   'sub_偏差値の差'),
    ('cur_prob_win',    'cur_prob_win'),
    ('sub_prob_win',    'sub_prob_win'),
    ('cur_gap',         'cur_gap'),
    ('sub_gap',         'sub_gap'),
    ('combo_gap',       'combo_gap'),
]
METRICS_LOW = [
    ('cur_ランカー順位', 'cur_ランカー順位'),
    ('sub_ランカー順位', 'sub_ランカー順位'),
]

def pick_top_per_race(df, col, ascending=False):
    """各レースで指標が最良の1頭を返す（同値の場合は先頭1頭）"""
    s = pd.to_numeric(df[col], errors='coerce')
    df2 = df.assign(_v=s).dropna(subset=['_v'])
    idx = (df2.sort_values('_v', ascending=ascending)
               .drop_duplicates('_rk', keep='first')
               .index)
    return df2.loc[idx]

def stats(sel, df_full):
    """選択された馬の統計を返す"""
    n      = len(sel)
    n_race = sel['_rk'].nunique()
    wr     = sel['着_num'].eq(1).mean()
    pr     = (sel['着_num'] <= 3).mean()
    avg_o  = sel['_odds'].mean() if '_odds' in sel.columns else np.nan
    # 単勝ROI
    sel_t  = sel.dropna(subset=['_tansho'])
    roi_t  = (sel_t[sel_t['着_num']==1]['_tansho'].sum() * BET / 100
              / (len(sel_t) * BET) - 1.0) if len(sel_t) > 0 else np.nan
    # 複勝ROI
    sel_f  = sel.dropna(subset=['_fuku'])
    roi_f  = (sel_f[sel_f['着_num']<=3]['_fuku'].sum() * BET / 100
              / (len(sel_f) * BET) - 1.0) if len(sel_f) > 0 else np.nan
    # 平均着順
    avg_rank = sel['着_num'].mean()
    return n, n_race, wr, pr, avg_o, roi_t, roi_f, avg_rank

print("=" * 90)
print("■ 全指標「レース内1位選択」比較表（各指標でレース内最高スコアの馬を1頭選択）")
print("=" * 90)
print(f"  {'指標':<20}  {'件数':>5}  {'勝率':>6}  {'複勝率':>6}  {'平均着順':>6}  "
      f"{'平均オッズ':>8}  {'単勝ROI':>8}  {'複勝ROI':>8}")
print("-" * 90)

for label, col in METRICS_HIGH:
    if col not in df_all.columns:
        print(f"  {label:<20}  (列なし)")
        continue
    sel = pick_top_per_race(df_all, col, ascending=False)
    n, nr, wr, pr, avg_o, roi_t, roi_f, avg_rank = stats(sel, df_all)
    mk = ' ◀' if (roi_t is not None and roi_t > 0) else ''
    print(f"  {label:<20}  {n:5d}  {wr:6.1%}  {pr:6.1%}  {avg_rank:6.2f}  "
          f"{avg_o:8.1f}倍  {roi_t:+7.1%}{mk}  {roi_f:+7.1%}")

for label, col in METRICS_LOW:
    if col not in df_all.columns:
        print(f"  {label:<20}  (列なし)")
        continue
    sel = pick_top_per_race(df_all, col, ascending=True)
    n, nr, wr, pr, avg_o, roi_t, roi_f, avg_rank = stats(sel, df_all)
    mk = ' ◀' if (roi_t is not None and roi_t > 0) else ''
    print(f"  {label:<20}  {n:5d}  {wr:6.1%}  {pr:6.1%}  {avg_rank:6.2f}  "
          f"{avg_o:8.1f}倍  {roi_t:+7.1%}{mk}  {roi_f:+7.1%}")

# ベースライン：人気順1位（単オッズ最低）
sel_pop = pick_top_per_race(df_all.dropna(subset=['_odds']), '_odds', ascending=True)
n, nr, wr, pr, avg_o, roi_t, roi_f, avg_rank = stats(sel_pop, df_all)
print("-" * 90)
print(f"  {'[参考] 1番人気':<20}  {n:5d}  {wr:6.1%}  {pr:6.1%}  {avg_rank:6.2f}  "
      f"{avg_o:8.1f}倍  {roi_t:+7.1%}  {roi_f:+7.1%}")

print()

# ── 週次レポート「印」ロジック再現 ROI ────────────────────────
# gen_weekly_report.py の assign_marks() と同じ条件
# cur_ランカー順位, sub_ランカー順位, cur_偏差値の差, sub_偏差値の差, _odds を使う
dm = df_all.copy()
dm['cur_r']    = pd.to_numeric(dm.get('cur_ランカー順位'), errors='coerce')
dm['sub_r']    = pd.to_numeric(dm.get('sub_ランカー順位'), errors='coerce')
dm['cur_diff'] = pd.to_numeric(dm.get('cur_偏差値の差'),   errors='coerce')
dm['sub_diff'] = pd.to_numeric(dm.get('sub_偏差値の差'),   errors='coerce')
dm['odds']     = pd.to_numeric(dm['_odds'], errors='coerce')

both_r1  = (dm['cur_r'] == 1) & (dm['sub_r'] == 1)
star     = (dm['cur_r'] <= 3) & (dm['sub_r'] <= 3) & ~both_r1
odds_ok3 = dm['odds'].isna() | (dm['odds'] >= 3)
odds_ok5 = dm['odds'].isna() | (dm['odds'] >= 5)

dm['_印'] = ''
dm.loc[star & ~((dm['cur_r']<=2)&(dm['sub_r']<=2)) & odds_ok5 & (dm['sub_diff']>=10),       '_印'] = '☆'
dm.loc[(dm['cur_r']<=2) & (dm['sub_r']<=2) & ~both_r1 & (dm['sub_diff']>=10) & odds_ok5,    '_印'] = '▲'
dm.loc[both_r1 & (dm['sub_diff']>=10) & odds_ok3 & ~(both_r1 & (dm['cur_diff']>=10) & (dm['sub_diff']>=10) & odds_ok5), '_印'] = '〇'
dm.loc[both_r1 & (dm['cur_diff']>=10) & (dm['sub_diff']>=10) & odds_ok5,                    '_印'] = '激熱'

def mark_roi(mask, label):
    st = dm[mask].dropna(subset=['_tansho'])
    sf = dm[mask].dropna(subset=['_fuku'])
    n  = mask.sum()
    if n == 0:
        print(f"  {label:<8}  該当なし")
        return
    wr    = dm[mask]['着_num'].eq(1).mean()
    pr    = (dm[mask]['着_num'] <= 3).mean()
    avg_o = dm[mask]['odds'].mean()
    avg_r = dm[mask]['着_num'].mean()
    roi_t = ((st[st['着_num']==1]['_tansho'].sum() * BET / 100) / (len(st)*BET) - 1.0) if len(st) else np.nan
    roi_f = ((sf[sf['着_num']<=3]['_fuku'].sum()   * BET / 100) / (len(sf)*BET) - 1.0) if len(sf) else np.nan
    mk = ' ◀' if (roi_t and roi_t > 0) else ''
    print(f"  {label:<8}  {n:5d}頭  {wr:6.1%}  {pr:6.1%}  {avg_r:5.2f}着  "
          f"{avg_o:7.1f}倍  {roi_t:+7.1%}{mk}  {roi_f:+7.1%}")

print("=" * 85)
print("■ 週次レポート 印別 ROI（gen_weekly_report.py と同一ロジック）")
print(f"  {'印':<8}  {'頭数':>5}  {'勝率':>6}  {'複勝率':>6}  {'平均着順':>6}  "
      f"{'平均オッズ':>7}  {'単勝ROI':>8}  {'複勝ROI':>8}")
print("-" * 85)
mark_roi(dm['_印'] == '激熱', '激熱')
mark_roi(dm['_印'] == '〇',   '〇')
mark_roi(dm['_印'] == '▲',   '▲')
mark_roi(dm['_印'] == '☆',   '☆')
any_mark = dm['_印'].isin(['激熱','〇','▲','☆'])
print("-" * 85)
mark_roi(any_mark,             '印あり計')
print("-" * 85)
# 参考：1番人気
pop = dm['odds'] == dm.groupby('_rk')['odds'].transform('min')
mark_roi(pop, '[1番人気]')
print()
print(f"  印あり総レース数: {dm[any_mark]['_rk'].nunique()} / {dm['_rk'].nunique()} R")
