# coding: utf-8
"""新指標候補をブルートフォースで探索"""
import sys, io, re, pickle, glob
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
import pandas as pd, numpy as np
from scipy.stats import spearmanr

pkl_files   = sorted(glob.glob('data/raw/cache/*.cache.pkl'))
result_csvs = sorted(glob.glob('data/raw/results/*.csv'))

VMAP = {'中山':'中','東京':'東','阪神':'阪','中京':'名','京都':'京',
        '函館':'函','新潟':'新','小倉':'小','札幌':'札','福島':'福'}

csv_by_date = {}
for rf in result_csvs:
    try:    tmp = pd.read_csv(rf, encoding='cp932', nrows=1, low_memory=False)
    except:
        try: tmp = pd.read_csv(rf, encoding='utf-8', nrows=1, low_memory=False)
        except: continue
    datecols = [c for c in tmp.columns if '日付' in c and 'S' in c]
    if not datecols: continue
    ds = str(tmp[datecols[0]].iloc[0]).replace('/', '.')
    parts = ds.split('.')
    try: d = (int(parts[0]) - 2000) * 10000 + int(parts[1]) * 100 + int(parts[2])
    except: continue
    csv_by_date[d] = rf

def zen(s):
    if pd.isna(s): return np.nan
    s = str(s).strip().translate(str.maketrans('０１２３４５６７８９', '0123456789'))
    m = re.search(r'\d+', s)
    return int(m.group()) if m else np.nan

records = []
for pf in pkl_files:
    with open(pf, 'rb') as f: c = pickle.load(f)
    dnum = c.get('target_date')
    if not dnum or int(dnum) not in csv_by_date: continue
    dnum = int(dnum)

    rf = csv_by_date[dnum]
    try:    dfr = pd.read_csv(rf, encoding='cp932', low_memory=False)
    except: dfr = pd.read_csv(rf, encoding='utf-8',  low_memory=False)
    dfr['着_num'] = dfr['着'].apply(zen)
    dfr['_tan']   = pd.to_numeric(dfr['単勝'], errors='coerce')
    dfr['_fuku']  = pd.to_numeric(dfr['複勝'], errors='coerce')
    result_map = dfr.dropna(subset=['着_num']).set_index('馬名S')['着_num'].to_dict()
    tan_map    = dfr.dropna(subset=['着_num']).set_index('馬名S')['_tan'].to_dict()
    fuku_map   = dfr.set_index('馬名S')['_fuku'].to_dict()
    win_tan    = (dfr[dfr['着_num']==1].drop_duplicates(['場所','Ｒ'])
                  .assign(_v=lambda d: VMAP.get(str(d['場所'].iloc[0]),str(d['場所'].iloc[0])))
                  if False else {})

    res = c['result']
    def gs(col): return pd.to_numeric(pd.Series(res[col].tolist() if col in res.columns else [np.nan]*len(res)), errors='coerce')

    cur_cs  = gs('cur_コース偏差値')
    sub_cs  = gs('sub_コース偏差値')
    cur_ri  = gs('cur_レース内偏差値')
    sub_ri  = gs('sub_レース内偏差値')
    cur_r   = gs('cur_ランカー順位')
    sub_r   = gs('sub_ランカー順位')
    cur_p   = gs('cur_prob_win')
    sub_p   = gs('sub_prob_win')
    odds    = gs('単勝オッズ')

    both    = ~(cur_cs.isna() | sub_cs.isna())
    pt      = pd.Series(np.nan, index=cur_cs.index)
    pt[both]  = (cur_cs[both] + sub_cs[both]) / 2
    pt[~both] = cur_cs.fillna(sub_cs)[~both]

    venue = pd.Series((res['会場'] if '会場' in res.columns else res['開催']).tolist())
    rnum  = pd.Series(res['Ｒ'].tolist())
    uma   = pd.Series(res['馬名S'].tolist())

    for i in range(len(res)):
        horse = str(uma.iloc[i])
        atch  = result_map.get(horse)
        if atch is None or pd.isna(atch): continue
        rk = f"{dnum}_{venue.iloc[i]}_{rnum.iloc[i]}"

        # 市場の勝率（オッズの逆数）
        o = odds.iloc[i]
        market_p = (1.0 / o) if (pd.notna(o) and o > 0) else np.nan

        records.append({
            '_rk':    rk,
            '着_num': float(atch),
            '_tan':   tan_map.get(horse, np.nan),
            '_fuku':  fuku_map.get(horse, np.nan),
            # 基本指標
            'cur_cs':  cur_cs.iloc[i],
            'sub_cs':  sub_cs.iloc[i],
            'cur_ri':  cur_ri.iloc[i],
            'sub_ri':  sub_ri.iloc[i],
            'cur_r':   cur_r.iloc[i],
            'sub_r':   sub_r.iloc[i],
            'cur_p':   cur_p.iloc[i],
            'sub_p':   sub_p.iloc[i],
            'odds':    o,
            'market_p':market_p,
            'pt':      pt.iloc[i],
            # ── 新指標候補 ──
            # 1. モデルvs市場：sub_prob / 市場確率（高い=モデルが市場より高評価）
            'sub_edge':  (sub_p.iloc[i] / market_p) if pd.notna(market_p) and market_p > 0 else np.nan,
            'cur_edge':  (cur_p.iloc[i] / market_p) if pd.notna(market_p) and market_p > 0 else np.nan,
            # 2. 両モデルのランカー順位を合算（小さいほど良い）
            'avg_rank':  (cur_r.iloc[i] + sub_r.iloc[i]) / 2 if pd.notna(cur_r.iloc[i]) and pd.notna(sub_r.iloc[i]) else sub_r.iloc[i],
            # 3. コース偏差値 × レース内偏差値（両方高い馬を選ぶ）
            'sub_cs_x_ri': sub_cs.iloc[i] * sub_ri.iloc[i] if pd.notna(sub_cs.iloc[i]) and pd.notna(sub_ri.iloc[i]) else np.nan,
            # 4. ランカー1位かつコース偏差値高い → sub_cs / sub_r（小さいランクに大きいcsを割る）
            'sub_cs_per_r': sub_cs.iloc[i] / sub_r.iloc[i] if pd.notna(sub_r.iloc[i]) and sub_r.iloc[i] > 0 else np.nan,
            # 5. sub_prob × オッズ（高いと穴馬だがモデルが高評価）
            'sub_p_x_odds': sub_p.iloc[i] * o if pd.notna(o) else np.nan,
            # 6. curとsubのランカー順位が一致している度合い（差が小さい=両モデルが合意）
            'rank_agree': -abs(cur_r.iloc[i] - sub_r.iloc[i]) if pd.notna(cur_r.iloc[i]) and pd.notna(sub_r.iloc[i]) else np.nan,
            # 7. レース内偏差値 × sub_edge
            'ri_x_edge': sub_ri.iloc[i] * (sub_p.iloc[i] / market_p) if pd.notna(market_p) and market_p > 0 and pd.notna(sub_ri.iloc[i]) else np.nan,
        })

# レース単位の単勝配当マップを再構築（結果CSVの1着馬からのみ取得）
tan_records = []
for dnum2, rf2 in csv_by_date.items():
    try:    dfr2 = pd.read_csv(rf2, encoding='cp932', low_memory=False)
    except: dfr2 = pd.read_csv(rf2, encoding='utf-8', low_memory=False)
    dfr2 = dfr2.copy()
    dfr2['着_num2'] = dfr2['着'].apply(zen)
    dfr2['_tan2']   = pd.to_numeric(dfr2['単勝'], errors='coerce')
    for _, row in dfr2[dfr2['着_num2'] == 1].drop_duplicates(['場所', 'Ｒ']).iterrows():
        v = VMAP.get(str(row['場所']), str(row['場所']))
        tan_records.append({'_key': f"{dnum2}_{v}_{row['Ｒ']}", '_tan': row['_tan2']})
tan_df_map = pd.DataFrame(tan_records).set_index('_key')['_tan'].to_dict()

df = pd.DataFrame(records)
df['_tan'] = df['_rk'].map(tan_df_map)  # レース単位で上書き（全馬に勝ち馬の配当を付与）
n_races = df['_rk'].nunique()
print(f"総データ: {len(df)}頭 / {n_races}レース\n")

prod_r  = (df['cur_r'] * df['sub_r']).clip(lower=0.25)
geo_r   = prod_r.pow(0.5)

# 暫定最強
df['cs_ri_sq']    = df['sub_cs'] * df['sub_ri'] / prod_r           # S: cs×ri÷積rank

# Round6: S のバリエーション
# 両モデルのcs情報を加える
df['both_ri_sq']  = (df['sub_cs'] * df['sub_ri'] + df['cur_cs'] * df['cur_ri']) / prod_r   # 両csri合計÷積
df['pt_ri_sq']    = df['pt'] * df['sub_ri'] / prod_r               # PT×ri÷積rank
df['cs_ri2_sq']   = df['sub_cs'] * (df['sub_ri'] ** 2) / prod_r   # cs×ri²÷積
df['cs2_ri_sq']   = (df['sub_cs'] ** 2) * df['sub_ri'] / prod_r   # cs²×ri÷積
df['cs_ri_sq2']   = df['sub_cs'] * df['sub_ri'] / (prod_r ** 2).clip(lower=0.0625)  # cs×ri÷積²（より厳しく）
df['cs_ri_geo']   = df['sub_cs'] * df['sub_ri'] / geo_r            # L: cs×ri÷幾何平均（比較用）

# ── オッズランクを計算してランク融合 ──
df['_odds_rank'] = df.groupby('_rk')['odds'].rank(ascending=True, method='min')   # 1番人気=1
df['_s_rank']    = df.groupby('_rk')['cs_ri_sq'].rank(ascending=False, method='min')

prod_r = (df['cur_r'] * df['sub_r']).clip(lower=0.25)

# S = sub_cs × sub_ri ÷ 積rank
df2_s = df.dropna(subset=['cs_ri_sq', '着_num', '_tan']).copy()
n_races_total = df2_s['_rk'].nunique()

print(f"1番人気ベースライン: 勝率31.3% / 複勝率65.3% / 相関0.552 / 単勝ROI-21.9% / 複勝ROI-15.3%\n")
print(f"  {'閾値S>':<10}  {'ベット数':>7}  {'カバー率':>8}  {'勝率':>7}  {'複勝率':>8}  {'単勝ROI':>8}  {'複勝ROI':>8}  {'平均オッズ':>9}")
print("  " + "-" * 80)

# 各レースでS最高値の馬を1頭だけ選ぶ（タイは最小着順優先ではなく先頭行で代表）
top1_per_race = (df2_s.sort_values('cs_ri_sq', ascending=False)
                       .drop_duplicates(subset='_rk', keep='first')
                       .copy())

def roi_row(subset):
    n_bet = len(subset)
    if n_bet == 0:
        return n_bet, np.nan, np.nan, np.nan, np.nan, np.nan
    wr  = subset['着_num'].eq(1).mean()
    pr  = (subset['着_num'] <= 3).mean()
    top_t = subset.dropna(subset=['_tan'])
    if len(top_t) > 0:
        roi_t = (top_t[top_t['着_num']==1]['_tan'].sum()/100 - len(top_t)) / len(top_t)
    else:
        roi_t = np.nan
    placed = subset[subset['着_num'] <= 3].dropna(subset=['_fuku'])
    roi_f = (placed['_fuku'].sum()/100 - n_bet) / n_bet if len(placed) > 0 else np.nan
    avg_odds = subset['odds'].mean()
    return n_bet, wr, pr, roi_t, roi_f, avg_odds

# ── 分析1: Sのみで閾値（参考）──
print(f"1番人気: 勝率31.3% / 複勝率65.3% / 単勝ROI-21.9% / 複勝ROI-15.3%\n")
print("── A: Sのみ閾値（1頭/レース）──")
print(f"  {'閾値S>':<8}  {'ベット数':>6}  {'勝率':>7}  {'複勝率':>7}  {'単勝ROI':>8}  {'複勝ROI':>8}  {'平均オッズ':>9}")
print("  " + "-" * 68)
for thr in [0, 1000, 2000, 3000, 4000]:
    sub = top1_per_race[top1_per_race['cs_ri_sq'] > thr]
    n, wr, pr, rt, rf, ao = roi_row(sub)
    if n < 10: break
    print(f"  {thr:<8}  {n:>6}  {wr:>7.1%}  {pr:>7.1%}  {rt:>+8.1%}  {rf:>+8.1%}  {ao:>9.1f}")

# ── 分析2: S閾値 × オッズ閾値の2次元表 ──
print(f"\n── B: S>X かつ オッズ>Y（1頭/レース）──")
odds_thrs = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
s_thrs    = [0, 1000, 2000, 3000]

for s_thr in s_thrs:
    base = top1_per_race[top1_per_race['cs_ri_sq'] > s_thr]
    print(f"\n  S>{s_thr} ベース({len(base)}頭)  | {'オッズ>':>8}", end="")
    print(f"  {'ベット数':>6}  {'勝率':>7}  {'複勝率':>7}  {'単勝ROI':>8}  {'複勝ROI':>8}  {'平均オッズ':>9}")
    print("  " + "-" * 75)
    for o_thr in odds_thrs:
        sub = base[base['odds'] > o_thr]
        n, wr, pr, rt, rf, ao = roi_row(sub)
        if n < 10: break
        rt_s = f"{rt:>+8.1%}" if pd.notna(rt) else "       -"
        rf_s = f"{rf:>+8.1%}" if pd.notna(rf) else "       -"
        print(f"  {'':>14}  {o_thr:>8.1f}  {n:>6}  {wr:>7.1%}  {pr:>7.1%}  {rt_s}  {rf_s}  {ao:>9.1f}")

CANDIDATES = []  # 以降の候補比較はスキップ

print(f"  {'指標':<22}  {'1位勝率':>7}  {'1位複勝率':>8}  {'着順相関':>8}  {'単勝ROI':>8}  {'複勝ROI':>8}")
print("  " + "-" * 78)

for label, col, high in CANDIDATES:
    if col is None:
        print(f"  {label}")
        continue
    df2 = df.dropna(subset=[col, '着_num']).copy()
    if len(df2) < 100: continue
    if high:
        df2['_r2'] = df2.groupby('_rk')[col].rank(ascending=False, method='min')
        corr, _   = spearmanr(-df2[col], df2['着_num'])
    else:
        df2['_r2'] = df2.groupby('_rk')[col].rank(ascending=True,  method='min')
        corr, _   = spearmanr( df2[col], df2['着_num'])

    top1  = df2[df2['_r2'] == 1]
    wr1   = top1['着_num'].eq(1).mean()
    pr1   = (top1['着_num'] <= 3).mean()

    top1_t = top1.dropna(subset=['_tan'])
    roi_t  = ((top1_t[top1_t['着_num']==1]['_tan'].sum()/100 - len(top1_t)) / len(top1_t)
               if len(top1_t) > 0 else np.nan)

    n_bet  = len(top1)
    placed = top1[top1['着_num'] <= 3].dropna(subset=['_fuku'])
    roi_f  = ((placed['_fuku'].sum()/100 - n_bet) / n_bet
               if n_bet > 0 and len(placed) > 0 else np.nan)

    rt = f"{roi_t:+7.1%}" if roi_t is not None and not np.isnan(roi_t) else "      -"
    rf2 = f"{roi_f:+7.1%}" if roi_f is not None and not np.isnan(roi_f) else "      -"
    print(f"  {label:<22}  {wr1:>7.1%}  {pr1:>8.1%}  {corr:>8.3f}  {rt:>8}  {rf2:>8}")
