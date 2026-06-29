# coding: utf-8
import sys, io, re, pickle, glob
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
import pandas as pd, numpy as np
from scipy.stats import spearmanr

pkl_files    = sorted(glob.glob('data/raw/cache/*.cache.pkl'))
result_csvs  = sorted(glob.glob('data/raw/results/*.csv'))

# 日付 → 結果CSVマップ
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

SCORE_COLS = [
    ('cur_コース偏差値',   True),
    ('sub_コース偏差値',   True),
    ('cur_レース内偏差値', True),
    ('sub_レース内偏差値', True),
    ('cur_偏差値の差',     True),
    ('sub_偏差値の差',     True),
    ('cur_ランカー順位',   False),
    ('sub_ランカー順位',   False),
    ('強さPT',             True),
    ('単勝オッズ',         False),
]

records = []
for pf in pkl_files:
    with open(pf, 'rb') as f:
        c = pickle.load(f)
    dnum = c.get('target_date')
    if not dnum or int(dnum) not in csv_by_date:
        continue
    dnum = int(dnum)

    rf = csv_by_date[dnum]
    try:    dfr = pd.read_csv(rf, encoding='cp932', low_memory=False)
    except: dfr = pd.read_csv(rf, encoding='utf-8',  low_memory=False)

    dfr['着_num'] = dfr['着'].apply(zen)
    dfr['_tan']   = pd.to_numeric(dfr['単勝'], errors='coerce')
    dfr['_rk']    = dfr['場所'].astype(str) + '_' + dfr['Ｒ'].astype(str)

    result_map = dfr.dropna(subset=['着_num']).set_index('馬名S')['着_num'].to_dict()
    # 勝ち馬の単勝配当をレース単位で
    win_tan = (dfr[dfr['着_num'] == 1]
               .drop_duplicates('_rk')
               .set_index('_rk')['_tan']
               .to_dict())
    # 各馬の複勝配当（その馬自身が3着以内なら値あり、それ以外はNaN）
    dfr['_fuku'] = pd.to_numeric(dfr['複勝'], errors='coerce')
    fuku_map = dfr.set_index('馬名S')['_fuku'].to_dict()

    res = c['result']
    cur_s = pd.to_numeric(pd.Series(res.get('cur_コース偏差値', [np.nan]*len(res)).tolist()), errors='coerce')
    sub_s = pd.to_numeric(pd.Series(res.get('sub_コース偏差値', [np.nan]*len(res)).tolist()), errors='coerce')
    both  = ~(cur_s.isna() | sub_s.isna())
    pt    = pd.Series(np.nan, index=cur_s.index)
    pt[both]  = (cur_s[both] + sub_s[both]) / 2
    pt[~both] = cur_s.fillna(sub_s)[~both]

    venue = pd.Series((res['会場'] if '会場' in res.columns else res['開催']).tolist())
    rnum  = pd.Series(res['Ｒ'].tolist())
    uma   = pd.Series(res['馬名S'].tolist())

    n = len(res)
    for i in range(n):
        horse  = str(uma.iloc[i])
        atch   = result_map.get(horse)
        if atch is None or pd.isna(atch):
            continue
        # レースキー：日付+会場+R で一意に
        rk     = f"{dnum}_{venue.iloc[i]}_{rnum.iloc[i]}"
        # 単勝配当：場所列はCSVの場所名で引く
        csv_rk = f"{dfr['場所'].iloc[0]}_{rnum.iloc[i]}"  # 同日同場で引く
        # より正確に：CSVのレースキーと合わせる
        tan_v  = win_tan.get(f"{dfr['場所'].mode()[0]}_{rnum.iloc[i]}", np.nan) \
                 if len(dfr) else np.nan

        rec = {
            '_rk':    rk,
            '着_num': float(atch),
            '強さPT': pt.iloc[i],
            '_fuku':  fuku_map.get(horse, np.nan),  # その馬自身の複勝配当
        }
        # 単勝オッズ（pkl の card_df か result から）
        odds_val = np.nan
        if '単勝オッズ' in res.columns:
            odds_val = pd.to_numeric(res['単勝オッズ'].iloc[i], errors='coerce')
        rec['単勝オッズ'] = odds_val

        for col, _ in SCORE_COLS:
            if col in ('強さPT', '単勝オッズ'):
                continue
            val = res[col].iloc[i] if col in res.columns else np.nan
            rec[col] = pd.to_numeric(val, errors='coerce')

        records.append(rec)

# 単勝配当を別途マージ（日付+場所+R で正確に）
# まず全結果CSVから配当を収集
VMAP = {'中山':'中','東京':'東','阪神':'阪','中京':'名','京都':'京',
        '函館':'函','新潟':'新','小倉':'小','札幌':'札','福島':'福'}

tan_records = []
for dnum, rf in csv_by_date.items():
    try:    dfr = pd.read_csv(rf, encoding='cp932', low_memory=False)
    except: dfr = pd.read_csv(rf, encoding='utf-8', low_memory=False)
    dfr['着_num'] = dfr['着'].apply(zen)
    dfr['_tan']   = pd.to_numeric(dfr['単勝'], errors='coerce')
    for _, row in dfr[dfr['着_num'] == 1].drop_duplicates(['場所','Ｒ']).iterrows():
        v = VMAP.get(str(row['場所']), str(row['場所']))
        tan_records.append({'_key': f"{dnum}_{v}_{row['Ｒ']}", '_tan': row['_tan']})

tan_df = pd.DataFrame(tan_records).set_index('_key')['_tan'].to_dict()

df = pd.DataFrame(records)
df['_tan'] = df['_rk'].map(tan_df)
# _fuku はすでに馬単位で records に入っている

n_races = df['_rk'].nunique()
print(f"総データ: {len(df)}頭 / {n_races}レース\n")
print(f"  {'指標':<16}  {'1位勝率':>7}  {'1位複勝率':>8}  {'着順相関':>8}  {'単勝ROI':>8}  {'複勝ROI':>8}  {'カバー':>6}")
print("  " + "-" * 78)

for col, high in SCORE_COLS:
    df2 = df.dropna(subset=[col, '着_num']).copy()
    if len(df2) < 100:
        continue
    if high:
        df2['_r'] = df2.groupby('_rk')[col].rank(ascending=False, method='min')
        corr, _  = spearmanr(-df2[col], df2['着_num'])
    else:
        df2['_r'] = df2.groupby('_rk')[col].rank(ascending=True,  method='min')
        corr, _  = spearmanr( df2[col], df2['着_num'])

    top1  = df2[df2['_r'] == 1]
    wr1   = top1['着_num'].eq(1).mean()
    pr1   = (top1['着_num'] <= 3).mean()
    cover = top1['_rk'].nunique() / n_races

    # 単勝ROI
    top1_t = top1.dropna(subset=['_tan'])
    if len(top1_t) > 0:
        ret_t   = top1_t[top1_t['着_num'] == 1]['_tan'].sum() / 100
        roi_t   = (ret_t - len(top1_t)) / len(top1_t)
        roi_t_s = f"{roi_t:+7.1%}"
    else:
        roi_t_s = "      -"

    # 複勝ROI：分母=全ベット数、分子=複勝に来た馬の配当合計
    n_bet_f = len(top1)
    placed  = top1[top1['着_num'] <= 3].dropna(subset=['_fuku'])
    if n_bet_f > 0 and len(placed) > 0:
        ret_f   = placed['_fuku'].sum() / 100
        roi_f   = (ret_f - n_bet_f) / n_bet_f
        roi_f_s = f"{roi_f:+7.1%}"
    else:
        roi_f_s = "      -"

    label = '1番人気' if col == '単勝オッズ' else col
    print(f"  {label:<16}  {wr1:>7.1%}  {pr1:>8.1%}  {corr:>8.3f}  {roi_t_s:>8}  {roi_f_s:>8}  {cover:>6.1%}")
