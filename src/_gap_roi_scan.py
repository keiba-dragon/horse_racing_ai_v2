# coding: utf-8
"""rank1-rank2 スコア差 × 単勝/複勝ROI スキャン"""
import sys, io, os, pickle
import numpy as np
import pandas as pd
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from save_conditional_logit import add_new_features, segment_softmax, prepare
from save_lambdarank_pace import add_pace_features

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, 'data', 'processed', 'all_venues_features.parquet')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

df = pd.read_parquet(DATA_FILE)
df['日付_num']   = pd.to_numeric(df['日付'], errors='coerce')
df['着順_num']   = pd.to_numeric(df['着順_num'], errors='coerce')
df['クラス_rank'] = pd.to_numeric(df['クラス_rank'], errors='coerce')
df = df.dropna(subset=['日付_num', '着順_num'])
df = df[df['着順_num'] < 99]
df['race_id'] = (df['日付_num'].astype(int).astype(str) + '_' +
                 df['開催'].astype(str).str.strip() + '_' +
                 df['Ｒ'].astype(str).str.strip())
df = add_pace_features(df)
df = add_new_features(df)
df['surface'] = df['距離'].astype(str).str.strip().str.extract(r'^([芝ダ])')[0].fillna('不明')
df = df[df['surface'].isin(['芝', 'ダ'])].copy()
df['is_maiden'] = (df['クラス_rank'] == 2)
df['n_horses']  = df.groupby('race_id')['race_id'].transform('count')

# 複勝配当
df['fukusho_pay'] = pd.to_numeric(df['複勝配当'], errors='coerce')
df['fuku_odds']   = df['fukusho_pay'] / 100.0

with open(os.path.join(MODEL_DIR, 'final_model.pkl'), 'rb') as f:
    pkg = pickle.load(f)
artifacts     = pkg['artifacts']
FACTOR_MAIDEN = pkg['factor_maiden']
FACTOR_OTHER  = pkg['factor_other']

def score_df(target):
    t = target.sort_values('race_id').reset_index(drop=True)
    calib_arr = np.zeros(len(t))
    odds_arr  = pd.to_numeric(t['単勝オッズ'], errors='coerce').values
    mprob     = 1.0 / np.clip(odds_arr, 1.0, None)
    for surf in ['芝', 'ダ']:
        art  = artifacts[surf]
        mask = (t['surface'] == surf).values
        sub  = t[mask].sort_values('race_id').reset_index(drop=True)
        if len(sub) == 0: continue
        X, _, gs, n, *_ = prepare(
            sub, art['feat_cols'],
            scaler=art['scaler'], poly2=art['poly2'],
            inter_scaler2=art['inter_scaler2'], top_idx=art['top_idx'],
            poly3=None, inter_scaler3=None, top_idx3=None, fit=False)
        raw   = segment_softmax(X @ art['coef'], gs, n)
        calib = art['isotonic'].predict(raw)
        calib_arr[t[mask].index] = calib
    t['calib']    = calib_arr
    t['odds_num'] = odds_arr
    t['mprob']    = mprob
    factor        = np.where(t['is_maiden'], FACTOR_MAIDEN, FACTOR_OTHER)
    t['score']    = t['calib'] - factor * t['mprob']
    t['ev']       = t['calib'] - t['mprob'] * 0.80
    t['rank']     = t.groupby('race_id')['score'].rank(ascending=False, method='first')
    t['yr']       = (t['日付_num'] // 10000).astype(int)

    def fukusho_hit(row):
        n = row['n_horses']; r = row['着順_num']
        if n <= 5:  return r == 1
        if n <= 7:  return r <= 2
        return r <= 3
    t['fuku_hit'] = t.apply(fukusho_hit, axis=1).astype(int)
    return t

print('スコア計算中...')
oos = score_df(df[df['日付_num'] >= 230101])

# rank1 に gap を付与
r1  = oos[oos['rank'] == 1][['race_id','score','calib','ev','odds_num','着順_num','fuku_hit','fuku_odds','yr']].copy()
r2  = oos[oos['rank'] == 2][['race_id','score']].rename(columns={'score':'score2'})
r1  = r1.merge(r2, on='race_id', how='left')
r1['gap'] = r1['score'] - r1['score2']

W = 70

def stats(sub):
    if len(sub) < 20: return None
    won = sub['着順_num'] == 1
    tan = (sub.loc[won,'odds_num']*100).sum() / (len(sub)*100) - 1
    fh  = sub['fuku_hit'].sum()
    fp  = sub.loc[sub['fuku_hit']==1,'fuku_odds'].sum()
    fuk = fp / len(sub) - 1 if len(sub) > 0 else float('nan')
    return len(sub), won.mean(), tan, fh/len(sub), fuk

# ── gap 閾値スキャン ──────────────────────────────────────────────────────
print(f'\n{"="*W}')
print('【gap（1位-2位スコア差）閾値別 ROI — OOS 2023+, rank=1全件】')
print(f'  {"gap>":>7}  {"件数":>6}  {"単勝勝率":>8}  {"単勝ROI":>9}  {"複勝的中":>8}  {"複勝ROI":>9}')
print(f'  {"-"*W}')
gap_thrs = np.arange(0.00, 0.35, 0.02)
gap_results = []
for thr in gap_thrs:
    sub = r1[r1['gap'] >= thr]
    s = stats(sub)
    if s is None: continue
    n, wr, tr, fhr, fr = s
    gap_results.append((thr, n, wr, tr, fhr, fr))
    mk_t = '★' if tr > 0 else ''
    mk_f = '★' if fr > 0 else ''
    print(f'  {thr:>+.2f}   {n:>6}  {wr:.3f}     {tr:>+8.4f}{mk_t}  {fhr:.3f}   {fr:>+8.4f}{mk_f}')

# ── gap × EV クロス ──────────────────────────────────────────────────────
print(f'\n{"="*W}')
print('【gap × EV クロス ROI — OOS 2023+】')
print(f'  {"gap≥":>6}  {"EV>":>6}  {"件数":>6}  {"単勝ROI":>9}  {"複勝ROI":>9}')
print(f'  {"-"*W}')
for gap_thr in [0.00, 0.05, 0.10, 0.15, 0.20]:
    for ev_thr in [-99, 0.00, 0.05, 0.10]:
        sub = r1[(r1['gap'] >= gap_thr) & (r1['ev'] > ev_thr)]
        s = stats(sub)
        if s is None: continue
        n, wr, tr, fhr, fr = s
        mk_t = '★' if tr > 0 else ''
        mk_f = '★' if fr > 0 else ''
        ev_str = f'>{ev_thr:+.2f}' if ev_thr > -99 else '(全)'
        print(f'  {gap_thr:>+.2f}   {ev_str:>6}  {n:>6}  {tr:>+8.4f}{mk_t}  {fr:>+8.4f}{mk_f}')
    print()

# ── 年別内訳（best combo付近）────────────────────────────────────────────
print(f'\n{"="*W}')
print('【年別 ROI — gap≥0.10, EV>0.05 / gap≥0.20 全件 / gap≥0.20 + EV>0.05】')
print(f'  {"年":>5}  {"件数(g10,e5)":>12}  {"単ROI":>8}  {"複ROI":>8}  |  {"件数(g20)":>9}  {"単ROI":>8}  {"複ROI":>8}  |  {"件数(g20,e5)":>11}  {"単ROI":>8}  {"複ROI":>8}')
print(f'  {"-"*90}')
for yr in sorted(r1['yr'].unique()):
    s = r1[r1['yr'] == yr]
    def yr_stats(sub):
        res = stats(sub)
        if res is None: return '--', '----', '----'
        n, wr, tr, fhr, fr = res
        return n, f'{tr:+.4f}', f'{fr:+.4f}'

    n1,t1,f1 = yr_stats(s[(s['gap']>=0.10)&(s['ev']>0.05)])
    n2,t2,f2 = yr_stats(s[s['gap']>=0.20])
    n3,t3,f3 = yr_stats(s[(s['gap']>=0.20)&(s['ev']>0.05)])
    print(f'  20{int(yr):02d}   {str(n1):>12}  {t1:>8}  {f1:>8}  |  {str(n2):>9}  {t2:>8}  {f2:>8}  |  {str(n3):>11}  {t3:>8}  {f3:>8}')

# ── gap≥0.20 × 複勝オッズ帯（※複勝配当は事後なので参考のみ）──────────────
print(f'\n{"="*W}')
print('【gap≥0.20, 単勝人気帯別 ROI (参考)】')
print(f'  {"単勝人気":>10}  {"件数":>6}  {"単勝ROI":>9}  {"複勝ROI":>9}')
print(f'  {"-"*W}')
r1['pop_rank'] = oos[oos['rank']==1].set_index('race_id').reindex(r1['race_id'].values)['odds_num'].values
r1['pop_rank2'] = r1.groupby('yr')['pop_rank'].transform(lambda x: x.rank(method='first', ascending=True))
# 人気は odds から近似
r1['market_pop'] = pd.cut(r1['odds_num'],
    bins=[0,3,6,10,20,999],
    labels=['1-2番人気(〜3倍)','3-5番人気(3-6倍)','6-9番人気(6-10倍)','10-20倍','20倍超'])
g20 = r1[r1['gap'] >= 0.20]
for lbl in ['1-2番人気(〜3倍)','3-5番人気(3-6倍)','6-9番人気(6-10倍)','10-20倍','20倍超']:
    sub = g20[g20['market_pop'] == lbl]
    s = stats(sub)
    if s is None: continue
    n, wr, tr, fhr, fr = s
    mk_t = '★' if tr > 0 else ''
    mk_f = '★' if fr > 0 else ''
    print(f'  {lbl:<18} {n:>6}  {tr:>+8.4f}{mk_t}  {fr:>+8.4f}{mk_f}')
