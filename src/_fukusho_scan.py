# coding: utf-8
"""複勝ROI分析: clogitランク別・EV閾値別"""
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

print('データ読み込み中...')
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

# 複勝配当 → 複勝オッズ (100円→払い戻し)
df['fukusho_pay'] = pd.to_numeric(df['複勝配当'], errors='coerce')
df['tansho_odds'] = pd.to_numeric(df['単勝オッズ'], errors='coerce')

# 頭数
df['n_horses'] = df.groupby('race_id')['race_id'].transform('count')

# 複勝の的中条件: 8頭以上→3着以内, 6〜7頭→2着以内, 5頭以下→1着のみ
def fukusho_hit(row):
    n = row['n_horses']
    r = row['着順_num']
    if n <= 5:  return r == 1
    if n <= 7:  return r <= 2
    return r <= 3

df['fuku_hit'] = df.apply(fukusho_hit, axis=1).astype(int)
# 複勝オッズ = 払い戻し / 100
df['fuku_odds'] = df['fukusho_pay'] / 100.0

# ── モデル ────────────────────────────────────────────────────────────────
with open(os.path.join(MODEL_DIR, 'roi_model.pkl'), 'rb') as f:
    pkg = pickle.load(f)
artifacts     = pkg['artifacts']
FACTOR_MAIDEN = pkg['factor_maiden']
FACTOR_OTHER  = pkg['factor_other']

def score_df(target: pd.DataFrame) -> pd.DataFrame:
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
    t['ev_tan']   = t['calib'] - t['mprob'] * 0.80   # 単勝EV
    t['rank']     = t.groupby('race_id')['score'].rank(ascending=False, method='first')
    t['yr']       = (t['日付_num'] // 10000).astype(int)
    # 複勝EV: 的中確率推定 - 市場複勝確率 * 0.8
    # 的中確率推定: clogit はwin probなので近似で place_prob ≈ calib * C (定数倍)
    # 複勝市場確率 = 1 / fuku_odds
    t['fuku_mprob'] = 1.0 / t['fuku_odds'].clip(lower=1.0)
    # 複勝EV (単勝EVを代理指標として使用)
    return t

print('OOS スコア計算中...')
oos = score_df(df[df['日付_num'] >= 230101])

W = 60

# ── 複勝ROI計算関数 ────────────────────────────────────────────────────────
def fuku_roi(subset, label=''):
    n    = len(subset)
    hit  = subset['fuku_hit'].sum()
    pay  = subset.loc[subset['fuku_hit']==1, 'fuku_odds'].sum()
    roi  = pay / n - 1 if n > 0 else float('nan')
    hr   = hit / n if n > 0 else float('nan')
    return n, hr, roi

# ── ランク別複勝ROI ────────────────────────────────────────────────────────
print('\n' + '='*W)
print('【OOS (2023+) ランク別 複勝ROI】')
print(f"{'ランク':<8} {'件数':>6} {'的中率':>7} {'複勝ROI':>9}")
print('-'*W)
for rk in [1, 2, 3, 4, 5]:
    sub = oos[oos['rank'] == rk]
    n, hr, r = fuku_roi(sub)
    print(f"  {rk}位    {n:>6}  {hr:.3f}  {r:>+9.4f}")

# ── 単勝EV別の複勝ROI (rank=1) ────────────────────────────────────────────
print('\n' + '='*W)
print('【OOS rank=1 × 単勝EV閾値別 複勝ROI】')
print(f"{'EV>':>6} {'件数':>6} {'的中率':>7} {'複勝ROI':>9}")
print('-'*W)
thresholds = [-0.10, 0.00, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.27]
for thr in thresholds:
    sub = oos[(oos['rank'] == 1) & (oos['ev_tan'] > thr)]
    if len(sub) < 30: continue
    n, hr, r = fuku_roi(sub)
    mk = ' ★' if r > 0 else ''
    print(f"{thr:>+.2f}  {n:>6}  {hr:.3f}  {r:>+9.4f}{mk}")

# ── 複勝オッズ帯別（rank=1）─────────────────────────────────────────────
print('\n' + '='*W)
print('【OOS rank=1 × 複勝オッズ帯別】')
print(f"{'複勝オッズ':>10} {'件数':>6} {'的中率':>7} {'複勝ROI':>9}")
print('-'*W)
bins = [(1.0,1.5),(1.5,2.0),(2.0,3.0),(3.0,5.0),(5.0,10.0),(10.0,99.0)]
top1 = oos[oos['rank'] == 1].copy()
for lo, hi in bins:
    sub = top1[(top1['fuku_odds'] >= lo) & (top1['fuku_odds'] < hi)]
    if len(sub) < 20: continue
    n, hr, r = fuku_roi(sub)
    mk = ' ★' if r > 0 else ''
    print(f"  {lo:.1f}〜{hi:.1f}倍  {n:>6}  {hr:.3f}  {r:>+9.4f}{mk}")

# ── 複勝EV（実測ベース）────────────────────────────────────────────────
print('\n' + '='*W)
print('【OOS rank=1 × 複勝EV（実複勝オッズ使用）別】')
print('  EV_fuku = calib - (1/fuku_odds) * 0.80  (近似)')
top1['ev_fuku'] = top1['calib'] - top1['fuku_mprob'] * 0.80
print(f"{'EV_fuku>':>9} {'件数':>6} {'的中率':>7} {'複勝ROI':>9}")
print('-'*W)
for thr in [-0.20, -0.10, 0.00, 0.05, 0.10, 0.15, 0.20]:
    sub = top1[top1['ev_fuku'] > thr]
    if len(sub) < 30: continue
    n, hr, r = fuku_roi(sub)
    mk = ' ★' if r > 0 else ''
    print(f"  {thr:>+.2f}    {n:>6}  {hr:.3f}  {r:>+9.4f}{mk}")

# ── 年別（rank=1, EV>0.10）──────────────────────────────────────────────
print('\n' + '='*W)
print('【年別 複勝ROI — rank=1 全 / EV_tan>0.10 / EV_fuku>0.00】')
print(f"{'年':>5}  {'全 件数':>6} {'全 複勝ROI':>10}  {'EV>0.10 件数':>12} {'EV>0.10 ROI':>11}  {'EV_fuku>0 件数':>14} {'EV_fuku>0 ROI':>13}")
print('-'*90)
top1['ev_fuku'] = top1['calib'] - top1['fuku_mprob'] * 0.80
for yr in sorted(top1['yr'].unique()):
    s = top1[top1['yr'] == yr]
    n0, h0, r0 = fuku_roi(s)
    s5 = s[s['ev_tan'] > 0.10]
    n5, h5, r5 = fuku_roi(s5)
    sf = s[s['ev_fuku'] > 0.00]
    nf, hf, rf = fuku_roi(sf)
    print(f"20{int(yr):02d}   {n0:>6}  {r0:>+10.4f}   {n5:>12}  {r5:>+11.4f}   {nf:>14}  {rf:>+13.4f}")

# ── 単勝+複勝の比較まとめ ────────────────────────────────────────────────
print('\n' + '='*W)
print('【まとめ: 単勝 vs 複勝 (rank=1, OOS全体)】')
t1_all = oos[oos['rank'] == 1]
tan_won = t1_all['着順_num'] == 1
tan_roi = (t1_all.loc[tan_won,'odds_num']*100).sum() / (len(t1_all)*100) - 1
fuk_n, fuk_hr, fuk_roi_val = fuku_roi(t1_all)
print(f"  単勝 rank=1全買い: {len(t1_all)}件  勝率={tan_won.mean():.3f}  ROI={tan_roi:+.4f}")
print(f"  複勝 rank=1全買い: {fuk_n}件  的中率={fuk_hr:.3f}  ROI={fuk_roi_val:+.4f}")

t1_ev10 = oos[(oos['rank'] == 1) & (oos['ev_tan'] > 0.10)]
tf_won = t1_ev10['着順_num'] == 1
tf_roi = (t1_ev10.loc[tf_won,'odds_num']*100).sum() / (len(t1_ev10)*100) - 1
fk2_n, fk2_hr, fk2_roi = fuku_roi(t1_ev10)
print(f"  単勝 EV>0.10:     {len(t1_ev10)}件  勝率={tf_won.mean():.3f}  ROI={tf_roi:+.4f}")
print(f"  複勝 EV>0.10:     {fk2_n}件  的中率={fk2_hr:.3f}  ROI={fk2_roi:+.4f}")
