# coding: utf-8
"""スコア・calib 別の3着以内率分析: どこで崖があるか"""
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
df['n_horses']  = df.groupby('race_id')['race_id'].transform('count')

with open(os.path.join(MODEL_DIR, 'roi_model.pkl'), 'rb') as f:
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
    t['calib']  = calib_arr
    t['odds_num'] = odds_arr
    t['mprob']  = mprob
    factor      = np.where(t['is_maiden'], FACTOR_MAIDEN, FACTOR_OTHER)
    t['score']  = t['calib'] - factor * t['mprob']
    t['ev']     = t['calib'] - t['mprob'] * 0.80
    t['rank']   = t.groupby('race_id')['score'].rank(ascending=False, method='first')
    # 3着以内フラグ（頭数考慮）
    def top3(row):
        n = row['n_horses']
        r = row['着順_num']
        if n <= 5: return r == 1
        if n <= 7: return r <= 2
        return r <= 3
    t['top3'] = t.apply(top3, axis=1).astype(int)
    t['win']  = (t['着順_num'] == 1).astype(int)
    return t

print('OOS スコア計算中...')
oos = score_df(df[df['日付_num'] >= 230101])

W = 62

# ── calib 別 3着以内率 ───────────────────────────────────────────────────
print(f'\n{"="*W}')
print('【calib（勝率推定）別 3着以内率 / 勝率】')
print(f'  {"calib範囲":>14}  {"件数":>6}  {"勝率":>6}  {"3着以内率":>9}  {"3着以内/calib":>12}')
print(f'  {"-"*W}')
edges = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 0.60, 1.01]
for lo, hi in zip(edges[:-1], edges[1:]):
    sub = oos[(oos['calib'] >= lo) & (oos['calib'] < hi)]
    if len(sub) < 30: continue
    wr  = sub['win'].mean()
    p3r = sub['top3'].mean()
    ratio = p3r / lo if lo > 0 else float('nan')
    print(f'  {lo:.2f}〜{hi:.2f}          {len(sub):>6}  {wr:.3f}  {p3r:.3f}     {ratio:>8.2f}x')

# ── score 別 3着以内率（全馬） ────────────────────────────────────────────
print(f'\n{"="*W}')
print('【score 別 3着以内率（全馬 OOS）】')
print(f'  {"score範囲":>14}  {"件数":>6}  {"勝率":>6}  {"3着以内率":>9}')
print(f'  {"-"*W}')
s_edges = [-0.30, -0.20, -0.15, -0.10, -0.08, -0.05, -0.03, -0.01, 0.00,
            0.01, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 1.00]
for lo, hi in zip(s_edges[:-1], s_edges[1:]):
    sub = oos[(oos['score'] >= lo) & (oos['score'] < hi)]
    if len(sub) < 50: continue
    wr  = sub['win'].mean()
    p3r = sub['top3'].mean()
    mk  = ' ←崖?' if p3r < 0.10 else ''
    print(f'  {lo:+.2f}〜{hi:+.2f}        {len(sub):>6}  {wr:.3f}  {p3r:.3f}{mk}')

# ── rank=1 の score 絶対値 分布 ──────────────────────────────────────────
print(f'\n{"="*W}')
print('【rank=1 の score 分布 → スコアが高いほど強いか】')
print(f'  {"score":>10}  {"件数":>6}  {"勝率":>6}  {"3着以内率":>9}')
print(f'  {"-"*W}')
top1 = oos[oos['rank'] == 1]
q_edges = [-99] + list(np.percentile(top1['score'], [10,20,30,40,50,60,70,80,90])) + [99]
for lo, hi in zip(q_edges[:-1], q_edges[1:]):
    sub = top1[(top1['score'] >= lo) & (top1['score'] < hi)]
    if len(sub) < 30: continue
    wr  = sub['win'].mean()
    p3r = sub['top3'].mean()
    print(f'  {lo:>+8.4f}〜{hi:+.4f}  {len(sub):>6}  {wr:.3f}  {p3r:.3f}')

# ── rank=1 の 1位と2位のスコア差 ────────────────────────────────────────
print(f'\n{"="*W}')
print('【rank=1 vs rank=2 のスコア差 → "抜け感"で3着以内率が変わるか】')
print(f'  {"score差":>12}  {"件数":>6}  {"勝率":>6}  {"3着以内率":>9}')
print(f'  {"-"*W}')
top2 = oos[oos['rank'].isin([1, 2])][['race_id','rank','score','win','top3']].copy()
sc1  = top2[top2['rank']==1].set_index('race_id')[['score','win','top3']]
sc2  = top2[top2['rank']==2].set_index('race_id')[['score']].rename(columns={'score':'score2'})
merged = sc1.join(sc2, how='inner')
merged['gap'] = merged['score'] - merged['score2']

gap_bins = [(-99,-0.05), (-0.05,-0.02), (-0.02,0.00), (0.00,0.02),
            (0.02,0.05), (0.05,0.10), (0.10,0.20), (0.20,99)]
for lo, hi in gap_bins:
    sub = merged[(merged['gap'] >= lo) & (merged['gap'] < hi)]
    if len(sub) < 30: continue
    wr  = sub['win'].mean()
    p3r = sub['top3'].mean()
    lbl = f'{lo:+.2f}〜{hi:+.2f}' if hi < 99 else f'{lo:+.2f}〜'
    mk  = ' ★' if wr > 0.35 else ''
    print(f'  {lbl:<14}  {len(sub):>6}  {wr:.3f}  {p3r:.3f}{mk}')

# ── calib と実際の3着以内率の整合性チェック ──────────────────────────────
print(f'\n{"="*W}')
print('【calib の calibration確認: 推定 vs 実際】')
print(f'  {"calib中央値":>12}  {"件数":>6}  {"実際の勝率":>10}  {"比率":>6}')
print(f'  {"-"*W}')
for lo, hi in zip(edges[:-1], edges[1:]):
    sub = oos[(oos['calib'] >= lo) & (oos['calib'] < hi)]
    if len(sub) < 30: continue
    mid = (lo + hi) / 2
    wr  = sub['win'].mean()
    ratio = wr / mid if mid > 0 else float('nan')
    bar = '█' * int(ratio * 10)
    print(f'  {mid:.3f}          {len(sub):>6}  {wr:.3f}        {ratio:.2f}x {bar}')
