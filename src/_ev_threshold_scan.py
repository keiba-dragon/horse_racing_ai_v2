# coding: utf-8
"""EV閾値スキャン：val で決めた閾値が OOS でどうなるか"""
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

# ── データ準備 ────────────────────────────────────────────────────────────
print('データ読み込み中...')
df = pd.read_parquet(DATA_FILE)
df['日付_num'] = pd.to_numeric(df['日付'], errors='coerce')
df['着順_num'] = pd.to_numeric(df['着順_num'], errors='coerce')
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

# ── モデル読み込み ────────────────────────────────────────────────────────
with open(os.path.join(MODEL_DIR, 'roi_model.pkl'), 'rb') as f:
    pkg = pickle.load(f)
artifacts     = pkg['artifacts']
FACTOR_MAIDEN = pkg['factor_maiden']
FACTOR_OTHER  = pkg['factor_other']

# ── スコア計算（save_final_model.py と同一ロジック）─────────────────────
def score_df(target: pd.DataFrame) -> pd.DataFrame:
    # ★ race_id 順にソートしてから reset → save_final_model.py と同じ
    t = target.sort_values('race_id').reset_index(drop=True)

    calib_arr = np.zeros(len(t))
    odds_arr  = pd.to_numeric(t['単勝オッズ'], errors='coerce').values
    mprob     = 1.0 / np.clip(odds_arr, 1.0, None)

    for surf in ['芝', 'ダ']:
        art  = artifacts[surf]
        mask = (t['surface'] == surf).values
        sub  = t[mask].sort_values('race_id').reset_index(drop=True)
        if len(sub) == 0:
            continue
        X, _, gs, n, *_ = prepare(
            sub, art['feat_cols'],
            scaler=art['scaler'], poly2=art['poly2'],
            inter_scaler2=art['inter_scaler2'], top_idx=art['top_idx'],
            poly3=None, inter_scaler3=None, top_idx3=None, fit=False)
        raw   = segment_softmax(X @ art['coef'], gs, n)
        calib = art['isotonic'].predict(raw)
        # t[mask].index は race_id 順で並んでいるので直接代入可
        calib_arr[t[mask].index] = calib

    t['calib']    = calib_arr
    t['odds_num'] = odds_arr
    t['mprob']    = mprob
    factor        = np.where(t['is_maiden'], FACTOR_MAIDEN, FACTOR_OTHER)
    t['score']    = t['calib'] - factor * t['mprob']
    t['ev']       = t['calib'] - t['mprob'] * 0.80
    t['rank']     = t.groupby('race_id')['score'].rank(ascending=False, method='first')
    t['yr']       = (t['日付_num'] // 10000).astype(int)
    return t


print('val スコア計算中...')
val_s = score_df(df[(df['日付_num'] >= 210101) & (df['日付_num'] <= 221231)])
print('OOS スコア計算中...')
oos_s = score_df(df[df['日付_num'] >= 230101])

# ── ROI計算 ─────────────────────────────────────────────────────────────
def roi(df_s, ev_thr):
    top1 = df_s[(df_s['rank'] == 1) & (df_s['ev'] > ev_thr)]
    if len(top1) < 50:
        return None, None, None
    won = top1['着順_num'] == 1
    r   = (top1.loc[won, 'odds_num'] * 100).sum() / (len(top1) * 100) - 1
    return len(top1), won.mean(), r

# サニティチェック
n0, w0, r0 = roi(oos_s, -99)
print(f'\nサニティチェック(全rank=1): {n0}件  勝率={w0:.3f}  ROI={r0:+.4f}')
print(f'  ※ save_final_model.py の結果と一致するか確認\n')

# ── val 閾値スキャン ─────────────────────────────────────────────────────
print('='*58)
print('【VAL (2021-2022) — EVしきい値スキャン】')
print(f"{'EV>':>6}  {'件数':>5}  {'勝率':>6}  {'ROI':>8}")
print('-'*58)
thresholds = np.arange(-0.10, 0.31, 0.01)
val_results = []
for thr in thresholds:
    n, wr, r = roi(val_s, thr)
    if n is None:
        continue
    val_results.append((thr, n, wr, r))

best_val = max(val_results, key=lambda x: x[3])
for thr, n, wr, r in val_results:
    show = (abs(thr % 0.05) < 0.005) or r > -0.10 or thr == best_val[0]
    if show:
        mk = ' ← val最良' if thr == best_val[0] else ''
        print(f"{thr:+.2f}   {n:>5}  {wr:.3f}  {r:+.4f}{mk}")

# ── OOS 閾値スキャン ─────────────────────────────────────────────────────
print('\n' + '='*58)
print('【OOS (2023+) — EVしきい値スキャン】')
print(f"{'EV>':>6}  {'件数':>5}  {'勝率':>6}  {'ROI':>8}  {'90%CI下限':>10}")
print('-'*58)
oos_results = []
for thr in thresholds:
    n, wr, r = roi(oos_s, thr)
    if n is None:
        continue
    # ROIの90%信頼区間（ブートストラップ近似: ±1.645σ/√n）
    # 簡易: std of ROI ~ sqrt(wr*(1-wr)/n) * avg_odds
    top1 = oos_s[(oos_s['rank'] == 1) & (oos_s['ev'] > thr)]
    avg_od = top1['odds_num'].mean()
    sigma  = avg_od * np.sqrt(wr * (1 - wr) / n)
    ci_lo  = r - 1.645 * sigma
    oos_results.append((thr, n, wr, r, ci_lo))

best_oos = max(oos_results, key=lambda x: x[3])
for thr, n, wr, r, ci_lo in oos_results:
    show = (abs(thr % 0.05) < 0.005) or r > -0.05 or thr == best_oos[0]
    if show:
        mk = ' ★' if r > 0 else (' ← OOS最良(リーク注意)' if thr == best_oos[0] else '')
        print(f"{thr:+.2f}   {n:>5}  {wr:.3f}  {r:+.4f}  {ci_lo:+.4f}{mk}")

# ── val最良閾値 → OOS適用 ────────────────────────────────────────────────
print('\n' + '='*58)
print(f'【val最良閾値 EV>{best_val[0]:+.2f} を OOS に適用】')
n, wr, r = roi(oos_s, best_val[0])
print(f'  OOS: {n}件  勝率={wr:.3f}  ROI={r:+.4f}')
print(f'  ※ val ROI={best_val[3]:+.4f} → OOS ROI={r:+.4f}  (劣化: {r-best_val[3]:+.4f})')

# ── OOS プラスになる閾値は存在するか ─────────────────────────────────────
print('\n' + '='*58)
plus_oos = [(thr, n, wr, r) for thr, n, wr, r, _ in oos_results if r > 0]
if plus_oos:
    print('【OOS でROI+になる閾値（リーク注意）】')
    for thr, n, wr, r in plus_oos:
        print(f'  EV>{thr:+.2f}  {n}件  勝率={wr:.3f}  ROI={r:+.4f}')
    print('  ただしOOS内で最適化した閾値は過学習。実運用では使えない。')
else:
    print('OOS でROIがプラスになる閾値は存在しない（リークなし条件下）')

# ── 年別 ROI ─────────────────────────────────────────────────────────────
print('\n' + '='*58)
print('【年別 OOS ROI — フィルタなし / EV>0.05 / EV>0.10】')
print(f"{'年':>6}  {'件数(全)':>7}  {'ROI(全)':>8}  {'件数(>0.05)':>10}  {'ROI(>0.05)':>10}  {'件数(>0.10)':>10}  {'ROI(>0.10)':>10}")
print('-'*80)
for yr in sorted(oos_s['yr'].unique()):
    s = oos_s[oos_s['yr'] == yr]
    def yr_roi(ev_thr):
        t = s[(s['rank'] == 1) & (s['ev'] > ev_thr)]
        if len(t) == 0: return 0, float('nan')
        w = t['着順_num'] == 1
        return len(t), (t.loc[w,'odds_num']*100).sum()/(len(t)*100)-1
    n0, r0 = yr_roi(-99)
    n5, r5 = yr_roi(0.05)
    n10,r10= yr_roi(0.10)
    print(f"20{int(yr):02d}   {n0:>7}  {r0:>+8.4f}  {n5:>10}  {r5:>+10.4f}  {n10:>10}  {r10:>+10.4f}")
