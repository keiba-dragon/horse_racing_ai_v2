# -*- coding: utf-8 -*-
"""
正ROI探索 v2 - 単勝/複勝 確率閾値ベース (カンニングなし)
train≤2018 / val 2019-2020 / OOS 2021+
"""
import sys, io, re, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression

df = pd.read_parquet('C:/horse_racing_ai/data/processed/all_venues_features.parquet')

def to_chak(s):
    s = str(s).strip().translate(str.maketrans('012345678901234567890123456789', '0123456789' * 3))
    s = s.translate(str.maketrans('０１２３４５６７８９',
                                   '0123456789'))
    m = re.match(r'^(\d+)', re.sub(r'[^0-9]', '', s))
    return int(m.group(1)) if m else None

df['着順_num'] = df['着順'].apply(to_chak).astype('float')
df = df[df['着順_num'].notna()].copy()
df['target_win']   = (df['着順_num'] == 1).astype(int)
df['target_place'] = (df['着順_num'] <= 3).astype(int)
df['日付_num'] = pd.to_numeric(df['日付'], errors='coerce').astype(int)
for c in ['単勝オッズ', '複勝配当']:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# 複勝倍率: 入着馬のみ有効 (入着外はNaN = ¥100全損)
df['複勝倍率'] = df['複勝配当'] / 100.0

def dist_band(s):
    m = re.search(r'(\d{3,4})', str(s))
    if not m: return None
    v = int(m.group(1))
    for lo, hi, name in [(0, 1400, '短'), (1401, 1800, 'マイ'), (1801, 2200, '中'), (2201, 9999, '長')]:
        if lo <= v <= hi: return name
    return None

df['距離帯'] = df['距離'].apply(dist_band)
df['surf']   = df['芝・ダ'].str.strip()
def gk(r):
    s, d = r['surf'], r['距離帯']
    if s not in ('芝', 'ダ') or not d: return None
    if s == 'ダ' and d in ('中', '長'): d = '中長'
    return f'{s}_{d}'
df['gk'] = df.apply(gk, axis=1)
df = df[df['gk'].notna()]
df['race_id'] = (df['日付'].astype(str) + '_' + df['開催'].astype(str)
                 + '_' + df['レース名'].astype(str))

FEATS = [
    '距離', '頭数', '馬番', '斤量', '馬体重', '馬体重変化',
    '1走前_着順_num', '2走前_着順_num', '3走前_着順_num',
    '4走前_着順_num', '5走前_着順_num',
    '1走前_タイム指数', '2走前_タイム指数', '3走前_タイム指数',
    '1走前_上り3F', '2走前_上り3F', '3走前_上り3F',
    '1走前_4角', '2走前_4角', '3走前_4角',
    '1走前_単勝オッズ', '2走前_単勝オッズ', '3走前_単勝オッズ',
    '1走前_走破タイム_sec',
    '近3走_平均着順', '近3走_勝率', '近5走_平均着順',
    '近5走_タイム指数平均', '近5走_上り3F平均', '近5走_タイム指数_std',
    '同馬場_平均着順_近5走', '同距離帯_平均着順_近5走',
    '同会場_平均着順_近5走', '同会場_複勝率_近5走',
    'タイム指数_近3走_slope', '近走_改善トレンド',
    'キャリア', 'クラス_rank',
    '騎手_r200_勝率', '騎手コース_r100_勝率',
    '馬コース_r20_勝率', '馬コース_r20_複勝率',
    '間隔', '前走コース一致',
]
avail = [c for c in FEATS if c in df.columns]
for c in avail:
    df[c] = pd.to_numeric(df[c], errors='coerce')
print(f'特徴量: {len(avail)}/{len(FEATS)}個')

def roi_tan(sub):
    s = sub.dropna(subset=['単勝オッズ'])
    if len(s) == 0: return np.nan, 0, 0
    w = s[s['target_win'] == 1]
    return w['単勝オッズ'].sum() / len(s) - 1, len(w), len(s)

def roi_fuku(sub):
    """複勝ROI: 入着馬は複勝倍率、未入着は0 (¥100ロス)"""
    n = len(sub)
    if n == 0: return np.nan, 0, 0
    wins = sub[sub['target_place'] == 1]
    total_ret = wins['複勝倍率'].dropna().sum()  # sum of multipliers for placing bets
    roi = total_ret / n - 1.0
    return roi, len(wins), n

# ── 学習 ──
print('学習中...')
all_w = []
all_p = []

for key in sorted(df['gk'].unique()):
    g   = df[df['gk'] == key].sort_values('日付_num')
    tr  = g[g['日付_num'] <= 181231]
    val = g[(g['日付_num'] > 181231) & (g['日付_num'] <= 201231)]
    te  = g[g['日付_num'] >= 210101]
    if len(tr) < 500 or len(te) < 200: continue
    feat = [c for c in avail if c in g.columns]

    def train_clf(target_col, min_pos=50):
        if tr[target_col].sum() < min_pos: return None, None
        clf = LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=31,
            min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0, class_weight='balanced',
            random_state=42, n_jobs=-1, verbose=-1)
        clf.fit(tr[feat].astype(float), tr[target_col])
        raw = clf.predict_proba(te[feat].astype(float))[:, 1]
        if len(val) >= 200:
            rv = clf.predict_proba(val[feat].astype(float))[:, 1]
            iso = IsotonicRegression(out_of_bounds='clip')
            iso.fit(rv, val[target_col].values)
            return clf, iso.predict(raw)
        return clf, raw

    _, prob_w = train_clf('target_win', 50)
    if prob_w is not None:
        te_w = te.copy()
        te_w['prob_win'] = prob_w
        te_w['ev_win']   = te_w['prob_win'] * te_w['単勝オッズ'].fillna(0)
        te_w['rank_w']   = te_w.groupby('race_id')['prob_win'].rank(ascending=False, method='min')
        te_w['gk']       = key
        all_w.append(te_w)

    _, prob_p = train_clf('target_place', 100)
    if prob_p is not None:
        te_p = te.copy()
        te_p['prob_place'] = prob_p
        te_p['rank_p']     = te_p.groupby('race_id')['prob_place'].rank(ascending=False, method='min')
        te_p['gk']         = key
        all_p.append(te_p)

    print(f'  {key}: tr={len(tr)} OOS={len(te)}')

oos_w = pd.concat(all_w, ignore_index=True)
oos_p = pd.concat(all_p, ignore_index=True)
print(f'OOS: {len(oos_w):,}行')

SEP = '=' * 65

# ── 単勝分析 ──────────────────────────────────────────────
print(f'\n{SEP}')
print(' 単勝 (win bet) OOS分析')
print(SEP)
r_rnd, _, n_rnd = roi_tan(oos_w)
print(f'  ランダム全馬: ROI={r_rnd:+.1%}  N={n_rnd:,}')
print()

print('  [確率閾値]')
for thr in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]:
    sub = oos_w[oos_w['prob_win'] >= thr]
    r, w, n = roi_tan(sub)
    if n < 50: continue
    print(f'  P≥{thr:.2f}: N={n:>7,} 的中={sub["target_win"].mean():.1%} ROI={r:+.1%}')

print()
print('  [EV閾値 全オッズ]')
for ev in [0.8, 1.0, 1.1, 1.2, 1.3, 1.5]:
    sub = oos_w[oos_w['ev_win'] >= ev]
    r, w, n = roi_tan(sub)
    if n < 50: continue
    print(f'  EV≥{ev:.1f}: N={n:>7,} 的中={sub["target_win"].mean():.1%} ROI={r:+.1%}')

print()
print('  [EV閾値 × オッズ5-30]')
base = oos_w[(oos_w['単勝オッズ'] >= 5) & (oos_w['単勝オッズ'] <= 30)]
for ev in [0.8, 1.0, 1.1, 1.2, 1.3]:
    sub = base[base['ev_win'] >= ev]
    r, w, n = roi_tan(sub)
    if n < 50: continue
    print(f'  EV≥{ev:.1f} [5-30]: N={n:>7,} 的中={sub["target_win"].mean():.1%} ROI={r:+.1%}')

print()
print('  [モデル1位]')
r1_w = oos_w[oos_w['rank_w'] == 1]
r, w, n = roi_tan(r1_w)
print(f'  1位全体: N={n:,} 的中={r1_w["target_win"].mean():.1%} ROI={r:+.1%}')
# 1位 × 確率高め
for thr in [0.10, 0.15, 0.20, 0.25]:
    sub = r1_w[r1_w['prob_win'] >= thr]
    r, w, n = roi_tan(sub)
    if n < 50: continue
    print(f'  1位 P≥{thr:.2f}: N={n:>7,} 的中={sub["target_win"].mean():.1%} ROI={r:+.1%}')

# ── 複勝分析 ──────────────────────────────────────────────
print(f'\n{SEP}')
print(' 複勝 (place bet) OOS分析')
print(SEP)
print(f'  複勝配当あり: {oos_p["複勝配当"].notna().sum():,}/{len(oos_p):,}件')
print(f'  (入着馬 {oos_p["target_place"].mean():.1%} に複勝配当あり)')
r_rnd_p, _, n_rnd_p = roi_fuku(oos_p)
print(f'  ランダム全馬: ROI={r_rnd_p:+.1%}  N={n_rnd_p:,}')
print()

print('  [確率閾値]')
for thr in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70]:
    sub = oos_p[oos_p['prob_place'] >= thr]
    r, w, n = roi_fuku(sub)
    if n < 50: continue
    print(f'  P≥{thr:.2f}: N={n:>7,} 入着={sub["target_place"].mean():.1%} ROI={r:+.1%}')

print()
print('  [モデルtop-N]')
for top_n in [1, 2, 3]:
    sub = oos_p[oos_p['rank_p'] <= top_n]
    r, w, n = roi_fuku(sub)
    print(f'  top-{top_n}: N={n:,} 入着={sub["target_place"].mean():.1%} ROI={r:+.1%}')

# ── 年別推移 ──────────────────────────────────────────────
print(f'\n{SEP}')
print(' 年別ROI推移')
print(SEP)
print(f'  {"年":>5} {"単勝1位N":>8} {"単勝1位ROI":>11} | {"複P≥0.40 N":>10} {"複勝ROI":>9}')
for yr in sorted(oos_w['日付_num'].astype(str).str[:4].unique()):
    yw = oos_w[oos_w['日付_num'].astype(str).str[:4] == yr]
    yp = oos_p[oos_p['日付_num'].astype(str).str[:4] == yr]
    r1 = yw[yw['rank_w'] == 1]
    rw, _, nw = roi_tan(r1)
    sp = yp[yp['prob_place'] >= 0.40]
    rp, _, np_ = roi_fuku(sp)
    print(f'  {yr}  {nw:>8,} {rw:>+11.1%}  | {np_:>10,} {rp:>+9.1%}')

# ── サマリー ──────────────────────────────────────────────
print(f'\n{SEP}')
print(' サマリー')
print(SEP)
avg_h = oos_w.groupby('race_id').size().mean()
print(f'  平均頭数: {avg_h:.1f}  ランダム的中率: {1/avg_h:.1%}')
print(f'  OOS期間: 2021-2026  レース数: {oos_w["race_id"].nunique():,}')
