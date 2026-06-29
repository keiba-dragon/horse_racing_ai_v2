# -*- coding: utf-8 -*-
"""
正ROI探索 v3 - フル特徴量 + EV model
train≤181231(2018末) / val 190101-201231 / OOS 210101+
"""
import sys, io, re, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression

df = pd.read_parquet('C:/horse_racing_ai/data/processed/all_venues_features.parquet')
print(f'読込: {len(df):,}行 × {len(df.columns)}列')

# 着順数値化
df['着順_num'] = pd.to_numeric(df['着順_num'], errors='coerce')
df = df[df['着順_num'].notna()].copy()
df['target_win']   = (df['着順_num'] == 1).astype(int)
df['target_place'] = (df['着順_num'] <= 3).astype(int)

for c in ['単勝オッズ', '複勝配当']:
    df[c] = pd.to_numeric(df[c], errors='coerce')

df['複勝倍率'] = df['複勝配当'] / 100.0

# コースグループ
def dist_band(s):
    m = re.search(r'(\d{3,4})', str(s))
    if not m: return None
    v = int(m.group(1))
    if v <= 1400: return '短'
    if v <= 1800: return 'マイ'
    if v <= 2200: return '中'
    return '長'

df['距離帯'] = df['距離'].apply(dist_band)
df['surf']   = df['芝・ダ'].str.strip() if '芝・ダ' in df.columns else df.get('今回_surface', pd.Series('', index=df.index))

MAJOR_VENUES = {'東', '中', '阪', '京'}  # 東京/中山/阪神/京都

def gk(r):
    s, d = r['surf'], r['距離帯']
    if s not in ('芝', 'ダ') or not d: return None
    if s == 'ダ' and d in ('中', '長'): d = '中長'
    v = str(r.get('今回_会場', '')).strip()
    venue_type = '主要' if v in MAJOR_VENUES else 'ロ'
    return f'{s}_{d}_{venue_type}'

df['gk'] = df.apply(gk, axis=1)
df = df[df['gk'].notna()]

df['race_id'] = (df['日付'].astype(str) + '_' + df['開催'].astype(str)
                 + '_' + df['レース名'].astype(str))

# ── 特徴量 ──────────────────────────────────────────────────
FEATS = [
    # 基本レース条件
    '距離', '頭数', '馬番', '斤量', '馬体重', '馬体重増減',
    '内外枠', '斤量変化', '間隔', '連闘フラグ', '休み明けフラグ',
    'クラス_rank',
    # 前走系
    '1走前_着順_num', '2走前_着順_num', '3走前_着順_num',
    '4走前_着順_num', '5走前_着順_num',
    '1走前_クラス_rank', '2走前_クラス_rank', '3走前_クラス_rank',
    '1走前_クラス差', '2走前_クラス差', '3走前_クラス差',
    '1走前_タイム指数', '2走前_タイム指数', '3走前_タイム指数',
    '4走前_タイム指数', '5走前_タイム指数',
    '1走前_上り3F', '2走前_上り3F', '3走前_上り3F',
    '1走前_4角', '2走前_4角', '3走前_4角',
    '1走前_単勝オッズ', '2走前_単勝オッズ', '3走前_単勝オッズ',
    '1走前_走破タイム_sec', '2走前_走破タイム_sec',
    '1走前_PCI', '2走前_PCI', '3走前_PCI',
    '1走前_脚質_num', '2走前_脚質_num',
    '1走前_頭数', '2走前_頭数',
    '1走前_馬番', '2走前_馬番',
    # 集計系
    '近3走_平均着順', '近3走_勝率', '近3走_複勝率',
    '近5走_平均着順', '近5走_複勝率',
    '近5走_タイム指数平均', '近5走_上り3F平均', '近5走_タイム指数_std',
    '近5走_タイム指数_max', '近5走_タイム指数_min',
    '近5走_上り3F_min', '近5走_上り3F_std',
    '近5走_クラス調整_平均着順', '格上経験数_近5走',
    '近5走_着差タイム_クラス補正平均',
    '近走_改善トレンド', 'タイム指数_近3走_slope',
    '前走_追い上げ度', '前走_4角位置', '近5走_平均4角位置',
    # 指数 (過去レース分のみ: 1走前以降はOK、当レース分はNG)
    '1走前_上り3F_指数', '2走前_上り3F_指数', '3走前_上り3F_指数',
    # 脚質・展開 (推定系: 近5走_平均4角位置ベースで当レース前に算出可能)
    '脚質フィット', '展開フィット_v2',
    'コース展開マッチ', '展開_コース_脚質フィット',
    'レース内_逃げ馬数', 'レース内_先行馬数',
    'レース内_相対脚質', 'コース_先行有利度', '推定ペース',
    # 騎手・調教師
    '騎手_r200_勝率', '騎手_r200_複勝率',
    '騎手コース_r100_勝率', '騎手コース_r100_複勝率',
    '騎手馬場_r100_勝率', '騎手馬場_r100_複勝率',
    '騎手距離_r100_勝率', '騎手距離_r100_複勝率',
    '騎手脚質_r100_勝率', '騎手脚質_r100_複勝率',
    # 種牡馬・血統
    '種牡馬_勝率', '種牡馬_複勝率',
    '種牡馬_芝_勝率', '種牡馬_ダ_勝率',
    '母父馬_勝率', '母父馬_複勝率',
    '産地_勝率', '生産者_勝率',
    # 馬コース適性
    '馬_r20_勝率', '馬_r20_複勝率',
    '馬コース_r20_勝率', '馬コース_r20_複勝率',
    '馬距離_勝率', '馬距離_複勝率',
    # 条件適性
    '同会場_平均着順_近5走', '同会場_複勝率_近5走',
    '同馬場_平均着順_近5走', '同距離帯_平均着順_近5走',
    '芝ダ一致_平均着順_近5走', '良馬場_平均着順_近5走',
    '道悪_平均着順_近5走',
    # その他
    'キャリア', 'キャリア_log',
    '馬体重トレンド_近5走', '前走コース一致',
    '芝ダ転向', '距離変化_前走',
    '騎手変更', '乗替り_近走不振',
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
    n = len(sub)
    if n == 0: return np.nan, 0, 0
    wins = sub[sub['target_place'] == 1]
    total_ret = wins['複勝倍率'].dropna().sum()
    roi = total_ret / n - 1.0
    return roi, len(wins), n

# ── 学習 ──────────────────────────────────────────────────
print('学習中...')
all_w, all_p = [], []

for key in sorted(df['gk'].unique()):
    g   = df[df['gk'] == key].sort_values('日付_num')
    tr  = g[g['日付_num'] <= 181231]
    val = g[(g['日付_num'] > 181231) & (g['日付_num'] <= 201231)]
    te  = g[g['日付_num'] >= 210101]
    if len(tr) < 500 or len(te) < 200: continue
    feat = [c for c in avail if c in g.columns]

    def train_clf(target_col, min_pos=50):
        if tr[target_col].sum() < min_pos: return None
        clf = LGBMClassifier(
            n_estimators=600, learning_rate=0.03, num_leaves=63,
            min_child_samples=20, subsample=0.8, colsample_bytree=0.7,
            reg_alpha=0.1, reg_lambda=1.0, class_weight='balanced',
            random_state=42, n_jobs=-1, verbose=-1)
        clf.fit(tr[feat].astype(float), tr[target_col])
        raw = clf.predict_proba(te[feat].astype(float))[:, 1]
        if len(val) >= 200:
            rv = clf.predict_proba(val[feat].astype(float))[:, 1]
            iso = IsotonicRegression(out_of_bounds='clip')
            iso.fit(rv, val[target_col].values)
            return iso.predict(raw)
        return raw

    prob_w = train_clf('target_win', 50)
    if prob_w is not None:
        te_w = te.copy()
        te_w['prob_win'] = prob_w
        te_w['ev_win']   = te_w['prob_win'] * te_w['単勝オッズ'].fillna(0)
        te_w['rank_w']   = te_w.groupby('race_id')['prob_win'].rank(ascending=False, method='min')
        te_w['gk']       = key
        all_w.append(te_w)

    prob_p = train_clf('target_place', 100)
    if prob_p is not None:
        te_p = te.copy()
        te_p['prob_place'] = prob_p
        te_p['ev_place']   = te_p['prob_place'] * te_p['単勝オッズ'].fillna(0)
        te_p['rank_p']     = te_p.groupby('race_id')['prob_place'].rank(ascending=False, method='min')
        te_p['gk']         = key
        all_p.append(te_p)

    print(f'  {key}: tr={len(tr)} val={len(val)} OOS={len(te)}')

oos_w = pd.concat(all_w, ignore_index=True)
oos_p = pd.concat(all_p, ignore_index=True)
print(f'OOS: 単勝={len(oos_w):,}行  複勝={len(oos_p):,}行')

SEP = '=' * 70

# ── 単勝分析 ──────────────────────────────────────────────
print(f'\n{SEP}')
print(' 単勝 OOS分析 (2021-2026)')
print(SEP)
r_rnd, _, n_rnd = roi_tan(oos_w)
print(f'  ランダム全馬: ROI={r_rnd:+.1%}  N={n_rnd:,}')
print()

print('  [確率閾値]')
for thr in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]:
    sub = oos_w[oos_w['prob_win'] >= thr]
    r, w, n = roi_tan(sub)
    if n < 50: continue
    print(f'  P≥{thr:.2f}: N={n:>7,}  的中={sub["target_win"].mean():.1%}  ROI={r:+.1%}')

print()
print('  [EV閾値 (単勝オッズ全域)]')
for ev in [0.8, 1.0, 1.1, 1.2, 1.3, 1.5, 2.0]:
    sub = oos_w[oos_w['ev_win'] >= ev]
    r, w, n = roi_tan(sub)
    if n < 50: continue
    print(f'  EV≥{ev:.1f}: N={n:>7,}  的中={sub["target_win"].mean():.1%}  ROI={r:+.1%}')

print()
print('  [EV閾値 × オッズ帯]')
for odds_lo, odds_hi in [(3,10), (5,20), (5,30), (10,50), (3,50)]:
    base = oos_w[(oos_w['単勝オッズ'] >= odds_lo) & (oos_w['単勝オッズ'] <= odds_hi)]
    for ev in [1.0, 1.2, 1.5]:
        sub = base[base['ev_win'] >= ev]
        r, w, n = roi_tan(sub)
        if n < 100: continue
        print(f'  EV≥{ev:.1f} [{odds_lo}-{odds_hi}倍]: N={n:>7,}  的中={sub["target_win"].mean():.1%}  ROI={r:+.1%}')

print()
print('  [モデル1位]')
r1_w = oos_w[oos_w['rank_w'] == 1]
r, _, n = roi_tan(r1_w)
print(f'  1位全体: N={n:,}  的中={r1_w["target_win"].mean():.1%}  ROI={r:+.1%}')
for thr in [0.10, 0.15, 0.20, 0.25, 0.30]:
    sub = r1_w[r1_w['prob_win'] >= thr]
    r, _, n = roi_tan(sub)
    if n < 50: continue
    print(f'  1位 P≥{thr:.2f}: N={n:>7,}  的中={sub["target_win"].mean():.1%}  ROI={r:+.1%}')

print()
print('  [EV≥1.2 × モデル上位]')
for rank_th in [1, 2, 3]:
    sub = oos_w[(oos_w['ev_win'] >= 1.2) & (oos_w['rank_w'] <= rank_th)]
    r, _, n = roi_tan(sub)
    if n < 50: continue
    print(f'  EV≥1.2 rank≤{rank_th}: N={n:>7,}  的中={sub["target_win"].mean():.1%}  ROI={r:+.1%}')

# ── 複勝分析 ──────────────────────────────────────────────
print(f'\n{SEP}')
print(' 複勝 OOS分析 (2021-2026)')
print(SEP)
r_rnd_p, _, n_rnd_p = roi_fuku(oos_p)
print(f'  ランダム全馬: ROI={r_rnd_p:+.1%}  N={n_rnd_p:,}')
print()

print('  [確率閾値]')
for thr in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80]:
    sub = oos_p[oos_p['prob_place'] >= thr]
    r, w, n = roi_fuku(sub)
    if n < 50: continue
    print(f'  P≥{thr:.2f}: N={n:>7,}  入着={sub["target_place"].mean():.1%}  ROI={r:+.1%}')

print()
print('  [モデルtop-N]')
for top_n in [1, 2, 3]:
    sub = oos_p[oos_p['rank_p'] <= top_n]
    r, w, n = roi_fuku(sub)
    print(f'  top-{top_n}: N={n:,}  入着={sub["target_place"].mean():.1%}  ROI={r:+.1%}')

print()
print('  [複勝: top-N × 単勝オッズ帯]')
for top_n in [1, 2]:
    for odds_lo, odds_hi in [(3,10), (5,20), (3,20)]:
        sub = oos_p[(oos_p['rank_p'] <= top_n)
                    & (oos_p['単勝オッズ'] >= odds_lo)
                    & (oos_p['単勝オッズ'] <= odds_hi)]
        r, w, n = roi_fuku(sub)
        if n < 100: continue
        print(f'  top-{top_n} [{odds_lo}-{odds_hi}倍]: N={n:>7,}  入着={sub["target_place"].mean():.1%}  ROI={r:+.1%}')

# ── gk別分析 ──────────────────────────────────────────────
print(f'\n{SEP}')
print(' コース種別 × EV1.2以上 ROI')
print(SEP)
print(f'  {"gk":12s}  {"単勝N":>8}  {"単勝ROI":>8}  {"複勝top2 N":>10}  {"複勝ROI":>8}')
for key in sorted(oos_w['gk'].unique()):
    gw = oos_w[(oos_w['gk'] == key) & (oos_w['ev_win'] >= 1.2)]
    gp = oos_p[(oos_p['gk'] == key) & (oos_p['rank_p'] <= 2)]
    rw, _, nw = roi_tan(gw)
    rp, _, np_ = roi_fuku(gp)
    if nw < 50 and np_ < 50: continue
    print(f'  {key:12s}  {nw:>8,}  {rw:>+8.1%}  {np_:>10,}  {rp:>+8.1%}')

# ── 年別推移 ──────────────────────────────────────────────
print(f'\n{SEP}')
print(' 年別ROI推移 (EV≥1.2 単勝 / 複勝top2)')
print(SEP)
print(f'  {"年":>6}  {"単勝N":>8}  {"単勝ROI":>10}  {"複勝N":>8}  {"複勝ROI":>10}')
for yr in sorted(oos_w['日付_num'].astype(str).str[:2].unique()):
    yw = oos_w[(oos_w['日付_num'].astype(str).str[:2] == yr) & (oos_w['ev_win'] >= 1.2)]
    yp = oos_p[(oos_p['日付_num'].astype(str).str[:2] == yr) & (oos_p['rank_p'] <= 2)]
    rw, _, nw = roi_tan(yw)
    rp, _, np_ = roi_fuku(yp)
    print(f'  20{yr}    {nw:>8,}  {rw:>+10.1%}  {np_:>8,}  {rp:>+10.1%}')

# ── サマリー ──────────────────────────────────────────────
print(f'\n{SEP}')
avg_h = oos_w.groupby('race_id').size().mean()
print(f'平均頭数: {avg_h:.1f}  OOS期間: 2021-2026  レース数: {oos_w["race_id"].nunique():,}')
print(f'特徴量数: {len(avail)}')
