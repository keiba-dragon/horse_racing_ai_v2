# -*- coding: utf-8 -*-
"""
正ROI探索 v4 - モデル順位 × 人気ズレ / オッズ帯別詳細分析
roi_search3.py の OOS結果（oos_w / oos_p）を引き継ぎ
"""
import sys, io, re, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression

df = pd.read_parquet('C:/horse_racing_ai/data/processed/all_venues_features.parquet')
df['着順_num'] = pd.to_numeric(df['着順_num'], errors='coerce')
df = df[df['着順_num'].notna()].copy()
df['target_win']   = (df['着順_num'] == 1).astype(int)
df['target_place'] = (df['着順_num'] <= 3).astype(int)
for c in ['単勝オッズ', '複勝配当', '人気']:
    df[c] = pd.to_numeric(df[c], errors='coerce')
df['複勝倍率'] = df['複勝配当'] / 100.0

def dist_band(s):
    m = re.search(r'(\d{3,4})', str(s))
    if not m: return None
    v = int(m.group(1))
    if v <= 1400: return '短'
    if v <= 1800: return 'マイ'
    if v <= 2200: return '中'
    return '長'
df['距離帯'] = df['距離'].apply(dist_band)
df['surf']   = df['芝・ダ'].str.strip()
MAJOR_VENUES = {'東', '中', '阪', '京'}
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

FEATS = [
    '距離', '頭数', '馬番', '斤量', '馬体重', '馬体重増減',
    '内外枠', '斤量変化', '間隔', '連闘フラグ', '休み明けフラグ', 'クラス_rank',
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
    '1走前_頭数', '2走前_頭数', '1走前_馬番', '2走前_馬番',
    '近3走_平均着順', '近3走_勝率', '近3走_複勝率',
    '近5走_平均着順', '近5走_複勝率',
    '近5走_タイム指数平均', '近5走_上り3F平均', '近5走_タイム指数_std',
    '近5走_タイム指数_max', '近5走_タイム指数_min',
    '近5走_上り3F_min', '近5走_上り3F_std',
    '近5走_クラス調整_平均着順', '格上経験数_近5走',
    '近5走_着差タイム_クラス補正平均',
    '近走_改善トレンド', 'タイム指数_近3走_slope',
    '前走_追い上げ度', '前走_4角位置', '近5走_平均4角位置',
    '1走前_上り3F_指数', '2走前_上り3F_指数', '3走前_上り3F_指数',
    '脚質フィット', '展開フィット_v2', 'コース展開マッチ', '展開_コース_脚質フィット',
    'レース内_逃げ馬数', 'レース内_先行馬数',
    'レース内_相対脚質', 'コース_先行有利度', '推定ペース',
    '騎手_r200_勝率', '騎手_r200_複勝率',
    '騎手コース_r100_勝率', '騎手コース_r100_複勝率',
    '騎手馬場_r100_勝率', '騎手馬場_r100_複勝率',
    '騎手距離_r100_勝率', '騎手距離_r100_複勝率',
    '騎手脚質_r100_勝率', '騎手脚質_r100_複勝率',
    '種牡馬_勝率', '種牡馬_複勝率', '種牡馬_芝_勝率', '種牡馬_ダ_勝率',
    '母父馬_勝率', '母父馬_複勝率', '産地_勝率', '生産者_勝率',
    '馬_r20_勝率', '馬_r20_複勝率',
    '馬コース_r20_勝率', '馬コース_r20_複勝率',
    '馬距離_勝率', '馬距離_複勝率',
    '同会場_平均着順_近5走', '同会場_複勝率_近5走',
    '同馬場_平均着順_近5走', '同距離帯_平均着順_近5走',
    '芝ダ一致_平均着順_近5走', '良馬場_平均着順_近5走', '道悪_平均着順_近5走',
    'キャリア', 'キャリア_log', '馬体重トレンド_近5走', '前走コース一致',
    '芝ダ転向', '距離変化_前走', '騎手変更', '乗替り_近走不振',
]
avail = list(dict.fromkeys([c for c in FEATS if c in df.columns]))
for c in avail:
    df[c] = pd.to_numeric(df[c], errors='coerce')
print(f'特徴量: {len(avail)}個')

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
    return total_ret / n - 1.0, len(wins), n

print('学習中...')
all_w, all_p = [], []
for key in sorted(df['gk'].unique()):
    g = df[df['gk'] == key].sort_values('日付_num')
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
    prob_w = train_clf('target_win')
    if prob_w is not None:
        tw = te.copy()
        tw['prob_win'] = prob_w
        tw['ev_win']   = tw['prob_win'] * tw['単勝オッズ'].fillna(0)
        tw['rank_w']   = tw.groupby('race_id')['prob_win'].rank(ascending=False, method='min')
        tw['pop_rank'] = tw['人気']
        tw['gk'] = key
        all_w.append(tw)
    prob_p = train_clf('target_place', 100)
    if prob_p is not None:
        tp = te.copy()
        tp['prob_place'] = prob_p
        tp['rank_p']     = tp.groupby('race_id')['prob_place'].rank(ascending=False, method='min')
        tp['pop_rank'] = tp['人気']
        tp['gk'] = key
        all_p.append(tp)
    print(f'  {key}: tr={len(tr)} val={len(val)} OOS={len(te)}')

oos_w = pd.concat(all_w, ignore_index=True)
oos_p = pd.concat(all_p, ignore_index=True)
print(f'OOS: {len(oos_w):,}行')

SEP = '=' * 70

# ── 単勝: モデル1位 × オッズ帯 詳細 ──────────────────────
print(f'\n{SEP}')
print(' 単勝: モデル1位 × 単勝オッズ帯別ROI')
print(SEP)
r1 = oos_w[oos_w['rank_w'] == 1]
odds_ranges = [(2,4,'2-4倍'),(4,7,'4-7倍'),(7,12,'7-12倍'),(12,20,'12-20倍'),
               (20,40,'20-40倍'),(40,100,'40-100倍'),(3,15,'3-15倍'),(5,15,'5-15倍')]
for lo, hi, label in odds_ranges:
    sub = r1[(r1['単勝オッズ'] >= lo) & (r1['単勝オッズ'] < hi)]
    r, w, n = roi_tan(sub)
    if n < 50: continue
    print(f'  1位 [{label:10s}]: N={n:>6,}  的中={sub["target_win"].mean():.1%}  ROI={r:+.1%}')

# ── 単勝: モデル1位 × 人気ズレ ──────────────────────────
print(f'\n{SEP}')
print(' 単勝: モデル1位 × 人気（市場）ランクのズレ')
print(SEP)
print(f'  {"人気ランク":^12}  {"N":>7}  {"的中率":>6}  {"ROI":>8}')
for pop in range(1, 10):
    sub = r1[r1['pop_rank'] == pop]
    r, w, n = roi_tan(sub)
    if n < 50: continue
    print(f'  モデル1位×人気{pop}番: N={n:>6,}  的中={sub["target_win"].mean():.1%}  ROI={r:+.1%}')

print()
print('  [モデル1位 × 人気外れ: 市場が2-4番人気の馬]')
sub_val = r1[(r1['pop_rank'] >= 2) & (r1['pop_rank'] <= 4)]
r, w, n = roi_tan(sub_val)
print(f'  1位 × 人気2-4: N={n:>6,}  的中={sub_val["target_win"].mean():.1%}  ROI={r:+.1%}')

for lo, hi, label in [(2,4,'2-4'),(2,3,'2-3'),(3,5,'3-5'),(5,8,'5-8')]:
    sub = r1[(r1['pop_rank'] >= lo) & (r1['pop_rank'] <= hi)]
    r, w, n = roi_tan(sub)
    if n < 50: continue
    print(f'  1位 × 人気{label}: N={n:>6,}  的中={sub["target_win"].mean():.1%}  ROI={r:+.1%}')

# ── 単勝: モデル1位 × 人気ズレ × オッズ帯 ────────────────
print(f'\n{SEP}')
print(' 単勝: モデル1位 × 人気外れ(2-5) × オッズ帯')
print(SEP)
r1_upset = r1[(r1['pop_rank'] >= 2) & (r1['pop_rank'] <= 5)]
for lo, hi, label in [(5,15,'5-15倍'),(7,20,'7-20倍'),(10,30,'10-30倍'),
                       (5,25,'5-25倍'),(3,10,'3-10倍')]:
    sub = r1_upset[(r1_upset['単勝オッズ'] >= lo) & (r1_upset['単勝オッズ'] < hi)]
    r, w, n = roi_tan(sub)
    if n < 50: continue
    print(f'  人気2-5 [{label:8s}]: N={n:>6,}  的中={sub["target_win"].mean():.1%}  ROI={r:+.1%}')

# ── 単勝: EV × 人気ズレ ──────────────────────────────────
print(f'\n{SEP}')
print(' 単勝: EV×人気ズレ（モデル高EV & 市場低評価）')
print(SEP)
for ev_lo in [1.0, 1.2, 1.5]:
    for pop_lo, pop_hi in [(2,5),(2,4),(3,6),(4,8)]:
        sub = oos_w[(oos_w['ev_win'] >= ev_lo)
                    & (oos_w['pop_rank'] >= pop_lo)
                    & (oos_w['pop_rank'] <= pop_hi)]
        r, w, n = roi_tan(sub)
        if n < 100: continue
        print(f'  EV≥{ev_lo:.1f} × 人気{pop_lo}-{pop_hi}: N={n:>6,}  的中={sub["target_win"].mean():.1%}  ROI={r:+.1%}')

# ── 単勝: モデル順位 × EV × オッズ 総合フィルタ ─────────
print(f'\n{SEP}')
print(' 単勝: 複合フィルタ探索')
print(SEP)
best_roi, best_desc = -1.0, ''
for rank_th in [1, 2]:
    for ev_lo in [0.8, 1.0, 1.2, 1.3]:
        for odds_lo, odds_hi in [(5,20),(5,30),(7,25),(10,40),(3,15)]:
            for pop_lo, pop_hi in [(1,99),(2,10),(1,5),(2,5)]:
                sub = oos_w[
                    (oos_w['rank_w'] <= rank_th) &
                    (oos_w['ev_win'] >= ev_lo) &
                    (oos_w['単勝オッズ'] >= odds_lo) &
                    (oos_w['単勝オッズ'] < odds_hi) &
                    (oos_w['pop_rank'] >= pop_lo) &
                    (oos_w['pop_rank'] <= pop_hi)
                ]
                r, w, n = roi_tan(sub)
                if n < 200 or np.isnan(r): continue
                if r > best_roi:
                    best_roi = r
                    best_desc = f'rank≤{rank_th} EV≥{ev_lo:.1f} オッズ{odds_lo}-{odds_hi} 人気{pop_lo}-{pop_hi}'
                if r > -0.10:
                    print(f'  ★ {best_desc}: N={n:>6,}  的中={sub["target_win"].mean():.1%}  ROI={r:+.1%}')
if best_roi <= -0.10:
    print(f'  N≥200で最良: ROI={best_roi:+.1%}  条件: {best_desc}')

# ── 複勝: モデル1位 × 人気・オッズ帯 ────────────────────
print(f'\n{SEP}')
print(' 複勝: モデルtop2 × 単勝オッズ帯')
print(SEP)
r2p = oos_p[oos_p['rank_p'] <= 2]
for lo, hi, label in [(3,8,'3-8倍'),(5,12,'5-12倍'),(8,20,'8-20倍'),
                       (10,30,'10-30倍'),(3,15,'3-15倍'),(2,6,'2-6倍')]:
    sub = r2p[(r2p['単勝オッズ'] >= lo) & (r2p['単勝オッズ'] < hi)]
    r, w, n = roi_fuku(sub)
    if n < 100: continue
    print(f'  top2 [{label:8s}]: N={n:>6,}  入着={sub["target_place"].mean():.1%}  ROI={r:+.1%}')

print()
print('  [複勝: top1 × オッズ帯]')
r1p = oos_p[oos_p['rank_p'] <= 1]
for lo, hi, label in [(3,8,'3-8倍'),(5,12,'5-12倍'),(8,20,'8-20倍'),(10,30,'10-30倍')]:
    sub = r1p[(r1p['単勝オッズ'] >= lo) & (r1p['単勝オッズ'] < hi)]
    r, w, n = roi_fuku(sub)
    if n < 100: continue
    print(f'  top1 [{label:8s}]: N={n:>6,}  入着={sub["target_place"].mean():.1%}  ROI={r:+.1%}')

# ── 複勝: 人気ズレ × top2 ─────────────────────────────
print(f'\n{SEP}')
print(' 複勝: モデルtop2 × 人気ズレ')
print(SEP)
for pop_lo, pop_hi in [(1,3),(2,4),(3,6),(4,8),(1,5),(2,6)]:
    sub = r2p[(r2p['pop_rank'] >= pop_lo) & (r2p['pop_rank'] <= pop_hi)]
    r, w, n = roi_fuku(sub)
    if n < 200: continue
    print(f'  top2 × 人気{pop_lo}-{pop_hi}: N={n:>6,}  入着={sub["target_place"].mean():.1%}  ROI={r:+.1%}')

# ── コース別 詳細ROI ─────────────────────────────────────
print(f'\n{SEP}')
print(' コース別: モデル1位×人気2-5 ROI')
print(SEP)
r1_upset_all = oos_w[(oos_w['rank_w'] == 1) & (oos_w['pop_rank'] >= 2) & (oos_w['pop_rank'] <= 5)]
for key in sorted(r1_upset_all['gk'].unique()):
    sub = r1_upset_all[r1_upset_all['gk'] == key]
    r, w, n = roi_tan(sub)
    if n < 50: continue
    print(f'  {key:10s}: N={n:>5,}  的中={sub["target_win"].mean():.1%}  ROI={r:+.1%}')

# ── 年別推移: ベスト戦略 ─────────────────────────────────
print(f'\n{SEP}')
print(' 年別: モデル1位×人気2-5 ROI推移')
print(SEP)
for yr in sorted(oos_w['日付_num'].astype(str).str[:2].unique()):
    yw = oos_w[oos_w['日付_num'].astype(str).str[:2] == yr]
    sub = yw[(yw['rank_w'] == 1) & (yw['pop_rank'] >= 2) & (yw['pop_rank'] <= 5)]
    r, w, n = roi_tan(sub)
    print(f'  20{yr}: N={n:>5,}  的中={sub["target_win"].mean():.1%}  ROI={r:+.1%}')

print(f'\n平均頭数: {oos_w.groupby("race_id").size().mean():.1f}  レース数: {oos_w["race_id"].nunique():,}')
