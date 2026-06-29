# -*- coding: utf-8 -*-
"""
正ROI探索 v5 - 市場特化特徴量 + 単勝オッズを特徴量化
train≤181231 / val 190101-201231 / OOS 210101+
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

df['着順_num'] = pd.to_numeric(df['着順_num'], errors='coerce')
df = df[df['着順_num'].notna()].copy()
df['target_win']   = (df['着順_num'] == 1).astype(int)
df['target_place'] = (df['着順_num'] <= 3).astype(int)
for c in ['単勝オッズ', '複勝配当', '人気']:
    df[c] = pd.to_numeric(df[c], errors='coerce')
df['複勝倍率'] = df['複勝配当'] / 100.0

# ── 市場特化特徴量を追加 ──────────────────────────────────
print('市場特化特徴量を生成中...')

# 騎手のコース特化度（全体比）
if '騎手コース_r100_勝率' in df.columns and '騎手_r200_勝率' in df.columns:
    df['騎手コース特化'] = (
        pd.to_numeric(df['騎手コース_r100_勝率'], errors='coerce') -
        pd.to_numeric(df['騎手_r200_勝率'],       errors='coerce')
    )

# 馬のこの会場への親和性（全体複勝率との差）
if '同会場_複勝率_近5走' in df.columns and '近5走_複勝率' in df.columns:
    df['会場親和性'] = (
        pd.to_numeric(df['同会場_複勝率_近5走'], errors='coerce') -
        pd.to_numeric(df['近5走_複勝率'],        errors='coerce')
    )

# 馬場(芝ダ)適性（全体比）
if '芝ダ一致_平均着順_近5走' in df.columns and '近5走_平均着順' in df.columns:
    df['馬場適性超過'] = (
        pd.to_numeric(df['近5走_平均着順'],           errors='coerce') -
        pd.to_numeric(df['芝ダ一致_平均着順_近5走'],  errors='coerce')
        # 正=今回馬場の方が成績良い
    )

# 種牡馬のこのコース特化度
df['種牡馬コース特化'] = np.nan
surf_col = '芝・ダ' if '芝・ダ' in df.columns else None
if surf_col and '種牡馬_芝_勝率' in df.columns and '種牡馬_ダ_勝率' in df.columns and '種牡馬_勝率' in df.columns:
    is_turf = df[surf_col].str.strip() == '芝'
    is_dirt = df[surf_col].str.strip() == 'ダ'
    turf_rate = pd.to_numeric(df['種牡馬_芝_勝率'], errors='coerce')
    dirt_rate = pd.to_numeric(df['種牡馬_ダ_勝率'], errors='coerce')
    base_rate = pd.to_numeric(df['種牡馬_勝率'],    errors='coerce')
    df.loc[is_turf, '種牡馬コース特化'] = turf_rate[is_turf] - base_rate[is_turf]
    df.loc[is_dirt, '種牡馬コース特化'] = dirt_rate[is_dirt] - base_rate[is_dirt]

# 前走からのクラス変化（負=降格=恩恵あり）
if '1走前_クラス差' in df.columns:
    df['クラス降格恩恵'] = -pd.to_numeric(df['1走前_クラス差'], errors='coerce')

# 市場含意確率（単勝オッズから）: 返還率75%想定
df['市場含意P'] = 0.75 / df['単勝オッズ'].clip(lower=1.0)

# 近走タイム指数 vs クラス補正期待値
if '近5走_タイム指数平均' in df.columns and 'クラス_rank' in df.columns:
    ti = pd.to_numeric(df['近5走_タイム指数平均'], errors='coerce')
    # クラスごとのタイム指数中央値との差
    cl = pd.to_numeric(df['クラス_rank'], errors='coerce')
    class_med = df.groupby('クラス_rank')['近5走_タイム指数平均'].transform(
        lambda x: pd.to_numeric(x, errors='coerce').median()
    )
    df['タイム指数クラス超過'] = ti - pd.to_numeric(class_med, errors='coerce')

print('特化特徴量を生成しました')

# ── コースグループ ────────────────────────────────────────
def dist_band(s):
    m = re.search(r'(\d{3,4})', str(s))
    if not m: return None
    v = int(m.group(1))
    if v <= 1400: return '短'
    if v <= 1800: return 'マイ'
    if v <= 2200: return '中'
    return '長'

df['距離帯'] = df['距離'].apply(dist_band)
df['surf']   = df['芝・ダ'].str.strip() if '芝・ダ' in df.columns else ''

# 右回り/左回り
LEFT_VENUES  = {'東', '名', '新', '福'}   # 東京/中京/新潟/福島
RIGHT_VENUES = {'中', '阪', '京', '小', '函', '札'}  # 中山/阪神/京都/小倉/函館/札幌

# 今回_会場を数値エンコード（特徴量として使う）
VENUE_MAP = {'東':0,'中':1,'阪':2,'京':3,'名':4,'新':5,'小':6,'福':7,'札':8,'函':9}
df['会場コード_num'] = df['今回_会場'].astype(str).str.strip().map(VENUE_MAP)

def gk(r):
    s, d = r['surf'], r['距離帯']
    if s not in ('芝', 'ダ') or not d: return None
    if s == 'ダ' and d in ('中', '長'): d = '中長'
    v = str(r.get('今回_会場', '')).strip()
    if v in LEFT_VENUES:   vt = '左'
    elif v in RIGHT_VENUES: vt = '右'
    else: return None
    return f'{s}_{d}_{vt}'

df['gk'] = df.apply(gk, axis=1)
df = df[df['gk'].notna()]
df['race_id'] = (df['日付'].astype(str) + '_' + df['開催'].astype(str)
                 + '_' + df['レース名'].astype(str))

# ── 特徴量リスト ──────────────────────────────────────────
FEATS_BASE = [
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

# 市場特化特徴量
FEATS_MARKET = [
    '単勝オッズ',       # 市場価格そのものを条件として学習
    '市場含意P',        # 0.75 / 単勝オッズ
    '人気',             # 市場人気順
    '会場コード_num',   # 会場そのもの（0=東京〜9=函館）
    '騎手コース特化',   # 騎手のコース特化度
    '会場親和性',       # 馬の会場親和性
    '馬場適性超過',     # 馬場(芝ダ)の得意度
    '種牡馬コース特化', # 種牡馬のコース特化度
    'クラス降格恩恵',   # クラス降格の恩恵
    'タイム指数クラス超過',  # クラス内でのタイム指数優位性
]

FEATS = FEATS_BASE + FEATS_MARKET
avail = list(dict.fromkeys([c for c in FEATS if c in df.columns]))
for c in avail:
    df[c] = pd.to_numeric(df[c], errors='coerce')
print(f'特徴量: {len(avail)}個 (うち市場特化: {len([c for c in FEATS_MARKET if c in df.columns])}個)')

def roi_tan(sub):
    s = sub.dropna(subset=['単勝オッズ'])
    if len(s) == 0: return np.nan, 0, 0
    w = s[s['target_win'] == 1]
    return w['単勝オッズ'].sum() / len(s) - 1, len(w), len(s)

def roi_fuku(sub):
    n = len(sub)
    if n == 0: return np.nan, 0, 0
    wins = sub[sub['target_place'] == 1]
    return wins['複勝倍率'].dropna().sum() / n - 1.0, len(wins), n

# ── 学習 ─────────────────────────────────────────────────
print('学習中...')
all_w, all_p = [], []

for key in sorted(df['gk'].unique()):
    g   = df[df['gk'] == key].sort_values('日付_num')
    tr  = g[g['日付_num'] <= 181231]
    val = g[(g['日付_num'] > 181231) & (g['日付_num'] <= 201231)]
    te  = g[g['日付_num'] >= 210101]
    if len(tr) < 300 or len(te) < 200: continue
    feat = [c for c in avail if c in g.columns]

    def train_clf(target_col, min_pos=30):
        if tr[target_col].sum() < min_pos: return None
        clf = LGBMClassifier(
            n_estimators=600, learning_rate=0.03, num_leaves=63,
            min_child_samples=20, subsample=0.8, colsample_bytree=0.7,
            reg_alpha=0.1, reg_lambda=1.0, class_weight='balanced',
            random_state=42, n_jobs=-1, verbose=-1)
        clf.fit(tr[feat].astype(float), tr[target_col])
        raw = clf.predict_proba(te[feat].astype(float))[:, 1]
        if len(val) >= 100:
            rv = clf.predict_proba(val[feat].astype(float))[:, 1]
            iso = IsotonicRegression(out_of_bounds='clip')
            iso.fit(rv, val[target_col].values)
            return iso.predict(raw)
        return raw

    prob_w = train_clf('target_win')
    if prob_w is not None:
        tw = te.copy()
        tw['prob_win']  = prob_w
        # 市場含意確率との差 = モデルエッジ
        tw['market_P']  = 0.75 / tw['単勝オッズ'].clip(lower=1.0)
        tw['edge']      = tw['prob_win'] - tw['market_P']
        tw['ev_win']    = tw['prob_win'] * tw['単勝オッズ'].fillna(0)
        tw['rank_w']    = tw.groupby('race_id')['prob_win'].rank(ascending=False, method='min')
        tw['rank_edge'] = tw.groupby('race_id')['edge'].rank(ascending=False, method='min')
        tw['gk'] = key
        all_w.append(tw)

    prob_p = train_clf('target_place', 60)
    if prob_p is not None:
        tp = te.copy()
        tp['prob_place'] = prob_p
        tp['rank_p']     = tp.groupby('race_id')['prob_place'].rank(ascending=False, method='min')
        tp['gk'] = key
        all_p.append(tp)

    print(f'  {key}: tr={len(tr)} val={len(val)} OOS={len(te)}')

oos_w = pd.concat(all_w, ignore_index=True)
oos_p = pd.concat(all_p, ignore_index=True)
print(f'OOS: {len(oos_w):,}行')

SEP = '=' * 70

# ── 単勝: edge（モデル - 市場）フィルタ ──────────────────
print(f'\n{SEP}')
print(' 単勝: モデルエッジ（model_P - market_P）フィルタ')
print(SEP)
r_rnd, _, n_rnd = roi_tan(oos_w)
print(f'  ランダム全馬: ROI={r_rnd:+.1%}  N={n_rnd:,}')
print()

for thr in [0.00, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10]:
    sub = oos_w[oos_w['edge'] >= thr]
    r, w, n = roi_tan(sub)
    if n < 100: continue
    print(f'  edge≥{thr:.2f}: N={n:>7,}  的中={sub["target_win"].mean():.1%}  ROI={r:+.1%}')

print()
print('  [edge上位 × オッズ帯]')
for odds_lo, odds_hi in [(3,10),(5,20),(7,25),(10,40)]:
    base = oos_w[(oos_w['単勝オッズ'] >= odds_lo) & (oos_w['単勝オッズ'] < odds_hi)]
    for thr in [0.01, 0.02, 0.03, 0.05]:
        sub = base[base['edge'] >= thr]
        r, w, n = roi_tan(sub)
        if n < 200: continue
        print(f'  edge≥{thr:.2f} [{odds_lo}-{odds_hi}倍]: N={n:>7,}  的中={sub["target_win"].mean():.1%}  ROI={r:+.1%}')

# ── 単勝: edge rank ────────────────────────────────────────
print(f'\n{SEP}')
print(' 単勝: レース内 edge 1位（最も市場が過小評価している馬）')
print(SEP)
r1_edge = oos_w[oos_w['rank_edge'] == 1]
r, _, n = roi_tan(r1_edge)
print(f'  edge1位 全体: N={n:,}  的中={r1_edge["target_win"].mean():.1%}  ROI={r:+.1%}')

for thr in [0.0, 0.01, 0.02, 0.03, 0.05]:
    sub = r1_edge[r1_edge['edge'] >= thr]
    r, _, n = roi_tan(sub)
    if n < 200: continue
    print(f'  edge1位 edge≥{thr:.2f}: N={n:>7,}  的中={sub["target_win"].mean():.1%}  ROI={r:+.1%}')

print()
for odds_lo, odds_hi in [(3,15),(5,20),(7,25),(10,40)]:
    sub = r1_edge[(r1_edge['単勝オッズ'] >= odds_lo) & (r1_edge['単勝オッズ'] < odds_hi)]
    r, _, n = roi_tan(sub)
    if n < 100: continue
    print(f'  edge1位 [{odds_lo}-{odds_hi}倍]: N={n:>7,}  的中={sub["target_win"].mean():.1%}  ROI={r:+.1%}')

# ── 単勝: EV（既存手法との比較） ─────────────────────────
print(f'\n{SEP}')
print(' 単勝: EV（比較用）')
print(SEP)
for ev in [1.0, 1.2, 1.5]:
    sub = oos_w[oos_w['ev_win'] >= ev]
    r, _, n = roi_tan(sub)
    if n < 100: continue
    print(f'  EV≥{ev:.1f}: N={n:>7,}  的中={sub["target_win"].mean():.1%}  ROI={r:+.1%}')

# ── コース別 edge1位 ROI ──────────────────────────────────
print(f'\n{SEP}')
print(' コース別: edge1位 ROI')
print(SEP)
print(f'  {"gk":16s}  {"N":>7}  {"的中":>6}  {"ROI":>8}  | edge≥0.02 ROI')
for key in sorted(r1_edge['gk'].unique()):
    g1 = r1_edge[r1_edge['gk'] == key]
    g2 = r1_edge[(r1_edge['gk'] == key) & (r1_edge['edge'] >= 0.02)]
    r1, _, n1 = roi_tan(g1)
    r2, _, n2 = roi_tan(g2)
    if n1 < 50: continue
    print(f'  {key:16s}  {n1:>7,}  {g1["target_win"].mean():>6.1%}  {r1:>+8.1%}  | N={n2:>5,} {r2:>+8.1%}')

# ── 複勝: edge フィルタ ───────────────────────────────────
print(f'\n{SEP}')
print(' 複勝: top2 & edge フィルタ')
print(SEP)
for top_n in [1, 2]:
    sub = oos_p[oos_p['rank_p'] <= top_n]
    r, _, n = roi_fuku(sub)
    print(f'  top-{top_n}: N={n:,}  入着={sub["target_place"].mean():.1%}  ROI={r:+.1%}')

# ── 年別ROI推移 ───────────────────────────────────────────
print(f'\n{SEP}')
print(' 年別: edge1位 ROI推移')
print(SEP)
for yr in sorted(oos_w['日付_num'].astype(str).str[:2].unique()):
    yw = oos_w[oos_w['日付_num'].astype(str).str[:2] == yr]
    sub = yw[yw['rank_edge'] == 1]
    r, _, n = roi_tan(sub)
    print(f'  20{yr}: N={n:>5,}  的中={sub["target_win"].mean():.1%}  ROI={r:+.1%}')

print(f'\n特徴量数: {len(avail)}  OOSレース数: {oos_w["race_id"].nunique():,}')
