# -*- coding: utf-8 -*-
"""
正ROI条件探索 - カンニングなし
train: 2013-2018 / val: 2019-2020 / OOS test: 2021+
"""
import sys, io, os, re, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FEAT_FILE = os.path.join(BASE_DIR, 'data', 'processed', 'all_venues_features.parquet')

TRAIN_END = 181231
VAL_END   = 201231
OOS_START = 210101

DIST_BANDS = [(0,1400,'短距離'),(1401,1800,'マイル'),(1801,2200,'中距離'),(2201,9999,'長距離')]

def to_num_chakujun(s):
    s = str(s).strip().translate(str.maketrans('０１２３４５６７８９','0123456789'))
    m = re.match(r'^(\d+)$', re.sub(r'[^0-9]','',s))
    return int(m.group(1)) if m else None

def dist_band(v):
    for lo,hi,name in DIST_BANDS:
        if lo <= v <= hi: return name
    return None

def group_key(surf, d):
    if surf not in ('芝','ダ') or not d: return None
    if surf == 'ダ' and d in ('中距離','長距離'): d = '中長距離'
    return f"{surf}_{d}"

def roi_calc(bets_df, odds_col='単勝オッズ'):
    """単勝100円ベースROI: Σ(win × オッズ) / N - 1"""
    sub = bets_df.dropna(subset=[odds_col])
    if len(sub) == 0: return np.nan, 0, 0
    wins = sub[sub['target_win'] == 1]
    return wins[odds_col].sum() / len(sub) - 1.0, len(wins), len(sub)

# ── 読み込み ──────────────────────────────────────────────
print("データ読み込み...")
df = pd.read_parquet(FEAT_FILE)
print(f"  shape: {df.shape}")

df['着順_num'] = df['着順'].apply(to_num_chakujun)
df = df[df['着順_num'].notna()].copy()
df['着順_num'] = df['着順_num'].astype(float)
df['target_win']   = (df['着順_num'] == 1).astype(int)
df['target_place'] = (df['着順_num'] <= 3).astype(int)

df['日付_num'] = pd.to_numeric(df['日付'], errors='coerce')
df = df.dropna(subset=['日付_num'])
df['日付_num'] = df['日付_num'].astype(int)

# オッズ
for c in ['単勝オッズ', '単勝配当', '複勝配当']:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

# 距離: 'ダ1400' → 1400
def parse_dist(s):
    m = re.search(r'(\d{3,4})', str(s))
    return int(m.group(1)) if m else None

df['距離_num'] = df['距離'].apply(parse_dist) if '距離' in df.columns else np.nan
df['距離帯']   = df['距離_num'].apply(lambda v: dist_band(v) if pd.notna(v) else None)

surf_col = '芝・ダ' if '芝・ダ' in df.columns else None
if surf_col:
    df['surface'] = df[surf_col].astype(str).str.strip()

df['group_key'] = df.apply(lambda r: group_key(r.get('surface',''), r.get('距離帯')), axis=1)

race_cols = ['日付','開催','レース名']
df['race_id'] = df[race_cols].astype(str).agg('_'.join, axis=1)

df = df[df['group_key'].notna()]
print(f"  有効行: {len(df):,}  レース: {df['race_id'].nunique():,}  グループ: {df['group_key'].nunique()}")

# ── 特徴量リスト ───────────────────────────────────────────
FEATS = [
    # レース条件
    '距離_num', '頭数', '馬番', '斤量',
    '馬体重', '馬体重変化',
    # 直近N走の生データ
    '1走前_着順_num', '2走前_着順_num', '3走前_着順_num',
    '4走前_着順_num', '5走前_着順_num',
    '1走前_タイム指数', '2走前_タイム指数', '3走前_タイム指数',
    '1走前_上り3F',    '2走前_上り3F',    '3走前_上り3F',
    '1走前_4角',       '2走前_4角',       '3走前_4角',
    '1走前_単勝オッズ','2走前_単勝オッズ','3走前_単勝オッズ',
    '1走前_走破タイム_sec',
    # 近走集計
    '近3走_平均着順', '近3走_複勝率', '近3走_勝率',
    '近5走_平均着順', '近5走_タイム指数平均', '近5走_上り3F平均',
    '近5走_平均4角位置',
    '近5走_タイム指数_std',
    '同馬場_平均着順_近5走', '同距離帯_平均着順_近5走',
    '同会場_平均着順_近5走', '同会場_複勝率_近5走',
    '芝ダ一致_平均着順_近5走',
    '良馬場_平均着順_近5走', '道悪_平均着順_近5走',
    '距離短縮時_平均着順_近5走', '距離延長時_平均着順_近5走',
    # トレンド
    'タイム指数_近3走_slope', '近走_改善トレンド', '馬体重トレンド_近5走',
    # キャリア
    'キャリア', 'キャリア_log',
    'クラス_rank',
    # 騎手
    '騎手_平均着順', '騎手_r200_勝率',
    '騎手コース_r100_勝率',
    # 馬コース
    '馬コース_r20_勝率', '馬コース_r20_複勝率',
    # 間隔
    '間隔',
    # コース変更
    '前走コース一致',
]

available = [c for c in FEATS if c in df.columns]
print(f"特徴量: {len(available)}/{len(FEATS)}個")

for c in available:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# ── 追加: マーケット誤差系特徴量 ─────────────────────────
# 前走で市場が大外れ (高オッズ馬が1着, または本命が大敗)
if '1走前_着順_num' in df.columns and '1走前_単勝オッズ' in df.columns:
    df['1走前_着順_num'] = pd.to_numeric(df['1走前_着順_num'], errors='coerce')
    df['前走_穴当選'] = (
        (df['1走前_着順_num'] == 1) & (df['1走前_単勝オッズ'] >= 10)
    ).astype(float)
    df['前走_本命失敗'] = (
        (df['1走前_着順_num'] >= 5) & (df['1走前_単勝オッズ'] <= 4)
    ).astype(float)
    df['前走_高オッズ負け'] = (
        (df['1走前_着順_num'] >= 6) & (df['1走前_単勝オッズ'] >= 15)
    ).astype(float)
    available += ['前走_穴当選', '前走_本命失敗', '前走_高オッズ負け']

# 長期 vs 短期ギャップ (市場バイアス利用)
if '近3走_勝率' in df.columns and 'キャリア' in df.columns:
    # 騎手勝率を使って通算勝率の代理として使う
    pass

print(f"最終特徴量: {len(available)}個")

# ── 学習 ──────────────────────────────────────────────────
print("\n学習中 (train≤2018, val 2019-2020, OOS 2021+)...")
all_oos = []

for key in sorted(df['group_key'].unique()):
    g     = df[df['group_key'] == key].sort_values('日付_num')
    train = g[g['日付_num'] <= TRAIN_END]
    val   = g[(g['日付_num'] > TRAIN_END) & (g['日付_num'] <= VAL_END)]
    test  = g[g['日付_num'] >= OOS_START]

    if len(train) < 500 or train['target_win'].sum() < 50: continue
    if len(test) < 200: continue

    feat = [c for c in available if c in g.columns]
    X_tr = train[feat].astype(float)
    y_tr = train['target_win']
    X_te = test[feat].astype(float)

    clf = LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=31,
        min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, class_weight='balanced',
        random_state=42, n_jobs=-1, verbose=-1,
    )
    clf.fit(X_tr, y_tr)
    raw_prob = clf.predict_proba(X_te)[:, 1]

    # Isotonic calibration on val set
    if len(val) >= 200:
        X_val = val[feat].astype(float)
        raw_val = clf.predict_proba(X_val)[:, 1]
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(raw_val, val['target_win'].values)
        cal_prob = iso.predict(raw_prob)
    else:
        cal_prob = raw_prob

    test = test.copy()
    test['prob'] = cal_prob
    test['ev']   = test['prob'] * test['単勝オッズ'].fillna(0)
    test['pred_rank'] = test.groupby('race_id')['prob'].rank(ascending=False, method='min')
    test['group_key'] = key
    all_oos.append(test)

    r1    = test[test['pred_rank']==1]
    hr_r1 = r1['target_win'].mean() if len(r1) > 0 else 0
    print(f"  {key:<18} tr={len(train):>6}  OOS={len(test):>6}  1位的中={hr_r1:.1%}")

oos = pd.concat(all_oos, ignore_index=True)
print(f"\nOOS: {len(oos):,}行 / {oos['race_id'].nunique():,}レース")

# ── 分析 ──────────────────────────────────────────────────
SEP = "=" * 65

def show_table(sub_base, label, ev_thrs, odds_info=""):
    n_base = len(sub_base)
    print(f"\n  [{label}] N={n_base:,} {odds_info}")
    print(f"  {'EV閾値':>7}  {'N':>7}  {'的中率':>7}  {'ROI':>8}")
    for thr in ev_thrs:
        s = sub_base[sub_base['ev'] >= thr].dropna(subset=['単勝オッズ'])
        if len(s) < 50: continue
        r, w, n = roi_calc(s)
        print(f"  EV≥{thr:.1f}   {n:>7,}  {s['target_win'].mean():>7.1%}  {r:>+8.1%}")

# [A] 全体 EV閾値
print(f"\n{SEP}\n[A] 全体: EV閾値 × ROI\n{SEP}")
show_table(oos, "全グループ全オッズ", [0.8, 1.0, 1.1, 1.2, 1.3, 1.5])

# [B] オッズ帯別
print(f"\n{SEP}\n[B] オッズ帯 × EV閾値\n{SEP}")
for olo, ohi, label in [(0,999,'全'),(3,999,'3+'),(5,999,'5+'),(5,30,'5-30'),(8,30,'8-30'),(5,20,'5-20')]:
    sub = oos[(oos['単勝オッズ'] >= olo) & (oos['単勝オッズ'] <= ohi)]
    show_table(sub, label, [1.0, 1.1, 1.2, 1.3], f"オッズ{olo}-{ohi}")

# [C] モデル1位 × EV × オッズ
print(f"\n{SEP}\n[C] モデル1位 × EV × オッズ\n{SEP}")
r1_all = oos[oos['pred_rank'] == 1]
rr, rw, rn = roi_calc(r1_all)
print(f"  モデル1位全体: N={rn:,}  的中={r1_all['target_win'].mean():.1%}  ROI={rr:+.1%}")
for olo, ohi in [(0,999),(5,30),(8,30),(5,20)]:
    sub = r1_all[(r1_all['単勝オッズ'] >= olo) & (r1_all['単勝オッズ'] <= ohi)]
    show_table(sub, f"1位+オッズ{olo}-{ohi}", [0.8, 1.0, 1.1, 1.2])

# [D] グループ × EV × オッズ の全組み合わせ → 正ROI一覧
print(f"\n{SEP}\n[D] グループ別 正ROI一覧 (N≥100)\n{SEP}")
best = []
for key in oos['group_key'].unique():
    sub = oos[oos['group_key'] == key]
    for olo, ohi in [(0,999),(5,30),(8,30)]:
        for ethr in [0.8, 1.0, 1.1, 1.2]:
            s = sub[(sub['単勝オッズ'] >= olo) & (sub['単勝オッズ'] <= ohi) & (sub['ev'] >= ethr)].dropna(subset=['単勝オッズ'])
            if len(s) < 100: continue
            r, w, n = roi_calc(s)
            best.append({'group':key,'odds':f"{olo}-{ohi}",'ev':ethr,'N':n,'hit%':w/n,'ROI':r})

bdf = pd.DataFrame(best).sort_values('ROI', ascending=False)
print(bdf[bdf['ROI'] > -0.05].head(20).to_string(index=False))
print(f"\n正ROI (N≥100) 件数: {(bdf['ROI'] > 0).sum()} / {len(bdf)}")

# [E] 前走穴当選フラグ
if '前走_穴当選' in oos.columns:
    print(f"\n{SEP}\n[E] 前走穴当選 (前走10倍以上で1着) フラグ効果\n{SEP}")
    for flag, label in [(1,'前走穴当選'),(0,'それ以外')]:
        sub = oos[oos['前走_穴当選'] == flag].dropna(subset=['単勝オッズ'])
        r,w,n = roi_calc(sub)
        print(f"  {label}: N={n:,}  的中={sub['target_win'].mean():.1%}  ROI={r:+.1%}")
    # 穴当選 × EV
    sub2 = oos[oos['前走_穴当選'] == 1]
    show_table(sub2, "前走穴当選 × EV", [0.8, 1.0, 1.1, 1.2])

# [F] 年別ROI推移 (最良条件)
print(f"\n{SEP}\n[F] 年別ROI推移 (EV≥1.1, オッズ5-30)\n{SEP}")
best_cond = oos[(oos['単勝オッズ'] >= 5) & (oos['単勝オッズ'] <= 30) & (oos['ev'] >= 1.1)].dropna(subset=['単勝オッズ'])
best_cond['year'] = best_cond['日付_num'].astype(str).str[:4]
print(f"  {'年':>6}  {'N':>7}  {'的中率':>8}  {'ROI':>8}")
for yr in sorted(best_cond['year'].unique()):
    ys = best_cond[best_cond['year'] == yr]
    r,w,n = roi_calc(ys)
    print(f"  {yr}  {n:>7,}  {ys['target_win'].mean():>8.1%}  {r:>+8.1%}")

# [G] 特徴量重要度 Top20
print(f"\n{SEP}\n[G] 特徴量重要度 (最後に学習したグループ)\n{SEP}")
try:
    fi = pd.Series(clf.feature_importances_, index=feat).sort_values(ascending=False)
    print(fi.head(20).to_string())
except: pass

# サマリー
print(f"\n{SEP}\nサマリー\n{SEP}")
avg_h = oos.groupby('race_id').size().mean()
print(f"  平均頭数: {avg_h:.1f}  ランダム勝率: {1/avg_h:.1%}  ランダムROI: ≈-20%")
r_rand, w_rand, n_rand = roi_calc(oos)
print(f"  全馬買い ROI: {r_rand:+.1%}")
r1s = oos[oos['pred_rank']==1].dropna(subset=['単勝オッズ'])
rr,rw,rn = roi_calc(r1s)
print(f"  モデル1位ROI: {rr:+.1%} ({rw}/{rn})")
