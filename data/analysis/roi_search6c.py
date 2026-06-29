# -*- coding: utf-8 -*-
"""主要4会場×芝ダ 8グループ - EV(prob×odds)フィルタ版 距離帯なし"""
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
df['target_win'] = (df['着順_num'] == 1).astype(int)
for c in ['単勝オッズ', '人気']: df[c] = pd.to_numeric(df[c], errors='coerce')

for col_a, col_b, new in [
    ('騎手コース_r100_勝率', '騎手_r200_勝率', '騎手コース特化'),
    ('同会場_複勝率_近5走',  '近5走_複勝率',   '会場親和性'),
    ('近5走_平均着順', '芝ダ一致_平均着順_近5走', '馬場適性超過'),
]:
    if col_a in df.columns and col_b in df.columns:
        df[new] = pd.to_numeric(df[col_a], errors='coerce') - pd.to_numeric(df[col_b], errors='coerce')

df['種牡馬コース特化'] = np.nan
if '芝・ダ' in df.columns:
    s_col = df['芝・ダ'].str.strip()
    for surf, scol in [('芝', '種牡馬_芝_勝率'), ('ダ', '種牡馬_ダ_勝率')]:
        m = s_col == surf
        df.loc[m, '種牡馬コース特化'] = (pd.to_numeric(df.loc[m, scol], errors='coerce')
                                         - pd.to_numeric(df.loc[m, '種牡馬_勝率'], errors='coerce'))
if '1走前_クラス差' in df.columns:
    df['クラス降格恩恵'] = -pd.to_numeric(df['1走前_クラス差'], errors='coerce')
df['市場含意P'] = 0.75 / df['単勝オッズ'].clip(lower=1.0)
if '近5走_タイム指数平均' in df.columns:
    ti = pd.to_numeric(df['近5走_タイム指数平均'], errors='coerce')
    cm = df.groupby('クラス_rank')['近5走_タイム指数平均'].transform(
        lambda x: pd.to_numeric(x, errors='coerce').median())
    df['タイム指数クラス超過'] = ti - pd.to_numeric(cm, errors='coerce')

df['surf'] = df['芝・ダ'].str.strip() if '芝・ダ' in df.columns else ''
MAJOR4 = {'東', '中', '阪', '京'}

def gk(r):
    v = str(r.get('今回_会場', '')).strip()
    if v not in MAJOR4: return None
    s = r['surf']
    if s not in ('芝', 'ダ'): return None
    return f'{v}_{s}'

df['gk'] = df.apply(gk, axis=1)
df = df[df['gk'].notna()]
df['race_id'] = df['日付'].astype(str) + '_' + df['開催'].astype(str) + '_' + df['レース名'].astype(str)

FEATS = list(dict.fromkeys([c for c in [
    '距離', '頭数', '馬番', '斤量', '馬体重', '馬体重増減', '内外枠', '斤量変化', '間隔', '連闘フラグ', '休み明けフラグ', 'クラス_rank',
    '1走前_着順_num', '2走前_着順_num', '3走前_着順_num', '4走前_着順_num', '5走前_着順_num',
    '1走前_クラス_rank', '2走前_クラス_rank', '3走前_クラス_rank', '1走前_クラス差', '2走前_クラス差', '3走前_クラス差',
    '1走前_タイム指数', '2走前_タイム指数', '3走前_タイム指数', '4走前_タイム指数', '5走前_タイム指数',
    '1走前_上り3F', '2走前_上り3F', '3走前_上り3F', '1走前_4角', '2走前_4角', '3走前_4角',
    '1走前_単勝オッズ', '2走前_単勝オッズ', '3走前_単勝オッズ', '1走前_走破タイム_sec', '2走前_走破タイム_sec',
    '1走前_PCI', '2走前_PCI', '3走前_PCI', '1走前_脚質_num', '2走前_脚質_num',
    '1走前_頭数', '2走前_頭数', '1走前_馬番', '2走前_馬番',
    '近3走_平均着順', '近3走_勝率', '近3走_複勝率', '近5走_平均着順', '近5走_複勝率',
    '近5走_タイム指数平均', '近5走_上り3F平均', '近5走_タイム指数_std', '近5走_タイム指数_max', '近5走_タイム指数_min',
    '近5走_上り3F_min', '近5走_上り3F_std', '近5走_クラス調整_平均着順', '格上経験数_近5走', '近5走_着差タイム_クラス補正平均',
    '近走_改善トレンド', 'タイム指数_近3走_slope', '前走_追い上げ度', '前走_4角位置', '近5走_平均4角位置',
    '1走前_上り3F_指数', '2走前_上り3F_指数', '3走前_上り3F_指数',
    '脚質フィット', '展開フィット_v2', 'コース展開マッチ', '展開_コース_脚質フィット', 'レース内_逃げ馬数', 'レース内_先行馬数',
    'レース内_相対脚質', 'コース_先行有利度', '推定ペース',
    '騎手_r200_勝率', '騎手_r200_複勝率', '騎手コース_r100_勝率', '騎手コース_r100_複勝率',
    '騎手馬場_r100_勝率', '騎手馬場_r100_複勝率', '騎手距離_r100_勝率', '騎手距離_r100_複勝率', '騎手脚質_r100_勝率', '騎手脚質_r100_複勝率',
    '種牡馬_勝率', '種牡馬_複勝率', '種牡馬_芝_勝率', '種牡馬_ダ_勝率', '母父馬_勝率', '母父馬_複勝率', '産地_勝率', '生産者_勝率',
    '馬_r20_勝率', '馬_r20_複勝率', '馬コース_r20_勝率', '馬コース_r20_複勝率', '馬距離_勝率', '馬距離_複勝率',
    '同会場_平均着順_近5走', '同会場_複勝率_近5走', '同馬場_平均着順_近5走', '同距離帯_平均着順_近5走',
    '芝ダ一致_平均着順_近5走', '良馬場_平均着順_近5走', '道悪_平均着順_近5走',
    'キャリア', 'キャリア_log', '馬体重トレンド_近5走', '前走コース一致', '芝ダ転向', '距離変化_前走', '騎手変更', '乗替り_近走不振',
    '単勝オッズ', '市場含意P', '人気',
    '騎手コース特化', '会場親和性', '馬場適性超過', '種牡馬コース特化', 'クラス降格恩恵', 'タイム指数クラス超過',
] if c in df.columns]))
for c in FEATS: df[c] = pd.to_numeric(df[c], errors='coerce')

def roi_tan(sub):
    s = sub.dropna(subset=['単勝オッズ'])
    if len(s) == 0: return np.nan, 0, 0
    w = s[s['target_win'] == 1]
    return w['単勝オッズ'].sum() / len(s) - 1, len(w), len(s)

print('学習中...')
all_w = []
for key in sorted(df['gk'].unique()):
    g = df[df['gk'] == key].sort_values('日付_num')
    tr  = g[g['日付_num'] <= 181231]
    val = g[(g['日付_num'] > 181231) & (g['日付_num'] <= 201231)]
    te  = g[g['日付_num'] >= 210101]
    if len(tr) < 300 or len(te) < 200 or tr['target_win'].sum() < 30: continue
    feat = [c for c in FEATS if c in g.columns]
    clf = LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=63, min_child_samples=20,
        subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0,
        class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1)
    clf.fit(tr[feat].astype(float), tr['target_win'])
    raw = clf.predict_proba(te[feat].astype(float))[:, 1]
    if len(val) >= 100:
        rv = clf.predict_proba(val[feat].astype(float))[:, 1]
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(rv, val['target_win'].values)
        prob = iso.predict(raw)
    else:
        prob = raw
    tw = te.copy()
    tw['prob_win']  = prob
    tw['market_P']  = 0.75 / tw['単勝オッズ'].clip(lower=1.0)
    tw['edge']      = tw['prob_win'] - tw['market_P']
    tw['ev']        = tw['prob_win'] * tw['単勝オッズ']          # モデル期待値
    tw['rank_edge'] = tw.groupby('race_id')['edge'].rank(ascending=False, method='min')
    tw['rank_ev']   = tw.groupby('race_id')['ev'].rank(ascending=False, method='min')
    tw['gk'] = key
    all_w.append(tw)
    print(f'  {key}: tr={len(tr):,} val={len(val):,} OOS={len(te):,}')

oos = pd.concat(all_w, ignore_index=True)
oos['year'] = oos['日付_num'].astype(str).str[:2]

SEP = '=' * 65

# ---- EVランク1位 vs edgeランク1位 比較 ----
print(f'\n{SEP}')
print(' EV1位 vs edge1位  全グループ比較')
print(SEP)
print(f'  {"グループ":<8}  {"edge1位_ROI":>11}  {"EV1位_ROI":>10}  {"edge1位_N":>9}  {"EV1位_N":>8}')
print(f'  {"-"*8}  {"-"*11}  {"-"*10}  {"-"*9}  {"-"*8}')
for key in sorted(oos['gk'].unique()):
    g = oos[oos['gk'] == key]
    re1, _, ne1 = roi_tan(g[g['rank_edge'] == 1])
    rv1, _, nv1 = roi_tan(g[g['rank_ev']   == 1])
    print(f'  {key:<8}  {re1:>+11.1%}  {rv1:>+10.1%}  {ne1:>9,}  {nv1:>8,}')

# ---- EV閾値フィルタ（rank_edge=1の中でEV≥X） ----
print(f'\n{SEP}')
print(' EV閾値フィルタ: rank_edge=1 かつ ev≥X  全グループ合計')
print(SEP)
r1e = oos[oos['rank_edge'] == 1]
print(f'  {"ev閾値":<8}  {"N":>6}  {"年間N":>6}  {"的中率":>6}  {"ROI":>8}')
print(f'  {"-"*8}  {"-"*6}  {"-"*6}  {"-"*6}  {"-"*8}')
for thr in [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5]:
    sub = r1e[r1e['ev'] >= thr]
    r, w, n = roi_tan(sub)
    if n < 20: continue
    nyear = n / 5.5
    print(f'  ev≥{thr:.1f}:  {n:>6,}  {nyear:>6.0f}  {sub["target_win"].mean():>6.1%}  {r:>+8.1%}')

# ---- グループ別 EV閾値スイープ ----
print(f'\n{SEP}')
print(' グループ別: rank_edge=1 × ev閾値')
print(SEP)
for thr in [0.8, 1.0, 1.2]:
    print(f'\n  [ev≥{thr:.1f}]')
    print(f'  {"グループ":<8}  {"N":>5}  {"的中率":>6}  {"ROI":>8}')
    for key in sorted(oos['gk'].unique()):
        sub = oos[(oos['gk'] == key) & (oos['rank_edge'] == 1) & (oos['ev'] >= thr)]
        r, w, n = roi_tan(sub)
        if n < 20: continue
        print(f'  {key:<8}  {n:>5,}  {sub["target_win"].mean():>6.1%}  {r:>+8.1%}')

# ---- 全体合計 年別ROI: edge1位 × ev閾値 ----
print(f'\n{SEP}')
print(' 全グループ合計 年別ROI: rank_edge=1 × ev閾値別')
print(SEP)
for thr in [0.8, 1.0, 1.2]:
    sub_all = oos[(oos['rank_edge'] == 1) & (oos['ev'] >= thr)]
    r_tot, _, n_tot = roi_tan(sub_all)
    print(f'\n  [ev≥{thr:.1f}] 合計N={n_tot:,}  全体ROI={r_tot:>+7.1%}')
    for yr in sorted(sub_all['year'].unique()):
        sub = sub_all[sub_all['year'] == yr]
        r, w, n = roi_tan(sub)
        cumr = roi_tan(sub_all[sub_all['year'] <= yr])[0]
        print(f'    20{yr}: N={n:>4,}  的中={sub["target_win"].mean():.1%}  ROI={r:>+7.1%}  累計={cumr:>+7.1%}')

# ---- ev1位 かつ ev≥X 年別 ----
print(f'\n{SEP}')
print(' 全グループ合計 年別ROI: rank_ev=1 × ev閾値別')
print(SEP)
r1v = oos[oos['rank_ev'] == 1]
for thr in [0.8, 1.0, 1.2]:
    sub_all = r1v[r1v['ev'] >= thr]
    r_tot, _, n_tot = roi_tan(sub_all)
    print(f'\n  [ev≥{thr:.1f}] 合計N={n_tot:,}  全体ROI={r_tot:>+7.1%}')
    for yr in sorted(sub_all['year'].unique()):
        sub = sub_all[sub_all['year'] == yr]
        r, w, n = roi_tan(sub)
        cumr = roi_tan(sub_all[sub_all['year'] <= yr])[0]
        print(f'    20{yr}: N={n:>4,}  的中={sub["target_win"].mean():.1%}  ROI={r:>+7.1%}  累計={cumr:>+7.1%}')

# ---- オッズ帯 × ev閾値 ----
print(f'\n{SEP}')
print(' 全グループ合計: rank_edge=1 × オッズ帯 × ev閾値')
print(SEP)
for lo, hi in [(3, 10), (5, 15), (10, 30), (3, 15)]:
    sub_o = r1e[(r1e['単勝オッズ'] >= lo) & (r1e['単勝オッズ'] < hi)]
    print(f'\n  [{lo}-{hi}倍]')
    for thr in [0.7, 0.8, 1.0, 1.2]:
        sub = sub_o[sub_o['ev'] >= thr]
        r, w, n = roi_tan(sub)
        if n < 30: continue
        print(f'    ev≥{thr:.1f}: N={n:>5,}  的中={sub["target_win"].mean():.1%}  ROI={r:>+7.1%}')
