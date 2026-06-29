# -*- coding: utf-8 -*-
"""主要4会場（東・中・阪・京）× 芝ダ = 8グループ 詳細分析"""
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
df['複勝倍率'] = pd.to_numeric(df['複勝配当'], errors='coerce') / 100.0

# 市場特化特徴量
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

# 距離帯
def dist_band(s):
    m = re.search(r'(\d{3,4})', str(s))
    if not m: return None
    v = int(m.group(1))
    if v <= 1400: return '短'
    if v <= 1800: return 'マイ'
    if v <= 2200: return '中'
    return '長'
df['距離帯'] = df['距離'].apply(dist_band)
df['surf'] = df['芝・ダ'].str.strip() if '芝・ダ' in df.columns else ''

# 主要4会場 × 芝ダ グループ
MAJOR4 = {'東', '中', '阪', '京'}
VENUE_NAME = {'東': '東京', '中': '中山', '阪': '阪神', '京': '京都'}

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
    tw['prob_win'] = prob
    tw['market_P'] = 0.75 / tw['単勝オッズ'].clip(lower=1.0)
    tw['edge']     = tw['prob_win'] - tw['market_P']
    tw['rank_edge'] = tw.groupby('race_id')['edge'].rank(ascending=False, method='min')
    tw['gk'] = key
    all_w.append(tw)
    print(f'  {key}: tr={len(tr):,} val={len(val):,} OOS={len(te):,}')

oos_w = pd.concat(all_w, ignore_index=True)
oos_w['year'] = oos_w['日付_num'].astype(str).str[:2]

SEP = '=' * 65

# グループ別サマリー
print(f'\n{SEP}')
print(' グループ別サマリー (OOS 2021-2026, edge1位)')
print(SEP)
print(f'  {"グループ":<8}  {"OOS_N":>7}  {"的中率":>6}  {"ROI":>8}  {"edge≥0.02_N":>11}  {"e≥0.02_ROI":>10}')
print(f'  {"-"*8}  {"-"*7}  {"-"*6}  {"-"*8}  {"-"*11}  {"-"*10}')
for key in sorted(oos_w['gk'].unique()):
    g = oos_w[oos_w['gk'] == key]
    r1e = g[g['rank_edge'] == 1]
    r, w, n = roi_tan(r1e)
    r2, w2, n2 = roi_tan(r1e[r1e['edge'] >= 0.02])
    hr = r1e['target_win'].mean() if len(r1e) > 0 else np.nan
    print(f'  {key:<8}  {n:>7,}  {hr:>6.1%}  {r:>+8.1%}  {n2:>11,}  {r2:>+10.1%}')

# 各グループの詳細
for key in sorted(oos_w['gk'].unique()):
    g = oos_w[oos_w['gk'] == key]
    r1e = g[g['rank_edge'] == 1]
    print(f'\n{SEP}')
    print(f' [{key}] (OOS N={len(g):,}  edge1位 N={len(r1e):,})')
    print(SEP)

    print('  年別 edge1位 ROI:')
    for yr in sorted(g['year'].unique()):
        sub = r1e[r1e['year'] == yr]
        r, w, n = roi_tan(sub)
        cumr = roi_tan(r1e[r1e['year'] <= yr])[0]
        print(f'    20{yr}: N={n:>4,}  的中={sub["target_win"].mean():.1%}  ROI={r:>+7.1%}  累計={cumr:>+7.1%}')

    print()
    print('  edge閾値別:')
    for thr in [0.0, 0.01, 0.02, 0.03, 0.05]:
        sub = r1e[r1e['edge'] >= thr]
        r, w, n = roi_tan(sub)
        if n < 30: continue
        print(f'    edge≥{thr:.2f}: N={n:>5,}  的中={sub["target_win"].mean():.1%}  ROI={r:>+7.1%}')

    print()
    print('  オッズ帯別 (edge1位):')
    for lo, hi in [(2, 5), (5, 10), (10, 20), (20, 50), (5, 15), (5, 30)]:
        sub = r1e[(r1e['単勝オッズ'] >= lo) & (r1e['単勝オッズ'] < hi)]
        r, w, n = roi_tan(sub)
        if n < 30: continue
        print(f'    [{lo:>2}-{hi:<3}倍]: N={n:>5,}  的中={sub["target_win"].mean():.1%}  ROI={r:>+7.1%}')

    print()
    print('  人気別 (edge1位):')
    for pop in range(1, 10):
        sub = r1e[r1e['人気'] == pop]
        r, w, n = roi_tan(sub)
        if n < 20: continue
        print(f'    人気{pop}番: N={n:>4,}  的中={sub["target_win"].mean():.1%}  ROI={r:>+7.1%}')

    print()
    print('  距離帯別 (edge1位):')
    for db in ['短', 'マイ', '中', '長']:
        sub = r1e[r1e['距離帯'] == db]
        r, w, n = roi_tan(sub)
        if n < 30: continue
        print(f'    {db}距離: N={n:>4,}  的中={sub["target_win"].mean():.1%}  ROI={r:>+7.1%}')

# 全体サマリー
print(f'\n{SEP}')
print(' 全8グループ合計 年別ROI (edge1位)')
print(SEP)
r1_all = oos_w[oos_w['rank_edge'] == 1]
for yr in sorted(r1_all['year'].unique()):
    sub = r1_all[r1_all['year'] == yr]
    r, w, n = roi_tan(sub)
    print(f'  20{yr}: N={n:>5,}  的中={sub["target_win"].mean():.1%}  ROI={r:>+7.1%}')
r, w, n = roi_tan(r1_all)
print(f'  合計:  N={n:>5,}  ROI={r:>+7.1%}')

print(f'\n{SEP}')
print(' 全8グループ合計 edge≥0.02 年別ROI')
print(SEP)
r1e2 = oos_w[(oos_w['rank_edge'] == 1) & (oos_w['edge'] >= 0.02)]
for yr in sorted(r1e2['year'].unique()):
    sub = r1e2[r1e2['year'] == yr]
    r, w, n = roi_tan(sub)
    print(f'  20{yr}: N={n:>5,}  的中={sub["target_win"].mean():.1%}  ROI={r:>+7.1%}')
r, w, n = roi_tan(r1e2)
print(f'  合計:  N={n:>5,}  ROI={r:>+7.1%}')
