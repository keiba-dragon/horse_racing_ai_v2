# -*- coding: utf-8 -*-
"""roi_search8.py - 特徴量選択 + Optunaハイパーパラメータ最適化"""
import sys, io, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')
import os
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ─── データ読み込み（roi_search7.pyと同一前処理） ───────────────────────
df = pd.read_parquet('C:/horse_racing_ai/data/processed/all_venues_features.parquet')
df['着順_num'] = pd.to_numeric(df['着順_num'], errors='coerce')
df = df[df['着順_num'].notna()].copy()
df['target_win'] = (df['着順_num'] == 1).astype(int)
for c in ['単勝オッズ', '人気']: df[c] = pd.to_numeric(df[c], errors='coerce')

BABA_MAP = {'良': 0, '稍': 1, '稍重': 1, '重': 2, '不': 3, '不良': 3}
for pfx in ['前走', '1走前_', '2走前_', '3走前_']:
    col = pfx + '馬場状態' if pfx == '前走' else pfx + '馬場状態'
    new = pfx + '馬場_num'
    if col in df.columns:
        df[new] = df[col].map(BABA_MAP)

for col_a, col_b, new in [
    ('騎手コース_r100_勝率',   '騎手_r200_勝率',   '騎手コース特化'),
    ('同会場_複勝率_近5走',    '近5走_複勝率',     '会場親和性'),
    ('近5走_平均着順',         '芝ダ一致_平均着順_近5走', '馬場適性超過'),
    ('調教師コース_r100_勝率', '調教師_r200_勝率', '調教師コース特化'),
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

df['race_id'] = df['日付'].astype(str) + '_' + df['開催'].astype(str) + '_' + df['レース名'].astype(str)
df['斤量_相対'] = df.groupby('race_id')['斤量'].transform(
    lambda x: pd.to_numeric(x, errors='coerce') - pd.to_numeric(x, errors='coerce').mean())

df['surf'] = df['芝・ダ'].str.strip() if '芝・ダ' in df.columns else ''
ALL10 = {'東', '中', '阪', '京', '新', '小', '福', '名', '札', '函'}

def gk(r):
    v = str(r.get('今回_会場', '')).strip()
    if v not in ALL10: return None
    s = r['surf']
    if s not in ('芝', 'ダ'): return None
    return f'{v}_{s}'

df['gk'] = df.apply(gk, axis=1)
df = df[df['gk'].notna()]

ALL_FEATS = list(dict.fromkeys([c for c in [
    '距離', '今回_距離_m', '頭数', '馬番', '斤量', '斤量_相対', '馬体重', '馬体重増減',
    '内外枠', '斤量変化', '間隔', '連闘フラグ', '休み明けフラグ', 'クラス_rank',
    '年齢', '性別_num', '所属_num', 'キャリア', 'キャリア_log', 'キャリア_浅い',
    '今回_馬場_num', '月', '季節',
    'ブリンカー_装着', '前走ブリンカー_装着', 'ブリンカー変更',
    '1走前_着順_num', '1走前_クラス_rank', '1走前_クラス差', '1走前_タイム指数',
    '1走前_上り3F', '1走前_上り3F_指数', '1走前_4角', '1走前_単勝オッズ', '1走前_走破タイム_sec',
    '1走前_PCI', '1走前_脚質_num', '1走前_頭数', '1走前_馬番', '1走前_斤量',
    '1走前_馬体重', '1走前_馬体重増減', '1走前_間隔', '1走前_前走着差タイム', '着差タイム_クラス補正_1走前',
    '1走前_クラス調整着順', '前走馬場_num', '前走_surface', '前走_距離_m', '前走人気',
    '2走前_着順_num', '2走前_クラス_rank', '2走前_クラス差', '2走前_タイム指数',
    '2走前_上り3F', '2走前_上り3F_指数', '2走前_4角', '2走前_単勝オッズ', '2走前_走破タイム_sec',
    '2走前_PCI', '2走前_脚質_num', '2走前_頭数', '2走前_馬番', '2走前_斤量',
    '2走前_馬体重', '2走前_馬体重増減', '2走前_間隔', '2走前_前走着差タイム', '着差タイム_クラス補正_2走前',
    '2走前_クラス調整着順',
    '3走前_着順_num', '3走前_クラス_rank', '3走前_クラス差', '3走前_タイム指数',
    '3走前_上り3F', '3走前_上り3F_指数', '3走前_4角', '3走前_単勝オッズ', '3走前_走破タイム_sec',
    '3走前_PCI', '3走前_脚質_num', '3走前_頭数', '3走前_馬番', '3走前_斤量',
    '3走前_馬体重', '3走前_馬体重増減', '3走前_間隔', '3走前_前走着差タイム', '着差タイム_クラス補正_3走前',
    '3走前_クラス調整着順',
    '4走前_着順_num', '4走前_クラス差', '4走前_タイム指数', '4走前_上り3F', '4走前_上り3F_指数',
    '4走前_4角', '4走前_単勝オッズ', '4走前_走破タイム_sec', '4走前_脚質_num', '4走前_頭数', '4走前_馬番',
    '5走前_着順_num', '5走前_クラス差', '5走前_タイム指数', '5走前_上り3F', '5走前_上り3F_指数',
    '5走前_4角', '5走前_単勝オッズ', '5走前_走破タイム_sec', '5走前_脚質_num', '5走前_頭数', '5走前_馬番',
    '6走前_着順_num', '6走前_タイム指数', '6走前_上り3F', '6走前_単勝オッズ', '6走前_4角',
    '7走前_着順_num', '7走前_タイム指数', '7走前_上り3F', '7走前_単勝オッズ', '7走前_4角',
    '8走前_着順_num', '8走前_タイム指数', '8走前_上り3F', '8走前_単勝オッズ',
    '9走前_着順_num', '9走前_タイム指数', '9走前_上り3F',
    '10走前_着順_num', '10走前_タイム指数',
    '近3走_平均着順', '近3走_勝率', '近3走_複勝率', '近3走_体重増減合計',
    '近5走_平均着順', '近5走_複勝率', '近5走_タイム指数平均', '近5走_上り3F平均',
    '近5走_タイム指数_std', '近5走_タイム指数_max', '近5走_タイム指数_min', '近5走_タイム指数_range',
    '近5走_上り3F_min', '近5走_上り3F_std', '近5走_上り3F指数平均',
    '近5走_クラス調整_平均着順', '近5走_クラス補正スコア', '近5走_平均相対着順',
    '格上経験数_近5走', '最大クラス差_近5走', '近5走_着差タイム_クラス補正平均', '近5走_走破タイム平均',
    '近10走_平均着順', '近10走_勝率', '近10走_複勝率',
    '近走_改善トレンド', 'タイム指数_近3走_slope', '前走_追い上げ度', '前走_4角位置', '近5走_平均4角位置',
    '馬体重トレンド_近5走',
    '脚質フィット', '展開フィット_v2', 'コース展開マッチ', '展開_コース_脚質フィット',
    'レース内_逃げ馬数', 'レース内_先行馬数', 'レース内_相対脚質', 'レース内_平均脚質', 'レース内_脚質std',
    'コース_先行有利度', '推定ペース', '推定_脚質率',
    '相手レベル_平均着順', '相手レベル_実力差',
    '騎手_r200_勝率', '騎手_r200_複勝率', '騎手コース_r100_勝率', '騎手コース_r100_複勝率',
    '騎手馬場_r100_勝率', '騎手馬場_r100_複勝率', '騎手距離_r100_勝率', '騎手距離_r100_複勝率',
    '騎手脚質_r100_勝率', '騎手脚質_r100_複勝率', '騎手_平均着順',
    '調教師_r200_勝率', '調教師_r200_複勝率', '調教師コース_r100_勝率', '調教師コース_r100_複勝率',
    '種牡馬_勝率', '種牡馬_複勝率', '種牡馬_芝_勝率', '種牡馬_ダ_勝率',
    '種牡馬_芝_複勝率', '種牡馬_ダ_複勝率',
    '母父馬_勝率', '母父馬_複勝率', '産地_勝率', '産地_複勝率', '生産者_勝率', '生産者_複勝率',
    '馬_r20_勝率', '馬_r20_複勝率', '馬コース_r20_勝率', '馬コース_r20_複勝率',
    '馬距離_勝率', '馬距離_複勝率',
    '同会場_平均着順_近5走', '同会場_複勝率_近5走', '同会場_出走数_近5走',
    '同馬場_平均着順_近5走', '同距離帯_平均着順_近5走', '芝ダ一致_平均着順_近5走',
    '芝ダ一致数_近5走', '良馬場_平均着順_近5走', '道悪_平均着順_近5走', '馬場適性差_近5走',
    '距離短縮時_平均着順_近5走', '距離延長時_平均着順_近5走',
    '前走コース一致', '芝ダ転向', '距離変化_前走', '騎手変更', '乗替り_近走不振',
    '単勝オッズ', '市場含意P', '人気',
    '騎手コース特化', '会場親和性', '馬場適性超過', '種牡馬コース特化',
    'クラス降格恩恵', 'タイム指数クラス超過', '調教師コース特化', '斤量_相対',
] if c in df.columns]))
for c in ALL_FEATS: df[c] = pd.to_numeric(df[c], errors='coerce')
print(f'全特徴量数: {len(ALL_FEATS)}')

# ─── Phase 1 & 2: 保存済みならスキップ ─────────────────────────────────
import json as _json
_params_path = 'C:/horse_racing_ai/data/processed/best_params_v2.json'

BASE_PARAMS = dict(n_estimators=800, learning_rate=0.02, num_leaves=63,
    min_child_samples=20, subsample=0.8, colsample_bytree=0.7,
    reg_alpha=0.1, reg_lambda=1.0, class_weight='balanced',
    random_state=42, n_jobs=-1, verbose=-1)

if os.path.exists(_params_path):
    with open(_params_path, encoding='utf-8') as _f:
        _saved = _json.load(_f)
    selected_feats = _saved['selected_feats']
    best_params_all = _saved['best_params']
    print(f'=== Phase 1 & 2 スキップ（保存済みパラメータ読み込み）===')
    print(f'  特徴量数: {len(selected_feats)}  グループ数: {len(best_params_all)}')
else:
    print('\n=== Phase 1: 特徴量重要度収集 ===')
    imp_all = pd.Series(0.0, index=ALL_FEATS)
    for key in sorted(df['gk'].unique()):
        g = df[df['gk'] == key].sort_values('日付_num')
        tr = g[g['日付_num'] <= 201231]
        if len(tr) < 300 or tr['target_win'].sum() < 30: continue
        feat = [c for c in ALL_FEATS if c in g.columns]
        clf = LGBMClassifier(**BASE_PARAMS)
        clf.fit(tr[feat].astype(float), tr['target_win'])
        imp = pd.Series(clf.feature_importances_, index=feat)
        imp_all = imp_all.add(imp, fill_value=0)
        print(f'  {key}: 学習完了')

    imp_all = imp_all.sort_values(ascending=False)
    TOP_N = 120
    selected_feats = imp_all.head(TOP_N).index.tolist()
    print(f'\n特徴量選択: {len(ALL_FEATS)} → {len(selected_feats)} (上位{TOP_N})')
    print('上位20特徴量:')
    for f, v in imp_all.head(20).items():
        print(f'  {f}: {v:.0f}')
    print('下位10特徴量（削除対象）:')
    for f, v in imp_all.tail(10).items():
        print(f'  {f}: {v:.0f}')

# ─── Phase 2: Optuna チューニング ──────────────────────────────────────
def roi_tan(sub):
    s = sub.dropna(subset=['単勝オッズ'])
    if len(s) == 0: return np.nan, 0, 0
    w = s[s['target_win'] == 1]
    return w['単勝オッズ'].sum() / len(s) - 1, len(w), len(s)

N_TRIALS = 40

print(f'\n=== Phase 2: Optunaチューニング (各グループ{N_TRIALS}試行) ===')
best_params_all = {}

for key in sorted(df['gk'].unique()):
    g = df[df['gk'] == key].sort_values('日付_num')
    tr  = g[g['日付_num'] <= 201231]
    val = g[(g['日付_num'] > 201231) & (g['日付_num'] <= 221231)]
    te  = g[g['日付_num'] >= 230101]
    if len(tr) < 300 or len(te) < 200 or tr['target_win'].sum() < 30: continue
    feat = [c for c in selected_feats if c in g.columns]

    def objective(trial):
        params = dict(
            n_estimators     = trial.suggest_int('n_estimators', 400, 1500),
            learning_rate    = trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
            num_leaves       = trial.suggest_int('num_leaves', 20, 80),
            min_child_samples= trial.suggest_int('min_child_samples', 20, 100),
            subsample        = trial.suggest_float('subsample', 0.6, 1.0),
            colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0),
            reg_alpha        = trial.suggest_float('reg_alpha', 0.01, 3.0, log=True),
            reg_lambda       = trial.suggest_float('reg_lambda', 0.1, 5.0, log=True),
            class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1,
        )
        clf = LGBMClassifier(**params)
        clf.fit(tr[feat].astype(float), tr['target_win'])
        from sklearn.metrics import log_loss
        rv = clf.predict_proba(val[feat].astype(float))[:, 1]
        return -log_loss(val['target_win'].values, rv)  # logloss最小化（安定）

    study = optuna.create_study(direction='maximize',  # loglossはマイナスなので最大化
        sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    best_params_all[key] = study.best_params
    print(f'  {key}: best_logloss={-study.best_value:.4f}  '
          f'(leaves={study.best_params["num_leaves"]}, '
          f'lr={study.best_params["learning_rate"]:.3f}, '
          f'min_child={study.best_params["min_child_samples"]})')

    # 完了次第すぐ保存（途中クラッシュ対策）
    with open(_params_path, 'w', encoding='utf-8') as _f:
        _json.dump({'best_params': best_params_all, 'selected_feats': selected_feats}, _f, ensure_ascii=False, indent=2)

# ─── Phase 3: 最終OOS評価 ────────────────────────────────────────────────
print('\n=== Phase 3: OOS最終評価 ===')
all_w = []
for key in sorted(df['gk'].unique()):
    g = df[df['gk'] == key].sort_values('日付_num')
    tr  = g[g['日付_num'] <= 201231]
    val = g[(g['日付_num'] > 201231) & (g['日付_num'] <= 221231)]
    te  = g[g['日付_num'] >= 230101]
    if len(tr) < 300 or len(te) < 200 or tr['target_win'].sum() < 30: continue
    feat = [c for c in selected_feats if c in g.columns]

    bp = best_params_all.get(key, {})
    params = {**BASE_PARAMS, **bp}
    clf = LGBMClassifier(**params)
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
    tw['rank_edge'] = tw.groupby('race_id')['edge'].rank(ascending=False, method='min')
    tw['gk'] = key
    all_w.append(tw)
    print(f'  {key}: tr={len(tr):,} val={len(val):,} OOS={len(te):,}  feat={len(feat)}')

oos = pd.concat(all_w, ignore_index=True)
oos['year'] = oos['日付_num'].astype(str).str[:2]

# OOSデータを保存（選別分析用）
save_cols = ['日付_num', 'year', 'gk', 'race_id', '馬名S', '着順_num', 'target_win',
             '単勝オッズ', '人気', '頭数', '今回_馬場_num', 'クラス_rank',
             'prob_win', 'market_P', 'edge', 'rank_edge']
oos_save = oos[[c for c in save_cols if c in oos.columns]]
oos_save.to_parquet('C:/horse_racing_ai/data/processed/oos_predictions.parquet', index=False)
print(f'OOS予測データ保存: {len(oos_save):,}行')

# ── Val予測も保存（条件探索用・OOSを汚さないため） ──
print('\n=== Val予測データ生成・保存 ===')
all_val = []
for key in sorted(df['gk'].unique()):
    g = df[df['gk'] == key].sort_values('日付_num')
    tr  = g[g['日付_num'] <= 201231]
    val = g[(g['日付_num'] > 201231) & (g['日付_num'] <= 221231)]
    if len(tr) < 300 or len(val) < 100 or tr['target_win'].sum() < 30: continue
    feat = [c for c in selected_feats if c in g.columns]

    bp = best_params_all.get(key, {})
    params = {**BASE_PARAMS, **bp}
    clf = LGBMClassifier(**params)
    clf.fit(tr[feat].astype(float), tr['target_win'])
    raw = clf.predict_proba(val[feat].astype(float))[:, 1]
    # valのIsotonic: val前半でfit → val後半に適用（val内リーク防止）
    val_sorted = val.copy()
    mid = len(val_sorted) // 2
    v1 = val_sorted.iloc[:mid]
    v2 = val_sorted.iloc[mid:]
    r1 = clf.predict_proba(v1[feat].astype(float))[:, 1]
    r2 = clf.predict_proba(v2[feat].astype(float))[:, 1]
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(r1, v1['target_win'].values)
    prob2 = iso.predict(r2)
    # val前半はキャリブレーション前の確率を使用
    prob1 = r1
    vw = val_sorted.copy()
    vw['prob_win'] = np.concatenate([prob1, prob2])
    vw['market_P']  = 0.75 / vw['単勝オッズ'].clip(lower=1.0)
    vw['edge']      = vw['prob_win'] - vw['market_P']
    vw['rank_edge'] = vw.groupby('race_id')['edge'].rank(ascending=False, method='min')
    vw['gk'] = key
    all_val.append(vw)

val_df = pd.concat(all_val, ignore_index=True)
val_df['year'] = val_df['日付_num'].astype(str).str[:2]
val_save = val_df[[c for c in save_cols if c in val_df.columns]]
val_save.to_parquet('C:/horse_racing_ai/data/processed/val_predictions.parquet', index=False)
print(f'Val予測データ保存: {len(val_save):,}行  ({val_save["year"].min()}〜{val_save["year"].max()})')

SEP = '=' * 65
print(f'\n{SEP}')
print(' グループ別サマリー (OOS 2023-2026, edge1位)')
print(SEP)
print(f'  {"グループ":<8}  {"N":>6}  {"的中率":>6}  {"ROI":>8}  {"e≥0.02_ROI":>10}')
for key in sorted(oos['gk'].unique()):
    g = oos[oos['gk'] == key]
    r1e = g[g['rank_edge'] == 1]
    r, w, n = roi_tan(r1e)
    r2, _, n2 = roi_tan(r1e[r1e['edge'] >= 0.02])
    print(f'  {key:<8}  {n:>6,}  {r1e["target_win"].mean():>6.1%}  {r:>+8.1%}  {r2:>+10.1%}')

print(f'\n{SEP}')
print(' 全グループ合計 年別ROI (edge1位)')
print(SEP)
r1_all = oos[oos['rank_edge'] == 1]
for yr in sorted(r1_all['year'].unique()):
    sub = r1_all[r1_all['year'] == yr]
    r, w, n = roi_tan(sub)
    cumr = roi_tan(r1_all[r1_all['year'] <= yr])[0]
    print(f'  20{yr}: N={n:>5,}  的中={sub["target_win"].mean():.1%}  ROI={r:>+7.1%}  累計={cumr:>+7.1%}')
r, w, n = roi_tan(r1_all)
print(f'  合計:  N={n:>5,}  ROI={r:>+7.1%}')
