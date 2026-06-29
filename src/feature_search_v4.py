# coding: utf-8
"""
feature_search v4: 訓練窓の最適化 + val-basedな特徴量選択
- 問題: v3はOOS(2023-24)で特徴量選択 → 2025で過学習
- 改善:
  A. 訓練窓を短く (2018-2022) → 最近のパターンを重視
  B. 特徴量選択をval(2022)のROIで行う → OOSは評価のみ
  C. OOS を 2023-24(評価) と 2025(真のホールドアウト) に分けて報告
"""
import sys, os, json, datetime
import numpy as np
import pandas as pd

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from save_conditional_logit import prepare, segment_softmax, BASE_DIR, DATA_FILE, LR, N_EPOCHS, PATIENCE

LOG_FILE = os.path.join(BASE_DIR, 'logs', 'feature_search_v4.jsonl')
os.makedirs(os.path.join(BASE_DIR, 'logs'), exist_ok=True)

# v303ベース特徴量（芝_長距離はベースライン維持版）
BASE_FEATURES = [
    '1走前_タイム指数', '1走前_上り3F', '前走着差タイム', '1走前_RPCI',
    '近5走_タイム指数平均', '近5走_タイム指数_max', 'タイム指数_近3走_slope',
    '1走前_クラス調整着順', '近5走_クラス調整_平均着順',
    '馬番', 'コース枠_r200_勝率',
    '騎手コース_r100_勝率', '騎手変更', '調教師コース_r100_勝率',
    'コース脚質_r200_勝率', '1走前_脚質_num', '種牡馬_勝率', '種牡馬_ダ_勝率',
    '斤量', '芝ダ転向', '間隔',
]

# v303採用済みの追加特徴量（セグメント別スタート地点）
SEG_INIT = {
    '芝_短距離':  BASE_FEATURES[:-3] + [f for f in BASE_FEATURES[-3:] if f != 'コース枠_r200_勝率' and f != '騎手変更'] + ['1走前_3角'],
    '芝_中距離':  BASE_FEATURES + ['馬体重', '間隔_短_flag', '血統_ダ優位度', '馬体重増減'],
    '芝_長距離':  list(BASE_FEATURES),
    'ダ_短距離':  [c for c in BASE_FEATURES if c != '芝ダ転向'] + ['馬体重増減', '展開フィット_v2', '乗替り_近走不振', '間隔_長_flag'],
    'ダ_中長距離': [c for c in BASE_FEATURES if c not in ('タイム指数_近3走_slope', 'コース脚質_r200_勝率', '芝ダ転向')] + ['間隔_長_flag', '1走前_上3F地点差'],
}

# 追加候補（新規 + v3で未採用）
ADD_CANDIDATES = [
    '1走前_3角', '馬体重', '間隔_短_flag', '血統_ダ優位度', '馬体重増減',
    '1走前_上3F地点差', '間隔_長_flag', '展開フィット_v2', '乗替り_近走不振',
    'キャリア', '母父馬_勝率', '近5走_上り3F指数平均', '近走連続入着数',
    '前走着順_比', 'タイム指数_不安定度', '騎手調教師_r100_勝率',
    '近3走_複勝率', '同会場_複勝率_近5走', '芝ダ一致数_近5走',
]

SEGMENTS_5 = [('芝', '短距離'), ('芝', '中距離'), ('芝', '長距離'),
              ('ダ', '短距離'), ('ダ', '中長距離')]

# 訓練窓のバリエーション
TRAIN_WINDOWS = [
    (130101, 220101, 220101, 221231),   # 従来: 2013-2022
    (180101, 220101, 220101, 221231),   # 短縮: 2018-2022
    (150101, 220101, 220101, 221231),   # 中間: 2015-2022
]


def add_computed_features(df):
    interval = pd.to_numeric(df['間隔'], errors='coerce') if '間隔' in df.columns else pd.Series(np.nan, index=df.index)
    df['間隔_長_flag'] = (interval >= 60).astype(float)
    df['間隔_短_flag'] = (interval <= 14).astype(float)
    heads = pd.to_numeric(df.get('1走前_頭数', np.nan), errors='coerce').clip(lower=1)
    df['前走着順_比'] = pd.to_numeric(df.get('1走前_着順_num', np.nan), errors='coerce') / heads
    da_r  = pd.to_numeric(df.get('種牡馬_ダ_勝率', np.nan), errors='coerce')
    all_r = pd.to_numeric(df.get('種牡馬_勝率',    np.nan), errors='coerce')
    df['血統_ダ優位度'] = da_r - all_r
    t5max  = pd.to_numeric(df.get('近5走_タイム指数_max',  np.nan), errors='coerce')
    t5mean = pd.to_numeric(df.get('近5走_タイム指数平均',  np.nan), errors='coerce')
    df['タイム指数_不安定度'] = t5max - t5mean
    return df


def load_data():
    print(f'データ読み込み: {DATA_FILE}')
    df = pd.read_parquet(DATA_FILE)
    df['日付_num'] = pd.to_numeric(df['日付'], errors='coerce')
    df['着順_num'] = pd.to_numeric(df['着順_num'], errors='coerce')
    df = df.dropna(subset=['日付_num', '着順_num'])
    df = df[df['着順_num'] < 99]
    df['race_id'] = (df['日付_num'].astype(int).astype(str) + '_' +
                     df['開催'].astype(str).str.strip() + '_' +
                     df['Ｒ'].astype(str).str.strip())
    df = df[df['開催'].notna()].copy()
    df['surface'] = df['距離'].astype(str).str.strip().str.extract(r'^([芝ダ])')[0].fillna('不明')
    df = df[df['surface'].isin(['芝', 'ダ'])].copy()
    if 'クラス_rank' in df.columns:
        df = df[df['クラス_rank'] != 1.0].copy()
    dm = pd.to_numeric(df['距離'].astype(str).str.extract(r'(\d+)')[0], errors='coerce')
    df['dist_m'] = dm
    shi, da = df['surface'] == '芝', df['surface'] == 'ダ'
    df['dist_band'] = ''
    df.loc[shi & (dm <= 1400),               'dist_band'] = '短距離'
    df.loc[shi & (dm > 1400) & (dm <= 2000), 'dist_band'] = '中距離'
    df.loc[shi & (dm > 2000),                'dist_band'] = '長距離'
    df.loc[da  & (dm <= 1400),               'dist_band'] = '短距離'
    df.loc[da  & (dm > 1400),                'dist_band'] = '中長距離'
    df = add_computed_features(df)
    print(f'有効データ: {len(df):,}行')
    return df


def _loss_grad(beta, X, y, gs, n, nr, alpha=1.0):
    probs = segment_softmax(X @ beta, gs, n)
    log_lik = np.sum(y * np.log(np.clip(probs, 1e-15, 1.0)))
    res = y - probs
    return (-log_lik + alpha * np.sum(beta**2)) / nr, (-(X.T @ res) + 2*alpha*beta) / nr


def adam_opt(X_tr, y_tr, gs_tr, n_tr, nr_tr, X_va, y_va, gs_va, n_va, nr_va, alpha=1.0):
    d = X_tr.shape[1]
    beta, m, v = np.zeros(d), np.zeros(d), np.zeros(d)
    b1, b2, eps = 0.9, 0.999, 1e-8
    t, best_val, best_beta, no_imp = 0, np.inf, np.zeros(d), 0
    for epoch in range(1, N_EPOCHS + 1):
        _, grad = _loss_grad(beta, X_tr, y_tr, gs_tr, n_tr, nr_tr, alpha)
        t += 1
        m = b1*m + (1-b1)*grad
        v = b2*v + (1-b2)*grad**2
        beta -= LR * (m/(1-b1**t)) / (np.sqrt(v/(1-b2**t)) + eps)
        if epoch % 10 == 0:
            vl, _ = _loss_grad(beta, X_va, y_va, gs_va, n_va, nr_va, alpha)
            if vl < best_val:
                best_val, best_beta, no_imp = vl, beta.copy(), 0
            else:
                no_imp += 1
            if no_imp >= PATIENCE // 10:
                break
    return best_beta, best_val


def score_df(df_oo, valid, beta, scaler, gs_oo, n_oo):
    df_s = df_oo.sort_values('race_id').reset_index(drop=True)
    X_oo, *_ = prepare(df_oo, valid, scaler=scaler, top_idx=None, top_idx3=None)
    probs = segment_softmax(X_oo @ beta, gs_oo, n_oo)
    df_s['model_prob'] = probs
    df_s['rank_model'] = df_s.groupby('race_id')['model_prob'].rank(ascending=False, method='first')
    df_s['odds_num']   = pd.to_numeric(df_s['単勝オッズ'], errors='coerce')
    top1 = df_s[df_s['rank_model'] == 1]
    if len(top1) == 0:
        return None, 0
    won = top1['着順_num'] == 1
    return float((top1.loc[won, 'odds_num'] * 100).sum() / (len(top1) * 100) - 1), len(top1)


def train_and_eval(df_s, feat_cols, trn_start, trn_end, val_start, val_end, alpha=1.0):
    """train/val で学習し val ROI + 各OOS ROI を返す。"""
    valid = [c for c in feat_cols if c in df_s.columns and df_s[c].isna().mean() <= 0.65]
    if len(valid) < 3:
        return None

    trn  = df_s[(df_s['日付_num'] >= trn_start) & (df_s['日付_num'] < trn_end)]
    val  = df_s[(df_s['日付_num'] >= val_start)  & (df_s['日付_num'] <= val_end)]
    oos2324 = df_s[(df_s['日付_num'] >= 230101) & (df_s['日付_num'] < 250101)].copy()
    oos25   = df_s[df_s['日付_num'] >= 250101].copy()

    if len(trn) < 300 or len(val) < 30:
        return None
    try:
        X_tr, y_tr, gs_tr, n_tr, nr_tr, scaler, *_ = prepare(trn, valid, top_idx=None, top_idx3=None, fit=True)
        X_va, y_va, gs_va, n_va, nr_va, *_          = prepare(val, valid, scaler=scaler, top_idx=None, top_idx3=None)
    except Exception as e:
        return None

    beta, best_val = adam_opt(X_tr, y_tr, gs_tr, n_tr, nr_tr, X_va, y_va, gs_va, n_va, nr_va, alpha)

    # val ROI（特徴量選択の基準）
    val_s = val.sort_values('race_id').reset_index(drop=True)
    val_probs = segment_softmax(X_va @ beta, gs_va, n_va)
    val_s['model_prob'] = val_probs
    val_s['rank_model'] = val_s.groupby('race_id')['model_prob'].rank(ascending=False, method='first')
    val_s['odds_num']   = pd.to_numeric(val_s['単勝オッズ'], errors='coerce')
    top1_val = val_s[val_s['rank_model'] == 1]
    won_val  = top1_val['着順_num'] == 1
    val_roi  = float((top1_val.loc[won_val, 'odds_num'] * 100).sum() / (len(top1_val) * 100) - 1) if len(top1_val) > 0 else None

    # OOS 2023-24 ROI
    roi2324 = None
    if len(oos2324) > 10:
        try:
            X_oo, y_oo, gs_oo, n_oo, *_ = prepare(oos2324, valid, scaler=scaler, top_idx=None, top_idx3=None)
            roi2324, _ = score_df(oos2324, valid, beta, scaler, gs_oo, n_oo)
        except:
            pass

    # OOS 2025 ROI
    roi25 = None
    if len(oos25) > 10:
        try:
            X_oo, y_oo, gs_oo, n_oo, *_ = prepare(oos25, valid, scaler=scaler, top_idx=None, top_idx3=None)
            roi25, _ = score_df(oos25, valid, beta, scaler, gs_oo, n_oo)
        except:
            pass

    return {'val_roi': val_roi, 'roi2324': roi2324, 'roi25': roi25,
            'valid': valid, 'beta': beta, 'scaler': scaler}


def log_result(entry):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')


def greedy_val_based(df, surf, dist_band, init_feats, candidates,
                     trn_start, trn_end, val_start, val_end, alpha=1.0, label=''):
    """val ROI を選択基準にした greedy search（OOSは評価のみ）。"""
    seg_key = f'{surf}_{dist_band}'
    df_s    = df[(df['surface'] == surf) & (df['dist_band'] == dist_band)].copy()
    current = list(init_feats)
    cands   = [c for c in candidates if c not in current
               and c in df_s.columns and df_s[c].isna().mean() <= 0.65]

    res = train_and_eval(df_s, current, trn_start, trn_end, val_start, val_end, alpha)
    if res is None:
        print(f'  [{seg_key}] ベースライン失敗')
        return current, None
    best_val_roi = res['val_roi']
    print(f'  [{seg_key}] ベースライン  val={best_val_roi:+.4f}  oos2324={res["roi2324"]:+.4f}  oos25={res["roi25"]:+.4f}')
    log_result({'ts': datetime.datetime.now().isoformat(), 'phase': f'{label}_baseline',
                'seg': seg_key, 'val_roi': best_val_roi, 'roi2324': res['roi2324'], 'roi25': res['roi25'],
                'n_feats': len(current)})

    improved = True
    while improved:
        improved = False
        # Forward (val ROI 基準)
        for cand in list(cands):
            trial = current + [cand]
            r = train_and_eval(df_s, trial, trn_start, trn_end, val_start, val_end, alpha)
            if r is None or r['val_roi'] is None:
                continue
            log_result({'ts': datetime.datetime.now().isoformat(), 'phase': f'{label}_fwd',
                        'seg': seg_key, 'action': f'+{cand}',
                        'val_roi': r['val_roi'], 'roi2324': r['roi2324'], 'roi25': r['roi25']})
            if r['val_roi'] > best_val_roi + 0.002:
                print(f'    ★ +{cand}: val {best_val_roi:+.4f}→{r["val_roi"]:+.4f}  oos2324={r["roi2324"]:+.4f}  oos25={r["roi25"]:+.4f}')
                best_val_roi, current = r['val_roi'], trial
                cands.remove(cand)
                improved = True
        # Backward
        for rem in list(current):
            if len(current) <= 3:
                break
            trial = [c for c in current if c != rem]
            r = train_and_eval(df_s, trial, trn_start, trn_end, val_start, val_end, alpha)
            if r is None or r['val_roi'] is None:
                continue
            log_result({'ts': datetime.datetime.now().isoformat(), 'phase': f'{label}_bwd',
                        'seg': seg_key, 'action': f'-{rem}',
                        'val_roi': r['val_roi'], 'roi2324': r['roi2324'], 'roi25': r['roi25']})
            if r['val_roi'] > best_val_roi + 0.002:
                print(f'    ★ -{rem}: val {best_val_roi:+.4f}→{r["val_roi"]:+.4f}  oos2324={r["roi2324"]:+.4f}  oos25={r["roi25"]:+.4f}')
                best_val_roi, current = r['val_roi'], trial
                cands.append(rem)
                improved = True

    res_final = train_and_eval(df_s, current, trn_start, trn_end, val_start, val_end, alpha)
    roi2324 = res_final['roi2324'] if res_final else None
    roi25   = res_final['roi25']   if res_final else None
    print(f'  [{seg_key}] 収束: {len(current)}特徴量  val={best_val_roi:+.4f}  oos2324={roi2324:+.4f}  oos25={roi25:+.4f}')
    log_result({'ts': datetime.datetime.now().isoformat(), 'phase': f'{label}_converged',
                'seg': seg_key, 'val_roi': best_val_roi, 'roi2324': roi2324, 'roi25': roi25,
                'features': current, 'n_feats': len(current)})
    return current, best_val_roi


def eval_window(df, seg_feats, segments, trn_start, trn_end, val_start, val_end, label=''):
    """全セグメント評価（訓練窓固定）。"""
    all_top1_2324, all_top1_25 = [], []
    for surf, dist_band in segments:
        seg_key = f'{surf}_{dist_band}'
        feats   = seg_feats.get(seg_key, BASE_FEATURES)
        df_s    = df[(df['surface'] == surf) & (df['dist_band'] == dist_band)].copy()
        res     = train_and_eval(df_s, feats, trn_start, trn_end, val_start, val_end)
        if res is None:
            continue
        r2324 = f'{res["roi2324"]:+.4f}' if res['roi2324'] is not None else 'N/A'
        r25   = f'{res["roi25"]:+.4f}'   if res['roi25']   is not None else 'N/A'
        print(f'  {seg_key}: val={res["val_roi"]:+.4f}  oos2324={r2324}  oos25={r25}')
    print()


def main():
    df = load_data()

    # ── Phase 1: 訓練窓比較（ベースライン特徴量のまま）──────────────────
    print('\n' + '='*60)
    print('Phase 1: 訓練窓比較（v302ベースライン特徴量）')
    print('='*60)
    seg_feats_base = {f'{s}_{b}': list(BASE_FEATURES) for s, b in SEGMENTS_5}
    for trn_start, trn_end, val_start, val_end in TRAIN_WINDOWS:
        label = f'{trn_start//10000}-{(trn_end-1)//10000}'
        print(f'\n--- 訓練窓 {label} ---')
        eval_window(df, seg_feats_base, SEGMENTS_5, trn_start, trn_end, val_start, val_end, label)
        log_result({'ts': datetime.datetime.now().isoformat(), 'phase': 'window_eval',
                    'trn_start': trn_start, 'trn_end': trn_end, 'label': label})

    # ── Phase 2: 最良訓練窓で val-based greedy search ───────────────────
    # Phase 1の結果を見て最良窓を選ぶ。ここでは2018-2022を仮採用（短い窓）
    print('\n' + '='*60)
    print('Phase 2: val-based greedy search (訓練窓 2018-2022)')
    print('='*60)
    best_trn_start, best_trn_end = 180101, 220101
    val_start, val_end = 220101, 221231

    seg_feats_opt = {}
    for surf, dist_band in SEGMENTS_5:
        seg_key = f'{surf}_{dist_band}'
        init    = SEG_INIT.get(seg_key, BASE_FEATURES)
        cands   = [c for c in ADD_CANDIDATES if c not in init]
        print(f'\n=== {seg_key} ===')
        best_feats, _ = greedy_val_based(
            df, surf, dist_band, init, cands,
            best_trn_start, best_trn_end, val_start, val_end, alpha=1.0, label='v4')
        seg_feats_opt[seg_key] = best_feats

    # ── 最終評価 ─────────────────────────────────────────────────────────
    print('\n' + '='*60)
    print('Phase 2 最終評価（全セグメント最適特徴量）')
    print('='*60)
    eval_window(df, seg_feats_opt, SEGMENTS_5, best_trn_start, best_trn_end, val_start, val_end, 'v4_final')
    log_result({'ts': datetime.datetime.now().isoformat(), 'phase': 'v4_final',
                'seg_feats': seg_feats_opt})

    print('\n=== 完了 ===')
    print('seg_feats_opt:', json.dumps(seg_feats_opt, ensure_ascii=False))


if __name__ == '__main__':
    main()
