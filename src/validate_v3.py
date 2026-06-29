# coding: utf-8
"""
feature_search_v3 結果の2025-only検証
- OOS(2023-2024)で選択した特徴量セットを 2025年データで独立評価
- 芝_長距離+6%がOOS過学習か真のシグナルかを判定
"""
import sys, os, json
import numpy as np
import pandas as pd

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from save_conditional_logit import prepare, segment_softmax, BASE_DIR, DATA_FILE, LR, N_EPOCHS, PATIENCE

# ── v3 per-segment 最適特徴量（feature_search_v3 収束結果）────────────────
BASE_FEATURES = [
    '1走前_タイム指数', '1走前_上り3F', '前走着差タイム', '1走前_RPCI',
    '近5走_タイム指数平均', '近5走_タイム指数_max', 'タイム指数_近3走_slope',
    '1走前_クラス調整着順', '近5走_クラス調整_平均着順',
    '馬番', 'コース枠_r200_勝率',
    '騎手コース_r100_勝率', '騎手変更', '調教師コース_r100_勝率',
    'コース脚質_r200_勝率', '1走前_脚質_num', '種牡馬_勝率', '種牡馬_ダ_勝率',
    '斤量', '芝ダ転向', '間隔',
]

SEG_FEATS_OPT = {
    '芝_短距離': [
        '1走前_タイム指数', '1走前_上り3F', '前走着差タイム', '1走前_RPCI',
        '近5走_タイム指数平均', '近5走_タイム指数_max', 'タイム指数_近3走_slope',
        '1走前_クラス調整着順', '近5走_クラス調整_平均着順',
        '馬番',
        '騎手コース_r100_勝率', '調教師コース_r100_勝率',
        'コース脚質_r200_勝率', '1走前_脚質_num', '種牡馬_勝率', '種牡馬_ダ_勝率',
        '斤量', '芝ダ転向', '間隔', '1走前_3角',
    ],
    '芝_中距離': BASE_FEATURES + ['馬体重', '間隔_短_flag', '血統_ダ優位度', '馬体重増減'],
    '芝_長距離': [
        '近5走_タイム指数平均', 'タイム指数_近3走_slope', '近5走_クラス調整_平均着順',
        '騎手コース_r100_勝率', '騎手変更', '調教師コース_r100_勝率',
        '芝ダ転向', '間隔', '血統_ダ優位度', '1走前_RPCI',
        '1走前_上3F地点差', '間隔_短_flag', 'キャリア', '母父馬_勝率', '斤量',
    ],
    'ダ_短距離': [
        '1走前_タイム指数', '1走前_上り3F', '前走着差タイム', '1走前_RPCI',
        '近5走_タイム指数平均', '近5走_タイム指数_max', 'タイム指数_近3走_slope',
        '1走前_クラス調整着順', '近5走_クラス調整_平均着順',
        '馬番', 'コース枠_r200_勝率',
        '騎手コース_r100_勝率', '騎手変更', '調教師コース_r100_勝率',
        'コース脚質_r200_勝率', '1走前_脚質_num', '種牡馬_勝率', '種牡馬_ダ_勝率',
        '斤量', '間隔',
        '馬体重増減', '展開フィット_v2', '乗替り_近走不振', '間隔_長_flag',
    ],
    'ダ_中長距離': [
        '1走前_タイム指数', '1走前_上り3F', '前走着差タイム', '1走前_RPCI',
        '近5走_タイム指数平均', '近5走_タイム指数_max',
        '1走前_クラス調整着順', '近5走_クラス調整_平均着順',
        '馬番', 'コース枠_r200_勝率',
        '騎手コース_r100_勝率', '騎手変更', '調教師コース_r100_勝率',
        '1走前_脚質_num', '種牡馬_勝率', '種牡馬_ダ_勝率',
        '斤量', '間隔',
        '間隔_長_flag', '1走前_上3F地点差',
    ],
}

SEGMENTS_5 = [('芝', '短距離'), ('芝', '中距離'), ('芝', '長距離'),
              ('ダ', '短距離'), ('ダ', '中長距離')]


def add_computed_features(df):
    interval = pd.to_numeric(df['間隔'], errors='coerce') if '間隔' in df.columns else pd.Series(np.nan, index=df.index)
    df['間隔_長_flag'] = (interval >= 60).astype(float)
    df['間隔_短_flag'] = (interval <= 14).astype(float)
    da_r  = pd.to_numeric(df.get('種牡馬_ダ_勝率', np.nan), errors='coerce')
    all_r = pd.to_numeric(df.get('種牡馬_勝率',    np.nan), errors='coerce')
    df['血統_ダ優位度'] = da_r - all_r
    return df


def load_data():
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
    return df


def _loss_grad(beta, X, y, gs, n, nr, alpha=1.0):
    probs = segment_softmax(X @ beta, gs, n)
    log_lik = np.sum(y * np.log(np.clip(probs, 1e-15, 1.0)))
    res = y - probs
    loss = (-log_lik + alpha * np.sum(beta**2)) / nr
    grad = (-(X.T @ res) + 2 * alpha * beta) / nr
    return loss, grad


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
    return best_beta


def eval_on_period(df, seg_feats, segments, trn_end, val_start, val_end, oos_start, label=''):
    """指定期間でtrain/val/OOSに分けて評価。"""
    all_top1_oos23, all_top1_oos25 = [], []
    results = {}
    for surf, dist_band in segments:
        seg_key = f'{surf}_{dist_band}'
        feats   = seg_feats.get(seg_key, BASE_FEATURES)
        df_s    = df[(df['surface'] == surf) & (df['dist_band'] == dist_band)].copy()

        valid = [c for c in feats if c in df_s.columns and df_s[c].isna().mean() <= 0.65]
        if len(valid) < 3:
            print(f'  {seg_key}: 列不足スキップ')
            continue

        trn = df_s[df_s['日付_num'] < val_start]
        val = df_s[(df_s['日付_num'] >= val_start) & (df_s['日付_num'] <= val_end)]
        # OOS 2023-2024（選択に使ったデータ）
        oos23 = df_s[(df_s['日付_num'] >= oos_start) & (df_s['日付_num'] < 250101)].copy()
        # OOS 2025（真のホールドアウト）
        oos25 = df_s[df_s['日付_num'] >= 250101].copy()

        if len(trn) < 300 or len(val) < 30:
            print(f'  {seg_key}: データ不足スキップ')
            continue

        try:
            X_tr, y_tr, gs_tr, n_tr, nr_tr, scaler, *_ = prepare(trn, valid, top_idx=None, top_idx3=None, fit=True)
            X_va, y_va, gs_va, n_va, nr_va, *_          = prepare(val, valid, scaler=scaler, top_idx=None, top_idx3=None)
        except Exception as e:
            print(f'  {seg_key}: prepare失敗 {e}')
            continue

        beta = adam_opt(X_tr, y_tr, gs_tr, n_tr, nr_tr, X_va, y_va, gs_va, n_va, nr_va)

        def score_period(oos_df):
            if len(oos_df) < 10:
                return None, 0, 0
            try:
                X_oo, y_oo, gs_oo, n_oo, *_ = prepare(oos_df, valid, scaler=scaler, top_idx=None, top_idx3=None)
            except:
                return None, 0, 0
            oos_df = oos_df.sort_values('race_id').reset_index(drop=True)
            probs = segment_softmax(X_oo @ beta, gs_oo, n_oo)
            oos_df['model_prob'] = probs
            oos_df['rank_model'] = oos_df.groupby('race_id')['model_prob'].rank(ascending=False, method='first')
            oos_df['odds_num']   = pd.to_numeric(oos_df['単勝オッズ'], errors='coerce')
            top1 = oos_df[oos_df['rank_model'] == 1]
            if len(top1) == 0:
                return None, 0, 0
            won = top1['着順_num'] == 1
            roi = (top1.loc[won, 'odds_num'] * 100).sum() / (len(top1) * 100) - 1
            return float(roi), len(top1), int(won.sum())

        roi23, n23, w23 = score_period(oos23)
        roi25, n25, w25 = score_period(oos25)

        roi23_s = f'{roi23:+.3f}' if roi23 is not None else 'N/A'
        roi25_s = f'{roi25:+.3f}' if roi25 is not None else 'N/A'
        print(f'  {seg_key}:  2023-24: {n23}R 勝率{w23/max(n23,1):.1%} ROI={roi23_s}  |  2025: {n25}R 勝率{w25/max(n25,1):.1%} ROI={roi25_s}')

        results[seg_key] = {'roi_2023': roi23, 'n_2023': n23, 'roi_2025': roi25, 'n_25': n25}
        if roi23 is not None and n23 > 0:
            top1_23 = oos23.copy() if len(oos23) > 0 else pd.DataFrame()
        if roi25 is not None and n25 > 0:
            # collect for aggregate
            try:
                X_oo, y_oo, gs_oo, n_oo, *_ = prepare(oos25, valid, scaler=scaler, top_idx=None, top_idx3=None)
                oos25s = oos25.sort_values('race_id').reset_index(drop=True)
                probs = segment_softmax(X_oo @ beta, gs_oo, n_oo)
                oos25s['model_prob'] = probs
                oos25s['rank_model'] = oos25s.groupby('race_id')['model_prob'].rank(ascending=False, method='first')
                oos25s['odds_num']   = pd.to_numeric(oos25s['単勝オッズ'], errors='coerce')
                all_top1_oos25.append(oos25s[oos25s['rank_model'] == 1])
            except:
                pass

    if all_top1_oos25:
        combined = pd.concat(all_top1_oos25, ignore_index=True)
        won = combined['着順_num'] == 1
        total = (combined.loc[won, 'odds_num'] * 100).sum() / (len(combined) * 100) - 1
        print(f'\n  {label} 2025合計: {len(combined)}R  ROI={total:+.4f}')
    else:
        total = None
        print(f'\n  {label} 2025: データなし')
    return total, results


def main():
    print('データ読み込み...')
    df = load_data()
    print(f'有効データ: {len(df):,}行')

    # ── 1. ベースライン（exp_v302, 全共通21特徴量）────────────────────────
    print('\n' + '='*60)
    print('【ベースライン: exp_v302  全共通21特徴量】')
    print('='*60)
    seg_feats_base = {f'{s}_{b}': list(BASE_FEATURES) for s, b in SEGMENTS_5}
    roi_base_25, _ = eval_on_period(
        df, seg_feats_base, SEGMENTS_5,
        trn_end=220101, val_start=220101, val_end=221231, oos_start=230101,
        label='baseline')

    # ── 2. v3 5seg最適特徴量────────────────────────────────────────────────
    print('\n' + '='*60)
    print('【v3 5seg per-segment最適特徴量】')
    print('='*60)
    roi_v3_25, seg_results = eval_on_period(
        df, SEG_FEATS_OPT, SEGMENTS_5,
        trn_end=220101, val_start=220101, val_end=221231, oos_start=230101,
        label='v3_5seg')

    # ── サマリー ─────────────────────────────────────────────────────────
    print('\n' + '='*60)
    print('=== 検証サマリー ===')
    print(f'ベースライン 2025: {roi_base_25:+.4f}' if roi_base_25 else 'ベースライン 2025: N/A')
    print(f'v3 5seg      2025: {roi_v3_25:+.4f}'   if roi_v3_25   else 'v3 5seg      2025: N/A')
    print()
    print('セグメント別 2025 ROI:')
    for seg, r in seg_results.items():
        r25 = f'{r["roi_2025"]:+.4f}' if r.get('roi_2025') is not None else 'N/A'
        r23 = f'{r["roi_2023"]:+.4f}' if r.get('roi_2023') is not None else 'N/A'
        print(f'  {seg}: 2023-24={r23}  2025={r25}')
    print('='*60)


if __name__ == '__main__':
    main()
