# coding: utf-8
"""
残存改善余地の精密探索
1. 表面別 hybrid factor (芝/ダで異なる factor)
2. 2022年のみ isotonic (2021=COVID除外)
3. log-ratio ranking: log(calib_prob × odds) = log(EV)
"""
import sys, os, pickle
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import save_lambdarank_pace as _slp
from save_lambdarank_pace import add_pace_features, EXCLUDE, ODDS_REMOVE
from save_conditional_logit import add_new_features, segment_softmax, prepare

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, 'data', 'processed', 'all_venues_features.parquet')
MODEL_DIR = os.path.join(BASE_DIR, 'models')


def calc_roi(df, score_col, odds_col='odds_num', win_col='着順_num'):
    df = df.copy()
    df['rank_'] = df.groupby('race_id')[score_col].rank(ascending=False, method='first')
    top1 = df[df['rank_'] == 1]
    won  = top1[win_col] == 1
    return (top1.loc[won, odds_col] * 100).sum() / (len(top1) * 100) - 1, len(top1), won.mean()


def main():
    # ── データ準備 ────────────────────────────────────────────────────────────
    df = pd.read_parquet(DATA_FILE)
    df['日付_num'] = pd.to_numeric(df['日付'], errors='coerce')
    df['着順_num'] = pd.to_numeric(df['着順_num'], errors='coerce')
    df = df.dropna(subset=['日付_num', '着順_num'])
    df = df[df['着順_num'] < 99]
    df['race_id'] = (df['日付_num'].astype(int).astype(str) + '_' +
                     df['開催'].astype(str).str.strip() + '_' +
                     df['Ｒ'].astype(str).str.strip())
    df = add_pace_features(df)
    df = add_new_features(df)
    df['surface'] = df['距離'].astype(str).str.strip().str.extract(r'^([芝ダ])')[0].fillna('不明')
    df = df[df['surface'].isin(['芝', 'ダ'])].copy()

    # v1 モデル読み込み
    with open(os.path.join(MODEL_DIR, 'surface_clogit.pkl'), 'rb') as f:
        pkg = pickle.load(f)
    artifacts = pkg['artifacts']
    feat_cols = pkg['feat_cols']

    # ── 各期間の予測を生成 ───────────────────────────────────────────────────
    for period_name, period_mask in [
        ('val_2021', (df['日付_num'] >= 210101) & (df['日付_num'] <= 211231)),
        ('val_2022', (df['日付_num'] >= 220101) & (df['日付_num'] <= 221231)),
        ('val_all',  (df['日付_num'] >= 210101) & (df['日付_num'] <= 221231)),
        ('oos',      (df['日付_num'] >= 230101)),
    ]:
        period_df = df[period_mask].sort_values('race_id').reset_index(drop=True)
        raw_probs = np.zeros(len(period_df))
        for surf in ['芝', 'ダ']:
            art = artifacts[surf]
            mask = (period_df['surface'] == surf).values
            pds = period_df[mask].sort_values('race_id').reset_index(drop=True)
            if len(pds) == 0:
                continue
            X, y, gs, n, *_ = prepare(
                pds, art['feat_cols'],
                scaler=art['scaler'], poly2=art['poly2'], inter_scaler2=art['inter_scaler2'],
                top_idx=art['top_idx'],
                poly3=None, inter_scaler3=None, top_idx3=None, fit=False)
            raw_probs[period_df[mask].index] = segment_softmax(X @ art['coef'], gs, n)
        period_df['raw_prob'] = raw_probs
        period_df['odds_num'] = pd.to_numeric(period_df['単勝オッズ'], errors='coerce')
        period_df['market_prob'] = 1.0 / period_df['odds_num'].clip(lower=1.0)
        period_df['yr'] = period_df['日付_num'] // 10000
        if period_name == 'val_2021':
            val2021 = period_df.copy()
        elif period_name == 'val_2022':
            val2022 = period_df.copy()
        elif period_name == 'val_all':
            val_all = period_df.copy()
        else:
            oos = period_df.copy()

    # ── 実験 1: 表面別 isotonic calibration の比較 ───────────────────────────
    # 2022年のみでisotonic fit → OOS に適用
    print('=== 実験1: val 2022のみ isotonic vs val 2021+2022 ===')
    for surf in ['芝', 'ダ']:
        art = artifacts[surf]
        # val_all (2021+2022) での raw prob
        v_all_s = val_all[val_all['surface'] == surf].sort_values('race_id').reset_index(drop=True)
        X, y, gs, n, *_ = prepare(
            v_all_s, art['feat_cols'],
            scaler=art['scaler'], poly2=art['poly2'], inter_scaler2=art['inter_scaler2'],
            top_idx=art['top_idx'], poly3=None, inter_scaler3=None, top_idx3=None, fit=False)
        raw_all = segment_softmax(X @ art['coef'], gs, n)
        v_all_s['raw_prob'] = raw_all

        # val_2022 のみ
        v22_s = val2022[val2022['surface'] == surf].sort_values('race_id').reset_index(drop=True)
        X2, y2, gs2, n2, *_ = prepare(
            v22_s, art['feat_cols'],
            scaler=art['scaler'], poly2=art['poly2'], inter_scaler2=art['inter_scaler2'],
            top_idx=art['top_idx'], poly3=None, inter_scaler3=None, top_idx3=None, fit=False)
        raw_22 = segment_softmax(X2 @ art['coef'], gs2, n2)

        # isotonic (2021+2022)
        ir_all = IsotonicRegression(out_of_bounds='clip', increasing=True)
        ir_all.fit(raw_all, y)

        # isotonic (2022 only)
        ir_22 = IsotonicRegression(out_of_bounds='clip', increasing=True)
        ir_22.fit(raw_22, y2)

        # OOS に適用
        oos_s = oos[oos['surface'] == surf].sort_values('race_id').reset_index(drop=True)
        X_oo, y_oo, gs_oo, n_oo, *_ = prepare(
            oos_s, art['feat_cols'],
            scaler=art['scaler'], poly2=art['poly2'], inter_scaler2=art['inter_scaler2'],
            top_idx=art['top_idx'], poly3=None, inter_scaler3=None, top_idx3=None, fit=False)
        raw_oo = segment_softmax(X_oo @ art['coef'], gs_oo, n_oo)
        oos_s['calib_all'] = ir_all.predict(raw_oo)
        oos_s['calib_22']  = ir_22.predict(raw_oo)
        oos_s['raw_prob']  = raw_oo
        oos_s['odds_num']  = pd.to_numeric(oos_s['単勝オッズ'], errors='coerce').values

        r_all, _, _ = calc_roi(oos_s, 'calib_all')
        r_22, _, _  = calc_roi(oos_s, 'calib_22')
        r_raw, _, _ = calc_roi(oos_s, 'raw_prob')
        print(f'  {surf}: raw={r_raw:+.3f}  isotonic(2021+2022)={r_all:+.3f}  isotonic(2022only)={r_22:+.3f}')

    # ── 実験 2: 表面別 hybrid factor 探索 ────────────────────────────────────
    print('\n=== 実験2: 表面別 hybrid factor (val_all, OOS) ===')
    # まず val_all で各 surf の best factor を探す
    for surf in ['芝', 'ダ']:
        art = artifacts[surf]
        vs  = val_all[val_all['surface'] == surf].copy()
        X, y, gs, n, *_ = prepare(
            vs.sort_values('race_id').reset_index(drop=True), art['feat_cols'],
            scaler=art['scaler'], poly2=art['poly2'], inter_scaler2=art['inter_scaler2'],
            top_idx=art['top_idx'], poly3=None, inter_scaler3=None, top_idx3=None, fit=False)
        raw = segment_softmax(X @ art['coef'], gs, n)
        calib = art['isotonic'].predict(raw)
        vs_s = vs.sort_values('race_id').reset_index(drop=True)
        vs_s['calib_prob'] = calib
        vs_s['market_prob'] = 1.0 / vs_s['odds_num'].clip(lower=1.0)
        print(f'  {surf} val:')
        for f in [0.0, 0.05, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25]:
            vs_s['score'] = vs_s['calib_prob'] - f * vs_s['market_prob']
            r, n_r, w = calc_roi(vs_s, 'score')
            print(f'    factor={f:.2f}: {n_r}R  win={w:.3f}  ROI={r:+.3f}')

    # ── 実験 3: log-ratio ranking ─────────────────────────────────────────────
    print('\n=== 実験3: log-ratio ranking log(calib_prob × odds) ===')
    # val_all で log-ratio ROI
    # まず全表面のcalib_prob を取得済みと仮定して計算
    for period_name, period_df in [('val_all', val_all), ('OOS', oos)]:
        preds = []
        for surf in ['芝', 'ダ']:
            art = artifacts[surf]
            ps  = period_df[period_df['surface'] == surf].sort_values('race_id').reset_index(drop=True)
            if len(ps) == 0:
                continue
            X, y, gs, n, *_ = prepare(
                ps, art['feat_cols'],
                scaler=art['scaler'], poly2=art['poly2'], inter_scaler2=art['inter_scaler2'],
                top_idx=art['top_idx'], poly3=None, inter_scaler3=None, top_idx3=None, fit=False)
            raw   = segment_softmax(X @ art['coef'], gs, n)
            calib = art['isotonic'].predict(raw)
            ps['calib_prob']    = calib
            ps['log_ev']        = np.log(np.clip(calib, 1e-9, 1.0)) + np.log(ps['odds_num'].clip(lower=1.0))
            ps['raw_prob']      = raw
            preds.append(ps)
        all_p = pd.concat(preds, ignore_index=True)
        all_p['odds_num'] = pd.to_numeric(all_p['odds_num'], errors='coerce')

        r_pure, n_r, w   = calc_roi(all_p, 'calib_prob')
        r_log,  n_r2, w2 = calc_roi(all_p, 'log_ev')
        print(f'  {period_name}:')
        print(f'    calib_prob (factor=0): {n_r}R  win={w:.3f}  ROI={r_pure:+.3f}')
        print(f'    log_ev ranking:        {n_r2}R  win={w2:.3f}  ROI={r_log:+.3f}')

        # hybrid との比較 (factor=0.15)
        all_p['market_prob'] = 1.0 / all_p['odds_num'].clip(lower=1.0)
        all_p['hybrid_15'] = all_p['calib_prob'] - 0.15 * all_p['market_prob']
        r_h15, n_h, w_h = calc_roi(all_p, 'hybrid_15')
        print(f'    hybrid factor=0.15:    {n_h}R  win={w_h:.3f}  ROI={r_h15:+.3f}')


if __name__ == '__main__':
    main()
