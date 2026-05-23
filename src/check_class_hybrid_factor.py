# coding: utf-8
"""
クラス別 hybrid factor + 2022年のみ isotonic の組み合わせ探索
- 未勝利 (クラス_rank=2): 市場優位性が高い → factor を大きく
- 非未勝利: factor を小さく
- val で最適化し OOS で検証
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


def get_preds(df, artifacts, feat_cols, period_mask):
    period_df = df[period_mask].sort_values('race_id').reset_index(drop=True)
    raw_arr   = np.zeros(len(period_df))
    for surf in ['芝', 'ダ']:
        art = artifacts[surf]
        mask = (period_df['surface'] == surf).values
        ps   = period_df[mask].sort_values('race_id').reset_index(drop=True)
        if len(ps) == 0:
            continue
        X, y, gs, n, *_ = prepare(
            ps, art['feat_cols'],
            scaler=art['scaler'], poly2=art['poly2'], inter_scaler2=art['inter_scaler2'],
            top_idx=art['top_idx'], poly3=None, inter_scaler3=None, top_idx3=None, fit=False)
        raw_arr[period_df[mask].index] = segment_softmax(X @ art['coef'], gs, n)
    period_df['raw_prob'] = raw_arr
    period_df['odds_num'] = pd.to_numeric(period_df['単勝オッズ'], errors='coerce')
    period_df['market_prob'] = 1.0 / period_df['odds_num'].clip(lower=1.0)
    period_df['yr'] = period_df['日付_num'] // 10000
    return period_df


def fit_isotonic(df, artifacts, feat_cols, period_mask, surf_filter=None):
    """各 surface の isotonic を fit して返す"""
    period_df = df[period_mask].sort_values('race_id').reset_index(drop=True)
    isotonics = {}
    for surf in ['芝', 'ダ']:
        if surf_filter is not None and surf != surf_filter:
            isotonics[surf] = artifacts[surf]['isotonic']
            continue
        art = artifacts[surf]
        mask = (period_df['surface'] == surf).values
        ps   = period_df[mask].sort_values('race_id').reset_index(drop=True)
        if len(ps) == 0:
            isotonics[surf] = art['isotonic']
            continue
        X, y, gs, n, *_ = prepare(
            ps, art['feat_cols'],
            scaler=art['scaler'], poly2=art['poly2'], inter_scaler2=art['inter_scaler2'],
            top_idx=art['top_idx'], poly3=None, inter_scaler3=None, top_idx3=None, fit=False)
        raw = segment_softmax(X @ art['coef'], gs, n)
        ir = IsotonicRegression(out_of_bounds='clip', increasing=True)
        ir.fit(raw, y)
        isotonics[surf] = ir
    return isotonics


def apply_calib(df, artifacts, feat_cols, isotonics):
    """raw → calib_prob を適用"""
    df = df.copy()
    calib_arr = np.zeros(len(df))
    for surf in ['芝', 'ダ']:
        art  = artifacts[surf]
        mask = (df['surface'] == surf).values
        ps   = df[mask].sort_values('race_id').reset_index(drop=True)
        if len(ps) == 0:
            continue
        X, y, gs, n, *_ = prepare(
            ps, art['feat_cols'],
            scaler=art['scaler'], poly2=art['poly2'], inter_scaler2=art['inter_scaler2'],
            top_idx=art['top_idx'], poly3=None, inter_scaler3=None, top_idx3=None, fit=False)
        raw   = segment_softmax(X @ art['coef'], gs, n)
        calib = isotonics[surf].predict(raw)
        calib_arr[df[mask].index] = calib
    df['calib_prob'] = calib_arr
    return df


def eval_roi(df, factor_maiden, factor_other):
    """クラス別 factor で rank=1 を選んで ROI を計算"""
    df = df.copy()
    df['is_maiden'] = (df.get('クラス_rank', pd.Series(0, index=df.index)) == 2)
    factor_arr = np.where(df['is_maiden'], factor_maiden, factor_other)
    df['score'] = df['calib_prob'] - factor_arr * df['market_prob']
    df['rank_'] = df.groupby('race_id')['score'].rank(ascending=False, method='first')
    top1 = df[df['rank_'] == 1]
    won  = top1['着順_num'] == 1
    roi  = (top1.loc[won, 'odds_num'] * 100).sum() / (len(top1) * 100) - 1
    return roi, len(top1), won.mean()


def main():
    df = pd.read_parquet(DATA_FILE)
    df['日付_num'] = pd.to_numeric(df['日付'], errors='coerce')
    df['着順_num'] = pd.to_numeric(df['着順_num'], errors='coerce')
    df['クラス_rank'] = pd.to_numeric(df['クラス_rank'], errors='coerce')
    df = df.dropna(subset=['日付_num', '着順_num'])
    df = df[df['着順_num'] < 99]
    df['race_id'] = (df['日付_num'].astype(int).astype(str) + '_' +
                     df['開催'].astype(str).str.strip() + '_' +
                     df['Ｒ'].astype(str).str.strip())
    df = add_pace_features(df)
    df = add_new_features(df)
    df['surface'] = df['距離'].astype(str).str.strip().str.extract(r'^([芝ダ])')[0].fillna('不明')
    df = df[df['surface'].isin(['芝', 'ダ'])].copy()

    with open(os.path.join(MODEL_DIR, 'surface_clogit.pkl'), 'rb') as f:
        pkg = pickle.load(f)
    artifacts = pkg['artifacts']
    feat_cols = pkg['feat_cols']

    # isotonic パターン: A=2021+2022, B=2022+芝のみ2022, C=2022のみ
    mask_val_all = (df['日付_num'] >= 210101) & (df['日付_num'] <= 221231)
    mask_val_22  = (df['日付_num'] >= 220101) & (df['日付_num'] <= 221231)
    mask_oos     = df['日付_num'] >= 230101

    iso_AB = {surf: artifacts[surf]['isotonic'] for surf in ['芝','ダ']}  # 現在の2021+2022
    # 2022年のみ isotonic (芝だけ)
    iso_B  = fit_isotonic(df, artifacts, feat_cols, mask_val_22, surf_filter='芝')
    iso_B['ダ'] = artifacts['ダ']['isotonic']  # ダは2021+2022のまま

    val_df  = get_preds(df, artifacts, feat_cols, mask_val_all)
    oos_df  = get_preds(df, artifacts, feat_cols, mask_oos)

    val_calib_AB = apply_calib(val_df,  artifacts, feat_cols, iso_AB)
    oos_calib_AB = apply_calib(oos_df,  artifacts, feat_cols, iso_AB)
    val_calib_B  = apply_calib(val_df,  artifacts, feat_cols, iso_B)
    oos_calib_B  = apply_calib(oos_df,  artifacts, feat_cols, iso_B)

    print('=== クラス別 hybrid factor グリッドサーチ ===')
    print(f'{"isotonic":6} {"f_maiden":>8} {"f_other":>8} {"val ROI":>9} {"OOS ROI":>9}')
    print('-'*55)

    best_val = -np.inf
    best_cfg = None

    for iso_name, val_c, oos_c in [('A:2021+22', val_calib_AB, oos_calib_AB),
                                    ('B:22only芝', val_calib_B,  oos_calib_B)]:
        for f_maiden in [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
            for f_other in [0.00, 0.05, 0.10, 0.15, 0.20]:
                vr, nv, wv = eval_roi(val_c, f_maiden, f_other)
                or_, no, wo = eval_roi(oos_c, f_maiden, f_other)
                marker = ' ←' if vr > best_val else ''
                if vr > best_val:
                    best_val = vr
                    best_cfg = (iso_name, f_maiden, f_other, vr, or_)
                if f_maiden == f_other or abs(f_maiden - f_other) <= 0.10:
                    print(f'{iso_name:10} {f_maiden:>8.2f} {f_other:>8.2f} {vr:>+9.3f} {or_:>+9.3f}{marker}')

    print(f'\n最良 val ROI = {best_cfg[3]:+.3f}  対応 OOS = {best_cfg[4]:+.3f}')
    print(f'設定: {best_cfg[0]}  f_maiden={best_cfg[1]:.2f}  f_other={best_cfg[2]:.2f}')

    # 詳細: best 設定周辺を表示
    print('\n=== Best設定周辺 詳細 ===')
    iso_name, iso_nm, iso_ot, _, _ = best_cfg[0], best_cfg[1], best_cfg[2], best_cfg[3], best_cfg[4]
    if 'B' in iso_name:
        val_use, oos_use = val_calib_B, oos_calib_B
    else:
        val_use, oos_use = val_calib_AB, oos_calib_AB
    for f_maiden in np.arange(max(0, iso_nm-0.05), iso_nm+0.10, 0.02):
        for f_other in np.arange(max(0, iso_ot-0.05), iso_ot+0.10, 0.02):
            vr, _, _ = eval_roi(val_use, round(f_maiden,3), round(f_other,3))
            or_, _, _ = eval_roi(oos_use, round(f_maiden,3), round(f_other,3))
            print(f'  f_maiden={f_maiden:.3f}  f_other={f_other:.3f}  val={vr:+.4f}  OOS={or_:+.4f}')

    # 年別 ROI (best 設定)
    print(f'\n=== 年別 OOS (best設定: {best_cfg[0]} f_maiden={best_cfg[1]:.2f} f_other={best_cfg[2]:.2f}) ===')
    oos_use = oos_calib_B if 'B' in best_cfg[0] else oos_calib_AB
    oos_use2 = oos_use.copy()
    oos_use2['is_maiden'] = (oos_use2['クラス_rank'] == 2)
    fac = np.where(oos_use2['is_maiden'], best_cfg[1], best_cfg[2])
    oos_use2['score'] = oos_use2['calib_prob'] - fac * oos_use2['market_prob']
    oos_use2['rank_'] = oos_use2.groupby('race_id')['score'].rank(ascending=False, method='first')
    t1 = oos_use2[oos_use2['rank_'] == 1]
    for yr in sorted(t1['yr'].unique()):
        s = t1[t1['yr'] == yr]
        w = s['着順_num'] == 1
        r = (s.loc[w,'odds_num']*100).sum()/(len(s)*100)-1
        print(f'  20{int(yr):02d}: {len(s):5d}R  win={w.mean():.3f}  ROI={r:+.3f}')


if __name__ == '__main__':
    main()
