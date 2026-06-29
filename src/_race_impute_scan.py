# coding: utf-8
"""
実験: レース内平均 + オッズ調整による NaN 補完スキャン
----------------------------------------------------------------------
対象列（能力系、NaN率10-20%）に対して:
  Step1: NaN を同レース内の非NaN馬の平均で埋める
  Step2: さらにオッズの相対強さで上下に調整する (α でスケール)

α=0 → レース内平均のみ
α>0 → レース内平均 + オッズ調整

【注意】モデルは再学習しない。同じ weights で imputation だけ変えてROIを比較。
"""
import os, sys, pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from save_lambdarank_pace import add_pace_features
from save_conditional_logit import add_new_features, segment_softmax, prepare
from save_final_model import make_race_id, get_surface

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, 'data', 'processed', 'all_venues_features.parquet')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# 補完対象列と方向 (+1=高い方が良い, -1=低い方が良い, 0=方向なし)
TARGET_COLS = {
    '近5走_タイム指数平均':       +1,
    '近5走_タイム指数_max':       +1,
    '近5走_タイム指数_min':       +1,
    '近5走_タイム指数_range':      0,
    '近5走_タイム指数_std':        0,
    '近5走_平均着順':             -1,
    '近5走_複勝率':               +1,
    '近5走_クラス調整_平均着順':  -1,
    '近5走_クラス補正スコア':     +1,
    '近3走_勝率':                 +1,
    '近3走_平均着順':             -1,
    '近3走_複勝率':               +1,
    '近10走_勝率':                +1,
    '近10走_平均着順':            -1,
    '近10走_複勝率':              +1,
    '1走前_タイム指数':           +1,
    '1走前_着順_num':             -1,
    '近走着順トレンド':            0,  # 方向が複雑なので調整なし
}


def apply_imputation(df, target_cols, alpha=0.0):
    """
    NaN補完を適用して df のコピーを返す。
    alpha=0: レース内平均のみ
    alpha>0: レース内平均 + オッズ相対差による調整
    """
    df = df.copy()

    # オッズ確率（補完には使わない場合も計算しておく）
    mp = pd.to_numeric(df['単勝オッズ'], errors='coerce')
    df['_mprob'] = 1.0 / np.clip(mp.values, 1.0, None)

    for col, sign in target_cols.items():
        if col not in df.columns:
            continue
        nan_mask = df[col].isna()
        if nan_mask.sum() == 0:
            continue

        # Step1: レース内平均で埋める
        race_mean = df.groupby('race_id')[col].transform(
            lambda x: x.mean()  # NaN は無視して計算
        )
        imputed = race_mean.copy()

        # Step2: オッズ調整（alpha > 0 かつ sign != 0 の列のみ）
        if alpha > 0 and sign != 0:
            race_mean_mp = df.groupby('race_id')['_mprob'].transform('mean')
            race_std_mp  = df.groupby('race_id')['_mprob'].transform('std').fillna(0.01)
            # オッズのレース内 z スコア (正 = 市場が強いと見ている)
            odds_z = (df['_mprob'] - race_mean_mp) / race_std_mp.clip(lower=0.01)
            # レース内の列の std（スケール感を合わせる）
            race_std_col = df.groupby('race_id')[col].transform('std').fillna(
                df[col].std()
            )
            imputed = imputed + sign * alpha * odds_z * race_std_col

        # NaN のところだけ更新（非NaNはそのまま）
        df.loc[nan_mask, col] = imputed[nan_mask]

    df = df.drop(columns=['_mprob'])
    return df


def compute_oos_roi(oos, artifacts, feat_cols_pkg, factor_maiden, factor_other,
                    target_cols, alpha, label):
    """imputationを適用してOOS ROIを計算して返す。"""
    oos_imp = apply_imputation(oos, target_cols, alpha=alpha)

    calib_arr = np.zeros(len(oos_imp))
    for surf in ['芝', 'ダ']:
        art  = artifacts[surf]
        mask = (oos_imp['surface'] == surf).values
        if mask.sum() == 0:
            continue
        oos_s = oos_imp[mask].sort_values('race_id').reset_index(drop=True)
        X, y, gs, n, *_ = prepare(
            oos_s, art['feat_cols'],
            scaler=art['scaler'], poly2=art['poly2'],
            inter_scaler2=art['inter_scaler2'], top_idx=art['top_idx'],
            poly3=None, inter_scaler3=None, top_idx3=None, fit=False)
        raw   = segment_softmax(X @ art['coef'], gs, n)
        calib = art['isotonic'].predict(raw)
        calib_arr[oos_imp[mask].index.values] = calib

    oos_imp = oos_imp.copy()
    oos_imp['calib_prob']  = calib_arr
    oos_imp['market_prob'] = 1.0 / np.clip(
        pd.to_numeric(oos_imp['単勝オッズ'], errors='coerce').values, 1.0, None)
    factor_arr = np.where(oos_imp['クラス_rank'] == 2, factor_maiden, factor_other)
    score = oos_imp['calib_prob'].values - factor_arr * oos_imp['market_prob'].values
    oos_imp['_rank'] = pd.Series(score, index=oos_imp.index).groupby(
        oos_imp['race_id']).rank(ascending=False, method='first')

    top1 = oos_imp[oos_imp['_rank'] == 1]
    won  = top1['着順_num'] == 1
    odds = pd.to_numeric(top1['単勝オッズ'], errors='coerce')
    roi  = (odds[won] * 100).sum() / (len(top1) * 100) - 1
    print(f'  {label:<55s}  ROI={roi:+.4f}  ({len(top1)}R  win={won.mean():.3f})')
    return roi


def main():
    print('モデル読み込み中...')
    with open(os.path.join(MODEL_DIR, 'roi_model.pkl'), 'rb') as f:
        pkg = pickle.load(f)
    artifacts     = pkg['artifacts']
    feat_cols_pkg = pkg['feat_cols']
    FACTOR_MAIDEN = pkg.get('factor_maiden', 0.0)
    FACTOR_OTHER  = pkg.get('factor_other',  0.16)
    print(f'  保存済みOOS ROI: {pkg.get("total_oos_roi", "不明"):+.4f}')

    print('データ読み込み・前処理中...')
    df = pd.read_parquet(DATA_FILE)
    df['日付_num']   = pd.to_numeric(df['日付'],       errors='coerce')
    df['着順_num']   = pd.to_numeric(df['着順_num'],   errors='coerce')
    df['クラス_rank'] = pd.to_numeric(df['クラス_rank'], errors='coerce')
    df = df.dropna(subset=['日付_num', '着順_num'])
    df = df[df['着順_num'] < 99]
    df = df[df['開催'].notna()].copy()
    df = make_race_id(df)
    df = add_pace_features(df)
    df = add_new_features(df)
    df['surface'] = get_surface(df)
    df = df[df['surface'].isin(['芝', 'ダ'])].copy()

    oos = df[df['日付_num'] >= 230101].sort_values('race_id').reset_index(drop=True)
    print(f'  OOS行数: {len(oos):,}')

    # 実際に feat_cols に含まれる列だけに絞る
    fc_set = set(feat_cols_pkg)
    active_targets = {c: s for c, s in TARGET_COLS.items() if c in fc_set and c in oos.columns}
    print(f'  補完対象列: {len(active_targets)}列')
    print()

    print('=== レース内平均 + オッズ調整スキャン (OOS 2023+) ===\n')

    # ベースライン（補完なし）
    compute_oos_roi(oos, artifacts, feat_cols_pkg, FACTOR_MAIDEN, FACTOR_OTHER,
                    {}, 0.0, 'A. ベースライン (補完なし)')

    print()

    # レース内平均のみ (α=0)
    compute_oos_roi(oos, artifacts, feat_cols_pkg, FACTOR_MAIDEN, FACTOR_OTHER,
                    active_targets, 0.0, 'B. レース内平均のみ (α=0)')

    print()

    # レース内平均 + オッズ調整 (α スキャン)
    for alpha in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
        compute_oos_roi(oos, artifacts, feat_cols_pkg, FACTOR_MAIDEN, FACTOR_OTHER,
                        active_targets, alpha, f'C. レース内平均+オッズ調整 α={alpha}')

    print()

    # 年別内訳（ベースライン vs ベスト候補）
    print('--- 年別内訳 (ベースライン) ---')
    oos_b = apply_imputation(oos, {}, 0.0)
    calib_arr = np.zeros(len(oos_b))
    for surf in ['芝', 'ダ']:
        art  = artifacts[surf]
        mask = (oos_b['surface'] == surf).values
        oos_s = oos_b[mask].sort_values('race_id').reset_index(drop=True)
        X, y, gs, n, *_ = prepare(
            oos_s, art['feat_cols'],
            scaler=art['scaler'], poly2=art['poly2'],
            inter_scaler2=art['inter_scaler2'], top_idx=art['top_idx'],
            poly3=None, inter_scaler3=None, top_idx3=None, fit=False)
        raw   = segment_softmax(X @ art['coef'], gs, n)
        calib = art['isotonic'].predict(raw)
        calib_arr[oos_b[mask].index.values] = calib
    oos_b['calib_prob']  = calib_arr
    oos_b['market_prob'] = 1.0 / np.clip(
        pd.to_numeric(oos_b['単勝オッズ'], errors='coerce').values, 1.0, None)
    factor_arr = np.where(oos_b['クラス_rank'] == 2, FACTOR_MAIDEN, FACTOR_OTHER)
    score = oos_b['calib_prob'].values - factor_arr * oos_b['market_prob'].values
    oos_b['_rank'] = pd.Series(score, index=oos_b.index).groupby(
        oos_b['race_id']).rank(ascending=False, method='first')
    oos_b['yr'] = oos_b['日付_num'] // 10000
    for yr in sorted(oos_b['yr'].unique()):
        s   = oos_b[(oos_b['yr'] == yr) & (oos_b['_rank'] == 1)]
        won = s['着順_num'] == 1
        r   = (pd.to_numeric(s['単勝オッズ'], errors='coerce')[won] * 100).sum() / (len(s)*100) - 1
        print(f'  20{int(yr):02d}: {len(s):5d}R  win={won.mean():.3f}  ROI={r:+.3f}')


if __name__ == '__main__':
    main()
