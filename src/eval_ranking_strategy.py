# coding: utf-8
"""
rank=1 選択基準を3通り比較: model_prob / ev_score / expected_return
目標: OOS rank=1全体 ROI >= -5%
"""
import os, sys, pickle
import numpy as np
import pandas as pd

# sys.stdout をラップしているモジュールを先に import（二重ラップ防止）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import save_lambdarank_pace as slp   # module-level stdout wrap はここで1回だけ起きる
from save_lambdarank_pace import add_pace_features, EXCLUDE, ODDS_REMOVE
from save_conditional_logit import add_new_features, segment_softmax, get_group_starts, prepare

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE  = os.path.join(BASE_DIR, 'data', 'processed', 'all_venues_features.parquet')
MODEL_DIR  = os.path.join(BASE_DIR, 'models')
CALIB_PATH = os.path.join(MODEL_DIR, 'conditional_logit_calibrated.pkl')


def predict_calib(df, artifact):
    """校正済み確率を返す"""
    ir        = artifact['isotonic']
    beta      = artifact['coef']
    feat_cols = artifact['feat_cols']

    X, y, gs, n, *_ = prepare(
        df, feat_cols,
        scaler=artifact['scaler'],
        poly2=artifact.get('poly2'),
        inter_scaler2=artifact.get('inter_scaler2'),
        top_idx=artifact.get('top_idx'),
        poly3=artifact.get('poly3'),
        inter_scaler3=artifact.get('inter_scaler3'),
        top_idx3=artifact.get('top_idx3'),
        fit=False,
    )
    raw_prob  = segment_softmax(X @ beta, gs, n)
    calib_prob = ir.predict(raw_prob)

    df_out = df.sort_values('race_id').reset_index(drop=True)
    df_out['model_prob']  = raw_prob
    df_out['calib_prob']  = calib_prob
    df_out['y']           = y
    return df_out


def roi_by_year(d, label):
    print(f'\n  {label}')
    for yr in sorted(d['yr'].unique()):
        s   = d[d['yr'] == yr]
        won = s['着順_num'] == 1
        r   = (s.loc[won, 'odds_num'] * 100).sum() / (len(s) * 100) - 1
        print(f'    20{yr:02d}: {len(s):5d}R  win={won.mean():.3f}  ROI={r:+.3f}')
    won = d['着順_num'] == 1
    r   = (d.loc[won, 'odds_num'] * 100).sum() / (len(d) * 100) - 1
    print(f'    Total: {len(d):5d}R  win={won.mean():.3f}  ROI={r:+.3f}')
    return r


def main():
    print('データ読み込み...')
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

    print('校正済みモデル読み込み...')
    with open(CALIB_PATH, 'rb') as f:
        artifact = pickle.load(f)

    oos = df[df['日付_num'] >= 230101].copy()
    print(f'OOS サイズ: {len(oos):,}行')

    print('確率計算中...')
    pred = predict_calib(oos, artifact)

    oos_s = oos.sort_values('race_id').reset_index(drop=True)
    pred['odds_num']    = pd.to_numeric(oos_s['単勝オッズ'], errors='coerce').values
    pred['market_prob'] = 1.0 / pred['odds_num']
    pred['ev_score']    = pred['calib_prob'] - pred['market_prob'] * 0.80
    pred['exp_return']  = pred['calib_prob'] * pred['odds_num']
    pred['yr']          = pred['日付_num'] // 10000

    print('\n' + '='*60)
    print('rank=1 選択基準別 OOS ROI 比較')
    print('='*60)

    results = {}

    # --- 戦略1: model_prob (生スコア) ---
    pred['rank_raw'] = pred.groupby('race_id')['model_prob'].rank(
        ascending=False, method='first')
    top1_raw = pred[pred['rank_raw'] == 1]
    r1 = roi_by_year(top1_raw, '1. rank=1 by model_prob (raw)')
    results['model_prob'] = r1

    # --- 戦略2: calib_prob (校正済み) ---
    pred['rank_calib'] = pred.groupby('race_id')['calib_prob'].rank(
        ascending=False, method='first')
    top1_calib = pred[pred['rank_calib'] == 1]
    r2 = roi_by_year(top1_calib, '2. rank=1 by calib_prob (isotonic)')
    results['calib_prob'] = r2

    # --- 戦略3: ev_score ---
    pred['rank_ev'] = pred.groupby('race_id')['ev_score'].rank(
        ascending=False, method='first')
    top1_ev = pred[pred['rank_ev'] == 1]
    r3 = roi_by_year(top1_ev, '3. rank=1 by EV score (calib - market×0.8)')
    results['ev_score'] = r3

    # --- 戦略4: expected_return = calib_prob × odds ---
    pred['rank_er'] = pred.groupby('race_id')['exp_return'].rank(
        ascending=False, method='first')
    top1_er = pred[pred['rank_er'] == 1]
    r4 = roi_by_year(top1_er, '4. rank=1 by expected return (calib × odds)')
    results['exp_return'] = r4

    print('\n' + '='*60)
    print('サマリ')
    print('='*60)
    for name, roi in results.items():
        mark = ' ← 目標達成!' if roi >= -0.05 else ''
        print(f'  {name:25s}: ROI={roi:+.3f}{mark}')
    print(f'  目標: ROI >= -0.050')


if __name__ == '__main__':
    main()
