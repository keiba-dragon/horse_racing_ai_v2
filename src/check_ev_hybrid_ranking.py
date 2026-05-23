# coding: utf-8
"""
surface_clogit v1 OOS 予測を使って、
ランキングスコアのブレンド比率を評価
score = calib_prob - market_factor * market_prob

rank=1全買い (pure calib_prob)    = v1 ベースライン -13.2%
rank=1 by EV (factor=0.8)        = 失敗
rank=1 by hybrid (factor=0~0.8)  = ここを評価
"""
import sys, os, pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import save_lambdarank_pace as _slp
from save_lambdarank_pace import add_pace_features, EXCLUDE, ODDS_REMOVE
from save_conditional_logit import add_new_features, segment_softmax, prepare

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, 'data', 'processed', 'all_venues_features.parquet')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

def main():
    with open(os.path.join(MODEL_DIR, 'surface_clogit.pkl'), 'rb') as f:
        pkg = pickle.load(f)
    artifacts = pkg['artifacts']
    feat_cols = pkg['feat_cols']

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

    all_pred = []
    for surf in ['芝', 'ダ']:
        art   = artifacts[surf]
        oos_s = df[(df['日付_num'] >= 230101) & (df['surface'] == surf)].sort_values('race_id').reset_index(drop=True)
        X_oo, y_oo, gs_oo, n_oo, *_ = prepare(
            oos_s, art['feat_cols'],
            scaler=art['scaler'], poly2=art['poly2'], inter_scaler2=art['inter_scaler2'],
            top_idx=art['top_idx'],
            poly3=None, inter_scaler3=None, top_idx3=None, fit=False)
        raw   = segment_softmax(X_oo @ art['coef'], gs_oo, n_oo)
        calib = art['isotonic'].predict(raw)
        pred  = oos_s.copy()
        pred['model_prob'] = raw
        pred['calib_prob'] = calib
        pred['odds_num']   = pd.to_numeric(oos_s['単勝オッズ'], errors='coerce').values
        pred['market_prob'] = 1.0 / pred['odds_num'].clip(lower=1.0)
        pred['yr'] = pred['日付_num'] // 10000
        pred['surface'] = surf
        all_pred.append(pred)

    all_pred = pd.concat(all_pred, ignore_index=True)

    print('=== hybrid ranking: score = calib_prob - factor × market_prob ===')
    print(f'{"factor":>8} {"R数":>6} {"win":>6} {"ROI":>8}  年別 (2023/2024/2025/2026)')
    print('-'*70)

    for factor in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.80]:
        all_pred['score'] = all_pred['calib_prob'] - factor * all_pred['market_prob']
        all_pred['rank_score'] = all_pred.groupby('race_id')['score'].rank(ascending=False, method='first')
        top1 = all_pred[all_pred['rank_score'] == 1]
        won  = top1['着順_num'] == 1
        total_roi = (top1.loc[won, 'odds_num'] * 100).sum() / (len(top1) * 100) - 1

        yr_rois = []
        for yr in [23, 24, 25, 26]:
            s = top1[top1['yr'] == yr]
            if len(s) == 0:
                yr_rois.append('  N/A')
                continue
            w = s['着順_num'] == 1
            r = (s.loc[w, 'odds_num'] * 100).sum() / (len(s) * 100) - 1
            yr_rois.append(f'{r:+.3f}')

        print(f'{factor:>8.2f} {len(top1):>6,} {won.mean():>6.3f} {total_roi:>+8.3f}  {" / ".join(yr_rois)}')

    # 補足: val期間 (2021-2022) でのハイブリッドランキング
    print('\n=== val期間 (2021-2022) での同分析 ===')
    val_pred = []
    for surf in ['芝', 'ダ']:
        art   = artifacts[surf]
        val_s = df[(df['日付_num'] >= 210101) & (df['日付_num'] <= 221231) & (df['surface'] == surf)].sort_values('race_id').reset_index(drop=True)
        X_va, y_va, gs_va, n_va, *_ = prepare(
            val_s, art['feat_cols'],
            scaler=art['scaler'], poly2=art['poly2'], inter_scaler2=art['inter_scaler2'],
            top_idx=art['top_idx'],
            poly3=None, inter_scaler3=None, top_idx3=None, fit=False)
        raw   = segment_softmax(X_va @ art['coef'], gs_va, n_va)
        calib = art['isotonic'].predict(raw)
        pred  = val_s.copy()
        pred['calib_prob'] = calib
        pred['odds_num']   = pd.to_numeric(val_s['単勝オッズ'], errors='coerce').values
        pred['market_prob'] = 1.0 / pred['odds_num'].clip(lower=1.0)
        val_pred.append(pred)
    val_pred = pd.concat(val_pred, ignore_index=True)

    print(f'{"factor":>8} {"R数":>6} {"win":>6} {"ROI":>8}')
    print('-'*40)
    for factor in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.80]:
        val_pred['score'] = val_pred['calib_prob'] - factor * val_pred['market_prob']
        val_pred['rank_score'] = val_pred.groupby('race_id')['score'].rank(ascending=False, method='first')
        top1 = val_pred[val_pred['rank_score'] == 1]
        won  = top1['着順_num'] == 1
        total_roi = (top1.loc[won, 'odds_num'] * 100).sum() / (len(top1) * 100) - 1
        print(f'{factor:>8.2f} {len(top1):>6,} {won.mean():>6.3f} {total_roi:>+8.3f}')


if __name__ == '__main__':
    main()
