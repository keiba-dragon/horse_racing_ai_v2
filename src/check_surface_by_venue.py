# coding: utf-8
"""surface_clogit v1 の OOS 結果を会場別に分解して分析"""
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
    # surface_clogit v1 モデルを読み込んで OOS 予測を再現
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
    df['venue']   = df['開催'].astype(str).str.strip().str[1:-1]
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
        pred['yr']         = pred['日付_num'] // 10000
        all_pred.append(pred)

    all_pred = pd.concat(all_pred, ignore_index=True)
    all_pred['rank_model'] = all_pred.groupby('race_id')['calib_prob'].rank(ascending=False, method='first')
    top1 = all_pred[all_pred['rank_model'] == 1].copy()

    print('=== surface_clogit v1: OOS 会場別 ROI ===')
    print(f'{"会場":6} {"サーフェス":6} {"R数":>6} {"win率":>6} {"ROI":>8}')
    print('-'*40)
    for surf in ['芝', 'ダ']:
        for venue in sorted(top1['venue'].unique()):
            s = top1[(top1['surface'] == surf) & (top1['venue'] == venue)]
            if len(s) < 50:
                continue
            won = s['着順_num'] == 1
            r   = (s.loc[won, 'odds_num'] * 100).sum() / (len(s) * 100) - 1
            print(f'{venue:6} {surf:6} {len(s):>6,}  {won.mean():.3f}  {r:+.3f}')

    print('\n=== 会場別合計 ===')
    for venue in sorted(top1['venue'].unique()):
        s = top1[top1['venue'] == venue]
        if len(s) < 50:
            continue
        won = s['着順_num'] == 1
        r   = (s.loc[won, 'odds_num'] * 100).sum() / (len(s) * 100) - 1
        print(f'{venue:6}: {len(s):>6,}R  win={won.mean():.3f}  ROI={r:+.3f}')

    print('\n=== 年別・会場別 ===')
    for yr in sorted(top1['yr'].unique()):
        s   = top1[top1['yr'] == yr]
        won = s['着順_num'] == 1
        r   = (s.loc[won, 'odds_num'] * 100).sum() / (len(s) * 100) - 1
        print(f'  20{int(yr):02d}: {len(s):5,}R  win={won.mean():.3f}  ROI={r:+.3f}')


if __name__ == '__main__':
    main()
