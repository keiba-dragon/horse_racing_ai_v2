# coding: utf-8
"""surface_clogit v1 OOS 結果を様々な軸で分解して弱点を特定"""
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


def print_breakdown(df, col, label, n_min=300):
    print(f'\n=== {label} ===')
    for val in sorted(df[col].unique()):
        s   = df[df[col] == val]
        if len(s) < n_min:
            continue
        won = s['着順_num'] == 1
        r   = (s.loc[won, 'odds_num'] * 100).sum() / (len(s) * 100) - 1
        print(f'  {val}: {len(s):>6,}R  win={won.mean():.3f}  ROI={r:+.3f}')


def main():
    with open(os.path.join(MODEL_DIR, 'surface_clogit.pkl'), 'rb') as f:
        pkg = pickle.load(f)
    artifacts = pkg['artifacts']
    feat_cols = pkg['feat_cols']

    df = pd.read_parquet(DATA_FILE)
    df['日付_num'] = pd.to_numeric(df['日付'], errors='coerce')
    df['着順_num'] = pd.to_numeric(df['着順_num'], errors='coerce')
    df['頭数']    = pd.to_numeric(df['頭数'], errors='coerce')
    df = df.dropna(subset=['日付_num', '着順_num'])
    df = df[df['着順_num'] < 99]
    df['race_id'] = (df['日付_num'].astype(int).astype(str) + '_' +
                     df['開催'].astype(str).str.strip() + '_' +
                     df['Ｒ'].astype(str).str.strip())
    df = add_pace_features(df)
    df = add_new_features(df)
    df['surface'] = df['距離'].astype(str).str.strip().str.extract(r'^([芝ダ])')[0].fillna('不明')
    df = df[df['surface'].isin(['芝', 'ダ'])].copy()
    df['venue']   = df['開催'].astype(str).str.strip().str[1:-1]

    # 距離 (m) を数値化
    df['dist_m'] = df['距離'].astype(str).str.extract(r'(\d+)')[0].astype(float)
    df['dist_bucket'] = pd.cut(df['dist_m'],
                                bins=[0, 1200, 1400, 1600, 1800, 2000, 9999],
                                labels=['≤1200', '1201-1400', '1401-1600', '1601-1800', '1801-2000', '2001+'])
    # 頭数バケット
    df['field_bucket'] = pd.cut(df['頭数'],
                                 bins=[0, 9, 12, 14, 16, 99],
                                 labels=['≤9', '10-12', '13-14', '15-16', '17+'])
    # レース名からクラス判定
    df['race_name'] = df['レース名'].astype(str)
    def classify_race(name):
        if any(k in name for k in ['新馬', 'maiden']): return '新馬'
        if '未勝利' in name: return '未勝利'
        if '1勝' in name or '500万' in name: return '1勝クラス'
        if '2勝' in name or '1000万' in name: return '2勝クラス'
        if '3勝' in name or '1600万' in name: return '3勝クラス'
        if any(k in name for k in ['オープン', 'OP', '重賞', 'Ｇ', 'G', '記念', '賞']): return 'OP以上'
        return 'その他'
    df['race_class'] = df['race_name'].apply(classify_race)

    # OOS予測を生成
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
        pred['yr'] = pred['日付_num'] // 10000
        pred['surface'] = surf
        all_pred.append(pred)

    all_pred = pd.concat(all_pred, ignore_index=True)
    all_pred['rank_model'] = all_pred.groupby('race_id')['calib_prob'].rank(ascending=False, method='first')
    top1 = all_pred[all_pred['rank_model'] == 1].copy()

    won = top1['着順_num'] == 1
    total_roi = (top1.loc[won, 'odds_num'] * 100).sum() / (len(top1) * 100) - 1
    print(f'OOS 全体: {len(top1):,}R  win={won.mean():.3f}  ROI={total_roi:+.3f}')

    # 各軸で分解
    print_breakdown(top1, 'race_class', 'レースクラス別')
    print_breakdown(top1, 'dist_bucket', '距離バケット別')
    print_breakdown(top1, 'field_bucket', '頭数バケット別')
    print_breakdown(top1, '今回_馬場_num', '馬場状態別 (0=良,1=稍,2=重,3=不良)', n_min=100)

    # 大きなサブグループ: 芝×クラス
    print('\n=== 芝ダ × クラス ===')
    for surf in ['芝', 'ダ']:
        for cls in ['新馬', '未勝利', '1勝クラス', '2勝クラス', '3勝クラス', 'OP以上']:
            s = top1[(top1['surface'] == surf) & (top1['race_class'] == cls)]
            if len(s) < 200:
                continue
            won = s['着順_num'] == 1
            r   = (s.loc[won, 'odds_num'] * 100).sum() / (len(s) * 100) - 1
            print(f'  {surf} {cls}: {len(s):>5,}R  win={won.mean():.3f}  ROI={r:+.3f}')

    # 勝率 × オッズ分布（pick の質を見る）
    print('\n=== rank=1 pick のオッズ分布 ===')
    for lo, hi, lbl in [(0,3,'≤3倍'),(3,6,'3-6倍'),(6,10,'6-10倍'),(10,20,'10-20倍'),(20,999,'20倍+')]:
        s = top1[(top1['odds_num'] > lo) & (top1['odds_num'] <= hi)]
        if len(s) < 50:
            continue
        won = s['着順_num'] == 1
        r   = (s.loc[won, 'odds_num'] * 100).sum() / (len(s) * 100) - 1
        print(f'  {lbl}: {len(s):>5,}R  win={won.mean():.3f}  ROI={r:+.3f}')


if __name__ == '__main__':
    main()
