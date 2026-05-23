# coding: utf-8
"""
条件付きロジットモデルの確率校正 (Isotonic Regression)

conditional_logit.pkl の予測確率を val データで校正し、
校正済みモデルとして保存する。

校正後の確率を使って EV フィルタ精度を確認。
"""
import sys, io, os, json, pickle
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from save_lambdarank_pace import add_pace_features, EXCLUDE, ODDS_REMOVE
from save_conditional_logit import segment_softmax, prepare, add_new_features

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE  = os.path.join(BASE_DIR, 'data', 'processed', 'all_venues_features.parquet')
MODEL_DIR  = os.path.join(BASE_DIR, 'models')
CLOGIT_PATH = os.path.join(MODEL_DIR, 'conditional_logit.pkl')
CALIB_PATH  = os.path.join(MODEL_DIR, 'conditional_logit_calibrated.pkl')


def predict_probs(df, artifact):
    beta      = artifact['coef']
    feat_cols = artifact['feat_cols']

    X, y, gs, n, *_ = prepare(
        df, feat_cols,
        scaler=artifact['scaler'],
        poly2=artifact.get('poly2'), inter_scaler2=artifact.get('inter_scaler2'),
        top_idx=artifact.get('top_idx'),
        poly3=artifact.get('poly3'), inter_scaler3=artifact.get('inter_scaler3'),
        top_idx3=artifact.get('top_idx3'),
        fit=False,
    )
    scores = X @ beta
    probs  = segment_softmax(scores, gs, n)
    df_out = df.sort_values('race_id').reset_index(drop=True)
    df_out['model_prob'] = probs
    df_out['y'] = y
    return df_out


def roi_table(d, label):
    print(f'\n=== {label} ===')
    for yr in sorted(d['yr'].unique()):
        sub = d[d['yr'] == yr]
        won = sub['着順_num'] == 1
        r   = (sub.loc[won, 'odds_num'] * 100).sum() / (len(sub) * 100) - 1
        print(f'  20{yr:02d}: {len(sub):5d}R  win={won.mean():.3f}  ROI={r:+.3f}')
    won = d['着順_num'] == 1
    r   = (d.loc[won, 'odds_num'] * 100).sum() / (len(d) * 100) - 1
    print(f'  Total: {len(d):5d}R  win={won.mean():.3f}  ROI={r:+.3f}')


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

    with open(CLOGIT_PATH, 'rb') as f:
        artifact = pickle.load(f)

    # val (2021-2022) で isotonic regression を学習
    print('val データで確率を予測中...')
    val = df[(df['日付_num'] >= 210101) & (df['日付_num'] <= 221231)].copy()
    val_pred = predict_probs(val, artifact)
    val_pred['odds_num'] = pd.to_numeric(val['単勝オッズ'], errors='coerce')

    print(f'val サイズ: {len(val_pred):,}行')

    # isotonic regression: 予測確率 → 実際の勝利ラベルをフィット
    ir = IsotonicRegression(out_of_bounds='clip', increasing=True)
    ir.fit(val_pred['model_prob'].values, val_pred['y'].values)

    print('\n校正前 vs 校正後（val）:')
    val_pred['calib_prob'] = ir.predict(val_pred['model_prob'].values)
    val_pred['prob_bin'] = pd.qcut(val_pred['model_prob'], 10, labels=False, duplicates='drop')
    cal = val_pred.groupby('prob_bin').agg(
        raw_pred=('model_prob', 'mean'),
        calib_pred=('calib_prob', 'mean'),
        actual=('y', 'mean'),
        n=('y', 'count'),
    )
    for _, row in cal.iterrows():
        print(f'  raw={row.raw_pred:.3f}  calib={row.calib_pred:.3f}  actual={row.actual:.3f}  n={int(row.n)}')

    # OOS評価 (2023+) で校正済み確率を使用
    print('\nOOS データで確率を予測中...')
    oos = df[df['日付_num'] >= 230101].copy()
    oos_pred = predict_probs(oos, artifact)
    oos_pred['calib_prob'] = ir.predict(oos_pred['model_prob'].values)

    oos_pred['pop_num']     = pd.to_numeric(oos['単勝オッズ'], errors='coerce').values  # 注: oos は sort 前
    # 正しくマージ
    oos_sorted = oos.sort_values('race_id').reset_index(drop=True)
    oos_pred['pop_num']    = pd.to_numeric(oos_sorted['人気'], errors='coerce').values
    oos_pred['odds_num']   = pd.to_numeric(oos_sorted['単勝オッズ'], errors='coerce').values
    oos_pred['market_prob']= 1.0 / oos_pred['odds_num']
    oos_pred['yr']         = oos_pred['日付_num'] // 10000

    # rank は校正済み確率で決まるが isotonic は単調変換なので rank は変わらない
    oos_pred['rank_model'] = oos_pred.groupby('race_id')['calib_prob'].rank(
        ascending=False, method='first')

    # EV: 校正済み確率 vs 市場確率 (控除率 20% 考慮)
    oos_pred['ev_score'] = oos_pred['calib_prob'] - oos_pred['market_prob'] * 0.80

    top1 = oos_pred[oos_pred['rank_model'] == 1]
    roi_table(top1, 'rank=1 全体（校正済み）')
    roi_table(top1[top1['pop_num'] >= 2], 'rank=1 × 2番人気以下（校正済み）')

    print('\n--- EV フィルタ効果 ---')
    for thr in [0.0, 0.01, 0.02, 0.03, 0.05]:
        ev_top1 = oos_pred[(oos_pred['rank_model'] == 1) & (oos_pred['ev_score'] > thr)]
        if len(ev_top1) >= 200:
            won = ev_top1['着順_num'] == 1
            r   = (ev_top1.loc[won, 'odds_num'] * 100).sum() / (len(ev_top1) * 100) - 1
            print(f'  rank=1 × EV>{thr:.2f}: {len(ev_top1):5d}件  win={won.mean():.3f}  ROI={r:+.3f}')

    print('\n=== OOS キャリブレーション（校正後） ===')
    oos_pred['prob_bin'] = pd.qcut(oos_pred['calib_prob'], 10, labels=False, duplicates='drop')
    cal2 = oos_pred.groupby('prob_bin').agg(
        calib_pred=('calib_prob', 'mean'),
        actual=('着順_num', lambda x: (x == 1).mean()),
        n=('calib_prob', 'count'),
    )
    for _, row in cal2.iterrows():
        print(f'  calib={row.calib_pred:.3f}  actual={row.actual:.3f}  n={int(row.n):6d}')

    # 保存
    artifact_calib = dict(artifact)
    artifact_calib['isotonic'] = ir
    with open(CALIB_PATH, 'wb') as f:
        pickle.dump(artifact_calib, f)
    print(f'\n保存完了: {CALIB_PATH}')


if __name__ == '__main__':
    main()
