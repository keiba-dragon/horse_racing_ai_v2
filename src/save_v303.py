# coding: utf-8
"""
exp_v303_best を保存するスクリプト
- 5セグメント別最適特徴量（feature_search_v3 結果）
- 芝_長距離はOOS過学習が確認されたためベースライン特徴量を採用
- train=2013-2022, val=2022, save to models/exp_v303_best/
"""
import sys, os, pickle, json
import numpy as np
import pandas as pd

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from save_conditional_logit import (
    prepare, segment_softmax, BASE_DIR, DATA_FILE, LR, N_EPOCHS, PATIENCE
)
from sklearn.isotonic import IsotonicRegression

OUT_DIR = os.path.join(BASE_DIR, 'models', 'exp_v303_best')
os.makedirs(OUT_DIR, exist_ok=True)

BASE_FEATURES = [
    '1走前_タイム指数', '1走前_上り3F', '前走着差タイム', '1走前_RPCI',
    '近5走_タイム指数平均', '近5走_タイム指数_max', 'タイム指数_近3走_slope',
    '1走前_クラス調整着順', '近5走_クラス調整_平均着順',
    '馬番', 'コース枠_r200_勝率',
    '騎手コース_r100_勝率', '騎手変更', '調教師コース_r100_勝率',
    'コース脚質_r200_勝率', '1走前_脚質_num', '種牡馬_勝率', '種牡馬_ダ_勝率',
    '斤量', '芝ダ転向', '間隔',
]

# 芝_長距離: 2025ホールドアウトで崩壊(-22%)確認 → ベースラインに戻す
SEG_FEATS = {
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
    '芝_長距離': list(BASE_FEATURES),   # OOS過学習確認 → ベースラインを維持
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

SEGMENTS = [('芝', '短距離'), ('芝', '中距離'), ('芝', '長距離'),
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
    print(f'読み込み: {DATA_FILE}')
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
    print(f'有効行: {len(df):,}')
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
    return best_beta


def main():
    df = load_data()
    all_val_top1 = []
    all_oos_top1 = []
    meta = {'segments': {}, 'seg_feats': SEG_FEATS}

    for surf, dist_band in SEGMENTS:
        seg_key = f'{surf}_{dist_band}'
        feats   = SEG_FEATS[seg_key]
        df_s    = df[(df['surface'] == surf) & (df['dist_band'] == dist_band)].copy()
        valid   = [c for c in feats if c in df_s.columns and df_s[c].isna().mean() <= 0.65]
        missing = [c for c in feats if c not in df_s.columns]
        if missing:
            print(f'  [{seg_key}] 列なし: {missing}')

        trn = df_s[(df_s['日付_num'] >= 130101) & (df_s['日付_num'] < 220101)]
        val = df_s[(df_s['日付_num'] >= 220101) & (df_s['日付_num'] <= 221231)]
        oos = df_s[df_s['日付_num'] >= 230101].copy()

        if len(trn) < 300 or len(val) < 30:
            print(f'  [{seg_key}] データ不足スキップ')
            continue

        print(f'\n[{seg_key}] {len(valid)}特徴量  train:{len(trn)} val:{len(val)} oos:{len(oos)}')
        X_tr, y_tr, gs_tr, n_tr, nr_tr, scaler, *_ = prepare(trn, valid, top_idx=None, top_idx3=None, fit=True)
        X_va, y_va, gs_va, n_va, nr_va, *_          = prepare(val, valid, scaler=scaler, top_idx=None, top_idx3=None)

        beta = adam_opt(X_tr, y_tr, gs_tr, n_tr, nr_tr, X_va, y_va, gs_va, n_va, nr_va)

        # val スコアリング → Isotonic Regression
        val_sorted = val.sort_values('race_id').reset_index(drop=True)
        val_probs  = segment_softmax(X_va @ beta, gs_va, n_va)
        val_sorted['raw_prob'] = val_probs
        val_sorted['rank_model'] = val_sorted.groupby('race_id')['raw_prob'].rank(ascending=False, method='first')
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(val_sorted['raw_prob'].values, (val_sorted['着順_num'] == 1).astype(float).values)
        all_val_top1.append(val_sorted[val_sorted['rank_model'] == 1])

        # OOS スコアリング
        if len(oos) > 0:
            X_oo, y_oo, gs_oo, n_oo, *_ = prepare(oos, valid, scaler=scaler, top_idx=None, top_idx3=None)
            oos_sorted = oos.sort_values('race_id').reset_index(drop=True)
            oos_probs  = segment_softmax(X_oo @ beta, gs_oo, n_oo)
            oos_sorted['model_prob'] = oos_probs
            oos_sorted['calib_prob'] = iso.predict(oos_probs)
            oos_sorted['rank_model'] = oos_sorted.groupby('race_id')['model_prob'].rank(ascending=False, method='first')
            oos_sorted['odds_num']   = pd.to_numeric(oos_sorted['単勝オッズ'], errors='coerce')
            top1 = oos_sorted[oos_sorted['rank_model'] == 1]
            won  = top1['着順_num'] == 1
            roi  = (top1.loc[won, 'odds_num'] * 100).sum() / (len(top1) * 100) - 1
            print(f'  OOS: {len(top1)}R  ROI={roi:+.4f}  勝率={won.mean():.1%}')
            all_oos_top1.append(top1)

        # 保存
        seg_data = {
            'beta': beta,
            'scaler': scaler,
            'iso': iso,
            'feat_cols': valid,
            'seg_key': seg_key,
        }
        seg_path = os.path.join(OUT_DIR, f'{seg_key}.pkl')
        with open(seg_path, 'wb') as f:
            pickle.dump(seg_data, f)
        print(f'  保存: {seg_path}')
        meta['segments'][seg_key] = {'feat_cols': valid, 'n_feats': len(valid)}

    # 合計OOS ROI
    if all_oos_top1:
        combined = pd.concat(all_oos_top1, ignore_index=True)
        won_all  = combined['着順_num'] == 1
        total_roi = (combined.loc[won_all, 'odds_num'] * 100).sum() / (len(combined) * 100) - 1
        print(f'\n=== OOS合計: {len(combined)}R  ROI={total_roi:+.4f} ===')
        meta['oos_roi'] = float(total_roi)
        meta['oos_n']   = len(combined)

    # メタデータ保存
    meta_path = os.path.join(OUT_DIR, 'meta.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump({k: v for k, v in meta.items() if k != 'seg_feats'}, f, ensure_ascii=False, indent=2)
    # seg_feats は別保存
    feats_path = os.path.join(OUT_DIR, 'seg_feats.json')
    with open(feats_path, 'w', encoding='utf-8') as f:
        json.dump(SEG_FEATS, f, ensure_ascii=False, indent=2)
    print(f'\nメタデータ保存: {meta_path}')
    print(f'特徴量マップ保存: {feats_path}')
    print('exp_v303_best 保存完了')


if __name__ == '__main__':
    main()
