# coding: utf-8
"""
save_da_16f.py - ダ_中長距離 最良モデル (16特徴) を保存
出力: models/exp_da16f/ダ_中長距離.pkl

特徴量選択経緯:
  v303(20特徴) → dead3本削除 → 17F
  17F → backward: 上り3F削除 → 16F
  16F → backward: 近5走_タイム指数平均削除 → 15F
  15F → forward: 近3走_複勝率追加 → 16F (最終)
  検証: backward elimination で両期間合意の削除なし = 安定

性能:
  2324: -24.82%  2025: -23.00%  2026: -24.15%  25+26: -23.32%
  vs v303: +4.81pp改善
"""
import sys, os, pickle, time
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from save_conditional_logit import (
    prepare, segment_softmax, BASE_DIR, DATA_FILE, LR, N_EPOCHS, PATIENCE
)
from save_v3 import add_computed_features, calc_roi

OUT_DIR = os.path.join(BASE_DIR, 'models', 'exp_da16f')
os.makedirs(OUT_DIR, exist_ok=True)

FINAL_16 = [
    '近5走_クラス調整_平均着順',
    '近5走_タイム指数_max', '1走前_タイム指数', '前走着差タイム',
    '騎手コース_r100_勝率', '1走前_クラス調整着順', '調教師コース_r100_勝率',
    '1走前_RPCI', '1走前_上3F地点差', '斤量', '種牡馬_勝率',
    '間隔_長_flag', '1走前_脚質_num', '騎手変更', '馬番',
    '近3走_複勝率',
]


def load_segment():
    df = pd.read_parquet(DATA_FILE)
    df['日付_num'] = pd.to_numeric(df['日付'], errors='coerce')
    df['着順_num'] = pd.to_numeric(df['着順_num'], errors='coerce')
    df = df.dropna(subset=['日付_num', '着順_num'])
    df = df[df['着順_num'] < 99]
    df['race_id'] = (df['日付_num'].astype(int).astype(str) + '_' +
                     df['開催'].astype(str).str.strip() + '_' +
                     df['Ｒ'].astype(str).str.strip())
    df = df[df['開催'].notna()].copy()
    df['surface'] = (df['距離'].astype(str).str.strip()
                      .str.extract(r'^([芝ダ])')[0].fillna('不明'))
    dm = pd.to_numeric(df['距離'].astype(str).str.extract(r'(\d+)')[0], errors='coerce')
    df = df[(df['surface'] == 'ダ') & (dm > 1400)].copy()
    df['dist_m'] = dm[df.index]
    if 'クラス_rank' in df.columns:
        df = df[df['クラス_rank'] != 1.0].copy()
    df = add_computed_features(df)
    return df


def _loss_grad(beta, X, y, gs, n, nr):
    probs = segment_softmax(X @ beta, gs, n)
    res   = y - probs
    loss  = -np.sum(y * np.log(np.clip(probs, 1e-15, 1.0))) / nr
    grad  = -(X.T @ res) / nr
    return loss, grad


def adam_fit(X_tr, y_tr, gs_tr, n_tr, nr_tr, X_va, y_va, gs_va, n_va, nr_va):
    d = X_tr.shape[1]
    beta, m, v = np.zeros(d), np.zeros(d), np.zeros(d)
    b1, b2, eps = 0.9, 0.999, 1e-8
    t, best_val, best_beta, no_imp = 0, np.inf, np.zeros(d), 0
    for epoch in range(1, N_EPOCHS + 1):
        _, grad = _loss_grad(beta, X_tr, y_tr, gs_tr, n_tr, nr_tr)
        t += 1
        m = b1*m + (1-b1)*grad
        v = b2*v + (1-b2)*grad**2
        beta -= LR * (m/(1-b1**t)) / (np.sqrt(v/(1-b2**t)) + eps)
        if epoch % 10 == 0:
            vl, _ = _loss_grad(beta, X_va, y_va, gs_va, n_va, nr_va)
            if vl < best_val:
                best_val, best_beta, no_imp = vl, beta.copy(), 0
            else:
                no_imp += 1
            if no_imp >= PATIENCE // 10:
                break
    return best_beta, best_val


def main():
    t0 = time.time()
    df = load_segment()
    print(f'ダ_中長距離: {len(df):,}行')

    df_trn = df[(df['日付_num'] >= 130101) & (df['日付_num'] < 220101)]
    df_val = df[(df['日付_num'] >= 220101) & (df['日付_num'] <= 221231)]
    oos    = df[df['日付_num'] >= 230101]
    oos_parts = {
        '2324': oos[oos['日付_num'] < 250101],
        '2025': oos[(oos['日付_num'] >= 250101) & (oos['日付_num'] < 260101)],
        '2026': oos[oos['日付_num'] >= 260101],
    }
    print(f'trn:{len(df_trn):,} val:{len(df_val):,}')

    # 有効特徴確認
    valid = [c for c in FINAL_16 if c in df_trn.columns and df_trn[c].isna().mean() <= 0.65]
    print(f'有効特徴: {len(valid)}/{len(FINAL_16)}')
    missing = [c for c in FINAL_16 if c not in valid]
    if missing:
        print(f'除外: {missing}')

    # 訓練
    X_tr, y_tr, gs_tr, n_tr, nr_tr, scaler, *_ = prepare(
        df_trn, valid, top_idx=None, top_idx3=None, fit=True)
    X_va, y_va, gs_va, n_va, nr_va, *_ = prepare(
        df_val, valid, scaler=scaler, top_idx=None, top_idx3=None)
    beta, val_nll = adam_fit(X_tr, y_tr, gs_tr, n_tr, nr_tr,
                              X_va, y_va, gs_va, n_va, nr_va)
    print(f'valNLL={val_nll:.5f}')

    # isotonic calibration
    val_probs = segment_softmax(X_va @ beta, gs_va, n_va)
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(val_probs, y_va)

    # OOS評価
    print('\nOOS評価:')
    for period, df_p in oos_parts.items():
        if len(df_p) == 0:
            continue
        valid_p = [c for c in valid if c in df_p.columns]
        X_p, _, gs_p, n_p, *_ = prepare(df_p, valid_p, scaler=scaler,
                                          top_idx=None, top_idx3=None)
        scored = df_p.sort_values('race_id').reset_index(drop=True)
        raw_prob = segment_softmax(X_p @ beta, gs_p, n_p)
        calib_prob = iso.predict(raw_prob)
        scored['prob'] = raw_prob
        scored['calib_prob'] = calib_prob
        scored['rank'] = scored.groupby('race_id')['prob'].rank(
            ascending=False, method='first')
        top1 = scored[scored['rank'] == 1]
        roi, wins = calc_roi(top1)
        n = len(top1)
        print(f'  {period}: {n}R  ROI={roi:+.4f}  勝率={wins/n:.1%}')

    # ベータ係数
    print('\nベータ係数:')
    for i, c in enumerate(valid):
        print(f'  {c:<35} β={beta[i]:+.4f}')

    # 保存
    pkg = {
        'beta': beta,
        'scaler': scaler,
        'iso': iso,
        'feat_cols': valid,
        'seg_key': 'ダ_中長距離',
        'val_nll': val_nll,
        'description': '16特徴最終モデル: v303から上り3F・タイム指数平均削除、近3走_複勝率追加',
    }
    out_path = os.path.join(OUT_DIR, 'ダ_中長距離.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(pkg, f)
    print(f'\n保存完了: {out_path}')
    print(f'総時間: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
