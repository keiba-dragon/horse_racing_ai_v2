# coding: utf-8
"""
try_mean_imputation.py - NaN の平均値補完テスト
  現行: fillna(0) → StandardScaler → NaN馬は実質マイナス大きく評価
  改善: fillna(train_mean) → StandardScaler → NaN馬は平均扱い (0 after scaling)
  BASE_24 + BASE_25(+TI_slope) で両方検証
"""
import sys, os
import numpy as np
import pandas as pd

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from save_conditional_logit import (
    prepare, segment_softmax, BASE_DIR, DATA_FILE, LR, N_EPOCHS, PATIENCE
)
from save_v3 import add_computed_features, calc_roi

BASE_24 = [
    '近5走_クラス調整_平均着順', '近5走_タイム指数_max', '1走前_タイム指数', '前走着差タイム',
    '騎手コース_r100_勝率', '1走前_クラス調整着順', '調教師コース_r100_勝率',
    '1走前_RPCI', '1走前_上3F地点差', '斤量', '種牡馬_勝率',
    '間隔_長_flag', '1走前_脚質_num', '騎手変更', '馬番',
    '近3走_複勝率', '調教師_r200_複勝率',
    'ブリンカー変更', '2走前_クラス差', '4走前_クラス差', '1走前_馬場状態',
    '性別_num', '所属_num', 'キャリア_浅い',
]
BASE_25 = BASE_24 + ['タイム指数_近5走_slope']
L2 = 0.006


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
    baba_map = {'良': 0, '稍重': 1, '重': 2, '不良': 3}
    for col in df.columns:
        if '馬場状態' in col:
            df[col] = df[col].map(baba_map)
    return df


def apply_mean_imputation(df_trn, df_val, oos_parts_dict, feats):
    """訓練データの各特徴量平均でNaNを埋める。val/OOSにも同じ平均を適用"""
    impute_vals = {}
    for c in feats:
        if c in df_trn.columns:
            mean_val = df_trn[c].mean()  # NaN除く平均
            impute_vals[c] = mean_val

    df_trn_imp = df_trn.copy()
    df_val_imp = df_val.copy()
    for c, v in impute_vals.items():
        if not np.isnan(v):
            df_trn_imp[c] = df_trn_imp[c].fillna(v)
            df_val_imp[c] = df_val_imp[c].fillna(v)

    oos_imp = {}
    for period, df_p in oos_parts_dict.items():
        df_p_imp = df_p.copy()
        for c, v in impute_vals.items():
            if c in df_p_imp.columns and not np.isnan(v):
                df_p_imp[c] = df_p_imp[c].fillna(v)
        oos_imp[period] = df_p_imp

    return df_trn_imp, df_val_imp, oos_imp


def _loss_grad(beta, X, y, gs, n, nr, l2=0.0):
    probs = segment_softmax(X @ beta, gs, n)
    loss = -np.sum(y * np.log(np.clip(probs, 1e-15, 1.0))) / nr + l2 * np.dot(beta, beta)
    grad = -(X.T @ (y - probs)) / nr + 2 * l2 * beta
    return loss, grad


def adam_fit(X_tr, y_tr, gs_tr, n_tr, nr_tr, X_va, y_va, gs_va, n_va, nr_va, l2=0.0):
    d = X_tr.shape[1]
    beta, m, v = np.zeros(d), np.zeros(d), np.zeros(d)
    b1, b2, eps = 0.9, 0.999, 1e-8
    t, best_val, best_beta, no_imp = 0, np.inf, np.zeros(d), 0
    for epoch in range(1, N_EPOCHS + 1):
        _, grad = _loss_grad(beta, X_tr, y_tr, gs_tr, n_tr, nr_tr, l2)
        t += 1
        m = b1*m + (1-b1)*grad
        v = b2*v + (1-b2)*grad**2
        beta -= LR * (m/(1-b1**t)) / (np.sqrt(v/(1-b2**t)) + eps)
        if epoch % 10 == 0:
            vl, _ = _loss_grad(beta, X_va, y_va, gs_va, n_va, nr_va, l2=0.0)
            if vl < best_val:
                best_val, best_beta, no_imp = vl, beta.copy(), 0
            else:
                no_imp += 1
            if no_imp >= PATIENCE // 10:
                break
    return best_beta


def evaluate(df_trn, df_val, oos_parts, feats, l2=0.0):
    valid = [c for c in feats if c in df_trn.columns and df_trn[c].isna().mean() <= 0.65]
    X_tr, y_tr, gs_tr, n_tr, nr_tr, scaler, *_ = prepare(
        df_trn, valid, top_idx=None, top_idx3=None, fit=True)
    X_va, y_va, gs_va, n_va, nr_va, *_ = prepare(
        df_val, valid, scaler=scaler, top_idx=None, top_idx3=None)
    beta = adam_fit(X_tr, y_tr, gs_tr, n_tr, nr_tr,
                    X_va, y_va, gs_va, n_va, nr_va, l2=l2)
    oos_roi = {}
    for period, df_p in oos_parts.items():
        if len(df_p) == 0:
            continue
        valid_p = [c for c in valid if c in df_p.columns]
        X_p, _, gs_p, n_p, *_ = prepare(df_p, valid_p, scaler=scaler,
                                          top_idx=None, top_idx3=None)
        scored = df_p.sort_values('race_id').reset_index(drop=True)
        scored['prob'] = segment_softmax(X_p @ beta, gs_p, n_p)
        scored['rank'] = scored.groupby('race_id')['prob'].rank(ascending=False, method='first')
        top1 = scored[scored['rank'] == 1]
        roi, wins = calc_roi(top1)
        oos_roi[period] = (roi, len(top1), wins)
    return oos_roi


def show(label, roi, c0=None):
    r24 = roi.get('2324', (0, 1, 0))[0]
    r25, n25, _ = roi.get('2025', (0, 1, 0))
    r26, n26, _ = roi.get('2026', (0, 1, 0))
    comb = (r25*n25 + r26*n26) / (n25+n26)
    delta = f'  Δ={comb-c0:+.2%}' if c0 is not None else ''
    mark = ' ★' if c0 is not None and comb > c0 else ''
    print(f'  {label:<45} 2324:{r24:+.2%}  2025:{r25:+.2%}  2026:{r26:+.2%}  25+26:{comb:+.2%}{delta}{mark}')
    return comb


def main():
    df = load_segment()
    df_trn = df[(df['日付_num'] >= 130101) & (df['日付_num'] < 220101)]
    df_val = df[(df['日付_num'] >= 220101) & (df['日付_num'] <= 221231)]
    oos = df[df['日付_num'] >= 230101]
    oos_parts = {
        '2324': oos[oos['日付_num'] < 250101],
        '2025': oos[(oos['日付_num'] >= 250101) & (oos['日付_num'] < 260101)],
        '2026': oos[oos['日付_num'] >= 260101],
    }

    print('=== NaN補完方法比較 ===')
    print()

    # BASE_24 ベースライン (現行: fillna(0))
    roi0 = evaluate(df_trn, df_val, oos_parts, BASE_24, l2=L2)
    c0 = show('BASE_24  現行fillna(0)', roi0)

    # BASE_24 + 平均補完
    df_trn_m, df_val_m, oos_m = apply_mean_imputation(df_trn, df_val, oos_parts, BASE_24)
    roi = evaluate(df_trn_m, df_val_m, oos_m, BASE_24, l2=L2)
    c = show('BASE_24  平均補完', roi, c0)

    print()

    # BASE_25 ベースライン
    roi1 = evaluate(df_trn, df_val, oos_parts, BASE_25, l2=L2)
    c1 = show('BASE_25  現行fillna(0)', roi1)

    # BASE_25 + 平均補完
    df_trn_m25, df_val_m25, oos_m25 = apply_mean_imputation(df_trn, df_val, oos_parts, BASE_25)
    roi = evaluate(df_trn_m25, df_val_m25, oos_m25, BASE_25, l2=L2)
    c = show('BASE_25  平均補完', roi, c1)

    print()

    # NaN補完 + 全候補再テスト (BASE_25 base, mean imputation)
    print('--- 平均補完 + 追加特徴量テスト (BASE_25 base) ---')
    ALL_EXTRA = [
        '近5走_タイム指数平均', '近5走_タイム指数_min', '近5走_タイム指数_range',
        '近5走_タイム指数_std', '近5走_上り3F_min', '近5走_上り3F平均',
        '近5走_上り3F指数平均', '1走前_上り3F_指数', 'タイム指数_近3走_slope',
        '上り3F_近3走_slope', '1走前_クラス_rank', '1走前_クラス差',
        '2走前_タイム指数', '3走前_タイム指数', '5走前_タイム指数',
        '馬体重', '馬体重トレンド_近5走',
        '4角位置_近3走_slope', 'タイム指数_加速度',
        '着順_近3走_slope', '着順_近5走_slope', '近走_改善トレンド',
    ]
    avail_extra = [c for c in ALL_EXTRA if c in df.columns]
    best_c25 = c1
    best_col = None

    for c_extra in avail_extra:
        feats_trial = BASE_25 + [c_extra]
        nan_frac = df_trn[c_extra].isna().mean()
        if nan_frac > 0.65:
            continue
        df_trn_t, df_val_t, oos_t = apply_mean_imputation(df_trn, df_val, oos_parts, feats_trial)
        roi = evaluate(df_trn_t, df_val_t, oos_t, feats_trial, l2=L2)
        comb = show(f'+{c_extra}', roi, c1)
        if comb > best_c25:
            best_c25 = comb
            best_col = c_extra

    print()
    if best_col:
        print(f'最良追加: {best_col} → {best_c25:+.2%}')
    else:
        print('追加有望なし')

    print()
    print(f'21F基準(-19.70%) から最良: {max(c0, c, c1)-(-0.197):+.2%}')


if __name__ == '__main__':
    main()
