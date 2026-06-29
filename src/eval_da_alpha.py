# coding: utf-8
"""
eval_da_alpha.py - ダ_中長距離 alpha（L2正則化）スイープ
greedy_da_naga2 で採用した26特徴に対してalpha=0〜10を試す
"""
import sys, os, time
import numpy as np
import pandas as pd

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from save_conditional_logit import (
    prepare, segment_softmax, BASE_DIR, DATA_FILE, LR, N_EPOCHS, PATIENCE
)
from save_v3 import add_computed_features, calc_roi

FEATS_26 = [
    # base 17
    '近5走_クラス調整_平均着順', '1走前_上り3F', '近5走_タイム指数平均',
    '近5走_タイム指数_max', '1走前_タイム指数', '前走着差タイム',
    '騎手コース_r100_勝率', '1走前_クラス調整着順', '調教師コース_r100_勝率',
    '1走前_RPCI', '1走前_上3F地点差', '斤量', '種牡馬_勝率',
    '間隔_長_flag', '1走前_脚質_num', '騎手変更', '馬番',
    # domain 9 (greedy_da_naga2 で採用)
    '馬コース_r20_勝率',
    '馬コース_r20_複勝率',
    '同距離帯_平均着順_近5走',
    '同会場_平均着順_近5走',
    '馬距離_勝率',
    '騎手コース距離_r100_勝率',
    '4角位置_近3走_slope',
    '距離変化_前走',
    '同馬場_平均着順_近5走',
]

FEATS_17 = FEATS_26[:17]

ALPHAS = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]


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


def _loss_grad(beta, X, y, gs, n, nr, alpha):
    probs = segment_softmax(X @ beta, gs, n)
    res   = y - probs
    loss  = (-np.sum(y * np.log(np.clip(probs, 1e-15, 1.0))) + alpha * np.sum(beta**2)) / nr
    grad  = (-(X.T @ res) + 2 * alpha * beta) / nr
    return loss, grad


def adam_fit(X_tr, y_tr, gs_tr, n_tr, nr_tr,
             X_va, y_va, gs_va, n_va, nr_va, alpha):
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
            # val NLLはalpha=0で評価（正則化項を含まない純粋なNLL）
            vl, _ = _loss_grad(beta, X_va, y_va, gs_va, n_va, nr_va, alpha=0.0)
            if vl < best_val:
                best_val, best_beta, no_imp = vl, beta.copy(), 0
            else:
                no_imp += 1
            if no_imp >= PATIENCE // 10:
                break
    return best_beta, best_val


def evaluate_alpha(df_trn, df_val, oos_parts, feats, alpha):
    valid = [c for c in feats if c in df_trn.columns and df_trn[c].isna().mean() <= 0.65]
    if not valid:
        return None, None, {}
    X_tr, y_tr, gs_tr, n_tr, nr_tr, scaler, *_ = prepare(
        df_trn, valid, top_idx=None, top_idx3=None, fit=True)
    X_va, y_va, gs_va, n_va, nr_va, *_ = prepare(
        df_val, valid, scaler=scaler, top_idx=None, top_idx3=None)
    beta, val_nll = adam_fit(X_tr, y_tr, gs_tr, n_tr, nr_tr,
                              X_va, y_va, gs_va, n_va, nr_va, alpha)
    oos_roi = {}
    for period, df_p in oos_parts.items():
        if len(df_p) == 0:
            continue
        valid_p = [c for c in valid if c in df_p.columns]
        X_p, _, gs_p, n_p, *_ = prepare(df_p, valid_p, scaler=scaler,
                                          top_idx=None, top_idx3=None)
        scored = df_p.sort_values('race_id').reset_index(drop=True)
        scored['prob'] = segment_softmax(X_p @ beta, gs_p, n_p)
        scored['rank'] = scored.groupby('race_id')['prob'].rank(
            ascending=False, method='first')
        top1 = scored[scored['rank'] == 1]
        roi, _ = calc_roi(top1)
        oos_roi[period] = (roi, len(top1))
    return val_nll, len(valid), oos_roi


def fmt(oos_roi):
    parts = []
    for p in ['2324', '2025', '2026']:
        if p in oos_roi:
            roi, n = oos_roi[p]
            parts.append(f'{p}:{roi:+.2%}({n}R)')
    if '2025' in oos_roi and '2026' in oos_roi:
        r5, n5 = oos_roi['2025']
        r6, n6 = oos_roi['2026']
        comb = (r5*n5 + r6*n6) / (n5+n6)
        parts.append(f'25+26:{comb:+.2%}({n5+n6}R)')
    return '  '.join(parts)


def main():
    print(f'読み込み: {DATA_FILE}')
    df = load_segment()
    print(f'ダ_中長距離: {len(df):,}行\n')

    df_trn = df[(df['日付_num'] >= 130101) & (df['日付_num'] < 220101)]
    df_val = df[(df['日付_num'] >= 220101) & (df['日付_num'] <= 221231)]
    oos    = df[df['日付_num'] >= 230101]
    oos_parts = {
        '2324': oos[oos['日付_num'] < 250101],
        '2025': oos[(oos['日付_num'] >= 250101) & (oos['日付_num'] < 260101)],
        '2026': oos[oos['日付_num'] >= 260101],
    }

    print('='*70)
    print(f'{"特徴量セット":<10} {"alpha":>6} {"valNLL":>9} {"結果"}')
    print('='*70)

    for feat_name, feats in [('17特徴(base)', FEATS_17), ('26特徴(dom)', FEATS_26)]:
        print(f'\n--- {feat_name} ---')
        for alpha in ALPHAS:
            t0 = time.time()
            nll, n_valid, oos_roi = evaluate_alpha(df_trn, df_val, oos_parts, feats, alpha)
            elapsed = time.time() - t0
            if nll is None:
                print(f'  alpha={alpha:.1f}: データ不足')
                continue
            print(f'  alpha={alpha:.1f}  valNLL={nll:.5f}  {fmt(oos_roi)}  [{elapsed:.0f}s]')

    print('\n完了')


if __name__ == '__main__':
    main()
