# coding: utf-8
"""
fine_tune_recency.py - recency weighting の精密チューニング
  prepare()がrace_idでソートするため、事前にソートして重みを合わせる
  BASE_24 + L2=0.006
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
    df['year'] = df['日付_num'] // 10000  # 13, 14, ..., 21
    return df


def _loss_grad(beta, X, y, gs, n, nr, l2=0.0, sample_w=None):
    probs = segment_softmax(X @ beta, gs, n)
    if sample_w is not None:
        loss = -np.sum(sample_w * y * np.log(np.clip(probs, 1e-15, 1.0))) / nr
        grad = -(X.T @ (sample_w * (y - probs))) / nr
    else:
        loss = -np.sum(y * np.log(np.clip(probs, 1e-15, 1.0))) / nr
        grad = -(X.T @ (y - probs)) / nr
    loss += l2 * np.dot(beta, beta)
    grad += 2 * l2 * beta
    return loss, grad


def adam_fit(X_tr, y_tr, gs_tr, n_tr, nr_tr, X_va, y_va, gs_va, n_va, nr_va,
             l2=0.0, sample_w=None):
    d = X_tr.shape[1]
    beta, m, v = np.zeros(d), np.zeros(d), np.zeros(d)
    b1, b2, eps = 0.9, 0.999, 1e-8
    t, best_val, best_beta, no_imp = 0, np.inf, np.zeros(d), 0
    for epoch in range(1, N_EPOCHS + 1):
        _, grad = _loss_grad(beta, X_tr, y_tr, gs_tr, n_tr, nr_tr, l2, sample_w)
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


def score_oos(beta, df_p, valid, scaler):
    valid_p = [c for c in valid if c in df_p.columns]
    X_p, _, gs_p, n_p, *_ = prepare(df_p, valid_p, scaler=scaler,
                                      top_idx=None, top_idx3=None)
    scored = df_p.sort_values('race_id').reset_index(drop=True)
    scored['prob'] = segment_softmax(X_p @ beta, gs_p, n_p)
    scored['rank'] = scored.groupby('race_id')['prob'].rank(ascending=False, method='first')
    top1 = scored[scored['rank'] == 1]
    roi, wins = calc_roi(top1)
    return roi, len(top1), wins


def run(df_trn, df_val, oos_parts, feats, l2=0.0, recency_alpha=0.0):
    valid = [c for c in feats if c in df_trn.columns and df_trn[c].isna().mean() <= 0.65]

    # prepare()はrace_idでソートするので事前ソートしてから重みを計算
    df_trn_s = df_trn.sort_values('race_id').reset_index(drop=True)

    X_tr, y_tr, gs_tr, n_tr, nr_tr, scaler, *_ = prepare(
        df_trn_s, valid, top_idx=None, top_idx3=None, fit=True)

    if recency_alpha > 0 and 'year' in df_trn_s.columns:
        years = df_trn_s['year'].values
        y_min = years.min()
        w = np.exp(recency_alpha * (years - y_min)).astype(float)
        w = w / w.mean()
    else:
        w = None

    X_va, y_va, gs_va, n_va, nr_va, *_ = prepare(
        df_val, valid, scaler=scaler, top_idx=None, top_idx3=None)
    beta = adam_fit(X_tr, y_tr, gs_tr, n_tr, nr_tr,
                    X_va, y_va, gs_va, n_va, nr_va, l2=l2, sample_w=w)

    results = {}
    for period, df_p in oos_parts.items():
        if len(df_p) == 0:
            continue
        roi, n, wins = score_oos(beta, df_p, valid, scaler)
        results[period] = (roi, n, wins)
    return results, beta, scaler, valid


def main():
    df = load_segment()
    df_trn = df[(df['日付_num'] >= 130101) & (df['日付_num'] < 220101)]
    df_val = df[(df['日付_num'] >= 220101) & (df['日付_num'] <= 221231)]
    oos    = df[df['日付_num'] >= 230101]
    oos_parts = {
        '2324': oos[oos['日付_num'] < 250101],
        '2025': oos[(oos['日付_num'] >= 250101) & (oos['日付_num'] < 260101)],
        '2026': oos[oos['日付_num'] >= 260101],
    }

    print(f'{"α":>6}  {"2324":>8}  {"2025":>8}  {"2026":>8}  {"25+26":>8}')
    print('='*50)

    best_comb = -99; best_alpha = 0; best_roi = {}
    for alpha in [0.00, 0.05, 0.08, 0.10, 0.12, 0.15, 0.17, 0.20, 0.22, 0.25, 0.28, 0.30, 0.35, 0.40]:
        roi, *_ = run(df_trn, df_val, oos_parts, BASE_24, l2=L2, recency_alpha=alpha)
        r24 = roi.get('2324', (0, 1, 0))[0]
        r25, n25, _ = roi.get('2025', (0, 1, 0))
        r26, n26, _ = roi.get('2026', (0, 1, 0))
        comb = (r25*n25 + r26*n26) / (n25+n26)
        mark = ' ★' if comb > best_comb else ''
        print(f'{alpha:>6.2f}  {r24:>+8.2%}  {r25:>+8.2%}  {r26:>+8.2%}  {comb:>+8.2%}{mark}')
        if comb > best_comb:
            best_comb = comb; best_alpha = alpha; best_roi = roi

    print(f'\n最良: α={best_alpha}, 25+26={best_comb:+.2%}')
    print(f'21F基準(-19.70%) から改善: {best_comb-(-0.197):+.2%}')

    # 最良設定の詳細
    print(f'\n【最良設定詳細: α={best_alpha}, L2={L2}】')
    for period, (roi, n, wins) in best_roi.items():
        print(f'  {period}: {n}R  ROI={roi:+.4f}  勝率={wins/n:.1%}')


if __name__ == '__main__':
    main()
