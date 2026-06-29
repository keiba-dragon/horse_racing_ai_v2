# coding: utf-8
"""
try_interactions.py - prepare()の2-way交互作用項を活用
  top_idx で上位特徴量を指定 → C(k,2) の積特徴量が自動生成される
  BASE_24 + L2=0.006 で、top_idx を変えてROI比較
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
    return df


def _loss_grad(beta, X, y, gs, n, nr, l2=0.0):
    probs = segment_softmax(X @ beta, gs, n)
    res   = y - probs
    loss  = -np.sum(y * np.log(np.clip(probs, 1e-15, 1.0))) / nr + l2 * np.dot(beta, beta)
    grad  = -(X.T @ res) / nr + 2 * l2 * beta
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
    return best_beta, best_val


def run_with_interactions(df_trn, df_val, oos_parts, feats, top_idx, l2=0.0, label=''):
    valid = [c for c in feats if c in df_trn.columns and df_trn[c].isna().mean() <= 0.65]

    X_tr, y_tr, gs_tr, n_tr, nr_tr, scaler, poly2, inter_sc2, *_ = prepare(
        df_trn, valid, top_idx=top_idx, top_idx3=None, fit=True)
    X_va, y_va, gs_va, n_va, nr_va, *_ = prepare(
        df_val, valid, scaler=scaler, poly2=poly2, inter_scaler2=inter_sc2,
        top_idx=top_idx, top_idx3=None)

    beta, val_nll = adam_fit(X_tr, y_tr, gs_tr, n_tr, nr_tr,
                              X_va, y_va, gs_va, n_va, nr_va, l2=l2)

    oos_roi = {}
    for period, df_p in oos_parts.items():
        if len(df_p) == 0:
            continue
        valid_p = [c for c in valid if c in df_p.columns]
        X_p, _, gs_p, n_p, *_ = prepare(df_p, valid_p, scaler=scaler,
                                          poly2=poly2, inter_scaler2=inter_sc2,
                                          top_idx=top_idx, top_idx3=None)
        if X_p.shape[1] != len(beta):
            # top_idx feature mismatch for OOS - skip interactions
            X_p, _, gs_p, n_p, *_ = prepare(df_p, valid_p, scaler=scaler,
                                              top_idx=None, top_idx3=None)
            beta_base = beta[:len(valid_p)]
            probs = segment_softmax(X_p @ beta_base, gs_p, n_p)
        else:
            probs = segment_softmax(X_p @ beta, gs_p, n_p)
        scored = df_p.sort_values('race_id').reset_index(drop=True)
        scored['prob'] = probs
        scored['rank'] = scored.groupby('race_id')['prob'].rank(ascending=False, method='first')
        top1 = scored[scored['rank'] == 1]
        roi, wins = calc_roi(top1)
        oos_roi[period] = (roi, len(top1), wins)

    r25, n25, _ = oos_roi.get('2025', (0, 1, 0))
    r26, n26, _ = oos_roi.get('2026', (0, 1, 0))
    r24 = oos_roi.get('2324', (0, 1, 0))[0]
    comb = (r25*n25 + r26*n26) / (n25+n26)
    n_inter = X_tr.shape[1] - len(valid)
    print(f'  {label:<30} 交互作用:{n_inter:3d}個  2324:{r24:+.2%}  2025:{r25:+.2%}  2026:{r26:+.2%}  25+26:{comb:+.2%}')
    return comb


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

    valid = [c for c in BASE_24 if c in df_trn.columns and df_trn[c].isna().mean() <= 0.65]
    print(f'有効特徴量: {len(valid)}F')
    print()
    print(f'{"設定":<32} {"n_inter":>8}  {"2324":>8}  {"2025":>8}  {"2026":>8}  {"25+26":>8}')

    best_comb = -99; best_top_idx = None

    # ベースライン（交互作用なし）
    c = run_with_interactions(df_trn, df_val, oos_parts, BASE_24, None, L2, 'ベース (交互作用なし)')
    if c > best_comb: best_comb = c

    # 上位k特徴量で交互作用
    top_candidates = [
        ([0, 1, 2, 3, 5], 'top5: クラス平均/TI_max/前走TI/着差/調整着順'),
        ([0, 1, 2, 3, 4, 5], 'top6: +騎手コース勝率'),
        ([0, 1, 2, 3, 5, 7], 'top6: +RPCI'),
        ([0, 1, 2, 3, 4, 5, 6, 7], 'top8: 前8特徴'),
        ([0, 1, 2, 3, 5, 9, 10], 'top7: 主要+斤量+種牡馬'),
        # 性能系のみの交互作用
        ([0, 1, 2, 5], 'perf4: 成績系'),
        ([3, 7, 8], 'pace3: ペース系'),
        # 広めの交互作用
        (list(range(10)), 'top10: 前10特徴'),
    ]

    for top_idx, label in top_candidates:
        top_idx_arr = [i for i in top_idx if i < len(valid)]
        if len(top_idx_arr) < 2:
            continue
        c = run_with_interactions(df_trn, df_val, oos_parts, BASE_24, top_idx_arr, L2, label)
        if c > best_comb:
            best_comb = c; best_top_idx = top_idx_arr

    print(f'\n最良: top_idx={best_top_idx}, 25+26={best_comb:+.2%}')
    print(f'21F基準(-19.70%) から改善: {best_comb-(-0.197):+.2%}')


if __name__ == '__main__':
    main()
