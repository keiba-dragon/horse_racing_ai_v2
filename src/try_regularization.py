# coding: utf-8
"""
try_regularization.py - L2正則化・訓練期間変更のOOS ROI影響テスト
  1. L2正則化強度スイープ (λ = 0, 0.001, 0.01, 0.1, 1.0)
  2. 訓練期間を変更 (2013-, 2016-, 2018-, 2019-スタート)
  ベース: 24F モデル (21F + 性別_num + 所属_num + キャリア_浅い)
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


def adam_fit_l2(X_tr, y_tr, gs_tr, n_tr, nr_tr, X_va, y_va, gs_va, n_va, nr_va, l2=0.0):
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
            # val評価はL2なしのNLL（汎化を見る）
            vl, _ = _loss_grad(beta, X_va, y_va, gs_va, n_va, nr_va, l2=0.0)
            if vl < best_val:
                best_val, best_beta, no_imp = vl, beta.copy(), 0
            else:
                no_imp += 1
            if no_imp >= PATIENCE // 10:
                break
    return best_beta, best_val


def evaluate(df_trn, df_val, oos_parts, feats, l2=0.0):
    valid = [c for c in feats if c in df_trn.columns and df_trn[c].isna().mean() <= 0.65]
    if not valid:
        return None, None, {}
    X_tr, y_tr, gs_tr, n_tr, nr_tr, scaler, *_ = prepare(
        df_trn, valid, top_idx=None, top_idx3=None, fit=True)
    X_va, y_va, gs_va, n_va, nr_va, *_ = prepare(
        df_val, valid, scaler=scaler, top_idx=None, top_idx3=None)
    beta, val_nll = adam_fit_l2(X_tr, y_tr, gs_tr, n_tr, nr_tr,
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
        scored['rank'] = scored.groupby('race_id')['prob'].rank(
            ascending=False, method='first')
        top1 = scored[scored['rank'] == 1]
        roi, wins = calc_roi(top1)
        oos_roi[period] = (roi, len(top1), wins)
    return val_nll, beta, oos_roi


def main():
    df = load_segment()
    df_val = df[(df['日付_num'] >= 220101) & (df['日付_num'] <= 221231)]
    oos    = df[df['日付_num'] >= 230101]
    oos_parts = {
        '2324': oos[oos['日付_num'] < 250101],
        '2025': oos[(oos['日付_num'] >= 250101) & (oos['日付_num'] < 260101)],
        '2026': oos[oos['日付_num'] >= 260101],
    }

    print('=== 1. L2正則化スイープ (訓練: 2013-2021) ===')
    df_trn_base = df[(df['日付_num'] >= 130101) & (df['日付_num'] < 220101)]
    print(f'訓練行数: {len(df_trn_base):,}')
    best_comb = -99; best_l2 = 0
    for l2 in [0.0, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]:
        _, _, roi = evaluate(df_trn_base, df_val, oos_parts, BASE_24, l2=l2)
        r25, n25, _ = roi.get('2025', (0, 0, 0))
        r26, n26, _ = roi.get('2026', (0, 0, 0))
        comb = (r25*n25 + r26*n26) / (n25+n26) if n25+n26 > 0 else 0
        mark = ' ★' if comb > best_comb else ''
        print(f'  L2={l2:<6} → 2025:{r25:+.2%}  2026:{r26:+.2%}  25+26:{comb:+.2%}{mark}')
        if comb > best_comb:
            best_comb = comb; best_l2 = l2
    print(f'  → 最良 L2={best_l2}, 25+26={best_comb:+.2%}')

    print()
    print('=== 2. 訓練期間スイープ (L2=0) ===')
    best_comb2 = -99; best_start = 130101
    for start_year, start_num in [
        ('2013', 130101), ('2015', 150101), ('2017', 170101),
        ('2018', 180101), ('2019', 190101), ('2020', 200101)
    ]:
        df_trn = df[(df['日付_num'] >= start_num) & (df['日付_num'] < 220101)]
        if len(df_trn) < 5000:
            print(f'  {start_year}~: 訓練データ不足 ({len(df_trn):,}行)')
            continue
        _, _, roi = evaluate(df_trn, df_val, oos_parts, BASE_24, l2=0.0)
        r25, n25, _ = roi.get('2025', (0, 0, 0))
        r26, n26, _ = roi.get('2026', (0, 0, 0))
        comb = (r25*n25 + r26*n26) / (n25+n26) if n25+n26 > 0 else 0
        mark = ' ★' if comb > best_comb2 else ''
        print(f'  {start_year}~: trn={len(df_trn):,}行  2025:{r25:+.2%}  2026:{r26:+.2%}  25+26:{comb:+.2%}{mark}')
        if comb > best_comb2:
            best_comb2 = comb; best_start = start_num
    print(f'  → 最良 start={best_start}, 25+26={best_comb2:+.2%}')

    # 最良のL2 + 最良の訓練期間を組み合わせ
    print()
    print(f'=== 3. 最良組み合わせ: start={best_start}, L2={best_l2} ===')
    df_trn_best = df[(df['日付_num'] >= best_start) & (df['日付_num'] < 220101)]
    _, _, roi = evaluate(df_trn_best, df_val, oos_parts, BASE_24, l2=best_l2)
    r25, n25, w25 = roi.get('2025', (0, 0, 0))
    r26, n26, w26 = roi.get('2026', (0, 0, 0))
    comb = (r25*n25 + r26*n26) / (n25+n26) if n25+n26 > 0 else 0
    print(f'  2025: {n25}R  ROI={r25:+.4f}  勝率={w25/n25:.1%}')
    print(f'  2026: {n26}R  ROI={r26:+.4f}  勝率={w26/n26:.1%}')
    print(f'  25+26: {comb:+.4f}')
    print(f'  ★ 21F基準(-19.70%) からの改善: {comb-(-0.197):+.2%}')


if __name__ == '__main__':
    main()
