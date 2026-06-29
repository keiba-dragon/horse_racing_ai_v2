# coding: utf-8
"""
try_temporal_shift.py - 訓練期間の前方シフト実験
  通常: train=2013-2021, val=2022, OOS=2023+
  案A: train=2013-2022, val=2023, OOS=2024+
  案B: train=2018-2022, val=2023, OOS=2024+  (直近5年)
  案C: train=2019-2022, val=2023, OOS=2024+  (直近4年)
  案D: train=2016-2021, val=2022, OOS=2023+  (直近6年)
  案E: train=2018-2021, val=2022, OOS=2023+  (直近4年)
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


def show(label, roi, n_trn, primary_keys):
    vals = []
    for k in primary_keys:
        r, n, _ = roi.get(k, (0, 1, 0))
        vals.append((k, r, n))
    # 25+26 combined (or whatever the last two periods are)
    r_last2 = [(r, n) for k, r, n in vals[-2:]]
    if len(r_last2) == 2:
        comb = (r_last2[0][0]*r_last2[0][1] + r_last2[1][0]*r_last2[1][1]) / (r_last2[0][1] + r_last2[1][1])
    else:
        comb = r_last2[0][0]
    parts = '  '.join(f'{k}:{r:+.2%}' for k, r, n in vals)
    print(f'  {label:<35} {parts}  combined:{comb:+.2%}  (train={n_trn}R)')
    return comb


def main():
    df = load_segment()

    print('=== 訓練期間シフト実験 (L2=0.006, BASE_24) ===')
    print()

    results = {}

    # ベースライン: train 2013-2021, val 2022, OOS 2023+
    df_trn = df[(df['日付_num'] >= 130101) & (df['日付_num'] < 220101)]
    df_val = df[(df['日付_num'] >= 220101) & (df['日付_num'] <= 221231)]
    oos = df[df['日付_num'] >= 230101]
    oos_parts = {
        '2324': oos[oos['日付_num'] < 250101],
        '2025': oos[(oos['日付_num'] >= 250101) & (oos['日付_num'] < 260101)],
        '2026': oos[oos['日付_num'] >= 260101],
    }
    roi = evaluate(df_trn, df_val, oos_parts, BASE_24, l2=L2)
    n_trn = df_trn.groupby('race_id').ngroups
    c = show('【ベース】2013-2021/val=2022', roi, n_trn, ['2324','2025','2026'])
    results['base'] = c
    print()

    # 案D: train 2016-2021 (直近6年), val 2022
    df_trn = df[(df['日付_num'] >= 160101) & (df['日付_num'] < 220101)]
    roi = evaluate(df_trn, df_val, oos_parts, BASE_24, l2=L2)
    n_trn = df_trn.groupby('race_id').ngroups
    c = show('案D: 2016-2021/val=2022', roi, n_trn, ['2324','2025','2026'])
    results['D'] = c

    # 案E: train 2018-2021 (直近4年), val 2022
    df_trn = df[(df['日付_num'] >= 180101) & (df['日付_num'] < 220101)]
    roi = evaluate(df_trn, df_val, oos_parts, BASE_24, l2=L2)
    n_trn = df_trn.groupby('race_id').ngroups
    c = show('案E: 2018-2021/val=2022', roi, n_trn, ['2324','2025','2026'])
    results['E'] = c

    # 案F: train 2019-2021 (直近3年), val 2022
    df_trn = df[(df['日付_num'] >= 190101) & (df['日付_num'] < 220101)]
    roi = evaluate(df_trn, df_val, oos_parts, BASE_24, l2=L2)
    n_trn = df_trn.groupby('race_id').ngroups
    c = show('案F: 2019-2021/val=2022', roi, n_trn, ['2324','2025','2026'])
    results['F'] = c

    print()

    # 前方シフト系: val=2023, OOS=2024+
    df_val23 = df[(df['日付_num'] >= 230101) & (df['日付_num'] <= 231231)]
    oos_parts24 = {
        '2024': df[(df['日付_num'] >= 240101) & (df['日付_num'] < 250101)],
        '2025': df[(df['日付_num'] >= 250101) & (df['日付_num'] < 260101)],
        '2026': df[df['日付_num'] >= 260101],
    }

    # 案A: train 2013-2022, val 2023, OOS 2024+
    df_trn = df[(df['日付_num'] >= 130101) & (df['日付_num'] < 230101)]
    roi = evaluate(df_trn, df_val23, oos_parts24, BASE_24, l2=L2)
    n_trn = df_trn.groupby('race_id').ngroups
    c = show('案A: 2013-2022/val=2023', roi, n_trn, ['2024','2025','2026'])
    results['A'] = c

    # 案B: train 2018-2022, val 2023, OOS 2024+
    df_trn = df[(df['日付_num'] >= 180101) & (df['日付_num'] < 230101)]
    roi = evaluate(df_trn, df_val23, oos_parts24, BASE_24, l2=L2)
    n_trn = df_trn.groupby('race_id').ngroups
    c = show('案B: 2018-2022/val=2023', roi, n_trn, ['2024','2025','2026'])
    results['B'] = c

    # 案C: train 2019-2022, val 2023, OOS 2024+
    df_trn = df[(df['日付_num'] >= 190101) & (df['日付_num'] < 230101)]
    roi = evaluate(df_trn, df_val23, oos_parts24, BASE_24, l2=L2)
    n_trn = df_trn.groupby('race_id').ngroups
    c = show('案C: 2019-2022/val=2023', roi, n_trn, ['2024','2025','2026'])
    results['C'] = c

    print()
    best_label = max(results, key=results.get)
    print(f'最良: 案{best_label} = {results[best_label]:+.2%}')
    print(f'ベースライン比: Δ={results[best_label]-results["base"]:+.2%}')


if __name__ == '__main__':
    main()
