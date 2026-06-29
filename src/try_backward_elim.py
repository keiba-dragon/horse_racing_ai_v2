# coding: utf-8
"""
try_backward_elim.py - BASE_25の後ろ向き消去
  各特徴量を除いたROIを計算し、削除すると改善する特徴を特定
  またBASE_25の係数上位/下位を表示
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


def evaluate_with_beta(df_trn, df_val, oos_parts, feats, l2=0.0):
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
    return oos_roi, beta, valid


def comb_roi(roi):
    r25, n25, _ = roi.get('2025', (0, 1, 0))
    r26, n26, _ = roi.get('2026', (0, 1, 0))
    return (r25*n25 + r26*n26) / (n25+n26)


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

    print('=== BASE_25 係数分析 ===')
    roi0, beta0, valid0 = evaluate_with_beta(df_trn, df_val, oos_parts, BASE_25, l2=L2)
    c0 = comb_roi(roi0)
    r25, n25, _ = roi0.get('2025', (0, 1, 0))
    r26, n26, _ = roi0.get('2026', (0, 1, 0))
    print(f'BASE_25: 2025={r25:+.2%}  2026={r26:+.2%}  25+26={c0:+.2%}')
    print()

    # 係数を表示
    sorted_idx = np.argsort(np.abs(beta0))[::-1]
    print('係数上位 (|β|順):')
    for i in sorted_idx:
        print(f'  {valid0[i]:<40} β={beta0[i]:+.4f}')
    print()

    # 後ろ向き消去
    print('=== 後ろ向き消去 (BASE_25から1特徴ずつ削除) ===')
    removal_results = []
    for c in BASE_25:
        trial = [f for f in BASE_25 if f != c]
        roi = evaluate_with_beta(df_trn, df_val, oos_parts, trial, l2=L2)[0]
        c_new = comb_roi(roi)
        delta = c_new - c0
        removal_results.append((delta, c, c_new))

    removal_results.sort(reverse=True)  # 削除で改善するものが上位

    print('削除効果 (改善幅順):')
    for delta, c, c_new in removal_results:
        mark = ' ★ ← 削除推奨' if delta > 0 else ''
        print(f'  -{c:<40} 25+26:{c_new:+.2%}  Δ={delta:+.2%}{mark}')

    # 改善する削除がある場合、ステップワイズ削除
    improves = [(delta, c) for delta, c, _ in removal_results if delta > 0]
    if improves:
        print()
        print('★ 削除で改善:')
        current_feats = BASE_25.copy()
        current_c = c0
        removed = []
        for delta, c in improves:
            trial = [f for f in current_feats if f != c]
            roi = evaluate_with_beta(df_trn, df_val, oos_parts, trial, l2=L2)[0]
            c_new = comb_roi(roi)
            if c_new > current_c:
                current_feats = trial
                current_c = c_new
                removed.append(c)
                print(f'  -{c} → {current_c:+.2%}')
            else:
                print(f'  -{c} → 棄却 ({c_new:+.2%})')
        print(f'最終: {len(current_feats)}F → {current_c:+.2%}')
        print(f'21F基準から: {current_c-(-0.197):+.2%}')
    else:
        print()
        print('後ろ向き消去で改善なし')
        print(f'21F基準から最良: {c0-(-0.197):+.2%}')


if __name__ == '__main__':
    main()
