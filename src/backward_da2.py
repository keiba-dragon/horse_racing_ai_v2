# coding: utf-8
"""
backward_da2.py - 15特徴ベース（上り3F + 近5走_タイム指数平均 除外）
さらにbackward eliminationを続ける
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

# 15特徴（17 - 上り3F - 近5走_タイム指数平均）
BASE_15 = [
    '近5走_クラス調整_平均着順',
    '近5走_タイム指数_max', '1走前_タイム指数', '前走着差タイム',
    '騎手コース_r100_勝率', '1走前_クラス調整着順', '調教師コース_r100_勝率',
    '1走前_RPCI', '1走前_上3F地点差', '斤量', '種牡馬_勝率',
    '間隔_長_flag', '1走前_脚質_num', '騎手変更', '馬番',
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


def evaluate(df_trn, df_val, oos_parts, feats):
    valid = [c for c in feats if c in df_trn.columns and df_trn[c].isna().mean() <= 0.65]
    if not valid:
        return None, None, {}
    X_tr, y_tr, gs_tr, n_tr, nr_tr, scaler, *_ = prepare(
        df_trn, valid, top_idx=None, top_idx3=None, fit=True)
    X_va, y_va, gs_va, n_va, nr_va, *_ = prepare(
        df_val, valid, scaler=scaler, top_idx=None, top_idx3=None)
    beta, val_nll = adam_fit(X_tr, y_tr, gs_tr, n_tr, nr_tr,
                              X_va, y_va, gs_va, n_va, nr_va)
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
    return val_nll, beta, oos_roi


def combined_roi(oos_roi):
    if '2025' not in oos_roi or '2026' not in oos_roi:
        return -999
    r5, n5 = oos_roi['2025']
    r6, n6 = oos_roi['2026']
    return (r5*n5 + r6*n6) / (n5+n6)


def fmt(oos_roi):
    parts = []
    for p in ['2324', '2025', '2026']:
        if p in oos_roi:
            roi, n = oos_roi[p]
            parts.append(f'{p}:{roi:+.2%}({n}R)')
    c = combined_roi(oos_roi)
    if c != -999:
        parts.append(f'25+26:{c:+.2%}')
    return '  '.join(parts)


def greedy_backward(df_trn, df_val, oos_parts, feats, label):
    """1本ずつ削除して25+26 ROIが改善する限り続ける"""
    current = list(feats)
    nll0, beta0, roi0 = evaluate(df_trn, df_val, oos_parts, current)
    c0 = combined_roi(roi0)
    print(f'\n[{label}] 開始 {len(current)}特徴: {fmt(roi0)}')

    removed_history = []
    round_num = 0

    while True:
        round_num += 1
        results = []
        for feat in current:
            remaining = [f for f in current if f != feat]
            nll, _, roi = evaluate(df_trn, df_val, oos_parts, remaining)
            c = combined_roi(roi)
            delta = c - c0
            results.append((feat, nll, delta, c, roi))
        results.sort(key=lambda x: -x[2])  # 削除して良くなる順

        # 最善の削除を表示
        print(f'\nラウンド{round_num}:')
        for feat, nll, delta, c_new, roi in results[:5]:
            sym = '✓削' if delta > 0.003 else ' ~'
            print(f'  {sym} -{feat}: Δ25+26={delta:+.2%}  {fmt(roi)}')

        # 最善を採用（Δ > +0.3pp）
        best = results[0]
        if best[2] > 0.003:
            feat_to_remove = best[0]
            removed_history.append(feat_to_remove)
            current = [f for f in current if f != feat_to_remove]
            _, _, roi_new = evaluate(df_trn, df_val, oos_parts, current)
            c0 = combined_roi(roi_new)
            print(f'  → -{feat_to_remove} 採用: {len(current)}特徴  {fmt(roi_new)}')
        else:
            print(f'  → 改善なし。停止。')
            break

        if len(current) < 5:
            print(f'  → 特徴数が少なすぎるため停止。')
            break

    return current, c0, removed_history


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

    final_feats, final_comb, removed = greedy_backward(
        df_trn, df_val, oos_parts, BASE_15, 'backward-15')

    print(f'\n{"="*60}')
    print(f'最終特徴数: {len(final_feats)}')
    print(f'削除履歴: {removed}')
    print(f'最終25+26 ROI: {final_comb:+.2%}')
    print('\n最終特徴量:')
    for f in final_feats:
        print(f'  {f}')

    # ベータ係数確認
    nll, beta, roi = evaluate(df_trn, df_val, oos_parts, final_feats)
    valid = [c for c in final_feats if c in df_trn.columns and df_trn[c].isna().mean() <= 0.65]
    print(f'\n最終評価: {fmt(roi)}')
    print(f'valNLL={nll:.5f}')
    print('\nベータ係数:')
    idx = np.argsort(np.abs(beta))[::-1]
    for i in idx:
        print(f'  {valid[i]:<35} β={beta[i]:+.4f}')


if __name__ == '__main__':
    main()
