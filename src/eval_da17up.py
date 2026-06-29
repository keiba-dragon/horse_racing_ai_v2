# coding: utf-8
"""
eval_da17up.py - ダ中長距離を 1700m以上に変更してROI評価
現在: dm > 1400 (1600m込み)
新定義: dm > 1600 (1700m以上)
21F特徴で比較
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

FINAL_21 = [
    '近5走_クラス調整_平均着順', '近5走_タイム指数_max', '1走前_タイム指数', '前走着差タイム',
    '騎手コース_r100_勝率', '1走前_クラス調整着順', '調教師コース_r100_勝率',
    '1走前_RPCI', '1走前_上3F地点差', '斤量', '種牡馬_勝率',
    '間隔_長_flag', '1走前_脚質_num', '騎手変更', '馬番',
    '近3走_複勝率', '調教師_r200_複勝率',
    'ブリンカー変更', '2走前_クラス差', '4走前_クラス差', '1走前_馬場状態',
]


def load_segment(min_dist):
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
    df = df[(df['surface'] == 'ダ') & (dm > min_dist)].copy()
    df['dist_m'] = dm[df.index]
    if 'クラス_rank' in df.columns:
        df = df[df['クラス_rank'] != 1.0].copy()
    df = add_computed_features(df)
    baba_map = {'良': 0, '稍重': 1, '重': 2, '不良': 3}
    for col in df.columns:
        if '馬場状態' in col:
            df[col] = df[col].map(baba_map)
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
        roi, wins = calc_roi(top1)
        oos_roi[period] = (roi, len(top1), wins)
    return val_nll, beta, oos_roi


def run_eval(label, min_dist):
    df = load_segment(min_dist)
    df_trn = df[(df['日付_num'] >= 130101) & (df['日付_num'] < 220101)]
    df_val = df[(df['日付_num'] >= 220101) & (df['日付_num'] <= 221231)]
    oos    = df[df['日付_num'] >= 230101]
    oos_parts = {
        '2324': oos[oos['日付_num'] < 250101],
        '2025': oos[(oos['日付_num'] >= 250101) & (oos['日付_num'] < 260101)],
        '2026': oos[oos['日付_num'] >= 260101],
    }
    print(f'\n{"="*55}')
    print(f'【{label}】 trn:{len(df_trn):,} val:{len(df_val):,}')
    nll, beta, oos_roi = evaluate(df_trn, df_val, oos_parts, FINAL_21)
    for period, (roi, n, wins) in oos_roi.items():
        print(f'  {period}: {n}R  ROI={roi:+.4f}  勝率={wins/n:.1%}')
    if '2025' in oos_roi and '2026' in oos_roi:
        r25, n25, _ = oos_roi['2025']
        r26, n26, _ = oos_roi['2026']
        comb = (r25*n25 + r26*n26) / (n25+n26)
        print(f'  ★ 25+26: {comb:+.4f}')
    return oos_roi, beta, nll


def main():
    print('=== ダ中長距離 境界変更テスト (21F特徴) ===')
    print('\n[現行] dm > 1400 (1600m込み)')
    roi_old, _, _ = run_eval('現行: dm>1400 (1600m+)', min_dist=1400)

    print('\n[新定義] dm > 1600 (1700m以上, 1600m→ダ短へ)')
    roi_new, beta_new, nll_new = run_eval('新定義: dm>1600 (1700m+)', min_dist=1600)

    # 差分サマリ
    print('\n\n=== 比較サマリ ===')
    for p in ['2324', '2025', '2026']:
        if p in roi_old and p in roi_new:
            r_old, n_old, _ = roi_old[p]
            r_new, n_new, _ = roi_new[p]
            print(f'  {p}: 現行={r_old:+.4f}({n_old}R) → 新={r_new:+.4f}({n_new}R)  Δ={r_new-r_old:+.4f}')

    if '2025' in roi_new and '2026' in roi_new:
        r25, n25, _ = roi_new['2025']
        r26, n26, _ = roi_new['2026']
        comb_new = (r25*n25 + r26*n26) / (n25+n26)
        r25o, n25o, _ = roi_old['2025']
        r26o, n26o, _ = roi_old['2026']
        comb_old = (r25o*n25o + r26o*n26o) / (n25o+n26o)
        print(f'\n  25+26: 現行={comb_old:+.4f} → 新={comb_new:+.4f}  Δ={comb_new-comb_old:+.4f}')


if __name__ == '__main__':
    main()
