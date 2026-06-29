# coding: utf-8
"""
split_da_dist.py - ダ中長距離を距離帯で分割してモデル比較
  ダ_マイル: 1700m以下 (1500-1700m)
  ダ_長距離: 1800m以上
21F特徴リストを各セグメントで評価
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


def load_base():
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


def run_segment(name, df):
    df_trn = df[(df['日付_num'] >= 130101) & (df['日付_num'] < 220101)]
    df_val = df[(df['日付_num'] >= 220101) & (df['日付_num'] <= 221231)]
    oos    = df[df['日付_num'] >= 230101]
    oos_parts = {
        '2324': oos[oos['日付_num'] < 250101],
        '2025': oos[(oos['日付_num'] >= 250101) & (oos['日付_num'] < 260101)],
        '2026': oos[oos['日付_num'] >= 260101],
    }
    print(f'\n{"="*55}')
    print(f'【{name}】 trn:{len(df_trn):,}  val:{len(df_val):,}')
    for p, d in oos_parts.items():
        print(f'  OOS {p}: {len(d):,}行')

    nll, beta, oos_roi = evaluate(df_trn, df_val, oos_parts, FINAL_21)
    if nll is None:
        print('評価不可')
        return

    print(f'valNLL={nll:.5f}')
    for period, (roi, n, wins) in oos_roi.items():
        print(f'  {period}: {n}R  ROI={roi:+.4f}  勝率={wins/n:.1%}')

    if '2025' in oos_roi and '2026' in oos_roi:
        r25, n25, _ = oos_roi['2025']
        r26, n26, _ = oos_roi['2026']
        comb = (r25*n25 + r26*n26) / (n25+n26)
        print(f'  25+26合算: {comb:+.4f}')

    valid = [c for c in FINAL_21 if c in df_trn.columns and df_trn[c].isna().mean() <= 0.65]
    print('\nβ係数（絶対値降順）:')
    idx = np.argsort(np.abs(beta))[::-1]
    for i in idx[:10]:
        print(f'  {valid[i]:<35} β={beta[i]:+.4f}')


def main():
    df = load_base()

    # 全体（参照）
    print('=== 参照: ダ中長距離 全体（21F） ===')
    run_segment('全体 (1500m+)', df)

    # 分割1: 1700m以下
    df_mile = df[df['dist_m'] <= 1700].copy()
    run_segment('ダ_マイル (1500-1700m)', df_mile)

    # 分割2: 1800m以上
    df_long = df[df['dist_m'] >= 1800].copy()
    run_segment('ダ_長距離 (1800m+)', df_long)

    # 距離ごとの詳細
    print('\n\n=== 距離別 OOS ROI（21F モデル全体で予測） ===')
    df_full = df.copy()
    df_trn_f = df_full[(df_full['日付_num'] >= 130101) & (df_full['日付_num'] < 220101)]
    df_val_f = df_full[(df_full['日付_num'] >= 220101) & (df_full['日付_num'] <= 221231)]
    valid_f = [c for c in FINAL_21 if c in df_trn_f.columns and df_trn_f[c].isna().mean() <= 0.65]
    X_tr, y_tr, gs_tr, n_tr, nr_tr, scaler, *_ = prepare(
        df_trn_f, valid_f, top_idx=None, top_idx3=None, fit=True)
    X_va, y_va, gs_va, n_va, nr_va, *_ = prepare(
        df_val_f, valid_f, scaler=scaler, top_idx=None, top_idx3=None)
    beta_f, _ = adam_fit(X_tr, y_tr, gs_tr, n_tr, nr_tr, X_va, y_va, gs_va, n_va, nr_va)

    oos_f = df_full[(df_full['日付_num'] >= 250101) & (df_full['日付_num'] < 270101)]
    valid_p = [c for c in valid_f if c in oos_f.columns]
    X_p, _, gs_p, n_p, *_ = prepare(oos_f, valid_p, scaler=scaler,
                                      top_idx=None, top_idx3=None)
    scored = oos_f.sort_values('race_id').reset_index(drop=True)
    scored['prob'] = segment_softmax(X_p @ beta_f, gs_p, n_p)
    scored['rank'] = scored.groupby('race_id')['prob'].rank(ascending=False, method='first')
    top1 = scored[scored['rank'] == 1]

    print(f'{"距離":>6}  {"R数":>5}  {"ROI":>8}  {"勝率":>6}')
    for d in sorted(top1['dist_m'].unique()):
        sub = top1[top1['dist_m'] == d]
        if len(sub) < 20:
            continue
        roi, wins = calc_roi(sub)
        print(f'{int(d):>6}m  {len(sub):>5}R  {roi:>+8.4f}  {wins/len(sub):>6.1%}')


if __name__ == '__main__':
    main()
