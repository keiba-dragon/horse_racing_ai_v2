# coding: utf-8
"""
add_jockey_feats.py - 騎手・調教師系未試験特徴 全投入
16Fベースに騎手系・調教師系・オッズ系を個別追加してOOS ROI確認
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

BASE_16 = [
    '近5走_クラス調整_平均着順',
    '近5走_タイム指数_max', '1走前_タイム指数', '前走着差タイム',
    '騎手コース_r100_勝率', '1走前_クラス調整着順', '調教師コース_r100_勝率',
    '1走前_RPCI', '1走前_上3F地点差', '斤量', '種牡馬_勝率',
    '間隔_長_flag', '1走前_脚質_num', '騎手変更', '馬番',
    '近3走_複勝率',
]

# 未試験の騎手・調教師系全候補
NEW_CANDIDATES = [
    # 騎手全体勝率（コースに依存しない）
    '騎手_r200_勝率',
    '騎手_r200_複勝率',
    '騎手_平均着順',
    # コース複勝率（勝率は入っている）
    '騎手コース_r100_複勝率',
    # 距離帯別（ダート長距離に特化）
    '騎手距離_r100_勝率',
    '騎手距離_r100_複勝率',
    # 馬場別（ダート特化）
    '騎手馬場_r100_勝率',
    '騎手馬場_r100_複勝率',
    # 騎手×コース×距離
    '騎手コース距離_r100_勝率',
    '騎手コース距離_r100_複勝率',
    # 騎手×調教師コンビ
    '騎手調教師_r100_勝率',
    '騎手調教師_r100_複勝率',
    # 脚質一致率
    '騎手脚質_r100_勝率',
    '騎手脚質_r100_複勝率',
    # 調教師全体勝率
    '調教師_r200_勝率',
    '調教師_r200_複勝率',
    # 調教師コース複勝率
    '調教師コース_r100_複勝率',
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

    # ベース16F確認
    nll0, _, roi0 = evaluate(df_trn, df_val, oos_parts, BASE_16)
    r25_0 = roi0['2025'][0]
    r26_0 = roi0['2026'][0]
    n25, n26 = roi0['2025'][1], roi0['2026'][1]
    comb0 = (r25_0*n25 + r26_0*n26) / (n25+n26)
    print(f'ベース16F: 2324:{roi0["2324"][0]:+.2%}  2025:{r25_0:+.2%}  '
          f'2026:{r26_0:+.2%}  25+26:{comb0:+.2%}')
    print()

    # NaN率確認
    print('各候補のNaN率（ダ中長距離データ）:')
    for cand in NEW_CANDIDATES:
        if cand in df.columns:
            nan_rate = df[cand].isna().mean()
            nan_trn = df_trn[cand].isna().mean() if cand in df_trn.columns else 1.0
            print(f'  {cand:<35} NaN全体:{nan_rate:.1%}  NaN訓練:{nan_trn:.1%}')
        else:
            print(f'  {cand:<35} [列なし]')
    print()

    # 個別追加効果
    print(f'{"特徴量":<35} {"Δ2025":>8} {"Δ2026":>8} {"Δ25+26":>8}  合意?  最終25+26')
    print('='*80)

    results = []
    for cand in NEW_CANDIDATES:
        if cand in BASE_16 or cand not in df.columns:
            continue
        nll, _, roi = evaluate(df_trn, df_val, oos_parts, BASE_16 + [cand])
        if nll is None:
            continue
        r25 = roi.get('2025', (0, 0))[0]
        r26 = roi.get('2026', (0, 0))[0]
        n25_t = roi.get('2025', (0, 0))[1]
        n26_t = roi.get('2026', (0, 0))[1]
        comb = (r25*n25_t + r26*n26_t) / (n25_t + n26_t) if n25_t+n26_t > 0 else 0
        d25 = r25 - r25_0
        d26 = r26 - r26_0
        d_comb = comb - comb0
        both = (d25 > 0 and d26 > 0)
        results.append((cand, d25, d26, d_comb, comb, both))

    results.sort(key=lambda x: -x[3])
    for cand, d25, d26, d_comb, comb, both in results:
        sym = '✓' if d_comb > 0.003 else ('✗' if d_comb < -0.003 else '~')
        agree = '両期間✓' if both else ''
        print(f'{sym} {cand:<33} {d25:>+8.2%} {d26:>+8.2%} {d_comb:>+8.2%}  {agree:<6}  {comb:+.2%}')

    print()
    print('【有望特徴 (Δ25+26 > 0、両期間合意)】:')
    promising = [(c, d) for c, _, _, d, _, b in results if d > 0.003 and b]
    for c, d in promising:
        print(f'  ✓ {c}: {d:+.2%} [両期間合意]')
    if not promising:
        promising2 = [(c, d) for c, _, _, d, _, _ in results if d > 0.003]
        if promising2:
            for c, d in promising2:
                print(f'  ~ {c}: {d:+.2%}')
        else:
            print('  なし')


if __name__ == '__main__':
    main()
