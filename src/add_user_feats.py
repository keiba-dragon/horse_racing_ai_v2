# coding: utf-8
"""
add_user_feats.py - ユーザー提案特徴を21Fベースに追加テスト
  - ダート経験率_近5走
  - 騎手馬場_r100_勝率（騎手ダート勝率）
  - 前走_地方_flag（前走が地方→JRA転戦）
  - 2走前_馬場状態, 3走前_馬場状態
  - 良馬場_平均着順_近5走
  - コース馬場_r200_勝率（馬場×コース）
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

BASE_21 = [
    '近5走_クラス調整_平均着順', '近5走_タイム指数_max', '1走前_タイム指数', '前走着差タイム',
    '騎手コース_r100_勝率', '1走前_クラス調整着順', '調教師コース_r100_勝率',
    '1走前_RPCI', '1走前_上3F地点差', '斤量', '種牡馬_勝率',
    '間隔_長_flag', '1走前_脚質_num', '騎手変更', '馬番',
    '近3走_複勝率', '調教師_r200_複勝率',
    'ブリンカー変更', '2走前_クラス差', '4走前_クラス差', '1走前_馬場状態',
]

NEW_CANDIDATES = [
    '騎手馬場_r100_勝率',      # 騎手のダート勝率（未使用）
    'ダート経験率_近5走',       # 近5走のダート経験率（新規計算）
    '前走_地方_flag',          # 前走が地方（1走前_開催がNaN）
    '2走前_馬場状態',          # 2走前の馬場状態
    '3走前_馬場状態',          # 3走前の馬場状態
    '良馬場_平均着順_近5走',    # 良馬場限定の平均着順
    'コース馬場_r200_勝率',     # コース×馬場の勝率
    '騎手馬場_r100_複勝率',
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

    # ダート経験率_近5走: 近5走のうちダートだった割合
    dist_cols = [f'{i}走前_距離' for i in range(1, 6) if f'{i}走前_距離' in df.columns]
    if dist_cols:
        dart_flags = pd.DataFrame({
            c: df[c].astype(str).str.startswith('ダ').where(df[c].notna(), np.nan)
            for c in dist_cols
        })
        df['ダート経験率_近5走'] = dart_flags.mean(axis=1)

    # 前走_地方_flag: 1走前_開催がNaN かつ 2走前_開催はある → 前走が地方
    if '1走前_開催' in df.columns and '2走前_開催' in df.columns:
        df['前走_地方_flag'] = (
            df['1走前_開催'].isna() & df['2走前_開催'].notna()
        ).astype(float)

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

    nll0, _, roi0 = evaluate(df_trn, df_val, oos_parts, BASE_21)
    r25_0 = roi0['2025'][0]; r26_0 = roi0['2026'][0]
    n25, n26 = roi0['2025'][1], roi0['2026'][1]
    comb0 = (r25_0*n25 + r26_0*n26) / (n25+n26)
    print(f'ベース21F: 2025:{r25_0:+.2%}  2026:{r26_0:+.2%}  25+26:{comb0:+.2%}\n')

    # NaN率表示
    print('各候補のNaN率（訓練データ）:')
    for cand in NEW_CANDIDATES:
        if cand in df.columns:
            nan_tr = df_trn[cand].isna().mean() if cand in df_trn.columns else 1.0
            print(f'  {cand:<30} NaN訓練:{nan_tr:.1%}')
        else:
            print(f'  {cand:<30} [列なし]')
    print()

    results = []
    for cand in NEW_CANDIDATES:
        if cand in BASE_21 or cand not in df.columns:
            continue
        nan_trn = df_trn[cand].isna().mean() if cand in df_trn.columns else 1.0
        if nan_trn > 0.65:
            print(f'  スキップ {cand} (NaN={nan_trn:.0%})')
            continue
        nll, _, roi = evaluate(df_trn, df_val, oos_parts, BASE_21 + [cand])
        if nll is None:
            continue
        r25 = roi.get('2025', (0, 0))[0]; r26 = roi.get('2026', (0, 0))[0]
        n25_t = roi.get('2025', (0, 0))[1]; n26_t = roi.get('2026', (0, 0))[1]
        comb = (r25*n25_t + r26*n26_t) / (n25_t+n26_t) if n25_t+n26_t > 0 else 0
        d25 = r25 - r25_0; d26 = r26 - r26_0; d_comb = comb - comb0
        both = (d25 > 0 and d26 > 0)
        results.append((cand, d25, d26, d_comb, comb, both))

    results.sort(key=lambda x: -x[3])
    print(f'{"特徴量":<32} {"Δ2025":>8} {"Δ2026":>8} {"Δ25+26":>8}  合意?')
    print('='*72)
    for cand, d25, d26, d_comb, comb, both in results:
        sym = '✓' if d_comb > 0.003 else ('✗' if d_comb < -0.003 else '~')
        agree = '[両✓]' if both else ''
        print(f'{sym} {cand:<30} {d25:>+8.2%} {d26:>+8.2%} {d_comb:>+8.2%}  {agree}')

    print()
    promising = [(c, d, b) for c, _, _, d, _, b in results if d > 0.003]
    if promising:
        print('【有望】:')
        for c, d, b in promising:
            print(f'  ✓ {c}: {d:+.2%}{"  [両✓]" if b else ""}')
    else:
        print('有望なし')


if __name__ == '__main__':
    main()
