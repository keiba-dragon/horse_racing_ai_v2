# coding: utf-8
"""
ablation_da.py - ダ_中長距離 domain特徴の累積効果確認
17 base → 各特徴を順次追加してOOS ROI(2025/2026)の変化を追跡
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

BASE_17 = [
    '近5走_クラス調整_平均着順', '1走前_上り3F', '近5走_タイム指数平均',
    '近5走_タイム指数_max', '1走前_タイム指数', '前走着差タイム',
    '騎手コース_r100_勝率', '1走前_クラス調整着順', '調教師コース_r100_勝率',
    '1走前_RPCI', '1走前_上3F地点差', '斤量', '種牡馬_勝率',
    '間隔_長_flag', '1走前_脚質_num', '騎手変更', '馬番',
]

DOMAIN_9 = [
    '馬コース_r20_勝率',
    '馬コース_r20_複勝率',
    '同距離帯_平均着順_近5走',
    '同会場_平均着順_近5走',
    '馬距離_勝率',
    '騎手コース距離_r100_勝率',
    '4角位置_近3走_slope',
    '距離変化_前走',
    '同馬場_平均着順_近5走',
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


def fmt(oos_roi):
    parts = []
    for p in ['2025', '2026']:
        if p in oos_roi:
            roi, n = oos_roi[p]
            parts.append(f'{p}:{roi:+.2%}({n}R)')
    if '2025' in oos_roi and '2026' in oos_roi:
        r5, n5 = oos_roi['2025']
        r6, n6 = oos_roi['2026']
        comb = (r5*n5 + r6*n6) / (n5+n6)
        parts.append(f'25+26:{comb:+.2%}')
    return '  '.join(parts)


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

    # ベース
    nll0, _, roi0 = evaluate(df_trn, df_val, oos_parts, BASE_17)
    r25_0 = roi0['2025'][0] if '2025' in roi0 else 0
    r26_0 = roi0['2026'][0] if '2026' in roi0 else 0
    print(f'ベース17: valNLL={nll0:.5f}  {fmt(roi0)}')
    print()

    # 各ドメイン特徴を累積追加
    print('累積追加（ドメイン特徴 順次）:')
    print(f'{"特徴量":<30} {"valNLL":>9} {"Δ2025":>8} {"Δ2026":>8} {"Δ25+26":>8}  {" 結果"}')
    print('-'*80)

    cur_feats = list(BASE_17)
    prev_roi25 = r25_0
    prev_roi26 = r26_0

    for feat in DOMAIN_9:
        cur_feats = cur_feats + [feat]
        nll, _, roi = evaluate(df_trn, df_val, oos_parts, cur_feats)
        r25 = roi['2025'][0] if '2025' in roi else 0
        r26 = roi['2026'][0] if '2026' in roi else 0
        d25 = r25 - prev_roi25
        d26 = r26 - prev_roi26
        if '2025' in roi and '2026' in roi:
            n5, n6 = roi['2025'][1], roi['2026'][1]
            comb = (r25*n5 + r26*n6)/(n5+n6)
            comb0 = (prev_roi25*n5 + prev_roi26*n6)/(n5+n6)
            d_comb = comb - comb0
        else:
            d_comb = 0
        print(f'{feat:<30} {nll:>9.5f} {d25:>+8.2%} {d26:>+8.2%} {d_comb:>+8.2%}  {fmt(roi)}')
        prev_roi25 = r25
        prev_roi26 = r26

    print()
    print('--- 個別効果（ベース17に1本ずつ追加） ---')
    print(f'{"特徴量":<30} {"valNLL":>9} {"Δ2025":>8} {"Δ2026":>8} {"Δ25+26":>8}')
    print('-'*65)
    for feat in DOMAIN_9:
        nll, _, roi = evaluate(df_trn, df_val, oos_parts, BASE_17 + [feat])
        r25 = roi['2025'][0] if '2025' in roi else 0
        r26 = roi['2026'][0] if '2026' in roi else 0
        d25 = r25 - r25_0
        d26 = r26 - r26_0
        if '2025' in roi and '2026' in roi:
            n5, n6 = roi['2025'][1], roi['2026'][1]
            comb = (r25*n5 + r26*n6)/(n5+n6)
            comb0 = (r25_0*n5 + r26_0*n6)/(n5+n6)
            d_comb = comb - comb0
        else:
            d_comb = 0
        print(f'{feat:<30} {nll:>9.5f} {d25:>+8.2%} {d26:>+8.2%} {d_comb:>+8.2%}')


if __name__ == '__main__':
    main()
