# coding: utf-8
"""
greedy_da_naga2.py - ダ_中長距離 ドメイン知識特徴量 greedy search
ベース: 17特徴（v303 - 死んでいる3本）
追加候補: リピーター・マクリ・ペース等のドメイン知識由来特徴量
"""
import sys, os, time
import numpy as np
import pandas as pd

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from save_conditional_logit import (
    prepare, segment_softmax, BASE_DIR, DATA_FILE, LR, N_EPOCHS, PATIENCE
)
from sklearn.isotonic import IsotonicRegression
from save_v3 import add_computed_features, calc_roi

BASE_FEATS = [
    '近5走_クラス調整_平均着順', '1走前_上り3F', '近5走_タイム指数平均',
    '近5走_タイム指数_max', '1走前_タイム指数', '前走着差タイム',
    '騎手コース_r100_勝率', '1走前_クラス調整着順', '調教師コース_r100_勝率',
    '1走前_RPCI', '1走前_上3F地点差', '斤量', '種牡馬_勝率',
    '間隔_長_flag', '1走前_脚質_num', '騎手変更', '馬番',
]

DOMAIN_CANDIDATES = [
    # リピーター（最重要 - ダ長距離専門家）
    '馬コース_r20_勝率',
    '馬コース_r20_複勝率',
    '同距離帯_平均着順_近5走',
    '同会場_平均着順_近5走',
    '同会場_複勝率_近5走',
    '同会場_出走数_近5走',
    '馬距離_勝率',
    '馬距離_複勝率',
    '騎手コース距離_r100_勝率',
    # マクリ・4角位置（3コーナーからのポジション変化）
    '近5走_平均4角位置',
    '4角位置_近3走_slope',
    '1走前_3角',
    '1走前_4角',
    '前走_4角位置',
    # ペース・展開（先行有利度・頭数）
    'コース_先行有利度',
    'レース内_逃げ馬数',
    'レース内_先行馬数',
    '推定ペース',
    '展開_コース_脚質フィット',
    'コース展開マッチ',
    # 距離適性・馬場
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
    return val_nll, valid, oos_roi


def fmt(oos_roi):
    parts = []
    for p in ['2324', '2025', '2026']:
        if p in oos_roi:
            roi, n = oos_roi[p]
            parts.append(f'{p}:{roi:+.2%}({n}R)')
    if '2025' in oos_roi and '2026' in oos_roi:
        r5, n5 = oos_roi['2025']
        r6, n6 = oos_roi['2026']
        comb = (r5*n5 + r6*n6) / (n5+n6)
        parts.append(f'25+26:{comb:+.2%}({n5+n6}R)')
    return '  '.join(parts)


def main():
    print(f'読み込み: {DATA_FILE}')
    df = load_segment()
    print(f'ダ_中長距離: {len(df):,}行\n')

    df_trn = df[(df['日付_num'] >= 130101) & (df['日付_num'] < 220101)]
    df_val = df[(df['日付_num'] >= 220101) & (df['日付_num'] <= 221231)]
    oos    = df[df['日付_num'] >= 230101]
    oos_parts = {
        '2324': oos[oos['日付_num'] < 250101],
        '2025': oos[(oos['日付_num'] >= 250101) & (oos['日付_num'] < 260101)],
        '2026': oos[oos['日付_num'] >= 260101],
    }

    t0 = time.time()
    val_nll, valid, oos_roi = evaluate(df_trn, df_val, oos_parts, BASE_FEATS)
    current_feats = list(valid)
    best_nll = val_nll
    print(f'【ベース {len(current_feats)}特徴】valNLL={best_nll:.5f}')
    print(f'  {fmt(oos_roi)}\n')

    added = []
    for cand in DOMAIN_CANDIDATES:
        if cand in current_feats:
            continue
        t1 = time.time()
        nll, valid_t, roi_t = evaluate(df_trn, df_val, oos_parts, current_feats + [cand])
        elapsed = time.time() - t1
        if nll is None:
            print(f'✗ +{cand} [列なし]')
            continue
        actually_added = cand in valid_t
        delta = nll - best_nll
        sym = '✓' if (delta < -1e-5 and actually_added) else '✗'
        note = '' if actually_added else ' [NaN>65%]'
        print(f'{sym} +{cand}{note}')
        print(f'    valNLL={nll:.5f}(Δ{delta:+.5f})  {fmt(roi_t)}  [{elapsed:.0f}s]')
        if delta < -1e-5 and actually_added:
            current_feats = list(valid_t)
            best_nll = nll
            added.append(cand)
            print(f'    → 採用 ({len(current_feats)}特徴)')
        print()

    print('='*60)
    print(f'ドメイン特徴追加結果 ({len(added)}本採用):')
    for f in added:
        print(f'  + {f}')
    print(f'\n最終 valNLL={best_nll:.5f}  特徴数={len(current_feats)}')
    print(f'総時間: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
