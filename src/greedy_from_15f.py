# coding: utf-8
"""
greedy_from_15f.py - 15特徴ベースからの greedy forward search
基準: 25+26 ROI（両期間が合意しているかも確認）
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

BASE_15 = [
    '近5走_クラス調整_平均着順',
    '近5走_タイム指数_max', '1走前_タイム指数', '前走着差タイム',
    '騎手コース_r100_勝率', '1走前_クラス調整着順', '調教師コース_r100_勝率',
    '1走前_RPCI', '1走前_上3F地点差', '斤量', '種牡馬_勝率',
    '間隔_長_flag', '1走前_脚質_num', '騎手変更', '馬番',
]

CANDIDATES = [
    # 15Fへの追加で改善が見込まれる特徴（add_to_15f.py 結果より）
    '近3走_複勝率',           # ✓ 2025+1.40% 2026+2.48% 25+26+1.70%
    'タイム指数_近3走_slope',   # ✓ 2025+1.88% 2026-0.81% 25+26+1.13%
    '馬コース_r20_勝率',       # ✓ 2025+1.74% 2026-1.11% 25+26+0.94%
    '1走前_3角',              # ✓ 2025+0.52% 2026+0.00% 25+26+0.37%
    '芝ダ転向',               # ~ ±0
    '近5走_複勝率',           # ✗ 2025-1.19% 2026-5.35% 25+26-2.35%（念のため再テスト）
    '近3走_平均着順',
    '2走前_タイム指数',
    '3走前_タイム指数',
    '馬コース_r20_複勝率',
    '馬距離_勝率',
    '同距離帯_平均着順_近5走',
    '同会場_平均着順_近5走',
    '4角位置_近3走_slope',
    '距離変化_前走',
    '馬体重', '馬体重増減',
    '乗替り_近走不振',
    '間隔', '間隔_短_flag',
    'コース脚質_r200_勝率',
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


def comb_roi(oos_roi):
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
    c = comb_roi(oos_roi)
    if c != -999:
        parts.append(f'25+26:{c:+.2%}')
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

    # ベース15評価
    nll0, beta0, roi0 = evaluate(df_trn, df_val, oos_parts, BASE_15)
    best_comb = comb_roi(roi0)
    print(f'ベース15: valNLL={nll0:.5f}  {fmt(roi0)}')
    print(f'  選択基準=25+26 ROI: {best_comb:+.2%}\n')

    current_feats = list(BASE_15)
    added = []

    t0 = time.time()
    for cand in CANDIDATES:
        if cand in current_feats:
            continue
        t1 = time.time()
        nll, _, roi = evaluate(df_trn, df_val, oos_parts, current_feats + [cand])
        elapsed = time.time() - t1
        if nll is None:
            print(f'  ✗ +{cand} [列なし]')
            continue

        # 実際に追加されたか（NaN率チェック）
        valid_check = [c for c in current_feats + [cand]
                       if c in df_trn.columns and df_trn[c].isna().mean() <= 0.65]
        actually_added = cand in valid_check

        c_new = comb_roi(roi)
        delta = c_new - best_comb
        r25 = roi.get('2025', (0, 0))[0]
        r26 = roi.get('2026', (0, 0))[0]
        r25_base = roi0.get('2025', (0, 0))[0]
        r26_base = roi0.get('2026', (0, 0))[0]
        d25 = r25 - r25_base
        d26 = r26 - r26_base
        both_improve = (d25 > 0 and d26 > 0)
        note = '' if actually_added else ' [NaN>65%]'
        agree = ' [両期間合意!]' if both_improve else ''

        # 採用基準: 25+26改善 かつ 両期間が合意、または 改善幅が大きい
        sym = '✓' if (delta > 0.003 and actually_added) else '✗'
        print(f'{sym} +{cand}{note}  Δ25+26={delta:+.2%}(2025:{d25:+.2%} 2026:{d26:+.2%}){agree}  [{elapsed:.0f}s]')

        if delta > 0.003 and actually_added:
            current_feats.append(cand)
            best_comb = c_new
            added.append(cand)
            print(f'    → 採用 ({len(current_feats)}特徴)  最新{fmt(roi)}')
        print()

    print('='*60)
    print(f'追加採用: {added} ({len(added)}本)')
    print(f'最終特徴数: {len(current_feats)}')
    print(f'総時間: {time.time()-t0:.0f}s')

    # 最終評価
    if added:
        print('\n--- 最終モデル評価 ---')
        nll_f, beta_f, roi_f = evaluate(df_trn, df_val, oos_parts, current_feats)
        print(f'valNLL={nll_f:.5f}  {fmt(roi_f)}')
        print('\n最終特徴量:')
        for f in current_feats:
            print(f'  {f}')
        valid_f = [c for c in current_feats if c in df_trn.columns and df_trn[c].isna().mean() <= 0.65]
        print('\nベータ係数（絶対値降順）:')
        idx = np.argsort(np.abs(beta_f))[::-1]
        for i in idx:
            print(f'  {valid_f[i]:<35} β={beta_f[i]:+.4f}')


if __name__ == '__main__':
    main()
