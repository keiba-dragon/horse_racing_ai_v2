# coding: utf-8
"""
greedy_da_roi.py - ダ_中長距離 greedy feature search (ROI基準)
選択基準: 2023-24 OOS ROI (val NLLではない)
真のOOS: 2025 / 2026
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

# ベース特徴（17本）
BASE_FEATS = [
    '近5走_クラス調整_平均着順', '1走前_上り3F', '近5走_タイム指数平均',
    '近5走_タイム指数_max', '1走前_タイム指数', '前走着差タイム',
    '騎手コース_r100_勝率', '1走前_クラス調整着順', '調教師コース_r100_勝率',
    '1走前_RPCI', '1走前_上3F地点差', '斤量', '種牡馬_勝率',
    '間隔_長_flag', '1走前_脚質_num', '騎手変更', '馬番',
]

# 候補特徴（ドメイン知識由来 + その他）
CANDIDATES = [
    # リピーター
    '馬コース_r20_勝率',
    '馬コース_r20_複勝率',
    '馬距離_勝率',
    '馬距離_複勝率',
    '同距離帯_平均着順_近5走',
    '同会場_平均着順_近5走',
    '同会場_複勝率_近5走',
    '同馬場_平均着順_近5走',
    '騎手コース距離_r100_勝率',
    # タイム・フォーム系
    '2走前_タイム指数',
    '3走前_タイム指数',
    'タイム指数_近3走_slope',
    '近5走_タイム指数_min',
    '近5走_タイム指数_std',
    '近3走_平均着順',
    '近3走_複勝率',
    '近5走_複勝率',
    '近10走_勝率',
    # 脚質・位置
    '4角位置_近3走_slope',
    '近5走_平均4角位置',
    '前走_4角位置',
    '1走前_3角',
    '1走前_4角',
    # コース・血統
    '距離変化_前走',
    '芝ダ転向',
    '血統_ダ優位度',
    # 体重・環境
    '馬体重',
    '馬体重増減',
    '出走頭数',
    '間隔',
    '間隔_短_flag',
    'クラス_rank',
    '乗替り_近走不振',
    '展開フィット_v2',
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


def get_roi(df_p, beta, scaler, valid):
    valid_p = [c for c in valid if c in df_p.columns]
    X_p, _, gs_p, n_p, *_ = prepare(df_p, valid_p, scaler=scaler,
                                      top_idx=None, top_idx3=None)
    scored = df_p.sort_values('race_id').reset_index(drop=True)
    scored['prob'] = segment_softmax(X_p @ beta, gs_p, n_p)
    scored['rank'] = scored.groupby('race_id')['prob'].rank(
        ascending=False, method='first')
    top1 = scored[scored['rank'] == 1]
    roi, _ = calc_roi(top1)
    return roi, len(top1)


def evaluate(df_trn, df_val, oos_2324, oos_2025, oos_2026, feats):
    valid = [c for c in feats if c in df_trn.columns and df_trn[c].isna().mean() <= 0.65]
    if not valid:
        return None, None, None, None, None, None
    X_tr, y_tr, gs_tr, n_tr, nr_tr, scaler, *_ = prepare(
        df_trn, valid, top_idx=None, top_idx3=None, fit=True)
    X_va, y_va, gs_va, n_va, nr_va, *_ = prepare(
        df_val, valid, scaler=scaler, top_idx=None, top_idx3=None)
    beta, val_nll = adam_fit(X_tr, y_tr, gs_tr, n_tr, nr_tr,
                              X_va, y_va, gs_va, n_va, nr_va)

    roi_2324, n2324 = get_roi(oos_2324, beta, scaler, valid) if len(oos_2324) > 0 else (0, 0)
    roi_2025, n2025 = get_roi(oos_2025, beta, scaler, valid) if len(oos_2025) > 0 else (0, 0)
    roi_2026, n2026 = get_roi(oos_2026, beta, scaler, valid) if len(oos_2026) > 0 else (0, 0)

    return val_nll, valid, roi_2324, roi_2025, roi_2026, (n2324, n2025, n2026)


def comb2526(r25, n25, r26, n26):
    if n25 + n26 == 0:
        return 0
    return (r25*n25 + r26*n26) / (n25+n26)


def main():
    print(f'読み込み: {DATA_FILE}')
    df = load_segment()

    df_trn  = df[(df['日付_num'] >= 130101) & (df['日付_num'] < 220101)]
    df_val  = df[(df['日付_num'] >= 220101) & (df['日付_num'] <= 221231)]
    oos_2324 = df[(df['日付_num'] >= 230101) & (df['日付_num'] < 250101)]
    oos_2025 = df[(df['日付_num'] >= 250101) & (df['日付_num'] < 260101)]
    oos_2026 = df[df['日付_num'] >= 260101]

    print(f'trn:{len(df_trn):,} val:{len(df_val):,} '
          f'oos2324:{len(oos_2324):,} oos2025:{len(oos_2025):,} oos2026:{len(oos_2026):,}\n')

    # ベース評価
    t0 = time.time()
    nll0, v0, r2324_0, r25_0, r26_0, ns = evaluate(
        df_trn, df_val, oos_2324, oos_2025, oos_2026, BASE_FEATS)
    n2324, n2025, n2026 = ns
    comb0 = comb2526(r25_0, n2025, r26_0, n2026)
    print(f'【ベース {len(v0)}特徴】 valNLL={nll0:.5f}')
    print(f'  2324:{r2324_0:+.2%}({n2324}R)  '
          f'2025:{r25_0:+.2%}({n2025}R)  '
          f'2026:{r26_0:+.2%}({n2026}R)  '
          f'25+26:{comb0:+.2%}({n2025+n2026}R)')
    print(f'  選択基準(2324 ROI)={r2324_0:+.2%}\n')

    current_feats = list(v0)
    best_2324_roi = r2324_0

    added = []
    for cand in CANDIDATES:
        if cand in current_feats:
            continue
        t1 = time.time()
        nll, valid_t, r2324, r25, r26, ns_t = evaluate(
            df_trn, df_val, oos_2324, oos_2025, oos_2026,
            current_feats + [cand])
        elapsed = time.time() - t1

        if nll is None:
            print(f'✗ +{cand} [列なし]')
            continue

        actually_added = cand in valid_t
        delta_2324 = r2324 - best_2324_roi
        n2324_t, n2025_t, n2026_t = ns_t
        comb_t = comb2526(r25, n2025_t, r26, n2026_t)
        delta_comb = comb_t - comb0

        sym = '✓' if (delta_2324 > 0.005 and actually_added) else '✗'
        note = '' if actually_added else ' [NaN>65%]'

        print(f'{sym} +{cand}{note}')
        print(f'    2324:{r2324:+.2%}(Δ{delta_2324:+.2%})  '
              f'2025:{r25:+.2%}  2026:{r26:+.2%}  25+26:{comb_t:+.2%}(Δ{delta_comb:+.2%})  '
              f'[{elapsed:.0f}s]')

        if delta_2324 > 0.005 and actually_added:
            current_feats = list(valid_t)
            best_2324_roi = r2324
            comb0 = comb_t
            added.append(cand)
            print(f'    → 採用 ({len(current_feats)}特徴) [選択基準(2324 ROI): {best_2324_roi:+.2%}]')
        print()

    print('='*60)
    print(f'ROI基準greedy結果 ({len(added)}本採用):')
    for f in added:
        print(f'  + {f}')
    print(f'\n最終特徴数={len(current_feats)}  総時間: {time.time()-t0:.0f}s')

    # 最終モデルを全期間で再評価
    print('\n--- 最終モデル再評価 ---')
    _, _, r2324f, r25f, r26f, nsf = evaluate(
        df_trn, df_val, oos_2324, oos_2025, oos_2026, current_feats)
    n2324f, n2025f, n2026f = nsf
    combf = comb2526(r25f, n2025f, r26f, n2026f)
    print(f'  2324:{r2324f:+.2%}({n2324f}R)  '
          f'2025:{r25f:+.2%}({n2025f}R)  '
          f'2026:{r26f:+.2%}({n2026f}R)  '
          f'25+26:{combf:+.2%}({n2025f+n2026f}R)')

    print('\n最終特徴量リスト:')
    for f in current_feats:
        print(f'  {f}')


if __name__ == '__main__':
    main()
