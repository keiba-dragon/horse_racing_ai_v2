# coding: utf-8
"""
greedy_da_naga.py - ダ_中長距離 greedy feature search
死んでいる3特徴量を削除したベースから1本ずつ候補を追加
基準: val NLL (2022, 選択バイアスなし)
OOS 2324/2025/2026 ROI も表示
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

SURF, DIST = 'ダ', '中長距離'
SEG_KEY    = f'{SURF}_{DIST}'

# ── ベース特徴量（v303 - 死んでいる3本） ──────────────────────────────────
BASE_FEATS = [
    '近5走_クラス調整_平均着順',
    '1走前_上り3F',
    '近5走_タイム指数平均',
    '近5走_タイム指数_max',
    '1走前_タイム指数',
    '前走着差タイム',
    '騎手コース_r100_勝率',
    '1走前_クラス調整着順',
    '調教師コース_r100_勝率',
    '1走前_RPCI',
    '1走前_上3F地点差',
    '斤量',
    '種牡馬_勝率',
    '間隔_長_flag',
    '1走前_脚質_num',
    '騎手変更',
    '馬番',
    # 削除済み: 種牡馬_ダ_勝率(0.000), 間隔(0.001), コース枠_r200_勝率(0.018)
]

# ── 追加候補（優先度順） ──────────────────────────────────────────────────
CANDIDATES = [
    # ダート長距離向き
    '展開フィット_v2',        # ペース適性（ダ短で有効）
    '馬体重',                 # 重い馬がダート長距離で有利
    '馬体重増減',             # 体重変化
    '乗替り_近走不振',        # 騎手変更×近走不振
    # タイム・フォーム系
    'タイム指数_近3走_slope', # 上昇トレンド
    '2走前_タイム指数',       # 2走前実績
    '3走前_タイム指数',       # 3走前実績
    '近5走_タイム指数_min',   # 最低タイム指数
    '近5走_タイム指数_std',   # タイム指数のばらつき
    # 複勝・勝率系
    '近3走_複勝率',
    '近5走_複勝率',
    '近10走_勝率',
    '近3走_平均着順',
    # コース適性
    'コース脚質_r200_勝率',
    '芝ダ転向',
    '血統_ダ優位度',          # 種牡馬ダ勝率 - 種牡馬全勝率
    # 出走環境
    'クラス_rank',
    '出走頭数',
    '間隔',                   # 再テスト
    '間隔_短_flag',
    'コース枠_r200_勝率',     # 再テスト
    '種牡馬_ダ_勝率',         # 再テスト
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
    df = df[df['surface'].isin(['芝', 'ダ'])].copy()
    if 'クラス_rank' in df.columns:
        df = df[df['クラス_rank'] != 1.0].copy()
    dm = pd.to_numeric(df['距離'].astype(str).str.extract(r'(\d+)')[0], errors='coerce')
    df = df[df['surface'] == SURF].copy()
    df['dist_m'] = dm[df.index]
    if DIST == '短距離':
        df = df[df['dist_m'] <= 1400]
    else:
        df = df[df['dist_m'] > 1400]
    df = add_computed_features(df)
    return df


def _loss_grad(beta, X, y, gs, n, nr, alpha=0.0):
    probs = segment_softmax(X @ beta, gs, n)
    res   = y - probs
    loss  = (-np.sum(y * np.log(np.clip(probs, 1e-15, 1.0))) + alpha * np.sum(beta**2)) / nr
    grad  = (-(X.T @ res) + 2 * alpha * beta) / nr
    return loss, grad


def adam_fit(X_tr, y_tr, gs_tr, n_tr, nr_tr,
             X_va, y_va, gs_va, n_va, nr_va, alpha=0.0):
    d = X_tr.shape[1]
    beta, m, v = np.zeros(d), np.zeros(d), np.zeros(d)
    b1, b2, eps = 0.9, 0.999, 1e-8
    t, best_val, best_beta, no_imp = 0, np.inf, np.zeros(d), 0
    for epoch in range(1, N_EPOCHS + 1):
        _, grad = _loss_grad(beta, X_tr, y_tr, gs_tr, n_tr, nr_tr, alpha)
        t += 1
        m = b1*m + (1-b1)*grad
        v = b2*v + (1-b2)*grad**2
        beta -= LR * (m/(1-b1**t)) / (np.sqrt(v/(1-b2**t)) + eps)
        if epoch % 10 == 0:
            vl, _ = _loss_grad(beta, X_va, y_va, gs_va, n_va, nr_va, alpha)
            if vl < best_val:
                best_val, best_beta, no_imp = vl, beta.copy(), 0
            else:
                no_imp += 1
            if no_imp >= PATIENCE // 10:
                break
    return best_beta, best_val


def evaluate(df_trn, df_val, oos_parts, feats):
    valid = [c for c in feats if c in df_trn.columns and df_trn[c].isna().mean() <= 0.65]
    if len(valid) == 0:
        return None, None, {}

    X_tr, y_tr, gs_tr, n_tr, nr_tr, scaler, *_ = prepare(
        df_trn, valid, top_idx=None, top_idx3=None, fit=True)
    X_va, y_va, gs_va, n_va, nr_va, *_ = prepare(
        df_val, valid, scaler=scaler, top_idx=None, top_idx3=None)

    beta, val_nll = adam_fit(X_tr, y_tr, gs_tr, n_tr, nr_tr,
                             X_va, y_va, gs_va, n_va, nr_va)

    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(segment_softmax(X_va @ beta, gs_va, n_va), y_va)

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


def main():
    print(f'読み込み: {DATA_FILE}')
    df = load_segment()
    print(f'[{SEG_KEY}] 有効行: {len(df):,}')

    df_trn = df[(df['日付_num'] >= 130101) & (df['日付_num'] < 220101)]
    df_val = df[(df['日付_num'] >= 220101) & (df['日付_num'] <= 221231)]
    oos    = df[df['日付_num'] >= 230101]
    oos_parts = {
        '2324': oos[oos['日付_num'] < 250101],
        '2025': oos[(oos['日付_num'] >= 250101) & (oos['日付_num'] < 260101)],
        '2026': oos[oos['日付_num'] >= 260101],
    }
    print(f'trn:{len(df_trn):,} val:{len(df_val):,} '
          f'oos2324:{len(oos_parts["2324"]):,} '
          f'oos2025:{len(oos_parts["2025"]):,} '
          f'oos2026:{len(oos_parts["2026"]):,}\n')

    # ── ベース評価 ──────────────────────────────────────────────────────────
    t0 = time.time()
    val_nll, valid, oos_roi = evaluate(df_trn, df_val, oos_parts, BASE_FEATS)
    current_feats = list(valid)
    best_nll      = val_nll

    def fmt_roi(oos_roi):
        parts = []
        total_n, total_roi = 0, 0
        for p in ['2324', '2025', '2026']:
            if p in oos_roi:
                roi, n = oos_roi[p]
                parts.append(f'{p}:{roi:+.2%}({n}R)')
                total_n += n
                total_roi += roi * n
        recent = [(oos_roi[p][0]*oos_roi[p][1], oos_roi[p][1])
                  for p in ['2025','2026'] if p in oos_roi]
        if recent:
            rn = sum(x[1] for x in recent)
            rr = sum(x[0] for x in recent) / rn
            parts.append(f'25+26:{rr:+.2%}({rn}R)')
        return '  '.join(parts)

    print(f'【ベース {len(current_feats)}特徴】valNLL={best_nll:.5f}')
    print(f'  {fmt_roi(oos_roi)}\n')

    # ── greedy 追加 ────────────────────────────────────────────────────────
    added = []
    skipped = []

    for cand in CANDIDATES:
        if cand in current_feats:
            continue
        trial = current_feats + [cand]
        t1 = time.time()
        nll, valid_t, roi_t = evaluate(df_trn, df_val, oos_parts, trial)
        elapsed = time.time() - t1

        if nll is None:
            skipped.append(f'{cand}(列なし)')
            continue

        # 候補が実際に使われたか確認
        actually_added = cand in valid_t

        delta = nll - best_nll
        symbol = '✓' if delta < -1e-5 else '✗'
        na_note = '' if actually_added else ' [NaN率>65%→除外]'
        print(f'{symbol} +{cand}{na_note}')
        print(f'    valNLL={nll:.5f} (Δ{delta:+.5f})  {fmt_roi(roi_t)}  [{elapsed:.0f}s]')

        if delta < -1e-5 and actually_added:
            current_feats = list(valid_t)
            best_nll      = nll
            added.append(cand)
            print(f'    → 採用  現在{len(current_feats)}特徴\n')
        else:
            print()

    # ── 最終結果 ─────────────────────────────────────────────────────────
    print('='*60)
    print(f'最終特徴量セット ({len(current_feats)}本):')
    for f in current_feats:
        print(f'  {f}')
    print(f'\n追加採用: {added}')
    if skipped:
        print(f'スキップ: {skipped}')
    print(f'\n最終 valNLL={best_nll:.5f}')
    print(f'総時間: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
