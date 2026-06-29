# coding: utf-8
"""
backward_da_short.py - ダート短距離 backward elimination
BASE_25+コース枠_r200_複勝率 から不要特徴を削る
削除基準: 削除で 2324 ROI が +0.3pp以上 かつ 25+26合算が悪化しない
削除後に forward greedy で追加候補を再試行
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

L2 = 0.006

START_FEATS = [
    '近5走_クラス調整_平均着順', '近5走_タイム指数_max', '1走前_タイム指数', '前走着差タイム',
    '騎手コース_r100_勝率', '1走前_クラス調整着順', '調教師コース_r100_勝率',
    '1走前_RPCI', '1走前_上3F地点差', '斤量', '種牡馬_勝率',
    '間隔_長_flag', '1走前_脚質_num', '騎手変更', '馬番',
    '近3走_複勝率', '調教師_r200_複勝率',
    'ブリンカー変更', '2走前_クラス差', '4走前_クラス差', '1走前_馬場状態',
    '性別_num', '所属_num', 'キャリア_浅い', 'タイム指数_近5走_slope',
    'コース枠_r200_複勝率',
]

# forward 追加候補 (greedy_da_short.py の全候補から重複除く)
ADD_CANDIDATES = [
    'コース枠_r200_勝率',
    'コース脚質_r200_勝率', 'コース脚質_r200_複勝率',
    'コース馬場_r200_勝率', 'コース馬場_r200_複勝率',
    '種牡馬_ダ_勝率', '種牡馬_ダ_複勝率',
    '展開フィット_v2', 'レース内_相対脚質', 'レース内_先行馬数',
    '乗替り_近走不振', '近走連続入着数', '近走_改善トレンド',
    '近3走_勝率', '近5走_複勝率', '近10走_勝率', '近10走_複勝率',
    '1走前_上り3F', 'タイム指数_近3走_slope',
    '近5走_タイム指数平均', '近5走_タイム指数_min', '近5走_タイム指数_std',
    '近5走_上り3F平均', '上り3F_近3走_slope',
    '2走前_タイム指数', '3走前_タイム指数',
    '馬コース_r20_勝率', '馬コース_r20_複勝率',
    '馬距離_勝率', '馬距離_複勝率',
    '同距離帯_平均着順_近5走', '同会場_平均着順_近5走', '同会場_複勝率_近5走',
    '同馬場_平均着順_近5走', '道悪_平均着順_近5走', '芝ダ一致_平均着順_近5走',
    '良馬場_平均着順_近5走',
    '騎手コース距離_r100_勝率', '騎手脚質_r100_勝率', '騎手馬場_r100_勝率',
    '騎手距離_r100_勝率', '騎手調教師_r100_勝率',
    '馬体重', '馬体重増減', '近3走_体重増減合計', '馬体重トレンド_近5走',
    '1走前_3角', '1走前_4角', '4角位置_近3走_slope', '近5走_平均4角位置', '前走_4角位置',
    '距離変化_前走', '芝ダ転向', '1走前_クラス差', '3走前_クラス差',
    'クラス_rank', '格上経験数_近5走',
    '年齢', 'キャリア', 'キャリア_log', '内外枠', '斤量変化', '頭数',
    '間隔', '間隔_短_flag',
    '近5走_着差タイム_クラス補正平均', '近5走_平均相対着順',
    '最大クラス差_近5走', '近5走_クラス補正スコア',
    '母父馬_勝率', '生産者_勝率', '産地_勝率', '脚質フィット',
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
    df = df[(df['surface'] == 'ダ') & (dm <= 1400)].copy()
    df['dist_m'] = dm[df.index]
    if 'クラス_rank' in df.columns:
        df = df[df['クラス_rank'] != 1.0].copy()
    df = add_computed_features(df)
    if '間隔_長_flag' not in df.columns or df['間隔_長_flag'].isna().all():
        interval = pd.to_numeric(df.get('間隔', pd.Series(np.nan, index=df.index)), errors='coerce')
        df['間隔_長_flag'] = (interval >= 60).astype(float)
    if '間隔_短_flag' not in df.columns or df['間隔_短_flag'].isna().all():
        interval = pd.to_numeric(df.get('間隔', pd.Series(np.nan, index=df.index)), errors='coerce')
        df['間隔_短_flag'] = (interval <= 14).astype(float)
    baba_map = {'良': 0, '稍重': 1, '重': 2, '不良': 3}
    for col in df.columns:
        if '馬場状態' in col:
            df[col] = df[col].map(baba_map)
    all_cands = list(set(START_FEATS + ADD_CANDIDATES))
    for col in all_cands:
        if col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception:
                df[col] = np.nan
    return df


def _loss_grad(beta, X, y, gs, n, nr, l2=0.0):
    probs = segment_softmax(X @ beta, gs, n)
    loss  = -np.sum(y * np.log(np.clip(probs, 1e-15, 1.0))) / nr + l2 * np.dot(beta, beta)
    grad  = -(X.T @ (y - probs)) / nr + 2 * l2 * beta
    return loss, grad


def adam_fit(X_tr, y_tr, gs_tr, n_tr, nr_tr, X_va, y_va, gs_va, n_va, nr_va, l2=L2):
    d = X_tr.shape[1]
    beta, m, v = np.zeros(d), np.zeros(d), np.zeros(d)
    b1, b2, eps = 0.9, 0.999, 1e-8
    t, best_val, best_beta, no_imp = 0, np.inf, np.zeros(d), 0
    for epoch in range(1, N_EPOCHS + 1):
        _, grad = _loss_grad(beta, X_tr, y_tr, gs_tr, n_tr, nr_tr, l2)
        t += 1
        m = b1*m + (1-b1)*grad
        v = b2*v + (1-b2)*grad**2
        beta -= LR * (m/(1-b1**t)) / (np.sqrt(v/(1-b2**t)) + eps)
        if epoch % 10 == 0:
            vl, _ = _loss_grad(beta, X_va, y_va, gs_va, n_va, nr_va, l2=0.0)
            if vl < best_val:
                best_val, best_beta, no_imp = vl, beta.copy(), 0
            else:
                no_imp += 1
            if no_imp >= PATIENCE // 10:
                break
    return best_beta


def roi_from_top1(top1):
    won  = top1['着順_num'] == 1
    odds = pd.to_numeric(top1['単勝オッズ'], errors='coerce')
    if len(top1) == 0:
        return float('nan'), 0
    return (odds[won] * 100).sum() / (len(top1) * 100) - 1, len(top1)


def comb2526(r25, n25, r26, n26):
    if n25 + n26 == 0:
        return 0.0
    return (r25 * n25 + r26 * n26) / (n25 + n26)


def evaluate(df_trn, df_val, oos_2324, oos_2025, oos_2026, feats):
    valid = [c for c in feats if c in df_trn.columns and df_trn[c].isna().mean() <= 0.65]
    if not valid:
        return None, valid, 0, 0, 0, (0, 0, 0)
    X_tr, y_tr, gs_tr, n_tr, nr_tr, scaler, *_ = prepare(
        df_trn, valid, top_idx=None, top_idx3=None, fit=True)
    X_va, y_va, gs_va, n_va, nr_va, *_ = prepare(
        df_val, valid, scaler=scaler, top_idx=None, top_idx3=None)
    beta = adam_fit(X_tr, y_tr, gs_tr, n_tr, nr_tr,
                    X_va, y_va, gs_va, n_va, nr_va)
    rois, ns = [], []
    for df_p in [oos_2324, oos_2025, oos_2026]:
        if len(df_p) == 0:
            rois.append(0.0); ns.append(0); continue
        valid_p = [c for c in valid if c in df_p.columns]
        X_p, _, gs_p, n_p, *_ = prepare(df_p, valid_p, scaler=scaler,
                                          top_idx=None, top_idx3=None)
        scored = df_p.sort_values('race_id').reset_index(drop=True)
        scored['prob'] = segment_softmax(X_p @ beta, gs_p, n_p)
        scored['rank'] = scored.groupby('race_id')['prob'].rank(ascending=False, method='first')
        top1 = scored[scored['rank'] == 1]
        r, n = roi_from_top1(top1)
        rois.append(r); ns.append(n)
    val_loss = _loss_grad(beta, X_va, y_va, gs_va, n_va, nr_va, l2=0.0)[0]
    return val_loss, valid, rois[0], rois[1], rois[2], tuple(ns)


def run_backward(label, start_feats, df_trn, df_val, oos_2324, oos_2025, oos_2026):
    print(f'\n{"="*65}')
    print(f'  Backward elimination: {label}')
    print(f'{"="*65}')
    _, _, r2324_0, r25_0, r26_0, ns_0 = evaluate(
        df_trn, df_val, oos_2324, oos_2025, oos_2026, start_feats)
    n2324_0, n2025_0, n2026_0 = ns_0
    comb_0 = comb2526(r25_0, n2025_0, r26_0, n2026_0)
    print(f'  開始: 2324:{r2324_0:+.2%}  2025:{r25_0:+.2%}  2026:{r26_0:+.2%}  25+26:{comb_0:+.2%}  ({len(start_feats)}特徴)')

    current = list(start_feats)
    removed = []
    improved = True
    while improved:
        improved = False
        best_gain = 0.0
        best_feat = None
        best_state = None
        for feat in current:
            trial = [f for f in current if f != feat]
            if len(trial) < 3:
                continue
            _, valid_t, r2324, r25, r26, ns_t = evaluate(
                df_trn, df_val, oos_2324, oos_2025, oos_2026, trial)
            n2324_t, n2025_t, n2026_t = ns_t
            comb_t = comb2526(r25, n2025_t, r26, n2026_t)
            delta_2324 = r2324 - r2324_0
            delta_comb = comb_t - comb_0
            # 削除で 2324 が改善 かつ 25+26 も悪化しない
            if delta_2324 > 0.003 and delta_comb >= -0.005 and delta_2324 > best_gain:
                best_gain = delta_2324
                best_feat = feat
                best_state = (r2324, r25, r26, comb_t, ns_t)
        if best_feat:
            r2324_0, r25_0, r26_0, comb_0, ns_0 = best_state
            n2324_0, n2025_0, n2026_0 = ns_0
            current.remove(best_feat)
            removed.append(best_feat)
            improved = True
            print(f'  削除: -{best_feat}  →  2324:{r2324_0:+.2%}  2025:{r25_0:+.2%}  2026:{r26_0:+.2%}  25+26:{comb_0:+.2%}  ({len(current)}特徴)')
    print(f'\n  削除完了: {len(removed)}本削除  残り{len(current)}特徴')
    print(f'  削除済み: {removed}')
    print(f'  最終: 2324:{r2324_0:+.2%}  2025:{r25_0:+.2%}  2026:{r26_0:+.2%}  25+26:{comb_0:+.2%}')
    return current, r2324_0, comb_0


def run_forward(label, start_feats, candidates, r2324_start, comb_start,
                df_trn, df_val, oos_2324, oos_2025, oos_2026):
    print(f'\n{"="*65}')
    print(f'  Forward greedy: {label}')
    print(f'{"="*65}')
    current = list(start_feats)
    best_2324 = r2324_start
    comb_0 = comb_start
    added = []
    for cand in candidates:
        if cand in current:
            continue
        _, valid_t, r2324, r25, r26, ns_t = evaluate(
            df_trn, df_val, oos_2324, oos_2025, oos_2026, current + [cand])
        n2324_t, n2025_t, n2026_t = ns_t
        comb_t = comb2526(r25, n2025_t, r26, n2026_t)
        delta_2324 = r2324 - best_2324
        delta_comb = comb_t - comb_0
        actually_added = cand in valid_t
        good = delta_2324 > 0.005 and actually_added and delta_comb >= -0.005
        sym = '✓' if good else '✗'
        note = '' if actually_added else ' [NaN>65%]'
        print(f'{sym} +{cand}{note}  2324:{r2324:+.2%}(Δ{delta_2324:+.2%})  '
              f'2025:{r25:+.2%}  2026:{r26:+.2%}  25+26:{comb_t:+.2%}(Δ{delta_comb:+.2%})')
        if good:
            current = list(valid_t)
            best_2324 = r2324
            comb_0 = comb_t
            added.append(cand)
            print(f'    → 採用 ({len(current)}特徴) [2324: {best_2324:+.2%}]')
    print(f'\n  Forward完了: {len(added)}本採用')
    # 最終評価
    _, _, r2324f, r25f, r26f, nsf = evaluate(
        df_trn, df_val, oos_2324, oos_2025, oos_2026, current)
    n2324f, n2025f, n2026f = nsf
    combf = comb2526(r25f, n2025f, r26f, n2026f)
    print(f'  最終: 2324:{r2324f:+.2%}({n2324f}R)  2025:{r25f:+.2%}({n2025f}R)  2026:{r26f:+.2%}({n2026f}R)  25+26:{combf:+.2%}')
    return current, r2324f, combf


def main():
    t0 = time.time()
    print('=== ダート短距離 Backward→Forward 最適化 ===')
    print()
    print('データ読み込み中...')
    df = load_segment()
    df_trn   = df[(df['日付_num'] >= 130101) & (df['日付_num'] < 220101)]
    df_val   = df[(df['日付_num'] >= 220101) & (df['日付_num'] <= 221231)]
    oos_2324 = df[(df['日付_num'] >= 230101) & (df['日付_num'] < 250101)]
    oos_2025 = df[(df['日付_num'] >= 250101) & (df['日付_num'] < 260101)]
    oos_2026 = df[df['日付_num'] >= 260101]
    print(f'train:{len(df_trn):,}  val:{len(df_val):,}  2324:{len(oos_2324):,}  2025:{len(oos_2025):,}  2026:{len(oos_2026):,}')

    # ── Phase 1: Backward elimination ───────────────────────────────────────
    slim, r2324_slim, comb_slim = run_backward(
        'BASE_25+コース枠複勝率 → 削減',
        START_FEATS, df_trn, df_val, oos_2324, oos_2025, oos_2026)

    # ── Phase 2: Forward greedy on slim set ─────────────────────────────────
    final_feats, r2324_f, comb_f = run_forward(
        '削減後 → Forward追加',
        slim, ADD_CANDIDATES, r2324_slim, comb_slim,
        df_trn, df_val, oos_2324, oos_2025, oos_2026)

    print(f'\n{"="*65}')
    print(f'総時間: {time.time()-t0:.0f}s')
    print(f'最終特徴量 ({len(final_feats)}個):')
    for f in final_feats:
        print(f'  {f}')


if __name__ == '__main__':
    main()
