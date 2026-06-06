# coding: utf-8
"""
compare_shiba_short6.py - 芝短距離 Round6
* R5 best P7=7特徴(+34.10%), Q1=8特徴(+27.93%)
* Q1(8特徴)をベースに 2 特徴追加で 10 特徴を目指す
* 合わせて 9 特徴の候補も確認
"""
import sys, os, time
import numpy as np
import pandas as pd

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from save_conditional_logit import prepare, segment_softmax, BASE_DIR, DATA_FILE, LR, N_EPOCHS, PATIENCE
from save_v3 import add_computed_features

L2 = 0.006

ALL_FEATS = [
    '1走前_3角', '芝ダ転向', '距離変化_前走', '1走前_脚質_num',
    '近5走_上り3F平均', '間隔', 'コース脚質_r200_勝率', '同会場_平均着順_近5走',
    '良馬場_平均着順_近5走', '2走前_クラス差', '1走前_クラス差',
    'コース枠_r200_勝率', '馬距離_勝率', '斤量', '馬番', '馬体重',
    '前走着差タイム', '1走前_タイム指数', '性別_num', '1走前_馬場状態',
]

# Round5 best: Q1 = P7 + 同会場_平均着順_近5走 (8特徴, +27.93%)
BASE_Q1 = ['1走前_3角', '芝ダ転向', '距離変化_前走', '1走前_脚質_num',
            '間隔', '近5走_上り3F平均', 'コース脚質_r200_勝率',
            '同会場_平均着順_近5走']

SETS = {
    'P7: 7特徴(参考)':               ['1走前_3角', '芝ダ転向', '距離変化_前走', '1走前_脚質_num',
                                       '間隔', '近5走_上り3F平均', 'コース脚質_r200_勝率'],
    'Q1: 8特徴(参考)':               BASE_Q1,
    'R1: Q1+良馬場着順':              BASE_Q1 + ['良馬場_平均着順_近5走'],
    'R2: Q1+2走クラス差':             BASE_Q1 + ['2走前_クラス差'],
    'R3: Q1+斤量':                    BASE_Q1 + ['斤量'],
    'R4: Q1+馬番':                    BASE_Q1 + ['馬番'],
    'R5: Q1+性別':                    BASE_Q1 + ['性別_num'],
    'R6: Q1+良馬場+斤量':             BASE_Q1 + ['良馬場_平均着順_近5走', '斤量'],
    'R7: Q1+2走クラス差+斤量':        BASE_Q1 + ['2走前_クラス差', '斤量'],
    'R8: Q1+2走クラス差+良馬場':       BASE_Q1 + ['2走前_クラス差', '良馬場_平均着順_近5走'],
    'R9: Q1+斤量+馬番':               BASE_Q1 + ['斤量', '馬番'],
    'R10: Q1+性別+斤量':              BASE_Q1 + ['性別_num', '斤量'],
    'R11: Q1+コース枠WR+斤量':        BASE_Q1 + ['コース枠_r200_勝率', '斤量'],
    'R12: Q1+前走着差タイム+斤量':     BASE_Q1 + ['前走着差タイム', '斤量'],
}


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
    df = df[(df['surface'] == '芝') & (dm <= 1400)].copy()
    df['dist_m'] = dm[df.index]
    df = add_computed_features(df)
    baba_map = {'良': 0, '稍重': 1, '重': 2, '不良': 3}
    for col in df.columns:
        if '馬場状態' in col:
            df[col] = df[col].map(baba_map)
    for col in ALL_FEATS:
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


def tie_rate_2026(scored_df):
    s = scored_df.copy()
    s['_rm'] = s.groupby('race_id')['prob'].rank(ascending=False, method='min')
    ties = s[s['_rm'] == 1].groupby('race_id').size()
    return (ties > 1).mean()


def evaluate_set(df_trn, df_val, oos_2324, oos_2025, oos_2026, feats):
    valid = [c for c in feats if c in df_trn.columns and df_trn[c].isna().mean() <= 0.65]
    if not valid:
        return {k: (float('nan'), 0) for k in ['2324', '2025', '2026']}, valid, None, float('nan')
    X_tr, y_tr, gs_tr, n_tr, nr_tr, scaler, *_ = prepare(
        df_trn, valid, top_idx=None, top_idx3=None, fit=True)
    X_va, y_va, gs_va, n_va, nr_va, *_ = prepare(
        df_val, valid, scaler=scaler, top_idx=None, top_idx3=None)
    beta = adam_fit(X_tr, y_tr, gs_tr, n_tr, nr_tr, X_va, y_va, gs_va, n_va, nr_va)
    results = {}
    tie26 = float('nan')
    for label, oos in [('2324', oos_2324), ('2025', oos_2025), ('2026', oos_2026)]:
        if len(oos) == 0:
            results[label] = (float('nan'), 0)
            continue
        valid_p = [c for c in valid if c in oos.columns]
        X_p, _, gs_p, n_p, *_ = prepare(oos, valid_p, scaler=scaler,
                                          top_idx=None, top_idx3=None)
        scored = oos.sort_values('race_id').reset_index(drop=True)
        scored['prob'] = segment_softmax(X_p @ beta, gs_p, n_p)
        scored['rank'] = scored.groupby('race_id')['prob'].rank(ascending=False, method='first')
        top1 = scored[scored['rank'] == 1]
        results[label] = roi_from_top1(top1)
        if label == '2026':
            tie26 = tie_rate_2026(scored)
    return results, valid, beta, tie26


def main():
    t0 = time.time()
    P7_ROI, Q1_ROI = +0.3410, +0.2793
    print("=" * 82)
    print("  芝短距離 Round6 — Q1(8特徴,+27.93%)に 2 特徴追加で 10 特徴へ")
    print(f"  ベースQ1: {BASE_Q1}")
    print("  ※ 2324は参考のみ。25+26で優劣を判断する")
    print("=" * 82)

    df = load_segment()
    df_trn   = df[(df['日付_num'] >= 130101) & (df['日付_num'] < 220101)]
    df_val   = df[(df['日付_num'] >= 220101) & (df['日付_num'] <= 221231)]
    oos_2324 = df[(df['日付_num'] >= 230101) & (df['日付_num'] < 250101)]
    oos_2025 = df[(df['日付_num'] >= 250101) & (df['日付_num'] < 260101)]
    oos_2026 = df[df['日付_num'] >= 260101]

    print(f"\ntrain:{len(df_trn):,}行({df_trn['race_id'].nunique()}R)  "
          f"val:{len(df_val):,}行({df_val['race_id'].nunique()}R)")
    print(f"2324:{oos_2324['race_id'].nunique()}R  "
          f"2025:{oos_2025['race_id'].nunique()}R  "
          f"2026:{oos_2026['race_id'].nunique()}R")

    print(f"\n{'='*82}")
    print(f"  {'セット':36s}  {'2324':>7}  {'2025':>7}  {'2026':>7}  {'25+26':>7}  "
          f"{'タイ率':>6}  特徴数")
    print(f"  {'-'*78}")

    best_comb, best_name = -999.0, None
    for name, feats in SETS.items():
        t1 = time.time()
        res, valid, beta, tie26 = evaluate_set(
            df_trn, df_val, oos_2324, oos_2025, oos_2026, feats)
        r2324, _ = res['2324']
        r25, n25 = res['2025']
        r26, n26 = res['2026']
        rcomb    = comb2526(r25, n25, r26, n26)
        marker   = ' ←best' if rcomb > best_comb else ''
        if rcomb > best_comb:
            best_comb, best_name = rcomb, name
        tie_str = f'{tie26*100:5.1f}%' if not np.isnan(tie26) else '  N/A'
        print(f"  {name:36s}  {r2324*100:+6.2f}%  {r25*100:+6.2f}%  {r26*100:+6.2f}%  "
              f"{rcomb*100:+6.2f}%  {tie_str}  {len(valid)}個  "
              f"({int(time.time()-t1)}s){marker}")
        if beta is not None:
            for f, b in zip(valid, beta):
                print(f"      β {f}: {b:+.4f}")

    print(f"\n{'='*82}")
    print(f"  Round6 ベスト: {best_name}  25+26={best_comb*100:.2f}%")
    print(f"  P7(7特徴)比: {(best_comb - P7_ROI)*100:+.2f}pp")
    print(f"  Q1(8特徴)比: {(best_comb - Q1_ROI)*100:+.2f}pp")
    print(f"  総時間: {int(time.time()-t0)}s")


if __name__ == '__main__':
    main()
