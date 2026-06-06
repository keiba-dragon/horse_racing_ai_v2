# coding: utf-8
"""
compare_shiba_short2.py - 芝短距離 Round2
* 目標: ~10特徴でタイ率を下げつつ ROI を維持/改善
* nv1 (4特徴, 25+26=+14.85%) をベースに拡張
* 2324は参考のみ。25+26 で優劣を判断する
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
    '1走前_タイム指数', '前走着差タイム', '1走前_クラス調整着順',
    '近5走_上り3F平均', '馬体重', 'コース枠_r200_勝率', 'コース枠_r200_複勝率',
    'コース脚質_r200_勝率', '馬距離_勝率', '馬距離_複勝率',
    '騎手コース_r100_勝率', '調教師コース_r100_勝率',
    '性別_num', '斤量', '馬番',
    '1走前_クラス差', '2走前_クラス差',
    '良馬場_平均着順_近5走', '同会場_平均着順_近5走',
    '1走前_馬場状態', '間隔',
    '近5走_クラス調整_平均着順', '騎手変更', 'ブリンカー変更',
]

# nv1 = ['1走前_3角', '芝ダ転向', '距離変化_前走', '1走前_脚質_num']  25+26=+14.85%
NV1 = ['1走前_3角', '芝ダ転向', '距離変化_前走', '1走前_脚質_num']

SETS = {
    'A: nv1(参考)':              NV1,
    'B: nv1+騎手+コース':        NV1 + ['騎手コース_r100_勝率', 'コース枠_r200_勝率'],
    'C: nv1+騎手+調教師+コース': NV1 + ['騎手コース_r100_勝率', '調教師コース_r100_勝率',
                                         'コース枠_r200_勝率'],
    'D: nv1+連続特徴4':          NV1 + ['前走着差タイム', '1走前_タイム指数',
                                         'コース枠_r200_勝率', '騎手コース_r100_勝率'],
    'E: nv1+10特徴':             NV1 + ['前走着差タイム', '1走前_クラス調整着順',
                                         'コース枠_r200_勝率', '騎手コース_r100_勝率',
                                         '調教師コース_r100_勝率', '間隔'],
    'F: nv1+クラス+コース+着差': NV1 + ['1走前_クラス差', '2走前_クラス差',
                                         'コース枠_r200_勝率', '騎手コース_r100_勝率',
                                         '前走着差タイム', '馬距離_勝率'],
    'G: 転向なし+10特徴':        ['1走前_3角', '距離変化_前走', '1走前_脚質_num',
                                   '前走着差タイム', '1走前_タイム指数',
                                   'コース枠_r200_勝率', '騎手コース_r100_勝率',
                                   '調教師コース_r100_勝率', '馬距離_勝率', '間隔'],
    'H: 転向なし+クラス補正': ['1走前_3角', '距離変化_前走', '1走前_脚質_num',
                                '1走前_クラス調整着順', '近5走_クラス調整_平均着順',
                                'コース枠_r200_勝率', '騎手コース_r100_勝率',
                                '調教師コース_r100_勝率', '馬距離_勝率', '間隔'],
    'I: 芝中風(10特徴)':         ['調教師コース_r100_勝率', '馬距離_勝率',
                                   '1走前_クラス調整着順', '近5走_クラス調整_平均着順', '間隔',
                                   '1走前_3角', '距離変化_前走', '前走着差タイム',
                                   'コース枠_r200_勝率', '騎手コース_r100_勝率'],
    'J: 幅広12特徴':             ['1走前_3角', '芝ダ転向', '距離変化_前走', '1走前_脚質_num',
                                   '前走着差タイム', '1走前_タイム指数', '1走前_クラス調整着順',
                                   'コース枠_r200_勝率', '騎手コース_r100_勝率',
                                   '調教師コース_r100_勝率', '馬距離_勝率', '間隔'],
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


def tie_rate(scored_df):
    """スコアタイ1位の発生率"""
    scored_df = scored_df.copy()
    scored_df['_rank_min'] = scored_df.groupby('race_id')['prob'].rank(
        ascending=False, method='min')
    ties = scored_df[scored_df['_rank_min'] == 1].groupby('race_id').size()
    return (ties > 1).mean()


def evaluate_set(df_trn, df_val, oos_2324, oos_2025, oos_2026, feats):
    valid = [c for c in feats if c in df_trn.columns and df_trn[c].isna().mean() <= 0.65]
    if not valid:
        return {k: (float('nan'), 0) for k in ['2324', '2025', '2026']}, valid, None, {}
    X_tr, y_tr, gs_tr, n_tr, nr_tr, scaler, *_ = prepare(
        df_trn, valid, top_idx=None, top_idx3=None, fit=True)
    X_va, y_va, gs_va, n_va, nr_va, *_ = prepare(
        df_val, valid, scaler=scaler, top_idx=None, top_idx3=None)
    beta = adam_fit(X_tr, y_tr, gs_tr, n_tr, nr_tr, X_va, y_va, gs_va, n_va, nr_va)
    results = {}
    tie_rates = {}
    for label, oos in [('2324', oos_2324), ('2025', oos_2025), ('2026', oos_2026)]:
        if len(oos) == 0:
            results[label] = (float('nan'), 0)
            tie_rates[label] = float('nan')
            continue
        valid_p = [c for c in valid if c in oos.columns]
        X_p, _, gs_p, n_p, *_ = prepare(oos, valid_p, scaler=scaler,
                                          top_idx=None, top_idx3=None)
        scored = oos.sort_values('race_id').reset_index(drop=True)
        scored['prob'] = segment_softmax(X_p @ beta, gs_p, n_p)
        scored['rank'] = scored.groupby('race_id')['prob'].rank(ascending=False, method='first')
        top1 = scored[scored['rank'] == 1]
        results[label] = roi_from_top1(top1)
        tie_rates[label] = tie_rate(scored)
    return results, valid, beta, tie_rates


def main():
    t0 = time.time()
    NV1_ROI = +0.1485
    print("=" * 80)
    print("  芝短距離 Round2 — ~10特徴でタイ率削減")
    print(f"  nv1 ベスト: +14.85% (4特徴, タイ1位=29.9%)")
    print("  ※ 2324は参考のみ。25+26で優劣を判断する")
    print("=" * 80)

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

    print(f"\n{'='*80}")
    print(f"  {'セット':30s}  {'2324':>7}  {'2025':>7}  {'2026':>7}  {'25+26':>7}  "
          f"{'タイ率':>6}  特徴数")
    print(f"  {'-'*76}")

    best_comb, best_name = -999.0, None
    for name, feats in SETS.items():
        t1 = time.time()
        res, valid, beta, ties = evaluate_set(
            df_trn, df_val, oos_2324, oos_2025, oos_2026, feats)
        r2324, _ = res['2324']
        r25, n25 = res['2025']
        r26, n26 = res['2026']
        rcomb    = comb2526(r25, n25, r26, n26)
        tie_26   = ties.get('2026', float('nan'))
        marker   = ' ←best' if rcomb > best_comb else ''
        if rcomb > best_comb:
            best_comb, best_name = rcomb, name
        print(f"  {name:30s}  {r2324*100:+6.2f}%  {r25*100:+6.2f}%  {r26*100:+6.2f}%  "
              f"{rcomb*100:+6.2f}%  {tie_26*100:5.1f}%  {len(valid)}個  "
              f"({int(time.time()-t1)}s){marker}")
        if beta is not None:
            for f, b in zip(valid, beta):
                print(f"      β {f}: {b:+.4f}")

    print(f"\n{'='*80}")
    print(f"  Round2 ベスト: {best_name}  25+26={best_comb*100:.2f}%")
    print(f"  nv1比: {(best_comb - NV1_ROI)*100:+.2f}pp")
    print(f"  総時間: {int(time.time()-t0)}s")


if __name__ == '__main__':
    main()
