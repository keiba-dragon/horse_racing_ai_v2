# coding: utf-8
"""
compare_da_short_10feat.py - ダ短距離 10特徴探索
  選択指標: 2023-24 OOS ROI  (25+26は報告のみ)
  現nv2(5特徴)をベースに greedy forward selection で 10 特徴を目指す
  セグメント: ダ & ≤1400m & クラス_rank≠1.0
"""
import sys, os, time
import numpy as np
import pandas as pd

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from save_conditional_logit import prepare, segment_softmax, BASE_DIR, DATA_FILE, LR, N_EPOCHS, PATIENCE
from save_v3 import add_computed_features

L2 = 0.006

# 現nv2の5特徴（ベース）
BASE_NV2 = ['近5走_上り3F平均', 'コース枠_r200_勝率', '1走前_馬場状態', '1走前_クラス差', '2走前_クラス差']

# 追加候補（約20個）
CANDIDATES = [
    '間隔',
    '1走前_脚質_num',
    '1走前_3角',
    '距離変化_前走',
    '芝ダ転向',
    '同会場_平均着順_近5走',
    '性別_num',
    '斤量',
    '馬番',
    '馬体重',
    '3走前_クラス差',
    '1走前_クラス調整着順',
    '近5走_クラス調整_平均着順',
    'コース脚質_r200_勝率',
    '馬距離_勝率',
    '前走着差タイム',
    '1走前_タイム指数',
    '近5走_上り3F_std',
    '良馬場_平均着順_近5走',
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
    kr = pd.to_numeric(df.get('クラス_rank', pd.Series(dtype=float)), errors='coerce')
    df = df[(df['surface'] == 'ダ') & (dm <= 1400) & (kr != 1.0)].copy()
    df['dist_m'] = dm[df.index]
    df = add_computed_features(df)
    baba_map = {'良': 0, '稍重': 1, '重': 2, '不良': 3}
    for col in df.columns:
        if '馬場状態' in col:
            df[col] = df[col].map(baba_map)
    all_feats = BASE_NV2 + CANDIDATES
    for col in all_feats:
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


def comb_roi(r_a, n_a, r_b, n_b):
    if n_a + n_b == 0:
        return 0.0
    return (r_a * n_a + r_b * n_b) / (n_a + n_b)


def tie_rate(scored_df):
    s = scored_df.copy()
    s['_rm'] = s.groupby('race_id')['prob'].rank(ascending=False, method='min')
    ties = s[s['_rm'] == 1].groupby('race_id').size()
    return (ties > 1).mean() if len(ties) > 0 else 0.0


def evaluate_set(df_trn, df_val, oos_2324, oos_2025, oos_2026, feats):
    valid = [c for c in feats if c in df_trn.columns and df_trn[c].isna().mean() <= 0.65]
    if not valid:
        return float('nan'), float('nan'), float('nan'), float('nan'), valid, None, float('nan')
    X_tr, y_tr, gs_tr, n_tr, nr_tr, scaler, *_ = prepare(
        df_trn, valid, top_idx=None, top_idx3=None, fit=True)
    X_va, y_va, gs_va, n_va, nr_va, *_ = prepare(
        df_val, valid, scaler=scaler, top_idx=None, top_idx3=None)
    beta = adam_fit(X_tr, y_tr, gs_tr, n_tr, nr_tr, X_va, y_va, gs_va, n_va, nr_va)

    r2324 = r25 = r26 = float('nan')
    n2324 = n25 = n26 = 0
    tie26 = float('nan')

    for label, oos in [('2324', oos_2324), ('2025', oos_2025), ('2026', oos_2026)]:
        if len(oos) == 0:
            continue
        vp = [c for c in valid if c in oos.columns]
        X_p, _, gs_p, n_p, *_ = prepare(oos, vp, scaler=scaler, top_idx=None, top_idx3=None)
        scored = oos.sort_values('race_id').reset_index(drop=True)
        scored['prob'] = segment_softmax(X_p @ beta, gs_p, n_p)
        scored['rank'] = scored.groupby('race_id')['prob'].rank(ascending=False, method='first')
        top1 = scored[scored['rank'] == 1]
        r, n = roi_from_top1(top1)
        if label == '2324':
            r2324, n2324 = r, n
        elif label == '2025':
            r25, n25 = r, n
        else:
            r26, n26 = r, n
            tie26 = tie_rate(scored)

    c2526 = comb_roi(r25, n25, r26, n26)
    return r2324, c2526, r25, r26, valid, beta, tie26


def main():
    t0 = time.time()
    print("=" * 90)
    print("  ダ短距離 10特徴探索 — 選択指標=2323 OOS ROI")
    print(f"  ベースnv2(5特徴): {BASE_NV2}")
    print(f"  候補追加: {len(CANDIDATES)}個")
    print("  ※ 25+26は報告のみ（選択に使わない）")
    print("=" * 90)

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

    # ── Greedy forward selection ────────────────────────────────────────────
    selected = list(BASE_NV2)
    remaining = list(CANDIDATES)

    print(f"\n{'='*90}")
    print("  Greedy forward selection (選択指標=2323)")
    print(f"{'='*90}")

    # ベースラインのスコア
    r_base, c_base, r25_base, r26_base, valid_base, beta_base, tie_base = evaluate_set(
        df_trn, df_val, oos_2324, oos_2025, oos_2026, selected)
    print(f"\n[初期ベース({len(valid_base)}特徴)]")
    print(f"  2323={r_base*100:+.2f}%  25+26={c_base*100:+.2f}%  "
          f"(2025={r25_base*100:+.2f}% / 2026={r26_base*100:+.2f}%)  tie={tie_base*100:.1f}%")

    round_num = 0
    while len(selected) < 10 and remaining:
        round_num += 1
        best_r2324, best_feat, best_comb_val = r_base, None, c_base
        print(f"\n  [Round {round_num}: 現在{len(selected)}特徴 → 目標10特徴]")
        print(f"  {'追加候補':30s}  {'2323':>8}  {'25+26':>8}  {'タイ':>6}  {'特徴数':>4}")
        print(f"  {'-'*70}")

        round_results = []
        for feat in remaining:
            trial = selected + [feat]
            r2324, c2526, r25, r26, valid, beta, tie26 = evaluate_set(
                df_trn, df_val, oos_2324, oos_2025, oos_2026, trial)
            round_results.append((feat, r2324, c2526, r25, r26, valid, beta, tie26))

        # 2323で降順ソートして表示
        round_results.sort(key=lambda x: -x[1] if not np.isnan(x[1]) else -999)
        for feat, r2324, c2526, r25, r26, valid, beta, tie26 in round_results:
            marker = ' ←BEST' if feat == round_results[0][0] else ''
            tie_str = f'{tie26*100:5.1f}%' if not np.isnan(tie26) else '  N/A'
            print(f"  +{feat:29s}  {r2324*100:+7.2f}%  {c2526*100:+7.2f}%  "
                  f"{tie_str}  {len(valid):3d}個{marker}")

        # 最良を選択（2323で判断）
        best_feat, best_r2324, best_c2526, best_r25, best_r26, best_valid, best_beta, best_tie = round_results[0]

        if not np.isnan(best_r2324) and best_r2324 > r_base:
            selected.append(best_feat)
            remaining.remove(best_feat)
            r_base = best_r2324
            c_base = best_c2526
            print(f"\n  ✓ 採用: +{best_feat}  "
                  f"2323={best_r2324*100:+.2f}% (+{(best_r2324-r_base+best_r2324-r_base)*0.5*100:.2f}pp相当)  "
                  f"25+26={best_c2526*100:+.2f}%")
            print(f"  現在の特徴量({len(selected)}個): {selected}")
            # β表示
            for f, b in zip(best_valid, best_beta):
                print(f"      β {f}: {b:+.4f}")
        else:
            print(f"\n  ✗ 改善なし。残り候補でROI改善なし")
            # 改善なくても10特徴未満なら2323最良を強制採用
            if len(selected) < 10 and not np.isnan(best_r2324):
                print(f"  → 10特徴目標のため強制採用: +{best_feat}  2323={best_r2324*100:+.2f}%")
                selected.append(best_feat)
                remaining.remove(best_feat)
                r_base = best_r2324
                c_base = best_c2526
                print(f"  現在の特徴量({len(selected)}個): {selected}")
            else:
                break

    # ── 最終結果 ────────────────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print(f"  最終特徴量({len(selected)}個):")
    for f in selected:
        print(f"    - {f}")

    r2324_f, c2526_f, r25_f, r26_f, valid_f, beta_f, tie_f = evaluate_set(
        df_trn, df_val, oos_2324, oos_2025, oos_2026, selected)
    print(f"\n  最終OOS:")
    print(f"    2323:  {r2324_f*100:+.2f}%")
    print(f"    2025:  {r25_f*100:+.2f}%")
    print(f"    2026:  {r26_f*100:+.2f}%")
    print(f"    25+26: {c2526_f*100:+.2f}%")
    print(f"    タイ率: {tie_f*100:.1f}%")
    print(f"  β係数:")
    for f, b in zip(valid_f, beta_f):
        print(f"    {f}: {b:+.4f}")
    print(f"\n  総時間: {int(time.time()-t0)}s")


if __name__ == '__main__':
    main()
