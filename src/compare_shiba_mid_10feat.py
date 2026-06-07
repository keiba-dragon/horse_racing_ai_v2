# coding: utf-8
"""
compare_shiba_mid_10feat.py - 芝中距離 10特徴探索
  選択指標: 2323 OOS ROI  (25+26は報告のみ)
  現AI(5特徴)をベースに greedy + pair search で 10 特徴を目指す
  セグメント: 芝 & 1401m-2000m
"""
import sys, os, time, itertools
import numpy as np
import pandas as pd

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from save_conditional_logit import prepare, segment_softmax, BASE_DIR, DATA_FILE, LR, N_EPOCHS, PATIENCE
from save_v3 import add_computed_features

L2 = 0.006

BASE_AI = ['調教師コース_r100_勝率', '馬距離_勝率', '1走前_クラス調整着順', '近5走_クラス調整_平均着順', '間隔']

CANDIDATES = [
    '性別_num',
    '斤量',
    '馬番',
    '馬体重',
    '1走前_クラス差',
    '2走前_クラス差',
    '3走前_クラス差',
    '1走前_脚質_num',
    '1走前_3角',
    '距離変化_前走',
    '近5走_上り3F平均',
    'コース脚質_r200_勝率',
    'コース枠_r200_勝率',
    '同会場_平均着順_近5走',
    '良馬場_平均着順_近5走',
    '前走着差タイム',
    '1走前_タイム指数',
    '近5走_上り3F_std',
    '芝ダ転向',
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
    df = df[(df['surface'] == '芝') & (dm >= 1401) & (dm <= 2000)].copy()
    df['dist_m'] = dm[df.index]
    df = add_computed_features(df)
    baba_map = {'良': 0, '稍重': 1, '重': 2, '不良': 3}
    for col in df.columns:
        if '馬場状態' in col:
            df[col] = df[col].map(baba_map)
    for col in BASE_AI + CANDIDATES:
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


def eval_feats(df_trn, df_val, oos_2324, oos_2025, oos_2026, feats):
    valid = [c for c in feats if c in df_trn.columns and df_trn[c].isna().mean() <= 0.65]
    if not valid:
        return float('nan'), float('nan'), float('nan'), float('nan'), valid, None
    X_tr, y_tr, gs_tr, n_tr, nr_tr, scaler, *_ = prepare(
        df_trn, valid, top_idx=None, top_idx3=None, fit=True)
    X_va, y_va, gs_va, n_va, nr_va, *_ = prepare(
        df_val, valid, scaler=scaler, top_idx=None, top_idx3=None)
    beta = adam_fit(X_tr, y_tr, gs_tr, n_tr, nr_tr, X_va, y_va, gs_va, n_va, nr_va)
    r2324 = r25 = r26 = float('nan')
    n2324 = n25 = n26 = 0
    for label, oos in [('2324', oos_2324), ('2025', oos_2025), ('2026', oos_2026)]:
        if len(oos) == 0:
            continue
        vp = [c for c in valid if c in oos.columns]
        X_p, _, gs_p, n_p, *_ = prepare(oos, vp, scaler=scaler, top_idx=None, top_idx3=None)
        scored = oos.sort_values('race_id').reset_index(drop=True)
        scored['prob'] = segment_softmax(X_p @ beta, gs_p, n_p)
        scored['rank'] = scored.groupby('race_id')['prob'].rank(ascending=False, method='first')
        top1 = scored[scored['rank'] == 1]
        won = top1['着順_num'] == 1
        odds = pd.to_numeric(top1['単勝オッズ'], errors='coerce')
        r = (odds[won] * 100).sum() / (len(top1) * 100) - 1 if len(top1) > 0 else float('nan')
        n = len(top1)
        if label == '2324':
            r2324, n2324 = r, n
        elif label == '2025':
            r25, n25 = r, n
        else:
            r26, n26 = r, n
    c2526 = (r25*n25 + r26*n26) / (n25+n26) if n25+n26 > 0 else float('nan')
    return r2324, c2526, r25, r26, valid, beta


def main():
    t0 = time.time()
    print("=" * 90)
    print("  芝中距離 10特徴探索 — 選択指標=2323 OOS ROI")
    print(f"  ベースAI(5特徴): {BASE_AI}")
    print("  Step1: Greedy forward selection  Step2: Pair search for 残り枠")
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

    # ── Step 1: Greedy ────────────────────────────────────────────────────
    selected = list(BASE_AI)
    remaining = list(CANDIDATES)

    r_base, c_base, r25_base, r26_base, valid_base, _ = eval_feats(
        df_trn, df_val, oos_2324, oos_2025, oos_2026, selected)
    print(f"\n[ベース({len(valid_base)}特徴)] 2323={r_base*100:+.2f}%  25+26={c_base*100:+.2f}%")

    print(f"\n{'='*90}")
    print("  Greedy forward selection (選択指標=2323)")
    print(f"{'='*90}")

    while len(selected) < 10 and remaining:
        print(f"\n  [Round: 現在{len(selected)}特徴 → 目標10特徴]")
        print(f"  {'追加候補':30s}  {'2323':>8}  {'25+26':>8}  特徴数")
        print(f"  {'-'*60}")
        round_res = []
        for feat in remaining:
            r2324, c2526, r25, r26, valid, beta = eval_feats(
                df_trn, df_val, oos_2324, oos_2025, oos_2026, selected + [feat])
            round_res.append((feat, r2324, c2526, r25, r26, valid, beta))
        round_res.sort(key=lambda x: -x[1] if not np.isnan(x[1]) else -999)
        for feat, r2324, c2526, r25, r26, valid, beta in round_res:
            marker = ' ←BEST' if feat == round_res[0][0] else ''
            print(f"  +{feat:29s}  {r2324*100:+7.2f}%  {c2526*100:+7.2f}%  {len(valid)}個{marker}")

        best = round_res[0]
        if not np.isnan(best[1]) and best[1] > r_base:
            selected.append(best[0])
            remaining.remove(best[0])
            r_base = best[1]
            c_base = best[2]
            print(f"\n  ✓ 採用: +{best[0]}  2323={best[1]*100:+.2f}%  25+26={best[2]*100:+.2f}%")
            for f, b in zip(best[5], best[6]):
                print(f"      β {f}: {b:+.4f}")
        else:
            print(f"\n  ✗ greedy改善なし。残り枠{10-len(selected)}個をペア検索へ")
            break

    # ── Step 2: Pair search for remaining slots ───────────────────────────
    slots_needed = 10 - len(selected)
    if slots_needed > 0 and len(remaining) >= slots_needed:
        print(f"\n{'='*90}")
        print(f"  Pair search: 残り{len(remaining)}候補から{slots_needed}個選択 (2323選択)")
        print(f"{'='*90}")

        if slots_needed == 1:
            combos = [(r,) for r in remaining]
        elif slots_needed == 2:
            combos = list(itertools.combinations(remaining, 2))
        else:
            combos = list(itertools.combinations(remaining, min(slots_needed, 3)))

        pair_res = []
        print(f"\n  {'組み合わせ':50s}  {'2323':>8}  {'25+26':>8}")
        print(f"  {'-'*72}")
        for combo in combos:
            trial = selected + list(combo)
            r2324, c2526, r25, r26, valid, beta = eval_feats(
                df_trn, df_val, oos_2324, oos_2025, oos_2026, trial)
            label = '+'.join(combo)
            print(f"  +{label[:49]:49s}  {r2324*100:+7.2f}%  {c2526*100:+7.2f}%")
            pair_res.append((r2324, c2526, list(combo), valid, beta))

        pair_res.sort(key=lambda x: -x[0] if not np.isnan(x[0]) else -999)
        best_combo = pair_res[0]
        selected = selected + best_combo[2]
        print(f"\n  ✓ 採用: {best_combo[2]}  2323={best_combo[0]*100:+.2f}%  25+26={best_combo[1]*100:+.2f}%")

    # ── 最終評価 ──────────────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print(f"  最終特徴量({len(selected)}個): {selected}")
    r2324_f, c2526_f, r25_f, r26_f, valid_f, beta_f = eval_feats(
        df_trn, df_val, oos_2324, oos_2025, oos_2026, selected)
    print(f"\n  2323={r2324_f*100:+.2f}%  2025={r25_f*100:+.2f}%  2026={r26_f*100:+.2f}%  25+26={c2526_f*100:+.2f}%")
    print("  β係数:")
    for f, b in zip(valid_f, beta_f):
        print(f"    {f}: {b:+.4f}")
    print(f"  総時間: {int(time.time()-t0)}s")


if __name__ == '__main__':
    main()
