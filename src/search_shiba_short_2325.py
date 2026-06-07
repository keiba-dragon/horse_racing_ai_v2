# coding: utf-8
"""
search_shiba_short_2325.py - 芝短距離 特徴量探索 (2325選択指標)

v2.0: NaN修正後の再探索。選択指標を2023-24 → 2023-25合算に変更。
  - supplementデータ(2025-05-17+)が実値を持つようになったため
    2323だけでは2026分布を代表しない
  - 選択指標: r2325 = (r2323×n2323 + r2025×n2025) / (n2323+n2025)
  - フレッシュスタート（固定ベースなし）
  - 最大10特徴
  セグメント: 芝 ≤1400m
"""
import sys, os, time, itertools
import numpy as np
import pandas as pd

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from save_conditional_logit import prepare, segment_softmax, BASE_DIR, DATA_FILE, LR, N_EPOCHS, PATIENCE
from save_v3 import add_computed_features

L2 = 0.006
MAX_FEATS = 10
ROI_CAP = 0.30   # 単年ROIをこの値でキャップ（外れ値年が支配するのを防ぐ）

# 常にベースに含める（タイ防止・必ず全馬に値あり）
FORCED_BASE = ['馬番', '斤量']

CANDIDATES = [
    # 前走ポジション・脚質
    '1走前_3角', '1走前_4角', '1走前_脚質_num',
    # コース適性
    '芝ダ転向', '距離変化_前走', '馬距離_勝率',
    # タイム系
    '前走着差タイム', '近5走_上り3F平均', '近5走_上り3F_std',
    '1走前_タイム指数', '近5走_タイム指数平均',
    # 馬体・装備
    '馬体重', '斤量', 'ブリンカー変更',
    # 馬番・枠
    '馬番', 'コース枠_r200_勝率',
    # クラス
    '1走前_クラス差', '2走前_クラス差', '1走前_クラス調整着順',
    '近5走_クラス調整_平均着順',
    # 馬場
    '1走前_馬場状態', '良馬場_平均着順_近5走', '道悪_平均着順_近5走',
    # 会場・騎手
    '同会場_平均着順_近5走', 'コース脚質_r200_勝率',
    '騎手コース_r100_勝率', '調教師コース_r100_勝率',
    # 間隔・その他
    '間隔', '性別_num', '騎手変更',
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
    df = df[(df['surface'] == '芝') & (dm <= 1400)].copy()
    df['dist_m'] = dm[df.index]
    df = add_computed_features(df)
    baba_map = {'良': 0, '稍重': 1, '重': 2, '不良': 3}
    for col in df.columns:
        if '馬場状態' in col and col != '馬場状態':
            df[col] = df[col].map(baba_map)
    for col in CANDIDATES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
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


NAN_IND_THRESHOLD = 0.05   # NaN率がこれ以上なら _isnan 指示変数を自動追加


def expand_with_nan_indicators(dfs, feats):
    """高NaN特徴量に isnan 指示変数を追加し、拡張された特徴量リストを返す。
    NaN率>threshold の特徴 f → (f, f_isnan) ペアで使う。
    実値列は fillna(0) 済みを前提。isnan 列は 0/1。
    """
    extended = []
    ref_df = dfs[0]
    for f in feats:
        extended.append(f)
        if f not in ref_df.columns:
            continue
        nan_rate = ref_df[f].isna().mean()
        if NAN_IND_THRESHOLD < nan_rate < 1.0:
            ind = f + '_isnan'
            for df in dfs:
                if f in df.columns and ind not in df.columns:
                    df[ind] = df[f].isna().astype(float)
            extended.append(ind)
    return extended


def eval_feats(df_trn, df_val, oos_2324, oos_2025, oos_2026, feats):
    all_dfs = [df_trn, df_val, oos_2324, oos_2025, oos_2026]
    # NaN指示変数を追加した拡張特徴量リスト
    expanded = expand_with_nan_indicators(all_dfs, feats)
    # NaN率100%（全欠損）の列だけ除外。それ以外はNaN率問わず使う
    valid = [c for c in expanded if c in df_trn.columns
             and df_trn[c].isna().mean() < 1.0
             and df_trn[c].std(ddof=0) > 0]
    if not valid:
        return float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), valid, None
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
        won  = top1['着順_num'] == 1
        odds = pd.to_numeric(top1['単勝オッズ'], errors='coerce')
        r = (odds[won] * 100).sum() / (len(top1) * 100) - 1 if len(top1) > 0 else float('nan')
        if label == '2324':
            r2324, n2324 = r, len(top1)
        elif label == '2025':
            r25, n25 = r, len(top1)
        else:
            r26, n26 = r, len(top1)

    # 選択指標: 2023-25合算。単年ROIをキャップして外れ値年の支配を防ぐ
    r2324c = np.clip(r2324, -ROI_CAP, ROI_CAP) if not np.isnan(r2324) else float('nan')
    r25c   = np.clip(r25,   -ROI_CAP, ROI_CAP) if not np.isnan(r25)   else float('nan')
    r2325  = (r2324c * n2324 + r25c * n25) / (n2324 + n25) if (n2324 + n25) > 0 else float('nan')
    c2526  = (r25 * n25 + r26 * n26) / (n25 + n26) if (n25 + n26) > 0 else float('nan')
    return r2325, r2324, c2526, r25, r26, valid, beta


def main():
    t0 = time.time()
    print("=" * 90)
    print("  芝短距離 特徴量探索 v2.0 — 選択指標=2325(2023-25合算) OOS ROI")
    print("  NaN修正後フレッシュスタート。2026は参考値のみ。")
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

    # FORCED_BASE はCANDIDATESから除外
    selected = list(FORCED_BASE)
    remaining = [c for c in CANDIDATES if c not in FORCED_BASE]
    best_r2325 = float('-inf')

    r_base, r2324_b, c_b, r25_b, r26_b, valid_b, _ = eval_feats(
        df_trn, df_val, oos_2324, oos_2025, oos_2026, selected)
    print(f"\n[強制ベース {FORCED_BASE}]  2325={r_base*100:+.2f}%  2323={r2324_b*100:+.2f}%  "
          f"2025={r25_b*100:+.2f}%  2026={r26_b*100:+.2f}%")
    best_r2325 = r_base if not np.isnan(r_base) else float('-inf')

    print("\n=== Greedy forward selection (2325キャップ選択) ===")
    while len(selected) < MAX_FEATS and remaining:
        round_res = []
        for feat in remaining:
            r2325, r2324, c2526, r25, r26, valid, beta = eval_feats(
                df_trn, df_val, oos_2324, oos_2025, oos_2026, selected + [feat])
            round_res.append((feat, r2325, r2324, c2526, r25, r26, valid, beta))
        round_res.sort(key=lambda x: -x[1] if not np.isnan(x[1]) else 999)

        print(f"\n[{len(selected)}特徴 → {MAX_FEATS}目標] 上位10候補:")
        print(f"  {'追加候補':30s}  {'2325(選択)':>10}  {'2323':>8}  {'2025':>8}  {'2026':>8}")
        for feat, r2325, r2324, c2526, r25, r26, *_ in round_res[:10]:
            marker = " ←BEST" if feat == round_res[0][0] else ""
            print(f"  +{feat:29s}  {r2325*100:+9.2f}%  {r2324*100:+7.2f}%  "
                  f"{r25*100:+7.2f}%  {r26*100:+7.2f}%{marker}")

        best = round_res[0]
        if not np.isnan(best[1]):
            improved = best[1] > best_r2325
            mark = "✓" if improved else "△"
            selected.append(best[0])
            remaining.remove(best[0])
            if improved:
                best_r2325 = best[1]
            print(f"\n  {mark} 採用: +{best[0]}  2325={best[1]*100:+.2f}%  "
                  f"2323={best[2]*100:+.2f}%  2025={best[4]*100:+.2f}%  2026={best[5]*100:+.2f}%")
        else:
            print(f"\n  ✗ 全候補NaN。探索終了。")
            break

    # 最終評価
    print(f"\n{'='*90}")
    print(f"  最終特徴量({len(selected)}個): {selected}")
    r2325_f, r2324_f, c2526_f, r25_f, r26_f, valid_f, beta_f = eval_feats(
        df_trn, df_val, oos_2324, oos_2025, oos_2026, selected)
    print(f"\n  2325(選択)={r2325_f*100:+.2f}%  2323={r2324_f*100:+.2f}%  "
          f"2025={r25_f*100:+.2f}%  2026={r26_f*100:+.2f}%  25+26={c2526_f*100:+.2f}%")
    print("  β係数:")
    if valid_f and beta_f is not None:
        for f, b in zip(valid_f, beta_f):
            print(f"    {f}: {b:+.4f}")
    print(f"  総時間: {int(time.time()-t0)}s")


if __name__ == '__main__':
    main()
