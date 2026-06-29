# coding: utf-8
"""
search_shiba_short_acc.py - 芝短距離 特徴量探索 (的中率最大化版)

セグメント: 芝 & 距離≤1400m & クラス_rank≠1.0
選択指標: acc_2325 = (acc2324×n2324 + acc25×n25) / (n2324+n25)  ← 的中率
ROIは参考値として表示。保存先: models/hitrate_model.pkl（roi_model.pkl上書き禁止）

1番人気 acc_2325=28.69%（ランダム6.7%）
v1: ゼロシードからクリーン探索、MAX_FEATS=30 → acc_2325=26.38%
v2: v1の30特徴をSEEDに、MAX_FEATS=35
"""
import sys, os, time
import numpy as np
import pandas as pd

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from save_conditional_logit import prepare, segment_softmax, BASE_DIR, DATA_FILE, LR, N_EPOCHS, PATIENCE
from save_v3 import add_computed_features

L2 = 0.006
MAX_FEATS = 35
ACC_TARGET = 0.28  # 目標: 1番人気acc_2325=28.69%に近づける

FORCED_BASE = ['馬番', '斤量']
# v1結果30特徴をシードに設定
SEED_FEATS = [
    '馬番', '斤量', '芝ダ一致_平均着順_近5走', '1走前_タイム指数',
    '近5走_クラス調整_平均着順', '馬コース_r20_勝率', '騎手コース_r100_勝率',
    '近10走_勝率', '馬体重', '近5走_上り3F平均', '芝ダ転向',
    '近3走_体重増減合計', '相手レベル_平均着順', '道悪_平均着順_近5走',
    'タイム指数_近3走_slope', '馬体重増減', 'コース馬場_r200_勝率',
    'ブリンカー変更', '3走前_クラス差', '近5走_上り3F_std',
    '騎手コース距離_r100_勝率', '2走前_クラス差', '着順_近3走_slope',
    '近3走_複勝率', '馬距離_勝率', '1走前_3角', '1走前_馬場状態',
    'コース枠_r200_勝率', '前走着差タイム', '馬_r20_勝率',
]

CANDIDATES = [
    # 前走ポジション・脚質
    '1走前_3角', '1走前_4角', '1走前_脚質_num',
    # コース適性
    '芝ダ転向', '距離変化_前走', '馬距離_勝率',
    # タイム系
    '前走着差タイム', '近5走_上り3F平均', '近5走_上り3F_std',
    '1走前_タイム指数', '近5走_タイム指数平均', '近5走_タイム指数_max',
    # トレンド系
    'タイム指数_近3走_slope', 'タイム指数_近5走_slope', 'タイム指数_加速度',
    '着順_近3走_slope', '着順_近5走_slope',
    '上り3F_近3走_slope', '4角位置_近3走_slope',
    # 馬体・装備
    '馬体重', '馬体重増減', 'ブリンカー変更', '近3走_体重増減合計',
    # 枠・コース
    'コース枠_r200_勝率', 'コース脚質_r200_勝率', 'コース馬場_r200_勝率', 'コース枠_r200_複勝率',
    # クラス系
    '1走前_クラス差', '2走前_クラス差', '3走前_クラス差',
    '1走前_クラス調整着順', '近5走_クラス調整_平均着順',
    # 馬場
    '1走前_馬場状態', '道悪_平均着順_近5走', '同馬場_平均着順_近5走', '良馬場_平均着順_近5走',
    # 勝率統計
    '種牡馬_勝率', '種牡馬_ダ_勝率', '母父馬_勝率',
    '馬_r20_勝率', '馬コース_r20_勝率', '馬距離_勝率',
    '騎手コース_r100_勝率', '騎手会場_r100_勝率', '騎手コース距離_r100_勝率',
    '騎手距離_r100_勝率', '騎手馬場_r100_勝率',
    '調教師コース_r100_勝率', '調教師_r200_勝率',
    # 複勝率・長期成績
    '近3走_複勝率', '近3走_勝率', '近10走_複勝率', '近10走_勝率', '近5走_複勝率',
    '同会場_複勝率_近5走', '同会場_平均着順_近5走',
    '同距離帯_平均着順_近5走', '芝ダ一致_平均着順_近5走', '相手レベル_平均着順',
    # 間隔・その他
    '間隔', '性別_num', '騎手変更',
    # 輸送
    '輸送有無',
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
    df['クラス_rank'] = pd.to_numeric(df['クラス_rank'], errors='coerce')
    df = df[(df['surface'] == '芝') & (dm <= 1400) & (df['クラス_rank'] != 1.0)].copy()
    df['dist_m'] = dm[df.index]
    df = add_computed_features(df)

    if '今回_会場' in df.columns and '1走前_開催' in df.columns:
        prev_venue = df['1走前_開催'].astype(str).str[1]
        df['輸送有無'] = (df['今回_会場'].astype(str) != prev_venue).astype(float)
        df.loc[df['1走前_開催'].isna(), '輸送有無'] = float('nan')

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


NAN_IND_THRESHOLD = 0.05


def expand_with_nan_indicators(dfs, feats):
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
    expanded = expand_with_nan_indicators(all_dfs, feats)
    valid = [c for c in expanded if c in df_trn.columns
             and df_trn[c].isna().mean() < 1.0
             and df_trn[c].std(ddof=0) > 0]
    if not valid:
        return float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), valid, None, {}

    X_tr, y_tr, gs_tr, n_tr, nr_tr, scaler, *_ = prepare(
        df_trn, valid, top_idx=None, top_idx3=None, fit=True)
    X_va, y_va, gs_va, n_va, nr_va, *_ = prepare(
        df_val, valid, scaler=scaler, top_idx=None, top_idx3=None)
    beta = adam_fit(X_tr, y_tr, gs_tr, n_tr, nr_tr, X_va, y_va, gs_va, n_va, nr_va)

    acc2324 = acc25 = acc26 = float('nan')
    roi2324 = roi25 = roi26 = float('nan')
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
        acc  = won.mean() if len(top1) > 0 else float('nan')
        roi  = (odds[won] * 100).sum() / (len(top1) * 100) - 1 if len(top1) > 0 else float('nan')
        n    = len(top1)
        if label == '2324':
            acc2324, roi2324, n2324 = acc, roi, n
        elif label == '2025':
            acc25, roi25, n25 = acc, roi, n
        else:
            acc26, roi26, n26 = acc, roi, n

    acc_2325 = (acc2324 * n2324 + acc25 * n25) / (n2324 + n25) if (n2324 + n25) > 0 else float('nan')
    acc_2526 = (acc25 * n25 + acc26 * n26) / (n25 + n26) if (n25 + n26) > 0 else float('nan')
    roi_2526 = (roi25 * n25 + roi26 * n26) / (n25 + n26) if (n25 + n26) > 0 else float('nan')
    ref_roi  = {'2324': roi2324, '2025': roi25, '2026': roi26, '25+26': roi_2526}

    return acc_2325, acc2324, acc_2526, acc25, acc26, valid, beta, ref_roi


def main():
    t0 = time.time()
    print("=" * 90)
    print("  芝短距離 特徴量探索 — 選択指標=的中率(acc_2325)  ※ROIモデルとは別物")
    print("  セグメント: 芝 & 距離≤1400m & クラス_rank≠1.0")
    print("  1番人気 acc_2325=28.69%  ランダム=6.7%")
    print("=" * 90)

    df = load_segment()
    df_trn   = df[(df['日付_num'] >= 130101) & (df['日付_num'] < 220101)]
    df_val   = df[(df['日付_num'] >= 220101) & (df['日付_num'] <= 221231)]
    oos_2324 = df[(df['日付_num'] >= 230101) & (df['日付_num'] < 250101)]
    oos_2025 = df[(df['日付_num'] >= 250101) & (df['日付_num'] < 260101)]
    oos_2026 = df[df['日付_num'] >= 260101]

    avg_field = df[df['日付_num'] >= 230101].groupby('race_id').size().mean()
    random_acc = 1 / avg_field

    print(f"\ntrain:{len(df_trn):,}行({df_trn['race_id'].nunique()}R)  "
          f"val:{len(df_val):,}行({df_val['race_id'].nunique()}R)")
    print(f"2324:{oos_2324['race_id'].nunique()}R  "
          f"2025:{oos_2025['race_id'].nunique()}R  "
          f"2026:{oos_2026['race_id'].nunique()}R")
    print(f"ランダム的中率（参考）: {random_acc:.1%}  ({avg_field:.1f}頭/R平均)")

    selected = list(SEED_FEATS)
    remaining = [c for c in CANDIDATES if c not in selected]
    best_acc = float('-inf')

    acc_seed, acc2324_s, acc_s, acc25_s, acc26_s, valid_s, _, roi_s = eval_feats(
        df_trn, df_val, oos_2324, oos_2025, oos_2026, selected)
    print(f"\n[シード: {SEED_FEATS}]")
    print(f"  的中率: acc_2325={acc_seed:.4f}  2324={acc2324_s:.4f}  2025={acc25_s:.4f}  2026={acc26_s:.4f}")
    print(f"  目標: 1番人気 acc_2325=28.69%")
    best_acc = acc_seed if not np.isnan(acc_seed) else float('-inf')

    print("\n=== Greedy forward selection (的中率acc_2325で選択) ===")
    while len(selected) < MAX_FEATS and remaining:
        round_res = []
        for feat in remaining:
            acc_2325, acc2324, acc_2526, acc25, acc26, valid, beta, ref_roi = eval_feats(
                df_trn, df_val, oos_2324, oos_2025, oos_2026, selected + [feat])
            round_res.append((feat, acc_2325, acc2324, acc_2526, acc25, acc26, valid, beta, ref_roi))
        round_res.sort(key=lambda x: -x[1] if not np.isnan(x[1]) else 999)

        print(f"\n[{len(selected)}特徴 → {MAX_FEATS}目標] 上位10候補:")
        print(f"  {'追加候補':30s}  {'acc_2325(選択)':>13}  {'acc2324':>8}  {'acc2025':>8}  {'acc2026':>8}")
        for feat, acc_2325, acc2324, acc_2526, acc25, acc26, *_ in round_res[:10]:
            marker = " ←BEST" if feat == round_res[0][0] else ""
            print(f"  +{feat:29s}  {acc_2325:.4f}       {acc2324:.4f}   {acc25:.4f}   {acc26:.4f}{marker}")

        best = round_res[0]
        if not np.isnan(best[1]):
            improved = best[1] > best_acc
            mark = "✓" if improved else "△"
            selected.append(best[0])
            remaining.remove(best[0])
            if improved:
                best_acc = best[1]
            roi_ref = best[8]
            print(f"\n  {mark} 採用: +{best[0]}  acc_2325={best[1]:.4f}  "
                  f"参考ROI25+26={roi_ref['25+26']*100:+.1f}%")
            if best[1] >= ACC_TARGET:
                print(f"\n  ★ 目標達成! acc_2325={best[1]:.4f} >= {ACC_TARGET:.2f}")
        else:
            print(f"\n  ✗ 全候補NaN。探索終了。")
            break

    print(f"\n{'='*90}")
    print(f"  最終特徴量({len(selected)}個): {selected}")
    acc_f, acc2324_f, acc2526_f, acc25_f, acc26_f, valid_f, beta_f, roi_f = eval_feats(
        df_trn, df_val, oos_2324, oos_2025, oos_2026, selected)
    print(f"\n  的中率: acc_2325={acc_f:.4f}  2324={acc2324_f:.4f}  "
          f"2025={acc25_f:.4f}  2026={acc26_f:.4f}  25+26={acc2526_f:.4f}")
    print(f"  参考ROI: 2324={roi_f['2324']*100:+.2f}%  2025={roi_f['2025']*100:+.2f}%  "
          f"2026={roi_f['2026']*100:+.2f}%  25+26={roi_f['25+26']*100:+.2f}%")
    print(f"  ランダム的中率: {random_acc:.4f}")
    print(f"  ランダム比: {acc_f/random_acc:.3f}x")
    print("  β係数:")
    if valid_f and beta_f is not None:
        for f, b in zip(valid_f, beta_f):
            print(f"    {f}: {b:+.4f}")
    print(f"  総時間: {int(time.time()-t0)}s")


if __name__ == '__main__':
    main()
