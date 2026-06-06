# coding: utf-8
"""
fresh_shiba_short.py - 芝短距離（芝≤1400m）Forward Greedy
* 選択基準: Δ2324 > +0.5pp のみ
* 2025/26 は最後にブラインド評価
* 空集合スタート → 芝短専用特徴量セットを構築
* 新馬込み（クラス_rank除外なし）
* ベースライン: -26.54% (旧芝artifact, 新馬除外)
"""
import sys, os, time
import numpy as np
import pandas as pd

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from save_conditional_logit import (
    prepare, segment_softmax, BASE_DIR, DATA_FILE, LR, N_EPOCHS, PATIENCE
)
from save_v3 import add_computed_features

L2 = 0.006

ALL_CANDIDATES = [
    # 近走成績
    '近5走_クラス調整_平均着順', '1走前_クラス調整着順',
    '近3走_複勝率', '近5走_複勝率', '近3走_勝率', '近10走_勝率', '近10走_複勝率',
    '近走連続入着数', '近走_改善トレンド',

    # スピード・タイム指数
    '近5走_タイム指数_max', '近5走_タイム指数平均', '近5走_タイム指数_min',
    '1走前_タイム指数', '2走前_タイム指数', '3走前_タイム指数',
    'タイム指数_近5走_slope', 'タイム指数_近3走_slope',
    '近5走_上り3F平均', '上り3F_近3走_slope', '1走前_上り3F',
    '1走前_RPCI', '前走着差タイム',

    # クラス差（全走数）※クラス_rankは全馬同値なので除外
    '1走前_クラス差', '2走前_クラス差', '3走前_クラス差', '4走前_クラス差',
    '格上経験数_近5走',

    # コース・距離・枠適性
    'コース枠_r200_勝率', 'コース枠_r200_複勝率',
    'コース脚質_r200_勝率', 'コース脚質_r200_複勝率',
    # コース馬場_r200_勝率/複勝率 は全馬同値（レース単位）→除外
    '馬距離_勝率', '馬距離_複勝率',
    '馬コース_r20_勝率', '馬コース_r20_複勝率',

    # 同条件実績
    '同距離帯_平均着順_近5走', '同会場_平均着順_近5走', '同会場_複勝率_近5走',
    '道悪_平均着順_近5走', '良馬場_平均着順_近5走',

    # 騎手・調教師
    '騎手コース_r100_勝率', '騎手距離_r100_勝率',
    '騎手脚質_r100_勝率', '騎手調教師_r100_勝率', '騎手馬場_r100_勝率',
    '調教師_r200_複勝率', '調教師コース_r100_勝率',

    # 脚質・展開（芝は差しが有利なことも多い）
    '1走前_脚質_num', 'レース内_相対脚質', '展開フィット_v2',
    '1走前_上3F地点差',
    '1走前_3角', '1走前_4角', '前走_4角位置', '4角位置_近3走_slope',

    # 馬場状態（前走・今走）
    '1走前_馬場状態',

    # 馬体重
    '馬体重', '馬体重増減', '馬体重トレンド_近5走',

    # キャリア・年齢・性別
    'キャリア_浅い', '年齢', 'キャリア', 'キャリア_log', '性別_num',

    # レース条件（頭数・クラス_rankは全馬同値→除外）
    '馬番', '斤量', '斤量変化', '内外枠', '所属_num',
    '間隔', '間隔_長_flag', '間隔_短_flag',

    # 距離・路線変化
    '距離変化_前走', '芝ダ転向',

    # 血統（芝限定）
    '種牡馬_芝_勝率', '種牡馬_芝_複勝率',
    '種牡馬_勝率', '母父馬_勝率',

    # その他
    '乗替り_近走不振', '騎手変更',
    '近5走_着差タイム_クラス補正平均', '近5走_平均相対着順',
    '近5走_クラス補正スコア',
]
seen = set()
ALL_CANDIDATES = [c for c in ALL_CANDIDATES if not (c in seen or seen.add(c))]


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
    for col in ALL_CANDIDATES:
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


def evaluate_2324(df_trn, df_val, oos_2324, feats):
    valid = [c for c in feats if c in df_trn.columns and df_trn[c].isna().mean() <= 0.65]
    if not valid:
        return None, valid, None
    X_tr, y_tr, gs_tr, n_tr, nr_tr, scaler, *_ = prepare(
        df_trn, valid, top_idx=None, top_idx3=None, fit=True)
    X_va, y_va, gs_va, n_va, nr_va, *_ = prepare(
        df_val, valid, scaler=scaler, top_idx=None, top_idx3=None)
    beta = adam_fit(X_tr, y_tr, gs_tr, n_tr, nr_tr,
                    X_va, y_va, gs_va, n_va, nr_va)
    valid_p = [c for c in valid if c in oos_2324.columns]
    X_p, _, gs_p, n_p, *_ = prepare(oos_2324, valid_p, scaler=scaler,
                                      top_idx=None, top_idx3=None)
    scored = oos_2324.sort_values('race_id').reset_index(drop=True)
    scored['prob'] = segment_softmax(X_p @ beta, gs_p, n_p)
    scored['rank'] = scored.groupby('race_id')['prob'].rank(ascending=False, method='first')
    top1 = scored[scored['rank'] == 1]
    r2324, n2324 = roi_from_top1(top1)
    return r2324, valid, beta


def evaluate_blind(df_trn, df_val, oos_2025, oos_2026, feats):
    valid = [c for c in feats if c in df_trn.columns and df_trn[c].isna().mean() <= 0.65]
    if not valid:
        return None, None, 0, 0
    X_tr, y_tr, gs_tr, n_tr, nr_tr, scaler, *_ = prepare(
        df_trn, valid, top_idx=None, top_idx3=None, fit=True)
    X_va, y_va, gs_va, n_va, nr_va, *_ = prepare(
        df_val, valid, scaler=scaler, top_idx=None, top_idx3=None)
    beta = adam_fit(X_tr, y_tr, gs_tr, n_tr, nr_tr,
                    X_va, y_va, gs_va, n_va, nr_va)
    results = []
    for oos in [oos_2025, oos_2026]:
        if len(oos) == 0:
            results.append((float('nan'), 0))
            continue
        valid_p = [c for c in valid if c in oos.columns]
        X_p, _, gs_p, n_p, *_ = prepare(oos, valid_p, scaler=scaler,
                                          top_idx=None, top_idx3=None)
        scored = oos.sort_values('race_id').reset_index(drop=True)
        scored['prob'] = segment_softmax(X_p @ beta, gs_p, n_p)
        scored['rank'] = scored.groupby('race_id')['prob'].rank(ascending=False, method='first')
        top1 = scored[scored['rank'] == 1]
        results.append(roi_from_top1(top1))
    (r25, n25), (r26, n26) = results
    return r25, r26, n25, n26


def main():
    t0 = time.time()
    print("=" * 65)
    print("  芝短距離（芝≤1400m）Forward Greedy")
    print("  選択基準: Δ2324 > +0.5pp のみ")
    print("  2025/26 は最後にブラインド評価")
    print("  ベースライン: -26.54% (旧芝artifact)")
    print("=" * 65)
    print()

    df = load_segment()
    df_trn   = df[(df['日付_num'] >= 130101) & (df['日付_num'] < 220101)]
    df_val   = df[(df['日付_num'] >= 220101) & (df['日付_num'] <= 221231)]
    oos_2324 = df[(df['日付_num'] >= 230101) & (df['日付_num'] < 250101)]
    oos_2025 = df[(df['日付_num'] >= 250101) & (df['日付_num'] < 260101)]
    oos_2026 = df[df['日付_num'] >= 260101]
    print(f"train:{len(df_trn):,}  val:{len(df_val):,}  2324:{len(oos_2324):,}")
    print(f"[ブラインド] 2025:{len(oos_2025):,}  2026:{len(oos_2026):,}")
    print()

    avail_raw = [c for c in ALL_CANDIDATES if c in df_trn.columns and
                 df_trn[c].notna().sum() > 1000]

    # レース内で実際に馬ごとに異なる特徴量のみ残す
    # （レース単位で全馬同値 → clogitで無効 → data-order artifactを防ぐ）
    df_all = pd.concat([df_trn, df_val, oos_2324])
    avail = []
    for c in avail_raw:
        col = pd.to_numeric(df_all[c], errors='coerce')
        race_std = df_all.assign(_c=col).groupby('race_id')['_c'].std()
        frac_nonzero = (race_std > 0).sum() / max(len(race_std), 1)
        if frac_nonzero >= 0.5:  # 50%以上のレースでレース内分散あり
            avail.append(c)
        else:
            print(f"  [除外] {c}: レース内分散ありレース={frac_nonzero*100:.1f}%")

    print(f"\n候補特徴量: {len(avail)}個 (レース内分散フィルタ後)\n")

    # ─── Round 1: 最良シングルトン ────────────────────────────────
    print("─" * 65)
    print("  Round 1: シングルトン評価 (2324 ROI最大を採用)")
    print("─" * 65)
    best_feat, best_r2324 = None, -999.0
    for cand in avail:
        r, valid_t, _ = evaluate_2324(df_trn, df_val, oos_2324, [cand])
        if r is None or cand not in valid_t or np.isnan(r):
            continue
        print(f"  [{cand}] 2324:{r*100:.2f}%")
        if r > best_r2324:
            best_r2324, best_feat = r, cand

    if best_feat is None:
        print("候補なし。終了。")
        return

    print(f"\n  → 採用: {best_feat}  2324:{best_r2324*100:.2f}%\n")
    current   = [best_feat]
    remaining = [c for c in avail if c not in current]

    # ─── Round 2+: Forward Greedy ────────────────────────────────
    print("─" * 65)
    print("  Forward Greedy (採用基準: Δ2324 > +0.5pp のみ)")
    print("─" * 65)

    round_num = 2
    while remaining:
        print(f"\n--- Round {round_num}: {len(current)}特徴, 2324:{best_r2324*100:.2f}% ---")
        best_delta, best_cand_this, best_valid_this = -999.0, None, None

        for cand in remaining:
            r, valid_t, _ = evaluate_2324(df_trn, df_val, oos_2324, current + [cand])
            if r is None or np.isnan(r):
                continue
            actually_added = cand in valid_t
            delta = r - best_r2324
            good  = delta > 0.005 and actually_added
            sym   = '✓' if good else '✗'
            note  = '' if actually_added else ' [NaN>65%]'
            print(f"  {sym} +{cand}{note}  2324:{r*100:.2f}%(Δ{delta*100:+.2f}%)")
            if good and delta > best_delta:
                best_delta, best_cand_this, best_r2324_new, best_valid_this = delta, cand, r, valid_t

        if best_cand_this is None:
            print("  → 採用なし。終了。")
            break

        current   = list(best_valid_this)
        best_r2324 = best_r2324_new
        remaining  = [c for c in remaining if c not in current]
        print(f"\n  → 採用: {best_cand_this}  ({len(current)}特徴)  2324:{best_r2324*100:.2f}%")
        round_num += 1

    # ─── 最終報告 ─────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print(f"総時間: {int(time.time()-t0)}s")
    print(f"\n最終特徴量 ({len(current)}個):")
    for f in current:
        print(f"  {f}")

    r2324_final, _, _ = evaluate_2324(df_trn, df_val, oos_2324, current)
    print(f"\n  2324 (選択基準): {r2324_final*100:.2f}%")

    print(f"\n{'─' * 65}")
    print("  ブラインド評価 (2025/2026) ← 選択中は一度も見ていない")
    print(f"{'─' * 65}")
    r25, r26, n25, n26 = evaluate_blind(df_trn, df_val, oos_2025, oos_2026, current)
    rcomb = comb2526(r25, n25, r26, n26)
    print(f"  2025: {r25*100:.2f}%  ({n25}R)")
    print(f"  2026: {r26*100:.2f}%  ({n26}R)")
    print(f"  25+26: {rcomb*100:.2f}%")
    print(f"\n  ベースライン比 (25+26): {rcomb*100:.2f}% vs -26.54%  (Δ{(rcomb+0.2654)*100:+.2f}pp)")


if __name__ == '__main__':
    main()
