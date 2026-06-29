# coding: utf-8
"""
search_all_acc_v2.py - 全5セグメント 的中率最大化 統合探索 v2

目標: 全セグメントで1番人気的中率を超える
新追加候補: 1走前_単勝オッズ, PCI系, 個別過去走タイム指数, 前走上り3F等

現在の各セグメント acc_2325 vs 1番人気:
  ダ長  30.02% vs 34.03%  (gap -4.01pp)
  ダ短  30.79% vs 34.90%  (gap -4.11pp)
  芝短  26.38% vs 28.69%  (gap -2.31pp)
  芝中  30.25% vs 33.21%  (gap -2.96pp)
  芝長  34.75% vs 36.05%  (gap -1.30pp)
"""
import sys, os, time, pickle
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from save_conditional_logit import prepare, segment_softmax, BASE_DIR, DATA_FILE, LR, N_EPOCHS, PATIENCE
from save_v3 import add_computed_features

L2 = 0.006
NAN_IND_THRESHOLD = 0.05
MAX_FEATS = 50

# ─── 各セグメントの現在のSEED（hitrate_model.pkl の特徴量） ───
SEEDS = {
    'ダ長': [
        '馬番', '斤量', '芝ダ一致_平均着順_近5走', '1走前_タイム指数',
        '近5走_クラス調整_平均着順', '輸送有無', '馬コース_r20_勝率',
        '近3走_体重増減合計', '性別_num', '1走前_クラス差', 'コース枠_r200_複勝率',
        'コース枠_r200_勝率', '3走前_クラス差', '近10走_勝率', '調教師_r200_勝率',
        'ブリンカー変更', '2走前_クラス差', '相手レベル_平均着順', 'コース馬場_r200_勝率',
        '種牡馬_ダ_勝率', '近5走_上り3F平均', '近5走_タイム指数_max', '1走前_馬場状態',
        'タイム指数_近5走_slope', '上り3F_近3走_slope', '近3走_勝率', 'タイム指数_加速度',
        '1走前_クラス調整着順', '距離変化_前走', '間隔',
        '騎手馬場_r100_勝率', '近5走_タイム指数平均', '馬体重増減',
        'タイム指数_近3走_slope', '近5走_上り3F_std',
        '着順_近3走_slope', '馬距離_勝率', '母父馬_勝率', '前走着差タイム', '騎手変更',
    ],
    'ダ短': [
        '馬番', '斤量', '芝ダ一致_平均着順_近5走', '1走前_タイム指数',
        '近5走_クラス調整_平均着順', '輸送有無', '馬コース_r20_勝率',
        '近3走_体重増減合計', '性別_num', '1走前_クラス差', 'コース枠_r200_複勝率',
        'コース枠_r200_勝率', '3走前_クラス差', '近10走_勝率', '調教師_r200_勝率',
        'ブリンカー変更', '2走前_クラス差', '相手レベル_平均着順', 'コース馬場_r200_勝率',
        '種牡馬_ダ_勝率', '近5走_上り3F平均', '近5走_タイム指数_max', '1走前_馬場状態',
        'タイム指数_近5走_slope', '上り3F_近3走_slope', '近3走_勝率', 'タイム指数_加速度',
        '1走前_クラス調整着順', '距離変化_前走', '間隔',
        '騎手馬場_r100_勝率', '近5走_タイム指数平均', '馬体重増減',
        'タイム指数_近3走_slope', '近5走_上り3F_std',
        '着順_近3走_slope', '馬距離_勝率', '母父馬_勝率', '前走着差タイム', '騎手変更',
    ],
    '芝短': [
        '馬番', '斤量', '芝ダ一致_平均着順_近5走', '1走前_タイム指数',
        '近5走_クラス調整_平均着順', '馬コース_r20_勝率', '騎手コース_r100_勝率',
        '近10走_勝率', '馬体重', '近5走_上り3F平均', '芝ダ転向',
        '近3走_体重増減合計', '相手レベル_平均着順', '道悪_平均着順_近5走',
        'タイム指数_近3走_slope', '馬体重増減', 'コース馬場_r200_勝率',
        'ブリンカー変更', '3走前_クラス差', '近5走_上り3F_std',
        '騎手コース距離_r100_勝率', '2走前_クラス差', '着順_近3走_slope',
        '近3走_複勝率', '馬距離_勝率', '1走前_3角', '1走前_馬場状態',
        'コース枠_r200_勝率', '前走着差タイム', '馬_r20_勝率',
    ],
    '芝中': [
        '馬番', '斤量', '芝ダ一致_平均着順_近5走', '騎手距離_r100_勝率',
        '1走前_タイム指数', '1走前_クラス調整着順', '馬コース_r20_勝率',
        '種牡馬_ダ_勝率', '馬体重', '前走着差タイム', '近5走_上り3F平均',
        '近5走_タイム指数平均', '1走前_クラス差', '同馬場_平均着順_近5走',
        'コース枠_r200_勝率', '輸送有無', '性別_num', '1走前_馬場状態',
        'ブリンカー変更', 'コース馬場_r200_勝率', '芝ダ転向', '近3走_体重増減合計',
        '間隔', '相手レベル_平均着順', 'タイム指数_加速度', '母父馬_勝率',
        '近3走_勝率', '騎手変更', '1走前_4角', '着順_近3走_slope',
        '上り3F_近3走_slope', '1走前_3角', '近5走_上り3F_std',
        '近5走_タイム指数_max', 'タイム指数_近5走_slope',
    ],
    '芝長': [
        '近3走_複勝率', '騎手距離_r100_勝率', '近5走_タイム指数平均',
        '馬コース_r20_勝率', 'タイム指数_近3走_slope', '調教師コース_r100_勝率',
        '同会場_複勝率_近5走', '近5走_上り3F_std', 'コース枠_r200_複勝率',
        '相手レベル_平均着順', 'タイム指数_加速度', '近10走_勝率', '近3走_体重増減合計',
        'コース馬場_r200_勝率', '1走前_馬場状態', '種牡馬_ダ_勝率', 'タイム指数_近5走_slope',
        '道悪_平均着順_近5走', '1走前_タイム指数', '距離変化_前走', '性別_num',
        '馬体重', '馬体重増減', '調教師_r200_勝率', '輸送有無', 'コース枠_r200_勝率',
        'ブリンカー変更', '種牡馬_勝率', '馬番', '斤量',
    ],
}

# ─── 追加する新候補（全セグメント共通） ───
NEW_CANDIDATES = [
    '1走前_単勝オッズ',      # 前走の市場評価（人気）
    '1走前_上り3F',          # 前走の具体的上がりタイム
    '1走前_PCI',             # 前走ペース指数（前傾・後傾）
    '1走前_RPCI',            # 前走相対ペース
    '1走前_頭数',            # 前走の出走頭数
    '2走前_タイム指数',       # 2走前の個別タイム指数
    '2走前_上り3F',          # 2走前上がり
    '2走前_着順_num',        # 2走前の着順（個別）
    '3走前_タイム指数',       # 3走前の個別タイム指数
    '3走前_上り3F',          # 3走前上がり
    '近10走_複勝率',         # 長期複勝率
    '近5走_複勝率',          # 中期複勝率
    '近5走_タイム指数_min',  # 最低タイム指数（安定性の逆）
    # 既存CANDIDATESから各セグメントでまだ試せていないもの
    '芝ダ一致_平均着順_近5走',
    '1走前_3角', '1走前_4角', '1走前_脚質_num',
    '芝ダ転向', '距離変化_前走', '馬距離_勝率',
    '前走着差タイム', '近5走_上り3F平均', '近5走_上り3F_std',
    '1走前_タイム指数', '近5走_タイム指数平均', '近5走_タイム指数_max',
    'タイム指数_近3走_slope', 'タイム指数_近5走_slope', 'タイム指数_加速度',
    '着順_近3走_slope', '着順_近5走_slope', '上り3F_近3走_slope', '4角位置_近3走_slope',
    '馬体重', '馬体重増減', 'ブリンカー変更', '近3走_体重増減合計',
    'コース枠_r200_勝率', 'コース脚質_r200_勝率', 'コース馬場_r200_勝率', 'コース枠_r200_複勝率',
    '1走前_クラス差', '2走前_クラス差', '3走前_クラス差',
    '1走前_クラス調整着順', '近5走_クラス調整_平均着順',
    '1走前_馬場状態', '道悪_平均着順_近5走', '同馬場_平均着順_近5走', '良馬場_平均着順_近5走',
    '種牡馬_勝率', '種牡馬_ダ_勝率', '母父馬_勝率',
    '馬_r20_勝率', '馬コース_r20_勝率',
    '騎手コース_r100_勝率', '騎手会場_r100_勝率', '騎手コース距離_r100_勝率',
    '騎手距離_r100_勝率', '騎手馬場_r100_勝率',
    '調教師コース_r100_勝率', '調教師_r200_勝率',
    '近3走_複勝率', '近3走_勝率', '近10走_複勝率', '近10走_勝率', '近5走_複勝率',
    '同会場_複勝率_近5走', '同会場_平均着順_近5走',
    '同距離帯_平均着順_近5走', '相手レベル_平均着順',
    '間隔', '性別_num', '騎手変更', '騎手変更', '輸送有無', '馬番', '斤量',
]

FAV_TARGETS = {'ダ長': 0.3403, 'ダ短': 0.3490, '芝短': 0.2869, '芝中': 0.3321, '芝長': 0.3605}
SEG_VERSIONS = {'ダ長': 'da_long_acc_v8', 'ダ短': 'da_short_acc_v4', '芝短': 'shiba_short_acc_v2',
                '芝中': 'shiba_mid_acc_v3', '芝長': 'shiba_long_acc_v2'}


def expand_nan_ind(dfs, feats):
    ref = dfs[0]
    extended = []
    for f in feats:
        extended.append(f)
        if f not in ref.columns: continue
        if NAN_IND_THRESHOLD < ref[f].isna().mean() < 1.0:
            ind = f + '_isnan'
            for df in dfs:
                if f in df.columns and ind not in df.columns:
                    df[ind] = df[f].isna().astype(float)
            extended.append(ind)
    return extended


def _loss_grad(beta, X, y, gs, n, nr, l2=0.0):
    probs = segment_softmax(X @ beta, gs, n)
    loss = -np.sum(y * np.log(np.clip(probs, 1e-15, 1.0))) / nr + l2 * np.dot(beta, beta)
    grad = -(X.T @ (y - probs)) / nr + 2 * l2 * beta
    return loss, grad


def adam_fit(X_tr, y_tr, gs_tr, n_tr, nr_tr, X_va, y_va, gs_va, n_va, nr_va, l2=L2):
    d = X_tr.shape[1]
    beta, m, v = np.zeros(d), np.zeros(d), np.zeros(d)
    b1, b2, eps = 0.9, 0.999, 1e-8
    t, best_val, best_beta, no_imp = 0, np.inf, np.zeros(d), 0
    for epoch in range(1, N_EPOCHS + 1):
        _, grad = _loss_grad(beta, X_tr, y_tr, gs_tr, n_tr, nr_tr, l2)
        t += 1
        m = b1 * m + (1 - b1) * grad
        v = b2 * v + (1 - b2) * grad ** 2
        beta -= LR * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
        if epoch % 10 == 0:
            vl, _ = _loss_grad(beta, X_va, y_va, gs_va, n_va, nr_va, l2=0.0)
            if vl < best_val:
                best_val, best_beta, no_imp = vl, beta.copy(), 0
            else:
                no_imp += 1
            if no_imp >= PATIENCE // 10:
                break
    return best_beta


def evaluate_feats(feats, dfs, scaler=None, fit=False):
    df_trn, df_val, oos_2324, oos_2025 = dfs
    all_dfs = list(dfs)
    expanded = expand_nan_ind(all_dfs, feats)
    valid = [c for c in expanded if c in df_trn.columns
             and df_trn[c].isna().mean() < 1.0 and df_trn[c].std(ddof=0) > 0]
    if not valid:
        return float('-inf'), None, None
    try:
        X_tr, y_tr, gs_tr, n_tr, nr_tr, sc, *_ = prepare(
            df_trn, valid, top_idx=None, top_idx3=None, fit=True)
        X_va, y_va, gs_va, n_va, nr_va, *_ = prepare(
            df_val, valid, scaler=sc, top_idx=None, top_idx3=None)
        beta = adam_fit(X_tr, y_tr, gs_tr, n_tr, nr_tr,
                        X_va, y_va, gs_va, n_va, nr_va)
    except Exception:
        return float('-inf'), None, None

    def acc_top1(oos, beta, sc, valid):
        vp = [c for c in valid if c in oos.columns]
        X_p, _, gs_p, n_p, *_ = prepare(oos, vp, scaler=sc, top_idx=None, top_idx3=None)
        s = oos.sort_values('race_id').reset_index(drop=True)
        s['prob'] = segment_softmax(X_p @ beta, gs_p, n_p)
        s['rank'] = s.groupby('race_id')['prob'].rank(ascending=False, method='first')
        t = s[s['rank'] == 1]
        n = s['race_id'].nunique()
        return (t['着順_num'] == 1).mean(), n

    a2324, n2324 = acc_top1(oos_2324, beta, sc, valid)
    a25, n25 = acc_top1(oos_2025, beta, sc, valid)
    acc_2325 = (a2324 * n2324 + a25 * n25) / (n2324 + n25) if (n2324 + n25) > 0 else float('-inf')
    return acc_2325, beta, sc


def load_data():
    df = pd.read_parquet(DATA_FILE)
    df['日付_num'] = pd.to_numeric(df['日付'], errors='coerce')
    df['着順_num'] = pd.to_numeric(df['着順_num'], errors='coerce')
    df = df.dropna(subset=['日付_num', '着順_num'])
    df = df[df['着順_num'] < 99]
    df['race_id'] = (df['日付_num'].astype(int).astype(str) + '_' +
                     df['開催'].astype(str).str.strip() + '_' +
                     df['Ｒ'].astype(str).str.strip())
    df = df[df['開催'].notna()].copy()
    df['surface'] = df['距離'].astype(str).str.strip().str.extract(r'^([芝ダ])')[0].fillna('不明')
    dm = pd.to_numeric(df['距離'].astype(str).str.extract(r'(\d+)')[0], errors='coerce')
    df['クラス_rank'] = pd.to_numeric(df['クラス_rank'], errors='coerce')
    df = add_computed_features(df)
    if '今回_会場' in df.columns and '1走前_開催' in df.columns:
        df['輸送有無'] = (df['今回_会場'].astype(str) != df['1走前_開催'].astype(str).str[1]).astype(float)
        df.loc[df['1走前_開催'].isna(), '輸送有無'] = float('nan')
    baba_map = {'良': 0, '稍重': 1, '重': 2, '不良': 3}
    for col in df.columns:
        if '馬場状態' in col and col != '馬場状態':
            df[col] = df[col].map(baba_map)
    return df, dm


def get_seg(df, dm, name):
    s = df['surface']
    r = df['クラス_rank']
    if name == 'ダ長':   mask = (s == 'ダ') & (dm > 1400) & (r != 1.0)
    elif name == 'ダ短': mask = (s == 'ダ') & (dm <= 1400) & (r != 1.0)
    elif name == '芝短': mask = (s == '芝') & (dm <= 1400) & (r != 1.0)
    elif name == '芝中': mask = (s == '芝') & (dm > 1400) & (dm <= 2000) & (r != 1.0)
    elif name == '芝長': mask = (s == '芝') & (dm > 2000) & (r != 1.0)
    seg = df[mask].copy()
    seg['dist_m'] = dm[seg.index]
    return seg


def search_segment(name, seg):
    print(f'\n{"="*65}')
    print(f'  {name}  目標: {FAV_TARGETS[name]:.2%}（1番人気）')
    print(f'{"="*65}')

    df_trn = seg[(seg['日付_num'] >= 130101) & (seg['日付_num'] < 220101)]
    df_val = seg[(seg['日付_num'] >= 220101) & (seg['日付_num'] <= 221231)]
    oos_2324 = seg[(seg['日付_num'] >= 230101) & (seg['日付_num'] < 250101)]
    oos_2025 = seg[(seg['日付_num'] >= 250101) & (seg['日付_num'] < 260101)]
    dfs = (df_trn, df_val, oos_2324, oos_2025)

    # 候補リスト構築（重複除去）
    seed = SEEDS[name]
    candidates = list(dict.fromkeys(
        [c for c in NEW_CANDIDATES if c in seg.columns] +
        [c for c in seed if c in seg.columns]
    ))

    # 現在のSEEDから開始
    current = [f for f in seed if f in seg.columns]
    best_score, best_beta, best_sc = evaluate_feats(current, dfs)
    print(f'  SEED baseline: acc_2325={best_score:.4f} ({len(current)}特徴)')

    # greedy forward selection（新候補を優先）
    remaining = [c for c in candidates if c not in current]
    improved_count = 0
    t0 = time.time()

    while len(current) < MAX_FEATS and remaining:
        best_add = None
        best_add_score = best_score

        for cand in remaining:
            trial = current + [cand]
            score, beta, sc = evaluate_feats(trial, dfs)
            if score > best_add_score:
                best_add_score = score
                best_add = cand
                best_add_beta = beta
                best_add_sc = sc

        if best_add is None:
            print(f'  改善なし → 終了 ({len(current)}特徴, {time.time()-t0:.0f}s)')
            break

        current.append(best_add)
        remaining.remove(best_add)
        best_score = best_add_score
        best_beta = best_add_beta
        best_sc = best_add_sc
        improved_count += 1
        gap = FAV_TARGETS[name] - best_score
        marker = '★' if best_score >= FAV_TARGETS[name] else ('↑' if improved_count % 5 == 0 else ' ')
        print(f'  {marker} +{best_add:35s}  acc_2325={best_score:.4f}  gap={gap:+.4f}  ({len(current)}特徴, {time.time()-t0:.0f}s)')
        sys.stdout.flush()

        if best_score >= FAV_TARGETS[name]:
            print(f'\n  ★★★ 1番人気超え達成! ★★★  {name}: {best_score:.4f} > {FAV_TARGETS[name]:.4f}')
            break

    print(f'\n  最終: {name}  acc_2325={best_score:.4f}  ({len(current)}特徴)  目標まで{FAV_TARGETS[name]-best_score:+.4f}')
    return best_score, best_beta, best_sc, current


def save_result(name, score, beta, sc, feats, seg):
    df_trn = seg[(seg['日付_num'] >= 130101) & (seg['日付_num'] < 220101)]
    df_val = seg[(seg['日付_num'] >= 220101) & (seg['日付_num'] <= 221231)]
    oos_2324 = seg[(seg['日付_num'] >= 230101) & (seg['日付_num'] < 250101)]
    oos_2025 = seg[(seg['日付_num'] >= 250101) & (seg['日付_num'] < 260101)]
    oos_2026 = seg[seg['日付_num'] >= 260101]

    all_dfs = [df_trn, df_val, oos_2324, oos_2025, oos_2026]
    expanded = expand_nan_ind(all_dfs, feats)
    valid = [c for c in expanded if c in df_trn.columns
             and df_trn[c].isna().mean() < 1.0 and df_trn[c].std(ddof=0) > 0]

    X_tr, y_tr, gs_tr, n_tr, nr_tr, scaler, *_ = prepare(
        df_trn, valid, top_idx=None, top_idx3=None, fit=True)
    X_va, y_va, gs_va, n_va, nr_va, *_ = prepare(
        df_val, valid, scaler=scaler, top_idx=None, top_idx3=None)
    beta_final = adam_fit(X_tr, y_tr, gs_tr, n_tr, nr_tr,
                          X_va, y_va, gs_va, n_va, nr_va)

    val_s = df_val.sort_values('race_id').reset_index(drop=True)
    raw_val = segment_softmax(X_va @ beta_final, gs_va, n_va)
    y_val = (val_s['着順_num'] == 1).astype(float).values
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(raw_val, y_val)

    results = {}
    for label, oos in [('2324', oos_2324), ('2025', oos_2025), ('2026', oos_2026)]:
        if len(oos) == 0: continue
        vp = [c for c in valid if c in oos.columns]
        X_p, _, gs_p, n_p, *_ = prepare(oos, vp, scaler=scaler, top_idx=None, top_idx3=None)
        s = oos.sort_values('race_id').reset_index(drop=True)
        s['prob'] = segment_softmax(X_p @ beta_final, gs_p, n_p)
        s['rank'] = s.groupby('race_id')['prob'].rank(ascending=False, method='first')
        t = s[s['rank'] == 1]
        nr = s['race_id'].nunique()
        acc = (t['着順_num'] == 1).mean()
        odds = pd.to_numeric(t['単勝オッズ'], errors='coerce')
        roi = (odds[t['着順_num'] == 1] * 100).sum() / (len(t) * 100) - 1
        results[label] = (acc, roi, nr)
        print(f'  {label}: acc={acc:.2%}  ROI={roi:+.2%}  ({nr}R)')

    n2324 = results.get('2324', (0, 0, 0))[2]
    n25 = results.get('2025', (0, 0, 0))[2]
    n26 = results.get('2026', (0, 0, 0))[2]
    a2324 = results.get('2324', (0, 0, 0))[0]
    a25 = results.get('2025', (0, 0, 0))[0]
    a26 = results.get('2026', (0, 0, 0))[0]
    r25 = results.get('2025', (0, 0, 0))[1]
    r26 = results.get('2026', (0, 0, 0))[1]
    acc_2325 = (a2324 * n2324 + a25 * n25) / (n2324 + n25) if (n2324 + n25) > 0 else 0.0
    acc_2526 = (a25 * n25 + a26 * n26) / (n25 + n26) if (n25 + n26) > 0 else 0.0
    roi_2526 = (r25 * n25 + r26 * n26) / (n25 + n26) if (n25 + n26) > 0 else 0.0
    print(f'  acc_2325={acc_2325:.4f}  25+26_acc={acc_2526:.4f}  25+26_ROI={roi_2526:+.2%}')

    acc_pkg = {
        'segment': name,
        'scaler': scaler,
        'coef': beta_final,
        'feat_cols': valid,
        'isotonic': iso,
        'acc_2325': acc_2325,
        'acc_2526': acc_2526,
        'oos_roi_2526': roi_2526,
        'version': SEG_VERSIONS[name],
        'note': f'{SEG_VERSIONS[name]}: {len(feats)}特徴 acc_2325={acc_2325:.4f} 1番人気{FAV_TARGETS[name]:.2%}',
    }

    acc_pkl = os.path.join(BASE_DIR, 'models', 'hitrate_model.pkl')
    if os.path.exists(acc_pkl):
        existing = pickle.load(open(acc_pkl, 'rb'))
        existing[name] = acc_pkg
        with open(acc_pkl, 'wb') as f:
            pickle.dump(existing, f)
    else:
        with open(acc_pkl, 'wb') as f:
            pickle.dump({name: acc_pkg}, f)
    print(f'  保存完了: {name} → hitrate_model.pkl')
    return acc_2325


def main():
    print('全5セグメント 的中率最大化 統合探索 v2')
    print(f'目標: 各セグメントで1番人気的中率を超える')
    print(f'MAX_FEATS={MAX_FEATS}  L2={L2}')

    df, dm = load_data()
    print('データ読み込み完了')

    results = {}
    for name in ['芝長', '芝中', '芝短', 'ダ長', 'ダ短']:
        seg = get_seg(df, dm, name)
        score, beta, sc, feats = search_segment(name, seg)
        results[name] = score

        # 既存より改善した場合のみ保存
        existing_acc = 0.0
        acc_pkl = os.path.join(BASE_DIR, 'models', 'hitrate_model.pkl')
        if os.path.exists(acc_pkl):
            pkg = pickle.load(open(acc_pkl, 'rb'))
            if name in pkg:
                existing_acc = pkg[name].get('acc_2325', 0.0)

        if score > existing_acc + 0.0001:
            print(f'\n  改善あり ({existing_acc:.4f} → {score:.4f})、保存します...')
            final = save_result(name, score, beta, sc, feats, seg)
        else:
            print(f'\n  改善なし ({existing_acc:.4f} ≥ {score:.4f})、スキップ')

    print('\n' + '='*65)
    print('全セグメント完了')
    for name, score in results.items():
        gap = FAV_TARGETS[name] - score
        mark = '★超え' if score >= FAV_TARGETS[name] else f'gap={gap:+.4f}'
        print(f'  {name}: acc_2325={score:.4f}  {mark}')


if __name__ == '__main__':
    main()
