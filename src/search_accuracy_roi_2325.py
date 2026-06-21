# coding: utf-8
"""
search_accuracy_roi_2325.py - 的中率モデル用 Greedy forward selection (ROI選択)

  損失: P(win) cross-entropy (clogit softmax, 的中率モデルと同じ)
  選択指標: 単勝ROI 2325合算 (23-24 + 2025, 単年±30%キャップ)
  特徴候補: search_acc_random.py の ALL_CANDS (タイム指数系・スロープ等を含む広いセット)
  特徴数: 最大 MAX_FEATS
  保存先: accuracy_model.pkl (新聞のランキングに使用)

usage:
  python src/search_accuracy_roi_2325.py ダ長
  python src/search_accuracy_roi_2325.py ダ短
  ...全セグメント
"""
import sys, os, time, pickle
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from save_conditional_logit import (prepare, segment_softmax,
                                    BASE_DIR, DATA_FILE, LR, N_EPOCHS, PATIENCE)
from save_v3 import add_computed_features

SEG_NAME  = sys.argv[1] if len(sys.argv) > 1 else 'ダ長'
MAX_FEATS = 10
ROI_CAP   = 0.30
L2        = 0.006
NAN_IND_THRESHOLD = 0.05

FORCED_BASE = ['馬番', '斤量']

SEG_FILTER = {
    'ダ長': lambda s, dm, cr: (s == 'ダ') & (dm > 1400)  & (cr != 1.0),
    'ダ短': lambda s, dm, cr: (s == 'ダ') & (dm <= 1400) & (cr != 1.0),
    '芝短': lambda s, dm, cr: (s == '芝') & (dm <= 1400) & (cr != 1.0),
    '芝中': lambda s, dm, cr: (s == '芝') & (dm > 1400)  & (dm <= 2000) & (cr != 1.0),
    '芝長': lambda s, dm, cr: (s == '芝') & (dm > 2000)  & (cr != 1.0),
}

# search_acc_random.py の ALL_CANDS より（タイム指数系・スロープ等を含む広いセット）
CANDIDATES = [
    '馬番', '斤量',
    '近3走_上り3F_min', '1走前_上り3F', '1走前_PCI', '1走前_頭数',
    '2走前_タイム指数', '2走前_上り3F', '3走前_タイム指数', '3走前_上り3F',
    '近10走_複勝率', '近10走_勝率', '近5走_複勝率', '近5走_タイム指数_min',
    '芝ダ一致_平均着順_近5走', '1走前_3角', '1走前_4角', '1走前_脚質_num',
    '芝ダ転向', '距離変化_前走', '馬距離_勝率', '前走着差タイム',
    '近5走_上り3F平均', '近5走_上り3F_std',
    '1走前_タイム指数', '近5走_タイム指数平均', '近5走_タイム指数_max',
    'タイム指数_近3走_slope', 'タイム指数_近5走_slope', 'タイム指数_加速度',
    '着順_近3走_slope', '上り3F_近3走_slope',
    '1走前_クラス差', '2走前_クラス差', '3走前_クラス差',
    '1走前_クラス調整着順', '近5走_クラス調整_平均着順',
    '1走前_馬場状態', '道悪_平均着順_近5走', '同馬場_平均着順_近5走', '良馬場_平均着順_近5走',
    '種牡馬_勝率', '種牡馬_ダ_勝率', '母父馬_勝率', '馬_r20_勝率', '馬コース_r20_勝率',
    '騎手コース_r100_勝率', '騎手会場_r100_勝率', '騎手距離_r100_勝率',
    '騎手馬場_r100_勝率', '調教師コース_r100_勝率', '調教師_r200_勝率',
    '近3走_複勝率', '近3走_勝率',
    '同会場_複勝率_近5走', '同会場_平均着順_近5走', '同距離帯_平均着順_近5走',
    '相手レベル_平均着順',
    '間隔', '性別_num', '騎手変更', '輸送有無',
    '馬体重', '馬体重増減', 'ブリンカー変更', '近3走_体重増減合計',
    'コース枠_r200_勝率', 'コース脚質_r200_勝率', 'コース馬場_r200_勝率', 'コース枠_r200_複勝率',
    '近5走_タイム指数_min',
    '種牡馬_芝_複勝率', '種牡馬_ダ_複勝率',
    'コース枠_r200_複勝率', '近3走_勝率', '近10走_勝率',
]
# 重複排除
CANDIDATES = list(dict.fromkeys(CANDIDATES))


def load_segment(seg_name):
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
    mask = SEG_FILTER[seg_name](df['surface'], dm, df['クラス_rank'])
    df = df[mask].copy()
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


def expand_nan_ind(dfs, feats):
    ref = dfs[0]
    extended = []
    for f in feats:
        extended.append(f)
        if f not in ref.columns:
            continue
        if NAN_IND_THRESHOLD < ref[f].isna().mean() < 1.0:
            ind = f + '_isnan'
            for df in dfs:
                if f in df.columns and ind not in df.columns:
                    df[ind] = df[f].isna().astype(float)
            extended.append(ind)
    return extended


def _loss_grad(beta, X, y, gs, n, nr):
    """P(win) cross-entropy with softmax (= clogit loss)"""
    probs = segment_softmax(X @ beta, gs, n)
    loss  = -np.sum(y * np.log(np.clip(probs, 1e-15, 1.0))) / nr + L2 * np.dot(beta, beta)
    grad  = -(X.T @ (y - probs)) / nr + 2 * L2 * beta
    return loss, grad


def adam_fit(X_tr, y_tr, gs_tr, n_tr, nr_tr,
             X_va, y_va, gs_va, n_va, nr_va):
    d = X_tr.shape[1]
    beta, m, v = np.zeros(d), np.zeros(d), np.zeros(d)
    b1, b2, eps = 0.9, 0.999, 1e-8
    t, best_val, best_beta, no_imp = 0, np.inf, np.zeros(d), 0
    for epoch in range(1, N_EPOCHS + 1):
        _, grad = _loss_grad(beta, X_tr, y_tr, gs_tr, n_tr, nr_tr)
        t += 1
        m = b1 * m + (1 - b1) * grad
        v = b2 * v + (1 - b2) * grad ** 2
        beta -= LR * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
        if epoch % 10 == 0:
            vl, _ = _loss_grad(beta, X_va, y_va, gs_va, n_va, nr_va)
            if vl < best_val:
                best_val, best_beta, no_imp = vl, beta.copy(), 0
            else:
                no_imp += 1
            if no_imp >= PATIENCE // 10:
                break
    return best_beta


def tansho_roi(oos_sorted, beta, valid, scaler):
    """モデルrank-1馬の単勝ROIを計算"""
    vp = [c for c in valid if c in oos_sorted.columns]
    X_p, _, gs_p, n_p, *_ = prepare(oos_sorted, vp, scaler=scaler,
                                     top_idx=None, top_idx3=None)
    s = oos_sorted.copy()
    s['_prob'] = segment_softmax(X_p @ beta, gs_p, n_p)
    s['_rank'] = s.groupby('race_id')['_prob'].rank(ascending=False, method='first')
    top1 = s[s['_rank'] == 1].copy()
    nr = len(top1)
    if nr == 0:
        return float('nan'), 0
    won  = top1['着順_num'] == 1
    odds = pd.to_numeric(top1['単勝オッズ'], errors='coerce')
    payout = (odds[won] * 100).sum()
    roi = payout / (nr * 100) - 1
    return roi, nr


def eval_feats(df_trn, df_val, oos_2324, oos_2025, oos_2026, feats):
    all_dfs = [df_trn, df_val, oos_2324, oos_2025, oos_2026]
    expanded = expand_nan_ind(all_dfs, feats)
    valid = [c for c in expanded
             if c in df_trn.columns
             and df_trn[c].isna().mean() < 1.0
             and df_trn[c].std(ddof=0) > 0]
    if not valid:
        return float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), valid, None, None

    try:
        X_tr, y_tr, gs_tr, n_tr, nr_tr, scaler, *_ = prepare(
            df_trn, valid, top_idx=None, top_idx3=None, fit=True)
        X_va, y_va, gs_va, n_va, nr_va, *_ = prepare(
            df_val, valid, scaler=scaler, top_idx=None, top_idx3=None)
        beta = adam_fit(X_tr, y_tr, gs_tr, n_tr, nr_tr,
                        X_va, y_va, gs_va, n_va, nr_va)
    except Exception as e:
        return float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), valid, None, None

    r2324, n2324 = tansho_roi(oos_2324.sort_values('race_id').reset_index(drop=True), beta, valid, scaler)
    r25,   n25   = tansho_roi(oos_2025.sort_values('race_id').reset_index(drop=True), beta, valid, scaler)
    r26,   n26   = tansho_roi(oos_2026.sort_values('race_id').reset_index(drop=True), beta, valid, scaler)

    r2324c = np.clip(r2324, -ROI_CAP, ROI_CAP) if not np.isnan(r2324) else float('nan')
    r25c   = np.clip(r25,   -ROI_CAP, ROI_CAP) if not np.isnan(r25)   else float('nan')
    r2325  = (r2324c * n2324 + r25c * n25) / (n2324 + n25) if (n2324 + n25) > 0 else float('nan')
    c2526  = (r25 * n25 + r26 * n26) / (n25 + n26) if (n25 + n26) > 0 else float('nan')
    return r2325, r2324, c2526, r25, r26, valid, beta, scaler


def main():
    t0 = time.time()
    print('=' * 90)
    print(f'  {SEG_NAME} 的中率モデル × ROI選択 Greedy forward selection')
    print(f'  損失: P(win) softmax CE  選択指標: 単勝ROI 2325 (±{ROI_CAP:.0%}キャップ)  最大{MAX_FEATS}特徴')
    print('=' * 90)

    df = load_segment(SEG_NAME)
    df_trn   = df[(df['日付_num'] >= 130101) & (df['日付_num'] < 220101)]
    df_val   = df[(df['日付_num'] >= 220101) & (df['日付_num'] <= 221231)]
    oos_2324 = df[(df['日付_num'] >= 230101) & (df['日付_num'] < 250101)]
    oos_2025 = df[(df['日付_num'] >= 250101) & (df['日付_num'] < 260101)]
    oos_2026 = df[df['日付_num'] >= 260101]

    print(f'\ntrain:{len(df_trn):,}行({df_trn["race_id"].nunique()}R)  '
          f'val:{len(df_val):,}行({df_val["race_id"].nunique()}R)  '
          f'2324:{oos_2324["race_id"].nunique()}R  '
          f'2025:{oos_2025["race_id"].nunique()}R  '
          f'2026:{oos_2026["race_id"].nunique()}R')

    selected  = list(FORCED_BASE)
    remaining = [c for c in CANDIDATES if c not in FORCED_BASE]
    best_r2325, best_valid, best_beta, best_scaler = float('-inf'), None, None, None

    r_base, r2324_b, c_b, r25_b, r26_b, valid_b, beta_b, sc_b = eval_feats(
        df_trn, df_val, oos_2324, oos_2025, oos_2026, selected)
    print(f'\n[強制ベース {FORCED_BASE}]  2325={r_base*100:+.2f}%  '
          f'2323={r2324_b*100:+.2f}%  2025={r25_b*100:+.2f}%  2026={r26_b*100:+.2f}%')
    best_r2325, best_valid, best_beta, best_scaler = (
        r_base if not np.isnan(r_base) else float('-inf'), valid_b, beta_b, sc_b)

    print('\n=== Greedy forward selection (単勝ROI 2325) ===')
    while len(selected) < MAX_FEATS and remaining:
        round_res = []
        for feat in remaining:
            r2325, r2324, c2526, r25, r26, valid, beta, sc = eval_feats(
                df_trn, df_val, oos_2324, oos_2025, oos_2026, selected + [feat])
            round_res.append((feat, r2325, r2324, c2526, r25, r26, valid, beta, sc))
        round_res.sort(key=lambda x: -x[1] if not np.isnan(x[1]) else 999)

        print(f'\n[{len(selected)}特徴 → {MAX_FEATS}目標] 上位10候補:')
        print(f"  {'追加候補':30s}  {'2325(選択)':>10}  {'2323':>8}  {'2025':>8}  {'2026':>8}")
        for feat, r2325, r2324, c2526, r25, r26, *_ in round_res[:10]:
            mark = ' ←BEST' if feat == round_res[0][0] else ''
            print(f'  +{feat:29s}  {r2325*100:+9.2f}%  {r2324*100:+7.2f}%  '
                  f'{r25*100:+7.2f}%  {r26*100:+7.2f}%{mark}')

        best = round_res[0]
        if not np.isnan(best[1]):
            improved = best[1] > best_r2325
            mark = '✓' if improved else '△'
            selected.append(best[0])
            remaining.remove(best[0])
            if improved:
                best_r2325, best_valid, best_beta, best_scaler = (
                    best[1], best[6], best[7], best[8])
            print(f'\n  {mark} 採用: +{best[0]}  2325={best[1]*100:+.2f}%  '
                  f'2323={best[2]*100:+.2f}%  2025={best[4]*100:+.2f}%  2026={best[5]*100:+.2f}%')
        else:
            print('\n  ✗ 全候補NaN。探索終了。')
            break

    print(f'\n\n{"="*90}')
    print(f'  最終特徴量 ({len(selected)}個): {selected}')
    print(f'  最良 2325単勝ROI: {best_r2325*100:+.2f}%')
    print(f'{"="*90}')

    if best_beta is None:
        print('保存スキップ（betaなし）')
        return

    # ── 最終フィット + Isotonic calibration ──────────────────────────────────
    all_dfs = [df_trn, df_val, oos_2324, oos_2025, oos_2026]
    expanded_final = expand_nan_ind(all_dfs, selected)
    valid_final = [c for c in expanded_final
                   if c in df_trn.columns
                   and df_trn[c].isna().mean() < 1.0
                   and df_trn[c].std(ddof=0) > 0]
    X_tr_f, y_tr_f, gs_tr_f, n_tr_f, nr_tr_f, sc_f, *_ = prepare(
        df_trn, valid_final, top_idx=None, top_idx3=None, fit=True)
    X_va_f, y_va_f, gs_va_f, n_va_f, nr_va_f, *_ = prepare(
        df_val, valid_final, scaler=sc_f, top_idx=None, top_idx3=None)
    beta_f = adam_fit(X_tr_f, y_tr_f, gs_tr_f, n_tr_f, nr_tr_f,
                      X_va_f, y_va_f, gs_va_f, n_va_f, nr_va_f)
    raw_val = segment_softmax(X_va_f @ beta_f, gs_va_f, n_va_f)
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(raw_val, y_va_f)

    # ── accuracy_model.pkl に保存 ─────────────────────────────────────────────
    model_path = os.path.join(BASE_DIR, 'models', 'accuracy_model.pkl')
    try:
        with open(model_path, 'rb') as f:
            pkg = pickle.load(f)
    except Exception:
        pkg = {}

    seg_key_map = {'ダ長': 'ダ', 'ダ短': 'ダ短', '芝短': '芝短', '芝中': '芝中', '芝長': '芝長'}
    pkg[seg_key_map[SEG_NAME]] = {
        'feat_cols': valid_final,
        'scaler':    sc_f,
        'coef':      beta_f,
        'isotonic':  iso,
        'best_roi_2325': best_r2325,
        'selected_feats': selected,
    }
    with open(model_path, 'wb') as f:
        pickle.dump(pkg, f)
    print(f'\n保存完了: {model_path}  [{SEG_NAME}]')
    print(f'特徴量: {valid_final}')
    print(f'経過時間: {(time.time()-t0)/60:.1f}分')


if __name__ == '__main__':
    main()
