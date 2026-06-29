# coding: utf-8
"""
feature_search v2: ALPHA grid search + 追加候補 greedy search
Phase 1: ALPHA (0.3, 0.5, 1.5, 3.0) × 現21特徴量でOOS ROI比較
Phase 2: 最良ALPHAで新候補をgreedy forward+backward
"""
import sys, os, json, datetime
import numpy as np
import pandas as pd

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from save_conditional_logit import (
    prepare, segment_softmax,
    BASE_DIR, DATA_FILE, LR, N_EPOCHS, PATIENCE
)


def _loss_grad(beta, X, y, group_starts, n, n_races, alpha):
    """alpha可変版のneg_log_lik_and_grad。"""
    scores = X @ beta
    probs  = segment_softmax(scores, group_starts, n)
    log_lik = np.sum(y * np.log(np.clip(probs, 1e-15, 1.0)))
    residuals = y - probs
    loss = (-log_lik + alpha * np.sum(beta ** 2)) / n_races
    grad = (-(X.T @ residuals) + 2 * alpha * beta) / n_races
    return loss, grad

LOG_FILE = os.path.join(BASE_DIR, 'logs', 'feature_search_v2.jsonl')
os.makedirs(os.path.join(BASE_DIR, 'logs'), exist_ok=True)

# ── 現時点最良の21特徴量 ──────────────────────────────────────────────────
BEST_FEATURES = [
    '1走前_タイム指数', '1走前_上り3F', '前走着差タイム', '1走前_RPCI',
    '近5走_タイム指数平均', '近5走_タイム指数_max', 'タイム指数_近3走_slope',
    '1走前_クラス調整着順', '近5走_クラス調整_平均着順',
    '馬番', 'コース枠_r200_勝率',
    '騎手コース_r100_勝率', '騎手変更', '調教師コース_r100_勝率',
    'コース脚質_r200_勝率', '1走前_脚質_num', '種牡馬_勝率', '種牡馬_ダ_勝率',
    '斤量', '芝ダ転向', '間隔',
]

# ── Phase 2 追加候補 ──────────────────────────────────────────────────────
ADD_CANDIDATES_V2 = [
    '騎手調教師_r100_勝率',   # 騎手×調教師コンビ勝率（前回未試験）
    '産地_勝率',               # 産地（北海道等）の勝率
    '生産者_勝率',             # 生産者の勝率
    '馬体重増減',              # 前走からの体重変化（NaN=10%）
    '近10走_勝率',             # 長期実績（NaN=11%）
    '馬距離_複勝率',           # 馬の距離別複勝率
    '脚質フィット',            # 脚質×コース適性スコア（NaN=12%）
    '展開_コース_脚質フィット', # 展開×コース×脚質（NaN=12%）
]

SEGMENTS = [('芝', '短距離'), ('芝', '中距離'), ('芝', '長距離'),
            ('ダ', '短距離'), ('ダ', '中長距離')]


def load_data():
    print(f'データ読み込み: {DATA_FILE}')
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
    df = df[df['surface'].isin(['芝', 'ダ'])].copy()
    if 'クラス_rank' in df.columns:
        df = df[df['クラス_rank'] != 1.0].copy()
    df['dist_m'] = pd.to_numeric(
        df['距離'].astype(str).str.extract(r'(\d+)')[0], errors='coerce')
    dm, shi, da = df['dist_m'], df['surface'] == '芝', df['surface'] == 'ダ'
    df['dist_band'] = ''
    df.loc[shi & (dm <= 1400),               'dist_band'] = '短距離'
    df.loc[shi & (dm > 1400) & (dm <= 2000), 'dist_band'] = '中距離'
    df.loc[shi & (dm > 2000),                'dist_band'] = '長距離'
    df.loc[da  & (dm <= 1400),               'dist_band'] = '短距離'
    df.loc[da  & (dm > 1400),                'dist_band'] = '中長距離'
    print(f'有効データ: {len(df):,}行')
    return df


def adam_optimize(X_tr, y_tr, gs_tr, n_tr, nr_tr,
                  X_va, y_va, gs_va, n_va, nr_va, alpha):
    d = X_tr.shape[1]
    beta, m, v = np.zeros(d), np.zeros(d), np.zeros(d)
    b1, b2, eps = 0.9, 0.999, 1e-8
    t, best_val, best_beta, no_improve = 0, np.inf, np.zeros(d), 0
    CHECK = 10
    for epoch in range(1, N_EPOCHS + 1):
        loss, grad = _loss_grad(beta, X_tr, y_tr, gs_tr, n_tr, nr_tr, alpha)
        t += 1
        m = b1 * m + (1 - b1) * grad
        v = b2 * v + (1 - b2) * grad ** 2
        beta -= LR * (m / (1 - b1**t)) / (np.sqrt(v / (1 - b2**t)) + eps)
        if epoch % CHECK == 0:
            vl, _ = _loss_grad(beta, X_va, y_va, gs_va, n_va, nr_va, alpha)
            if vl < best_val:
                best_val, best_beta, no_improve = vl, beta.copy(), 0
            else:
                no_improve += 1
            if no_improve >= PATIENCE // CHECK:
                break
    return best_beta, best_val


def eval_features(df, feat_cols, alpha=1.0):
    valid_cols = [c for c in feat_cols if c in df.columns]
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        print(f'  [列なし] {missing}')
    if not valid_cols:
        return None, {}

    # NaN率チェック
    high_nan = [c for c in valid_cols if df[c].isna().mean() > 0.6]
    if high_nan:
        print(f'  [NaN>60%スキップ] {high_nan}')
        return None, {}

    all_top1, seg_rois = [], {}
    for surf, dist_band in SEGMENTS:
        df_s = df[(df['surface'] == surf) & (df['dist_band'] == dist_band)].copy()
        trn = df_s[(df_s['日付_num'] >= 130101) & (df_s['日付_num'] < 220101)]
        val = df_s[(df_s['日付_num'] >= 220101) & (df_s['日付_num'] <= 221231)]
        oos = df_s[df_s['日付_num'] >= 230101].copy()
        if len(trn) < 500 or len(val) < 50 or len(oos) < 50:
            continue
        try:
            X_tr, y_tr, gs_tr, n_tr, nr_tr, scaler, *_ = prepare(
                trn, valid_cols, top_idx=None, top_idx3=None, fit=True)
            X_va, y_va, gs_va, n_va, nr_va, *_ = prepare(
                val, valid_cols, scaler=scaler, top_idx=None, top_idx3=None)
            X_oo, y_oo, gs_oo, n_oo, nr_oo, *_ = prepare(
                oos, valid_cols, scaler=scaler, top_idx=None, top_idx3=None)
        except Exception as e:
            print(f'  [{surf}_{dist_band}] prepare失敗: {e}')
            continue
        beta, _ = adam_optimize(X_tr, y_tr, gs_tr, n_tr, nr_tr,
                                 X_va, y_va, gs_va, n_va, nr_va, alpha)
        oos = oos.sort_values('race_id').reset_index(drop=True)
        probs = segment_softmax(X_oo @ beta, gs_oo, n_oo)
        oos['model_prob'] = probs
        oos['rank_model'] = oos.groupby('race_id')['model_prob'].rank(
            ascending=False, method='first')
        oos['odds_num'] = pd.to_numeric(oos['単勝オッズ'], errors='coerce')
        top1 = oos[oos['rank_model'] == 1].copy()
        all_top1.append(top1)
        won = top1['着順_num'] == 1
        roi = (top1.loc[won, 'odds_num'] * 100).sum() / (len(top1) * 100) - 1
        seg_rois[f'{surf}_{dist_band}'] = round(float(roi), 4)
        print(f'  {surf}_{dist_band}: {len(top1)}R  ROI={roi:+.3f}')

    if not all_top1:
        return None, seg_rois
    combined = pd.concat(all_top1, ignore_index=True)
    won_all = combined['着順_num'] == 1
    total_roi = (combined.loc[won_all, 'odds_num'] * 100).sum() / (len(combined) * 100) - 1
    print(f'  ▶ 合計: {len(combined)}R  ROI={total_roi:+.3f}')
    return float(total_roi), seg_rois


def log_result(entry):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')


def greedy_search(df, init_feats, candidates, alpha, label=''):
    current = list(init_feats)
    cands = [c for c in candidates if c not in current and c in df.columns
             and df[c].isna().mean() <= 0.6]
    print(f'\n有効追加候補: {cands}')

    # ベースライン
    print(f'\n--- {label} ベースライン ({len(current)}特徴量, α={alpha}) ---')
    best_roi, best_seg = eval_features(df, current, alpha)
    log_result({'ts': datetime.datetime.now().isoformat(), 'phase': f'{label}_baseline',
                'alpha': alpha, 'n_feats': len(current), 'total_roi': best_roi, 'seg_rois': best_seg})

    round_num = 0
    while True:
        round_num += 1
        improved = False
        print(f'\n### {label} Round {round_num}  ROI={best_roi:+.4f} ({len(current)}特徴量) ###')

        # Forward
        for cand in list(cands):
            trial = current + [cand]
            print(f'\n[+] {cand}  ({len(trial)}特徴量)')
            roi, seg = eval_features(df, trial, alpha)
            if roi is None:
                continue
            log_result({'ts': datetime.datetime.now().isoformat(), 'phase': f'{label}_fwd_r{round_num}',
                        'action': f'+{cand}', 'alpha': alpha, 'n_feats': len(trial),
                        'total_roi': roi, 'seg_rois': seg})
            if roi > best_roi + 0.001:
                print(f'  ★ 採用: {cand}  {best_roi:+.4f} → {roi:+.4f}')
                best_roi, best_seg, current = roi, seg, trial
                cands.remove(cand)
                improved = True
                log_result({'ts': datetime.datetime.now().isoformat(), 'phase': f'{label}_adopted',
                            'action': f'+{cand}', 'alpha': alpha, 'n_feats': len(current),
                            'total_roi': best_roi, 'features': current})

        # Backward
        for rem in list(current):
            if len(current) <= 3:
                break
            trial = [c for c in current if c != rem]
            print(f'\n[-] {rem}  ({len(trial)}特徴量)')
            roi, seg = eval_features(df, trial, alpha)
            if roi is None:
                continue
            log_result({'ts': datetime.datetime.now().isoformat(), 'phase': f'{label}_bwd_r{round_num}',
                        'action': f'-{rem}', 'alpha': alpha, 'n_feats': len(trial),
                        'total_roi': roi, 'seg_rois': seg})
            if roi > best_roi + 0.001:
                print(f'  ★ 除外採用: {rem}  {best_roi:+.4f} → {roi:+.4f}')
                best_roi, best_seg, current = roi, seg, trial
                cands.append(rem)
                improved = True
                log_result({'ts': datetime.datetime.now().isoformat(), 'phase': f'{label}_adopted',
                            'action': f'-{rem}', 'alpha': alpha, 'n_feats': len(current),
                            'total_roi': best_roi, 'features': current})

        if not improved:
            print(f'\n収束: {label}  ROI={best_roi:+.4f}  {len(current)}特徴量')
            log_result({'ts': datetime.datetime.now().isoformat(), 'phase': f'{label}_converged',
                        'alpha': alpha, 'n_feats': len(current), 'features': current,
                        'total_roi': best_roi, 'seg_rois': best_seg})
            break
    return best_roi, current


def main():
    df = load_data()

    # ── Phase 1: ALPHA grid search ───────────────────────────────────────
    print('\n' + '='*60)
    print('Phase 1: ALPHA grid search (現21特徴量)')
    print('='*60)
    alpha_results = {}
    for alpha in [0.3, 0.5, 1.0, 1.5, 3.0]:
        print(f'\n--- α={alpha} ---')
        roi, seg = eval_features(df, BEST_FEATURES, alpha)
        alpha_results[alpha] = roi
        log_result({'ts': datetime.datetime.now().isoformat(), 'phase': 'alpha_grid',
                    'alpha': alpha, 'n_feats': len(BEST_FEATURES),
                    'total_roi': roi, 'seg_rois': seg})
        print(f'  α={alpha}: ROI={roi:+.4f}')

    print('\n=== ALPHA grid 結果 ===')
    for a, r in sorted(alpha_results.items(), key=lambda x: -x[1]):
        print(f'  α={a}: {r:+.4f}')
    best_alpha = max(alpha_results, key=alpha_results.get)
    best_alpha_roi = alpha_results[best_alpha]
    print(f'最良α={best_alpha} (ROI={best_alpha_roi:+.4f})')
    log_result({'ts': datetime.datetime.now().isoformat(), 'phase': 'alpha_best',
                'best_alpha': best_alpha, 'best_roi': best_alpha_roi, 'all': alpha_results})

    # ── Phase 2: 新候補 greedy search (最良α使用) ─────────────────────
    print('\n' + '='*60)
    print(f'Phase 2: 新候補 greedy search (α={best_alpha})')
    print('='*60)
    final_roi, final_feats = greedy_search(
        df, BEST_FEATURES, ADD_CANDIDATES_V2, best_alpha, label='v2')

    print('\n' + '='*60)
    print('=== 最終結果 ===')
    print(f'ROI: {final_roi:+.4f}  特徴量数: {len(final_feats)}')
    print(f'特徴量: {final_feats}')
    print('='*60)


if __name__ == '__main__':
    main()
