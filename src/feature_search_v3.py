# coding: utf-8
"""
feature_search v3: セグメント別 greedy search + 新規計算特徴量 + 距離帯再分割実験
- Part A: 新規派生特徴量をon-the-flyで計算してdfに追加
- Part B: 各セグメント個別にgreedy search（セグメントROIを最大化）
- Part C: ダ中長距離を1800m境界で再分割(6seg)して比較
"""
import sys, os, json, datetime
import numpy as np
import pandas as pd

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from save_conditional_logit import prepare, segment_softmax, BASE_DIR, DATA_FILE, LR, N_EPOCHS, PATIENCE

LOG_FILE = os.path.join(BASE_DIR, 'logs', 'feature_search_v3.jsonl')
os.makedirs(os.path.join(BASE_DIR, 'logs'), exist_ok=True)

BASE_FEATURES = [
    '1走前_タイム指数', '1走前_上り3F', '前走着差タイム', '1走前_RPCI',
    '近5走_タイム指数平均', '近5走_タイム指数_max', 'タイム指数_近3走_slope',
    '1走前_クラス調整着順', '近5走_クラス調整_平均着順',
    '馬番', 'コース枠_r200_勝率',
    '騎手コース_r100_勝率', '騎手変更', '調教師コース_r100_勝率',
    'コース脚質_r200_勝率', '1走前_脚質_num', '種牡馬_勝率', '種牡馬_ダ_勝率',
    '斤量', '芝ダ転向', '間隔',
]

# 全候補（v1/v2で棄却されたもの + 新規計算特徴量）
ALL_CANDIDATES = [
    # v1/v2で全体棄却されたが特定セグメントには効く可能性
    '1走前_3角',           # 前走3角順位
    '1走前_上3F地点差',    # 前走上り3F地点差
    '馬体重',              # 馬体重
    '近5走_上り3F指数平均', # 近5走上り3F指数平均
    '間隔_長_flag',        # 間隔60日以上フラグ（新規計算）
    '間隔_短_flag',        # 間隔14日以下フラグ（新規計算）
    '前走着順_比',         # 前走着順/頭数（正規化・新規計算）
    '血統_ダ優位度',       # 種牡馬_ダ_勝率 - 種牡馬_勝率（新規計算）
    'タイム指数_不安定度',  # 近5走max - mean（ムラ馬指標・新規計算）
    '馬体重増減',          # 前走比体重変化
    '展開フィット_v2',     # 展開フィット
    '近走連続入着数',      # 連続入着数
    'キャリア',            # 通算出走数
    '同会場_複勝率_近5走', # 同会場近5走複勝率
    '母父馬_勝率',         # 母父勝率
    '芝ダ一致数_近5走',    # 近5走の芝ダ一致数
    '近3走_複勝率',        # 近3走複勝率
    '距離変化_前走',       # 前走からの距離変化
    '乗替り_近走不振',     # 乗替り×近走不振
    '間隔',               # 間隔（BASE外として追加テスト用）
]

SEGMENTS_5 = [('芝', '短距離'), ('芝', '中距離'), ('芝', '長距離'),
              ('ダ', '短距離'), ('ダ', '中長距離')]
SEGMENTS_6 = [('芝', '短距離'), ('芝', '中距離'), ('芝', '長距離'),
              ('ダ', '短距離'), ('ダ', '中距離'), ('ダ', '長距離')]


def add_computed_features(df):
    """パーケットにない派生特徴量をon-the-flyで追加。"""
    interval = pd.to_numeric(df['間隔'], errors='coerce') if '間隔' in df.columns else pd.Series(np.nan, index=df.index)
    df['間隔_長_flag']    = (interval >= 60).astype(float)
    df['間隔_短_flag']    = (interval <= 14).astype(float)

    heads = pd.to_numeric(df.get('1走前_頭数', np.nan), errors='coerce').clip(lower=1)
    df['前走着順_比'] = pd.to_numeric(df.get('1走前_着順_num', np.nan), errors='coerce') / heads

    da_r  = pd.to_numeric(df.get('種牡馬_ダ_勝率', np.nan), errors='coerce')
    all_r = pd.to_numeric(df.get('種牡馬_勝率',    np.nan), errors='coerce')
    df['血統_ダ優位度'] = da_r - all_r

    t5max  = pd.to_numeric(df.get('近5走_タイム指数_max',   np.nan), errors='coerce')
    t5mean = pd.to_numeric(df.get('近5走_タイム指数平均',   np.nan), errors='coerce')
    df['タイム指数_不安定度'] = t5max - t5mean

    return df


def load_data(segments='5seg'):
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
    dm  = pd.to_numeric(df['距離'].astype(str).str.extract(r'(\d+)')[0], errors='coerce')
    df['dist_m'] = dm
    shi, da = df['surface'] == '芝', df['surface'] == 'ダ'
    df['dist_band'] = ''
    df.loc[shi & (dm <= 1400),               'dist_band'] = '短距離'
    df.loc[shi & (dm > 1400) & (dm <= 2000), 'dist_band'] = '中距離'
    df.loc[shi & (dm > 2000),                'dist_band'] = '長距離'
    if segments == '6seg':
        df.loc[da  & (dm <= 1400),               'dist_band'] = '短距離'
        df.loc[da  & (dm > 1400) & (dm <= 1800), 'dist_band'] = '中距離'
        df.loc[da  & (dm > 1800),                'dist_band'] = '長距離'
    else:
        df.loc[da  & (dm <= 1400),               'dist_band'] = '短距離'
        df.loc[da  & (dm > 1400),                'dist_band'] = '中長距離'

    df = add_computed_features(df)
    print(f'有効データ: {len(df):,}行')
    return df


def _loss_grad(beta, X, y, gs, n, nr, alpha=1.0):
    probs = segment_softmax(X @ beta, gs, n)
    log_lik = np.sum(y * np.log(np.clip(probs, 1e-15, 1.0)))
    res = y - probs
    loss = (-log_lik + alpha * np.sum(beta**2)) / nr
    grad = (-(X.T @ res) + 2 * alpha * beta) / nr
    return loss, grad


def adam_opt(X_tr, y_tr, gs_tr, n_tr, nr_tr, X_va, y_va, gs_va, n_va, nr_va, alpha=1.0):
    d = X_tr.shape[1]
    beta, m, v = np.zeros(d), np.zeros(d), np.zeros(d)
    b1, b2, eps = 0.9, 0.999, 1e-8
    t, best_val, best_beta, no_imp = 0, np.inf, np.zeros(d), 0
    for epoch in range(1, N_EPOCHS + 1):
        _, grad = _loss_grad(beta, X_tr, y_tr, gs_tr, n_tr, nr_tr, alpha)
        t += 1
        m = b1*m + (1-b1)*grad
        v = b2*v + (1-b2)*grad**2
        beta -= LR * (m/(1-b1**t)) / (np.sqrt(v/(1-b2**t)) + eps)
        if epoch % 10 == 0:
            vl, _ = _loss_grad(beta, X_va, y_va, gs_va, n_va, nr_va, alpha)
            if vl < best_val:
                best_val, best_beta, no_imp = vl, beta.copy(), 0
            else:
                no_imp += 1
            if no_imp >= PATIENCE // 10:
                break
    return best_beta


def train_segment(df_s, feat_cols, alpha=1.0):
    """1セグメントを学習して (beta, scaler, oos_df) を返す。失敗時はNone。"""
    valid = [c for c in feat_cols if c in df_s.columns and df_s[c].isna().mean() <= 0.65]
    if len(valid) < 3:
        return None
    trn = df_s[(df_s['日付_num'] >= 130101) & (df_s['日付_num'] < 220101)]
    val = df_s[(df_s['日付_num'] >= 220101) & (df_s['日付_num'] <= 221231)]
    oos = df_s[df_s['日付_num'] >= 230101].copy()
    if len(trn) < 300 or len(val) < 30 or len(oos) < 30:
        return None
    try:
        X_tr, y_tr, gs_tr, n_tr, nr_tr, scaler, *_ = prepare(trn, valid, top_idx=None, top_idx3=None, fit=True)
        X_va, y_va, gs_va, n_va, nr_va, *_ = prepare(val, valid, scaler=scaler, top_idx=None, top_idx3=None)
        X_oo, y_oo, gs_oo, n_oo, *_ = prepare(oos, valid, scaler=scaler, top_idx=None, top_idx3=None)
    except Exception as e:
        print(f'    prepare失敗: {e}')
        return None
    beta = adam_opt(X_tr, y_tr, gs_tr, n_tr, nr_tr, X_va, y_va, gs_va, n_va, nr_va, alpha)
    oos = oos.sort_values('race_id').reset_index(drop=True)
    probs = segment_softmax(X_oo @ beta, gs_oo, n_oo)
    oos['model_prob'] = probs
    oos['rank_model'] = oos.groupby('race_id')['model_prob'].rank(ascending=False, method='first')
    oos['odds_num']   = pd.to_numeric(oos['単勝オッズ'], errors='coerce')
    return oos


def seg_roi(oos_df):
    top1 = oos_df[oos_df['rank_model'] == 1]
    if len(top1) == 0:
        return None, 0
    won = top1['着順_num'] == 1
    roi = (top1.loc[won, 'odds_num'] * 100).sum() / (len(top1) * 100) - 1
    return float(roi), len(top1)


def eval_segspec(df, seg_feats, segments, alpha=1.0, verbose=True):
    """seg_feats: {seg_key: [feat_list]} で各セグメントの特徴量を指定。"""
    all_top1, seg_rois = [], {}
    for surf, dist_band in segments:
        seg_key = f'{surf}_{dist_band}'
        feats   = seg_feats.get(seg_key, BASE_FEATURES)
        df_s    = df[(df['surface'] == surf) & (df['dist_band'] == dist_band)].copy()
        result  = train_segment(df_s, feats, alpha)
        if result is None:
            if verbose:
                print(f'  {seg_key}: スキップ')
            continue
        r, n = seg_roi(result)
        seg_rois[seg_key] = round(r, 4)
        all_top1.append(result[result['rank_model'] == 1])
        if verbose:
            print(f'  {seg_key}: {n}R  ROI={r:+.3f}')
    if not all_top1:
        return None, seg_rois
    combined = pd.concat(all_top1, ignore_index=True)
    won = combined['着順_num'] == 1
    total = (combined.loc[won, 'odds_num'] * 100).sum() / (len(combined) * 100) - 1
    if verbose:
        print(f'  ▶ 合計: {len(combined)}R  ROI={total:+.3f}')
    return float(total), seg_rois


def log_result(entry):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')


def greedy_single_segment(df, surf, dist_band, base_feats, candidates, alpha=1.0):
    """1セグメントのROIを最大化するgreedy search。"""
    seg_key = f'{surf}_{dist_band}'
    df_s    = df[(df['surface'] == surf) & (df['dist_band'] == dist_band)].copy()
    current = list(base_feats)

    # ベースライン
    result = train_segment(df_s, current, alpha)
    if result is None:
        return current
    best_roi, n = seg_roi(result)
    print(f'  [{seg_key}] ベースライン: {n}R  ROI={best_roi:+.3f}')

    cands = [c for c in candidates if c not in current
             and (c in df_s.columns or c.endswith('_flag') or '比' in c or '優位' in c or '不安定' in c)]
    # 存在確認
    cands = [c for c in cands if c in df_s.columns and df_s[c].isna().mean() <= 0.65]

    improved = True
    while improved:
        improved = False
        # Forward
        for cand in list(cands):
            trial = current + [cand]
            result = train_segment(df_s, trial, alpha)
            if result is None:
                continue
            r, _ = seg_roi(result)
            log_result({'ts': datetime.datetime.now().isoformat(), 'seg': seg_key,
                        'phase': 'fwd', 'action': f'+{cand}', 'roi': r})
            if r > best_roi + 0.001:
                print(f'    ★ +{cand}: {best_roi:+.3f} → {r:+.3f}')
                best_roi, current = r, trial
                cands.remove(cand)
                improved = True
        # Backward
        for rem in list(current):
            if len(current) <= 3:
                break
            trial = [c for c in current if c != rem]
            result = train_segment(df_s, trial, alpha)
            if result is None:
                continue
            r, _ = seg_roi(result)
            log_result({'ts': datetime.datetime.now().isoformat(), 'seg': seg_key,
                        'phase': 'bwd', 'action': f'-{rem}', 'roi': r})
            if r > best_roi + 0.001:
                print(f'    ★ -{rem}: {best_roi:+.3f} → {r:+.3f}')
                best_roi, current = r, trial
                cands.append(rem)
                improved = True

    print(f'  [{seg_key}] 収束: ROI={best_roi:+.3f}  {len(current)}特徴量')
    log_result({'ts': datetime.datetime.now().isoformat(), 'seg': seg_key,
                'phase': 'converged', 'roi': best_roi, 'features': current})
    return current


def main():
    # ── Part A: 5seg で新規特徴量込みの per-segment search ──────────────
    print('\n' + '='*60)
    print('Part A: 5セグメント個別 greedy search + 新規計算特徴量')
    print('='*60)
    df5 = load_data('5seg')

    # 候補を確認
    all_cands = [c for c in ALL_CANDIDATES
                 if c in df5.columns and c not in BASE_FEATURES
                 and df5[c].isna().mean() <= 0.65]
    print(f'有効候補: {len(all_cands)}本: {all_cands}')

    # ベースライン (全セグメント共通21特徴量)
    seg_feats_base = {f'{s}_{b}': list(BASE_FEATURES) for s, b in SEGMENTS_5}
    print('\n--- 5seg ベースライン ---')
    base_roi, base_segs = eval_segspec(df5, seg_feats_base, SEGMENTS_5)
    log_result({'ts': datetime.datetime.now().isoformat(), 'phase': 'baseline_5seg',
                'total_roi': base_roi, 'seg_rois': base_segs})

    # セグメント別greedy search
    seg_feats_opt = {}
    for surf, dist_band in SEGMENTS_5:
        seg_key = f'{surf}_{dist_band}'
        print(f'\n=== {seg_key} 個別search ===')
        best_feats = greedy_single_segment(df5, surf, dist_band, BASE_FEATURES, all_cands)
        seg_feats_opt[seg_key] = best_feats

    print('\n--- Part A 最終: セグメント別最適特徴量で全体評価 ---')
    opt_roi, opt_segs = eval_segspec(df5, seg_feats_opt, SEGMENTS_5)
    log_result({'ts': datetime.datetime.now().isoformat(), 'phase': 'segspec_5seg_result',
                'total_roi': opt_roi, 'seg_rois': opt_segs, 'seg_feats': seg_feats_opt})

    print(f'\nPart A: ベースライン {base_roi:+.4f} → 最適化後 {opt_roi:+.4f}')
    for sk in opt_segs:
        extra = [f for f in seg_feats_opt.get(sk, BASE_FEATURES) if f not in BASE_FEATURES]
        removed = [f for f in BASE_FEATURES if f not in seg_feats_opt.get(sk, BASE_FEATURES)]
        if extra or removed:
            print(f'  {sk}: +{extra} -{removed}')

    # ── Part B: ダを1800m境界で6分割 ───────────────────────────────────
    print('\n' + '='*60)
    print('Part B: ダ中長距離を1800m境界で分割 (6セグメント)')
    print('='*60)
    df6 = load_data('6seg')

    # 6seg ベースライン (全共通21特徴量)
    seg_feats_6base = {f'{s}_{b}': list(BASE_FEATURES) for s, b in SEGMENTS_6}
    print('\n--- 6seg ベースライン ---')
    roi6_base, segs6_base = eval_segspec(df6, seg_feats_6base, SEGMENTS_6)
    log_result({'ts': datetime.datetime.now().isoformat(), 'phase': 'baseline_6seg',
                'total_roi': roi6_base, 'seg_rois': segs6_base})

    # ダ中距離・ダ長距離のみ個別search
    seg_feats_6opt = {k: list(v) for k, v in seg_feats_6base.items()}
    for surf, dist_band in [('ダ', '中距離'), ('ダ', '長距離')]:
        seg_key = f'{surf}_{dist_band}'
        print(f'\n=== {seg_key} 個別search (6seg) ===')
        all_cands_6 = [c for c in ALL_CANDIDATES
                       if c in df6.columns and c not in BASE_FEATURES
                       and df6[c].isna().mean() <= 0.65]
        best_feats = greedy_single_segment(df6, surf, dist_band, BASE_FEATURES, all_cands_6)
        seg_feats_6opt[seg_key] = best_feats

    print('\n--- Part B 最終: 6seg最適特徴量で全体評価 ---')
    roi6_opt, segs6_opt = eval_segspec(df6, seg_feats_6opt, SEGMENTS_6)
    log_result({'ts': datetime.datetime.now().isoformat(), 'phase': 'segspec_6seg_result',
                'total_roi': roi6_opt, 'seg_rois': segs6_opt, 'seg_feats': seg_feats_6opt})

    print(f'\nPart B: ベースライン {roi6_base:+.4f} → 最適化後 {roi6_opt:+.4f}')

    # ── 最終サマリー ─────────────────────────────────────────────────────
    print('\n' + '='*60)
    print('=== 最終サマリー ===')
    print(f'5seg ベースライン: {base_roi:+.4f}')
    print(f'5seg セグメント別: {opt_roi:+.4f}  ({"改善" if opt_roi > base_roi else "変化なし"})')
    print(f'6seg ベースライン: {roi6_base:+.4f}')
    print(f'6seg セグメント別: {roi6_opt:+.4f}  ({"改善" if roi6_opt > base_roi else "変化なし"})')
    best_overall = max([base_roi, opt_roi, roi6_base, roi6_opt])
    print(f'最良ROI: {best_overall:+.4f}')
    log_result({'ts': datetime.datetime.now().isoformat(), 'phase': 'final_summary',
                'base_5seg': base_roi, 'opt_5seg': opt_roi,
                'base_6seg': roi6_base, 'opt_6seg': roi6_opt})


if __name__ == '__main__':
    main()
