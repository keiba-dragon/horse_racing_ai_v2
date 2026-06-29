# coding: utf-8
"""
feature_search v5: OOS 2023-2025全体でgreedy選択 + 2026ホールドアウト
- 選択基準: OOS 2023-2025 ROI (11000R, SE≈1.2%)
- 真のホールドアウト: 2026データのみ（一度も評価に使わない）
- 改良点:
  * グリーディ閾値を0.3pp（v3の0.1ppより厳格）にしてノイズ耐性を上げる
  * セグメント別greedy (v3方式)
  * 新候補特徴量を追加
"""
import sys, os, json, datetime
import numpy as np
import pandas as pd

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from save_conditional_logit import prepare, segment_softmax, BASE_DIR, DATA_FILE, LR, N_EPOCHS, PATIENCE

LOG_FILE = os.path.join(BASE_DIR, 'logs', 'feature_search_v5.jsonl')
os.makedirs(os.path.join(BASE_DIR, 'logs'), exist_ok=True)

# v303確定特徴量をスタート地点とする
SEG_INIT = {
    '芝_短距離': [
        '1走前_タイム指数','1走前_上り3F','前走着差タイム','1走前_RPCI',
        '近5走_タイム指数平均','近5走_タイム指数_max','タイム指数_近3走_slope',
        '1走前_クラス調整着順','近5走_クラス調整_平均着順',
        '馬番','騎手コース_r100_勝率','調教師コース_r100_勝率',
        'コース脚質_r200_勝率','1走前_脚質_num','種牡馬_勝率','種牡馬_ダ_勝率',
        '斤量','芝ダ転向','間隔','1走前_3角',
    ],
    '芝_中距離': [
        '1走前_タイム指数','1走前_上り3F','前走着差タイム','1走前_RPCI',
        '近5走_タイム指数平均','近5走_タイム指数_max','タイム指数_近3走_slope',
        '1走前_クラス調整着順','近5走_クラス調整_平均着順',
        '馬番','コース枠_r200_勝率','騎手コース_r100_勝率','騎手変更','調教師コース_r100_勝率',
        'コース脚質_r200_勝率','1走前_脚質_num','種牡馬_勝率','種牡馬_ダ_勝率',
        '斤量','芝ダ転向','間隔','馬体重','間隔_短_flag','血統_ダ優位度','馬体重増減',
    ],
    '芝_長距離': [
        '1走前_タイム指数','1走前_上り3F','前走着差タイム','1走前_RPCI',
        '近5走_タイム指数平均','近5走_タイム指数_max','タイム指数_近3走_slope',
        '1走前_クラス調整着順','近5走_クラス調整_平均着順',
        '馬番','コース枠_r200_勝率','騎手コース_r100_勝率','騎手変更','調教師コース_r100_勝率',
        'コース脚質_r200_勝率','1走前_脚質_num','種牡馬_勝率','種牡馬_ダ_勝率',
        '斤量','芝ダ転向','間隔',
    ],
    'ダ_短距離': [
        '1走前_タイム指数','1走前_上り3F','前走着差タイム','1走前_RPCI',
        '近5走_タイム指数平均','近5走_タイム指数_max','タイム指数_近3走_slope',
        '1走前_クラス調整着順','近5走_クラス調整_平均着順',
        '馬番','コース枠_r200_勝率','騎手コース_r100_勝率','騎手変更','調教師コース_r100_勝率',
        'コース脚質_r200_勝率','1走前_脚質_num','種牡馬_勝率','種牡馬_ダ_勝率',
        '斤量','間隔','馬体重増減','展開フィット_v2','乗替り_近走不振','間隔_長_flag',
    ],
    'ダ_中長距離': [
        '1走前_タイム指数','1走前_上り3F','前走着差タイム','1走前_RPCI',
        '近5走_タイム指数平均','近5走_タイム指数_max',
        '1走前_クラス調整着順','近5走_クラス調整_平均着順',
        '馬番','コース枠_r200_勝率','騎手コース_r100_勝率','騎手変更','調教師コース_r100_勝率',
        '1走前_脚質_num','種牡馬_勝率','種牡馬_ダ_勝率',
        '斤量','間隔','間隔_長_flag','1走前_上3F地点差',
    ],
}

# 追加候補（v3未採用 + 新規）
ADD_CANDIDATES = [
    # v3で候補だったが未採用
    'コース枠_r200_勝率','騎手変更',                  # 芝_短距離で除外されたもの
    'タイム指数_近3走_slope','コース脚質_r200_勝率','芝ダ転向',  # ダ_中長距離で除外
    '馬体重','間隔_短_flag','血統_ダ優位度','馬体重増減',        # 芝_長距離未採用
    'キャリア','母父馬_勝率','1走前_上3F地点差',
    # 新規
    '前走着順_比',       # 前走着順/頭数（正規化済み）
    'タイム指数_不安定度',# 近5走max-mean
    '近5走_上り3F指数平均',
    '前走_人気',         # 前走の人気順位（市場評価の変化を捉える）
    '騎手調教師_r100_勝率',
    '近走連続入着数',
    '近3走_複勝率',
    '間隔_長_flag','間隔_短_flag',   # ダセグメントで未採用のもの
]

SEGMENTS_5 = [('芝','短距離'),('芝','中距離'),('芝','長距離'),('ダ','短距離'),('ダ','中長距離')]
THRESHOLD = 0.003   # 0.3pp (v3の0.1ppより厳格)


def add_computed_features(df):
    interval = pd.to_numeric(df['間隔'], errors='coerce') if '間隔' in df.columns else pd.Series(np.nan, index=df.index)
    df['間隔_長_flag'] = (interval >= 60).astype(float)
    df['間隔_短_flag'] = (interval <= 14).astype(float)
    heads = pd.to_numeric(df.get('1走前_頭数', np.nan), errors='coerce').clip(lower=1)
    df['前走着順_比'] = pd.to_numeric(df.get('1走前_着順_num', np.nan), errors='coerce') / heads
    da_r  = pd.to_numeric(df.get('種牡馬_ダ_勝率', np.nan), errors='coerce')
    all_r = pd.to_numeric(df.get('種牡馬_勝率',    np.nan), errors='coerce')
    df['血統_ダ優位度'] = da_r - all_r
    t5max  = pd.to_numeric(df.get('近5走_タイム指数_max',  np.nan), errors='coerce')
    t5mean = pd.to_numeric(df.get('近5走_タイム指数平均',  np.nan), errors='coerce')
    df['タイム指数_不安定度'] = t5max - t5mean
    # 前走人気: 1走前_単勝オッズから人気換算（低オッズ=高人気=低数字）
    if '1走前_単勝オッズ' in df.columns:
        prev_odds = pd.to_numeric(df['1走前_単勝オッズ'], errors='coerce')
        df['前走_人気'] = prev_odds   # オッズを直接使う（低=人気）
    return df


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
    dm = pd.to_numeric(df['距離'].astype(str).str.extract(r'(\d+)')[0], errors='coerce')
    df['dist_m'] = dm
    shi, da = df['surface'] == '芝', df['surface'] == 'ダ'
    df['dist_band'] = ''
    df.loc[shi & (dm <= 1400),               'dist_band'] = '短距離'
    df.loc[shi & (dm > 1400) & (dm <= 2000), 'dist_band'] = '中距離'
    df.loc[shi & (dm > 2000),                'dist_band'] = '長距離'
    df.loc[da  & (dm <= 1400),               'dist_band'] = '短距離'
    df.loc[da  & (dm > 1400),                'dist_band'] = '中長距離'
    df = add_computed_features(df)
    print(f'有効データ: {len(df):,}行')
    return df


def _loss_grad(beta, X, y, gs, n, nr, alpha=1.0):
    probs = segment_softmax(X @ beta, gs, n)
    res = y - probs
    return (-np.sum(y * np.log(np.clip(probs, 1e-15, 1.0))) + alpha * np.sum(beta**2)) / nr, \
           (-(X.T @ res) + 2 * alpha * beta) / nr


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
    valid = [c for c in feat_cols if c in df_s.columns and df_s[c].isna().mean() <= 0.65]
    if len(valid) < 3:
        return None
    trn = df_s[(df_s['日付_num'] >= 130101) & (df_s['日付_num'] < 220101)]
    val = df_s[(df_s['日付_num'] >= 220101) & (df_s['日付_num'] <= 221231)]
    oos_sel = df_s[(df_s['日付_num'] >= 230101) & (df_s['日付_num'] < 260101)].copy()  # 2023-2025
    oos_ho  = df_s[df_s['日付_num'] >= 260101].copy()                                   # 2026 holdout
    if len(trn) < 300 or len(val) < 30 or len(oos_sel) < 50:
        return None
    try:
        X_tr, y_tr, gs_tr, n_tr, nr_tr, scaler, *_ = prepare(trn, valid, top_idx=None, top_idx3=None, fit=True)
        X_va, y_va, gs_va, n_va, nr_va, *_          = prepare(val, valid, scaler=scaler, top_idx=None, top_idx3=None)
        X_oo, y_oo, gs_oo, n_oo, *_                 = prepare(oos_sel, valid, scaler=scaler, top_idx=None, top_idx3=None)
    except Exception as e:
        print(f'    prepare失敗: {e}')
        return None
    beta = adam_opt(X_tr, y_tr, gs_tr, n_tr, nr_tr, X_va, y_va, gs_va, n_va, nr_va, alpha)

    def calc_roi(df_period, X_p, gs_p, n_p):
        df_s2 = df_period.sort_values('race_id').reset_index(drop=True)
        probs = segment_softmax(X_p @ beta, gs_p, n_p)
        df_s2['model_prob'] = probs
        df_s2['rank_model'] = df_s2.groupby('race_id')['model_prob'].rank(ascending=False, method='first')
        df_s2['odds_num']   = pd.to_numeric(df_s2['単勝オッズ'], errors='coerce')
        top1 = df_s2[df_s2['rank_model'] == 1]
        if len(top1) == 0:
            return None, 0
        won = top1['着順_num'] == 1
        return float((top1.loc[won, 'odds_num'] * 100).sum() / (len(top1) * 100) - 1), len(top1)

    roi_sel, n_sel = calc_roi(oos_sel, X_oo, gs_oo, n_oo)

    roi_ho, n_ho = None, 0
    if len(oos_ho) > 10:
        try:
            X_ho, y_ho, gs_ho, n_ho2, *_ = prepare(oos_ho, valid, scaler=scaler, top_idx=None, top_idx3=None)
            roi_ho, n_ho = calc_roi(oos_ho, X_ho, gs_ho, n_ho2)
        except:
            pass

    return {'roi_sel': roi_sel, 'n_sel': n_sel, 'roi_ho': roi_ho, 'n_ho': n_ho,
            'valid': valid, 'beta': beta, 'scaler': scaler,
            'oos_sel': oos_sel, 'oos_ho': oos_ho, 'X_oo': X_oo, 'gs_oo': gs_oo, 'n_oo': n_oo}


def log_result(entry):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')


def greedy_segment(df, surf, dist_band, init_feats, candidates, alpha=1.0):
    """OOS 2023-2025 ROIを基準にしたgreedy search。2026は評価のみ。"""
    seg_key = f'{surf}_{dist_band}'
    df_s    = df[(df['surface'] == surf) & (df['dist_band'] == dist_band)].copy()
    current = list(init_feats)
    cands   = [c for c in candidates if c not in current
               and c in df_s.columns and df_s[c].isna().mean() <= 0.65]

    res = train_segment(df_s, current, alpha)
    if res is None:
        print(f'  [{seg_key}] ベースライン失敗')
        return current
    best_roi = res['roi_sel']
    ho_s = f'{res["roi_ho"]:+.4f}({res["n_ho"]}R)' if res['roi_ho'] is not None else 'N/A'
    print(f'  [{seg_key}] ベースライン: {res["n_sel"]}R OOS={best_roi:+.4f}  2026={ho_s}')
    log_result({'ts': datetime.datetime.now().isoformat(), 'phase': 'v5_baseline',
                'seg': seg_key, 'roi_sel': best_roi, 'roi_ho': res['roi_ho'],
                'n_sel': res['n_sel'], 'n_ho': res['n_ho'], 'n_feats': len(current)})

    improved = True
    while improved:
        improved = False
        # Forward
        for cand in list(cands):
            r = train_segment(df_s, current + [cand], alpha)
            if r is None or r['roi_sel'] is None:
                continue
            log_result({'ts': datetime.datetime.now().isoformat(), 'phase': 'v5_fwd',
                        'seg': seg_key, 'action': f'+{cand}',
                        'roi_sel': r['roi_sel'], 'roi_ho': r['roi_ho']})
            if r['roi_sel'] > best_roi + THRESHOLD:
                ho_s = f'{r["roi_ho"]:+.4f}' if r['roi_ho'] is not None else 'N/A'
                print(f'    ★ +{cand}: {best_roi:+.4f}→{r["roi_sel"]:+.4f}  2026={ho_s}')
                best_roi, current = r['roi_sel'], current + [cand]
                cands.remove(cand)
                improved = True
        # Backward
        for rem in list(current):
            if len(current) <= 3:
                break
            trial = [c for c in current if c != rem]
            r = train_segment(df_s, trial, alpha)
            if r is None or r['roi_sel'] is None:
                continue
            log_result({'ts': datetime.datetime.now().isoformat(), 'phase': 'v5_bwd',
                        'seg': seg_key, 'action': f'-{rem}',
                        'roi_sel': r['roi_sel'], 'roi_ho': r['roi_ho']})
            if r['roi_sel'] > best_roi + THRESHOLD:
                ho_s = f'{r["roi_ho"]:+.4f}' if r['roi_ho'] is not None else 'N/A'
                print(f'    ★ -{rem}: {best_roi:+.4f}→{r["roi_sel"]:+.4f}  2026={ho_s}')
                best_roi, current = r['roi_sel'], trial
                cands.append(rem)
                improved = True

    res_final = train_segment(df_s, current, alpha)
    roi_ho = res_final['roi_ho'] if res_final else None
    n_ho   = res_final['n_ho']   if res_final else 0
    ho_s   = f'{roi_ho:+.4f}({n_ho}R)' if roi_ho is not None else 'N/A'
    print(f'  [{seg_key}] 収束: {len(current)}本  OOS={best_roi:+.4f}  2026={ho_s}')
    log_result({'ts': datetime.datetime.now().isoformat(), 'phase': 'v5_converged',
                'seg': seg_key, 'roi_sel': best_roi, 'roi_ho': roi_ho,
                'n_sel': res_final['n_sel'] if res_final else 0,
                'n_ho': n_ho, 'features': current, 'n_feats': len(current)})
    return current


def main():
    df = load_data()

    # 有効候補確認
    all_cands = list(dict.fromkeys(ADD_CANDIDATES))  # 重複除去
    print(f'\n候補特徴量: {[c for c in all_cands if c in df.columns]}')
    missing = [c for c in all_cands if c not in df.columns]
    if missing:
        print(f'列なし: {missing}')

    print('\n' + '='*60)
    print('feature_search v5: OOS 2023-2025 greedy + 2026ホールドアウト')
    print(f'閾値: {THRESHOLD*100:.1f}pp')
    print('='*60)

    seg_feats_opt = {}
    for surf, dist_band in SEGMENTS_5:
        seg_key = f'{surf}_{dist_band}'
        init  = SEG_INIT.get(seg_key, list(SEG_INIT['芝_長距離']))
        cands = [c for c in all_cands if c not in init]
        print(f'\n=== {seg_key} (init:{len(init)}本, 候補:{len([c for c in cands if c in df.columns])}本) ===')
        best_feats = greedy_segment(df, surf, dist_band, init, cands)
        seg_feats_opt[seg_key] = best_feats

    print('\n' + '='*60)
    print('=== 最終結果 ===')
    for sk, feats in seg_feats_opt.items():
        print(f'  {sk}: {len(feats)}本')
    print(json.dumps(seg_feats_opt, ensure_ascii=False))


if __name__ == '__main__':
    main()
