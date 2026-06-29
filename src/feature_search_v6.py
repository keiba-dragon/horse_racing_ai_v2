# coding: utf-8
"""
feature_search_v6.py - フル特徴量セット + L2正則化強度スイープ
47特徴量 x 5α x 5セグメント
早期停止: val NLL (正則化なし) → α比較を公平に
出力: logs/feature_search_v6.jsonl
"""
import sys, os, json, time
import numpy as np
import pandas as pd

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from save_conditional_logit import (
    prepare, segment_softmax, BASE_DIR, DATA_FILE, LR, N_EPOCHS, PATIENCE
)
from sklearn.isotonic import IsotonicRegression

LOG_FILE = os.path.join(BASE_DIR, 'logs', 'feature_search_v6.jsonl')
os.makedirs(os.path.join(BASE_DIR, 'logs'), exist_ok=True)

ALPHAS = [0.5, 1.0, 2.0, 3.0, 5.0]

FULL_FEATURES = [
    # Core time/pace
    '1走前_タイム指数', '1走前_上り3F', '前走着差タイム', '1走前_RPCI',
    '近5走_タイム指数平均', '近5走_タイム指数_max', 'タイム指数_近3走_slope',
    '近5走_上り3F指数平均',
    # 2走前/3走前
    '2走前_タイム指数', '2走前_上り3F', '2走前_着順_num',
    '3走前_タイム指数', '3走前_着順_num',
    # Class/rank
    '1走前_クラス調整着順', '近5走_クラス調整_平均着順',
    '1走前_着順_num',
    # Race position
    '馬番', 'コース枠_r200_勝率',
    # Jockey/trainer (multiple granularities)
    '騎手コース_r100_勝率', '騎手変更', '調教師コース_r100_勝率',
    '騎手_r200_勝率', '調教師_r200_勝率', '騎手調教師_r100_勝率',
    # Running style/corner
    'コース脚質_r200_勝率', '1走前_脚質_num', '1走前_3角', '1走前_4角',
    '展開フィット_v2', '乗替り_近走不振',
    # Sire/dam sire
    '種牡馬_勝率', '種牡馬_ダ_勝率', '母父馬_勝率',
    # Weight/physical
    '斤量', '馬体重', '馬体重増減', '1走前_馬体重', '1走前_馬体重増減',
    # Track switch/interval
    '芝ダ転向', '間隔', '間隔_長_flag', '間隔_短_flag',
    # Pace differential (previous race result, no leakage)
    '1走前_上3F地点差',
    # Career/history win/place rates
    'キャリア', '近3走_複勝率', '近10走_勝率', '馬距離_複勝率', '同会場_複勝率_近5走',
    # Derived (computed below)
    '血統_ダ優位度', '前走着順_比',
    # Previous race odds (market signal, previous race only)
    '1走前_単勝オッズ',
]

SEGMENTS = [('芝', '短距離'), ('芝', '中距離'), ('芝', '長距離'),
            ('ダ', '短距離'), ('ダ', '中長距離')]

# v303ベースライン OOS ROI（参考値）
V303_BASELINE = {
    '芝_短距離':  -0.1583,
    '芝_中距離':  -0.1749,
    '芝_長距離':  -0.1378,
    'ダ_短距離':  -0.1613,
    'ダ_中長距離': -0.2306,
}


def add_computed_features(df):
    interval = (pd.to_numeric(df['間隔'], errors='coerce')
                if '間隔' in df.columns
                else pd.Series(np.nan, index=df.index))
    df['間隔_長_flag'] = (interval >= 60).astype(float)
    df['間隔_短_flag'] = (interval <= 14).astype(float)

    da_r  = (pd.to_numeric(df['種牡馬_ダ_勝率'], errors='coerce')
             if '種牡馬_ダ_勝率' in df.columns
             else pd.Series(np.nan, index=df.index))
    all_r = (pd.to_numeric(df['種牡馬_勝率'], errors='coerce')
             if '種牡馬_勝率' in df.columns
             else pd.Series(np.nan, index=df.index))
    df['血統_ダ優位度'] = da_r - all_r

    if '1走前_頭数' in df.columns and '1走前_クラス調整着順' in df.columns:
        head = pd.to_numeric(df['1走前_頭数'], errors='coerce').replace(0, np.nan)
        rank = pd.to_numeric(df['1走前_クラス調整着順'], errors='coerce')
        df['前走着順_比'] = rank / head
    else:
        df['前走着順_比'] = np.nan

    return df


def log_result(data):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')


def load_data():
    print(f'読み込み: {DATA_FILE}')
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
    print(f'有効行: {len(df):,}')
    return df


def _nll(beta, X, y, gs, n):
    probs = segment_softmax(X @ beta, gs, n)
    return -np.sum(y * np.log(np.clip(probs, 1e-15, 1.0))) / n


def _loss_grad(beta, X, y, gs, n, nr, alpha):
    probs = segment_softmax(X @ beta, gs, n)
    log_lik = np.sum(y * np.log(np.clip(probs, 1e-15, 1.0)))
    res = y - probs
    loss = (-log_lik + alpha * np.sum(beta**2)) / nr
    grad = (-(X.T @ res) + 2 * alpha * beta) / nr
    return loss, grad


def adam_opt(X_tr, y_tr, gs_tr, n_tr, nr_tr,
             X_va, y_va, gs_va, n_va, alpha):
    d = X_tr.shape[1]
    beta, m, v = np.zeros(d), np.zeros(d), np.zeros(d)
    b1, b2, eps_a = 0.9, 0.999, 1e-8
    t, best_val, best_beta, no_imp = 0, np.inf, np.zeros(d), 0
    for epoch in range(1, N_EPOCHS + 1):
        _, grad = _loss_grad(beta, X_tr, y_tr, gs_tr, n_tr, nr_tr, alpha)
        t += 1
        m = b1 * m + (1 - b1) * grad
        v = b2 * v + (1 - b2) * grad ** 2
        beta -= LR * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps_a)
        if epoch % 10 == 0:
            # val NLL without regularization for fair α comparison
            vl = _nll(beta, X_va, y_va, gs_va, n_va)
            if vl < best_val:
                best_val, best_beta, no_imp = vl, beta.copy(), 0
            else:
                no_imp += 1
            if no_imp >= PATIENCE // 10:
                break
    return best_beta, float(best_val)


def score_and_roi(df_sub, valid, scaler, beta, label=''):
    if len(df_sub) == 0:
        return {'roi': float('nan'), 'n': 0, 'wins': 0}
    Xo, yo, gso, no, *_ = prepare(df_sub, valid, scaler=scaler,
                                   top_idx=None, top_idx3=None)
    sub = df_sub.sort_values('race_id').reset_index(drop=True)
    probs = segment_softmax(Xo @ beta, gso, no)
    sub['model_prob'] = probs
    sub['rank_model'] = sub.groupby('race_id')['model_prob'].rank(
        ascending=False, method='first')
    top1 = sub[sub['rank_model'] == 1]
    if len(top1) == 0:
        return {'roi': float('nan'), 'n': 0, 'wins': 0}
    won = top1['着順_num'] == 1
    odds = pd.to_numeric(top1['単勝オッズ'], errors='coerce')
    roi = float((odds[won] * 100).sum() / (len(top1) * 100) - 1)
    return {'roi': roi, 'n': int(len(top1)), 'wins': int(won.sum())}


def main():
    df = load_data()
    log_result({'event': 'start', 'ts': time.time(),
                'alphas': ALPHAS, 'n_feats': len(FULL_FEATURES),
                'features': FULL_FEATURES})

    all_results = {}  # seg -> list of (alpha, roi_2324, roi_2025, roi_2026)

    for surf, dist_band in SEGMENTS:
        seg_key = f'{surf}_{dist_band}'
        df_s = df[(df['surface'] == surf) & (df['dist_band'] == dist_band)].copy()

        valid = [c for c in FULL_FEATURES
                 if c in df_s.columns and df_s[c].isna().mean() <= 0.65]
        missing   = [c for c in FULL_FEATURES if c not in df_s.columns]
        high_nan  = [c for c in FULL_FEATURES
                     if c in df_s.columns and df_s[c].isna().mean() > 0.65]

        trn = df_s[(df_s['日付_num'] >= 130101) & (df_s['日付_num'] < 220101)]
        val = df_s[(df_s['日付_num'] >= 220101) & (df_s['日付_num'] <= 221231)]
        oos = df_s[df_s['日付_num'] >= 230101].copy()
        oos_2324 = oos[oos['日付_num'] < 250101]
        oos_2025 = oos[(oos['日付_num'] >= 250101) & (oos['日付_num'] < 260101)]
        oos_2026 = oos[oos['日付_num'] >= 260101]

        if len(trn) < 300 or len(val) < 30:
            print(f'  [{seg_key}] データ不足スキップ')
            continue

        print(f'\n[{seg_key}] {len(valid)}特徴量 (欠{len(missing)} 高NaN{len(high_nan)})'
              f'  trn:{len(trn)} val:{len(val)} oos:{len(oos)}'
              f'  [2026:{len(oos_2026)}R]')
        if missing:
            print(f'  欠: {missing}')
        if high_nan:
            print(f'  高NaN(>65%): {high_nan}')

        log_result({'event': 'seg_start', 'seg': seg_key,
                    'n_valid': len(valid), 'valid': valid,
                    'missing': missing, 'high_nan': high_nan,
                    'n_trn': len(trn), 'n_val': len(val),
                    'n_oos': len(oos), 'n_2026': len(oos_2026)})

        X_tr, y_tr, gs_tr, n_tr, nr_tr, scaler, *_ = prepare(
            trn, valid, top_idx=None, top_idx3=None, fit=True)
        X_va, y_va, gs_va, n_va, *_ = prepare(
            val, valid, scaler=scaler, top_idx=None, top_idx3=None)

        seg_results = []
        best_alpha, best_nll = None, np.inf

        for alpha in ALPHAS:
            t0 = time.time()
            beta, val_nll = adam_opt(X_tr, y_tr, gs_tr, n_tr, nr_tr,
                                      X_va, y_va, gs_va, n_va, alpha)
            elapsed = time.time() - t0

            r2324 = score_and_roi(oos_2324, valid, scaler, beta)
            r2025 = score_and_roi(oos_2025, valid, scaler, beta)
            r2026 = score_and_roi(oos_2026, valid, scaler, beta)

            v303 = V303_BASELINE.get(seg_key, float('nan'))
            diff_2324 = r2324['roi'] - v303
            print(f'  α={alpha:.1f} valNLL={val_nll:.5f} '
                  f'2324={r2324["roi"]:+.4f}({r2324["n"]}R,Δ{diff_2324:+.4f}) '
                  f'2025={r2025["roi"]:+.4f}({r2025["n"]}R) '
                  f'2026={r2026["roi"]:+.4f}({r2026["n"]}R) [{elapsed:.0f}s]')

            rec = {'event': 'alpha_result', 'seg': seg_key, 'alpha': alpha,
                   'val_nll': val_nll, 'elapsed': elapsed,
                   'roi_2324': r2324['roi'], 'n_2324': r2324['n'],
                   'roi_2025': r2025['roi'], 'n_2025': r2025['n'],
                   'roi_2026': r2026['roi'], 'n_2026': r2026['n']}
            log_result(rec)
            seg_results.append(rec)

            if val_nll < best_nll:
                best_nll, best_alpha = val_nll, alpha

        log_result({'event': 'seg_best', 'seg': seg_key,
                    'best_alpha': best_alpha, 'best_val_nll': best_nll})
        print(f'  → best α={best_alpha} (val NLL={best_nll:.5f})')
        all_results[seg_key] = seg_results

    # Summary
    print('\n=== feature_search_v6 サマリー ===')
    print(f'{"seg":12} {"α":5} {"2324":>10} {"2025":>10} {"2026":>10}  valNLL')
    for seg_key, seg_results in all_results.items():
        best = min(seg_results, key=lambda x: x['val_nll'])
        v303 = V303_BASELINE.get(seg_key, float('nan'))
        print(f'{seg_key:12} {best["alpha"]:5.1f}'
              f' {best["roi_2324"]:+10.4f}'
              f' {best["roi_2025"]:+10.4f}'
              f' {best["roi_2026"]:+10.4f}'
              f'  {best["val_nll"]:.5f}'
              f'  [v303_base={v303:+.4f}]')

    log_result({'event': 'done', 'ts': time.time()})
    print('\n=== 完了 ===')


if __name__ == '__main__':
    main()
