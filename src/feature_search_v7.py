# coding: utf-8
"""
feature_search_v7.py - v303特徴量 + レース内平均NaN補完 + オッズ調整スイープ
apply_race_impute(alpha_impute) を全セグメントで試す
alpha_impute=0.0: レース内平均（オッズ調整なし）
alpha_impute>0 : オッズ相対差で補完値を調整（低人気馬は平均より低く評価）
出力: logs/feature_search_v7.jsonl
"""
import sys, os, json, time
import numpy as np
import pandas as pd

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from save_conditional_logit import (
    prepare, segment_softmax, apply_race_impute,
    BASE_DIR, DATA_FILE, LR, N_EPOCHS, PATIENCE
)
from sklearn.isotonic import IsotonicRegression

LOG_FILE = os.path.join(BASE_DIR, 'logs', 'feature_search_v7.jsonl')
os.makedirs(os.path.join(BASE_DIR, 'logs'), exist_ok=True)

IMPUTE_ALPHAS = [0.0, 0.5, 1.0, 2.0]

BASE_FEATURES = [
    '1走前_タイム指数', '1走前_上り3F', '前走着差タイム', '1走前_RPCI',
    '近5走_タイム指数平均', '近5走_タイム指数_max', 'タイム指数_近3走_slope',
    '1走前_クラス調整着順', '近5走_クラス調整_平均着順',
    '馬番', 'コース枠_r200_勝率',
    '騎手コース_r100_勝率', '騎手変更', '調教師コース_r100_勝率',
    'コース脚質_r200_勝率', '1走前_脚質_num', '種牡馬_勝率', '種牡馬_ダ_勝率',
    '斤量', '芝ダ転向', '間隔',
]

SEG_FEATS = {
    '芝_短距離': [
        '1走前_タイム指数', '1走前_上り3F', '前走着差タイム', '1走前_RPCI',
        '近5走_タイム指数平均', '近5走_タイム指数_max', 'タイム指数_近3走_slope',
        '1走前_クラス調整着順', '近5走_クラス調整_平均着順',
        '馬番',
        '騎手コース_r100_勝率', '調教師コース_r100_勝率',
        'コース脚質_r200_勝率', '1走前_脚質_num', '種牡馬_勝率', '種牡馬_ダ_勝率',
        '斤量', '芝ダ転向', '間隔', '1走前_3角',
    ],
    '芝_中距離': BASE_FEATURES + ['馬体重', '間隔_短_flag', '血統_ダ優位度', '馬体重増減'],
    '芝_長距離': list(BASE_FEATURES),
    'ダ_短距離': [
        '1走前_タイム指数', '1走前_上り3F', '前走着差タイム', '1走前_RPCI',
        '近5走_タイム指数平均', '近5走_タイム指数_max', 'タイム指数_近3走_slope',
        '1走前_クラス調整着順', '近5走_クラス調整_平均着順',
        '馬番', 'コース枠_r200_勝率',
        '騎手コース_r100_勝率', '騎手変更', '調教師コース_r100_勝率',
        'コース脚質_r200_勝率', '1走前_脚質_num', '種牡馬_勝率', '種牡馬_ダ_勝率',
        '斤量', '間隔',
        '馬体重増減', '展開フィット_v2', '乗替り_近走不振', '間隔_長_flag',
    ],
    'ダ_中長距離': [
        '1走前_タイム指数', '1走前_上り3F', '前走着差タイム', '1走前_RPCI',
        '近5走_タイム指数平均', '近5走_タイム指数_max',
        '1走前_クラス調整着順', '近5走_クラス調整_平均着順',
        '馬番', 'コース枠_r200_勝率',
        '騎手コース_r100_勝率', '騎手変更', '調教師コース_r100_勝率',
        '1走前_脚質_num', '種牡馬_勝率', '種牡馬_ダ_勝率',
        '斤量', '間隔',
        '間隔_長_flag', '1走前_上3F地点差',
    ],
}

SEGMENTS = [('芝', '短距離'), ('芝', '中距離'), ('芝', '長距離'),
            ('ダ', '短距離'), ('ダ', '中長距離')]

V303_2026 = {
    '芝_短距離':  -0.2455,
    '芝_中距離':  -0.2047,
    '芝_長距離':  -0.3214,
    'ダ_短距離':  -0.0470,
    'ダ_中長距離': float('nan'),
}
V303_2025 = {
    '芝_短距離':  -0.2377,
    '芝_中距離':  -0.2017,
    '芝_長距離':  -0.2237,
    'ダ_短距離':  -0.2641,
    'ダ_中長距離': float('nan'),
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


def _loss_grad(beta, X, y, gs, n, nr, alpha=1.0):
    probs = segment_softmax(X @ beta, gs, n)
    log_lik = np.sum(y * np.log(np.clip(probs, 1e-15, 1.0)))
    res = y - probs
    loss = (-log_lik + alpha * np.sum(beta**2)) / nr
    grad = (-(X.T @ res) + 2 * alpha * beta) / nr
    return loss, grad


def adam_opt(X_tr, y_tr, gs_tr, n_tr, nr_tr,
             X_va, y_va, gs_va, n_va):
    d = X_tr.shape[1]
    beta, m, v = np.zeros(d), np.zeros(d), np.zeros(d)
    b1, b2, eps_a = 0.9, 0.999, 1e-8
    t, best_val, best_beta, no_imp = 0, np.inf, np.zeros(d), 0
    for epoch in range(1, N_EPOCHS + 1):
        _, grad = _loss_grad(beta, X_tr, y_tr, gs_tr, n_tr, nr_tr, alpha=1.0)
        t += 1
        m = b1 * m + (1 - b1) * grad
        v = b2 * v + (1 - b2) * grad ** 2
        beta -= LR * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps_a)
        if epoch % 10 == 0:
            vl = _nll(beta, X_va, y_va, gs_va, n_va)
            if vl < best_val:
                best_val, best_beta, no_imp = vl, beta.copy(), 0
            else:
                no_imp += 1
            if no_imp >= PATIENCE // 10:
                break
    return best_beta, float(best_val)


def score_and_roi(df_sub, valid, scaler, beta):
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
    return {'roi': float((odds[won] * 100).sum() / (len(top1) * 100) - 1),
            'n': int(len(top1)), 'wins': int(won.sum())}


def main():
    df = load_data()
    log_result({'event': 'start', 'ts': time.time(),
                'impute_alphas': IMPUTE_ALPHAS})

    summary = []

    for surf, dist_band in SEGMENTS:
        seg_key = f'{surf}_{dist_band}'
        feats = SEG_FEATS[seg_key]
        df_s = df[(df['surface'] == surf) & (df['dist_band'] == dist_band)].copy()

        valid = [c for c in feats
                 if c in df_s.columns and df_s[c].isna().mean() <= 0.65]
        missing = [c for c in feats if c not in df_s.columns]

        trn_raw = df_s[(df_s['日付_num'] >= 130101) & (df_s['日付_num'] < 220101)]
        val_raw = df_s[(df_s['日付_num'] >= 220101) & (df_s['日付_num'] <= 221231)]
        oos_raw = df_s[df_s['日付_num'] >= 230101].copy()
        oos_2324_raw = oos_raw[oos_raw['日付_num'] < 250101]
        oos_2025_raw = oos_raw[(oos_raw['日付_num'] >= 250101) & (oos_raw['日付_num'] < 260101)]
        oos_2026_raw = oos_raw[oos_raw['日付_num'] >= 260101]

        if len(trn_raw) < 300 or len(val_raw) < 30:
            print(f'  [{seg_key}] データ不足スキップ')
            continue

        print(f'\n[{seg_key}] {len(valid)}特徴量'
              f'  trn:{len(trn_raw)} val:{len(val_raw)} oos:{len(oos_raw)}'
              f'  [2026:{len(oos_2026_raw)}R]')
        if missing:
            print(f'  欠: {missing}')

        log_result({'event': 'seg_start', 'seg': seg_key,
                    'n_valid': len(valid), 'valid': valid, 'missing': missing,
                    'n_trn': len(trn_raw), 'n_val': len(val_raw),
                    'n_2026': len(oos_2026_raw)})

        seg_rows = []
        best_alpha_imp, best_nll = None, np.inf

        for alpha_imp in IMPUTE_ALPHAS:
            t0 = time.time()

            # Apply race-level imputation separately to each split (no leakage)
            trn = apply_race_impute(trn_raw, alpha=alpha_imp)
            val = apply_race_impute(val_raw, alpha=alpha_imp)
            oos_2324 = apply_race_impute(oos_2324_raw, alpha=alpha_imp)
            oos_2025 = apply_race_impute(oos_2025_raw, alpha=alpha_imp)
            oos_2026 = apply_race_impute(oos_2026_raw, alpha=alpha_imp)

            X_tr, y_tr, gs_tr, n_tr, nr_tr, scaler, *_ = prepare(
                trn, valid, top_idx=None, top_idx3=None, fit=True)
            X_va, y_va, gs_va, n_va, *_ = prepare(
                val, valid, scaler=scaler, top_idx=None, top_idx3=None)

            beta, val_nll = adam_opt(X_tr, y_tr, gs_tr, n_tr, nr_tr,
                                      X_va, y_va, gs_va, n_va)
            elapsed = time.time() - t0

            r2324 = score_and_roi(oos_2324, valid, scaler, beta)
            r2025 = score_and_roi(oos_2025, valid, scaler, beta)
            r2026 = score_and_roi(oos_2026, valid, scaler, beta)

            v303_26 = V303_2026.get(seg_key, float('nan'))
            v303_25 = V303_2025.get(seg_key, float('nan'))
            d26 = r2026['roi'] - v303_26 if not np.isnan(v303_26) else float('nan')
            d25 = r2025['roi'] - v303_25 if not np.isnan(v303_25) else float('nan')

            print(f'  imp_α={alpha_imp:.1f} valNLL={val_nll:.5f}'
                  f'  2025={r2025["roi"]:+.4f}(Δ{d25:+.4f})'
                  f'  2026={r2026["roi"]:+.4f}(Δ{d26:+.4f})'
                  f'  [{elapsed:.0f}s]')

            rec = {'event': 'impute_result', 'seg': seg_key, 'alpha_imp': alpha_imp,
                   'val_nll': val_nll, 'elapsed': elapsed,
                   'roi_2324': r2324['roi'], 'n_2324': r2324['n'],
                   'roi_2025': r2025['roi'], 'n_2025': r2025['n'],
                   'roi_2026': r2026['roi'], 'n_2026': r2026['n'],
                   'v303_2025': v303_25, 'v303_2026': v303_26}
            log_result(rec)
            seg_rows.append(rec)

            if val_nll < best_nll:
                best_nll, best_alpha_imp = val_nll, alpha_imp

        log_result({'event': 'seg_best', 'seg': seg_key,
                    'best_alpha_imp': best_alpha_imp, 'best_val_nll': best_nll})
        print(f'  → best imp_α={best_alpha_imp} (val NLL={best_nll:.5f})')

        best_rec = min(seg_rows, key=lambda x: x['val_nll'])
        summary.append({'seg': seg_key, 'best_alpha_imp': best_alpha_imp,
                        'roi_2025': best_rec['roi_2025'],
                        'roi_2026': best_rec['roi_2026'],
                        'v303_2025': V303_2025.get(seg_key, float('nan')),
                        'v303_2026': V303_2026.get(seg_key, float('nan'))})

    # Final summary
    print('\n=== feature_search_v7 サマリー ===')
    print(f'{"seg":12} {"imp_α":6}  {"v303_2025":>10} {"v7_2025":>10} {"Δ2025":>8}'
          f'  {"v303_2026":>10} {"v7_2026":>10} {"Δ2026":>8}')
    for row in summary:
        seg = row['seg']
        d25 = row['roi_2025'] - row['v303_2025'] if not np.isnan(row['v303_2025']) else float('nan')
        d26 = row['roi_2026'] - row['v303_2026'] if not np.isnan(row['v303_2026']) else float('nan')
        print(f'{seg:12} {row["best_alpha_imp"]:6.1f}'
              f'  {row["v303_2025"]:+10.4f} {row["roi_2025"]:+10.4f} {d25:+8.4f}'
              f'  {row["v303_2026"]:+10.4f} {row["roi_2026"]:+10.4f} {d26:+8.4f}')

    log_result({'event': 'done', 'ts': time.time(), 'summary': summary})
    print('\n=== 完了 ===')


if __name__ == '__main__':
    main()
