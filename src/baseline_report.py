# coding: utf-8
"""
baseline_report.py
v303 と final_model をセグメント別×期間別に比較する現状マップ
"""
import sys, os, pickle, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from save_conditional_logit import prepare, segment_softmax, BASE_DIR, DATA_FILE
from save_v3 import add_computed_features, calc_roi

PERIODS = {
    '2324': (230101, 250101),
    '2025': (250101, 260101),
    '2026': (260101, 999999),
}
SEGMENTS = [('芝', '短距離'), ('芝', '中距離'), ('芝', '長距離'),
            ('ダ', '短距離'), ('ダ', '中長距離')]


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
    return df


# ── v303 評価 ──────────────────────────────────────────────────────────────
def eval_v303(df):
    model_dir = os.path.join(BASE_DIR, 'models', 'exp_v303_best')
    results = {}
    for surf, dist_band in SEGMENTS:
        seg_key = f'{surf}_{dist_band}'
        pkl = os.path.join(model_dir, f'{seg_key}.pkl')
        if not os.path.exists(pkl):
            continue
        with open(pkl, 'rb') as f:
            seg = pickle.load(f)
        beta, scaler, feats, iso = seg['beta'], seg['scaler'], seg['feat_cols'], seg['iso']

        df_s = df[(df['surface'] == surf) & (df['dist_band'] == dist_band)].copy()
        oos  = df_s[df_s['日付_num'] >= 230101].copy()

        seg_res = {}
        for pname, (lo, hi) in PERIODS.items():
            df_p = oos[(oos['日付_num'] >= lo) & (oos['日付_num'] < hi)]
            if len(df_p) == 0:
                continue
            valid = [c for c in feats if c in df_p.columns]
            X, _, gs, n, *_ = prepare(df_p, valid, scaler=scaler, top_idx=None, top_idx3=None)
            scored = df_p.sort_values('race_id').reset_index(drop=True)
            probs = segment_softmax(X @ beta, gs, n)
            scored['prob'] = probs
            scored['rank'] = scored.groupby('race_id')['prob'].rank(ascending=False, method='first')
            top1 = scored[scored['rank'] == 1]
            roi, wins = calc_roi(top1)
            seg_res[pname] = (roi, len(top1))
        results[seg_key] = seg_res
    return results


# ── final_model 評価 ────────────────────────────────────────────────────────
def eval_final(df):
    pkl_path = os.path.join(BASE_DIR, 'models', 'roi_model.pkl')
    if not os.path.exists(pkl_path):
        return {}
    with open(pkl_path, 'rb') as f:
        pkg = pickle.load(f)

    arts      = pkg['artifacts']
    feat_cols = pkg['feat_cols']
    f_other   = pkg.get('factor_other', 0.16)

    # 欠損列を 0 埋め（クラス変化など parquet に存在しない列）
    df = df.copy()
    for col in feat_cols:
        if col not in df.columns:
            df[col] = 0.0

    oos = df[df['日付_num'] >= 230101].copy()

    # セグメント定義（final_modelは芝/ダ 2分類のみ）
    dm = df['dist_m']
    results_surf = {}
    for surf in ['芝', 'ダ']:
        art = arts[surf]
        df_s = oos[oos['surface'] == surf].copy()
        if len(df_s) == 0:
            continue
        valid = [c for c in feat_cols if c in df_s.columns]
        df_sorted = df_s.sort_values('race_id').reset_index(drop=True)
        X_raw = df_sorted[valid].astype(float).fillna(0).values
        X_sc  = art['scaler'].transform(X_raw)
        parts_list = [X_sc]
        if art.get('top_idx') is not None:
            X2 = art['poly2'].transform(X_sc[:, art['top_idx']])
            X2 = X2[:, art['top_idx'].shape[0]:]
            X2 = art['inter_scaler2'].transform(X2)
            parts_list.append(X2)
        if art.get('top_idx3') is not None:
            X3 = art['poly3'].transform(X_sc[:, art['top_idx3']])
            from sklearn.preprocessing import PolynomialFeatures
            n2 = art['top_idx3'].shape[0]
            n2way = n2*(n2-1)//2
            X3 = X3[:, n2 + n2way:]
            X3 = art['inter_scaler3'].transform(X3)
            parts_list.append(X3)
        X_full = np.hstack(parts_list)
        raw_score = X_full @ art['coef']
        calib = art['isotonic'].predict(raw_score)

        from save_conditional_logit import get_group_starts
        gs = get_group_starts(df_sorted['race_id'].values)
        sizes = np.diff(np.append(gs, len(df_sorted)))

        # 新聞用EVスコア（non-maidenのみ）
        odds = pd.to_numeric(df_sorted['単勝オッズ'], errors='coerce').fillna(99).values
        ev_score = calib - f_other * (1.0 / np.clip(odds, 1.0, None))
        df_sorted['ev_score'] = ev_score

        seg_res = {}
        for pname, (lo, hi) in PERIODS.items():
            mask = (df_sorted['日付_num'] >= lo) & (df_sorted['日付_num'] < hi)
            df_p = df_sorted[mask].copy()
            if len(df_p) == 0:
                continue
            df_p['rank'] = df_p.groupby('race_id')['ev_score'].rank(ascending=False, method='first')
            top1 = df_p[df_p['rank'] == 1]
            roi, wins = calc_roi(top1)
            seg_res[pname] = (roi, len(top1))
        results_surf[surf] = seg_res

    # セグメント別（dist_band × surface）に分解
    results = {}
    for surf, dist_band in SEGMENTS:
        seg_key = f'{surf}_{dist_band}'
        art = arts[surf]
        df_s = oos[(oos['surface'] == surf)].copy()
        dm_s = df_s['dist_m']
        if surf == '芝':
            if dist_band == '短距離':
                df_s = df_s[dm_s <= 1400]
            elif dist_band == '中距離':
                df_s = df_s[(dm_s > 1400) & (dm_s <= 2000)]
            else:
                df_s = df_s[dm_s > 2000]
        else:
            if dist_band == '短距離':
                df_s = df_s[dm_s <= 1400]
            else:
                df_s = df_s[dm_s > 1400]

        if len(df_s) == 0:
            continue
        valid = [c for c in feat_cols if c in df_s.columns]
        df_sorted = df_s.sort_values('race_id').reset_index(drop=True)
        X_raw = df_sorted[valid].astype(float).fillna(0).values
        X_sc  = art['scaler'].transform(X_raw)
        parts_list = [X_sc]
        if art.get('top_idx') is not None:
            X2 = art['poly2'].transform(X_sc[:, art['top_idx']])
            X2 = X2[:, art['top_idx'].shape[0]:]
            X2 = art['inter_scaler2'].transform(X2)
            parts_list.append(X2)
        if art.get('top_idx3') is not None:
            X3 = art['poly3'].transform(X_sc[:, art['top_idx3']])
            n3 = art['top_idx3'].shape[0]
            n2way = n3*(n3-1)//2
            X3 = X3[:, n3 + n2way:]
            X3 = art['inter_scaler3'].transform(X3)
            parts_list.append(X3)
        X_full = np.hstack(parts_list)
        raw_score = X_full @ art['coef']
        calib = art['isotonic'].predict(raw_score)
        odds = pd.to_numeric(df_sorted['単勝オッズ'], errors='coerce').fillna(99).values
        ev_score = calib - f_other * (1.0 / np.clip(odds, 1.0, None))
        df_sorted['ev_score'] = ev_score

        seg_res = {}
        for pname, (lo, hi) in PERIODS.items():
            mask = (df_sorted['日付_num'] >= lo) & (df_sorted['日付_num'] < hi)
            df_p = df_sorted[mask].copy()
            if len(df_p) == 0:
                continue
            df_p['rank'] = df_p.groupby('race_id')['ev_score'].rank(ascending=False, method='first')
            top1 = df_p[df_p['rank'] == 1]
            roi, wins = calc_roi(top1)
            seg_res[pname] = (roi, len(top1))
        results[seg_key] = seg_res
    return results


def print_table(title, results):
    print(f'\n{"="*60}')
    print(f' {title}')
    print(f'{"="*60}')
    print(f'{"セグメント":<14} {"2324ROI":>9} {"2324R":>6} {"2025ROI":>9} {"2025R":>6} {"2026ROI":>9} {"2026R":>6}')
    print('-'*60)
    all_tops = {p: [] for p in PERIODS}
    for seg_key, seg_res in results.items():
        row = f'{seg_key:<14}'
        for pname in ['2324', '2025', '2026']:
            if pname in seg_res:
                roi, n = seg_res[pname]
                row += f' {roi:>+9.2%} {n:>6}'
                all_tops[pname].append((roi, n))
            else:
                row += f' {"":>9} {"":>6}'
        print(row)
    print('-'*60)
    # 合計行
    row = f'{"合計":<14}'
    for pname in ['2324', '2025', '2026']:
        items = all_tops[pname]
        if items:
            total_n = sum(n for _, n in items)
            # レース数加重平均
            weighted = sum(roi * n for roi, n in items) / total_n
            row += f' {weighted:>+9.2%} {total_n:>6}'
        else:
            row += f' {"":>9} {"":>6}'
    print(row)


def main():
    print(f'読み込み: {DATA_FILE}')
    df = load_data()
    print(f'有効行: {len(df):,}')

    print('\n--- v303 評価中 ---')
    r303 = eval_v303(df)
    print_table('v303 (純clogit, 5セグメント)', r303)

    print('\n--- final_model 評価中 ---')
    rfin = eval_final(df)
    print_table('final_model (320特徴+交差項+EV)', rfin)

    # 差分サマリ
    print(f'\n{"="*60}')
    print(' 差分: final_model - v303  (+ = final_modelが良い)')
    print(f'{"="*60}')
    print(f'{"セグメント":<14} {"Δ2324":>9} {"Δ2025":>9} {"Δ2026":>9}')
    print('-'*40)
    for seg_key in [f'{s}_{d}' for s, d in SEGMENTS]:
        r3 = r303.get(seg_key, {})
        rf = rfin.get(seg_key, {})
        row = f'{seg_key:<14}'
        for p in ['2324', '2025', '2026']:
            if p in r3 and p in rf:
                delta = rf[p][0] - r3[p][0]
                row += f' {delta:>+9.2%}'
            else:
                row += f' {"—":>9}'
        print(row)


if __name__ == '__main__':
    main()
