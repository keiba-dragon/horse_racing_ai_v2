# coding: utf-8
"""
eval_v3_weights.py
models/v3/ の clogit + lgbm 両スコアを使い、
セグメント別に clogit_w を 0.0→1.0 でスイープして最適ブレンドを探索する。

出力: 各セグメント × 各 clogit_w × 各期間 の ROI テーブル
"""
import sys, os, pickle
import numpy as np
import pandas as pd

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from save_conditional_logit import (
    prepare, segment_softmax, BASE_DIR, DATA_FILE
)
from sklearn.isotonic import IsotonicRegression
from save_v3 import (
    add_computed_features, race_normalize, calc_roi,
    CLOGIT_FEATS, _lgbm_feats, SEGMENTS
)

MODEL_DIR = os.path.join(BASE_DIR, 'models', 'v3')
WEIGHTS   = [round(w * 0.1, 1) for w in range(0, 11)]   # 0.0 ~ 1.0


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


def score_oos(df_eval, beta, scaler, iso, clogit_feats,
              lgbm_model, lgbm_feats):
    """clogit_prob と lgbm_prob を付けた DataFrame を返す（race_id ソート済み）。"""
    valid_c = [c for c in clogit_feats if c in df_eval.columns]
    X_c, _, gs, n_total, *_ = prepare(df_eval, valid_c, scaler=scaler,
                                       top_idx=None, top_idx3=None)
    raw_p   = segment_softmax(X_c @ beta, gs, n_total)
    calib_p = iso.predict(raw_p)
    clogit_p = race_normalize(calib_p, gs, n_total)

    df_s = df_eval.sort_values('race_id').reset_index(drop=True)
    valid_l = [c for c in lgbm_feats if c in df_s.columns]
    X_l = df_s[valid_l].astype(float).fillna(0).values
    lgbm_raw = lgbm_model.predict_proba(X_l)[:, 1]
    lgbm_p   = race_normalize(lgbm_raw, gs, n_total)

    df_s['clogit_prob'] = clogit_p
    df_s['lgbm_prob']   = lgbm_p
    df_s['gs_idx']      = np.repeat(np.arange(len(gs)), np.diff(np.append(gs, n_total)))
    return df_s, gs, n_total


def main():
    print(f'読み込み: {DATA_FILE}')
    df = load_data()
    print(f'有効行: {len(df):,}\n')

    # 全体集計用: weight → period → top1 rows list
    global_oos = {w: {p: [] for p in ['2324', '2025', '2026']} for w in WEIGHTS}

    for surf, dist_band in SEGMENTS:
        seg_key = f'{surf}_{dist_band}'
        seg_dir = os.path.join(MODEL_DIR, seg_key)
        clogit_path = os.path.join(seg_dir, 'clogit.pkl')
        lgbm_path   = os.path.join(seg_dir, 'lgbm.pkl')

        if not (os.path.exists(clogit_path) and os.path.exists(lgbm_path)):
            print(f'[{seg_key}] モデルなし → スキップ')
            continue

        with open(clogit_path, 'rb') as f:
            cp = pickle.load(f)
        with open(lgbm_path, 'rb') as f:
            lp = pickle.load(f)

        beta, scaler, iso = cp['beta'], cp['scaler'], cp['iso']
        clogit_feats = cp['feat_cols']
        lgbm_feats   = lp['feat_cols']
        lgbm_model   = lp['model']

        df_s = df[(df['surface'] == surf) & (df['dist_band'] == dist_band)].copy()
        oos  = df_s[df_s['日付_num'] >= 230101].copy()
        parts = {
            '2324': oos[oos['日付_num'] < 250101],
            '2025': oos[(oos['日付_num'] >= 250101) & (oos['日付_num'] < 260101)],
            '2026': oos[oos['日付_num'] >= 260101],
        }

        # 各期間を一度だけスコアリング
        scored_parts = {}
        for period, df_p in parts.items():
            if len(df_p) == 0:
                continue
            scored_df, gs, n_total = score_oos(
                df_p, beta, scaler, iso, clogit_feats, lgbm_model, lgbm_feats)
            scored_parts[period] = (scored_df, gs, n_total)

        # ── ヘッダ表示
        hdr = f'clogit_w  ' + '  '.join(
            f'{p}({len(scored_parts[p][0])//len(scored_parts[p][0].race_id.unique()):>3}R/race?)'
            if p in scored_parts else f'{p}(なし)'
            for p in ['2324', '2025', '2026']
        )
        print(f'[{seg_key}]')
        period_labels = {
            '2324': f"2324({len(scored_parts['2324'][0].race_id.unique())}R)" if '2324' in scored_parts else '2324(なし)',
            '2025': f"2025({len(scored_parts['2025'][0].race_id.unique())}R)" if '2025' in scored_parts else '2025(なし)',
            '2026': f"2026({len(scored_parts['2026'][0].race_id.unique())}R)" if '2026' in scored_parts else '2026(なし)',
        }
        print(f'  {"clogit_w":>9}  {period_labels["2324"]:>12}  {period_labels["2025"]:>12}  {period_labels["2026"]:>12}')

        for w in WEIGHTS:
            lw = round(1.0 - w, 1)
            row_parts = []
            for period in ['2324', '2025', '2026']:
                if period not in scored_parts:
                    row_parts.append('        ')
                    continue
                scored_df, gs, n_total = scored_parts[period]
                final_p = w * scored_df['clogit_prob'].values + lw * scored_df['lgbm_prob'].values
                scored_df = scored_df.copy()
                scored_df['final_prob'] = final_p
                scored_df['rank_model'] = scored_df.groupby('race_id')['final_prob'].rank(
                    ascending=False, method='first')
                top1 = scored_df[scored_df['rank_model'] == 1].copy()
                roi, wins = calc_roi(top1)
                row_parts.append(f'{roi:+.4f}')
                global_oos[w][period].append(top1)

            print(f'  clogit={w:.1f}  {row_parts[0]:>12}  {row_parts[1]:>12}  {row_parts[2]:>12}')
        print()

    # ── 全体集計
    print('=== 全体 (全セグメント合計) ===')
    print(f'  {"clogit_w":>9}  {"2324":>12}  {"2025":>12}  {"2026":>12}')
    for w in WEIGHTS:
        row = []
        for period in ['2324', '2025', '2026']:
            tops = global_oos[w][period]
            if not tops:
                row.append('        ')
                continue
            combined = pd.concat(tops, ignore_index=True)
            roi, _ = calc_roi(combined)
            row.append(f'{roi:+.4f}')
        print(f'  clogit={w:.1f}  {row[0]:>12}  {row[1]:>12}  {row[2]:>12}')


if __name__ == '__main__':
    main()
