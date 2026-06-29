# coding: utf-8
"""
eval_v303.py - 保存済み exp_v303_best を現在の parquet で再評価
2023-24 / 2025 / 2026 ホールドアウト ROI を計算
"""
import sys, os, pickle
import numpy as np
import pandas as pd

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from save_conditional_logit import (
    prepare, segment_softmax, BASE_DIR, DATA_FILE
)

MODEL_DIR = os.path.join(BASE_DIR, 'models', 'exp_v303_best')
SEGMENTS = [('芝', '短距離'), ('芝', '中距離'), ('芝', '長距離'),
            ('ダ', '短距離'), ('ダ', '中長距離')]


def add_computed_features(df):
    interval = (pd.to_numeric(df['間隔'], errors='coerce')
                if '間隔' in df.columns else pd.Series(np.nan, index=df.index))
    df['間隔_長_flag'] = (interval >= 60).astype(float)
    df['間隔_短_flag'] = (interval <= 14).astype(float)
    da_r  = (pd.to_numeric(df['種牡馬_ダ_勝率'], errors='coerce')
             if '種牡馬_ダ_勝率' in df.columns else pd.Series(np.nan, index=df.index))
    all_r = (pd.to_numeric(df['種牡馬_勝率'], errors='coerce')
             if '種牡馬_勝率' in df.columns else pd.Series(np.nan, index=df.index))
    df['血統_ダ優位度'] = da_r - all_r
    return df


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


def calc_roi(top1):
    if len(top1) == 0:
        return float('nan'), 0
    won = top1['着順_num'] == 1
    odds = pd.to_numeric(top1['単勝オッズ'], errors='coerce')
    return float((odds[won] * 100).sum() / (len(top1) * 100) - 1), int(won.sum())


def main():
    print(f'読み込み: {DATA_FILE}')
    df = load_data()
    print(f'有効行: {len(df):,}\n')

    all_top1 = {k: [] for k in ['2324', '2025', '2026']}

    for surf, dist_band in SEGMENTS:
        seg_key = f'{surf}_{dist_band}'
        pkl_path = os.path.join(MODEL_DIR, f'{seg_key}.pkl')
        if not os.path.exists(pkl_path):
            print(f'[{seg_key}] pkl なし → スキップ')
            continue

        with open(pkl_path, 'rb') as f:
            seg = pickle.load(f)

        beta    = seg['beta']
        scaler  = seg['scaler']
        feats   = seg['feat_cols']
        iso     = seg['iso']

        df_s = df[(df['surface'] == surf) & (df['dist_band'] == dist_band)].copy()
        oos = df_s[df_s['日付_num'] >= 230101].copy()
        parts = {
            '2324': oos[oos['日付_num'] < 250101],
            '2025': oos[(oos['日付_num'] >= 250101) & (oos['日付_num'] < 260101)],
            '2026': oos[oos['日付_num'] >= 260101],
        }

        print(f'[{seg_key}] {len(feats)}特徴量')
        for period, df_p in parts.items():
            if len(df_p) == 0:
                print(f'  {period}: データなし')
                continue
            valid = [c for c in feats if c in df_p.columns]
            Xo, yo, gso, no, *_ = prepare(df_p, valid, scaler=scaler,
                                           top_idx=None, top_idx3=None)
            scored = df_p.sort_values('race_id').reset_index(drop=True)
            probs = segment_softmax(Xo @ beta, gso, no)
            scored['model_prob'] = probs
            scored['rank_model'] = scored.groupby('race_id')['model_prob'].rank(
                ascending=False, method='first')
            top1 = scored[scored['rank_model'] == 1].copy()
            roi, wins = calc_roi(top1)
            n = len(top1)
            print(f'  {period}: {n}R  ROI={roi:+.4f}  勝率={wins/n:.1%}')
            all_top1[period].append(top1)

    print('\n=== 全体 ===')
    for period, tops in all_top1.items():
        if not tops:
            continue
        combined = pd.concat(tops, ignore_index=True)
        roi, wins = calc_roi(combined)
        n = len(combined)
        print(f'{period}: {n}R  ROI={roi:+.4f}  勝率={wins/n:.1%}')


if __name__ == '__main__':
    main()
