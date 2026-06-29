# coding: utf-8
"""
eval_da_short_strata.py - strata_clogit.pkl の ダ_短距離 ストラタムをOOSで個別評価
"""
import sys, os, pickle
import numpy as np
import pandas as pd

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from save_conditional_logit import prepare, segment_softmax, BASE_DIR, DATA_FILE
from save_v3 import add_computed_features

MODEL_DIR = os.path.join(BASE_DIR, 'models')


def load_da_short():
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
    dm = pd.to_numeric(df['距離'].astype(str).str.extract(r'(\d+)')[0], errors='coerce')
    df = df[(df['surface'] == 'ダ') & (dm <= 1400)].copy()
    df['dist_m'] = dm[df.index]
    if 'クラス_rank' in df.columns:
        df = df[df['クラス_rank'] != 1.0].copy()
    df = add_computed_features(df)
    baba_map = {'良': 0, '稍重': 1, '重': 2, '不良': 3}
    for col in df.columns:
        if '馬場状態' in col:
            df[col] = df[col].map(baba_map)
    return df


def calc_roi_top1(top1):
    won  = top1['着順_num'] == 1
    odds = pd.to_numeric(top1['単勝オッズ'], errors='coerce')
    n    = len(top1)
    if n == 0:
        return float('nan'), 0, 0
    roi  = (odds[won] * 100).sum() / (n * 100) - 1
    return roi, n, won.sum()


def eval_stratum(art, df, feats_old):
    beta       = art['coef']
    scaler_old = art['scaler']
    feats      = art['feat_cols']

    # 欠損列をNaNで補完
    df2 = df.copy()
    for fc in feats:
        if fc not in df2.columns:
            df2[fc] = np.nan

    valid = [c for c in feats if c in df2.columns]
    try:
        X_p, _, gs_p, n_p, *_ = prepare(
            df2, valid, scaler=scaler_old,
            poly2=art.get('poly2'),
            inter_scaler2=art.get('inter_scaler2'),
            top_idx=art.get('top_idx'),
            poly3=art.get('poly3'),
            inter_scaler3=art.get('inter_scaler3'),
            top_idx3=art.get('top_idx3'),
        )
    except Exception as e:
        print(f'  prepare エラー: {e}')
        return None

    scored = df2.sort_values('race_id').reset_index(drop=True)
    scored['prob'] = segment_softmax(X_p @ beta, gs_p, n_p)
    scored['rank'] = scored.groupby('race_id')['prob'].rank(ascending=False, method='first')
    return scored


def main():
    print('=== strata_clogit.pkl  ダ_短距離 個別評価 ===')
    print()

    with open(os.path.join(MODEL_DIR, 'strata_clogit.pkl'), 'rb') as f:
        pkg = pickle.load(f)

    art   = pkg['artifacts']['ダ_短距離']
    feats = art['feat_cols']
    print(f'特徴量数: {len(feats)}個')
    print(f'パッケージ total_oos_roi: {pkg.get("total_oos_roi", "?")}')

    print('\nデータ読み込み中...')
    df = load_da_short()
    print(f'ダート短距離 全データ: {len(df):,}行')

    oos = df[df['日付_num'] >= 230101].copy()
    periods = {
        '2023-24': oos[oos['日付_num'] < 250101],
        '2025':    oos[(oos['日付_num'] >= 250101) & (oos['日付_num'] < 260101)],
        '2026':    oos[oos['日付_num'] >= 260101],
    }

    print()
    print(f'  {"期間":<10} {"R数":>6} {"勝利":>5} {"勝率":>7} {"ROI":>10}')
    print('  ' + '─' * 46)

    results = {}
    scored_all = None
    for period, df_p in periods.items():
        if len(df_p) == 0:
            continue
        scored = eval_stratum(art, df_p, feats)
        if scored is None:
            continue
        top1 = scored[scored['rank'] == 1]
        roi, n, w = calc_roi_top1(top1)
        wr = w / n if n > 0 else 0
        print(f'  {period:<10} {n:>6,} {w:>5}  {wr:>6.1%}  {roi:>+9.2%}')
        results[period] = (roi, n, w)

    # 25+26 combined
    r25, n25, _ = results.get('2025', (0, 1, 0))
    r26, n26, _ = results.get('2026', (0, 1, 0))
    n25, n26 = int(n25), int(n26)
    comb = (r25 * n25 + r26 * n26) / (n25 + n26) if (n25 + n26) > 0 else 0
    print('  ' + '─' * 46)
    print(f'  {"25+26合算":<10}                    {comb:>+9.2%}')

    print()
    print(f'【まとめ】')
    print(f'  ダ_短距離 ストラタム (strata_clogit.pkl)')
    print(f'  25+26 ROI = {comb:+.2%}')
    for p, (roi, n, w) in results.items():
        print(f'  {p}: {roi:+.2%} ({n}R)')


if __name__ == '__main__':
    main()
