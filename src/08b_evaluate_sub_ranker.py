import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import os
import pickle
import json
import re

OOS_START = 210101

def get_distance_band(dist):
    try:
        m = re.search(r'\d+', str(dist))
        if not m: return None
        d = int(m.group())
    except: return None
    if d <= 1400:   return '短距離'
    elif d <= 1800: return 'マイル'
    elif d <= 2200: return '中距離'
    else:           return '長距離'

def get_class_group(class_rank):
    try:
        r = int(float(class_rank))
    except: return None
    if np.isnan(float(class_rank)): return None
    if r == 1: return '新馬'
    elif r == 2: return '未勝利'
    elif r == 3: return '1勝'
    elif r == 4: return '2勝'
    elif r >= 5: return '3勝以上'
    return None

def deviation_score(values, mean, std):
    if std == 0:
        return pd.Series([50.0] * len(values), index=values.index)
    return 50 + 10 * (values - mean) / std

def calc_roi(bets_df, payout_col='単勝配当'):
    total_bets = len(bets_df)
    if total_bets == 0:
        return 0.0, 0, 0
    wins = bets_df[bets_df['target_win'] == 1]
    total_paid = wins[payout_col].sum()
    roi = total_paid / (total_bets * 100) - 1.0
    return roi, len(wins), total_bets

def main():
    print("--- サブモデル + サブランカー OOS評価 ---")
    print(f"OOS評価期間: {OOS_START}以降（2013–2020学習 / 2021+テスト）\n")

    base_dir     = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if 'src' in os.path.abspath(__file__) else '.'
    input_file   = os.path.join(base_dir, 'data', 'processed', 'all_venues_features.csv')
    sub_dir      = os.path.join(base_dir, 'models', 'submodel')
    ranker_dir   = os.path.join(base_dir, 'models', 'sub_ranker')

    with open(os.path.join(sub_dir, 'submodel_info.json'), 'r', encoding='utf-8') as f:
        sub_info = json.load(f)
    features     = sub_info['features']
    sub_models   = sub_info['models']

    with open(os.path.join(ranker_dir, 'sub_ranker_info.json'), 'r', encoding='utf-8') as f:
        ranker_info = json.load(f)
    trained_rankers = ranker_info.get('rankers', {})

    print(f"サブモデル: {len(sub_models)}グループ / サブランカー: {len(trained_rankers)}グループ")

    print("データを読み込んでいます...")
    df = pd.read_csv(input_file, low_memory=False)
    df['日付_num'] = pd.to_numeric(df['日付'], errors='coerce')
    if '着順_num' in df.columns:
        df['着順_num'] = pd.to_numeric(df['着順_num'], errors='coerce')
    else:
        df['着順_num'] = (df['着順'].astype(str)
            .str.translate(str.maketrans('０１２３４５６７８９', '0123456789'))
            .pipe(pd.to_numeric, errors='coerce'))
    df = df.dropna(subset=['日付_num', '着順_num'])
    df['target_win']   = (df['着順_num'] == 1).astype(int)
    df['target_place'] = (df['着順_num'] <= 3).astype(int)

    # グループキー（サブモデルと同じロジック）
    df['_surface']    = df['芝・ダ'].astype(str).str.strip()
    df['_dist_band']  = df['距離'].apply(get_distance_band)
    mask_da = (df['_surface'] == 'ダ') & (df['_dist_band'].isin(['中距離', '長距離']))
    df.loc[mask_da, '_dist_band'] = '中長距離'
    df['_class_group'] = df['クラス_rank'].apply(get_class_group) if 'クラス_rank' in df.columns else np.nan
    df['model_key'] = df['_surface'] + '_' + df['_dist_band'].astype(str) + '_' + df['_class_group'].astype(str)
    df = df[~df['model_key'].str.contains('None|nan', na=True)]

    if '単勝オッズ' in df.columns:
        df['単勝オッズ'] = pd.to_numeric(df['単勝オッズ'], errors='coerce')
        has_payout = df['単勝オッズ'].notna().sum() > 1000
        if has_payout:
            df['単勝配当'] = df['単勝オッズ'] * 100
            print(f"単勝オッズあり（{df['単勝オッズ'].notna().sum():,}件）")
    elif '単勝配当' in df.columns:
        df['単勝配当'] = pd.to_numeric(df['単勝配当'], errors='coerce')
        has_payout = df['単勝配当'].notna().sum() > 1000
    else:
        has_payout = False

    for col in features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df['race_id'] = df['日付_num'].astype(str) + '_' + df['開催'].astype(str) + '_' + df['レース名'].astype(str)

    # ────────────────────────────────────────────────
    # OOS予測
    # ────────────────────────────────────────────────
    all_test_rows = []

    for key in sub_models:
        g    = df[df['model_key'] == key].sort_values('日付_num').reset_index(drop=True)
        test = g[g['日付_num'] >= OOS_START].copy()
        if len(test) < 50:
            continue

        win_path = os.path.join(sub_dir, sub_models[key]['win'])
        if not os.path.exists(win_path):
            continue
        with open(win_path, 'rb') as f:
            m_win = pickle.load(f)
        wf = m_win.booster_.feature_name()
        test['prob_win'] = m_win.predict_proba(test[wf])[:, 1]

        stats  = sub_models[key].get('stats', {})
        w_mean = stats.get('win_mean', test['prob_win'].mean())
        w_std  = stats.get('win_std',  test['prob_win'].std())
        test['コース偏差値'] = deviation_score(test['prob_win'], w_mean, w_std).values

        test['レース内偏差値'] = np.nan
        for _, rdf in test.groupby('race_id'):
            if len(rdf) < 2: continue
            r_mean = rdf['prob_win'].mean()
            r_std  = rdf['prob_win'].std()
            test.loc[rdf.index, 'レース内偏差値'] = deviation_score(
                rdf['prob_win'], r_mean, r_std if r_std > 0 else 1
            ).values
        test['偏差値の差'] = test['レース内偏差値'] - test['コース偏差値']

        # サブランカースコア
        test['ランカースコア'] = np.nan
        if key in trained_rankers:
            rpath = os.path.join(ranker_dir, trained_rankers[key])
            if os.path.exists(rpath):
                with open(rpath, 'rb') as f:
                    ranker = pickle.load(f)
                test['ランカースコア'] = ranker.predict(test[features])

        # レース内ランカー順位
        test['ランカー順位'] = np.nan
        for _, rdf in test.groupby('race_id'):
            if rdf['ランカースコア'].isna().all(): continue
            ranked = rdf['ランカースコア'].rank(ascending=False, method='min').astype(int)
            test.loc[rdf.index, 'ランカー順位'] = ranked

        test['model_key'] = key
        all_test_rows.append(test)

    if not all_test_rows:
        print("テストデータが空です。")
        return

    all_test = pd.concat(all_test_rows, ignore_index=True)
    print(f"\nテストデータ: {len(all_test):,}件（{all_test['model_key'].nunique()}グループ）\n")

    # ────────────────────────────────────────────────
    # 1. 偏差値の差 × ROI
    # ────────────────────────────────────────────────
    print(f"{'='*70}")
    print(f" [1] バイナリモデル：偏差値の差 × 単勝ROI")
    print(f"{'='*70}")
    print(f"{'閾値':>6}  {'対象馬':>7}  {'的中数':>6}  {'的中率':>8}", end='')
    print(f"  {'ROI':>8}" if has_payout else "")
    print(f"\n{'-'*70}")

    data_nodiff = all_test.dropna(subset=['偏差値の差'])
    for thr in [-5, 0, 5, 10, 15, 20, 25]:
        sub = data_nodiff[data_nodiff['偏差値の差'] >= thr]
        if len(sub) < 30: continue
        hits = sub['target_win'].sum()
        rate = sub['target_win'].mean()
        line = f"  {thr:>+4}以上  {len(sub):>7,}頭  {hits:>6}  {rate:>8.1%}"
        if has_payout:
            sub_pay = sub.dropna(subset=['単勝配当'])
            roi, w, n = calc_roi(sub_pay)
            line += f"  {roi:>+8.1%}"
        print(line)

    # ────────────────────────────────────────────────
    # 2. ランカー順位 × ROI
    # ────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f" [2] サブランカー：上位N位に賭けた場合")
    print(f"{'='*70}")
    print(f"{'対象':>10}  {'対象馬':>7}  {'的中数':>6}  {'的中率':>8}", end='')
    print(f"  {'ROI':>8}" if has_payout else "")
    print(f"\n{'-'*70}")

    data_ranked = all_test.dropna(subset=['ランカー順位'])
    for top_n in [1, 2, 3]:
        sub = data_ranked[data_ranked['ランカー順位'] <= top_n]
        if len(sub) < 30: continue
        hits = sub['target_win'].sum()
        rate = sub['target_win'].mean()
        line = f"  上位{top_n}位以内  {len(sub):>7,}頭  {hits:>6}  {rate:>8.1%}"
        if has_payout:
            sub_pay = sub.dropna(subset=['単勝配当'])
            roi, w, n = calc_roi(sub_pay)
            line += f"  {roi:>+8.1%}"
        print(line)

    # ────────────────────────────────────────────────
    # 3. 組み合わせ：ランカー1位 × 偏差値の差
    # ────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f" [3] 組み合わせ：ランカー1位 & 偏差値の差 ≥ 閾値")
    print(f"{'='*70}")
    print(f"{'閾値':>6}  {'対象馬':>7}  {'的中数':>6}  {'的中率':>8}", end='')
    print(f"  {'ROI':>8}" if has_payout else "")
    print(f"\n{'-'*70}")

    data_combo = all_test.dropna(subset=['ランカー順位', '偏差値の差'])
    data_combo = data_combo[data_combo['ランカー順位'] == 1]
    for thr in [-5, 0, 5, 10, 15, 20]:
        sub = data_combo[data_combo['偏差値の差'] >= thr]
        if len(sub) < 20: continue
        hits = sub['target_win'].sum()
        rate = sub['target_win'].mean()
        line = f"  {thr:>+4}以上  {len(sub):>7,}頭  {hits:>6}  {rate:>8.1%}"
        if has_payout:
            sub_pay = sub.dropna(subset=['単勝配当'])
            roi, w, n = calc_roi(sub_pay)
            line += f"  {roi:>+8.1%}"
        print(line)

    # ────────────────────────────────────────────────
    # 4. グループ別 ランカー1位的中率 TOP10
    # ────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f" [4] サブランカー1位的中率 TOP10（グループ別）")
    print(f"{'='*70}")
    rank1 = data_ranked[data_ranked['ランカー順位'] == 1].copy()
    grp_stats = rank1.groupby('model_key').agg(
        対象=('target_win', 'count'),
        的中=('target_win', 'sum'),
    )
    grp_stats['的中率'] = grp_stats['的中'] / grp_stats['対象']
    if has_payout:
        pay_stats = rank1.dropna(subset=['単勝配当']).groupby('model_key').apply(
            lambda x: (x['単勝配当'].where(x['target_win']==1, 0).sum()) / (len(x)*100) - 1
        )
        grp_stats['ROI'] = pay_stats
    top10 = grp_stats[grp_stats['対象'] >= 20].sort_values('的中率', ascending=False).head(10)
    print(top10.to_string())

    # ────────────────────────────────────────────────
    # 5. 全体サマリー
    # ────────────────────────────────────────────────
    avg_horses = all_test.groupby('race_id').size().mean()
    print(f"\n{'='*70}")
    print(f" サマリー")
    print(f"{'='*70}")
    print(f"  平均頭数           : {avg_horses:.1f}頭")
    print(f"  単純ランダム単勝   : {1/avg_horses:.1%}")
    print(f"  サブランカー1位的中: {rank1['target_win'].mean():.1%}（{rank1['target_win'].sum()}/{len(rank1)}）")
    data_15 = data_nodiff[data_nodiff['偏差値の差'] >= 15]
    if len(data_15) > 0:
        print(f"  偏差値の差≥15      : {data_15['target_win'].mean():.1%}（{data_15['target_win'].sum()}/{len(data_15)}）")
    if has_payout:
        r1_pay = rank1.dropna(subset=['単勝配当'])
        roi_r1, w_r1, n_r1 = calc_roi(r1_pay)
        print(f"  サブランカー1位ROI : {roi_r1:+.1%}（{w_r1}/{n_r1}）")
        r15_pay = data_15.dropna(subset=['単勝配当']) if len(data_15) > 0 else pd.DataFrame()
        if len(r15_pay) > 0:
            roi_15, w_15, n_15 = calc_roi(r15_pay)
            print(f"  偏差値の差≥15 ROI  : {roi_15:+.1%}（{w_15}/{n_15}）")

if __name__ == "__main__":
    main()
