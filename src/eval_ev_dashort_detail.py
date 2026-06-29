# coding: utf-8
"""
eval_ev_dashort_detail.py - ダ短 EVフィルタ詳細検証
的中率モデル × EV閾値=1.20 の+28.44%が本物か検証

・四半期別ROI
・オッズ帯別ROI
・的中馬の分布
・異なる閾値の詳細スキャン(0.05刻み)

usage: python src/eval_ev_dashort_detail.py
"""
import os, sys, pickle
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from save_conditional_logit import prepare, segment_softmax, BASE_DIR, DATA_FILE
from save_v3 import add_computed_features


def load_data():
    df = pd.read_parquet(DATA_FILE)
    df['日付_num'] = pd.to_numeric(df['日付'], errors='coerce')
    df['着順_num'] = pd.to_numeric(df['着順_num'], errors='coerce')
    df = df.dropna(subset=['日付_num', '着順_num'])
    df = df[df['着順_num'] < 99]
    df['race_id'] = (df['日付_num'].astype(int).astype(str) + '_' +
                     df['開催'].astype(str).str.strip() + '_' + df['Ｒ'].astype(str).str.strip())
    df = df[df['開催'].notna()].copy()
    df['surface'] = df['距離'].astype(str).str.strip().str.extract(r'^([芝ダ])')[0].fillna('不明')
    dm = pd.to_numeric(df['距離'].astype(str).str.extract(r'(\d+)')[0], errors='coerce')
    df['クラス_rank'] = pd.to_numeric(df['クラス_rank'], errors='coerce')
    df = add_computed_features(df)
    if '今回_会場' in df.columns and '1走前_開催' in df.columns:
        df['輸送有無'] = (df['今回_会場'].astype(str) != df['1走前_開催'].astype(str).str[1]).astype(float)
        df.loc[df['1走前_開催'].isna(), '輸送有無'] = float('nan')
    baba_map = {'良': 0, '稍重': 1, '重': 2, '不良': 3}
    for col in df.columns:
        if '馬場状態' in col and col != '馬場状態':
            df[col] = df[col].map(baba_map)
    df['dist_m'] = dm
    return df


def predict_probs(seg_df, pkg):
    feat_cols = pkg['feat_cols']
    for f in feat_cols:
        if f.endswith('_isnan'):
            base = f[:-6]
            if base in seg_df.columns and f not in seg_df.columns:
                seg_df[f] = seg_df[base].isna().astype(float)
    valid = [c for c in feat_cols if c in seg_df.columns]
    s = seg_df.sort_values('race_id').reset_index(drop=True)
    X, _, gs, n, *_ = prepare(s, valid, scaler=pkg['scaler'], top_idx=None, top_idx3=None)
    raw = segment_softmax(X @ pkg['coef'], gs, n)
    cal = pkg['isotonic'].transform(raw)
    s['prob_raw'] = raw
    s['prob_cal'] = cal
    return s


def get_bets(df_pred, ev_threshold):
    """threshold以上のEVを持つtop1馬を選び、bet recordを返す"""
    results = []
    for race_id, grp in df_pred.groupby('race_id'):
        grp = grp.copy()
        odds = pd.to_numeric(grp['単勝オッズ'], errors='coerce')
        grp['ev'] = grp['prob_cal'] * odds
        top1_idx = grp['prob_raw'].idxmax()
        top1 = grp.loc[top1_idx]
        ev = top1['ev']
        o = pd.to_numeric(top1['単勝オッズ'], errors='coerce')
        if pd.isna(ev) or ev < ev_threshold or pd.isna(o):
            continue
        date_num = grp['日付_num'].iloc[0]
        won = int(top1['着順_num'] == 1)
        payout = o * 100 if won else 0
        results.append({
            'race_id': race_id, 'won': won, 'payout': payout,
            'ev': ev, 'odds': o, 'date_num': int(date_num),
            'prob_cal': top1['prob_cal'],
        })
    return pd.DataFrame(results) if results else pd.DataFrame()


def roi_summary(df_bets):
    if len(df_bets) == 0:
        return 0.0, 0.0, 0
    n = len(df_bets)
    return df_bets['payout'].sum()/(n*100)-1, df_bets['won'].mean(), n


def main():
    print('データ読み込み中...')
    df = load_data()
    acc_pkl = os.path.join(BASE_DIR, 'models', 'hitrate_model.pkl')
    MODEL = pickle.load(open(acc_pkl, 'rb'))

    if 'ダ短' not in MODEL:
        print('ダ短モデルなし')
        return

    pkg = MODEL['ダ短']
    s = df['surface']; dm = df['dist_m']; r = df['クラス_rank']
    mask = (s == 'ダ') & (dm <= 1400) & (r != 1.0)
    seg = df[mask].copy()

    oos_2324 = seg[(seg['日付_num'] >= 230101) & (seg['日付_num'] < 250101)].copy()
    oos_2025 = seg[(seg['日付_num'] >= 250101) & (seg['日付_num'] < 260101)].copy()
    oos_2026 = seg[seg['日付_num'] >= 260101].copy()

    pred_2324 = predict_probs(oos_2324, pkg)
    pred_2025 = predict_probs(oos_2025, pkg)
    pred_2026 = predict_probs(oos_2026, pkg) if len(oos_2026) > 0 else None

    print(f'\n{"="*65}')
    print(f'ダ短 EVフィルタ 詳細検証 (的中率モデル)')
    print(f'{"="*65}')
    print(f'acc_2325={pkg.get("acc_2325", 0):.2%}  特徴数={len(pkg["feat_cols"])}')
    print(f'特徴量: {pkg["feat_cols"]}')

    # ベースライン
    bets_b24 = get_bets(pred_2324, 0.0)
    bets_b25 = get_bets(pred_2025, 0.0)
    bets_b26 = get_bets(pred_2026, 0.0) if pred_2026 is not None else pd.DataFrame()
    roi_b24, acc_b24, n_b24 = roi_summary(bets_b24)
    roi_b25, acc_b25, n_b25 = roi_summary(bets_b25)
    roi_b26, acc_b26, n_b26 = roi_summary(bets_b26)
    print(f'\nベースライン: 2324={roi_b24:+.2%} acc={acc_b24:.2%}({n_b24}R) | '
          f'2025={roi_b25:+.2%} acc={acc_b25:.2%}({n_b25}R) | '
          f'2026={roi_b26:+.2%} acc={acc_b26:.2%}({n_b26}R)')

    # 詳細閾値スキャン(0.90〜1.50, 0.05刻み)
    print(f'\n--- 詳細閾値スキャン (0.05刻み) ---')
    print(f'  {"閾値":>5} | {"買い率":>6} | {"2324ROI":>8} | {"2324acc":>8} | {"avg_odds":>8} | {"2025ROI":>8} | {"2025acc":>8} | {"2026ROI":>8}')
    print(f'  {"─"*5}-+-{"─"*6}-+-{"─"*8}-+-{"─"*8}-+-{"─"*8}-+-{"─"*8}-+-{"─"*8}-+-{"─"*8}')
    n_races_2324 = pred_2324['race_id'].nunique()
    for th in np.arange(0.90, 1.51, 0.05):
        bets_t = get_bets(pred_2324, th)
        bets_25 = get_bets(pred_2025, th)
        bets_26 = get_bets(pred_2026, th) if pred_2026 is not None else pd.DataFrame()
        roi_t, acc_t, n_t = roi_summary(bets_t)
        roi_25, acc_25, n_25 = roi_summary(bets_25)
        roi_26, acc_26, n_26 = roi_summary(bets_26)
        buy_rate = n_t / max(n_races_2324, 1)
        avg_o = bets_t['odds'].mean() if len(bets_t) > 0 else 0
        print(f'  {th:5.2f} | {buy_rate:6.1%} | {roi_t:+8.2%} | {acc_t:8.2%} | {avg_o:8.1f} | {roi_25:+8.2%} | {acc_25:8.2%} | {roi_26:+8.2%}')

    # EV=1.20 の詳細分析
    TH = 1.20
    print(f'\n{"="*65}')
    print(f'EV閾値={TH:.2f} 詳細分析')
    print(f'{"="*65}')

    for year_label, bets in [('2023-24', get_bets(pred_2324, TH)),
                              ('2025', get_bets(pred_2025, TH)),
                              ('2026', get_bets(pred_2026, TH) if pred_2026 is not None else pd.DataFrame())]:
        if len(bets) == 0:
            print(f'\n{year_label}: データなし')
            continue
        roi, acc, n = roi_summary(bets)
        print(f'\n【{year_label}】 ROI={roi:+.2%}  acc={acc:.2%}  N={n}R  avg_odds={bets["odds"].mean():.1f}  avg_EV={bets["ev"].mean():.2f}')

        # オッズ帯別
        bins = [0, 3, 5, 8, 12, 20, 999]
        labels = ['〜3', '3〜5', '5〜8', '8〜12', '12〜20', '20+']
        bets = bets.copy()
        bets['odds_bin'] = pd.cut(bets['odds'], bins=bins, labels=labels)
        print(f'  オッズ帯別:')
        for band in labels:
            b = bets[bets['odds_bin'] == band]
            if len(b) == 0: continue
            r_roi, r_acc, r_n = roi_summary(b)
            print(f'    オッズ{band:6}: ROI={r_roi:+8.2%}  acc={r_acc:6.2%}  N={r_n:4}R  wins={b["won"].sum():3}')

        # 四半期別
        bets['quarter'] = (bets['date_num'] // 100).apply(
            lambda ym: f'{ym//100}Q{(ym%100-1)//3+1}')
        print(f'  四半期別:')
        for q in sorted(bets['quarter'].unique()):
            b = bets[bets['quarter'] == q]
            r_roi, r_acc, r_n = roi_summary(b)
            print(f'    {q}: ROI={r_roi:+8.2%}  acc={r_acc:6.2%}  N={r_n:4}R  wins={b["won"].sum():3}')

    # 「的中馬」のオッズ分布
    bets_all = pd.concat([get_bets(pred_2324, TH), get_bets(pred_2025, TH)], ignore_index=True)
    wins = bets_all[bets_all['won'] == 1]
    losses = bets_all[bets_all['won'] == 0]
    if len(wins) > 0 and len(losses) > 0:
        print(f'\n【2323-25合計 的中馬と外れ馬のオッズ比較 (EV>{TH:.2f})】')
        print(f'  的中: 平均オッズ={wins["odds"].mean():.1f}  中央={wins["odds"].median():.1f}  N={len(wins)}')
        print(f'  外れ: 平均オッズ={losses["odds"].mean():.1f}  中央={losses["odds"].median():.1f}  N={len(losses)}')
        print(f'  全体: 平均オッズ={bets_all["odds"].mean():.1f}  平均EV={bets_all["ev"].mean():.2f}')

    # prob_cal vs 実際の的中率（キャリブレーション確認）
    bets_25 = get_bets(pred_2025, TH)
    if len(bets_25) > 10:
        print(f'\n【2025 キャリブレーション確認 (EV>{TH:.2f})】')
        print(f'  平均prob_cal={bets_25["prob_cal"].mean():.3f}  実際的中率={bets_25["won"].mean():.3f}')
        pb = bets_25['prob_cal']
        bins_p = [0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 1.0]
        labels_p = ['<10%', '10-15%', '15-20%', '20-25%', '25-30%', '30-40%', '40%+']
        bets_25 = bets_25.copy()
        bets_25['prob_bin'] = pd.cut(bets_25['prob_cal'], bins=bins_p, labels=labels_p)
        print(f'  prob帯別 (pred vs 実績):')
        for pb_label in labels_p:
            b = bets_25[bets_25['prob_bin'] == pb_label]
            if len(b) < 3: continue
            print(f'    prob={pb_label}: pred={b["prob_cal"].mean():.3f}  実績={b["won"].mean():.3f}  N={len(b)}')


if __name__ == '__main__':
    main()
