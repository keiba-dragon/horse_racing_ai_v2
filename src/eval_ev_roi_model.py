# coding: utf-8
"""
eval_ev_roi_model.py - ROIモデル(roi_model.pkl) × EVフィルタ評価
EV = prob_cal × 単勝オッズ
確率1位馬のEVが閾値を超えたときだけ買う

ROIモデルは既に25+26 ROI ≥-10%達成済み。EVフィルタでさらに改善できるか検証。

usage: python src/eval_ev_roi_model.py
"""
import os, sys, pickle
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from save_conditional_logit import prepare, segment_softmax, BASE_DIR, DATA_FILE
from save_v3 import add_computed_features

# ROIモデルのartifactキーとセグメント条件
SEG_DEFS = {
    'ダ':  lambda s, dm, r: (s == 'ダ') & (dm > 1400) & (r != 1.0),
    'ダ短': lambda s, dm, r: (s == 'ダ') & (dm <= 1400) & (r != 1.0),
    '芝短': lambda s, dm, r: (s == '芝') & (dm <= 1400) & (r != 1.0),
    '芝中': lambda s, dm, r: (s == '芝') & (dm > 1400) & (dm <= 2000) & (r != 1.0),
    '芝長': lambda s, dm, r: (s == '芝') & (dm > 2000) & (r != 1.0),
}
# CLAUDE.md記載のROIモデル25+26実績
ROI_BASELINE = {'ダ': +0.0111, 'ダ短': -0.0700, '芝短': -0.0489, '芝中': +0.0731, '芝長': +0.0781}
THRESHOLDS = np.arange(0.0, 1.61, 0.10)


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


def eval_strategy(df_pred, ev_threshold):
    results = []
    for race_id, grp in df_pred.groupby('race_id'):
        grp = grp.copy()
        odds = pd.to_numeric(grp['単勝オッズ'], errors='coerce')
        grp['ev'] = grp['prob_cal'] * odds
        top1_idx = grp['prob_raw'].idxmax()
        top1 = grp.loc[top1_idx]
        top1_ev = top1['ev']
        if pd.isna(top1_ev) or top1_ev < ev_threshold:
            continue
        won = int(top1['着順_num'] == 1)
        o = pd.to_numeric(top1['単勝オッズ'], errors='coerce')
        if pd.isna(o):
            continue
        payout = o * 100 if won else 0
        results.append({'race_id': race_id, 'won': won, 'payout': payout,
                        'ev': top1_ev, 'odds': o})
    if not results:
        return 0.0, 0.0, 0, 0.0
    df_r = pd.DataFrame(results)
    n_bets = len(df_r)
    return df_r['payout'].sum()/(n_bets*100)-1, df_r['won'].mean(), n_bets, df_r['odds'].mean()


def main():
    print('データ読み込み中...')
    df = load_data()
    roi_pkl = os.path.join(BASE_DIR, 'models', 'roi_model.pkl')
    raw_model = pickle.load(open(roi_pkl, 'rb'))
    # roi_model.pkl は {'artifacts': {seg: pkg, ...}, ...} の構造
    MODEL = raw_model.get('artifacts', raw_model)

    print('\n' + '='*72)
    print('ROIモデル × EVフィルタ評価')
    print('  EV = prob_cal × 単勝オッズ  |  確率1位馬のEV > threshold のときだけ買う')
    print('='*72)

    all_results = {}

    for seg_key, mask_fn in SEG_DEFS.items():
        if seg_key not in MODEL:
            print(f'\n{seg_key}: モデルなし スキップ')
            continue

        pkg = MODEL[seg_key]
        s = df['surface']; r = df['クラス_rank']; dm = df['dist_m']
        mask = mask_fn(s, dm, r)
        seg = df[mask].copy()

        oos_2324 = seg[(seg['日付_num'] >= 230101) & (seg['日付_num'] < 250101)].copy()
        oos_2025 = seg[(seg['日付_num'] >= 250101) & (seg['日付_num'] < 260101)].copy()
        oos_2026 = seg[seg['日付_num'] >= 260101].copy()

        pred_2324 = predict_probs(oos_2324, pkg)
        pred_2025 = predict_probs(oos_2025, pkg)
        pred_2026 = predict_probs(oos_2026, pkg) if len(oos_2026) > 0 else None

        # ベースライン
        roi_b_2324, acc_b_2324, n_b_2324, avg_o_2324 = eval_strategy(pred_2324, 0.0)
        roi_b_2025, acc_b_2025, n_b_2025, avg_o_2025 = eval_strategy(pred_2025, 0.0)
        roi_b_2026, acc_b_2026, n_b_2026, avg_o_2026 = (eval_strategy(pred_2026, 0.0) if pred_2026 is not None else (0,0,0,0))
        n25b, n26b = n_b_2025, n_b_2026
        roi_base_2526 = (roi_b_2025*n25b + roi_b_2026*n26b)/(n25b+n26b) if (n25b+n26b)>0 else 0.0

        print(f'\n{"─"*65}')
        print(f'【{seg_key}】  ROIモデル特徴数={len(pkg["feat_cols"])}  ROI25+26基準={ROI_BASELINE[seg_key]:+.2%}')
        print(f'{"─"*65}')
        print(f'  ベースライン(全買い): 2324={roi_b_2324:+.2%}({n_b_2324}R) '
              f'2025={roi_b_2025:+.2%}({n_b_2025}R) '
              f'2026={roi_b_2026:+.2%}({n_b_2026}R) '
              f'25+26={roi_base_2526:+.2%}')

        # EV分布
        all_evs = []
        for pred_df in [pred_2324, pred_2025]:
            for rid, grp in pred_df.groupby('race_id'):
                top1_idx = grp['prob_raw'].idxmax()
                o = pd.to_numeric(grp.loc[top1_idx, '単勝オッズ'], errors='coerce')
                ev = grp.loc[top1_idx, 'prob_cal'] * o if not pd.isna(o) else np.nan
                if not pd.isna(ev):
                    all_evs.append(ev)
        if all_evs:
            q = np.percentile(all_evs, [10,25,50,75,90])
            print(f'  top1馬EV分布(2323-25): p10={q[0]:.2f} p25={q[1]:.2f} p50={q[2]:.2f} p75={q[3]:.2f} p90={q[4]:.2f}')

        # 閾値スキャン
        print(f'\n  {"EV閾値":>6} | {"買い率":>6} | {"2324ROI":>8} | {"2324acc":>8} | {"2025ROI":>8} | {"2025acc":>8} | {"2026ROI":>8} | {"25+26ROI":>9}')
        print(f'  {"─"*6}-+-{"─"*6}-+-{"─"*8}-+-{"─"*8}-+-{"─"*8}-+-{"─"*8}-+-{"─"*8}-+-{"─"*9}')

        n_races_2324 = pred_2324['race_id'].nunique()
        best_thresh, best_roi_tune = 0.0, roi_b_2324

        for th in THRESHOLDS:
            roi_t, acc_t, n_t, avg_o = eval_strategy(pred_2324, th)
            roi_25, acc_25, n_25, _ = eval_strategy(pred_2025, th)
            roi_26, acc_26, n_26, _ = (eval_strategy(pred_2026, th) if pred_2026 is not None else (0,0,0,0))
            buy_rate = n_t / max(n_races_2324, 1)
            roi_2526 = (roi_25*n_25 + roi_26*n_26)/(n_25+n_26) if (n_25+n_26)>0 else 0.0
            marker = ' ←best' if roi_t > best_roi_tune + 0.001 else ''
            if roi_t > best_roi_tune + 0.001:
                best_roi_tune = roi_t; best_thresh = th
            print(f'  {th:6.2f} | {buy_rate:6.1%} | {roi_t:+8.2%} | {acc_t:8.2%} | {roi_25:+8.2%} | {acc_25:8.2%} | {roi_26:+8.2%} | {roi_2526:+9.2%}{marker}')

        roi_best_2324, _, n_best_2324, _ = eval_strategy(pred_2324, best_thresh)
        roi_best_2025, _, n_best_2025, _ = eval_strategy(pred_2025, best_thresh)
        roi_best_2026, _, n_best_2026, _ = (eval_strategy(pred_2026, best_thresh) if pred_2026 is not None else (0,0,0,0))
        n25, n26 = n_best_2025, n_best_2026
        roi_2526_best = (roi_best_2025*n25 + roi_best_2026*n26)/(n25+n26) if (n25+n26)>0 else 0.0

        improve = roi_2526_best - roi_base_2526
        print(f'\n  ★最良閾値={best_thresh:.2f}: '
              f'2324={roi_best_2324:+.2%}({n_best_2324}R) '
              f'2025={roi_best_2025:+.2%}({n_best_2025}R) '
              f'2026={roi_best_2026:+.2%}({n_best_2026}R) '
              f'25+26={roi_2526_best:+.2%} '
              f'(ベース比{improve:+.2%})')

        all_results[seg_key] = {
            'best_thresh': best_thresh,
            'roi_2324': roi_best_2324, 'n_2324': n_best_2324,
            'roi_2025': roi_best_2025, 'n_2025': n_best_2025,
            'roi_2026': roi_best_2026, 'n_2026': n_best_2026,
            'roi_2526': roi_2526_best,
            'base_roi_2526': roi_base_2526,
        }

    # 統合サマリー
    print(f'\n{"="*72}')
    print('全セグメント統合サマリー（ROIモデル + EVフィルタ）')
    print(f'{"="*72}')
    print(f'{"seg":5} | {"閾値":>5} | {"2324ROI":>8} | {"2025ROI":>8} | {"2026ROI":>8} | {"25+26ROI":>9} | {"vs基準":>7}')
    print(f'{"─"*5}-+-{"─"*5}-+-{"─"*8}-+-{"─"*8}-+-{"─"*8}-+-{"─"*9}-+-{"─"*7}')
    total_n25 = total_n26 = total_pay25 = total_pay26 = 0
    for k, res in all_results.items():
        vs = res['roi_2526'] - ROI_BASELINE[k]
        print(f'{k:5} | {res["best_thresh"]:5.2f} | '
              f'{res["roi_2324"]:+8.2%} | {res["roi_2025"]:+8.2%} | '
              f'{res["roi_2026"]:+8.2%} | {res["roi_2526"]:+9.2%} | {vs:+7.2%}')
        n25 = res['n_2025']; n26 = res['n_2026']
        total_n25 += n25; total_n26 += n26
        total_pay25 += (res['roi_2025']+1)*n25*100
        total_pay26 += (res['roi_2026']+1)*n26*100
    total_bets = total_n25 + total_n26
    if total_bets > 0:
        total_roi = (total_pay25+total_pay26)/(total_bets*100) - 1
        print(f'{"合計":5} |       |          |          |          | {total_roi:+9.2%}  ({total_bets}R)')


if __name__ == '__main__':
    main()
