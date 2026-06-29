# coding: utf-8
"""
eval_odds_filter.py - ROIモデル top1馬 × オッズ帯フィルタ評価

戦略: 各レースでROIモデルが最も高い確率を付けた馬を選び
      オッズが [min_odds, max_odds] の範囲に入るときのみ買う

EV計算のキャリブレーション精度に依存しない。
短オッズ(過剰評価)・超長期(高バリアンス)を除外して安定性を狙う。

usage: python src/eval_odds_filter.py
"""
import os, sys, pickle
import numpy as np
import pandas as pd
from itertools import product

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from save_conditional_logit import prepare, segment_softmax, BASE_DIR, DATA_FILE
from save_v3 import add_computed_features

SEG_DEFS = {
    'ダ':  lambda s, dm, r: (s == 'ダ') & (dm > 1400) & (r != 1.0),
    'ダ短': lambda s, dm, r: (s == 'ダ') & (dm <= 1400) & (r != 1.0),
    '芝短': lambda s, dm, r: (s == '芝') & (dm <= 1400) & (r != 1.0),
    '芝中': lambda s, dm, r: (s == '芝') & (dm > 1400) & (dm <= 2000) & (r != 1.0),
    '芝長': lambda s, dm, r: (s == '芝') & (dm > 2000) & (r != 1.0),
}

# スキャングリッド
MIN_ODDS_LIST = [2.0, 3.0, 4.0, 5.0, 6.0]
MAX_ODDS_LIST = [8.0, 10.0, 12.0, 15.0, 20.0, 30.0, 999.0]


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
    s['prob_raw'] = raw
    return s


def eval_odds_band(df_pred, min_o, max_o):
    """top1馬のオッズが[min_o, max_o]に入るときだけ買う"""
    results = []
    for race_id, grp in df_pred.groupby('race_id'):
        top1_idx = grp['prob_raw'].idxmax()
        top1 = grp.loc[top1_idx]
        o = pd.to_numeric(top1['単勝オッズ'], errors='coerce')
        if pd.isna(o) or o < min_o or o > max_o:
            continue
        won = int(top1['着順_num'] == 1)
        payout = o * 100 if won else 0
        results.append({'won': won, 'payout': payout, 'odds': o})
    if not results:
        return 0.0, 0.0, 0, 0.0
    df_r = pd.DataFrame(results)
    n = len(df_r)
    return df_r['payout'].sum()/(n*100)-1, df_r['won'].mean(), n, df_r['odds'].mean()


def main():
    print('データ読み込み中...')
    df = load_data()
    raw_model = pickle.load(open(os.path.join(BASE_DIR, 'models', 'roi_model.pkl'), 'rb'))
    MODEL = raw_model.get('artifacts', raw_model)

    print('\n' + '='*72)
    print('ROIモデル × オッズ帯フィルタ評価')
    print('  top1馬(by prob_raw)のオッズが帯内に入るときだけ買う')
    print('='*72)

    summary = {}  # seg -> [(min_o, max_o, roi24, roi25, roi26, roi2526, n25, n26)]

    for seg_key, mask_fn in SEG_DEFS.items():
        if seg_key not in MODEL:
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

        # ベースライン（全買い）
        roi_b24, acc_b24, n_b24, avg_o24 = eval_odds_band(pred_2324, 0, 999)
        roi_b25, acc_b25, n_b25, avg_o25 = eval_odds_band(pred_2025, 0, 999)
        roi_b26, acc_b26, n_b26, avg_o26 = (eval_odds_band(pred_2026, 0, 999) if pred_2026 is not None else (0,0,0,0))
        n2526 = n_b25 + n_b26
        roi_b2526 = (roi_b25*n_b25 + roi_b26*n_b26)/n2526 if n2526 > 0 else 0

        print(f'\n{"─"*65}')
        print(f'【{seg_key}】  feats={len(pkg["feat_cols"])}')
        print(f'  ベースライン(全買い): 2324={roi_b24:+.2%}({n_b24}R,acc={acc_b24:.2%}) | 2025={roi_b25:+.2%}({n_b25}R) | 2026={roi_b26:+.2%}({n_b26}R)')
        print(f'{"─"*65}')

        # オッズ帯スキャン
        print(f'  {"帯":>11} | {"買い率":>5} | {"2324ROI":>8} | {"2324acc":>7} | {"2025ROI":>8} | {"2025acc":>7} | {"2026ROI":>8} | {"25+26":>8} | 判定')
        print(f'  {"─"*11}-+-{"─"*5}-+-{"─"*8}-+-{"─"*7}-+-{"─"*8}-+-{"─"*7}-+-{"─"*8}-+-{"─"*8}-+─────')

        results_seg = []
        n_races_2324 = pred_2324['race_id'].nunique()

        for min_o, max_o in product(MIN_ODDS_LIST, MAX_ODDS_LIST):
            if min_o >= max_o:
                continue
            roi_t, acc_t, n_t, avg_o = eval_odds_band(pred_2324, min_o, max_o)
            roi_25, acc_25, n_25, _ = eval_odds_band(pred_2025, min_o, max_o)
            roi_26, acc_26, n_26, _ = (eval_odds_band(pred_2026, min_o, max_o) if pred_2026 is not None else (0,0,0,0))
            n2526 = n_25 + n_26
            roi_2526 = (roi_25*n_25 + roi_26*n_26)/n2526 if n2526 > 0 else 0

            if n_t < 30 or n_25 < 20:  # サンプル少なすぎはスキップ
                continue

            buy_rate = n_t / max(n_races_2324, 1)
            both_pos = roi_t > 0 and roi_25 > 0
            marker = ' ★' if both_pos else ''

            results_seg.append((min_o, max_o, roi_t, acc_t, n_t, roi_25, acc_25, n_25, roi_26, n_26, roi_2526, both_pos, buy_rate))

            label = f'{min_o:.0f}〜{max_o:.0f}倍' if max_o < 999 else f'{min_o:.0f}倍〜'
            print(f'  {label:>11} | {buy_rate:5.1%} | {roi_t:+8.2%} | {acc_t:7.2%} | {roi_25:+8.2%} | {acc_25:7.2%} | {roi_26:+8.2%} | {roi_2526:+8.2%} |{marker}')

        summary[seg_key] = results_seg

    # ★両期間プラスの結果まとめ
    print(f'\n{"="*72}')
    print('★ 2323-24 AND 2025 の両方プラスになったオッズ帯')
    print(f'{"="*72}')
    for seg_key, results_seg in summary.items():
        both_list = [(r[0], r[1], r[2], r[5], r[8], r[10], r[4], r[7])
                     for r in results_seg if r[11]]
        if not both_list:
            print(f'{seg_key}: なし')
            continue
        print(f'\n【{seg_key}】')
        print(f'  {"帯":>11} | {"2324ROI":>8} | {"2025ROI":>8} | {"2026ROI":>8} | {"25+26ROI":>9} | {"n(25)":>6}')
        print(f'  {"─"*11}-+-{"─"*8}-+-{"─"*8}-+-{"─"*8}-+-{"─"*9}-+-{"─"*6}')
        for min_o, max_o, r24, r25, r26, r2526, n24, n25 in both_list:
            label = f'{min_o:.0f}〜{max_o:.0f}倍' if max_o < 999 else f'{min_o:.0f}倍〜'
            print(f'  {label:>11} | {r24:+8.2%} | {r25:+8.2%} | {r26:+8.2%} | {r2526:+9.2%} | {n25:6}')

    # 各セグメントで最良の「両方プラス帯」を使った統合ROI
    print(f'\n{"="*72}')
    print('推奨帯（両方プラス・最大25+26ROI）での統合結果')
    print(f'{"="*72}')
    best_per_seg = {}
    for seg_key, results_seg in summary.items():
        both_list = [r for r in results_seg if r[11]]
        if not both_list:
            continue
        # 25+26ROIが最大のものを選ぶ
        best = max(both_list, key=lambda x: x[10])
        best_per_seg[seg_key] = best

    if best_per_seg:
        print(f'{"seg":5} | {"帯":>11} | {"2324ROI":>8} | {"2025ROI":>8} | {"2026ROI":>8} | {"25+26ROI":>9}')
        print(f'{"─"*5}-+-{"─"*11}-+-{"─"*8}-+-{"─"*8}-+-{"─"*8}-+-{"─"*9}')
        total_pay25 = total_pay26 = total_n25 = total_n26 = 0
        for seg_key, r in best_per_seg.items():
            min_o, max_o, r24, _, n24, r25, _, n25, r26, n26, r2526, _, _ = r
            label = f'{min_o:.0f}〜{max_o:.0f}倍' if max_o < 999 else f'{min_o:.0f}倍〜'
            print(f'{seg_key:5} | {label:>11} | {r24:+8.2%} | {r25:+8.2%} | {r26:+8.2%} | {r2526:+9.2%}')
            total_n25 += n25; total_n26 += n26
            total_pay25 += (r25+1)*n25*100
            total_pay26 += (r26+1)*n26*100
        total_bets = total_n25 + total_n26
        if total_bets > 0:
            total_roi = (total_pay25+total_pay26)/(total_bets*100) - 1
            print(f'{"合計":5} |             |          | {"":>8} | {"":>8} | {total_roi:+9.2%}  ({total_bets}R)')


if __name__ == '__main__':
    main()
