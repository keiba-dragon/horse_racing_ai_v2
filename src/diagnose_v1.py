# coding: utf-8
"""
diagnose_v1.py - New v1 ダート中長距離 ROI ドリルダウン診断
"""
import sys, os, pickle
import numpy as np
import pandas as pd

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from save_conditional_logit import prepare, segment_softmax, BASE_DIR, DATA_FILE
from save_v3 import add_computed_features

MODEL_DIR = os.path.join(BASE_DIR, 'models')


def load_segment():
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
    df = df[(df['surface'] == 'ダ') & (dm > 1400)].copy()
    df['dist_m'] = dm[df.index]
    if 'クラス_rank' in df.columns:
        df = df[df['クラス_rank'] != 1.0].copy()
    df = add_computed_features(df)
    baba_map = {'良': 0, '稍重': 1, '重': 2, '不良': 3}
    for col in df.columns:
        if '馬場状態' in col:
            df[col] = df[col].map(baba_map)
    df['クラス_rank'] = pd.to_numeric(df['クラス_rank'], errors='coerce')
    return df


def roi_stats(df):
    won  = df['着順_num'] == 1
    odds = pd.to_numeric(df['単勝オッズ'], errors='coerce')
    n    = len(df)
    w    = won.sum()
    if n == 0:
        return float('nan'), float('nan'), 0, 0
    roi  = (odds[won] * 100).sum() / (n * 100) - 1
    return roi, won.mean(), n, w


def print_breakdown(top1, col, label, min_n=50, top_n=None, sort_by_roi=True):
    rows = []
    for val in sorted(top1[col].dropna().unique()):
        s = top1[top1[col] == val]
        roi, wr, n, w = roi_stats(s)
        if n < min_n or np.isnan(roi):
            continue
        rows.append((val, roi, wr, n, w))
    if sort_by_roi:
        rows.sort(key=lambda x: x[1])
    if top_n:
        rows = rows[:top_n]
    print(f'  {label:<18} {"R数":>6} {"勝率":>7} {"ROI":>10}  バー')
    print('  ' + '─' * 58)
    for val, roi, wr, n, w in rows:
        bar_len = int((roi + 0.35) * 40)
        bar_len = max(0, min(bar_len, 40))
        bar = '█' * bar_len
        mark = ' ★' if roi > -0.15 else (' ▼' if roi < -0.25 else '')
        print(f'  {str(val):<18} {n:>6,}  {wr:>6.1%}  {roi:>+9.2%}  {bar}{mark}')
    return rows


def hr(char='─', n=62):
    print(char * n)


def main():
    print()
    hr('═')
    print('  New v1 ダート中長距離 ROI ドリルダウン診断 (OOS 2023+)')
    hr('═')

    with open(os.path.join(MODEL_DIR, 'roi_model.pkl'), 'rb') as f:
        pkg = pickle.load(f)
    art    = pkg['artifacts']['ダ']
    beta   = art['coef']
    scaler = art['scaler']
    feats  = art['feat_cols']

    print('\nデータ読み込み中...')
    df = load_segment()
    oos = df[df['日付_num'] >= 230101].copy()

    valid_p = [c for c in feats if c in oos.columns]
    X_p, _, gs_p, n_p, *_ = prepare(oos, valid_p, scaler=scaler,
                                      top_idx=None, top_idx3=None)
    oos_s = oos.sort_values('race_id').reset_index(drop=True)
    oos_s['prob']  = segment_softmax(X_p @ beta, gs_p, n_p)
    oos_s['rank']  = oos_s.groupby('race_id')['prob'].rank(ascending=False, method='first')
    oos_s['odds_num'] = pd.to_numeric(oos_s['単勝オッズ'], errors='coerce')
    oos_s['yr']    = (oos_s['日付_num'] // 10000).astype(int)
    oos_s['month'] = (oos_s['日付_num'] % 10000 // 100).astype(int)
    oos_s['season'] = oos_s['month'].map({
        1:'冬(1-2月)', 2:'冬(1-2月)', 3:'春(3-5月)', 4:'春(3-5月)', 5:'春(3-5月)',
        6:'夏(6-8月)', 7:'夏(6-8月)', 8:'夏(6-8月)', 9:'秋(9-11月)',
        10:'秋(9-11月)', 11:'秋(9-11月)', 12:'冬(12月)',
    }).fillna('不明')
    oos_s['class_lbl'] = oos_s['クラス_rank'].map({
        2.0: '未勝利(2)', 3.0: '1勝クラス(3)', 4.0: '2勝クラス(4)',
    }).fillna('OP以上(5+)')
    # 会場名抽出 (開催列 "1東京" → "東京")
    import re
    def extract_venue(kaikai):
        m = re.search(r'\d+([^\d]+)', str(kaikai))
        return m.group(1).strip() if m else str(kaikai)
    oos_s['venue'] = oos_s['開催'].apply(extract_venue)
    oos_s['dist_band'] = pd.cut(
        oos_s['dist_m'],
        bins=[1400, 1600, 1800, 2000, 9999],
        labels=['1401-1600m', '1601-1800m', '1801-2000m', '2001m以上'],
    )

    top1 = oos_s[oos_s['rank'] == 1].copy()
    roi_all, wr_all, n_all, w_all = roi_stats(top1)
    print(f'\n  全体 OOS 2023+: ROI={roi_all:+.2%}  ({n_all:,}R, 勝率={wr_all:.1%})')

    # ══════════════════════════════════════════════════════════════════
    print()
    hr('═')
    print('  1. クラス別  (ROI 昇順)')
    hr('═')
    print_breakdown(top1, 'class_lbl', 'クラス', min_n=50)

    # ══════════════════════════════════════════════════════════════════
    print()
    hr('═')
    print('  2. 会場別  (ROI 昇順, 50R以上)')
    hr('═')
    print_breakdown(top1, 'venue', '会場', min_n=50)

    # ══════════════════════════════════════════════════════════════════
    print()
    hr('═')
    print('  3. 季節別  (ROI 昇順)')
    hr('═')
    print_breakdown(top1, 'season', '季節', min_n=50)

    # ══════════════════════════════════════════════════════════════════
    print()
    hr('═')
    print('  4. 距離帯別  (ROI 昇順)')
    hr('═')
    print_breakdown(top1, 'dist_band', '距離帯', min_n=30)

    # ══════════════════════════════════════════════════════════════════
    # 悪い組み合わせを探す: クラス × 会場
    print()
    hr('═')
    print('  5. クラス × 距離帯  (ROI 昇順, 30R以上)')
    hr('═')
    top1['cls_dist'] = top1['class_lbl'].astype(str) + ' × ' + top1['dist_band'].astype(str)
    print_breakdown(top1, 'cls_dist', 'クラス×距離帯', min_n=30)

    # ══════════════════════════════════════════════════════════════════
    print()
    hr('═')
    print('  6. 会場 × クラス  (ワースト/ベスト, 30R以上)')
    hr('═')
    top1['venue_cls'] = top1['venue'] + ' × ' + top1['class_lbl']
    rows = []
    for val in top1['venue_cls'].dropna().unique():
        s = top1[top1['venue_cls'] == val]
        roi, wr, n, w = roi_stats(s)
        if n >= 30 and not np.isnan(roi):
            rows.append((val, roi, wr, n, w))
    rows.sort(key=lambda x: x[1])
    print('  ワースト10:')
    print(f'  {"会場×クラス":<30} {"R数":>5} {"勝率":>7} {"ROI":>10}')
    print('  ' + '─' * 55)
    for val, roi, wr, n, w in rows[:10]:
        print(f'  {val:<30} {n:>5,}  {wr:>6.1%}  {roi:>+9.2%}')
    print()
    print('  ベスト10:')
    print(f'  {"会場×クラス":<30} {"R数":>5} {"勝率":>7} {"ROI":>10}')
    print('  ' + '─' * 55)
    for val, roi, wr, n, w in rows[-10:][::-1]:
        print(f'  {val:<30} {n:>5,}  {wr:>6.1%}  {roi:>+9.2%}')

    # ══════════════════════════════════════════════════════════════════
    # 25+26 vs 23-24 比較
    print()
    hr('═')
    print('  7. 年度別×クラス (改善を確認)')
    hr('═')
    print(f'  {"クラス":<18} {"2023-24":>12} {"2025-26":>12} {"Δ":>8}')
    print('  ' + '─' * 55)
    for cls in ['未勝利(2)', '1勝クラス(3)', '2勝クラス(4)', 'OP以上(5+)']:
        s_old = top1[(top1['class_lbl'] == cls) & (top1['yr'] <= 24)]
        s_new = top1[(top1['class_lbl'] == cls) & (top1['yr'] >= 25)]
        if len(s_old) < 10 or len(s_new) < 10:
            continue
        roi_old, _, n_old, _ = roi_stats(s_old)
        roi_new, _, n_new, _ = roi_stats(s_new)
        delta = roi_new - roi_old
        mark = ' ↑' if delta > 0.03 else (' ↓' if delta < -0.03 else '')
        print(f'  {cls:<18} {roi_old:>+10.2%}  {roi_new:>+10.2%}  {delta:>+6.2%}{mark}')

    print()
    hr('═')
    print('  8. 年度別×会場 (悪い会場が特定年に集中してないか)')
    hr('═')
    print(f'  {"会場":<12} {"2023":>9} {"2024":>9} {"2025":>9} {"2026":>9}')
    print('  ' + '─' * 48)
    venues_sorted = []
    for v in top1['venue'].dropna().unique():
        s_all = top1[top1['venue'] == v]
        if len(s_all) < 30:
            continue
        roi_t, _, n_t, _ = roi_stats(s_all)
        venues_sorted.append((roi_t, v))
    venues_sorted.sort()
    for _, v in venues_sorted[:15]:
        parts = []
        for yr in [23, 24, 25, 26]:
            s = top1[(top1['venue'] == v) & (top1['yr'] == yr)]
            if len(s) < 10:
                parts.append('  ---  ')
            else:
                roi, _, n, _ = roi_stats(s)
                parts.append(f'{roi:>+7.1%}')
        print(f'  {v:<12}' + '  '.join(parts))

    print()
    hr('═')
    print('  【診断まとめ】')
    hr('═')
    # find worst class
    cls_rows = []
    for cls in ['未勝利(2)', '1勝クラス(3)', '2勝クラス(4)', 'OP以上(5+)']:
        s = top1[top1['class_lbl'] == cls]
        if len(s) > 0:
            roi, wr, n, w = roi_stats(s)
            cls_rows.append((roi, cls, n))
    cls_rows.sort()
    worst_cls = cls_rows[0] if cls_rows else (0, '?', 0)
    best_cls  = cls_rows[-1] if cls_rows else (0, '?', 0)

    venue_rows = []
    for v in top1['venue'].dropna().unique():
        s = top1[top1['venue'] == v]
        if len(s) >= 50:
            roi, wr, n, w = roi_stats(s)
            venue_rows.append((roi, v, n))
    venue_rows.sort()
    worst_v = venue_rows[0] if venue_rows else (0, '?', 0)
    best_v  = venue_rows[-1] if venue_rows else (0, '?', 0)

    print(f'  クラス: 最悪={worst_cls[1]}({worst_cls[0]:+.2%}, {worst_cls[2]}R)')
    print(f'          最良={best_cls[1]}({best_cls[0]:+.2%}, {best_cls[2]}R)')
    print(f'  会場  : 最悪={worst_v[1]}({worst_v[0]:+.2%}, {worst_v[2]}R)')
    print(f'          最良={best_v[1]}({best_v[0]:+.2%}, {best_v[2]}R)')
    print()
    roi_23, _, n_23, _ = roi_stats(top1[top1['yr'] == 23])
    roi_24, _, n_24, _ = roi_stats(top1[top1['yr'] == 24])
    roi_25, _, n_25, _ = roi_stats(top1[top1['yr'] == 25])
    roi_26, _, n_26, _ = roi_stats(top1[top1['yr'] == 26])
    print(f'  年度別: 2023={roi_23:+.2%} 2024={roi_24:+.2%} 2025={roi_25:+.2%} 2026={roi_26:+.2%}')
    print(f'  → 2023が突出して悪い ({roi_23:+.2%})。構造的問題か学習外分布か要確認')
    hr('═')
    print()


if __name__ == '__main__':
    main()
