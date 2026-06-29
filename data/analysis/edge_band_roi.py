# coding: utf-8
"""
E指標（edge）帯別 単勝・複勝ROI実測分析

- edge帯を細かく分割してどの帯でROIがプラスか確認
- ◎(rank_edge=1) × ダート15頭以上 × 2023+
- 実際の配当（単勝配当・複勝配当）を使用

実行:
    cd C:/horse_racing_ai_v2
    python data/analysis/edge_band_roi.py
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import pandas as pd
import numpy as np

# ── 1. OOS予測読み込み ────────────────────────────────────────────────
oos = pd.read_parquet('data/processed/oos_predictions.parquet')
oos.columns = ['日付_num','year','gk','race_id','馬名S','着順_num','target_win',
               '単勝オッズ','人気','頭数','今回_馬場_num','クラス_rank',
               'prob_win','market_P','edge','rank_edge']

oos['surface'] = oos['gk'].str.split('_').str[-1]
oos['kaisai']  = oos['race_id'].str.split('_').str[1]

# ── 2. 結果CSV読み込み ─────────────────────────────────────────────────
with open('data/raw/2023年～の結果.csv', 'rb') as f:
    raw = f.read()
res = pd.read_csv(io.BytesIO(raw), encoding='cp932')
res.columns = res.columns.str.strip()

res['日付_num'] = pd.to_numeric(res['日付'], errors='coerce').astype('Int64')
res['tan_pay']  = pd.to_numeric(res['単勝配当'], errors='coerce')
res['fuku_pay'] = pd.to_numeric(res['複勝配当'], errors='coerce')
res['着順_num'] = pd.to_numeric(
    res['着順'].astype(str)
    .str.translate(str.maketrans('０１２３４５６７８９', '0123456789'))
    .str.extract(r'(\d+)')[0], errors='coerce'
)
res = res[['日付_num','開催','馬名S','着順_num','tan_pay','fuku_pay']].rename(
    columns={'開催':'kaisai'})

# ── 3. JOIN ────────────────────────────────────────────────────────────
merged = oos.merge(res, on=['日付_num','kaisai','馬名S'], how='inner', suffixes=('_oos','_res'))
print(f"JOIN: {len(merged):,}行  マッチ率: {len(merged)/len(oos):.1%}")

# ── 4. 分析ベース ──────────────────────────────────────────────────────
# ◎ × ダート15頭以上 × 2023+
base = merged[
    (merged['surface']  == 'ダ') &
    (merged['頭数']     >= 15) &
    (merged['rank_edge'] == 1) &
    (merged['year']     >= '23')
].copy()
print(f"分析対象（◎ダ15頭+2023+）: {len(base):,}行\n")


def roi_stats(sub):
    n = len(sub)
    if n == 0:
        return None
    winners = sub[sub['target_win'] == 1]
    placed  = sub[sub['着順_num_res'].between(1, 3)].dropna(subset=['fuku_pay'])
    roi_tan  = winners['tan_pay'].sum() / 100 / n - 1
    roi_fuku = placed['fuku_pay'].sum() / 100 / n - 1
    return dict(
        n          = n,
        win_rate   = sub['target_win'].mean(),
        place_rate = (sub['着順_num_res'] <= 3).mean(),
        avg_odds   = sub['単勝オッズ'].mean(),
        avg_edge   = sub['edge'].mean(),
        roi_tan    = roi_tan,
        roi_fuku   = roi_fuku,
    )


SEP = '=' * 95


# ── 5. edge帯別ROI ─────────────────────────────────────────────────────
print(f"\n{SEP}")
print(" E指標帯別 単勝・複勝ROI  （◎ ダ15頭+ 2023+）")
print(SEP)
print(f"  {'edge帯':^14}  {'N':>6}  {'勝率':>6}  {'複勝率':>6}  {'avg_OD':>7}"
      f"  {'単ROI':>8}  {'複ROI':>8}")
print("  " + "-" * 93)

bands = [
    (float('-inf'), -0.05, 'edge< -0.05'),
    (-0.05,  0.00, '-0.05〜0.00'),
    ( 0.00,  0.02, '0.00〜0.02'),
    ( 0.02,  0.04, '0.02〜0.04'),
    ( 0.04,  0.06, '0.04〜0.06'),
    ( 0.06,  0.08, '0.06〜0.08'),
    ( 0.08,  0.10, '0.08〜0.10'),
    ( 0.10,  0.15, '0.10〜0.15'),
    ( 0.15, float('inf'), '0.15+'),
]

for lo, hi, lbl in bands:
    sub = base[(base['edge'] >= lo) & (base['edge'] < hi)]
    r = roi_stats(sub)
    if r is None or r['n'] < 3:
        continue
    print(f"  {lbl:^14}  {r['n']:>6,}  {r['win_rate']:>6.1%}"
          f"  {r['place_rate']:>6.1%}  {r['avg_odds']:>7.1f}"
          f"  {r['roi_tan']:>+8.1%}  {r['roi_fuku']:>+8.1%}")

# 全体
r_all = roi_stats(base)
print("  " + "-" * 93)
print(f"  {'全体':^14}  {r_all['n']:>6,}  {r_all['win_rate']:>6.1%}"
      f"  {r_all['place_rate']:>6.1%}  {r_all['avg_odds']:>7.1f}"
      f"  {r_all['roi_tan']:>+8.1%}  {r_all['roi_fuku']:>+8.1%}")


# ── 6. edge閾値別ROI（累積） ───────────────────────────────────────────
print(f"\n{SEP}")
print(" edge閾値別ROI（edge≥THR 全件）  （◎ ダ15頭+ 2023+）")
print(SEP)
print(f"  {'閾値':>8}  {'N':>6}  {'勝率':>6}  {'複勝率':>6}  {'avg_OD':>7}"
      f"  {'単ROI':>8}  {'複ROI':>8}")
print("  " + "-" * 93)

for thr in [-0.05, 0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10]:
    sub = base[base['edge'] >= thr]
    r = roi_stats(sub)
    if r is None or r['n'] < 5:
        continue
    print(f"  edge≥{thr:>+5.2f}  {r['n']:>6,}  {r['win_rate']:>6.1%}"
          f"  {r['place_rate']:>6.1%}  {r['avg_odds']:>7.1f}"
          f"  {r['roi_tan']:>+8.1%}  {r['roi_fuku']:>+8.1%}")


# ── 7. オッズ帯 × edge閾値 ────────────────────────────────────────────
print(f"\n{SEP}")
print(" オッズ帯 × edge閾値  （◎ ダ15頭+ 2023+）")
print(SEP)
print(f"  {'条件':^22}  {'N':>5}  {'勝率':>6}  {'複勝率':>6}  {'単ROI':>8}  {'複ROI':>8}")
print("  " + "-" * 73)

odds_bands = [(1,5,'〜5倍'), (5,10,'5〜10'), (10,20,'10〜20'), (20,99,'20倍+')]
edge_thrs = [0.00, 0.02, 0.04, 0.06]

for lo, hi, olbl in odds_bands:
    for thr in edge_thrs:
        sub = base[
            (base['単勝オッズ'] >= lo) &
            (base['単勝オッズ'] <  hi) &
            (base['edge']      >= thr)
        ]
        r = roi_stats(sub)
        if r is None or r['n'] < 5:
            continue
        lbl = f"{olbl} edge≥{thr:.2f}"
        print(f"  {lbl:^22}  {r['n']:>5,}  {r['win_rate']:>6.1%}"
              f"  {r['place_rate']:>6.1%}  {r['roi_tan']:>+8.1%}  {r['roi_fuku']:>+8.1%}")
    print()


# ── 8. 年別 × edge閾値 ────────────────────────────────────────────────
print(f"\n{SEP}")
print(" 年別ROI × edge閾値  （◎ ダ15頭+ 2023+）")
print(SEP)

for thr in [0.00, 0.02, 0.04]:
    sub_thr = base[base['edge'] >= thr]
    r_total = roi_stats(sub_thr)
    if r_total is None:
        continue
    print(f"\n  [edge≥{thr:.2f}]  全体 N={r_total['n']:,}  単ROI={r_total['roi_tan']:+.1%}  複ROI={r_total['roi_fuku']:+.1%}")
    print(f"  {'年':>4}  {'N':>5}  {'勝率':>6}  {'複勝率':>6}  {'単ROI':>8}  {'複ROI':>8}")
    for yr in sorted(sub_thr['year'].unique()):
        s = sub_thr[sub_thr['year'] == yr]
        r = roi_stats(s)
        if r is None or r['n'] < 3:
            continue
        print(f"  20{yr}  {r['n']:>5,}  {r['win_rate']:>6.1%}"
              f"  {r['place_rate']:>6.1%}  {r['roi_tan']:>+8.1%}  {r['roi_fuku']:>+8.1%}")


# ── 9. 複勝ROI+帯の特定 ──────────────────────────────────────────────
print(f"\n{SEP}")
print(" 複勝ROI+ 帯の絞り込み  （◎ ダ15頭+ 2023+）")
print(SEP)
print(" オッズ5倍以上 × edge各閾値")
print(f"  {'条件':^20}  {'N':>5}  {'複勝率':>6}  {'単ROI':>8}  {'複ROI':>8}")
print("  " + "-" * 58)

for thr in [0.00, 0.01, 0.02, 0.03, 0.04, 0.05]:
    sub = base[(base['単勝オッズ'] >= 5) & (base['edge'] >= thr)]
    r = roi_stats(sub)
    if r is None or r['n'] < 5:
        continue
    lbl = f"OD≥5 edge≥{thr:.2f}"
    print(f"  {lbl:^20}  {r['n']:>5,}  {r['place_rate']:>6.1%}"
          f"  {r['roi_tan']:>+8.1%}  {r['roi_fuku']:>+8.1%}")

print()
for thr in [0.00, 0.01, 0.02, 0.03, 0.04, 0.05]:
    sub = base[(base['単勝オッズ'] >= 7) & (base['edge'] >= thr)]
    r = roi_stats(sub)
    if r is None or r['n'] < 5:
        continue
    lbl = f"OD≥7 edge≥{thr:.2f}"
    print(f"  {lbl:^20}  {r['n']:>5,}  {r['place_rate']:>6.1%}"
          f"  {r['roi_tan']:>+8.1%}  {r['roi_fuku']:>+8.1%}")
