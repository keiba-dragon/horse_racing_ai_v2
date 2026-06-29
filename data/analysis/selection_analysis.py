# -*- coding: utf-8 -*-
"""OOS予測データで様々な選別条件のROIを分析する"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np, pandas as pd

oos = pd.read_parquet('C:/horse_racing_ai/data/processed/oos_predictions.parquet')
oos['単勝オッズ'] = pd.to_numeric(oos['単勝オッズ'], errors='coerce')
oos['人気'] = pd.to_numeric(oos['人気'], errors='coerce')
oos['頭数'] = pd.to_numeric(oos['頭数'], errors='coerce')

# 芝ダ分離
oos['surf'] = oos['gk'].str.split('_').str[1]
oos['venue'] = oos['gk'].str.split('_').str[0]

def roi_tan(sub):
    s = sub.dropna(subset=['単勝オッズ'])
    if len(s) == 0: return np.nan, 0
    w = s[s['target_win'] == 1]
    return w['単勝オッズ'].sum() / len(s) - 1, len(s)

SEP = '=' * 60
r1 = oos[oos['rank_edge'] == 1]

print(f'\n{SEP}')
print(' ① 芝・ダ別')
print(SEP)
for surf in ['芝', 'ダ']:
    g = r1[r1['surf'] == surf]
    r, n = roi_tan(g)
    print(f'  {surf}:  N={n:>5,}  ROI={r:>+7.1%}')

print(f'\n{SEP}')
print(' ② edge閾値別（全グループ）')
print(SEP)
print(f'  {"閾値":<10}  {"N":>6}  {"ROI":>8}  {"週あたり件数":>12}')
weeks = 5.5 * 52
for thr in [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10]:
    g = r1[r1['edge'] >= thr]
    r, n = roi_tan(g)
    per_week = n / weeks
    print(f'  edge≥{thr:.2f}:  N={n:>5,}  ROI={r:>+7.1%}  {per_week:>8.1f}件/週')

print(f'\n{SEP}')
print(' ③ オッズ帯別（edge1位・全グループ）')
print(SEP)
for lo, hi in [(1.5,3),(3,5),(5,8),(8,12),(12,20),(20,50)]:
    g = r1[(r1['単勝オッズ'] >= lo) & (r1['単勝オッズ'] < hi)]
    r, n = roi_tan(g)
    if n < 20: continue
    print(f'  {lo:.0f}-{hi:.0f}倍:  N={n:>5,}  ROI={r:>+7.1%}')

print(f'\n{SEP}')
print(' ④ edge閾値 × 芝ダ')
print(SEP)
print(f'  {"条件":<18}  {"N":>6}  {"ROI":>8}  {"週あたり":>8}')
for surf in ['ダ', '芝']:
    for thr in [0.00, 0.02, 0.03, 0.05]:
        g = r1[(r1['surf'] == surf) & (r1['edge'] >= thr)]
        r, n = roi_tan(g)
        per_week = n / weeks
        print(f'  {surf} edge≥{thr:.2f}:  N={n:>5,}  ROI={r:>+7.1%}  {per_week:>5.1f}件/週')

print(f'\n{SEP}')
print(' ⑤ edge閾値 × オッズ帯（ダートのみ）')
print(SEP)
dart = r1[r1['surf'] == 'ダ']
print(f'  {"条件":<25}  {"N":>5}  {"ROI":>8}  {"週あたり":>8}')
for thr in [0.00, 0.02, 0.03, 0.05]:
    for lo, hi in [(2,5),(5,10),(10,20),(2,10),(3,15)]:
        g = dart[(dart['edge'] >= thr) & (dart['単勝オッズ'] >= lo) & (dart['単勝オッズ'] < hi)]
        r, n = roi_tan(g)
        if n < 30: continue
        per_week = n / weeks
        print(f'  ダ edge≥{thr:.2f} {lo}-{hi}倍:  N={n:>5,}  ROI={r:>+7.1%}  {per_week:>5.1f}件/週')

print(f'\n{SEP}')
print(' ⑥ 人気帯別（edge1位・全グループ）')
print(SEP)
for lo, hi in [(1,1),(2,3),(4,6),(7,10),(11,99)]:
    g = r1[(r1['人気'] >= lo) & (r1['人気'] <= hi)]
    r, n = roi_tan(g)
    lbl = f'{lo}人気' if lo == hi else f'{lo}-{hi}人気'
    if n < 20: continue
    print(f'  {lbl}:  N={n:>5,}  ROI={r:>+7.1%}')

print(f'\n{SEP}')
print(' ⑦ 頭数別（edge1位・全グループ）')
print(SEP)
for lo, hi in [(8,10),(11,14),(15,18),(8,14),(10,16)]:
    g = r1[(r1['頭数'] >= lo) & (r1['頭数'] <= hi)]
    r, n = roi_tan(g)
    if n < 50: continue
    print(f'  {lo}-{hi}頭:  N={n:>5,}  ROI={r:>+7.1%}')

print(f'\n{SEP}')
print(' ⑧ 馬場状態別（edge1位・全グループ）')
print(SEP)
baba_map = {0:'良', 1:'稍重', 2:'重', 3:'不良'}
for bnum, blbl in baba_map.items():
    g = r1[r1['今回_馬場_num'] == bnum]
    r, n = roi_tan(g)
    if n < 15: continue
    print(f'  {blbl}:  N={n:>5,}  ROI={r:>+7.1%}')

print(f'\n{SEP}')
print(' ⑨ 会場×ダート × edge≥0.02 年別')
print(SEP)
dart_e2 = r1[(r1['surf'] == 'ダ') & (r1['edge'] >= 0.02)]
for yr in sorted(dart_e2['year'].unique()):
    g = dart_e2[dart_e2['year'] == yr]
    r, n = roi_tan(g)
    print(f'  20{yr}: N={n:>4,}  ROI={r:>+7.1%}')
r, n = roi_tan(dart_e2)
print(f'  合計: N={n:>4,}  ROI={r:>+7.1%}  ({n/weeks:.1f}件/週)')

# ── edge差・レース内相対特徴量の分析 ──────────────────────────────────

# レースごとのedge統計を計算（全馬対象）
race_stats = oos.groupby('race_id').agg(
    edge_1st  = ('edge', lambda x: x.nlargest(1).values[0]),
    edge_2nd  = ('edge', lambda x: x.nlargest(2).values[-1] if len(x) >= 2 else np.nan),
    edge_mean = ('edge', 'mean'),
    prob_mean = ('prob_win', 'mean'),
    prob_std  = ('prob_win', 'std'),
).reset_index()
race_stats['edge_gap'] = race_stats['edge_1st'] - race_stats['edge_2nd']  # 1位と2位の差

# rank_edge=1 の行にレース統計をマージ
r1_ext = r1.merge(race_stats, on='race_id', how='left')
r1_ext['prob_vs_mean'] = r1_ext['prob_win'] - r1_ext['prob_mean']   # 平均からの抜け
r1_ext['prob_vs_mean_ratio'] = r1_ext['prob_win'] / r1_ext['prob_mean'].clip(lower=0.001)  # 比率

print(f'\n{SEP}')
print(' ⑩ 1位-2位 edge差（edge_gap）別')
print(SEP)
print(f'  {"条件":<20}  {"N":>6}  {"ROI":>8}  {"週あたり":>8}')
for lo, hi in [(0.00,0.02),(0.02,0.04),(0.04,0.06),(0.06,0.10),(0.10,0.20),(0.20,1.0)]:
    g = r1_ext[(r1_ext['edge_gap'] >= lo) & (r1_ext['edge_gap'] < hi)]
    r, n = roi_tan(g)
    if n < 30: continue
    per_week = n / weeks
    print(f'  gap {lo:.2f}〜{hi:.2f}:  N={n:>5,}  ROI={r:>+7.1%}  {per_week:>5.1f}件/週')

print(f'\n{SEP}')
print(' ⑪ edge_gap閾値別')
print(SEP)
print(f'  {"閾値":<15}  {"N":>6}  {"ROI":>8}  {"週あたり":>8}')
for thr in [0.00, 0.02, 0.04, 0.05, 0.06, 0.08, 0.10, 0.15]:
    g = r1_ext[r1_ext['edge_gap'] >= thr]
    r, n = roi_tan(g)
    per_week = n / weeks
    print(f'  gap≥{thr:.2f}:  N={n:>5,}  ROI={r:>+7.1%}  {per_week:>5.1f}件/週')

print(f'\n{SEP}')
print(' ⑫ prob_vs_mean（レース内勝率平均からの抜け）別')
print(SEP)
print(f'  {"閾値":<20}  {"N":>6}  {"ROI":>8}  {"週あたり":>8}')
for thr in [0.00, 0.02, 0.04, 0.06, 0.08, 0.10]:
    g = r1_ext[r1_ext['prob_vs_mean'] >= thr]
    r, n = roi_tan(g)
    per_week = n / weeks
    print(f'  prob_抜け≥{thr:.2f}:  N={n:>5,}  ROI={r:>+7.1%}  {per_week:>5.1f}件/週')

print(f'\n{SEP}')
print(' ⑬ edge_gap × ダート 組み合わせ')
print(SEP)
dart_ext = r1_ext[r1_ext['surf'] == 'ダ']
print(f'  {"条件":<28}  {"N":>5}  {"ROI":>8}  {"週あたり":>8}')
for ethr in [0.00, 0.02, 0.03]:
    for gthr in [0.00, 0.03, 0.05, 0.08, 0.10]:
        g = dart_ext[(dart_ext['edge'] >= ethr) & (dart_ext['edge_gap'] >= gthr)]
        r, n = roi_tan(g)
        if n < 50: continue
        per_week = n / weeks
        print(f'  ダ edge≥{ethr:.2f} gap≥{gthr:.2f}:  N={n:>5,}  ROI={r:>+7.1%}  {per_week:>5.1f}件/週')

print(f'\n{SEP}')
print(' ⑮ 複勝分析（rank_edge=1・単勝と比較）')
print(SEP)
# 複勝的中 = 3着以内
oos['target_place'] = (pd.to_numeric(oos['着順_num'], errors='coerce') <= 3).astype(int)
r1_p = r1.copy()
r1_p['target_place'] = (pd.to_numeric(r1_p['着順_num'], errors='coerce') <= 3).astype(int)
r1_p['頭数_num'] = pd.to_numeric(r1_p['頭数'], errors='coerce')
r1_p['expected_place_rate'] = 3 / r1_p['頭数_num']  # 期待複勝率

# 複勝ROI推定: 実的中率/期待的中率 × 0.75 - 1 (複勝の控除率約25%と仮定)
def fukusho_stats(sub):
    s = sub.dropna(subset=['頭数_num'])
    if len(s) == 0: return
    hit_rate = s['target_place'].mean()
    exp_rate = (3 / s['頭数_num']).mean()
    # 複勝オッズ概算: 単勝オッズから推定 (簡易)
    # 複勝ROI ≈ hit_rate × avg_fukusho_odds - 1
    # avg_fukusho_odds ≈ 0.75 / expected_place_rate (市場の複勝オッズ水準)
    avg_mkt_odds = 0.75 / exp_rate
    est_roi = hit_rate * avg_mkt_odds - 1
    n = len(s)
    return hit_rate, exp_rate, est_roi, n

print(f'  {"条件":<30}  {"的中率":>7}  {"期待率":>7}  {"推定ROI":>9}  {"N":>5}')

conds = [
    ('全グループ edge1位',        r1_p),
    ('ダート edge1位',            r1_p[r1_p['surf']=='ダ']),
    ('ダート 15頭+ edge1位',      r1_p[(r1_p['surf']=='ダ') & (r1_p['頭数_num']>=15)]),
    ('ダート 15頭+ e≥0.02',       r1_p[(r1_p['surf']=='ダ') & (r1_p['頭数_num']>=15) & (r1_p['edge']>=0.02)]),
    ('ダート e≥0.02',             r1_p[(r1_p['surf']=='ダ') & (r1_p['edge']>=0.02)]),
    ('全 e≥0.05',                 r1_p[r1_p['edge']>=0.05]),
    ('7-10人気',                   r1_p[(r1_p['人気']>=7) & (r1_p['人気']<=10)]),
]
for label, sub in conds:
    res = fukusho_stats(sub)
    if res is None: continue
    hit, exp, roi, n = res
    print(f'  {label:<30}  {hit:>7.1%}  {exp:>7.1%}  {roi:>+9.1%}  {n:>5,}')

print(f'\n{SEP}')
print(' ⑯ 複勝ROI推定（単勝オッズから複勝オッズを逆算）')
print(SEP)
# 複勝オッズ推定: win_P → place_P → fukusho_odds
# market_win_P = 0.75 / 単勝オッズ
# place_P = 3 × win_P (市場の複勝確率の近似), cap 0.85 (約1.0倍の複勝オッズが下限)
# fukusho_odds = 0.75 / place_P (最低1.0倍)
r1_p2 = r1.copy()
r1_p2['着順_num2'] = pd.to_numeric(r1_p2['着順_num'], errors='coerce')
r1_p2['target_place'] = (r1_p2['着順_num2'] <= 3).astype(int)
r1_p2['頭数_num'] = pd.to_numeric(r1_p2['頭数'], errors='coerce')
r1_p2['単勝オッズ_n'] = pd.to_numeric(r1_p2['単勝オッズ'], errors='coerce').clip(lower=1.0)
r1_p2['win_P'] = 0.75 / r1_p2['単勝オッズ_n']
r1_p2['place_P'] = (3 * r1_p2['win_P']).clip(upper=0.85)
r1_p2['est_fuku_odds'] = (0.75 / r1_p2['place_P']).clip(lower=1.0)

def fuku_roi(sub):
    s = sub.dropna(subset=['est_fuku_odds'])
    if len(s) == 0: return np.nan, 0, 0
    hit = s[s['target_place'] == 1]
    roi = hit['est_fuku_odds'].sum() / len(s) - 1
    return roi, s['target_place'].mean(), len(s)

print(f'  {"条件":<35}  {"複勝ROI":>9}  {"的中率":>7}  {"平均複勝推定":>11}  {"N":>5}  {"週":>5}')
conds2 = [
    ('全 edge1位',                    r1_p2),
    ('ダート edge1位',                 r1_p2[r1_p2['surf']=='ダ']),
    ('芝 edge1位',                     r1_p2[r1_p2['surf']=='芝']),
    ('ダート 15頭+ edge1位',           r1_p2[(r1_p2['surf']=='ダ') & (r1_p2['頭数_num']>=15)]),
    ('ダート 15頭+ e≥0.02',            r1_p2[(r1_p2['surf']=='ダ') & (r1_p2['頭数_num']>=15) & (r1_p2['edge']>=0.02)]),
    ('ダート 15頭+ e≥0.03',            r1_p2[(r1_p2['surf']=='ダ') & (r1_p2['頭数_num']>=15) & (r1_p2['edge']>=0.03)]),
    ('ダート e≥0.05',                  r1_p2[(r1_p2['surf']=='ダ') & (r1_p2['edge']>=0.05)]),
    ('全 e≥0.05',                      r1_p2[r1_p2['edge']>=0.05]),
    ('ダート 10-20倍 e≥0.02',          r1_p2[(r1_p2['surf']=='ダ') & (r1_p2['単勝オッズ_n']>=10) & (r1_p2['単勝オッズ_n']<20) & (r1_p2['edge']>=0.02)]),
    ('ダート 15頭+ 5-20倍 e≥0.02',    r1_p2[(r1_p2['surf']=='ダ') & (r1_p2['頭数_num']>=15) & (r1_p2['単勝オッズ_n']>=5) & (r1_p2['単勝オッズ_n']<20) & (r1_p2['edge']>=0.02)]),
]
for label, sub in conds2:
    r, hit, n = fuku_roi(sub)
    if n < 30: continue
    avg_odds = sub['est_fuku_odds'].mean()
    per_week = n / weeks
    print(f'  {label:<35}  {r:>+9.1%}  {hit:>7.1%}  {avg_odds:>11.2f}倍  {n:>5,}  {per_week:>4.1f}件')

# 年別
print(f'\n  --- ダート 15頭+ e≥0.02 年別 ---')
sub_yr = r1_p2[(r1_p2['surf']=='ダ') & (r1_p2['頭数_num']>=15) & (r1_p2['edge']>=0.02)]
for yr in sorted(sub_yr['year'].unique()):
    g = sub_yr[sub_yr['year']==yr]
    r, hit, n = fuku_roi(g)
    print(f'    20{yr}: N={n:>4,}  複勝ROI={r:>+7.1%}  的中率={hit:.1%}')

print(f'\n{SEP}')
print(' ⑭ 有望組み合わせ：ダート×頭数15+×edge_gap')
print(SEP)
big_dart = r1_ext[(r1_ext['surf'] == 'ダ') & (r1_ext['頭数'] >= 15)]
print(f'  {"条件":<25}  {"N":>5}  {"ROI":>8}  {"週あたり":>8}')
for ethr in [0.00, 0.02]:
    for gthr in [0.00, 0.03, 0.05, 0.08]:
        g = big_dart[(big_dart['edge'] >= ethr) & (big_dart['edge_gap'] >= gthr)]
        r, n = roi_tan(g)
        if n < 30: continue
        per_week = n / weeks
        print(f'  大頭数ダ e≥{ethr:.2f} g≥{gthr:.2f}:  N={n:>5,}  ROI={r:>+7.1%}  {per_week:>5.1f}件/週')
