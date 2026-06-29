# coding: utf-8
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
from sklearn.isotonic import IsotonicRegression
import pickle, os

oos = pd.read_parquet('data/processed/oos_predictions.parquet')
oos.columns = ['日付_num','year','gk','race_id','馬名S','着順_num','target_win',
               '単勝オッズ','人気','頭数','今回_馬場_num','クラス_rank',
               'prob_win','market_P','edge','rank_edge']
oos['surface']       = oos['gk'].str.split('_').str[-1]
oos['kaisai']        = oos['race_id'].str.split('_').str[1]
oos['rank_edge_fix'] = oos.groupby('race_id')['edge'].rank(ascending=False, method='first')

print('OOS:', len(oos), '行  year:', sorted(oos['year'].unique()))

# 2021-2024 で Isotonic calibration 学習
train = oos[oos['year'] <= '24'].copy()
print('学習:', len(train), '行 (2021-2024)')

ir = IsotonicRegression(out_of_bounds='clip', increasing=True)
ir.fit(train['prob_win'].values, train['target_win'].values)

oos['prob_cal'] = ir.predict(oos['prob_win'].values)
oos['edge_cal'] = oos['prob_cal'] - oos['market_P']
oos['rank_cal'] = oos.groupby('race_id')['edge_cal'].rank(ascending=False, method='first')

SEP = '=' * 78

# ── prob帯別 補正前後比較
print()
print(SEP)
print(' prob_win帯別: 補正前 vs 補正後 vs 実勝率')
print(SEP)
hdr = '  {:^14}  {:>6}  {:>10}  {:>10}  {:>8}  {:>8}  {:>8}'.format(
    'prob帯', 'N', '補正前pred', '補正後pred', '実勝率', '前誤差', '後誤差')
print(hdr)
print('  ' + '-'*76)

bins = [0, 0.02, 0.05, 0.08, 0.12, 0.16, 0.20, 0.25, 0.30, 0.40, 0.60, 1.0]
for lo, hi in zip(bins[:-1], bins[1:]):
    sub = oos[(oos['prob_win']>=lo) & (oos['prob_win']<hi)]
    if len(sub) < 50: continue
    pb = sub['prob_win'].mean()
    pa = sub['prob_cal'].mean()
    ac = sub['target_win'].mean()
    print('  {:^14}  {:>6,}  {:>10.4f}  {:>10.4f}  {:>8.4f}  {:>+8.4f}  {:>+8.4f}'.format(
        f'{lo:.2f}-{hi:.2f}', len(sub), pb, pa, ac, pb-ac, pa-ac))

# ── 2025 完全OOS検証
val = oos[oos['year'] == '25'].copy()
print()
print(SEP)
print(f' 2025 完全OOS検証 N={len(val):,}')
print(SEP)
print('  {:^14}  {:>5}  {:>10}  {:>10}  {:>8}'.format('prob帯','N','補正前err','補正後err','改善'))
print('  ' + '-'*56)
better, worse = 0, 0
for lo, hi in zip(bins[:-1], bins[1:]):
    sub = val[(val['prob_win']>=lo) & (val['prob_win']<hi)]
    if len(sub) < 30: continue
    ac = sub['target_win'].mean()
    eb = sub['prob_win'].mean() - ac
    ea = sub['prob_cal'].mean() - ac
    ok = 'OK' if abs(ea) < abs(eb) else 'NG'
    if ok == 'OK': better += 1
    else: worse += 1
    print('  {:^14}  {:>5,}  {:>+10.4f}  {:>+10.4f}  {:>8}'.format(
        f'{lo:.2f}-{hi:.2f}', len(sub), eb, ea, ok))
print(f'\n  改善: {better}帯 / 悪化: {worse}帯')

# ── rank変化
top_fix = oos[oos['rank_edge_fix']==1][['race_id','馬名S']].rename(columns={'馬名S':'top_fix'})
top_cal = oos[oos['rank_cal']==1][['race_id','馬名S']].rename(columns={'馬名S':'top_cal'})
compare = top_fix.merge(top_cal, on='race_id')
changed = compare[compare['top_fix'] != compare['top_cal']]
total_races = oos['race_id'].nunique()
print()
print(SEP)
print(' rank変化')
print(SEP)
print(f'  全レース: {total_races:,}  変化したレース: {len(changed):,} ({len(changed)/total_races:.1%})')

# ── ROI比較
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
res = res[['日付_num','開催','馬名S','着順_num','tan_pay','fuku_pay']].rename(columns={'開催':'kaisai'})
merged = oos.merge(res, on=['日付_num','kaisai','馬名S'], how='inner', suffixes=('_oos','_res'))

print()
print(SEP)
print(' ROI比較: 補正前 vs 補正後 (ダ15頭+ 2023+)')
print(SEP)
print('  {:^14}  {:>5}  {:>6}  {:>6}  {:>7}  {:>8}  {:>8}'.format(
    '', 'N', '勝率', '複勝率', 'avg_OD', '単ROI', '複ROI'))
print('  ' + '-'*64)

for label, col in [('補正前', 'rank_edge_fix'), ('補正後_cal', 'rank_cal')]:
    base = merged[(merged['surface']=='ダ') & (merged['頭数']>=15) &
                  (merged[col]==1) & (merged['year']>='23')]
    n = len(base)
    w  = base[base['target_win']==1]
    pl = base[base['着順_num_res'].between(1,3)].dropna(subset=['fuku_pay'])
    roi_t = w['tan_pay'].sum()/100/n-1
    roi_f = pl['fuku_pay'].sum()/100/n-1
    print('  {:^14}  {:>5,}  {:>6.1%}  {:>6.1%}  {:>7.1f}  {:>+8.1%}  {:>+8.1%}'.format(
        label, n, base['target_win'].mean(), (base['着順_num_res']<=3).mean(),
        base['単勝オッズ'].mean(), roi_t, roi_f))

# 年別
print()
for label, col in [('補正前', 'rank_edge_fix'), ('補正後', 'rank_cal')]:
    print(f'  [{label}] 年別単ROI:')
    base = merged[(merged['surface']=='ダ') & (merged['頭数']>=15) &
                  (merged[col]==1) & (merged['year']>='23')]
    for yr in sorted(base['year'].unique()):
        s = base[base['year']==yr]
        w = s[s['target_win']==1]
        roi = w['tan_pay'].sum()/100/len(s)-1
        print(f'    20{yr}: N={len(s):>4}  単ROI={roi:+.1%}')
    print()

# ── 2025単年で補正後edgeの効果を確認
print(SEP)
print(' 補正後edge_cal 閾値別ROI (2025単年・ダ15頭+・rank_cal=1)')
print(SEP)
base25 = merged[(merged['surface']=='ダ') & (merged['頭数']>=15) &
                (merged['rank_cal']==1) & (merged['year']=='25')]
print('  {:>12}  {:>5}  {:>6}  {:>8}'.format('閾値', 'N', '勝率', '単ROI'))
print('  ' + '-'*36)
for thr in [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]:
    sub = base25[base25['edge_cal'] >= thr]
    if len(sub) < 10: continue
    w = sub[sub['target_win']==1]
    roi = w['tan_pay'].sum()/100/len(sub)-1
    print('  edge_cal>={:+.2f}  {:>5,}  {:>6.1%}  {:>+8.1%}'.format(
        thr, len(sub), sub['target_win'].mean(), roi))

# calibration保存
with open('data/processed/isotonic_cal.pkl', 'wb') as f:
    pickle.dump(ir, f)
print('\ncalibrationモデル保存: data/processed/isotonic_cal.pkl')
