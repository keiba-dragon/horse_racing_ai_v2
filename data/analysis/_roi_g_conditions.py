# coding: utf-8
"""G指標を条件別に分解してROIを探す"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd

df = pd.read_parquet('data/processed/oos_predictions.parquet')

ranked = df.sort_values(['race_id', 'prob_win'], ascending=[True, False])
ranked['rank_prob'] = ranked.groupby('race_id')['prob_win'].rank(ascending=False, method='first')

top1 = ranked[ranked['rank_prob'] == 1].copy()
top2 = ranked[ranked['rank_prob'] == 2][['race_id','prob_win']].rename(columns={'prob_win':'prob2'})
top1 = top1.merge(top2, on='race_id', how='left')
top1['gap'] = top1['prob_win'] - top1['prob2']
top1['payout'] = top1['単勝オッズ'] * 100 * top1['target_win']

bins   = [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 1.0]
labels = ['~0.05','0.05~0.10','0.10~0.15','0.15~0.20','0.20~0.25','0.25~0.30','0.30~']
top1['gap_band'] = pd.cut(top1['gap'], bins=bins, labels=labels, right=False)
band_avg = top1.groupby('gap_band', observed=True)['単勝オッズ'].mean()
top1['band_avg_odds'] = top1['gap_band'].map(band_avg).astype(float)
top1 = top1.reset_index(drop=True)

# G指標フィルタ
g = top1[(top1['gap'] >= 0.15) & (top1['単勝オッズ'] >= top1['band_avg_odds'] * 1.3)].copy()
BET = 100

def roi_summary(sub, label=''):
    if len(sub) < 10:
        return None
    n    = len(sub)
    wins = sub['target_win'].sum()
    ret  = sub['payout'].sum()
    roi  = (ret - n*BET) / (n*BET) * 100
    wr   = wins / n * 100
    ao   = sub['単勝オッズ'].mean()
    return {'label': label, 'n': n, 'wins': int(wins), 'wr': wr, 'avg_odds': ao, 'roi': roi}

# ── クラス別 ──
print('=== クラス別 ===')
print(f'{"クラス":<12} {"件数":>5} {"的中率":>7} {"avg_odds":>9} {"ROI":>8}')
for cls, sub in g.groupby('クラス_rank'):
    r = roi_summary(sub, str(cls))
    if r:
        mark = ' ★' if r['roi'] > 0 else ''
        print(f'{r["label"]:<12} {r["n"]:>5} {r["wr"]:>7.1f}% {r["avg_odds"]:>9.2f} {r["roi"]:>8.1f}%{mark}')

# クラス名を確認
print(f'\nクラス_rank値: {sorted(g["クラス_rank"].unique())}')
# race_idからクラス名を取得
g['race_class'] = g['race_id'].str.extract(r'_[^_]+_(.+)$')
print()

# ── 芝/ダート別 ──
print('=== 芝・ダート別 ===')
print(f'{"種別":<8} {"件数":>5} {"的中率":>7} {"avg_odds":>9} {"ROI":>8}')
for key, sub in g.groupby('gk'):
    r = roi_summary(sub, key)
    if r:
        mark = ' ★' if r['roi'] > 0 else ''
        print(f'{r["label"]:<8} {r["n"]:>5} {r["wr"]:>7.1f}% {r["avg_odds"]:>9.2f} {r["roi"]:>8.1f}%{mark}')

print()

# ── 頭数別 ──
print('=== 頭数別 ===')
g['heads_band'] = pd.cut(g['頭数'], bins=[0,10,12,14,16,18,99], labels=['~10','11-12','13-14','15-16','17-18','19~'])
print(f'{"頭数帯":<10} {"件数":>5} {"的中率":>7} {"avg_odds":>9} {"ROI":>8}')
for key, sub in g.groupby('heads_band', observed=True):
    r = roi_summary(sub, str(key))
    if r:
        mark = ' ★' if r['roi'] > 0 else ''
        print(f'{r["label"]:<10} {r["n"]:>5} {r["wr"]:>7.1f}% {r["avg_odds"]:>9.2f} {r["roi"]:>8.1f}%{mark}')

print()

# ── 馬場状態別（今回_馬場_num）──
print('=== 馬場状態別 ===')
print(f'{"馬場":>6} {"件数":>5} {"的中率":>7} {"avg_odds":>9} {"ROI":>8}')
for key, sub in g.groupby('今回_馬場_num'):
    r = roi_summary(sub, str(key))
    if r:
        mark = ' ★' if r['roi'] > 0 else ''
        print(f'{r["label"]:>6} {r["n"]:>5} {r["wr"]:>7.1f}% {r["avg_odds"]:>9.2f} {r["roi"]:>8.1f}%{mark}')

print()

# ── 条件の組み合わせ: 有望なものを掘る ──
print('=== 組み合わせ (件数>=20のみ) ===')
print(f'{"条件":<25} {"件数":>5} {"的中率":>7} {"avg_odds":>9} {"ROI":>8}')
results = []
for gk, sub_gk in g.groupby('gk'):
    for cls, sub in sub_gk.groupby('クラス_rank'):
        r = roi_summary(sub, f'{gk}_cls{cls}')
        if r and r['n'] >= 20:
            results.append(r)

results.sort(key=lambda x: x['roi'], reverse=True)
for r in results:
    mark = ' ★' if r['roi'] > 0 else ''
    print(f'{r["label"]:<25} {r["n"]:>5} {r["wr"]:>7.1f}% {r["avg_odds"]:>9.2f} {r["roi"]:>8.1f}%{mark}')
