# coding: utf-8
"""G指標（gap>=0.15 & odds>=band_avg*1.3）の複勝ROI分析"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd

# ── OOS: G指標 top1 馬を抽出 ──
df = pd.read_parquet('data/processed/oos_predictions.parquet')

ranked = df.sort_values(['race_id', 'prob_win'], ascending=[True, False])
ranked['rank_prob'] = ranked.groupby('race_id')['prob_win'].rank(ascending=False, method='first')

top1 = ranked[ranked['rank_prob'] == 1].copy()
top2 = ranked[ranked['rank_prob'] == 2][['race_id','prob_win']].rename(columns={'prob_win':'prob2'})
top1 = top1.merge(top2, on='race_id', how='left')
top1['gap'] = top1['prob_win'] - top1['prob2']

bins   = [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 1.0]
labels = ['~0.05','0.05~0.10','0.10~0.15','0.15~0.20','0.20~0.25','0.25~0.30','0.30~']
top1['gap_band'] = pd.cut(top1['gap'], bins=bins, labels=labels, right=False)
band_avg = top1.groupby('gap_band', observed=True)['単勝オッズ'].mean().rename('band_avg_odds')
top1 = top1.join(band_avg, on='gap_band')

# G指標フィルタ (gap>=0.15 & odds>=avg*1.3)
g = top1[(top1['gap'] >= 0.15) & (top1['単勝オッズ'] >= top1['band_avg_odds'] * 1.3)].copy()
print(f'G指標選択: {len(g)}件 (2021〜2026/4)')

# ── result CSV: 複勝配当を取得 ──
result = pd.read_csv('data/raw/result_utf8_tmp.csv',
                     usecols=['日付','馬名S','着順','複勝配当'],
                     dtype={'日付': str})
result['複勝配当'] = pd.to_numeric(result['複勝配当'], errors='coerce')
result = result.rename(columns={'日付': '日付_num'})
result['日付_num'] = result['日付_num'].astype(int)

# ── 結合 ──
g['日付_num'] = g['日付_num'].astype(int)
merged = g.merge(result[['日付_num','馬名S','複勝配当']],
                 on=['日付_num','馬名S'], how='left')

BET = 100
merged['fuku_hit']   = merged['複勝配当'].notna()
merged['fuku_pay']   = merged['複勝配当'].fillna(0)
merged['tan_hit']    = merged['target_win'] == 1
merged['tan_pay']    = merged['単勝オッズ'] * 100 * merged['target_win']

joined = merged.dropna(subset=['複勝配当'], how='all')  # 結合できた行のみ
n_joined = merged['複勝配当'].notna().sum() + (merged['fuku_hit'] == False).sum()
print(f'result結合: {len(merged)}件中 複勝配当あり={merged["fuku_hit"].sum()}件\n')

# ── 全体比較 ──
print('=== G指標: 単勝 vs 複勝 (全体) ===')
for label, hit, pay in [
    ('単勝', merged['tan_hit'], merged['tan_pay']),
    ('複勝', merged['fuku_hit'], merged['fuku_pay']),
]:
    n   = len(merged)
    w   = hit.sum()
    ret = pay.sum()
    bet = n * BET
    roi = (ret - bet) / bet * 100
    ao  = pay[hit].mean() if hit.sum() > 0 else 0
    print(f'{label}: {n}件 的中{int(w)}件 ({w/n*100:.1f}%) 平均払戻{ao:.0f}円 ROI={roi:+.1f}%')

print()

# ── 年別 ──
print('=== 年別: 複勝 ===')
yr = merged.groupby('year').agg(
    n=('fuku_hit','count'),
    wins=('fuku_hit','sum'),
    ret=('fuku_pay','sum'),
    avg_pay=('fuku_pay', lambda x: x[x>0].mean() if (x>0).any() else 0),
)
yr['bet'] = yr['n'] * BET
yr['roi'] = (yr['ret'] - yr['bet']) / yr['bet'] * 100
yr['wr']  = yr['wins'] / yr['n'] * 100
print(yr[['n','wins','wr','avg_pay','bet','ret','roi']].to_string(float_format='%.1f'))
print()

# ── gap閾値×倍率: 複勝ROIスキャン ──
print('=== gap & 倍率スキャン: 複勝ROI ===')
print(f'{"gap>=":>6} {"倍率":>5} {"件数":>6} {"複勝率":>7} {"avg払戻":>8} {"ROI":>8}')
for gap_thr in [0.10, 0.12, 0.15, 0.18, 0.20]:
    for mult in [1.0, 1.1, 1.2, 1.3]:
        s = top1[(top1['gap'] >= gap_thr) & (top1['単勝オッズ'] >= top1['band_avg_odds'] * mult)].copy()
        s = s.merge(result[['日付_num','馬名S','複勝配当']], on=['日付_num','馬名S'], how='left')
        if len(s) < 30:
            continue
        hit = s['複勝配当'].notna()
        pay = s['複勝配当'].fillna(0)
        roi = (pay.sum() - len(s)*BET) / (len(s)*BET) * 100
        wr  = hit.mean() * 100
        ap  = pay[hit].mean() if hit.sum() > 0 else 0
        print(f'{gap_thr:>6.2f} {mult:>5.1f} {len(s):>6} {wr:>7.1f}% {ap:>8.0f}円 {roi:>8.1f}%')
    print()
