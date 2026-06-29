# coding: utf-8
"""
単勝・複勝ROI実測分析 + 印レベル別比較

- OOS予測 × 結果CSV(複勝配当) をJOINして実測値を計算
- rank_edge × edge閾値 × 頭数 を「印」として定義
- 単勝ROI / 複勝ROI を印レベル別に出力

実行:
    cd C:/horse_racing_ai_v2
    python data/analysis/fuku_roi_analysis.py
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
# race_id = 'YYMMDD_開催_レース名' → 開催を抽出
oos['kaisai'] = oos['race_id'].str.split('_').str[1]

print(f"OOS: {len(oos):,}行  period: {oos['year'].min()}〜{oos['year'].max()}")

# ── 2. 結果CSV読み込み（複勝配当） ──────────────────────────────────
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

print(f"結果CSV: {len(res):,}行  period: {res['日付_num'].min()}〜{res['日付_num'].max()}")

# ── 3. JOIN ────────────────────────────────────────────────────────
merged = oos.merge(res, on=['日付_num','kaisai','馬名S'], how='inner', suffixes=('_oos','_res'))
print(f"JOIN後: {len(merged):,}行  マッチ率: {len(merged)/len(oos):.1%}")
print()

# ── 4. 印定義 ──────────────────────────────────────────────────────
# ◎◎: edge上位かつrank_edge=1（最高信頼）
# ◎  : rank_edge=1
# ○  : rank_edge≤2
# ▲  : rank_edge≤3

def calc_roi(sub, label):
    """単勝・複勝ROIを計算して表示用dictを返す"""
    n = len(sub)
    if n == 0:
        return None
    # 単勝ROI: 1着馬のtan_pay（1着馬の行にのみ存在）
    winners = sub[sub['target_win'] == 1]
    roi_tan = winners['tan_pay'].sum() / 100 / n - 1
    # 複勝ROI: 3着以内馬のfuku_pay
    placed = sub[sub['着順_num_res'].between(1, 3)].dropna(subset=['fuku_pay'])
    roi_fuku = placed['fuku_pay'].sum() / 100 / n - 1
    win_rate   = sub['target_win'].mean()
    place_rate = (sub['着順_num_res'] <= 3).mean()
    avg_odds   = sub['単勝オッズ'].mean()
    return dict(label=label, n=n, win_rate=win_rate, place_rate=place_rate,
                avg_odds=avg_odds, roi_tan=roi_tan, roi_fuku=roi_fuku)


def show_table(rows, title):
    SEP = '=' * 85
    print(f"\n{SEP}")
    print(f" {title}")
    print(SEP)
    print(f"  {'印':^8}  {'N':>6}  {'勝率':>6}  {'複勝率':>6}  {'avg_OD':>7}"
          f"  {'単ROI':>7}  {'複ROI':>7}")
    print("  " + "-" * 83)
    for r in rows:
        if r is None:
            continue
        print(f"  {r['label']:^8}  {r['n']:>6,}  {r['win_rate']:>6.1%}"
              f"  {r['place_rate']:>6.1%}  {r['avg_odds']:>7.1f}"
              f"  {r['roi_tan']:>+7.1%}  {r['roi_fuku']:>+7.1%}")


# ── 5. ダート15頭以上 × 印レベル ─────────────────────────────────
base = merged[(merged['surface']=='ダ') & (merged['頭数']>=15)]
base23 = base[base['year']>='23']

rows_all = [
    calc_roi(base[base['rank_edge']==1],              '◎（全期）'),
    calc_roi(base[base['rank_edge']<=2],              '◎○（全期）'),
    calc_roi(base[base['rank_edge']<=3],              '◎○▲（全期）'),
    calc_roi(base23[base23['rank_edge']==1],          '◎（23+）'),
    calc_roi(base23[base23['rank_edge']<=2],          '◎○（23+）'),
    calc_roi(base23[base23['rank_edge']<=3],          '◎○▲（23+）'),
]
show_table(rows_all, 'ダート15頭以上 × 印レベル')

# ── 6. edge閾値 × ダート15頭 ─────────────────────────────────────
rows_edge = []
for thr in [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]:
    sub = base23[base23['edge'] >= thr]
    rows_edge.append(calc_roi(sub, f'edge≥{thr:.2f}'))
show_table(rows_edge, 'ダート15頭以上 / 2023+ / rank_edge=1 × edge閾値')

# ── 7. オッズ帯 × ダート15頭 ─────────────────────────────────────
rows_odds = []
for lo, hi, lbl in [(1,5,'〜5倍'), (5,10,'5〜10'), (10,20,'10〜20'), (20,99,'20倍+')]:
    sub = base23[(base23['rank_edge']==1) &
                 (base23['単勝オッズ']>=lo) & (base23['単勝オッズ']<hi)]
    rows_odds.append(calc_roi(sub, lbl))
show_table(rows_odds, 'ダート15頭以上 / 2023+ / rank_edge=1 × オッズ帯')

# ── 8. 会場別 ───────────────────────────────────────────────────
rows_venue = []
for venue in ['東','中','阪','京']:
    sub = merged[(merged['surface']=='ダ') & (merged['頭数']>=15) &
                 (merged['year']>='23') & (merged['rank_edge']==1) &
                 (merged['gk'].str.startswith(venue))]
    rows_venue.append(calc_roi(sub, venue+'場'))
show_table(rows_venue, 'ダート15頭以上 / 2023+ / rank_edge=1 × 会場')

# ── 9. 芝 × 印レベル ─────────────────────────────────────────────
shiba = merged[(merged['surface']=='芝') & (merged['頭数']>=15) &
               (merged['クラス_rank']>=2)]
shiba23 = shiba[shiba['year']>='23']

rows_shiba = [
    calc_roi(shiba[shiba['rank_edge']==1],             '◎（全期）'),
    calc_roi(shiba[shiba['edge']>=0.06],               '◎ edge≥0.06（全期）'),
    calc_roi(shiba23[shiba23['rank_edge']==1],          '◎（23+）'),
    calc_roi(shiba23[shiba23['edge']>=0.06],            '◎ edge≥0.06（23+）'),
]
show_table(rows_shiba, '芝15頭以上 / cls≥2（未勝利以上） × 印レベル')

# ── 10. 年別詳細（最良条件） ─────────────────────────────────────
SEP = '=' * 85
print(f'\n{SEP}')
print(' 年別詳細: ダート15頭以上 / rank_edge=1 / 2021〜')
print(SEP)
print(f"  {'年':>4}  {'N':>6}  {'勝率':>6}  {'複勝率':>6}  {'単ROI':>8}  {'複ROI':>8}")
print("  " + "-" * 55)
best = merged[(merged['surface']=='ダ') & (merged['頭数']>=15) & (merged['rank_edge']==1)]
for yr in sorted(best['year'].unique()):
    s = best[best['year']==yr]
    if len(s) < 5:
        continue
    w  = s[s['target_win']==1]
    pl = s[s['着順_num_res'].between(1,3)].dropna(subset=['fuku_pay'])
    roi_t = w['tan_pay'].sum() / 100 / len(s) - 1
    roi_f = pl['fuku_pay'].sum() / 100 / len(s) - 1
    pr = (s['着順_num_res'] <= 3).mean()
    print(f"  20{yr}  {len(s):>6,}  {s['target_win'].mean():>6.1%}  {pr:>6.1%}"
          f"  {roi_t:>+8.1%}  {roi_f:>+8.1%}")

# ── ログ ─────────────────────────────────────────────────────────
import sys as _sys; _sys.path.insert(0, 'src')
try:
    from roi_logger import log
    best23 = merged[(merged['surface']=='ダ') & (merged['頭数']>=15) &
                    (merged['rank_edge']==1) & (merged['year']>='23')]
    w23 = best23[best23['target_win']==1]
    pl23 = best23[best23['着順_num_res'].between(1,3)].dropna(subset=['fuku_pay'])
    log(
        name         = '単勝・複勝ROI実測（OOS×結果CSV JOIN）',
        hypothesis   = 'rank_edge=1のダート15頭以上で複勝ROIはプラスか',
        train_period = '2013-2022',
        test_period  = '2021-2026 OOS / 結果CSV 2023-2026',
        cheat_risk   = '中: val(2023-24)でIsotonic calibration使用',
        bet_type     = '単勝・複勝（実測配当）',
        selection    = {'surface':'ダ','heads_min':15,'rank_edge_max':1,'year_filter':'23+'},
        results      = {
            'N': len(best23),
            'win_rate': round(best23['target_win'].mean(), 3),
            'place_rate': round((best23['着順_num_res']<=3).mean(), 3),
            'roi_tan':  round(w23['tan_pay'].sum()/100/len(best23)-1, 3),
            'roi_fuku': round(pl23['fuku_pay'].sum()/100/len(best23)-1, 3),
        },
        conclusion   = '実行後に記入',
        next_action  = 'オッズ帯・会場別の深掘り',
    )
except Exception as e:
    print(f'[log skip] {e}')
