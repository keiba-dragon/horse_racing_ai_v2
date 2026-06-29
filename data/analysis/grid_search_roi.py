# coding: utf-8
"""
OOS予測データでROI+条件をグリッドサーチ

探索軸:
  - edge閾値: 0.00～0.06
  - rank_edge上限: 1, 2, 3
  - 頭数下限: 8, 10, 12, 15
  - 芝ダ: all / 芝 / ダ
  - クラス: all / 未勝利以上(>=2) / 1勝以上(>=3)
  - 年次フィルタ: 全期間 / 2023以降

出力:
  - 単勝ROI上位条件
  - 年別安定性
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import itertools

# ── データ読み込み & 列名修正 ───────────────────────────────────────────
df = pd.read_parquet('C:/horse_racing_ai/data/processed/oos_predictions.parquet')
df.columns = ['日付_num', 'year', 'gk', 'race_id', '馬名S', '着順_num', 'target_win',
              '単勝オッズ', '人気', '頭数', '今回_馬場_num', 'クラス_rank',
              'prob_win', 'market_P', 'edge', 'rank_edge']

df['単勝オッズ']  = pd.to_numeric(df['単勝オッズ'],  errors='coerce')
df['頭数']       = pd.to_numeric(df['頭数'],       errors='coerce')
df['クラス_rank'] = pd.to_numeric(df['クラス_rank'], errors='coerce')
df['着順_num']   = pd.to_numeric(df['着順_num'],   errors='coerce')
df['target_place'] = (df['着順_num'] <= 3).astype(float)

# gkから芝ダを抽出
df['surface'] = df['gk'].str.split('_').str[-1]  # '芝' or 'ダ'

print(f"OOSデータ: {len(df):,}行")
print(f"期間: year {sorted(df['year'].unique())}")
print(f"gk種類: {sorted(df['gk'].unique())}")
print(f"surface: {sorted(df['surface'].unique())}")
print(f"クラス_rank: {sorted(df['クラス_rank'].dropna().unique())}")
print(f"頭数: {df['頭数'].min():.0f} ～ {df['頭数'].max():.0f}")
print()


# ── ROI計算関数 ───────────────────────────────────────────────────────
def calc_roi(sub):
    """単勝ROI・複勝ROI・N・勝率・複勝率を返す"""
    sub = sub.dropna(subset=['単勝オッズ'])
    n = len(sub)
    if n == 0:
        return dict(n=0, win_rate=np.nan, place_rate=np.nan,
                    roi_tan=np.nan, roi_fuku_est=np.nan, avg_odds=np.nan)
    winners = sub[sub['target_win'] == 1]
    placed  = sub[sub['target_place'] == 1]
    win_rate   = len(winners) / n
    place_rate = len(placed)  / n
    roi_tan    = winners['単勝オッズ'].sum() / n - 1
    # 複勝ROI推定: place_rate × 推定複勝オッズ - 1
    # 推定複勝オッズ = 0.75 / min(3*prob_win, 0.85) ただし最低1.1倍
    if '単勝オッズ' in sub.columns:
        est_fuku_odds = sub['単勝オッズ'].apply(
            lambda o: max(1.1, min(0.75 / max(3 * (0.75 / max(o, 1.5)), 0.05), o * 0.4))
            if pd.notna(o) else np.nan
        )
        roi_fuku_est = place_rate * est_fuku_odds.mean() - 1
    else:
        roi_fuku_est = np.nan
    return dict(n=n, win_rate=win_rate, place_rate=place_rate,
                roi_tan=roi_tan, roi_fuku_est=roi_fuku_est,
                avg_odds=sub['単勝オッズ'].mean())


def year_stability(sub, label=''):
    """年別ROIを計算して安定性を返す（min年別ROI, プラス年数/総年数）"""
    yearly = []
    for yr in sorted(sub['year'].unique()):
        s = sub[sub['year'] == yr].dropna(subset=['単勝オッズ'])
        if len(s) < 5:
            continue
        winners = s[s['target_win'] == 1]
        roi = winners['単勝オッズ'].sum() / len(s) - 1
        yearly.append((yr, roi, len(s)))
    if not yearly:
        return np.nan, 0, 0
    rois = [r for _, r, _ in yearly]
    plus_years = sum(1 for r in rois if r > 0)
    return min(rois), plus_years, len(yearly)


# ── グリッドサーチ ───────────────────────────────────────────────────
edge_thresholds  = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
rank_edge_maxes  = [1, 2, 3]
heads_mins       = [8, 10, 12, 15]
surfaces         = ['all', '芝', 'ダ']
class_mins       = [1, 2, 3]   # 1=全て, 2=未勝利以上, 3=1勝以上
year_filters     = ['all', '23+']   # 全期間 / 2023以降

results = []
total = (len(edge_thresholds) * len(rank_edge_maxes) * len(heads_mins)
         * len(surfaces) * len(class_mins) * len(year_filters))
print(f"探索パターン数: {total:,}")
print("計算中...")

for edge_thr, rank_max, heads_min, surf, cls_min, yr_filter in itertools.product(
        edge_thresholds, rank_edge_maxes, heads_mins, surfaces, class_mins, year_filters):

    sub = df.copy()

    # 年次フィルタ
    if yr_filter == '23+':
        sub = sub[sub['year'] >= '23']

    # 芝ダ
    if surf != 'all':
        sub = sub[sub['surface'] == surf]

    # edge
    sub = sub[sub['edge'] >= edge_thr]

    # rank_edge
    sub = sub[sub['rank_edge'] <= rank_max]

    # 頭数
    sub = sub[sub['頭数'] >= heads_min]

    # クラス
    if cls_min > 1:
        sub = sub[sub['クラス_rank'] >= cls_min]

    r = calc_roi(sub)
    if r['n'] < 20:
        continue

    min_yr_roi, plus_years, total_years = year_stability(sub)

    results.append({
        'edge_thr':   edge_thr,
        'rank_max':   rank_max,
        'heads_min':  heads_min,
        'surface':    surf,
        'cls_min':    cls_min,
        'yr_filter':  yr_filter,
        **r,
        'min_yr_roi':  min_yr_roi,
        'plus_years':  plus_years,
        'total_years': total_years,
    })

res = pd.DataFrame(results)
print(f"有効パターン: {len(res):,} (N≥20)\n")


# ── 表示 ─────────────────────────────────────────────────────────────
SEP = '=' * 90

def show_top(df_r, title, sort_col='roi_tan', n=30, min_n=30, min_roi=0.0):
    df_r = df_r[df_r['n'] >= min_n]
    df_r = df_r[df_r[sort_col] >= min_roi]
    df_r = df_r.sort_values(sort_col, ascending=False).head(n)
    print(f"\n{SEP}")
    print(f" {title}  (N≥{min_n}, ROI≥{min_roi:.0%})")
    print(SEP)
    if len(df_r) == 0:
        print("  該当なし")
        return
    print(f"  {'edge':>5} {'rank':>4} {'heads':>5} {'surf':>4} {'cls':>3} {'yr':>4}"
          f" {'N':>6} {'勝率':>6} {'複勝率':>6} {'avg_OD':>7}"
          f" {'単ROI':>7} {'安定_min':>8} {'プラス':>6}")
    print("  " + "-" * 88)
    for _, row in df_r.iterrows():
        print(f"  {row['edge_thr']:>5.2f} {row['rank_max']:>4.0f} {row['heads_min']:>5.0f}"
              f" {row['surface']:>4} {row['cls_min']:>3.0f} {row['yr_filter']:>4}"
              f" {row['n']:>6,} {row['win_rate']:>6.1%} {row['place_rate']:>6.1%}"
              f" {row['avg_odds']:>7.1f}"
              f" {row['roi_tan']:>+7.1%}"
              f" {row['min_yr_roi']:>+8.1%}"
              f" {row['plus_years']:.0f}/{row['total_years']:.0f}")


# 1. 単勝ROI上位（全条件）
show_top(res, "単勝ROI 上位30 (全期間・2023+混在)", sort_col='roi_tan', min_n=30, min_roi=0.0)

# 2. ROI+かつプラス年が多いもの（安定条件）
stable = res[(res['roi_tan'] > 0) & (res['plus_years'] >= res['total_years'] * 0.6)]
show_top(stable, "安定ROI+条件 (プラス年≥60%)", sort_col='roi_tan', min_n=30, min_roi=0.0)

# 3. 2023以降のみ（最近の傾向）
show_top(res[res['yr_filter'] == '23+'], "2023以降限定 単勝ROI上位", sort_col='roi_tan', min_n=20, min_roi=0.0)

# 4. ダート限定
show_top(res[res['surface'] == 'ダ'], "ダート限定 単勝ROI上位", sort_col='roi_tan', min_n=30, min_roi=0.0)

# 5. 芝限定
show_top(res[res['surface'] == '芝'], "芝限定 単勝ROI上位", sort_col='roi_tan', min_n=30, min_roi=0.0)

# ── ベスト条件の年別詳細 ──────────────────────────────────────────────
best_rows = res[res['roi_tan'] > 0].sort_values('roi_tan', ascending=False).head(5)
if len(best_rows) > 0:
    print(f"\n{SEP}")
    print(" ベスト5条件の年別ROI詳細")
    print(SEP)
    for _, row in best_rows.iterrows():
        cond_label = (f"edge≥{row['edge_thr']:.2f} rank≤{row['rank_max']:.0f}"
                      f" heads≥{row['heads_min']:.0f} surf={row['surface']}"
                      f" cls≥{row['cls_min']:.0f} yr={row['yr_filter']}")
        sub = df.copy()
        if row['yr_filter'] == '23+':
            sub = sub[sub['year'] >= '23']
        if row['surface'] != 'all':
            sub = sub[sub['surface'] == row['surface']]
        sub = sub[sub['edge'] >= row['edge_thr']]
        sub = sub[sub['rank_edge'] <= row['rank_max']]
        sub = sub[sub['頭数'] >= row['heads_min']]
        if row['cls_min'] > 1:
            sub = sub[sub['クラス_rank'] >= row['cls_min']]
        print(f"\n  [{cond_label}]  全体ROI={row['roi_tan']:+.1%}  N={row['n']:,}")
        print(f"  {'年':>4}  {'N':>6}  {'勝率':>6}  {'単ROI':>8}")
        for yr in sorted(sub['year'].unique()):
            s = sub[sub['year'] == yr].dropna(subset=['単勝オッズ'])
            if len(s) < 5:
                continue
            w = s[s['target_win'] == 1]
            roi = w['単勝オッズ'].sum() / len(s) - 1
            print(f"  20{yr}  {len(s):>6,}  {s['target_win'].mean():>6.1%}  {roi:>+8.1%}")
