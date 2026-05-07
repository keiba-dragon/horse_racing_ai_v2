# coding: utf-8
"""D指標 当日予測
使い方:
  python daily.py               # 最新の cache pkl を使用
  python daily.py 260419        # 日付指定
"""
import sys, io, os, re, glob, pickle, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
from config import CACHE_DIR

# ── 引数 ──
parser = argparse.ArgumentParser()
parser.add_argument('date', nargs='?', default=None, help='対象日付 (例: 260419)')
args = parser.parse_args()

# ── cache pkl 特定（target_date フィールドで照合） ──
pkl_files = sorted(glob.glob(os.path.join(CACHE_DIR, '*.cache.pkl')))
if not pkl_files:
    print("cache pkl が見つかりません"); sys.exit(1)

# target_date → ファイルパスのマップを作成
date_map = {}
for p in pkl_files:
    try:
        with open(p, 'rb') as f:
            c = pickle.load(f)
        d = str(c.get('target_date', ''))
        if d:
            date_map[d] = p
    except: pass

if args.date:
    if args.date not in date_map:
        print(f"{args.date} に対応する cache pkl が見つかりません")
        print(f"利用可能: {sorted(date_map.keys())[-5:]}")
        sys.exit(1)
    pkl_path = date_map[args.date]
else:
    pkl_path = date_map[max(date_map.keys())]

with open(pkl_path, 'rb') as f:
    cache = pickle.load(f)

target_date = cache.get('target_date', '?')
res = cache['result']
print(f"対象日付: {target_date}  ({os.path.basename(pkl_path)})")
print(f"総行数: {len(res)}頭\n")

# ── D指標計算（pklのスコアをそのまま使用） ──
def gs(col):
    return pd.to_numeric(
        pd.Series(res[col].tolist() if col in res.columns else [np.nan]*len(res)),
        errors='coerce')

sub_cs = gs('sub_コース偏差値')
sub_ri = gs('sub_レース内偏差値')
cur_r  = gs('cur_ランカー順位')
sub_r  = gs('sub_ランカー順位')
odds   = gs('単勝オッズ')
venue  = pd.Series((res['会場'] if '会場' in res.columns else res['開催']).tolist())
rnum   = pd.Series(res['Ｒ'].tolist())
uma    = pd.Series(res['馬名S'].tolist())
banum  = pd.Series(res['馬番'].tolist()) if '馬番' in res.columns else pd.Series(['-']*len(res))
rname  = pd.Series(res['レース名'].tolist()) if 'レース名' in res.columns else pd.Series(['']*len(res))

prod_r = (cur_r * sub_r).clip(lower=0.25)
D = sub_cs * sub_ri / prod_r

df = pd.DataFrame({
    'venue': venue, 'R': rnum, 'race_name': rname,
    'uma': uma, 'banum': banum, 'odds': odds, 'D': D
})
df = df.dropna(subset=['D'])
df['race_key'] = df['venue'].astype(str) + '_' + df['R'].astype(str)

# ── レース単位でD順位・gap計算 ──
df['D_rank'] = df.groupby('race_key')['D'].rank(ascending=False, method='min')
df['D_mean'] = df.groupby('race_key')['D'].transform('mean').clip(lower=1)
df['D_pct']  = (df['D'] - df['D_mean']) / df['D_mean'] * 100

def get_gap(g):
    g2 = g.sort_values('D', ascending=False).reset_index(drop=True)
    d1 = g2.iloc[0]['D'] if len(g2) >= 1 else np.nan
    d2 = g2.iloc[1]['D'] if len(g2) >= 2 else np.nan
    return pd.Series({
        'gap_ratio': d1/d2 if pd.notna(d2) and d2 > 0 else np.nan,
        'gap_pct':   (d1-d2)/d2*100 if pd.notna(d2) and d2 > 0 else np.nan,
    })

gap_df = df.groupby('race_key').apply(get_gap)

# D_pct>200% の頭数（レース単位）
n_qual = df.groupby('race_key')['D_pct'].transform(lambda x: (x > 200).sum())
df['n_qual'] = n_qual

# ── 出力 ──
BUY_MARK = '★'  # 買い推奨条件に合致

print("=" * 72)
for rk in df['race_key'].unique():
    sub = df[df['race_key'] == rk].sort_values('D', ascending=False).reset_index(drop=True)
    race_name = sub.iloc[0]['race_name']
    venue_str = sub.iloc[0]['venue']
    r_num     = sub.iloc[0]['R']

    d1 = sub.iloc[0]['D']
    d2 = sub.iloc[1]['D'] if len(sub) > 1 else np.nan
    gr = d1/d2 if pd.notna(d2) and d2 > 0 else np.nan
    gp = (d1-d2)/d2*100 if pd.notna(d2) and d2 > 0 else np.nan
    n_q = int(sub.iloc[0]['n_qual'])

    # 買い条件判定
    d1_odds = sub.iloc[0]['odds']
    flags = []
    if pd.notna(d1_odds) and d1_odds > 6:
        if pd.notna(gr) and gr > 3:
            flags.append(f"gap>{gr:.1f}x")
        if pd.notna(gp) and gp > 200 and n_q == 1:
            flags.append("1頭抜け")
        if pd.notna(d1_odds) and not flags:
            flags.append("OD>6")

    buy_str = f" {BUY_MARK} {'・'.join(flags)}" if flags else ""
    gr_str = f"{gr:.1f}x" if pd.notna(gr) else "  -"
    print(f"【{venue_str}{int(r_num):2d}R】{str(race_name)[:16]:<16}  gap:{gr_str:>5}  頭数:{len(sub):>2}{buy_str}")

    # D上位5頭表示
    for i, row in sub.head(5).iterrows():
        od_str = f"{row['odds']:.1f}倍" if pd.notna(row['odds']) else "  -"
        d_str  = f"{row['D']:>7.0f}"
        rank_mark = {0:'◎', 1:'○', 2:'▲', 3:'△', 4:'×'}.get(i, ' ')
        print(f"  {rank_mark} {int(row['banum']) if str(row['banum']).isdigit() else row['banum']:>2}番 "
              f"{str(row['uma'])[:8]:<9} D={d_str}  OD={od_str}")
    print()

# ── 買い推奨レースのまとめ ──
print("=" * 72)
print(f"【買い推奨まとめ】D1位 & OD>6 & (gap>3x or 1頭抜け)\n")
found = False
for rk in df['race_key'].unique():
    sub = df[df['race_key'] == rk].sort_values('D', ascending=False).reset_index(drop=True)
    d1_row = sub.iloc[0]
    d2 = sub.iloc[1]['D'] if len(sub) > 1 else np.nan
    gr = d1_row['D']/d2 if pd.notna(d2) and d2 > 0 else np.nan
    gp = (d1_row['D']-d2)/d2*100 if pd.notna(d2) and d2 > 0 else np.nan
    n_q = int(d1_row['n_qual'])
    od  = d1_row['odds']

    if pd.notna(od) and od > 6:
        if (pd.notna(gr) and gr > 3) or (pd.notna(gp) and gp > 200 and n_q == 1):
            od_str = f"{od:.1f}倍"
            gr_str = f"{gr:.1f}x" if pd.notna(gr) else "-"
            cond = []
            if pd.notna(gr) and gr > 3: cond.append(f"gap={gr_str}")
            if pd.notna(gp) and gp > 200 and n_q == 1: cond.append("1頭抜け")
            print(f"  {BUY_MARK} {d1_row['venue']}{int(d1_row['R']):2d}R  "
                  f"{str(d1_row['race_name'])[:12]:<12}  "
                  f"{str(d1_row['uma'])[:8]:<9} OD={od_str}  {'・'.join(cond)}")
            found = True

if not found:
    print("  該当レースなし（OD>6 & gap>3x or 1頭抜け）")
