# coding: utf-8
import pickle, json, os, pandas as pd, numpy as np, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

with open('data/raw/cache/20260517.cache.pkl', 'rb') as f:
    data = pickle.load(f)
df = data['result']

# Yahoo競馬オッズ読み込み
_odds_path = 'data/raw/cache/20260517.odds.json'
_yahoo_odds = {}
if os.path.exists(_odds_path):
    with open(_odds_path, encoding='utf-8') as f:
        _yahoo_odds = json.load(f)

# 列名を動的に探す（エンコードに依存しない）
def find_col(df, candidates):
    for c in df.columns:
        if c in candidates:
            return c
    return None

col_race = find_col(df, ['場 R', '場R'])
col_surf = find_col(df, ['芝・ダ', '苝・ダ', '苝ダ', '_surface'])
col_dist = find_col(df, ['距離'])
col_name = find_col(df, ['馬名S'])

# 各レース内でYahooオッズ順に人気を計算
df['_yahoo_odds'] = df[col_name].map(_yahoo_odds) if col_name else np.nan
df['_pop_rank'] = df.groupby(col_race)['_yahoo_odds'].rank(method='first', ascending=True)

rows = []
for race, grp in df.groupby(col_race, sort=False):
    grp = grp.copy()
    grp['clogit_rank'] = grp['clogit_rank'].fillna(99).astype(int)
    top3 = grp.nsmallest(3, 'clogit_rank')
    surf = str(grp[col_surf].iloc[0]) if col_surf else '?'
    dist_raw = str(grp[col_dist].iloc[0]) if col_dist else '?'
    dist_num = dist_raw.replace(surf, '')
    n = len(grp)
    for _, row in top3.iterrows():
        horse = row[col_name] if col_name else '?'
        yo  = _yahoo_odds.get(horse, np.nan)
        pop = row.get('_pop_rank', np.nan)
        rows.append({
            'r':  race,
            'c':  surf + dist_num + 'm',
            'n':  n,
            'rk': int(row['clogit_rank']),
            'nm': horse,
            'p':  int(pop) if pd.notna(pop) else '-',
            'o':  f'{yo:.1f}' if pd.notna(yo) else '-',
            's':  round(float(row['clogit_score']), 3),
        })

W = 72
print('=' * W)
print('  2026.5.17（土）  clogit 予測  /  新モデル OOS ROI -12.7%')
print('=' * W)
print(f"{'R':<7} {'コース':<9} {'頭':>2}    {'馬名':<18} {'人気':>3}  {'オッズ':>6}  {'score':>6}")
print('-' * W)

prev = None
for r in rows:
    if r['r'] != prev and prev is not None:
        print()
    prev = r['r']
    mk = '★' if r['rk'] == 1 else ' '
    print(f"{r['r']:<7} {r['c']:<9} {r['n']:>2}  {mk}  {r['nm']:<18} {str(r['p']):>3}  {r['o']:>6}  {r['s']:>6.3f}")

print('=' * W)
