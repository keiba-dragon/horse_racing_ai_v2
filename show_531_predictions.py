import sys, io, pickle, pandas as pd, numpy as np, json, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

with open(r'data/raw/cache/出馬表形式05月31日_api.cache.pkl', 'rb') as f:
    c = pickle.load(f)
df = c['result']

# オッズを最新JSONから取得
odds_path = r'data/raw/cache/20260531.odds.json'
odds_dict = {}
if os.path.exists(odds_path):
    with open(odds_path, encoding='utf-8') as f:
        odds_dict = json.load(f)

print("=== 5/31 clogit予測結果 ===\n")
race_keys = ['開催']
if 'Ｒ' in df.columns:
    race_keys.append('Ｒ')
if 'レース名' in df.columns:
    race_keys.append('レース名')

groups = df.groupby(race_keys, sort=True, dropna=False)
for key, idx in groups.groups.items():
    sub = df.loc[idx].copy()
    kaikai = str(key[0]) if isinstance(key, tuple) else str(key)
    r_num = key[1] if isinstance(key, tuple) and len(key) > 1 else ''
    race_name = key[-1] if isinstance(key, tuple) and len(key) > 2 else ''
    kyori = sub['距離'].iloc[0] if '距離' in sub.columns else ''

    calib = pd.to_numeric(sub['clogit_calib'], errors='coerce')
    if calib.isna().all():
        continue

    # ランク付け
    sub['_rank'] = calib.rank(ascending=False, method='first')
    sub = sub.sort_values('_rank')

    # オッズ取得
    def get_odds(name):
        o = odds_dict.get(name, np.nan)
        return f'{o:.1f}' if pd.notna(o) and o else '-'

    top3 = sub.head(3)
    print(f"【{kaikai} {r_num}R {race_name} {kyori}】")
    for _, row in top3.iterrows():
        name = row['馬名S']
        prob = pd.to_numeric(row['clogit_calib'], errors='coerce')
        rank = int(row['_rank'])
        star = '★' if rank == 1 else f'{rank}位'
        odds = get_odds(name)
        print(f"  {star} {name}  {prob*100:.1f}%  オッズ{odds}")
    print()
