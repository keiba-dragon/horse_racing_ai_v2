import pickle, sys
sys.stdout.reconfigure(encoding='utf-8')

with open(r'C:\horse_racing_ai\data\raw\cache\出馬表形式5月24日2.cache.pkl', 'rb') as f:
    cache = pickle.load(f)

for c in cache:
    r = c.get('result')
    if r is None:
        r = c.get('card_df')
    if r is None:
        continue
    if 'ロングトールサリー' not in list(r.get('馬名S', [])):
        continue
    row = r[r['馬名S'] == 'ロングトールサリー'].iloc[0]
    print(f"レース: {c.get('race_id', '?')}")
    print(f"クラス_rank: {row.get('クラス_rank', 'なし')}")
    print(f"クラス変化: {row.get('クラス変化', 'なし')}")
    print(f"clogit_calib: {row.get('clogit_calib', 'なし')}")
    print(f"clogit_rank:  {row.get('clogit_rank', 'なし')}")
