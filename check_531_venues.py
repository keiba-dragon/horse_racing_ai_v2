import pickle, pandas as pd, sys
sys.stdout.reconfigure(encoding='utf-8')
with open('data/raw/cache/出馬表形式05月31日_api.cache.pkl', 'rb') as f:
    cache = pickle.load(f)
df = cache['result']
print("会場一覧:", df['場 R'].str.extract(r'^(\s*[^\d]+)')[0].str.strip().unique().tolist())
print("レース数:", df['場 R'].nunique())
