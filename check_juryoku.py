import pickle, sys, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')

with open(r'C:\horse_racing_ai\data\raw\cache\20260524.cache.pkl', 'rb') as f:
    cache = pickle.load(f)

r = cache['result']
row = r[r['馬名S'] == 'ジュウリョクピエロ'].iloc[0]

# タイム指数関連
ti_cols = [c for c in row.index if 'タイム指数' in c or '走破タイム' in c or '上り3F' in c]
ti_cols += ['クラス_rank', '馬体重', '近5走_タイム指数平均', '近5走_タイム指数_max']
print("=== ジュウリョクピエロ タイム指数・関連特徴量 ===")
for c in sorted(set(ti_cols)):
    val = row.get(c, 'N/A')
    print(f"  {c}: {val}")
