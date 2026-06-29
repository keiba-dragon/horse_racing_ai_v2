import pickle, pandas as pd, sys
sys.stdout.reconfigure(encoding='utf-8')

with open('data/raw/cache/出馬表形式05月31日_api.cache.pkl', 'rb') as f:
    cache = pickle.load(f)
df = cache['result'].copy()
derby = df[df['場 R'].str.contains(r'東11$')].copy()

# softmax前のraw scoreとcalib後の確率を並べる
cols = ['馬名S','clogit_score','clogit_calib']
print("=== clogit_score（softmax前）vs calib ===")
print(derby.sort_values('clogit_score', ascending=False)[cols].to_string(index=False))

print()
# 主要特徴量の有無
pq = pd.read_parquet('data/processed/all_venues_features.parquet')
uma_col = '馬名S'
derby_horses = derby['馬名S'].tolist()

print("=== parquet最新行の主要特徴量 ===")
key_feats = ['日付','近3走_平均着順','1走前_クラス調整着順','近5走_タイム指数平均','クラス_rank']
for horse in derby['馬名S'].tolist():
    rows = pq[pq[uma_col] == horse]
    if rows.empty:
        print(f'{horse}: parquetに存在しない')
        continue
    latest = rows.sort_values('日付').iloc[-1]
    nan_cnt = latest[key_feats].isna().sum()
    print(f'{horse}: 最終日付={latest["日付"]} NaN={nan_cnt}/{len(key_feats)} score={derby[derby["馬名S"]==horse]["clogit_score"].values[0]:.2f}')
