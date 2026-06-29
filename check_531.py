import pickle, pandas as pd, sys
sys.stdout.reconfigure(encoding='utf-8')

with open('data/raw/cache/出馬表形式05月31日_api.cache.pkl', 'rb') as f:
    cache = pickle.load(f)
df = cache['result'].copy()

derby = df[df['場 R'].str.contains(r'東11$')].copy()
derby = derby.sort_values('clogit_calib', ascending=False)
print(f"=== 東京11R（日本ダービー）{len(derby)}頭 ===")
# 前走オッズ関連列
show_cols = ['馬名S','clogit_calib','clogit_rank']
# オッズ列を探す
for c in df.columns:
    if ('オッズ' in c or 'odds' in c.lower()) and '走前' not in c:
        show_cols.append(c)
        break
print(derby[show_cols].to_string(index=False))

# 修正前後の比較: 1走前_クラス調整着順
print()
print("=== エムズビギンの特徴量確認 ===")
emz = derby[derby['馬名S'] == 'エムズビギン']
if not emz.empty:
    feat_cols = [c for c in df.columns if 'クラス' in c or '近3走' in c or '近5走' in c]
    print(emz[feat_cols].T.to_string())
