import pickle, pandas as pd, numpy as np, sys
sys.stdout.reconfigure(encoding='utf-8')

with open('models/final_model.pkl', 'rb') as f:
    fm = pickle.load(f)

print("artifacts keys:", list(fm['artifacts'].keys()))
for seg, art in fm['artifacts'].items():
    print(f"  [{seg}] feat_cols={len(art['feat_cols'])} coef shape={np.array(art['coef']).shape}")

# 芝セグメントのコーフを取得
seg_key = next(k for k in fm['artifacts'] if '芝' in str(k) or '1' in str(k))
print(f"\n使用セグメント: {seg_key}")
art = fm['artifacts'][seg_key]
coef = np.array(art['coef'])
feat_cols = art['feat_cols']

# キャッシュからダービー馬の特徴量
with open('data/raw/cache/出馬表形式05月31日_api.cache.pkl', 'rb') as f:
    cache = pickle.load(f)
df = cache['result'].copy()
derby = df[df['場 R'].str.contains(r'東11$')].copy()

emz = derby[derby['馬名S'] == 'エムズビギン'].iloc[0]
rob = derby[derby['馬名S'] == 'ロブチェン'].iloc[0]

common = [c for c in feat_cols if c in derby.columns]
print(f"特徴量一致: {len(common)}/{len(feat_cols)}")

coef_map = dict(zip(feat_cols, coef))
contribs = []
for col in common:
    w = coef_map[col]
    ve = pd.to_numeric(emz.get(col, np.nan), errors='coerce')
    vr = pd.to_numeric(rob.get(col, np.nan), errors='coerce')
    if pd.isna(ve) and pd.isna(vr):
        continue
    ve = 0.0 if pd.isna(ve) else float(ve)
    vr = 0.0 if pd.isna(vr) else float(vr)
    diff = (ve - vr) * w
    contribs.append({'feat': col, 'w': round(w,4), 'emz': round(ve,3), 'rob': round(vr,3), 'diff': round(diff,4)})

cdf = pd.DataFrame(contribs).sort_values('diff', key=abs, ascending=False)
print()
print("=== エムズビギン vs ロブチェン  寄与度差TOP25 ===")
print("diff>0 = エムズビギン有利, diff<0 = ロブチェン有利")
print(cdf.head(25)[['feat','w','emz','rob','diff']].to_string(index=False))
print()
print(f"エムズ有利合計: {cdf[cdf['diff']>0]['diff'].sum():.4f}")
print(f"ロブチェン有利合計: {cdf[cdf['diff']<0]['diff'].sum():.4f}")
print(f"net差: {cdf['diff'].sum():.4f}")
