# coding: utf-8
import pickle, io, sys, numpy as np, pandas as pd
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

with open('models/final_model_placing.pkl', 'rb') as f:
    pkg = pickle.load(f)

for surf, art in pkg['artifacts'].items():
    print(f'surface={surf}')
    print(f'  coef shape: {art["coef"].shape}')
    print(f'  scaler n_features_in_: {art["scaler"].n_features_in_}')
    print(f'  poly2: {art["poly2"]}')
    print(f'  top_idx: {art["top_idx"][:5] if art["top_idx"] is not None else None}')
    print(f'  top_idx len: {len(art["top_idx"]) if art["top_idx"] is not None else "None"}')

# 読み込んでみて予測できるか確認
sys.path.insert(0, 'src')
from save_conditional_logit import prepare, segment_softmax

df = pd.read_parquet('data/processed/all_venues_features.parquet')
df['日付_num'] = pd.to_numeric(df['日付'], errors='coerce')
df['着順_num'] = pd.to_numeric(df['着順_num'], errors='coerce')
df = df.dropna(subset=['日付_num', '着順_num'])
df = df[df['着順_num'] < 99]
df['surface'] = df['距離'].astype(str).str.strip().str.extract(r'^([芝ダ])')[0].fillna('不明')

feat_cols = pkg['feat_cols']

# 1レースだけ試す
oos = df[df['日付_num'] >= 230101].copy()
oos['race_id'] = (oos['日付_num'].astype(int).astype(str) + '_' +
                  oos['開催'].astype(str).str.strip() + '_' +
                  oos['Ｒ'].astype(str).str.strip())

rid0 = oos['race_id'].iloc[0]
s = oos[oos['race_id'] == rid0].copy()
surf = s['surface'].iloc[0]
art = pkg['artifacts'].get(surf, pkg['artifacts']['芝'])

for fc in feat_cols:
    if fc not in s.columns:
        s[fc] = np.nan

print(f'\nテスト予測 race_id={rid0}, surface={surf}, n={len(s)}')
try:
    X, _, gs, n, *_ = prepare(
        s, feat_cols,
        scaler=art['scaler'], poly2=art['poly2'],
        inter_scaler2=art['inter_scaler2'], top_idx=art['top_idx'],
        poly3=None, inter_scaler3=None, top_idx3=None, fit=False)
    print(f'  X shape: {X.shape}')
    raw = segment_softmax(X @ art['coef'], gs, n)
    top2 = art['isotonic_top2'].predict(raw)
    top3 = art['isotonic_top3'].predict(raw)
    print(f'  top2 probs: {np.round(top2, 3)}')
    print(f'  top3 probs: {np.round(top3, 3)}')
    print('  予測成功!')
except Exception as e:
    print(f'  ERROR: {e}')
