import sys, io, subprocess, os, pickle, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# pkg の構造だけ確認
with open('models/final_model.pkl', 'rb') as f:
    pkg = pickle.load(f)

art = pkg['artifacts']['芝']
print("poly2 type:", type(art['poly2']))
print("inter_scaler2 type:", type(art['inter_scaler2']))
print("top_idx len:", len(art['top_idx']))
print("feat_cols len:", len(art['feat_cols']))
print("coef len:", len(art['coef']))
# poly2が sklearn PolynomialFeatures なら n_output_features_ がある
p2 = art['poly2']
if hasattr(p2, 'n_output_features_'):
    print("poly2.n_output_features_:", p2.n_output_features_)
if hasattr(p2, 'n_features_in_'):
    print("poly2.n_features_in_:", p2.n_features_in_)
