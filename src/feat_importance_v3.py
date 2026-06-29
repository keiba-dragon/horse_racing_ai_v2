# coding: utf-8
"""LGB feature importance per segment (v3モデルから)"""
import sys, os, pickle
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from save_conditional_logit import BASE_DIR

MODEL_DIR = os.path.join(BASE_DIR, 'models', 'v3')
SEGMENTS  = ['芝_短距離', '芝_中距離', '芝_長距離', 'ダ_短距離', 'ダ_中長距離']
TOP_N     = 20

for seg in SEGMENTS:
    pkl = os.path.join(MODEL_DIR, seg, 'lgbm.pkl')
    if not os.path.exists(pkl):
        print(f'[{seg}] なし')
        continue
    with open(pkl, 'rb') as f:
        lp = pickle.load(f)
    model = lp['model']
    feats = lp['feat_cols']
    imp   = model.feature_importances_
    sr = pd.Series(imp, index=feats).sort_values(ascending=False)
    print(f'\n[{seg}] top{TOP_N}')
    for i, (feat, val) in enumerate(sr.head(TOP_N).items(), 1):
        bar = '█' * int(val / sr.iloc[0] * 30)
        print(f'  {i:>2}. {feat:<35} {val:>6}  {bar}')
