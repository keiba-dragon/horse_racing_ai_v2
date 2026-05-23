# coding: utf-8
"""lambdarank 重要度上位特徴量と、現在除外されている有用候補を確認"""
import sys, os, json, pickle
import pandas as pd, numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import save_lambdarank_pace as _slp
from save_lambdarank_pace import add_pace_features, EXCLUDE, ODDS_REMOVE
from save_conditional_logit import add_new_features

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'models')

with open(os.path.join(MODEL_DIR, 'lambdarank_pace.pkl'), 'rb') as f:
    lgbm = pickle.load(f)
with open(os.path.join(MODEL_DIR, 'lambdarank_pace_info.json'), encoding='utf-8') as f:
    info = json.load(f)

feat_names = info['feat_cols']
imps = lgbm.feature_importance(importance_type='gain')
ranked = sorted(zip(feat_names, imps), key=lambda x: -x[1])

print('=== lambdarank 重要度 TOP50 ===')
for i, (name, imp) in enumerate(ranked[:50]):
    in_odds_remove = name in ODDS_REMOVE
    marker = ' [ODDS_REMOVE]' if in_odds_remove else ''
    print(f'{i+1:3d}. {name:<45} {imp:>12.1f}{marker}')

print('\n=== ODDS_REMOVE 内で重要度が高い特徴量 ===')
odds_in_lgbm = [(n, i) for n, i in zip(feat_names, imps) if n in ODDS_REMOVE]
odds_ranked  = sorted(odds_in_lgbm, key=lambda x: -x[1])
for name, imp in odds_ranked[:20]:
    print(f'  {name:<40} {imp:>12.1f}')
