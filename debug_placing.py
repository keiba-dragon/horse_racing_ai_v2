# coding: utf-8
"""placingモデルの生スコアを確認するデバッグスクリプト"""
import pickle, io, sys, numpy as np, pandas as pd
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, 'src')

from save_conditional_logit import prepare, segment_softmax, add_new_features
from save_lambdarank_pace import add_pace_features

# キャッシュからresultを読む
with open(r'data\raw\cache\出馬表形式05月30日_api.cache.pkl', 'rb') as f:
    cache = pickle.load(f)
result = cache['result']

# 今日の1レース目を取る（京1R）
r1 = result.iloc[:16].copy()  # 16頭
print('レース:', r1[['馬名S','clogit_calib','clogit_calib_top2']].to_string())

# main model(final_model.pkl)で予測
with open('models/final_model.pkl', 'rb') as f:
    main_pkg = pickle.load(f)
with open('models/final_model_placing.pkl', 'rb') as f:
    placing_pkg = pickle.load(f)

feat_cols = placing_pkg['feat_cols']
_ps = r1.reset_index(drop=True).copy()
_ps['race_id'] = 'tmp'
if '着順_num' not in _ps.columns:
    _ps['着順_num'] = 0

# 同じ前処理
_ps = add_pace_features(_ps)
_ps = add_new_features(_ps)

# 距離変化・休養日数補完
if '距離変化_m' not in _ps.columns or _ps['距離変化_m'].isna().all():
    if '距離変化_前走' in _ps.columns:
        _ps['距離変化_m'] = pd.to_numeric(_ps['距離変化_前走'], errors='coerce')
if '休養日数' not in _ps.columns or _ps['休養日数'].isna().all():
    if '間隔' in _ps.columns:
        _ps['休養日数'] = (pd.to_numeric(_ps['間隔'], errors='coerce') * 7).clip(0, 365)

surf = 'ダ'  # 京1Rはダート
art = placing_pkg['artifacts'][surf]

for fc in feat_cols:
    if fc not in _ps.columns:
        _ps[fc] = np.nan

pX, _, pgs, pn, *_ = prepare(
    _ps, feat_cols,
    scaler=art['scaler'], poly2=art.get('poly2'),
    inter_scaler2=art.get('inter_scaler2'), top_idx=art.get('top_idx'),
    poly3=None, inter_scaler3=None, top_idx3=None, fit=False)

lin_scores = pX @ art['coef']
raw = segment_softmax(lin_scores, pgs, pn)

print('\n生スコア vs clogit_calib:')
for i, (h, lin, r, c) in enumerate(zip(_ps['馬名S'], lin_scores, raw, r1['clogit_calib'])):
    top2 = art['isotonic_top2'].predict([r])[0]
    top3 = art['isotonic_top3'].predict([r])[0]
    print(f'  {h:15s} lin={lin:+.3f}  raw={r:.4f}  top2={top2:.3f}  clogit={c:.3f}')

print(f'\nisotonic_top2 threshold range: [{art["isotonic_top2"].X_thresholds_.min():.4f}, {art["isotonic_top2"].X_thresholds_.max():.4f}]')
print(f'isotonic_top2 y range: [{art["isotonic_top2"].y_thresholds_.min():.4f}, {art["isotonic_top2"].y_thresholds_.max():.4f}]')
print(f'raw probs range today: [{raw.min():.4f}, {raw.max():.4f}]')
