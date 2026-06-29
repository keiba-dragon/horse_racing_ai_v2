# coding: utf-8
"""東5R カノープス clogit_calib=1.0 バグ調査"""
import sys, io, pickle, numpy as np, pandas as pd, os

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

BASE_DIR = r'C:\horse_racing_ai_v2'
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))

from save_conditional_logit import prepare as _prep, segment_softmax as _ss
from save_conditional_logit import add_new_features as _anf

# ── モデルロード ────────────────────────────────────────────
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'roi_model.pkl')
with open(MODEL_PATH, 'rb') as f:
    pkg = pickle.load(f)

# ── キャッシュのresult（予測済みDF）から東5R取得 ────────────
CACHE_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'cache', '20260530.cache.pkl')
with open(CACHE_PATH, 'rb') as f:
    cache = pickle.load(f)

result_df = cache['result']
r5_result = result_df[result_df['場 R'].astype(str).str.strip() == '東5'].copy()
print(f"東5R result: {len(r5_result)}頭")

# ── 東5Rの特徴量を result_df から取り出してモデルに通す ───
_s = r5_result.reset_index(drop=True).copy()
_s['race_id'] = 'tmp'
if '着順_num' not in _s.columns:
    _s['着順_num'] = 0

try:
    _s = _anf(_s)
    print("add_new_features: OK")
except Exception as e:
    print(f"[WARN] add_new_features: {e}")

art = pkg['artifacts']['芝']
feat_cols = art['feat_cols']
print(f"特徴量数: {len(feat_cols)}")

for fc in feat_cols:
    if fc not in _s.columns:
        _s[fc] = np.nan

_nan_mask = _s[feat_cols].isna()
_nan_counts = _nan_mask.sum(axis=1).values
_total = len(feat_cols)

print("\n── NaN率 ──")
for i in range(len(_s)):
    name = _s.at[i, '馬名S']
    nc = _nan_counts[i]
    print(f"  {name}: {nc}/{_total} ({100*nc/_total:.0f}%)")

_X, _, _gs, _n, *_ = _prep(
    _s, feat_cols,
    scaler=art['scaler'], poly2=art['poly2'],
    inter_scaler2=art['inter_scaler2'], top_idx=art['top_idx'],
    poly3=None, inter_scaler3=None, top_idx3=None, fit=False)

_high_nan = _nan_counts / _total >= 0.5
_lin = _X @ art['coef']

print("\n── 線形スコア ──")
for i in range(len(_s)):
    name = _s.at[i, '馬名S']
    print(f"  {name}: lin={_lin[i]:.4f}  high_nan={_high_nan[i]}")

_lin_adj = _lin.copy()
_lin_adj[_high_nan] = 0.0

_raw = _ss(_lin_adj, _gs, _n)
_calib = art['isotonic'].predict(_raw)

print("\n── softmax確率 → キャリブ ──")
for i in range(len(_s)):
    name = _s.at[i, '馬名S']
    print(f"  {name}: raw={_raw[i]:.4f}  calib={_calib[i]:.4f}")

# ── カノープスの寄与TOP20 ──────────────────────────────────
kano_i = next((i for i in range(len(_s)) if 'カノープス' in str(_s.at[i, '馬名S'])), None)
if kano_i is not None:
    contribs = _X[kano_i] * art['coef']
    top20 = np.argsort(np.abs(contribs))[::-1][:20]
    print(f"\n── カノープス 寄与TOP20 (lin={_lin[kano_i]:.4f}) ──")
    for j in top20:
        fname = feat_cols[j] if j < len(feat_cols) else f'poly_{j}'
        raw_val = _s[fname].iloc[kano_i] if fname in _s.columns else '?'
        print(f"  {fname}: val={raw_val}  X_scaled={_X[kano_i,j]:.3f}  coef={art['coef'][j]:.4f}  contrib={contribs[j]:.4f}")

# ── isotonic境界 ────────────────────────────────────────────
iso = art['isotonic']
print(f"\n── isotonic calibration ──")
print(f"  X range: [{iso.X_thresholds_.min():.4f}, {iso.X_thresholds_.max():.4f}]")
print(f"  Y range: [{iso.y_thresholds_.min():.4f}, {iso.y_thresholds_.max():.4f}]")
print("  上位8スレッショルド (X→Y):")
for x, y in zip(iso.X_thresholds_[-8:], iso.y_thresholds_[-8:]):
    print(f"    {x:.4f} → {y:.4f}")
print(f"  カノープスのraw={_raw[kano_i]:.4f} → calib={_calib[kano_i]:.4f}")
