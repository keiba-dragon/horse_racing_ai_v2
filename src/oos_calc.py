"""src/ 内から実行するOOS勝率計算（sys.stdout問題回避）"""
import sys, io, os, pickle
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE_DIR)

import pandas as pd, numpy as np
from save_conditional_logit import prepare, segment_softmax, add_new_features
from save_lambdarank_pace import add_pace_features

with open(os.path.join(BASE_DIR, 'models', 'roi_model.pkl'), 'rb') as f:
    pkg = pickle.load(f)

print("parquet読み込み中...")
pq = os.path.join(BASE_DIR, 'data', 'processed', 'all_venues_features.parquet')
df = pd.read_parquet(pq)
df['日付_num'] = pd.to_numeric(df['日付'], errors='coerce')
oos_raw = df[(df['日付_num'] >= 230101) & (df['日付_num'] < 260531)].copy()
print(f"OOS行数: {len(oos_raw):,}")

results = []
for surf in ['芝', 'ダ']:
    art = pkg['artifacts'].get(surf)
    if art is None:
        continue
    mask = oos_raw['距離'].astype(str).str.startswith(surf)
    sub = oos_raw[mask].copy().reset_index(drop=True)
    if len(sub) == 0:
        continue
    r_col = 'Ｒ' if 'Ｒ' in sub.columns else ('R' if 'R' in sub.columns else None)
    if r_col:
        sub['race_id'] = (sub['日付'].fillna('').astype(str) + '_' +
                          sub['開催'].fillna('').astype(str) + '_' +
                          sub[r_col].fillna('').astype(str))
    else:
        sub['race_id'] = (sub['日付'].fillna('').astype(str) + '_' +
                          sub['開催'].fillna('').astype(str))
    sub['race_id'] = sub['race_id'].fillna('unknown')
    sub = sub.sort_values('race_id').reset_index(drop=True)
    sub['着順_num_v'] = pd.to_numeric(sub['着順_num'], errors='coerce').replace(99, np.nan)
    sub['着順_num'] = sub['着順_num_v'].fillna(0)
    for fc in art['feat_cols']:
        if fc not in sub.columns:
            sub[fc] = np.nan
    try:
        sub = add_pace_features(sub)
        sub = add_new_features(sub)
    except Exception:
        pass
    X, _, gs, n, *_ = prepare(
        sub, art['feat_cols'],
        scaler=art['scaler'], poly2=art['poly2'],
        inter_scaler2=art['inter_scaler2'], top_idx=art['top_idx'],
        poly3=None, inter_scaler3=None, top_idx3=None, fit=False)
    lin = X @ art['coef']
    raw = segment_softmax(lin, gs, n)
    sub['model_prob'] = raw
    results.append(sub[['着順_num_v', 'model_prob']])
    print(f"  {surf}: {len(sub):,}行")

valid = pd.concat(results, ignore_index=True)
valid = valid[valid['着順_num_v'].notna() & valid['model_prob'].notna()]
valid['win'] = (valid['着順_num_v'] == 1).astype(int)
print(f"\nOOS有効行数: {len(valid):,}")

bands  = [0, 1, 2, 5, 10, 20, 101]
labels = ['0-1%', '1-2%', '2-5%', '5-10%', '10-20%', '20%+']
valid['band'] = pd.cut(valid['model_prob'] * 100, bins=bands, labels=labels)

print("\n【OOS：確率帯別 実勝率】")
print(f"{'確率帯':>8} {'頭数':>8} {'勝ち':>6} {'実勝率':>8}")
for b in labels:
    g = valid[valid['band'] == b]
    n_h = len(g)
    wins = int(g['win'].sum())
    actual = wins / n_h if n_h > 0 else 0
    print(f"{b:>8} {n_h:>8,} {wins:>6} {actual*100:>7.2f}%")
