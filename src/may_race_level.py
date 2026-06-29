# coding: utf-8
"""2026年5月 parquetベース(フォールバックなし): レース単位で1着馬の予測確率帯"""
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

df = pd.read_parquet(os.path.join(BASE_DIR, 'data', 'processed', 'all_venues_features.parquet'))
df['日付_num'] = pd.to_numeric(df['日付'], errors='coerce')

# 2026年5月全体
may = df[(df['日付_num'] >= 260501) & (df['日付_num'] <= 260531)].copy()
print(f'5月全体行数: {len(may):,}')
print(f'日付ユニーク数: {may["日付_num"].nunique()}  ({may["日付_num"].min()}〜{may["日付_num"].max()})')

results = []
for surf in ['芝', 'ダ']:
    art = pkg['artifacts'].get(surf)
    if art is None:
        continue
    mask = may['距離'].astype(str).str.startswith(surf)
    sub = may[mask].copy().reset_index(drop=True)
    if len(sub) == 0:
        continue
    r_col = 'Ｒ' if 'Ｒ' in sub.columns else 'R'
    sub['race_id'] = (sub['日付'].fillna('').astype(str) + '_' +
                      sub['開催'].fillna('').astype(str) + '_' +
                      sub[r_col].fillna('').astype(str))
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
    sub = sub.sort_values('race_id').reset_index(drop=True)
    X, _, gs, n, *_ = prepare(
        sub, art['feat_cols'],
        scaler=art['scaler'], poly2=art['poly2'],
        inter_scaler2=art['inter_scaler2'], top_idx=art['top_idx'],
        poly3=None, inter_scaler3=None, top_idx3=None, fit=False)
    lin = X @ art['coef']
    raw = segment_softmax(lin, gs, n)
    sub['model_prob'] = raw
    results.append(sub[['race_id', '着順_num_v', 'model_prob']])
    print(f'  {surf}: {len(sub):,}行')

valid = pd.concat(results, ignore_index=True)
valid = valid[valid['着順_num_v'].notna() & valid['model_prob'].notna()]
valid['win'] = (valid['着順_num_v'] == 1).astype(int)
valid['prob_pct'] = valid['model_prob'] * 100

winners = valid[valid['win'] == 1].copy()
n_races = valid['race_id'].nunique()
print(f'\n5月レース数: {n_races:,}  1着馬: {len(winners):,}')

bands  = [0, 1, 2, 5, 10, 20, 101]
labels = ['0-1%', '1-2%', '2-5%', '5-10%', '10-20%', '20%+']
winners['band'] = pd.cut(winners['prob_pct'], bins=bands, labels=labels, include_lowest=True)
valid['band']   = pd.cut(valid['prob_pct'],   bins=bands, labels=labels, include_lowest=True)

print()
print('【2026年5月 parquet予測: レース単位 1着馬の予測確率帯】')
print(f'{"確率帯":>8} {"レース数":>10} {"割合":>8}  | OOS長期比較')
print('-' * 50)

oos_pct = {'0-1%': 1.5, '1-2%': 2.5, '2-5%': 10.8, '5-10%': 18.2, '10-20%': 27.8, '20%+': 39.3}
for b in labels:
    gw = winners[winners['band'] == b]
    nw = len(gw)
    pct = nw / len(winners) * 100
    oos = oos_pct[b]
    diff = pct - oos
    sign = '+' if diff >= 0 else ''
    print(f'{b:>8} {nw:>10,} {pct:>7.1f}%  | OOS {oos:>4.1f}%  差={sign}{diff:.1f}pp')
