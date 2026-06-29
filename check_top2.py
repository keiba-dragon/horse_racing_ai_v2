# coding: utf-8
import pickle, io, sys, numpy as np, pandas as pd
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

with open(r'data\raw\cache\出馬表形式05月30日_api.cache.pkl', 'rb') as f:
    cache = pickle.load(f)
result = cache['result']

# 最初のレース（16頭）
r1 = result.iloc[:16].copy()
cols = ['馬名S', 'clogit_calib', 'clogit_calib_top2', 'clogit_calib_top3', 'clogit_rank']
print('京1R（16頭）:')
print(r1[cols].sort_values('clogit_rank').to_string(float_format=lambda x: f'{x:.3f}'))

# P(top2) >= P(win) のチェック
mask = result['clogit_calib'].notna() & result['clogit_calib_top2'].notna()
sub = result[mask]
violations = sub[sub['clogit_calib_top2'] < sub['clogit_calib']]
print(f'\nP(top2) < P(win) 違反: {len(violations)}件 / {len(sub)}件')

# 全体の統計
print(f'\nclogit_calib 平均: {sub["clogit_calib"].mean():.3f}')
print(f'clogit_calib_top2 平均: {sub["clogit_calib_top2"].mean():.3f}')
print(f'clogit_calib_top3 平均: {sub["clogit_calib_top3"].mean():.3f}')
