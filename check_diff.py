import pickle, pandas as pd, sys

sys.stdout.reconfigure(encoding='utf-8')

with open('data/raw/cache/20260530.cache.pkl', 'rb') as f:
    old = pickle.load(f)
with open('data/raw/cache/出馬表形式05月30日_api.cache.pkl', 'rb') as f:
    new = pickle.load(f)

old_df = old['result'].copy()
new_df = new['result'].copy()

merged = old_df[['場 R','馬名S','clogit_calib']].merge(
    new_df[['場 R','馬名S','clogit_calib']],
    on=['場 R','馬名S'], suffixes=('_old','_new'))
merged['diff'] = merged['clogit_calib_new'] - merged['clogit_calib_old']
merged['diff_abs'] = merged['diff'].abs()

print(f"比較馬数: {len(merged)}")
print(f"変化あり(>0.001): {(merged['diff_abs'] > 0.001).sum()}頭")
print()
top = merged.nlargest(20, 'diff_abs')[['場 R','馬名S','clogit_calib_old','clogit_calib_new','diff']]
print(top.to_string(index=False))
