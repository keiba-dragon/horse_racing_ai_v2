import pickle, pandas as pd, json, sys
sys.stdout.reconfigure(encoding='utf-8')

# 05-31予測
with open('data/raw/cache/出馬表形式05月31日_api.cache.pkl', 'rb') as f:
    cache = pickle.load(f)
df = cache['result'].copy()

# オッズ
try:
    with open('data/raw/cache/20260531.odds.json', encoding='utf-8') as f:
        odds = json.load(f)
except:
    odds = {}

derby = df[df['場 R'].str.contains(r'東11$')].copy()
derby['odds'] = derby['馬名S'].map(odds)
derby = derby.sort_values('clogit_calib', ascending=False)

cols = ['馬名S','clogit_calib','clogit_calib_top2','clogit_calib_top3','odds']
cols = [c for c in cols if c in derby.columns]
print("=== 東京11R 日本ダービー（修正後）===")
print(derby[cols].to_string(index=False))
print()

# 修正前後の3着内率比較（05-30で検証）
print("=== 05-30: 3着内率も修正前後で変わったか ===")
with open('data/raw/cache/20260530_old.cache.pkl', 'rb') as f:
    old = pickle.load(f)
with open('data/raw/cache/20260530_new.cache.pkl', 'rb') as f:
    new = pickle.load(f)

if 'clogit_calib_top3' in old['result'].columns:
    merged = old['result'][['場 R','馬名S','clogit_calib','clogit_calib_top3']].merge(
        new['result'][['場 R','馬名S','clogit_calib','clogit_calib_top3']],
        on=['場 R','馬名S'], suffixes=('_old','_new'))
    merged['diff_top3'] = (merged['clogit_calib_top3_new'] - merged['clogit_calib_top3_old']).abs()
    merged['diff_win']  = (merged['clogit_calib_new'] - merged['clogit_calib_old']).abs()
    print(f"3着内率が変化(>0.01): {(merged['diff_top3']>0.01).sum()}頭 / {len(merged)}頭")
    print(f"勝率が変化(>0.01):    {(merged['diff_win']>0.01).sum()}頭 / {len(merged)}頭")
    print()
    top = merged.nlargest(10,'diff_top3')[['場 R','馬名S','clogit_calib_top3_old','clogit_calib_top3_new','diff_top3']]
    print(top.to_string(index=False))
else:
    print("clogit_calib_top3 列なし")
