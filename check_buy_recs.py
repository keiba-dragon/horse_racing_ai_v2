# coding: utf-8
import pickle, re, json, sys, io
import pandas as pd, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

with open('data/raw/cache/20260530.cache.pkl', 'rb') as f:
    cache = pickle.load(f)
df = cache['result'].copy()
df.columns = df.columns.astype(object)

with open('data/raw/cache/20260530.odds.json', encoding='utf-8') as f:
    odds_dict = json.load(f)

print(f'行数: {len(df)}, オッズ件数: {len(odds_dict)}')
print(f'列: {[c for c in df.columns if c in ["馬名S","開催","Ｒ","clogit_calib","clogit_score","clogit_factor","_cls_group","キャリア"]]}')

df['_horse'] = df['馬名S'].astype(str).str.strip()
df['_venue'] = df['開催'].astype(str).str.extract(r'([^\d]+)')[0]
df['_R']     = pd.to_numeric(df['Ｒ'], errors='coerce')
# レース単位グループ: 開催+R で一意にする
df['_race']  = df['開催'].astype(str) + '_' + df['Ｒ'].astype(str)

df['_yahoo_odds'] = df['_horse'].map(odds_dict)
df['_mprob']      = 1.0 / df['_yahoo_odds'].clip(lower=1.0)
print(f'オッズNaN: {df["_yahoo_odds"].isna().sum()}頭 / {len(df)}頭')

factor = df['clogit_factor'].fillna(0.16) if 'clogit_factor' in df.columns else pd.Series(0.16, index=df.index)
has_odds = df['_mprob'].notna()
df.loc[has_odds, 'clogit_score'] = df.loc[has_odds, 'clogit_calib'] - factor[has_odds] * df.loc[has_odds, '_mprob']
df['clogit_rank'] = df.groupby('_race')['clogit_score'].rank(ascending=False, method='first')
df['_ev'] = df['clogit_calib'] - df['_mprob'] * 0.80
df['_gap'] = df.groupby('_race', sort=False)['clogit_calib'].transform(
    lambda x: x.nlargest(2).iloc[0] - x.nlargest(2).iloc[1] if x.dropna().shape[0] >= 2 else 0.0
)

def cls(v):
    v = str(v)
    if '未勝利' in v: return '未勝利'
    if '新馬' in v: return '新馬'
    return '1勝+'
df['_class'] = df['_cls_group'].apply(cls) if '_cls_group' in df.columns else '1勝+'
df['_career'] = pd.to_numeric(df.get('キャリア', 0), errors='coerce').fillna(0)
df['_rank'] = df['clogit_rank'].fillna(99).astype(int)

df['_buy'] = (
    (df['_rank'] == 1) &
    (df['_gap'] >= 0.15) &
    (df['_ev'] >= 0.0) &
    ((df['_class'] == '1勝+') | ((df['_class'] == '未勝利') & (df['_career'] >= 5)))
)

print()
print('rank=1 全レース (gap/ev/買い):')
print(f'{"":3} {"R":>2}  {"馬名":<18} {"calib":>6} {"gap":>6} {"ev":>7} {"クラス":>6} {"買い"}')
print('-' * 72)
rank1 = df[df['_rank'] == 1].sort_values(['_venue', '_R'])
for _, r in rank1.iterrows():
    mark = '★買い' if r['_buy'] else '見送'
    print(f'{str(r["_venue"]):>3}{int(r["_R"]):>2}R  {str(r["_horse"]):<18} {r["clogit_calib"]*100:5.1f}%  {r["_gap"]:+.3f}  {r["_ev"]:+.3f}  {str(r["_class"]):>6}  {mark}')

print(f'\n買い推奨合計: {int(df["_buy"].sum())}件')
