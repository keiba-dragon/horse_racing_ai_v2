# coding: utf-8
import sys, os, pickle, pandas as pd, numpy as np, re
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'src')
from save_conditional_logit import prepare

with open('models/final_model.pkl', 'rb') as f:
    artifacts = pickle.load(f)
arts = artifacts['artifacts']

df_full = pd.read_parquet('data/processed/all_venues_features.parquet')
if '距離_num' not in df_full.columns:
    df_full['距離_num'] = pd.to_numeric(
        df_full['距離'].astype(str).str.extract(r'(\d+)')[0], errors='coerce')

VENUE_MAP = {'東':'05','中':'06','中京':'07','名':'07','京':'08','阪':'09',
             '新':'04','福':'03','函':'02','札':'01','小':'10'}

def make_race_id(row):
    d = str(int(row['_date_num'])).zfill(6)
    k = str(row['開催'])
    rn = str(int(float(row['Ｒ']))).zfill(2)
    m = re.match(r'^(\d+)([^\d]+)(\d+)$', k.strip())
    if not m: return ''
    kai, v_let, day = m.group(1).zfill(2), m.group(2), m.group(3).zfill(2)
    vc = VENUE_MAP.get(v_let, '00')
    return '20' + d[:2] + vc + kai + day + rn

df_full['_date_num'] = pd.to_numeric(df_full['日付'], errors='coerce')
df_full['race_id'] = df_full.apply(make_race_id, axis=1)
df_full['着順_num'] = pd.to_numeric(
    df_full.get('着順_num', df_full.get('着順', pd.Series(index=df_full.index))),
    errors='coerce')

baba_map = {'良':0, '稍重':1, '重':2, '不良':3}
if '1走前_馬場状態' in df_full.columns:
    df_full['1走前_馬場状態'] = (df_full['1走前_馬場状態'].map(baba_map)
        .combine_first(pd.to_numeric(df_full['1走前_馬場状態'], errors='coerce')))

seg_def = {
    '芝短': lambda d: (d['芝・ダ']=='芝') & (d['距離_num'] <= 1400),
    '芝中': lambda d: (d['芝・ダ']=='芝') & (d['距離_num'].between(1401, 2000)),
    '芝長': lambda d: (d['芝・ダ']=='芝') & (d['距離_num'] > 2000),
    'ダ':   lambda d: (d['芝・ダ']=='ダ') & (d['距離_num'] > 1400),
    'ダ短': lambda d: (d['芝・ダ']=='ダ') & (d['距離_num'] <= 1400),
}
seg_label = {'芝短':'芝短','芝中':'芝中','芝長':'芝長','ダ':'ダ長','ダ短':'ダ短'}

all_rows = []

for seg_key, art in arts.items():
    if seg_key not in seg_def:
        continue
    feat_cols = art['feat_cols']
    isnan_feats = [f for f in feat_cols if f.endswith('_isnan')]

    df = df_full.copy()
    if 'クラス_rank' in df.columns:
        df = df[df['クラス_rank'] != 1.0]
    df = df[seg_def[seg_key](df)].copy()
    df = df[df['_date_num'] >= 230101].copy()
    df = df[df['着順_num'].notna()].copy()
    if len(df) < 100:
        continue

    for iso_col in isnan_feats:
        base_col = iso_col.replace('_isnan', '')
        df[iso_col] = df[base_col].isna().astype(float) if base_col in df.columns else 0.0

    df2 = df.reset_index(drop=True)
    try:
        X, y, gs, n, n_races, *_ = prepare(
            df2, feat_cols,
            scaler=art['scaler'], poly2=art['poly2'],
            inter_scaler2=art['inter_scaler2'], top_idx=art['top_idx'],
            poly3=art['poly3'], inter_scaler3=art['inter_scaler3'], top_idx3=art['top_idx3'])
    except Exception as e:
        print(f'{seg_key} skip: {e}')
        continue

    lin = X @ art['coef']
    df2['_lin'] = lin

    probs_dict = {}
    for rk, grp in df2.groupby('race_id', sort=False):
        v = grp['_lin'].values
        e = np.exp(v - v.max())
        p = e / e.sum()
        for i, pi in zip(grp.index, p):
            probs_dict[i] = pi

    df2['_prob_raw'] = pd.Series(probs_dict)
    df2['_prob_calib'] = art['isotonic'].predict(df2['_prob_raw'].values)
    df2['won'] = (df2['着順_num'] == 1).astype(int)
    df2['_is_top'] = df2.groupby('race_id')['_prob_raw'].transform(lambda x: x == x.max())
    df2['_n'] = df2.groupby('race_id')['race_id'].transform('count')
    df2['_seg'] = seg_label[seg_key]
    all_rows.append(df2[['race_id', '_prob_calib', 'won', '_is_top', '_n', '_seg']].copy())

df_all = pd.concat(all_rows, ignore_index=True)

print(f'全体平均頭数: {df_all["_n"].mean():.1f}頭')
print(f'全体平均勝率: {df_all["won"].mean()*100:.1f}% (=1/{1/df_all["won"].mean():.1f}頭に1頭)')
print()

# rank=1馬（各レース最高確率）の勝率
df_top = df_all[df_all['_is_top']].copy()
print(f'rank=1: {len(df_top):,}レース, 勝率={df_top["won"].mean()*100:.1f}%')
print()

# rank=1のキャリブレーション
bins   = [0, 0.05, 0.08, 0.11, 0.14, 0.18, 0.25, 1.0]
labels = ['<5%','5-8%','8-11%','11-14%','14-18%','18-25%','25%+']
df_top['bin'] = pd.cut(df_top['_prob_calib'], bins=bins, labels=labels)
ct = df_top.groupby('bin', observed=True).agg(
    n=('won','count'), wins=('won','sum'), avg_prob=('_prob_calib','mean')
).reset_index()
ct['actual'] = ct['wins'] / ct['n']

print('=== rank=1馬のキャリブレーション（OOS 2023-2026）===')
print(f'{"AI確率帯":10} {"N":>6} {"勝ち":>5} {"AI平均":>8} {"実績勝率":>8} {"乖離":>7}')
for _, row in ct.iterrows():
    diff = (row['actual'] - row['avg_prob']) * 100
    print(f'{str(row["bin"]):10} {row["n"]:6,} {row["wins"]:5,}  '
          f'{row["avg_prob"]*100:6.1f}%   {row["actual"]*100:6.1f}%  {diff:+.1f}pp')

print()
# 全馬の完全キャリブレーション
df_all['bin'] = pd.cut(df_all['_prob_calib'], bins=bins, labels=labels)
ca = df_all.groupby('bin', observed=True).agg(
    n=('won','count'), wins=('won','sum'), avg_prob=('_prob_calib','mean')
).reset_index()
ca['actual'] = ca['wins'] / ca['n']
print('=== 全馬キャリブレーション（再掲）===')
print(f'{"AI確率帯":10} {"N":>7} {"勝ち":>5} {"AI平均":>8} {"実績勝率":>8} {"乖離":>7}')
for _, row in ca.iterrows():
    diff = (row['actual'] - row['avg_prob']) * 100
    print(f'{str(row["bin"]):10} {row["n"]:7,} {row["wins"]:5,}  '
          f'{row["avg_prob"]*100:6.1f}%   {row["actual"]*100:6.1f}%  {diff:+.1f}pp')

# <5%帯の解説
df_low = df_all[df_all['_prob_calib'] < 0.05]
low_avg_n = df_low['_n'].mean()
print(f'\n<5%帯の平均頭数: {low_avg_n:.1f}頭 → random勝率={1/low_avg_n*100:.1f}%')
