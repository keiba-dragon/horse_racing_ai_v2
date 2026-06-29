# coding: utf-8
import sys, os, pickle, pandas as pd, numpy as np, re
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'src')

with open('models/accuracy_model.pkl', 'rb') as f:
    acc_model = pickle.load(f)

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
    m = re.match(r'^(\d+)([^\d]+)(\d+)', k.strip())
    if not m: return ''
    kai, v_let, day = m.group(1).zfill(2), m.group(2), m.group(3).zfill(2)
    return '20' + d[:2] + VENUE_MAP.get(v_let, '00') + kai + day + rn

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
    'ダ長': lambda d: (d['芝・ダ']=='ダ') & (d['距離_num'] > 1400),
    'ダ短': lambda d: (d['芝・ダ']=='ダ') & (d['距離_num'] <= 1400),
}

all_rows = []

for seg_key, art in acc_model.items():
    if seg_key not in seg_def:
        continue
    feat_cols = art['feat_cols']
    scaler    = art['scaler']
    coef      = art['coef']
    iso       = art.get('isotonic')

    df = df_full.copy()
    if 'クラス_rank' in df.columns:
        df = df[df['クラス_rank'] != 1.0]
    df = df[seg_def[seg_key](df)].copy()
    df = df[df['_date_num'] >= 230101].copy()
    df = df[df['着順_num'].notna()].copy()
    if len(df) < 100:
        continue

    # 特徴量ベクトル構築
    rows_X = []
    for _, row in df.iterrows():
        fv = []
        for f in feat_cols:
            if f.endswith('_isnan'):
                base_f = f[:-6]
                fv.append(1.0 if pd.isna(row.get(base_f)) else 0.0)
            else:
                v = row.get(f, np.nan)
                try:
                    fv.append(float(v) if not pd.isna(v) else 0.0)
                except (ValueError, TypeError):
                    fv.append(0.0)
        rows_X.append(fv)

    X = np.array(rows_X, dtype=float)
    try:
        scores = scaler.transform(X) @ coef
    except Exception as e:
        print(f'{seg_key} skip: {e}')
        continue

    df2 = df.reset_index(drop=True).copy()
    df2['_score'] = scores

    # race-wise softmax → prob_raw
    probs_dict = {}
    for rk, grp in df2.groupby('race_id', sort=False):
        v = grp['_score'].values
        e = np.exp(v - v.max())
        p = e / e.sum()
        for i, pi in zip(grp.index, p):
            probs_dict[i] = pi
    df2['_prob_raw'] = pd.Series(probs_dict)

    # isotonic calibration があれば適用
    if iso is not None:
        df2['_prob_calib'] = iso.predict(df2['_prob_raw'].values)
    else:
        df2['_prob_calib'] = df2['_prob_raw']

    df2['won']     = (df2['着順_num'] == 1).astype(int)
    df2['_is_top'] = df2.groupby('race_id')['_score'].transform(lambda x: x == x.max())
    df2['_seg']    = seg_key
    all_rows.append(df2[['race_id', '_prob_calib', 'won', '_is_top', '_seg']].copy())
    print(f'{seg_key}: {len(df2):,}頭, 勝ち={df2["won"].sum()}頭')

df_all = pd.concat(all_rows, ignore_index=True)

print(f'\n全体平均勝率: {df_all["won"].mean()*100:.1f}%')
print()

# ── 全馬キャリブレーション ────────────────────────────────────────
bins   = [0, 0.05, 0.08, 0.11, 0.14, 0.18, 0.25, 1.0]
labels = ['<5%','5-8%','8-11%','11-14%','14-18%','18-25%','25%+']
df_all['bin'] = pd.cut(df_all['_prob_calib'], bins=bins, labels=labels)
ca = df_all.groupby('bin', observed=True).agg(
    n=('won','count'), wins=('won','sum'), avg_prob=('_prob_calib','mean')
).reset_index()
ca['actual'] = ca['wins'] / ca['n']

print('=== accuracy_model OOS(2023-2026) キャリブレーション（全馬）===')
print(f'{"AI確率帯":10} {"N":>7} {"勝ち":>5} {"AI平均":>8} {"実績勝率":>8} {"乖離":>7}')
for _, row in ca.iterrows():
    diff = (row['actual'] - row['avg_prob']) * 100
    print(f'{str(row["bin"]):10} {row["n"]:7,} {row["wins"]:5,}  '
          f'{row["avg_prob"]*100:6.1f}%   {row["actual"]*100:6.1f}%  {diff:+.1f}pp')

print()

# ── rank=1（各レース最高スコア馬）だけ ──────────────────────────────
df_top = df_all[df_all['_is_top']].copy()
df_top['bin'] = pd.cut(df_top['_prob_calib'], bins=bins, labels=labels)
ct = df_top.groupby('bin', observed=True).agg(
    n=('won','count'), wins=('won','sum'), avg_prob=('_prob_calib','mean')
).reset_index()
ct['actual'] = ct['wins'] / ct['n']

print(f'=== accuracy_model rank=1 の実績（{len(df_top):,}レース）===')
print(f'{"AI確率帯":10} {"N":>6} {"勝ち":>5} {"AI平均":>8} {"実績勝率":>8} {"乖離":>7}')
for _, row in ct.iterrows():
    diff = (row['actual'] - row['avg_prob']) * 100
    print(f'{str(row["bin"]):10} {row["n"]:6,} {row["wins"]:5,}  '
          f'{row["avg_prob"]*100:6.1f}%   {row["actual"]*100:6.1f}%  {diff:+.1f}pp')

print(f'\nrank=1 全体勝率: {df_top["won"].mean()*100:.1f}%  '
      f'（全体平均{df_all["won"].mean()*100:.1f}%比 '
      f'{(df_top["won"].mean()-df_all["won"].mean())*100:+.1f}pp）')

# ── セグメント別 rank=1 勝率 ────────────────────────────────────────
print()
print('=== セグメント別 rank=1 勝率 ===')
for seg in ['ダ長','ダ短','芝短','芝中','芝長']:
    s = df_top[df_top['_seg']==seg]
    if len(s) == 0: continue
    wr = s['won'].mean()*100
    avg_p = s['_prob_calib'].mean()*100
    print(f'  {seg}: {len(s):,}レース  勝率={wr:.1f}%  AI平均={avg_p:.1f}%')

# ── ランク別実績（芝中のみ）─────────────────────────────────────────
print()
print('=== 芝中: AIランク別 実績勝率 ===')
df_mid = df_all[df_all['_seg']=='芝中'].copy()
df_mid = df_mid.join(
    df_mid.groupby('race_id')['_prob_calib'].rank(ascending=False, method='first').rename('_rank'),
    how='left'
)
rk_gr = df_mid.groupby('_rank').agg(n=('won','count'), wins=('won','sum'), avg_prob=('_prob_calib','mean')).reset_index()
rk_gr['actual'] = rk_gr['wins'] / rk_gr['n']
for _, row in rk_gr[rk_gr['_rank'] <= 8].iterrows():
    bar = '#' * int(row['actual'] * 200)
    print(f'  rank{int(row["_rank"]):2d}: 実績={row["actual"]*100:5.1f}%  AI={row["avg_prob"]*100:5.1f}%  {bar}')
