# coding: utf-8
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import pandas as pd, numpy as np, pickle, json
import lightgbm as lgb

# ── データ読み込み
df = pd.read_parquet('data/processed/all_venues_features.parquet')
df['日付_num'] = pd.to_numeric(df['日付'], errors='coerce')
df['着順_num'] = pd.to_numeric(df['着順_num'], errors='coerce')
df = df.dropna(subset=['日付_num', '着順_num'])
df = df[df['着順_num'] < 99]
df['race_id'] = (df['日付_num'].astype(int).astype(str) + '_' +
                 df['開催'].astype(str).str.strip() + '_' +
                 df['Ｒ'].astype(str).str.strip())
df['surface'] = df['芝・ダ'].astype(str).str.strip()
df['kaisai']  = df['race_id'].str.split('_').str[1]
df['heads']   = df.groupby('race_id')['race_id'].transform('count')

# ── 特徴量（odds除外）
with open('models/lambdarank_info.json', encoding='utf-8') as f:
    info = json.load(f)
ODDS_REMOVE = set(
    ['単勝オッズ', '人気', '前走単勝オッズ', '前走人気'] +
    [f'{n}走前_単勝オッズ' for n in range(1, 11)] +
    [f'{n}走前_人気'       for n in range(1, 11)]
)
feat_cols = [c for c in info['feat_cols'] if c not in ODDS_REMOVE]

def make_label(s):
    s = pd.to_numeric(s, errors='coerce').fillna(99).astype(int)
    l = np.zeros(len(s), dtype=np.int32)
    l[s == 1] = 3; l[s == 2] = 2; l[s == 3] = 1
    return l

def build(sub, cols):
    sub = sub.sort_values('race_id').reset_index(drop=True)
    X = sub[cols].astype(float).values
    y = make_label(sub['着順_num'])
    g = sub.groupby('race_id', sort=False).size().values
    return X, y, g, sub

PARAMS = dict(objective='lambdarank', metric='ndcg', ndcg_eval_at=[1, 3],
              learning_rate=0.05, num_leaves=63, min_child_samples=20,
              subsample=0.8, colsample_bytree=0.8,
              reg_alpha=0.1, reg_lambda=1.0,
              n_jobs=-1, verbose=-1, random_state=42)

# ── 学習（train=2013-2020, val=2021-2022）
tr  = df[(df['日付_num'] >= 130101) & (df['日付_num'] <= 221231)]
val = tr[tr['日付_num'] >= 210101]
trn = tr[tr['日付_num'] <  210101]
X_tr, y_tr, g_tr, _ = build(trn, feat_cols)
X_va, y_va, g_va, _ = build(val, feat_cols)
ds_tr = lgb.Dataset(X_tr, label=y_tr, group=g_tr, free_raw_data=False)
ds_va = lgb.Dataset(X_va, label=y_va, group=g_va, free_raw_data=False, reference=ds_tr)
print('学習中...')
model = lgb.train(PARAMS, ds_tr, num_boost_round=500, valid_sets=[ds_va],
    callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(period=-1)])
print('完了')

# ── OOS予測
oos = df[df['日付_num'] >= 230101].copy().sort_values('race_id').reset_index(drop=True)
X_oos, _, _, oos = build(oos, feat_cols)
oos['score']      = model.predict(X_oos)
oos['rank_model'] = oos.groupby('race_id')['score'].rank(ascending=False, method='first')
oos['単勝オッズ'] = pd.to_numeric(oos['単勝オッズ'], errors='coerce')
oos['人気']       = pd.to_numeric(oos['人気'], errors='coerce')
oos['距離']       = pd.to_numeric(oos['距離'], errors='coerce')

# ── 配当CSV
raw = open(r'C:/horse_racing_ai/data/raw/2023年～の結果.csv', 'rb').read()
res = pd.read_csv(io.BytesIO(raw), encoding='cp932')
res.columns = res.columns.str.strip()
res['開催']    = res['開催'].astype(str).str.strip()
res['日付_num'] = pd.to_numeric(res['日付'], errors='coerce').astype('Int64')
res['tan_pay'] = pd.to_numeric(res['単勝配当'], errors='coerce')
res = res[['日付_num', '開催', '馬名S', 'tan_pay']]

# ── rank_model=1 の行とJOIN
top1 = oos[oos['rank_model'] == 1].copy()
m = top1.merge(res,
    left_on=['日付_num', 'kaisai', '馬名S'],
    right_on=['日付_num', '開催',  '馬名S'], how='inner')

# ── 表示ユーティリティ
SEP = '=' * 72
def show(sub, label, min_n=30):
    if len(sub) < min_n:
        return
    wins = sub[sub['着順_num'] == 1]
    roi  = wins['tan_pay'].sum() / 100 / len(sub) - 1
    hr   = len(wins) / len(sub)
    avg_od = sub['単勝オッズ'].mean()
    print(f'  {label:38s}  N={len(sub):>5,}  勝率={hr:.1%}  ROI={roi:+.1%}  avg_OD={avg_od:.1f}')

print()
print(SEP); print(' 芝ダ別'); print(SEP)
for s in ['芝', 'ダ']:
    show(m[m['surface'] == s], s)

print()
print(SEP); print(' 頭数別'); print(SEP)
for lo, hi, lbl in [(1,7,'～7頭'), (8,11,'8-11頭'), (12,14,'12-14頭'), (15,99,'15頭+')]:
    show(m[(m['heads'] >= lo) & (m['heads'] <= hi)], lbl)

print()
print(SEP); print(' 距離帯別'); print(SEP)
for lo, hi, lbl in [(0,1400,'短距離～1400m'), (1401,1800,'マイル1401-1800'),
                    (1801,2200,'中距離1801-2200'), (2201,9999,'長距離2200m+')]:
    show(m[(m['距離'] >= lo) & (m['距離'] <= hi)], lbl)

print()
print(SEP); print(' 芝ダ × 頭数'); print(SEP)
for s in ['芝', 'ダ']:
    for lo, hi, lbl in [(1,11,'～11頭'), (12,14,'12-14頭'), (15,99,'15頭+')]:
        show(m[(m['surface'] == s) & (m['heads'] >= lo) & (m['heads'] <= hi)],
             f'{s}  {lbl}')

print()
print(SEP); print(' 芝ダ × 距離'); print(SEP)
for s in ['芝', 'ダ']:
    for lo, hi, lbl in [(0,1400,'短距離'), (1401,1800,'マイル'), (1801,2200,'中距離'), (2201,9999,'長距離')]:
        show(m[(m['surface'] == s) & (m['距離'] >= lo) & (m['距離'] <= hi)],
             f'{s}  {lbl}')

print()
print(SEP); print(' オッズ帯別'); print(SEP)
for lo, hi, lbl in [(1,2.9,'1-2.9倍'), (3,4.9,'3-4.9倍'), (5,9.9,'5-9.9倍'),
                    (10,19.9,'10-19.9倍'), (20,999,'20倍+')]:
    show(m[(m['単勝オッズ'] >= lo) & (m['単勝オッズ'] <= hi)], lbl)

print()
print(SEP); print(' 人気別（モデルが選んだ馬の人気）'); print(SEP)
for pop in range(1, 11):
    show(m[m['人気'] == pop], f'{pop}番人気', min_n=20)
show(m[m['人気'] >= 11], '11番人気以上', min_n=20)

print()
print(SEP); print(' 年別'); print(SEP)
for yr_prefix, lbl in [('23','2023'), ('24','2024'), ('25','2025'), ('26','2026')]:
    sub = m[m['日付_num'].astype(str).str[:2] == yr_prefix]
    show(sub, lbl)

print()
print(SEP); print(' 頭数 × オッズ帯（ダート15頭+）'); print(SEP)
base = m[(m['surface'] == 'ダ') & (m['heads'] >= 15)]
show(base, 'ダート 15頭+  全オッズ')
for lo, hi, lbl in [(1,9.9,'～9.9倍'), (10,19.9,'10-19.9倍'), (20,999,'20倍+')]:
    show(base[(base['単勝オッズ'] >= lo) & (base['単勝オッズ'] <= hi)],
         f'ダート 15頭+  {lbl}')
