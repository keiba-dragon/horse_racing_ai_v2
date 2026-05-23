# coding: utf-8
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import pandas as pd, numpy as np, pickle, json
import lightgbm as lgb

df = pd.read_parquet('data/processed/all_venues_features.parquet')
df['日付_num'] = pd.to_numeric(df['日付'], errors='coerce')
df['着順_num'] = pd.to_numeric(df['着順_num'], errors='coerce')
df = df.dropna(subset=['日付_num', '着順_num'])
df = df[df['着順_num'] < 99]
df['race_id'] = (df['日付_num'].astype(int).astype(str) + '_' +
                 df['開催'].astype(str).str.strip() + '_' +
                 df['Ｒ'].astype(str).str.strip())
df['surface'] = df['芝・ダ'].astype(str).str.strip()
df['year']    = df['日付_num'].astype(str).str[:2]
df['kaisai']  = df['race_id'].str.split('_').str[1]

with open('models/lambdarank_info.json', encoding='utf-8') as f:
    info = json.load(f)

ODDS_REMOVE = set([
    '単勝オッズ', '人気', '前走単勝オッズ', '前走人気',
    '1走前_単勝オッズ', '1走前_人気',
    '2走前_単勝オッズ', '2走前_人気',
    '3走前_単勝オッズ', '3走前_人気',
    '4走前_単勝オッズ', '4走前_人気',
    '5走前_単勝オッズ', '5走前_人気',
    '6走前_単勝オッズ', '6走前_人気',
    '7走前_単勝オッズ', '7走前_人気',
    '8走前_単勝オッズ', '8走前_人気',
    '9走前_単勝オッズ', '9走前_人気',
    '10走前_単勝オッズ', '10走前_人気',
])
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

with open(r'C:\horse_racing_ai\data\raw\2023年～の結果.csv', 'rb') as f:
    raw = f.read()
res = pd.read_csv(io.BytesIO(raw), encoding='cp932')
res.columns = res.columns.str.strip()
res['開催']    = res['開催'].astype(str).str.strip()
res['日付_num'] = pd.to_numeric(res['日付'], errors='coerce').astype('Int64')
res['tan_pay'] = pd.to_numeric(res['単勝配当'], errors='coerce')
res = res[['日付_num', '開催', '馬名S', 'tan_pay']]

SEP = '=' * 62

def check_roi(label, period_df):
    period_df = period_df.copy().sort_values('race_id').reset_index(drop=True)
    X, _, _, period_df = build(period_df, feat_cols)
    period_df['score']      = model.predict(X)
    period_df['rank_model'] = period_df.groupby('race_id')['score'].rank(
        ascending=False, method='first')
    m = period_df.merge(res,
        left_on=['日付_num', 'kaisai', '馬名S'],
        right_on=['日付_num', '開催',  '馬名S'], how='inner')
    top1 = m[m['rank_model'] == 1]
    if len(top1) < 30:
        print(f'  {label}: データ不足')
        return
    wins = top1[top1['着順_num'] == 1]
    roi  = wins['tan_pay'].sum() / 100 / len(top1) - 1
    hr   = len(wins) / len(top1)
    print(f'  {label:35s}  N={len(top1):>5,}  勝率={hr:.1%}  単ROI={roi:+.1%}')

# ── テスト2: val期間 vs OOS ROI比較
print()
print(SEP)
print(' テスト2: val期間 vs OOS のROI比較')
print(' リークなら: val << OOS は起きにくい（どちらも高い or 均一）')
print(' リークがあれば: 学習に近いほど異常に高くなる')
print(SEP)
check_roi('val 2021-2022 (early stop用)',
    df[(df['日付_num'] >= 210101) & (df['日付_num'] <= 221231)])
check_roi('OOS 2023', df[(df['日付_num'] >= 230101) & (df['日付_num'] <= 231231)])
check_roi('OOS 2024', df[(df['日付_num'] >= 240101) & (df['日付_num'] <= 241231)])
check_roi('OOS 2025', df[(df['日付_num'] >= 250101) & (df['日付_num'] <= 251231)])
check_roi('OOS 2026', df[df['日付_num'] >= 260101])

# ── テスト3: 選ばれた馬の人気分布
print()
print(SEP)
print(' テスト3: モデルが選ぶ馬の人気分布')
print(' 市場コピーなら1番人気ばかり選ぶ')
print(SEP)
oos_all = df[df['日付_num'] >= 230101].copy().sort_values('race_id').reset_index(drop=True)
X, _, _, oos_all = build(oos_all, feat_cols)
oos_all['score']      = model.predict(X)
oos_all['rank_model'] = oos_all.groupby('race_id')['score'].rank(ascending=False, method='first')
top1 = oos_all[oos_all['rank_model'] == 1]
pop_dist = top1['人気'].value_counts().sort_index().head(10)
total = len(top1)
print(f'  N={total:,}')
for pop, cnt in pop_dist.items():
    bar = '#' * int(cnt / total * 30)
    print(f'  {int(pop):2d}番人気: {cnt:>4,} ({cnt/total:4.1%})  {bar}')

# 1番人気と何番人気を選んでいるかの比較
pop1_pct  = (top1['人気'] == 1).mean()
pop23_pct = top1['人気'].between(2, 3).mean()
pop4p_pct = (top1['人気'] >= 4).mean()
print(f'\n  1番人気選択率: {pop1_pct:.1%}  (ランダムなら約1/平均頭数≈10%)')
print(f'  2-3番人気:     {pop23_pct:.1%}')
print(f'  4番人気以下:   {pop4p_pct:.1%}')
