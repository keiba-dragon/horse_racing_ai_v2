# coding: utf-8
"""
リーク有モデル(脚質_num込み) + 前走脚質_num代入のROI検証
"""
import sys, io, pickle, json
import numpy as np
import pandas as pd
import lightgbm as lgb

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, 'data', 'processed', 'all_venues_features.parquet')

df = pd.read_parquet(DATA_FILE)
df['日付_num'] = pd.to_numeric(df['日付'], errors='coerce')
df['着順_num'] = pd.to_numeric(df['着順_num'], errors='coerce')
df = df.dropna(subset=['日付_num', '着順_num'])
df = df[df['着順_num'] < 99]
df['race_id'] = (df['日付_num'].astype(int).astype(str) + '_' +
                 df['開催'].astype(str).str.strip() + '_' +
                 df['Ｒ'].astype(str).str.strip())

ODDS_REMOVE = set(
    ['単勝オッズ', '人気', '前走単勝オッズ', '前走人気'] +
    [f'{n}走前_単勝オッズ' for n in range(1, 11)] +
    [f'{n}走前_人気' for n in range(1, 11)]
)
# 脚質_num は除外しない（リークあり学習）
EXCLUDE = {
    '走破タイム', '走破タイム_sec', '着差', '2角', '3角', '4角',
    '上り3F', 'PCI', 'PCI3', 'RPCI_x', 'RPCI_y',
    '上3F地点差_x', '上3F地点差_y',
    'タイム指数', '上り3F_指数', 'Ave-3F', '平均速度', '-3F平均速度', '上り3F平均速度',
    '単勝配当', '複勝配当', '枠連', '馬連', '馬単', '３連複', '３連単',
    '好走', '賞金',
    'Ｍ', '日付', '開催', 'Ｒ', 'レース名', '限定', '馬名S', 'Ｃ',
    '性別', '騎手', '調教師', '種牡馬', '母父馬', '生産者', '毛色',
    '馬記号', '生年月日', '市場取引価格(万/最終)', '取引市場(最終)', '産地',
    '前走開催', '前走レース名', '替', '前騎手',
    '1走前_開催', '2走前_開催', '3走前_開催', '4走前_開催', '5走前_開催',
    '6走前_開催', '7走前_開催', '8走前_開催', '9走前_開催', '10走前_開催',
    '前走日付', '前走Ｒ',
    '着順', '着順_num', '前走着順_num',
    '日付_num', 'race_id',
}

num_cols = df.select_dtypes(include='number').columns.tolist()
feat_cols = [c for c in num_cols if c not in EXCLUDE and c not in ODDS_REMOVE]
print(f'特徴量: {len(feat_cols)} 列 (脚質_num 含む: {"脚質_num" in feat_cols})')
print(f'前走脚質_num 含む: {"前走脚質_num" in feat_cols}')


def make_label(s):
    s = pd.to_numeric(s, errors='coerce').fillna(99).astype(int)
    l = np.zeros(len(s), dtype=np.int32)
    l[s == 1] = 3; l[s == 2] = 2; l[s == 3] = 1
    return l


def build(d):
    d = d.sort_values('race_id').reset_index(drop=True)
    X = d[feat_cols].astype(float).values
    y = make_label(d['着順_num'])
    g = d.groupby('race_id', sort=False).size().values
    return X, y, g


tr = df[(df['日付_num'] >= 130101) & (df['日付_num'] <= 221231)]
val = tr[tr['日付_num'] >= 210101]
trn = tr[tr['日付_num'] < 210101]
print(f'学習: {len(trn):,}行 / valid: {len(val):,}行')

X_tr, y_tr, g_tr = build(trn)
X_va, y_va, g_va = build(val)
ds_tr = lgb.Dataset(X_tr, label=y_tr, group=g_tr, free_raw_data=False)
ds_va = lgb.Dataset(X_va, label=y_va, group=g_va, free_raw_data=False, reference=ds_tr)

LGBM_PARAMS = dict(
    objective='lambdarank', metric='ndcg', ndcg_eval_at=[1, 3],
    learning_rate=0.05, num_leaves=63, min_child_samples=20,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0,
    n_jobs=-1, verbose=-1, random_state=42,
)
print('学習中...')
model_leak = lgb.train(
    LGBM_PARAMS, ds_tr, num_boost_round=500, valid_sets=[ds_va],
    callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(50)],
)
print(f'best_iter={model_leak.best_iteration}')

# OOS: 脚質_num を 前走脚質_num で代入して予測
oos = df[df['日付_num'] >= 230101].copy()
idx_kashitsu = feat_cols.index('脚質_num') if '脚質_num' in feat_cols else None
print(f'脚質_num 列インデックス: {idx_kashitsu}')

X_oos = np.full((len(oos), len(feat_cols)), np.nan, dtype=np.float32)
for i, col in enumerate(feat_cols):
    src = '前走脚質_num' if col == '脚質_num' else col
    if src in oos.columns:
        X_oos[:, i] = pd.to_numeric(oos[src], errors='coerce').values

oos['_score'] = model_leak.predict(X_oos)
oos['rank_model'] = oos.groupby('race_id')['_score'].rank(ascending=False, method='first')
oos['pop_num'] = pd.to_numeric(oos['人気'], errors='coerce')
oos['odds_num'] = pd.to_numeric(oos['単勝オッズ'], errors='coerce')
oos['yr'] = oos['日付_num'] // 10000

top1 = oos[oos['rank_model'] == 1]


def roi_table(d, label):
    print(f'\n=== {label} ===')
    for yr in sorted(d['yr'].unique()):
        sub = d[d['yr'] == yr]
        won = sub['着順_num'] == 1
        r = (sub.loc[won, 'odds_num'] * 100).sum() / (len(sub) * 100) - 1
        print(f'  20{yr:02d}: {len(sub)}R  win={won.mean():.3f}  ROI={r:+.3f}')
    won = d['着順_num'] == 1
    r = (d.loc[won, 'odds_num'] * 100).sum() / (len(d) * 100) - 1
    print(f'  Total: {len(d)}R  win={won.mean():.3f}  ROI={r:+.3f}')


roi_table(top1, 'リーク学習 + 前走脚質代入 (rank=1 全体)')
roi_table(top1[top1['pop_num'] >= 2], 'リーク学習 + 前走脚質代入 (rank=1 × pop>=2)')
roi_table(top1[top1['pop_num'] >= 4], 'リーク学習 + 前走脚質代入 (rank=1 × pop>=4)')
