# coding: utf-8
"""
オッズなし lambdarank モデルを学習して保存する（初回のみ実行）

出力:
  models/lambdarank_no_odds.pkl
  models/lambdarank_no_odds_info.json
"""
import sys, io, os, json, pickle
import numpy as np
import pandas as pd
import lightgbm as lgb

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE  = os.path.join(BASE_DIR, 'data', 'processed', 'all_venues_features.parquet')
MODEL_DIR  = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'lambdarank_no_odds.pkl')
INFO_PATH  = os.path.join(MODEL_DIR, 'lambdarank_no_odds_info.json')

ODDS_REMOVE = set(
    ['単勝オッズ', '人気', '前走単勝オッズ', '前走人気'] +
    [f'{n}走前_単勝オッズ' for n in range(1, 11)] +
    [f'{n}走前_人気'       for n in range(1, 11)]
)

EXCLUDE = {
    '走破タイム', '走破タイム_sec', '着差',
    '2角', '3角', '4角',
    '上り3F', 'PCI', 'PCI3', 'RPCI_x', 'RPCI_y',
    '上3F地点差_x', '上3F地点差_y',
    'タイム指数', '上り3F_指数',
    'Ave-3F', '平均速度', '-3F平均速度', '上り3F平均速度',
    '単勝配当', '複勝配当', '枠連', '馬連', '馬単', '３連複', '３連単',
    '好走', '賞金',
    'Ｍ', '日付', '開催', 'Ｒ', 'レース名', '限定', '馬名S', 'Ｃ',
    '性別', '騎手', '調教師', '種牡馬', '母父馬', '生産者', '毛色',
    '馬記号', '生年月日', '市場取引価格(万/最終)', '取引市場(最終)', '産地',
    '前走開催', '前走レース名', '替', '前騎手',
    '1走前_開催', '2走前_開催', '3走前_開催', '4走前_開催',
    '5走前_開催', '6走前_開催', '7走前_開催', '8走前_開催',
    '9走前_開催', '10走前_開催',
    '前走日付', '前走Ｒ',
    '着順', '着順_num', '前走着順_num',
    '日付_num', 'race_id',
    '脚質_num',  # 当レースの実走法 → レース後確定のためリーク
}

LGBM_PARAMS = dict(
    objective='lambdarank', metric='ndcg', ndcg_eval_at=[1, 3],
    learning_rate=0.05, num_leaves=63, min_child_samples=20,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0,
    n_jobs=-1, verbose=-1, random_state=42,
)


def make_label(s):
    s = pd.to_numeric(s, errors='coerce').fillna(99).astype(int)
    l = np.zeros(len(s), dtype=np.int32)
    l[s == 1] = 3; l[s == 2] = 2; l[s == 3] = 1
    return l


def build(df, feat_cols):
    df = df.sort_values('race_id').reset_index(drop=True)
    X  = df[feat_cols].astype(float).values
    y  = make_label(df['着順_num'])
    g  = df.groupby('race_id', sort=False).size().values
    return X, y, g


def main():
    print(f'データ読み込み: {DATA_FILE}')
    df = pd.read_parquet(DATA_FILE)
    df['日付_num'] = pd.to_numeric(df['日付'], errors='coerce')
    df['着順_num'] = pd.to_numeric(df['着順_num'], errors='coerce')
    df = df.dropna(subset=['日付_num', '着順_num'])
    df = df[df['着順_num'] < 99]
    df['race_id'] = (df['日付_num'].astype(int).astype(str) + '_' +
                     df['開催'].astype(str).str.strip() + '_' +
                     df['Ｒ'].astype(str).str.strip())

    num_cols  = df.select_dtypes(include='number').columns.tolist()
    feat_cols = [c for c in num_cols if c not in EXCLUDE and c not in ODDS_REMOVE]
    print(f'特徴量: {len(feat_cols)}列（オッズ・人気除外済み）')

    tr  = df[(df['日付_num'] >= 130101) & (df['日付_num'] <= 221231)]
    val = tr[tr['日付_num'] >= 210101]
    trn = tr[tr['日付_num'] <  210101]
    print(f'学習: {len(trn):,}行 / valid: {len(val):,}行')

    X_tr, y_tr, g_tr = build(trn, feat_cols)
    X_va, y_va, g_va = build(val,  feat_cols)
    ds_tr = lgb.Dataset(X_tr, label=y_tr, group=g_tr, free_raw_data=False)
    ds_va = lgb.Dataset(X_va, label=y_va, group=g_va, free_raw_data=False, reference=ds_tr)

    print('学習中...')
    model = lgb.train(
        LGBM_PARAMS, ds_tr, num_boost_round=500, valid_sets=[ds_va],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(50)],
    )

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)

    info = {
        'feat_cols':      feat_cols,
        'train_range':    [130101, 221231],
        'best_iteration': model.best_iteration,
        'odds_removed':   True,
    }
    with open(INFO_PATH, 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print(f'保存完了: {MODEL_PATH}  (best_iter={model.best_iteration})')
    print(f'         {INFO_PATH}')


if __name__ == '__main__':
    main()
