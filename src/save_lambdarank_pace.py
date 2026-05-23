# coding: utf-8
"""
展開予想特徴量（レース単位の前走脚質集計）を追加してlambdarankを学習。

既存の 01_make_features.py の レース内_* 特徴量は「開催×日付」レベルで
計算されており、1レースの出馬表単位ではない。
このスクリプトでは race_id（1レース）単位で前走脚質_numを集計し直す。

追加する特徴量 (全て前走データ=事前情報、リークなし):
  race_前走逃げ数        … そのレースに逃げ馬が何頭いるか
  race_前走先行数        … 先行馬の頭数
  race_前走前目頭数      … 逃げ+先行の合計
  race_前走前目割合      … 前目馬/(出走頭数)
  race_前走脚質平均      … レース全体の平均推定脚質
  race_前走脚質std       … レース全体の脚質多様性
  pace_単独逃げ          … 自馬が逃げ馬かつ他に逃げ馬がいない(1/0)
  pace_前目競争度        … 自馬の前走脚質が「前目」な馬に対する競争相手密度
  pace_有利スコア        … 自馬の推定脚質 × レース展開の相性
  自馬_対_レース脚質差   … 自馬の前走脚質 - レース平均脚質

出力:
  models/lambdarank_pace.pkl
  models/lambdarank_pace_info.json
"""
import sys, io, os, json, pickle
import numpy as np
import pandas as pd
import lightgbm as lgb

if not (isinstance(sys.stdout, io.TextIOWrapper) and (sys.stdout.encoding or '').lower().startswith('utf')):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if not (isinstance(sys.stderr, io.TextIOWrapper) and (sys.stderr.encoding or '').lower().startswith('utf')):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE  = os.path.join(BASE_DIR, 'data', 'processed', 'all_venues_features.parquet')
MODEL_DIR  = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'lambdarank_pace.pkl')
INFO_PATH  = os.path.join(MODEL_DIR, 'lambdarank_pace_info.json')

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
    '脚質_num',  # 当レース実走法 → リーク
    'レース印１', '前走レース印１',  # 事前取得不可（レース後に付く評価）
}

PACE_COLS = [
    'race_前走逃げ数', 'race_前走先行数', 'race_前走前目頭数',
    'race_前走前目割合', 'race_前走脚質平均', 'race_前走脚質std',
    'pace_単独逃げ', 'pace_前目競争度', 'pace_有利スコア',
    '自馬_対_レース脚質差',
]

LGBM_PARAMS = dict(
    objective='lambdarank', metric='ndcg', ndcg_eval_at=[1, 3],
    learning_rate=0.05, num_leaves=63, min_child_samples=20,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0,
    n_jobs=-1, verbose=-1, random_state=42,
)


def add_pace_features(df: pd.DataFrame) -> pd.DataFrame:
    """前走脚質_numを使ったレース単位の展開予想特徴量を追加。"""
    df = df.copy()

    # 推定脚質: 前走脚質_num を使い、NaN は近走平均で補完
    est = pd.to_numeric(df.get('前走脚質_num', pd.Series(np.nan, index=df.index)), errors='coerce')
    if '平均_脚質番号' in df.columns:
        est = est.fillna(pd.to_numeric(df['平均_脚質番号'], errors='coerce'))
    est = est.fillna(1.5)  # 未知なら中団扱い
    df['_est'] = est

    # 脚質カテゴリ別フラグ（前走ベース）
    df['_is_nige']   = (df['_est'] < 0.5).astype(float)   # 逃げ(0)
    df['_is_senkou'] = ((df['_est'] >= 0.5) & (df['_est'] < 1.5)).astype(float)  # 先行(1)

    # レース単位集計
    race_agg = df.groupby('race_id').agg(
        race_前走逃げ数=('_is_nige',   'sum'),
        race_前走先行数=('_is_senkou', 'sum'),
        race_前走脚質平均=('_est', 'mean'),
        race_前走脚質std=('_est',  'std'),
        _race_n=('_est', 'count'),
    ).fillna({'race_前走脚質std': 0}).reset_index()
    race_agg['race_前走前目頭数']  = race_agg['race_前走逃げ数'] + race_agg['race_前走先行数']
    race_agg['race_前走前目割合']  = race_agg['race_前走前目頭数'] / race_agg['_race_n'].clip(1)

    merge_cols = ['race_id'] + [c for c in race_agg.columns if c.startswith('race_') and c != 'race_id']
    df = df.merge(race_agg[merge_cols], on='race_id', how='left')

    # 自馬 × レース展開の相性スコア（ベクトル化）
    s   = df['_est']
    fp  = df['race_前走前目割合'].fillna(0.5)
    esc = df['race_前走逃げ数'].fillna(1)

    # 逃げ馬: 他の逃げ馬が少ないほど有利
    # 先行馬: 前目が少なめのほど有利
    # 差し・追い込み: 前目が多いほど有利（ハイペース）
    pace_adv = np.where(
        s < 0.5,  1.0 - fp,
        np.where(
            s < 1.5, 0.5 - fp * 0.5,
            np.where(s < 2.5, fp - 0.4, fp - 0.3)
        )
    )
    df['pace_有利スコア']      = pace_adv
    df['pace_単独逃げ']        = ((df['_is_nige'] == 1) & (esc <= 1)).astype(float)
    df['pace_前目競争度']      = df['_is_nige'] * esc + df['_is_senkou'] * df['race_前走先行数'].fillna(0)
    df['自馬_対_レース脚質差'] = s - df['race_前走脚質平均'].fillna(1.5)

    df.drop(columns=['_est', '_is_nige', '_is_senkou'], inplace=True)
    return df


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


def roi_table(d, label):
    print(f'\n=== {label} ===')
    for yr in sorted(d['yr'].unique()):
        sub = d[d['yr'] == yr]
        won = sub['着順_num'] == 1
        r   = (sub.loc[won, 'odds_num'] * 100).sum() / (len(sub) * 100) - 1
        print(f'  20{yr:02d}: {len(sub)}R  win={won.mean():.3f}  ROI={r:+.3f}')
    won = d['着順_num'] == 1
    r   = (d.loc[won, 'odds_num'] * 100).sum() / (len(d) * 100) - 1
    print(f'  Total: {len(d)}R  win={won.mean():.3f}  ROI={r:+.3f}')


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

    print('展開予想特徴量を追加中...')
    df = add_pace_features(df)
    new_cols = [c for c in PACE_COLS if c in df.columns]
    print(f'  追加列: {new_cols}')

    num_cols  = df.select_dtypes(include='number').columns.tolist()
    feat_cols = [c for c in num_cols if c not in EXCLUDE and c not in ODDS_REMOVE]
    print(f'特徴量: {len(feat_cols)}列（脚質リーク除外済み・展開特徴量込み）')
    pace_in_feat = [c for c in PACE_COLS if c in feat_cols]
    print(f'  うち展開特徴量: {pace_in_feat}')

    tr  = df[(df['日付_num'] >= 130101) & (df['日付_num'] <= 221231)]
    val = tr[tr['日付_num'] >= 210101]
    trn = tr[tr['日付_num'] <  210101]
    print(f'学習: {len(trn):,}行 / valid: {len(val):,}行')

    X_tr, y_tr, g_tr = build(trn, feat_cols)
    X_va, y_va, g_va = build(val, feat_cols)
    ds_tr = lgb.Dataset(X_tr, label=y_tr, group=g_tr, free_raw_data=False)
    ds_va = lgb.Dataset(X_va, label=y_va, group=g_va, free_raw_data=False, reference=ds_tr)

    print('学習中...')
    model = lgb.train(
        LGBM_PARAMS, ds_tr, num_boost_round=500, valid_sets=[ds_va],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(50)],
    )
    print(f'best_iter={model.best_iteration}')

    # ── OOS ROI評価 ──────────────────────────────────────────────────────
    print('\n=== OOS ROI評価 (2023+) ===')
    oos = df[df['日付_num'] >= 230101].copy()
    X_oos = np.full((len(oos), len(feat_cols)), np.nan, dtype=np.float32)
    for i, col in enumerate(feat_cols):
        if col in oos.columns:
            X_oos[:, i] = pd.to_numeric(oos[col], errors='coerce').values

    oos['_score']    = model.predict(X_oos)
    oos['rank_model'] = oos.groupby('race_id')['_score'].rank(ascending=False, method='first')
    oos['pop_num']   = pd.to_numeric(oos['人気'], errors='coerce')
    oos['odds_num']  = pd.to_numeric(oos['単勝オッズ'], errors='coerce')
    oos['yr']        = oos['日付_num'] // 10000

    top1 = oos[oos['rank_model'] == 1]
    roi_table(top1, 'rank=1 (全体)')
    roi_table(top1[top1['pop_num'] >= 2], 'rank=1 × 2番人気以下')
    roi_table(top1[top1['pop_num'] >= 4], 'rank=1 × 4番人気以上')

    # ── 保存 ──────────────────────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    info = {
        'feat_cols':      feat_cols,
        'pace_cols':      pace_in_feat,
        'train_range':    [130101, 221231],
        'best_iteration': model.best_iteration,
        'odds_removed':   True,
        '脚質_leakage':   False,
    }
    with open(INFO_PATH, 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f'\n保存完了: {MODEL_PATH}')


if __name__ == '__main__':
    main()
