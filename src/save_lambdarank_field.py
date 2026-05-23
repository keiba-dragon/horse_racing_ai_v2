# coding: utf-8
"""
フィールド相対特徴量（競合馬との比較）を追加してlambdarankを学習。

lambdarankは各馬を独立にスコアリングするため、「このレースの他の馬より強いか」
という相対情報を自分では計算できない。事前に計算して与えることで改善を狙う。

追加特徴量 (全てカード取得可能な過去走データから計算):
  相対_{col}          … 自馬の値 - レース内平均（正=フィールド平均より優秀）
  優位率_{col}        … フィールド内での相対順位 (1=最強, 0=最弱)
  レース内_{col}_std  … フィールドの実力差の広がり（高=差がある、低=拮抗）
  レース内_タイム指数_max  … フィールド内最高タイム指数（レースのレベル感）

出力:
  models/lambdarank_field.pkl
  models/lambdarank_field_info.json
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
MODEL_PATH = os.path.join(MODEL_DIR, 'lambdarank_field.pkl')
INFO_PATH  = os.path.join(MODEL_DIR, 'lambdarank_field_info.json')

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
}

# フィールド相対化する対象列
# (列名, lower_is_better)  True=低いほど良い(着順系), False=高いほど良い(勝率・タイム指数系)
RELATIVE_COLS = [
    ('近3走_平均着順',              True),
    ('1走前_クラス調整着順',        True),
    ('芝ダ一致_平均着順_近5走',     True),
    ('近10走_平均着順',             True),
    ('近5走_平均相対着順',          True),
    ('前走着差タイム_クラス補正',   True),   # 負=勝利、正=負け → 低いほど良い
    ('近5走_着差タイム_クラス補正平均', True),
    ('近5走_タイム指数_max',        False),  # 高いほど速い → 高いほど良い
    ('近5走_タイム指数平均',        False),
    ('1走前_着順_num',              True),
    ('騎手コース_r100_複勝率',      False),
    ('騎手脚質_r100_複勝率',        False),
    ('調教師コース_r100_複勝率',    False),
    ('騎手_r200_複勝率',            False),
]

LGBM_PARAMS = dict(
    objective='lambdarank', metric='ndcg', ndcg_eval_at=[1, 3],
    learning_rate=0.05, num_leaves=63, min_child_samples=20,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0,
    n_jobs=-1, verbose=-1, random_state=42,
)


def add_pace_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    est = pd.to_numeric(df.get('前走脚質_num', pd.Series(np.nan, index=df.index)), errors='coerce')
    if '平均_脚質番号' in df.columns:
        est = est.fillna(pd.to_numeric(df['平均_脚質番号'], errors='coerce'))
    est = est.fillna(1.5)
    df['_est'] = est
    df['_is_nige']   = (df['_est'] < 0.5).astype(float)
    df['_is_senkou'] = ((df['_est'] >= 0.5) & (df['_est'] < 1.5)).astype(float)
    ra = df.groupby('race_id').agg(
        race_前走逃げ数=('_is_nige', 'sum'), race_前走先行数=('_is_senkou', 'sum'),
        race_前走脚質平均=('_est', 'mean'), race_前走脚質std=('_est', 'std'),
        _n=('_est', 'count'),
    ).fillna({'race_前走脚質std': 0}).reset_index()
    ra['race_前走前目頭数'] = ra['race_前走逃げ数'] + ra['race_前走先行数']
    ra['race_前走前目割合'] = ra['race_前走前目頭数'] / ra['_n'].clip(1)
    mc = ['race_id'] + [c for c in ra.columns if c.startswith('race_') and c != 'race_id']
    df = df.merge(ra[mc], on='race_id', how='left')
    s  = df['_est']; fp = df['race_前走前目割合'].fillna(0.5); esc = df['race_前走逃げ数'].fillna(1)
    df['pace_有利スコア']      = np.where(s < 0.5, 1.0 - fp, np.where(s < 1.5, 0.5 - fp * 0.5, np.where(s < 2.5, fp - 0.4, fp - 0.3)))
    df['pace_単独逃げ']        = ((df['_is_nige'] == 1) & (esc <= 1)).astype(float)
    df['pace_前目競争度']      = df['_is_nige'] * esc + df['_is_senkou'] * df['race_前走先行数'].fillna(0)
    df['自馬_対_レース脚質差'] = s - df['race_前走脚質平均'].fillna(1.5)
    df.drop(columns=['_est', '_is_nige', '_is_senkou'], inplace=True)
    return df


def add_field_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    """レース内フィールド相対特徴量を追加。全てカード取得可能な過去走データから計算。"""
    df = df.copy()
    added = []

    for col, lower_is_better in RELATIVE_COLS:
        if col not in df.columns:
            continue
        v = pd.to_numeric(df[col], errors='coerce')
        df['_v'] = v

        # レース内統計
        race_mean = df.groupby('race_id')['_v'].transform('mean')
        race_std  = df.groupby('race_id')['_v'].transform('std').fillna(0)
        race_n    = df.groupby('race_id')['_v'].transform('count')

        # 相対値: 正 = 自馬がフィールド平均より優秀
        sign = -1 if lower_is_better else 1
        rel_col = f'相対_{col}'
        df[rel_col] = (v - race_mean) * sign
        added.append(rel_col)

        # フィールド内優位率: 1=最強, 0=最弱 (NaNは中間扱い)
        ascending = lower_is_better
        rank = df.groupby('race_id')['_v'].rank(ascending=ascending, method='average', na_option='keep')
        adv_col = f'優位率_{col}'
        df[adv_col] = 1 - (rank - 1) / race_n.clip(1)
        added.append(adv_col)

        # フィールドの実力差の広がり (高=差あり, 低=拮抗)
        std_col = f'レース内std_{col}'
        df[std_col] = race_std
        added.append(std_col)

        df.drop(columns=['_v'], inplace=True)

    # タイム指数系のレースレベル特徴量
    if '近5走_タイム指数_max' in df.columns:
        v = pd.to_numeric(df['近5走_タイム指数_max'], errors='coerce')
        df['_v'] = v
        df['レース内_タイム指数_max']  = df.groupby('race_id')['_v'].transform('max')
        df['レース内_タイム指数_mean'] = df.groupby('race_id')['_v'].transform('mean')
        # 自馬がレース内最速タイム指数馬かどうか
        df['自馬_タイム指数_対_レース最速'] = v - df['レース内_タイム指数_max']
        added += ['レース内_タイム指数_max', 'レース内_タイム指数_mean', '自馬_タイム指数_対_レース最速']
        df.drop(columns=['_v'], inplace=True)

    # 騎手勝率のレース内優位
    if '騎手コース_r100_複勝率' in df.columns:
        v = pd.to_numeric(df['騎手コース_r100_複勝率'], errors='coerce')
        df['_v'] = v
        df['レース内_騎手複勝率_max']  = df.groupby('race_id')['_v'].transform('max')
        df['自馬_騎手複勝率_対_最強騎手'] = v - df['レース内_騎手複勝率_max']
        added += ['レース内_騎手複勝率_max', '自馬_騎手複勝率_対_最強騎手']
        df.drop(columns=['_v'], inplace=True)

    print(f'  フィールド相対特徴量: {len(added)}列追加')
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
        sub = d[d['yr'] == yr]; won = sub['着順_num'] == 1
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

    print('フィールド相対特徴量を追加中...')
    df = add_field_relative_features(df)

    num_cols  = df.select_dtypes(include='number').columns.tolist()
    feat_cols = [c for c in num_cols if c not in EXCLUDE and c not in ODDS_REMOVE]
    print(f'特徴量: {len(feat_cols)}列')

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

    # ── OOS ROI評価 ───────────────────────────────────────────────────────
    print('\nOOS予測中...')
    oos = df[df['日付_num'] >= 230101].copy()
    X_oos = np.full((len(oos), len(feat_cols)), np.nan, dtype=np.float32)
    for i, col in enumerate(feat_cols):
        if col in oos.columns:
            X_oos[:, i] = pd.to_numeric(oos[col], errors='coerce').values

    oos['_score']     = model.predict(X_oos)
    oos['rank_model'] = oos.groupby('race_id')['_score'].rank(ascending=False, method='first')
    oos['pop_num']    = pd.to_numeric(oos['人気'], errors='coerce')
    oos['odds_num']   = pd.to_numeric(oos['単勝オッズ'], errors='coerce')
    oos['surf']       = oos['芝・ダ'].astype(str).str.strip() if '芝・ダ' in oos.columns else ''
    oos['yr']         = oos['日付_num'] // 10000

    top1 = oos[oos['rank_model'] == 1]
    roi_table(top1, 'rank=1 全買い')
    roi_table(top1[top1['pop_num'] >= 2], 'rank=1 × 2番人気以下')

    # 展開エッジ条件
    prev_style = pd.to_numeric(oos.get('前走脚質_num', pd.Series(np.nan, index=oos.index)), errors='coerce')
    oos['_pace_edge'] = ((oos['surf'] == 'ダ') & prev_style.isin([0, 3])).astype(int)
    top1 = oos[oos['rank_model'] == 1]
    roi_table(top1[(top1['pop_num'] >= 2) & (top1['_pace_edge'] == 1)],
              'rank=1 × 2番人気以下 × ダ×前走逃げ/後方')

    # 上位特徴量
    imp = model.feature_importance(importance_type='gain')
    fi  = sorted(zip(feat_cols, imp), key=lambda x: -x[1])
    print('\n=== 上位15特徴量 ===')
    for name, gain in fi[:15]:
        print(f'  {gain:8.0f}  {name}')
    print('\n=== フィールド相対特徴量の重要度 (上位10) ===')
    field_feats = [(n, g) for n, g in fi if n.startswith(('相対_', '優位率_', 'レース内std_',
                                                           'レース内_タイム指数', '自馬_タイム',
                                                           'レース内_騎手', '自馬_騎手'))]
    for name, gain in field_feats[:10]:
        rank = next(i for i, (n, g) in enumerate(fi) if n == name) + 1
        print(f'  rank={rank:3d}  {gain:8.0f}  {name}')

    # ── 保存 ────────────────────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(INFO_PATH, 'w', encoding='utf-8') as f:
        json.dump({
            'feat_cols': feat_cols,
            'train_range': [130101, 221231],
            'best_iteration': model.best_iteration,
            'odds_removed': True,
            '脚質_leakage': False,
            'field_relative': True,
        }, f, ensure_ascii=False, indent=2)
    print(f'\n保存完了: {MODEL_PATH}')


if __name__ == '__main__':
    main()
