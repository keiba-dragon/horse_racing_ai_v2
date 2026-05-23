# coding: utf-8
"""
LightGBM lambdarank: レース内順位を直接最適化

binary分類の問題点:
  - 1頭ずつ独立にP(勝つ)を予測 → 強い馬を過大評価するバイアス
  - 市場相関(0.3787) > モデル相関(0.3392) という結果につながっていた

lambdarankの改善点:
  - レース単位でgroup指定 → 「このレースで誰が強いか」を直接最適化
  - 確率の合計が1になる制約が暗黙的に入る

使い方:
  python src/03_train_lambdarank.py           # 学習 + OOS評価
  python src/03_train_lambdarank.py --eval-only
"""
import sys, io, os, json, pickle, argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, 'data', 'processed', 'all_venues_features.parquet')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'lambdarank.pkl')
INFO_PATH  = os.path.join(MODEL_DIR, 'lambdarank_info.json')

TRAIN_START = 130101   # 2013-01-01
TRAIN_END   = 221231   # 2022-12-31
OOS_START   = 230101   # 2023-01-01

# ── 当レース結果・識別子（特徴量から除外）
EXCLUDE = {
    # リーク: 当レースの結果
    '走破タイム', '走破タイム_sec', '着差',
    '2角', '3角', '4角',
    '上り3F', 'PCI', 'PCI3', 'RPCI_x', 'RPCI_y',
    '上3F地点差_x', '上3F地点差_y',
    'タイム指数', '上り3F_指数',
    'Ave-3F', '平均速度', '-3F平均速度', '上り3F平均速度',
    '単勝配当', '複勝配当', '枠連', '馬連', '馬単', '３連複', '３連単',
    '好走',
    '賞金',  # 当レース獲得賞金（着順に完全連動）→ リーク
    # 識別子・文字列メタ
    'Ｍ', '日付', '開催', 'Ｒ', 'レース名', '限定', '馬名S', 'Ｃ',
    '性別', '騎手', '調教師', '種牡馬', '母父馬', '生産者', '毛色',
    '馬記号', '生年月日', '市場取引価格(万/最終)', '取引市場(最終)', '産地',
    '前走開催', '前走レース名', '替', '前騎手',
    '1走前_開催', '2走前_開催', '3走前_開催', '4走前_開催',
    '5走前_開催', '6走前_開催', '7走前_開催', '8走前_開催',
    '9走前_開催', '10走前_開催',
    '前走日付', '前走Ｒ',
    # ターゲット・派生
    '着順', '着順_num', '前走着順_num',
    '日付_num', 'race_id',
}

LGBM_PARAMS = dict(
    objective        = 'lambdarank',
    metric           = 'ndcg',
    ndcg_eval_at     = [1, 3],
    learning_rate    = 0.05,
    num_leaves       = 63,
    max_depth        = -1,
    min_child_samples= 20,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    reg_alpha        = 0.1,
    reg_lambda       = 1.0,
    n_jobs           = -1,
    verbose          = -1,
    random_state     = 42,
)
N_ROUNDS     = 500
EARLY_STOP   = 30


def make_label(着順_num):
    """序数ラベル: 1着=3, 2着=2, 3着=1, 他=0"""
    s = pd.to_numeric(着順_num, errors='coerce').fillna(99).astype(int)
    label = np.zeros(len(s), dtype=np.int32)
    label[s == 1] = 3
    label[s == 2] = 2
    label[s == 3] = 1
    return label


def load_data():
    print(f"データ読み込み: {DATA_FILE}")
    df = pd.read_parquet(DATA_FILE)
    print(f"  {len(df):,}行 × {len(df.columns)}列")

    df['日付_num'] = pd.to_numeric(df['日付'], errors='coerce')
    df['着順_num'] = pd.to_numeric(df['着順_num'], errors='coerce')
    df = df.dropna(subset=['日付_num', '着順_num'])
    df = df[df['着順_num'] < 99]

    df['race_id'] = (df['日付_num'].astype(int).astype(str) + '_' +
                     df['開催'].astype(str).str.strip() + '_' +
                     df['Ｒ'].astype(str).str.strip())
    df['surface'] = df['芝・ダ'].astype(str).str.strip()

    # 特徴量列を決定（数値列 - 除外リスト）
    num_cols = df.select_dtypes(include='number').columns.tolist()
    feat_cols = [c for c in num_cols if c not in EXCLUDE]
    print(f"  特徴量: {len(feat_cols)}列")

    return df, feat_cols


def build_dataset(df, feat_cols):
    """race_idでソートしてlambdarank用データセットを作成"""
    df = df.sort_values('race_id').reset_index(drop=True)
    X = df[feat_cols].astype(float).values
    y = make_label(df['着順_num'])
    groups = df.groupby('race_id', sort=False).size().values
    return X, y, groups, df


def train(df, feat_cols):
    os.makedirs(MODEL_DIR, exist_ok=True)

    tr = df[(df['日付_num'] >= TRAIN_START) & (df['日付_num'] <= TRAIN_END)]
    # validationに直近2年（2021-22）を使う
    val_mask = (tr['日付_num'] >= 210101)
    val = tr[val_mask]
    trn = tr[~val_mask]

    print(f"\n学習: {len(trn):,}行 / {trn['race_id'].nunique():,}レース")
    print(f"valid: {len(val):,}行 / {val['race_id'].nunique():,}レース")

    X_tr, y_tr, g_tr, _ = build_dataset(trn, feat_cols)
    X_va, y_va, g_va, _ = build_dataset(val,  feat_cols)

    ds_tr = lgb.Dataset(X_tr, label=y_tr, group=g_tr, free_raw_data=False)
    ds_va = lgb.Dataset(X_va, label=y_va, group=g_va, free_raw_data=False, reference=ds_tr)

    print(f"\nLightGBM lambdarank 学習中...")
    model = lgb.train(
        LGBM_PARAMS,
        ds_tr,
        num_boost_round   = N_ROUNDS,
        valid_sets        = [ds_va],
        callbacks         = [
            lgb.early_stopping(EARLY_STOP, verbose=False),
            lgb.log_evaluation(50),
        ],
    )

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)

    info = {
        'feat_cols':   feat_cols,
        'train_range': [TRAIN_START, TRAIN_END],
        'oos_start':   OOS_START,
        'best_iteration': model.best_iteration,
        'params':      LGBM_PARAMS,
    }
    with open(INFO_PATH, 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print(f"\nモデル保存: {MODEL_PATH}  (best_iter={model.best_iteration})")
    return model, feat_cols


def evaluate(df, feat_cols, model):
    oos = df[df['日付_num'] >= OOS_START].copy()
    print(f"\nOOS: {len(oos):,}行 / {oos['race_id'].nunique():,}レース ({OOS_START}+)")

    X_oos, _, _, oos = build_dataset(oos, feat_cols)
    oos['score'] = model.predict(X_oos)
    oos['rank_model'] = oos.groupby('race_id')['score'].rank(ascending=False, method='first')

    # 市場との相関比較
    oos['market_P'] = 0.75 / pd.to_numeric(oos['単勝オッズ'], errors='coerce')
    valid = oos.dropna(subset=['market_P'])
    corr_m  = valid['score'].corr(valid['着順_num'].apply(lambda x: -x))  # 着順は小さいほど良い
    corr_mk = valid['market_P'].corr(valid['着順_num'].apply(lambda x: -x))

    SEP = '=' * 60
    print(f"\n{SEP}")
    print(" OOS評価結果")
    print(SEP)

    # 1位的中率（全体 + 年別）
    top1 = oos[oos['rank_model'] == 1]
    hit_all = top1['着順_num'].eq(1).mean()
    print(f"\n[1位的中率]")
    print(f"  全体: {hit_all:.1%}  (N={len(top1):,})")
    for yr in sorted(oos['日付_num'].astype(str).str[:2].unique()):
        if yr < '23': continue
        mask = oos['日付_num'].astype(str).str[:2] == yr
        sub = oos[mask & (oos['rank_model'] == 1)]
        if len(sub) < 50: continue
        print(f"  20{yr}: {sub['着順_num'].eq(1).mean():.1%}  (N={len(sub):,})")

    # 市場との相関比較
    print(f"\n[market_P vs 着順 相関]")
    print(f"  lambdarank score: {corr_m:.4f}")
    print(f"  market_P (基準):  {corr_mk:.4f}")
    diff = corr_m - corr_mk
    print(f"  差分: {diff:+.4f}  {'✓ 市場超え' if diff > 0 else '✗ 市場未満'}")

    # 旧モデル(binary)との比較参考
    print(f"\n[参考: binary分類 OOS結果]")
    print(f"  相関: 0.3392  1位的中率: 27.6%  単ROI: -20.0%  (旧プロジェクト計測)")

    # 芝ダ別
    print(f"\n[芝ダ別 1位的中率]")
    for surf in ['芝', 'ダ']:
        sub = top1[top1['surface'] == surf]
        if len(sub) < 50: continue
        hr = sub['着順_num'].eq(1).mean()
        print(f"  {surf}: {hr:.1%}  (N={len(sub):,})")

    return oos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval-only', action='store_true')
    args = ap.parse_args()

    df, feat_cols = load_data()

    if args.eval_only:
        with open(INFO_PATH, 'r', encoding='utf-8') as f:
            info = json.load(f)
        feat_cols = info['feat_cols']
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        print("モデル読み込み完了")
    else:
        model, feat_cols = train(df, feat_cols)

    evaluate(df, feat_cols, model)


if __name__ == '__main__':
    main()
