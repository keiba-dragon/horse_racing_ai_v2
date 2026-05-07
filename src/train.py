# -*- coding: utf-8 -*-
"""
モデル学習 (horse_racing_ai_v2)

入力 : data/processed/features.parquet
出力 : models/model.pkl + models/model_info.json

設計:
  - 芝ダ × 距離帯 のサブグループごとに LightGBM 分類器を学習
  - target: 1着かどうか (単勝を当てる確率 = win probability)
  - 学習期間: TRAIN_START 〜 TRAIN_END
  - OOSテスト: OOS_START 以降 (in-sample bias なし)
  - EV計算用の probability calibration は predict.py 側で行う

実行:
  python src/train.py
  python src/train.py --eval-only   # 学習スキップ・OOS評価のみ
"""
import sys, io, os, json, pickle, argparse
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT_FILE  = os.path.join(BASE_DIR, 'data', 'processed', 'features.parquet')
MODEL_DIR  = os.path.join(BASE_DIR, 'models')

TRAIN_START = 130101
TRAIN_END   = 201231
OOS_START   = 210101

# サブグループ: 芝ダ × 距離帯
SURFACES      = ['芝', 'ダ']
DIST_BANDS    = ['短距離', 'マイル', '中距離', '長距離']
# ダは '中距離' + '長距離' を統合
DA_DIST_BANDS = ['短距離', 'マイル', '中長距離']

LGBM_PARAMS = dict(
    n_estimators    = 500,
    learning_rate   = 0.05,
    num_leaves      = 31,
    max_depth       = -1,
    min_child_samples = 30,
    subsample       = 0.8,
    colsample_bytree= 0.8,
    reg_alpha       = 0.1,
    reg_lambda      = 1.0,
    class_weight    = 'balanced',
    random_state    = 42,
    n_jobs          = -1,
    verbose         = -1,
)

FEATURE_COLS = [
    '距離', '頭数', '馬番',
    '斤量', '馬体重', '馬体重変化',
    '出走回数',
    '直近1走_着順', '直近2走_着順', '直近3走_着順', '直近5走_着順',
    '直近3走_着順_平均', '直近5走_着順_平均', '直近3走_着順_slope',
    '連続入着数', '連続連対数', '連続1着数',
    '通算勝率', '通算入着率',
    '前走間隔_日',
    '芝ダ変更', '距離帯変更',
    '同コース_出走数', '同コース_着順_平均', '同コース_勝率',
    '直近1走_オッズ', '直近3走_オッズ_平均',
    '直近3走_走破タイム_平均', '直近1走_走破タイム',
    '直近3走_上り3F_平均', '直近1走_上り3F', '直近3走_上り3F_slope',
    '直近3走_4角_平均', '直近1走_4角',
]


# ─────────────────────────────────────────────────────────
# サブグループキー
# ─────────────────────────────────────────────────────────
def get_group_key(surface, dist_band):
    if surface == 'ダ' and dist_band in ('中距離', '長距離'):
        dist_band = '中長距離'
    return f"{surface}_{dist_band}"


def assign_group(df):
    df = df.copy()
    def _key(row):
        s = str(row.get('芝ダ', '')).strip()
        d = str(row.get('距離帯', '')).strip()
        if not s or not d or s not in ('芝', 'ダ'):
            return None
        return get_group_key(s, d)
    df['group_key'] = df.apply(_key, axis=1)
    return df


# ─────────────────────────────────────────────────────────
# 学習
# ─────────────────────────────────────────────────────────
def train_submodels(df):
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 使える特徴量のみ
    available = [c for c in FEATURE_COLS if c in df.columns]
    print(f"特徴量: {len(available)}個 / 定義: {len(FEATURE_COLS)}個")

    df = assign_group(df)
    df = df.dropna(subset=['着順', 'group_key'])
    df['target'] = (df['着順'] == 1).astype(int)
    df['日付_num'] = pd.to_numeric(df['日付_num'], errors='coerce')

    model_info = {'features': available, 'models': {}}
    oos_records = []

    for key in df['group_key'].unique():
        g = df[df['group_key'] == key].sort_values('日付_num')
        train = g[(g['日付_num'] >= TRAIN_START) & (g['日付_num'] <= TRAIN_END)]
        test  = g[g['日付_num'] >= OOS_START]

        if len(train) < 500 or train['target'].sum() < 50:
            print(f"  skip {key}: train={len(train)} wins={train['target'].sum()}")
            continue

        X_train = train[available].astype(float)
        y_train = train['target']

        clf = LGBMClassifier(**LGBM_PARAMS)
        clf.fit(X_train, y_train,
                eval_set=[(X_train, y_train)],
                callbacks=[])

        model_path = os.path.join(MODEL_DIR, f'model_{key}.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(clf, f)

        # OOS スコア
        if len(test) >= 100:
            X_test = test[available].astype(float)
            proba  = clf.predict_proba(X_test)[:, 1]
            test   = test.copy()
            test['prob_win'] = proba

            # レース内1位的中率
            test['pred_rank'] = test.groupby('race_id')['prob_win'].rank(
                ascending=False, method='min')
            rank1 = test[test['pred_rank'] == 1]
            hit_rate = rank1['target'].mean() if len(rank1) > 0 else np.nan

            # 期待値中央値
            test['ev'] = test['prob_win'] * test['単勝オッズ'].fillna(0)
            ev_median = test['ev'].median()

            oos_records.append({
                'group': key,
                'train_n': len(train),
                'test_n':  len(test),
                'hit_rate': round(hit_rate, 4) if not np.isnan(hit_rate) else None,
                'ev_median': round(ev_median, 3),
            })
            print(f"  {key:<20} train={len(train):>6}  OOS={len(test):>6}  1位的中={hit_rate:.1%}")
        else:
            print(f"  {key:<20} train={len(train):>6}  OOS={len(test):>4} (少)")

        model_info['models'][key] = {
            'path':   f'model_{key}.pkl',
            'group':  key,
            'train_n': int(len(train)),
        }

    # メタ情報保存
    model_info['train_range'] = [TRAIN_START, TRAIN_END]
    model_info['oos_start']   = OOS_START
    model_info['oos_summary'] = oos_records

    with open(os.path.join(MODEL_DIR, 'model_info.json'), 'w', encoding='utf-8') as f:
        json.dump(model_info, f, ensure_ascii=False, indent=2)

    print(f"\nモデル {len(model_info['models'])}グループ保存 → {MODEL_DIR}/")
    return model_info, oos_records


# ─────────────────────────────────────────────────────────
# OOS評価レポート
# ─────────────────────────────────────────────────────────
def eval_oos(df, model_info):
    available = model_info['features']
    df = assign_group(df)
    df = df.dropna(subset=['着順', 'group_key'])
    df['target']   = (df['着順'] == 1).astype(int)
    df['日付_num'] = pd.to_numeric(df['日付_num'], errors='coerce')
    test_all = df[df['日付_num'] >= OOS_START].copy()

    all_rows = []
    for key, minfo in model_info['models'].items():
        mpath = os.path.join(MODEL_DIR, minfo['path'])
        if not os.path.exists(mpath):
            continue
        with open(mpath, 'rb') as f:
            clf = pickle.load(f)
        sub = test_all[test_all['group_key'] == key].copy()
        if len(sub) < 50:
            continue
        X = sub[available].astype(float)
        sub['prob_win'] = clf.predict_proba(X)[:, 1]
        all_rows.append(sub)

    if not all_rows:
        print("OOSデータなし")
        return

    test = pd.concat(all_rows, ignore_index=True)
    test['pred_rank'] = test.groupby('race_id')['prob_win'].rank(
        ascending=False, method='min')
    test['ev'] = test['prob_win'] * test['単勝オッズ'].fillna(0)

    has_odds = test['単勝オッズ'].notna().mean() > 0.3

    print(f"\n{'='*65}")
    print(f" OOS評価  ({OOS_START}以降  {len(test):,}行 / {test['race_id'].nunique():,}レース)")
    print(f"{'='*65}")

    # [1] モデル1位
    rank1 = test[test['pred_rank'] == 1]
    print(f"\n[1] モデル予測1位に単勝買い")
    print(f"  的中率: {rank1['target'].mean():.1%}  ({rank1['target'].sum()}/{len(rank1)})")
    if has_odds:
        pays = rank1[rank1['target'] == 1]['単勝オッズ'] * 100
        roi  = pays.sum() / (len(rank1) * 100) - 1
        print(f"  ROI:    {roi:+.1%}")

    # [2] EV閾値別
    if has_odds:
        print(f"\n[2] EV ≥ 閾値 の馬だけ買い")
        print(f"  {'EV閾値':>8}  {'対象':>8}  {'的中率':>8}  {'ROI':>8}")
        for thr in [0.8, 1.0, 1.1, 1.2, 1.3, 1.5]:
            sub = test[test['ev'] >= thr].dropna(subset=['単勝オッズ'])
            if len(sub) < 30: continue
            hits = sub['target'].sum()
            hit_rate = sub['target'].mean()
            pays = sub[sub['target'] == 1]['単勝オッズ'] * 100
            roi  = pays.sum() / (len(sub) * 100) - 1
            print(f"  EV≥{thr:.1f}   {len(sub):>8,}  {hit_rate:>8.1%}  {roi:>+8.1%}")

    # [3] EV + モデル1位
    if has_odds:
        print(f"\n[3] モデル1位 & EV ≥ 閾値")
        print(f"  {'EV閾値':>8}  {'対象':>8}  {'的中率':>8}  {'ROI':>8}")
        r1_ev = test[test['pred_rank'] == 1].dropna(subset=['単勝オッズ'])
        for thr in [0.8, 1.0, 1.1, 1.2, 1.3]:
            sub = r1_ev[r1_ev['ev'] >= thr]
            if len(sub) < 20: continue
            pays = sub[sub['target'] == 1]['単勝オッズ'] * 100
            roi  = pays.sum() / (len(sub) * 100) - 1
            print(f"  EV≥{thr:.1f}   {len(sub):>8,}  {sub['target'].mean():>8.1%}  {roi:>+8.1%}")

    avg_horses = test.groupby('race_id').size().mean()
    print(f"\n  平均頭数: {avg_horses:.1f}頭  ランダム的中率: {1/avg_horses:.1%}")


# ─────────────────────────────────────────────────────────
# エントリポイント
# ─────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval-only', action='store_true', help='OOS評価のみ (学習スキップ)')
    args = ap.parse_args()

    print(f"特徴量読み込み: {FEAT_FILE}")
    if FEAT_FILE.endswith('.parquet'):
        df = pd.read_parquet(FEAT_FILE)
    else:
        df = pd.read_csv(FEAT_FILE, encoding='utf-8', dtype=str)
        for c in FEATURE_COLS + ['着順', '日付_num', '単勝オッズ']:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')

    print(f"読み込み: {len(df):,}行 × {len(df.columns)}列")

    if args.eval_only:
        info_path = os.path.join(MODEL_DIR, 'model_info.json')
        if not os.path.exists(info_path):
            print("model_info.json が見つかりません。先に学習を実行してください。")
            sys.exit(1)
        with open(info_path, 'r', encoding='utf-8') as f:
            model_info = json.load(f)
    else:
        print(f"\n学習期間: {TRAIN_START} 〜 {TRAIN_END}  |  OOS: {OOS_START}+")
        model_info, _ = train_submodels(df)

    eval_oos(df, model_info)


if __name__ == '__main__':
    main()
