# coding: utf-8
"""
LightGBM + 条件付きロジット カスタム目的関数
- ツリーの非線形性 × レース内競争構造（softmax-within-race）
- 目標: rank=1全買い OOS ROI >= -5%
- リークなし・API情報のみ
"""
import os, sys, json, pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import save_lambdarank_pace as _slp
from save_lambdarank_pace import add_pace_features, EXCLUDE, ODDS_REMOVE
from save_conditional_logit import add_new_features

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, 'data', 'processed', 'all_venues_features.parquet')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
OUT_PKL   = os.path.join(MODEL_DIR, 'lgb_clogit.pkl')
OUT_JSON  = os.path.join(MODEL_DIR, 'lgb_clogit_info.json')


def make_race_id(df):
    df['race_id'] = (df['日付_num'].astype(int).astype(str) + '_' +
                     df['開催'].astype(str).str.strip() + '_' +
                     df['Ｒ'].astype(str).str.strip())
    return df


def compute_groups(race_ids):
    """race_id配列 → グループサイズリスト（LightGBM Dataset.set_group用）"""
    _, counts = np.unique(race_ids, return_counts=True)
    # race_id でソート済みが前提 → 連続する同一 race_id の連長
    sizes = []
    prev = None
    cnt = 0
    for r in race_ids:
        if r == prev:
            cnt += 1
        else:
            if prev is not None:
                sizes.append(cnt)
            cnt = 1
            prev = r
    if cnt > 0:
        sizes.append(cnt)
    return sizes


# ---- カスタム目的関数: 条件付きロジット (softmax-within-race) ----

_GROUPS_TR = None   # グローバルに保持（train_data では get_group() が使えないため）

def make_clogit_obj(groups):
    """グループリストをクロージャに閉じ込めたカスタム目的関数を返す"""
    groups_arr = np.asarray(groups, dtype=np.int64)

    def clogit_obj(preds, dataset):
        labels = dataset.get_label()
        n = len(preds)
        grad = np.zeros(n, dtype=np.float64)
        hess = np.zeros(n, dtype=np.float64)

        idx = 0
        for g in groups_arr:
            s = preds[idx:idx + g].copy()
            s -= s.max()
            exp_s = np.exp(s)
            p = exp_s / (exp_s.sum() + 1e-300)
            y = labels[idx:idx + g]
            grad[idx:idx + g] = p - y
            hess[idx:idx + g] = np.maximum(p * (1.0 - p), 1e-6)
            idx += g
        return grad, hess

    return clogit_obj


def make_clogit_metric(groups):
    """グループリストをクロージャに閉じ込めた評価指標"""
    groups_arr = np.asarray(groups, dtype=np.int64)
    n_races    = len(groups_arr)

    def clogit_metric(preds, dataset):
        labels = dataset.get_label()
        log_lik = 0.0
        idx = 0
        for g in groups_arr:
            s = preds[idx:idx + g].copy()
            s -= s.max()
            exp_s = np.exp(s)
            p = exp_s / (exp_s.sum() + 1e-300)
            y = labels[idx:idx + g]
            log_lik += np.sum(y * np.log(np.clip(p, 1e-15, 1.0)))
            idx += g
        nll = -log_lik / n_races
        return 'clogit_nll', nll, False   # lower is better

    return clogit_metric


def segment_softmax_np(scores, groups):
    """numpy実装のsegment softmax → 確率ベクトル"""
    probs = np.zeros_like(scores)
    idx = 0
    for g in groups:
        s = scores[idx:idx + g]
        s = s - s.max()
        exp_s = np.exp(s)
        probs[idx:idx + g] = exp_s / (exp_s.sum() + 1e-300)
        idx += g
    return probs


def roi_table(d, label=''):
    print(f'\n  {label}')
    total_won_payout = 0.0
    total_n = 0
    for yr in sorted(d['yr'].unique()):
        s   = d[d['yr'] == yr]
        won = s['着順_num'] == 1
        r   = (s.loc[won, 'odds_num'] * 100).sum() / (len(s) * 100) - 1
        total_won_payout += (s.loc[won, 'odds_num'] * 100).sum()
        total_n += len(s)
        print(f'    20{int(yr):02d}: {len(s):5d}R  win={won.mean():.3f}  ROI={r:+.3f}')
    won = d['着順_num'] == 1
    r   = (d.loc[won, 'odds_num'] * 100).sum() / (len(d) * 100) - 1
    print(f'    Total: {len(d):5d}R  win={won.mean():.3f}  ROI={r:+.3f}')
    return r


def main():
    print('データ読み込み中...')
    df = pd.read_parquet(DATA_FILE)
    df['日付_num'] = pd.to_numeric(df['日付'], errors='coerce')
    df['着順_num'] = pd.to_numeric(df['着順_num'], errors='coerce')
    df = df.dropna(subset=['日付_num', '着順_num'])
    df = df[df['着順_num'] < 99]
    df = make_race_id(df)
    df = add_pace_features(df)
    df = add_new_features(df)

    num_cols  = df.select_dtypes(include='number').columns.tolist()
    feat_cols = [c for c in num_cols if c not in EXCLUDE and c not in ODDS_REMOVE]
    print(f'特徴量: {len(feat_cols)}列')

    # race_id でソートして group が連続になるようにする
    df = df.sort_values('race_id').reset_index(drop=True)

    trn = df[(df['日付_num'] >= 130101) & (df['日付_num'] < 210101)].copy()
    val = df[(df['日付_num'] >= 210101) & (df['日付_num'] <= 221231)].copy()
    oos = df[df['日付_num'] >= 230101].copy()
    print(f'学習: {len(trn):,}行  val: {len(val):,}行  OOS: {len(oos):,}行')

    trn_groups = compute_groups(trn['race_id'].values)
    val_groups = compute_groups(val['race_id'].values)
    oos_groups = compute_groups(oos['race_id'].values)
    print(f'学習レース数: {len(trn_groups):,}  val: {len(val_groups):,}  OOS: {len(oos_groups):,}')

    X_tr = trn[feat_cols].astype(float).fillna(0).values
    y_tr = (trn['着順_num'] == 1).astype(float).values
    X_va = val[feat_cols].astype(float).fillna(0).values
    y_va = (val['着順_num'] == 1).astype(float).values

    dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=feat_cols, free_raw_data=False)
    dtrain.set_group(trn_groups)
    dval   = lgb.Dataset(X_va, label=y_va, reference=dtrain, feature_name=feat_cols, free_raw_data=False)
    dval.set_group(val_groups)

    clogit_obj    = make_clogit_obj(trn_groups)
    clogit_metric = make_clogit_metric(val_groups)

    params = {
        'objective'        : clogit_obj,   # LightGBM 4.x: function直接指定
        'learning_rate'    : 0.02,
        'num_leaves'       : 31,
        'max_depth'        : 6,
        'min_child_samples': 100,
        'feature_fraction' : 0.6,
        'bagging_fraction' : 0.7,
        'bagging_freq'     : 5,
        'lambda_l1'        : 0.5,
        'lambda_l2'        : 5.0,
        'min_gain_to_split': 0.01,
        'verbose'          : -1,
        'seed'             : 42,
    }

    print('\nLightGBM + conditional logit 学習中...')
    callbacks = [
        lgb.early_stopping(stopping_rounds=100, verbose=True),
        lgb.log_evaluation(period=100),
    ]
    model = lgb.train(
        params, dtrain,
        num_boost_round=2000,
        valid_sets=[dval],
        feval=clogit_metric,
        callbacks=callbacks,
    )

    # --- Isotonic calibration on val (softmax 確率で校正) ---
    val_raw  = model.predict(X_va)
    val_prob = segment_softmax_np(val_raw, val_groups)
    ir = IsotonicRegression(out_of_bounds='clip', increasing=True)
    ir.fit(val_prob, y_va)

    print('\n校正チェック (val):')
    calib_v = ir.predict(val_prob)
    bins = pd.qcut(val_prob, 10, labels=False, duplicates='drop')
    for b in sorted(set(bins)):
        mask = bins == b
        print(f'  raw={val_prob[mask].mean():.3f}  calib={calib_v[mask].mean():.3f}  '
              f'actual={y_va[mask].mean():.3f}  n={mask.sum()}')

    # --- OOS 評価 ---
    print('\nOOS 評価中...')
    X_oo   = oos[feat_cols].astype(float).fillna(0).values
    oos_raw  = model.predict(X_oo)
    oos_prob = segment_softmax_np(oos_raw, oos_groups)
    oos_calib = ir.predict(oos_prob)

    oos = oos.copy()
    oos['model_prob']  = oos_prob
    oos['calib_prob']  = oos_calib
    oos['odds_num']    = pd.to_numeric(oos['単勝オッズ'], errors='coerce')
    oos['market_prob'] = 1.0 / oos['odds_num']
    oos['ev_score']    = oos['calib_prob'] - oos['market_prob'] * 0.80
    oos['yr']          = oos['日付_num'] // 10000

    oos['rank_model'] = oos.groupby('race_id')['calib_prob'].rank(
        ascending=False, method='first')

    print('\n' + '='*60)
    print('OOS ROI (LightGBM + conditional logit)')
    print('='*60)

    top1 = oos[oos['rank_model'] == 1]
    total_roi = roi_table(top1, 'rank=1 全体')

    print('\n--- EV フィルタ ---')
    for thr in [0.00, 0.01, 0.02, 0.03, 0.05]:
        ev = oos[(oos['rank_model'] == 1) & (oos['ev_score'] > thr)]
        if len(ev) >= 200:
            won = ev['着順_num'] == 1
            r   = (ev.loc[won, 'odds_num'] * 100).sum() / (len(ev) * 100) - 1
            print(f'  EV>{thr:.2f}: {len(ev):5d}件  win={won.mean():.3f}  ROI={r:+.3f}')

    print('\n--- 特徴量重要度 Top20 (gain) ---')
    imps  = model.feature_importance(importance_type='gain')
    order = np.argsort(imps)[::-1][:20]
    for i in order:
        print(f'  {feat_cols[i]:40s} {imps[i]:10.1f}')

    # 保存
    artifact = {
        'model'       : model,
        'isotonic'    : ir,
        'feat_cols'   : feat_cols,
        'oos_groups_size': oos_groups,
        'total_oos_roi': total_roi,
    }
    with open(OUT_PKL, 'wb') as f:
        pickle.dump(artifact, f)

    info = {
        'feat_cols'      : feat_cols,
        'n_features'     : len(feat_cols),
        'best_iteration' : model.best_iteration,
        'total_oos_roi'  : total_roi,
    }
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print(f'\n保存完了: {OUT_PKL}')
    mark = ' ← 目標達成!' if total_roi >= -0.05 else f'  (目標まであと{abs(-0.05 - total_roi):.3f})'
    print(f'rank=1全体 OOS ROI={total_roi:+.3f}{mark}')


if __name__ == '__main__':
    main()
