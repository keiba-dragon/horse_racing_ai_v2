# coding: utf-8
"""
LightGBM 直接勝率予測モデル (binary classification)
- 目標: rank=1全買い OOS ROI >= -5%
- 損失: binary logloss
- 後処理: val で Isotonic calibration
- リークなし・API情報のみ
"""
import os, sys, json, pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import save_lambdarank_pace as _slp   # stdout wrap 1回のみ
from save_lambdarank_pace import add_pace_features, EXCLUDE, ODDS_REMOVE
from save_conditional_logit import add_new_features

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, 'data', 'processed', 'all_venues_features.parquet')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
OUT_PKL   = os.path.join(MODEL_DIR, 'lgb_winprob.pkl')
OUT_JSON  = os.path.join(MODEL_DIR, 'lgb_winprob_info.json')

# レース内の競争構造を表す追加特徴量（リークなし）
RACE_AGG_FEATS = [
    'race_頭数平均斤量', 'race_頭数std斤量',
]


def make_race_id(df):
    df['race_id'] = (df['日付_num'].astype(int).astype(str) + '_' +
                     df['開催'].astype(str).str.strip() + '_' +
                     df['Ｒ'].astype(str).str.strip())
    return df


def load_and_prep():
    print('データ読み込み中...')
    df = pd.read_parquet(DATA_FILE)
    df['日付_num'] = pd.to_numeric(df['日付'], errors='coerce')
    df['着順_num'] = pd.to_numeric(df['着順_num'], errors='coerce')
    df = df.dropna(subset=['日付_num', '着順_num'])
    df = df[df['着順_num'] < 99]
    df = make_race_id(df)
    df = add_pace_features(df)
    df = add_new_features(df)
    return df


def build_feat_cols(df):
    num_cols  = df.select_dtypes(include='number').columns.tolist()
    feat_cols = [c for c in num_cols if c not in EXCLUDE and c not in ODDS_REMOVE]
    return feat_cols


def roi_table(d, label=''):
    print(f'\n  {label}')
    for yr in sorted(d['yr'].unique()):
        s   = d[d['yr'] == yr]
        won = s['着順_num'] == 1
        r   = (s.loc[won, 'odds_num'] * 100).sum() / (len(s) * 100) - 1
        print(f'    20{int(yr):02d}: {len(s):5d}R  win={won.mean():.3f}  ROI={r:+.3f}')
    won = d['着順_num'] == 1
    r   = (d.loc[won, 'odds_num'] * 100).sum() / (len(d) * 100) - 1
    print(f'    Total: {len(d):5d}R  win={won.mean():.3f}  ROI={r:+.3f}')
    return r


def main():
    df = load_and_prep()
    feat_cols = build_feat_cols(df)
    print(f'特徴量: {len(feat_cols)}列')

    trn = df[(df['日付_num'] >= 130101) & (df['日付_num'] < 210101)]
    val = df[(df['日付_num'] >= 210101) & (df['日付_num'] <= 221231)]
    oos = df[df['日付_num'] >= 230101]
    print(f'学習: {len(trn):,}行 / val: {len(val):,}行 / OOS: {len(oos):,}行')

    X_tr = trn[feat_cols].astype(float).fillna(0).values
    y_tr = (trn['着順_num'] == 1).astype(int).values
    X_va = val[feat_cols].astype(float).fillna(0).values
    y_va = (val['着順_num'] == 1).astype(int).values

    dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=feat_cols)
    dval   = lgb.Dataset(X_va, label=y_va, reference=dtrain, feature_name=feat_cols)

    params = {
        'objective'       : 'binary',
        'metric'          : 'binary_logloss',
        'learning_rate'   : 0.05,
        'num_leaves'      : 63,
        'min_child_samples': 50,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.8,
        'bagging_freq'    : 5,
        'lambda_l1'       : 0.1,
        'lambda_l2'       : 1.0,
        'verbose'         : -1,
        'seed'            : 42,
    }

    print('\nLightGBM 学習中...')
    callbacks = [
        lgb.early_stopping(stopping_rounds=100, verbose=True),
        lgb.log_evaluation(period=100),
    ]
    model = lgb.train(
        params, dtrain,
        num_boost_round=2000,
        valid_sets=[dval],
        callbacks=callbacks,
    )

    # --- Isotonic calibration on val ---
    val_pred = model.predict(X_va)
    val_s    = val.sort_values('race_id').reset_index(drop=True)
    ir = IsotonicRegression(out_of_bounds='clip', increasing=True)
    ir.fit(val_pred, y_va)

    print('\n校正チェック (val):')
    calib_v = ir.predict(val_pred)
    bins = pd.qcut(val_pred, 10, labels=False, duplicates='drop')
    for b in sorted(set(bins)):
        mask = bins == b
        print(f'  raw={val_pred[mask].mean():.3f}  calib={calib_v[mask].mean():.3f}  '
              f'actual={y_va[mask].mean():.3f}  n={mask.sum()}')

    # --- OOS 評価 ---
    print('\nOOS 評価中...')
    X_oo = oos[feat_cols].astype(float).fillna(0).values
    raw_oo   = model.predict(X_oo)
    calib_oo = ir.predict(raw_oo)

    oos_s = oos.sort_values('race_id').reset_index(drop=True)
    oos_s = oos_s.copy()
    oos_s['model_prob']  = raw_oo
    oos_s['calib_prob']  = calib_oo
    oos_s['odds_num']    = pd.to_numeric(oos_s['単勝オッズ'], errors='coerce')
    oos_s['market_prob'] = 1.0 / oos_s['odds_num']
    oos_s['ev_score']    = oos_s['calib_prob'] - oos_s['market_prob'] * 0.80
    oos_s['yr']          = oos_s['日付_num'] // 10000

    oos_s['rank_model'] = oos_s.groupby('race_id')['calib_prob'].rank(
        ascending=False, method='first')

    print('\n' + '='*60)
    print('OOS ROI (LightGBM binary win-prob, calib)')
    print('='*60)

    top1 = oos_s[oos_s['rank_model'] == 1]
    total_roi = roi_table(top1, 'rank=1 全体')

    print('\n--- EV フィルタ ---')
    for thr in [0.00, 0.01, 0.02, 0.03, 0.05]:
        ev = oos_s[(oos_s['rank_model'] == 1) & (oos_s['ev_score'] > thr)]
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
        'model'     : model,
        'isotonic'  : ir,
        'feat_cols' : feat_cols,
        'total_oos_roi': total_roi,
    }
    with open(OUT_PKL, 'wb') as f:
        pickle.dump(artifact, f)

    info = {
        'feat_cols'    : feat_cols,
        'n_features'   : len(feat_cols),
        'best_iteration': model.best_iteration,
        'total_oos_roi': total_roi,
    }
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print(f'\n保存完了: {OUT_PKL}')
    mark = ' ← 目標達成!' if total_roi >= -0.05 else f'  (目標まであと{-0.05 - total_roi:+.3f})'
    print(f'rank=1全体 OOS ROI={total_roi:+.3f}{mark}')


if __name__ == '__main__':
    main()
