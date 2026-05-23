# coding: utf-8
"""
芝ダ別 条件付きロジット v2
- 芝専用 / ダート専用 lambdarank で TOP_K 特徴量を選択（表面特化型）
- 複数の TOP_K を試してベストを採用
- 目標: rank=1全買い OOS ROI >= -10%
"""
import os, sys, json, pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import save_lambdarank_pace as _slp
from save_lambdarank_pace import add_pace_features, EXCLUDE, ODDS_REMOVE
from save_conditional_logit import (
    add_new_features, segment_softmax, get_group_starts, prepare
)

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, 'data', 'processed', 'all_venues_features.parquet')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

LR          = 0.001
N_EPOCHS    = 800
PATIENCE    = 100
CHECK_EVERY = 10
ALPHA       = 1.0
TOP_K_LIST  = [25, 35, 50]   # 表面ごとに試すTOP_K候補

LGBM_PARAMS = dict(
    objective='lambdarank', metric='ndcg', ndcg_eval_at=[1, 3],
    learning_rate=0.05, num_leaves=63, min_child_samples=20,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0,
    n_jobs=-1, verbose=-1, random_state=42,
)


def make_race_id(df):
    df['race_id'] = (df['日付_num'].astype(int).astype(str) + '_' +
                     df['開催'].astype(str).str.strip() + '_' +
                     df['Ｒ'].astype(str).str.strip())
    return df


def get_surface(df):
    return df['距離'].astype(str).str.strip().str.extract(r'^([芝ダ])')[0].fillna('不明')


def make_label(s):
    s = pd.to_numeric(s, errors='coerce').fillna(99).astype(int)
    l = np.zeros(len(s), dtype=np.int32)
    l[s == 1] = 3; l[s == 2] = 2; l[s == 3] = 1
    return l


def train_surface_lambdarank(df_surf, feat_cols, surface_name):
    """表面別 lambdarank を学習し gain 重要度を返す"""
    print(f'  [{surface_name}] lambdarank 学習中...')
    trn = df_surf[(df_surf['日付_num'] >= 130101) & (df_surf['日付_num'] < 210101)].sort_values('race_id')
    val = df_surf[(df_surf['日付_num'] >= 210101) & (df_surf['日付_num'] <= 221231)].sort_values('race_id')

    X_tr = trn[feat_cols].astype(float).fillna(0).values
    y_tr = make_label(trn['着順_num'])
    g_tr = trn.groupby('race_id', sort=False).size().values

    X_va = val[feat_cols].astype(float).fillna(0).values
    y_va = make_label(val['着順_num'])
    g_va = val.groupby('race_id', sort=False).size().values

    dtrain = lgb.Dataset(X_tr, label=y_tr, group=g_tr, feature_name=feat_cols)
    dval   = lgb.Dataset(X_va, label=y_va, group=g_va, reference=dtrain, feature_name=feat_cols)

    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=200),
    ]
    model = lgb.train(LGBM_PARAMS, dtrain, num_boost_round=500,
                      valid_sets=[dval], callbacks=callbacks)

    imps = model.feature_importance(importance_type='gain')
    print(f'  [{surface_name}] best_iter={model.best_iteration}  TOP5={[feat_cols[i] for i in np.argsort(imps)[::-1][:5]]}')
    return imps


def get_top_idx_from_imps(imps, feat_cols, top_k):
    ranked = np.argsort(imps)[::-1][:top_k]
    return ranked


def train_adam(X_tr, y_tr, gs_tr, nr_tr, X_va, y_va, gs_va, nr_va, alpha):
    d = X_tr.shape[1]
    beta = np.zeros(d)
    m, v = np.zeros(d), np.zeros(d)
    b1, b2, eps = 0.9, 0.999, 1e-8
    t = 0
    best_val, best_beta, no_improve = np.inf, beta.copy(), 0
    n_tr, n_va = len(y_tr), len(y_va)

    for epoch in range(1, N_EPOCHS + 1):
        scores    = X_tr @ beta
        probs     = segment_softmax(scores, gs_tr, n_tr)
        log_lik   = np.sum(y_tr * np.log(np.clip(probs, 1e-15, 1.0)))
        residuals = y_tr - probs
        loss = (-log_lik + alpha * np.sum(beta**2)) / nr_tr
        grad = (-(X_tr.T @ residuals) + 2 * alpha * beta) / nr_tr

        t += 1
        m = b1 * m + (1 - b1) * grad
        v = b2 * v + (1 - b2) * grad**2
        m_hat = m / (1 - b1**t)
        v_hat = v / (1 - b2**t)
        beta -= LR * m_hat / (np.sqrt(v_hat) + eps)

        if epoch % CHECK_EVERY == 0:
            sc_v = X_va @ beta
            pr_v = segment_softmax(sc_v, gs_va, n_va)
            ll_v = np.sum(y_va * np.log(np.clip(pr_v, 1e-15, 1.0)))
            vl   = (-ll_v + alpha * np.sum(beta**2)) / nr_va
            if vl < best_val:
                best_val = vl; best_beta = beta.copy(); no_improve = 0
            else:
                no_improve += 1
            if no_improve >= PATIENCE // CHECK_EVERY:
                break

    return best_beta, best_val


def train_one(trn_s, val_s, feat_cols, top_idx):
    X_tr, y_tr, gs_tr, n_tr, nr_tr, scaler, poly2, iscaler2, _, _ = prepare(
        trn_s, feat_cols,
        scaler=None, poly2=None, inter_scaler2=None, top_idx=top_idx,
        poly3=None, inter_scaler3=None, top_idx3=None, fit=True)
    X_va, y_va, gs_va, n_va, nr_va, *_ = prepare(
        val_s, feat_cols,
        scaler=scaler, poly2=poly2, inter_scaler2=iscaler2, top_idx=top_idx,
        poly3=None, inter_scaler3=None, top_idx3=None, fit=False)

    beta, best_val = train_adam(X_tr, y_tr, gs_tr, nr_tr,
                                X_va, y_va, gs_va, nr_va, ALPHA)
    pr_v = segment_softmax(X_va @ beta, gs_va, n_va)
    ir = IsotonicRegression(out_of_bounds='clip', increasing=True)
    ir.fit(pr_v, y_va)

    return {
        'coef': beta, 'scaler': scaler, 'poly2': poly2,
        'inter_scaler2': iscaler2, 'top_idx': top_idx,
        'poly3': None, 'inter_scaler3': None, 'top_idx3': None,
        'feat_cols': feat_cols, 'isotonic': ir, 'best_val': best_val,
    }


def predict_one(oos_s, art):
    X_oo, y_oo, gs_oo, n_oo, *_ = prepare(
        oos_s, art['feat_cols'],
        scaler=art['scaler'], poly2=art['poly2'], inter_scaler2=art['inter_scaler2'],
        top_idx=art['top_idx'],
        poly3=None, inter_scaler3=None, top_idx3=None, fit=False)
    raw   = segment_softmax(X_oo @ art['coef'], gs_oo, n_oo)
    calib = art['isotonic'].predict(raw)
    out   = oos_s.sort_values('race_id').reset_index(drop=True).copy()
    out['model_prob'] = raw
    out['calib_prob'] = calib
    out['y']          = y_oo
    return out


def eval_oos(all_pred, label=''):
    all_pred = all_pred.copy()
    all_pred['rank_model'] = all_pred.groupby('race_id')['calib_prob'].rank(
        ascending=False, method='first')
    top1 = all_pred[all_pred['rank_model'] == 1]
    won  = top1['着順_num'] == 1
    total_roi = (top1.loc[won, 'odds_num'] * 100).sum() / (len(top1) * 100) - 1

    print(f'\n  {label}')
    for yr in sorted(top1['yr'].unique()):
        s   = top1[top1['yr'] == yr]
        w   = s['着順_num'] == 1
        r   = (s.loc[w, 'odds_num'] * 100).sum() / (len(s) * 100) - 1
        print(f'    20{int(yr):02d}: {len(s):5d}R  win={w.mean():.3f}  ROI={r:+.3f}')
    print(f'    Total: {len(top1):5d}R  win={won.mean():.3f}  ROI={total_roi:+.3f}')
    return total_roi, all_pred


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
    df['surface'] = get_surface(df)
    df = df[df['surface'].isin(['芝', 'ダ'])].copy()

    num_cols  = df.select_dtypes(include='number').columns.tolist()
    feat_cols = [c for c in num_cols if c not in EXCLUDE and c not in ODDS_REMOVE]
    print(f'特徴量: {len(feat_cols)}列')

    # ── 表面別 lambdarank 学習 ────────────────────────────────────────────
    print('\n表面別 lambdarank 特徴量重要度を計算中...')
    surface_imps = {}
    for surf in ['芝', 'ダ']:
        df_surf = df[df['surface'] == surf].copy()
        surface_imps[surf] = train_surface_lambdarank(df_surf, feat_cols, surf)

    # ── TOP_K チューニング（val ROI で選択） ─────────────────────────────
    print('\nTOP_K チューニング中（表面別）...')
    best_combo = None
    best_val_roi = -np.inf

    for top_k_turf in TOP_K_LIST:
        for top_k_dirt in TOP_K_LIST:
            all_val_preds = []
            combo_arts = {}

            for surf, top_k in [('芝', top_k_turf), ('ダ', top_k_dirt)]:
                top_idx = get_top_idx_from_imps(surface_imps[surf], feat_cols, top_k)
                trn_s = df[(df['日付_num'] >= 130101) & (df['日付_num'] < 210101) & (df['surface'] == surf)].sort_values('race_id').reset_index(drop=True)
                val_s = df[(df['日付_num'] >= 210101) & (df['日付_num'] <= 221231) & (df['surface'] == surf)].sort_values('race_id').reset_index(drop=True)

                art = train_one(trn_s, val_s, feat_cols, top_idx)

                # val 予測でROI計算（チューニング用）
                X_va, y_va, gs_va, n_va, *_ = prepare(
                    val_s, feat_cols,
                    scaler=art['scaler'], poly2=art['poly2'], inter_scaler2=art['inter_scaler2'],
                    top_idx=top_idx, poly3=None, inter_scaler3=None, top_idx3=None, fit=False)
                raw_v   = segment_softmax(X_va @ art['coef'], gs_va, n_va)
                calib_v = art['isotonic'].predict(raw_v)
                val_df  = val_s.copy()
                val_df['calib_prob'] = calib_v
                val_df['odds_num']   = pd.to_numeric(val_s['単勝オッズ'], errors='coerce')
                all_val_preds.append(val_df)
                combo_arts[surf] = art

            val_all = pd.concat(all_val_preds, ignore_index=True)
            val_all['rank_model'] = val_all.groupby('race_id')['calib_prob'].rank(
                ascending=False, method='first')
            v_top1 = val_all[val_all['rank_model'] == 1]
            v_won  = v_top1['着順_num'] == 1
            v_roi  = (v_top1.loc[v_won, 'odds_num'] * 100).sum() / (len(v_top1) * 100) - 1
            print(f'  芝TOP_K={top_k_turf:2d}  ダTOP_K={top_k_dirt:2d} → val ROI={v_roi:+.3f}  val_芝={combo_arts["芝"]["best_val"]:.4f}  val_ダ={combo_arts["ダ"]["best_val"]:.4f}')

            if v_roi > best_val_roi:
                best_val_roi = v_roi
                best_combo = (top_k_turf, top_k_dirt, combo_arts)

    best_k_turf, best_k_dirt, best_arts = best_combo
    print(f'\nベスト組み合わせ: 芝TOP_K={best_k_turf}  ダTOP_K={best_k_dirt}  val ROI={best_val_roi:+.3f}')

    # ── OOS 評価 ──────────────────────────────────────────────────────────
    print('\nOOS 評価中...')
    all_oos = []
    for surf in ['芝', 'ダ']:
        art   = best_arts[surf]
        oos_s = df[(df['日付_num'] >= 230101) & (df['surface'] == surf)].sort_values('race_id').reset_index(drop=True)
        pred  = predict_one(oos_s, art)
        oos_sorted = oos_s.sort_values('race_id').reset_index(drop=True)
        pred['odds_num']    = pd.to_numeric(oos_sorted['単勝オッズ'], errors='coerce').values
        pred['market_prob'] = 1.0 / pred['odds_num']
        pred['ev_score']    = pred['calib_prob'] - pred['market_prob'] * 0.80
        pred['yr']          = pred['日付_num'] // 10000
        pred['surface']     = surf
        all_oos.append(pred)

    all_pred_df = pd.concat(all_oos, ignore_index=True)

    print('\n' + '='*55)
    print(f'表面別lambdarank + surface_clogit v2')
    print(f'芝TOP_K={best_k_turf}  ダTOP_K={best_k_dirt}')
    print('='*55)
    total_roi, ranked_df = eval_oos(all_pred_df, 'rank=1 全体')

    print('\n--- EV フィルタ ---')
    for thr in [0.00, 0.01, 0.02, 0.03, 0.05]:
        ev = ranked_df[(ranked_df['rank_model'] == 1) & (ranked_df['ev_score'] > thr)]
        if len(ev) >= 200:
            won = ev['着順_num'] == 1
            r   = (ev.loc[won, 'odds_num'] * 100).sum() / (len(ev) * 100) - 1
            print(f'  EV>{thr:.2f}: {len(ev):5d}件  win={won.mean():.3f}  ROI={r:+.3f}')

    print('\n--- 表面別サマリ ---')
    for surf in ['芝', 'ダ']:
        sub = ranked_df[(ranked_df['rank_model'] == 1) & (ranked_df['surface'] == surf)]
        won = sub['着順_num'] == 1
        r   = (sub.loc[won, 'odds_num'] * 100).sum() / (len(sub) * 100) - 1
        print(f'  {surf}: {len(sub):5d}R  win={won.mean():.3f}  ROI={r:+.3f}')

    print(f'\n合算 ROI: {total_roi:+.3f}')
    mark = ' ← 目標達成(-10%)!' if total_roi >= -0.10 else f'  (あと{abs(-0.10 - total_roi):.3f})'
    print(mark)

    # 比較ベースライン
    print(f'\n[参考] 前モデル(surface_clogit v1, 全体TOP_K=35): ROI=-0.132')

    # 保存
    out_pkl = os.path.join(MODEL_DIR, 'surface_clogit_v2.pkl')
    with open(out_pkl, 'wb') as f:
        pickle.dump({
            'artifacts': best_arts,
            'feat_cols': feat_cols,
            'top_k_turf': best_k_turf,
            'top_k_dirt': best_k_dirt,
            'total_oos_roi': total_roi,
        }, f)
    print(f'保存完了: {out_pkl}')


if __name__ == '__main__':
    main()
