# coding: utf-8
"""
芝ダ別 条件付きロジット v5
- 市場確率 (1/オッズ) を学習特徴量に追加
- 近年重み付け学習 (exp decay で直近を重視)
- 目標: rank=1全買い OOS ROI >= -10%
"""
import os, sys, json, pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import save_lambdarank_pace as _slp
from save_lambdarank_pace import add_pace_features, EXCLUDE, ODDS_REMOVE
from save_conditional_logit import (
    add_new_features, segment_softmax, get_group_starts
)

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, 'data', 'processed', 'all_venues_features.parquet')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

LR          = 0.001
N_EPOCHS    = 800
PATIENCE    = 100
CHECK_EVERY = 10
ALPHA       = 1.0
TOP_K       = 35
DECAY_GAMMA = 0.2   # 年あたりの重み減衰 (gamma=0.2: 2013年 → 0.25x, 2020年 → 1.0x)


def make_race_id(df):
    df['race_id'] = (df['日付_num'].astype(int).astype(str) + '_' +
                     df['開催'].astype(str).str.strip() + '_' +
                     df['Ｒ'].astype(str).str.strip())
    return df


def get_surface(df):
    return df['距離'].astype(str).str.strip().str.extract(r'^([芝ダ])')[0].fillna('不明')


def get_top_idx(feat_cols):
    with open(os.path.join(MODEL_DIR, 'lambdarank_pace.pkl'), 'rb') as f:
        lgbm = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'lambdarank_pace_info.json'), encoding='utf-8') as f:
        info = json.load(f)
    lgbm_feats = info['feat_cols']
    imps = lgbm.feature_importance(importance_type='gain')
    feat_set = set(feat_cols)
    ranked = sorted(
        [(lgbm_feats[i], imps[i]) for i in range(len(lgbm_feats)) if lgbm_feats[i] in feat_set],
        key=lambda x: -x[1]
    )
    top_names = [f for f, _ in ranked[:TOP_K] if f in feat_cols]
    return np.array([feat_cols.index(f) for f in top_names])


def build_X(df, feat_cols, scaler=None, poly2=None, iscaler2=None, top_idx=None, fit=False):
    """特徴量行列を構築 (market_prob 列を含む)"""
    df = df.sort_values('race_id').reset_index(drop=True)
    X_raw = df[feat_cols].astype(float).fillna(0).values

    if fit:
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X_raw)
    else:
        X_sc = scaler.transform(X_raw)

    if top_idx is not None:
        X_top = X_sc[:, top_idx]
        if fit:
            poly2  = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
            X_p2   = poly2.fit_transform(X_top)
            iscaler2 = StandardScaler()
            X_inter = iscaler2.fit_transform(X_p2[:, len(top_idx):])
        else:
            X_p2    = poly2.transform(X_top)
            X_inter = iscaler2.transform(X_p2[:, len(top_idx):])
        X = np.hstack([X_sc, X_inter])
    else:
        X = X_sc

    y  = (df['着順_num'] == 1).astype(float).values
    gs = get_group_starts(df['race_id'].values)
    return X, y, gs, len(y), len(gs), scaler, poly2, iscaler2


def make_sample_weights(df, base_year=20):
    """各行に近年重みを付与。base_year (2桁) からの経過年数で exp decay"""
    yr2 = df['日付_num'].astype(str).str[:2].astype(float)
    w   = np.exp(DECAY_GAMMA * (yr2 - base_year + 7))   # 2013(=base-7) → 1x, 2020(=base) → e^(0.2*7)≈4x
    w   = w / w.mean()
    return w.values


def make_race_weights(sample_w, gs, n):
    """サンプル重みから先頭馬の重みをレース重みとして使用"""
    race_w = sample_w[gs]  # 各レースの先頭馬の重み
    return race_w / race_w.mean()


def train_adam_weighted(X_tr, y_tr, gs_tr, nr_tr, sample_w_tr,
                        X_va, y_va, gs_va, nr_va):
    d = X_tr.shape[1]
    n_tr, n_va = len(y_tr), len(y_va)
    race_w = make_race_weights(sample_w_tr, gs_tr, n_tr)  # (nr_tr,)
    group_sizes = np.diff(np.append(gs_tr, n_tr))
    sample_w_tr_expanded = np.repeat(race_w, group_sizes)  # (n_tr,) — 各馬にレース重み

    beta = np.zeros(d)
    m, v = np.zeros(d), np.zeros(d)
    b1, b2, eps = 0.9, 0.999, 1e-8
    t = 0
    best_val, best_beta, no_improve = np.inf, beta.copy(), 0
    W_sum = race_w.sum()

    for epoch in range(1, N_EPOCHS + 1):
        scores    = X_tr @ beta
        probs     = segment_softmax(scores, gs_tr, n_tr)
        log_lik   = np.sum(sample_w_tr_expanded * y_tr * np.log(np.clip(probs, 1e-15, 1.0)))
        residuals = sample_w_tr_expanded * (y_tr - probs)
        loss = (-log_lik + ALPHA * np.sum(beta**2)) / W_sum
        grad = (-(X_tr.T @ residuals) + 2 * ALPHA * beta) / W_sum

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
            vl   = (-ll_v + ALPHA * np.sum(beta**2)) / nr_va
            if vl < best_val:
                best_val = vl; best_beta = beta.copy(); no_improve = 0
            else:
                no_improve += 1
            if no_improve >= PATIENCE // CHECK_EVERY:
                break

    return best_beta, best_val


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

    # 市場確率を特徴量として追加（事前情報）
    df['market_prob'] = 1.0 / pd.to_numeric(df['単勝オッズ'], errors='coerce').clip(lower=1.0)
    df['market_prob'] = df['market_prob'].fillna(df.groupby('race_id')['market_prob'].transform('mean'))

    num_cols  = df.select_dtypes(include='number').columns.tolist()
    feat_cols_base = [c for c in num_cols if c not in EXCLUDE and c not in ODDS_REMOVE]

    # market_prob を末尾に追加（まだ ODDS_REMOVE にはない派生列）
    if 'market_prob' not in feat_cols_base:
        feat_cols = feat_cols_base + ['market_prob']
    else:
        feat_cols = feat_cols_base
    print(f'特徴量: {len(feat_cols)}列 (market_prob 含む)')

    top_idx = get_top_idx(feat_cols_base)   # market_prob は TOP_K 対象外（全体で使用）

    all_oos = []
    artifacts = {}

    for surf in ['芝', 'ダ']:
        print(f'\n{"="*50}')
        print(f'馬場面: {surf}')

        trn_s = df[(df['日付_num'] >= 130101) & (df['日付_num'] < 210101) & (df['surface'] == surf)].sort_values('race_id').reset_index(drop=True)
        val_s = df[(df['日付_num'] >= 210101) & (df['日付_num'] <= 221231) & (df['surface'] == surf)].sort_values('race_id').reset_index(drop=True)
        oos_s = df[(df['日付_num'] >= 230101) & (df['surface'] == surf)].sort_values('race_id').reset_index(drop=True)

        print(f'  学習: {trn_s["race_id"].nunique():,}R  val: {val_s["race_id"].nunique():,}R  OOS: {oos_s["race_id"].nunique():,}R')

        X_tr, y_tr, gs_tr, n_tr, nr_tr, scaler, poly2, iscaler2 = build_X(
            trn_s, feat_cols, fit=True, top_idx=top_idx)[:8]
        X_va, y_va, gs_va, n_va, nr_va, *_ = build_X(
            val_s, feat_cols, scaler=scaler, poly2=poly2, iscaler2=iscaler2,
            top_idx=top_idx, fit=False)

        sample_w = make_sample_weights(trn_s.sort_values('race_id').reset_index(drop=True))
        print(f'  重み確認: yr2013≈{np.exp(DECAY_GAMMA*0):.2f}x, yr2020≈{np.exp(DECAY_GAMMA*7):.2f}x')

        print('  Adam（重み付き）最適化中...')
        beta, best_val = train_adam_weighted(
            X_tr, y_tr, gs_tr, nr_tr, sample_w,
            X_va, y_va, gs_va, nr_va)
        print(f'  best_val={best_val:.4f}')

        pr_v = segment_softmax(X_va @ beta, gs_va, n_va)
        ir = IsotonicRegression(out_of_bounds='clip', increasing=True)
        ir.fit(pr_v, y_va)

        X_oo, y_oo, gs_oo, n_oo, *_ = build_X(
            oos_s, feat_cols, scaler=scaler, poly2=poly2, iscaler2=iscaler2,
            top_idx=top_idx, fit=False)
        raw   = segment_softmax(X_oo @ beta, gs_oo, n_oo)
        calib = ir.predict(raw)

        oos_out = oos_s.sort_values('race_id').reset_index(drop=True).copy()
        oos_out['model_prob']  = raw
        oos_out['calib_prob']  = calib
        oos_out['odds_num']    = pd.to_numeric(oos_out['単勝オッズ'], errors='coerce')
        oos_out['market_prob_ev'] = 1.0 / oos_out['odds_num']
        oos_out['ev_score']    = oos_out['calib_prob'] - oos_out['market_prob_ev'] * 0.80
        oos_out['yr']          = oos_out['日付_num'] // 10000
        oos_out['surface']     = surf

        oos_out['rank_model'] = oos_out.groupby('race_id')['calib_prob'].rank(
            ascending=False, method='first')
        top1 = oos_out[oos_out['rank_model'] == 1]
        won  = top1['着順_num'] == 1
        r    = (top1.loc[won, 'odds_num'] * 100).sum() / (len(top1) * 100) - 1
        print(f'  OOS ROI={r:+.3f}  ({len(top1)}R  win={won.mean():.3f})')

        all_oos.append(oos_out)
        artifacts[surf] = {
            'coef': beta, 'scaler': scaler, 'poly2': poly2,
            'inter_scaler2': iscaler2, 'top_idx': top_idx,
            'poly3': None, 'inter_scaler3': None, 'top_idx3': None,
            'feat_cols': feat_cols, 'isotonic': ir,
        }

    print(f'\n{"="*50}')
    print('芝+ダ合算 OOS (v5: market_prob特徴量 + 近年重み付け)')
    print('='*50)

    all_pred = pd.concat(all_oos, ignore_index=True)
    all_pred['rank_model'] = all_pred.groupby('race_id')['calib_prob'].rank(
        ascending=False, method='first')

    top1_all = all_pred[all_pred['rank_model'] == 1]
    print('\n  rank=1 全体')
    for yr in sorted(top1_all['yr'].unique()):
        s   = top1_all[top1_all['yr'] == yr]
        won = s['着順_num'] == 1
        r   = (s.loc[won, 'odds_num'] * 100).sum() / (len(s) * 100) - 1
        print(f'    20{int(yr):02d}: {len(s):5d}R  win={won.mean():.3f}  ROI={r:+.3f}')
    won = top1_all['着順_num'] == 1
    total_roi = (top1_all.loc[won, 'odds_num'] * 100).sum() / (len(top1_all) * 100) - 1
    print(f'    Total: {len(top1_all):5d}R  win={won.mean():.3f}  ROI={total_roi:+.3f}')

    print('\n--- EV フィルタ ---')
    for thr in [0.00, 0.02, 0.03, 0.05]:
        ev = all_pred[(all_pred['rank_model'] == 1) & (all_pred['ev_score'] > thr)]
        if len(ev) >= 200:
            won = ev['着順_num'] == 1
            r   = (ev.loc[won, 'odds_num'] * 100).sum() / (len(ev) * 100) - 1
            print(f'  EV>{thr:.2f}: {len(ev):5d}件  win={won.mean():.3f}  ROI={r:+.3f}')

    print(f'\n合算 ROI: {total_roi:+.3f}')
    mark = ' ← 目標達成(-10%)!' if total_roi >= -0.10 else f'  (あと{abs(-0.10 - total_roi):.3f})'
    print(mark)
    print('[比較] v1(ベースライン): -0.132')

    out_pkl = os.path.join(MODEL_DIR, 'surface_clogit_v5.pkl')
    with open(out_pkl, 'wb') as f:
        pickle.dump({'artifacts': artifacts, 'feat_cols': feat_cols, 'total_oos_roi': total_roi}, f)
    print(f'保存完了: {out_pkl}')


if __name__ == '__main__':
    main()
