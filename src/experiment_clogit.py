# coding: utf-8
"""
条件付きロジット 3実験を一括実行

実験1: alpha=2.0 + TOP_K=50（強正則化で過学習抑制）
実験2: 学習データを2016〜に絞る（古い市場特性を除外）
実験3: 相互情報量ベースの交互作用特徴量選択
"""
import sys, io, os, json, pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.isotonic import IsotonicRegression
from sklearn.feature_selection import mutual_info_classif

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from save_lambdarank_pace import add_pace_features, EXCLUDE, ODDS_REMOVE
from save_conditional_logit import (
    add_new_features, segment_softmax, neg_log_lik_and_grad, get_group_starts
)

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, 'data', 'processed', 'all_venues_features.parquet')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

LR       = 0.001
N_EPOCHS = 800
PATIENCE = 100
CHECK_EVERY = 10


def prepare_data(df, feat_cols, scaler, poly2, iscaler2, top_idx, fit=False):
    df = df.sort_values('race_id').reset_index(drop=True)
    X_raw = df[feat_cols].astype(float).fillna(0).values
    if fit:
        X_sc = scaler.fit_transform(X_raw)
    else:
        X_sc = scaler.transform(X_raw)

    if top_idx is not None:
        X_top = X_sc[:, top_idx]
        if fit:
            X_p2 = poly2.fit_transform(X_top)
            X_inter = iscaler2.fit_transform(X_p2[:, len(top_idx):])
        else:
            X_p2 = poly2.transform(X_top)
            X_inter = iscaler2.transform(X_p2[:, len(top_idx):])
        X = np.hstack([X_sc, X_inter])
    else:
        X = X_sc

    y  = (df['着順_num'] == 1).astype(float).values
    gs = get_group_starts(df['race_id'].values)
    return X, y, gs, len(y), len(gs)


def train_adam(X_tr, y_tr, gs_tr, n_tr, nr_tr,
               X_va, y_va, gs_va, n_va, nr_va, alpha):
    d = X_tr.shape[1]
    beta = np.zeros(d)
    m, v = np.zeros(d), np.zeros(d)
    b1, b2, eps = 0.9, 0.999, 1e-8
    t = 0
    best_val, best_beta, no_improve = np.inf, beta.copy(), 0

    for epoch in range(1, N_EPOCHS + 1):
        loss, grad = neg_log_lik_and_grad(beta, X_tr, y_tr, gs_tr, n_tr, nr_tr)
        # alpha を動的に渡すため、neg_log_lik_and_grad の ALPHA を上書き
        scores = X_tr @ beta
        probs  = segment_softmax(scores, gs_tr, n_tr)
        log_lik = np.sum(y_tr * np.log(np.clip(probs, 1e-15, 1.0)))
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
                best_val = vl
                best_beta = beta.copy()
                no_improve = 0
            else:
                no_improve += 1
            if epoch % 50 == 0:
                print(f'    epoch={epoch:4d}  val={vl:.4f}  best={best_val:.4f}')
            if no_improve >= PATIENCE // CHECK_EVERY:
                print(f'    早期停止 epoch={epoch}')
                break

    return best_beta, best_val


def calibrate_and_eval(df_val, df_oos, feat_cols, scaler, poly2, iscaler2, top_idx, beta):
    """val でcalibrate して oos で評価"""
    # val
    val_s = df_val.sort_values('race_id').reset_index(drop=True)
    X_v, y_v, gs_v, n_v, _ = prepare_data(val_s, feat_cols, scaler, poly2, iscaler2, top_idx)
    pr_v = segment_softmax(X_v @ beta, gs_v, n_v)
    ir = IsotonicRegression(out_of_bounds='clip', increasing=True)
    ir.fit(pr_v, y_v)

    # oos
    oos_s = df_oos.sort_values('race_id').reset_index(drop=True)
    X_o, y_o, gs_o, n_o, _ = prepare_data(oos_s, feat_cols, scaler, poly2, iscaler2, top_idx)
    pr_o   = segment_softmax(X_o @ beta, gs_o, n_o)
    cp_o   = ir.predict(pr_o)

    oos_s['model_prob'] = cp_o
    oos_s['rank_model'] = oos_s.groupby('race_id')['model_prob'].rank(ascending=False, method='first')
    oos_s['odds_num']   = pd.to_numeric(df_oos.sort_values('race_id').reset_index(drop=True)['単勝オッズ'], errors='coerce')
    oos_s['market_prob']= 1.0 / oos_s['odds_num']
    oos_s['ev_score']   = oos_s['model_prob'] - oos_s['market_prob'] * 0.8
    oos_s['yr']         = oos_s['日付_num'] // 10000

    top1 = oos_s[oos_s['rank_model'] == 1]
    won = top1['着順_num'] == 1
    roi_all = (top1.loc[won, 'odds_num'] * 100).sum() / (len(top1) * 100) - 1

    results = {'rank=1全体': roi_all}
    for thr in [0.00, 0.02, 0.03, 0.05]:
        ev = oos_s[(oos_s['rank_model'] == 1) & (oos_s['ev_score'] > thr)]
        if len(ev) >= 200:
            w = ev['着順_num'] == 1
            r = (ev.loc[w, 'odds_num'] * 100).sum() / (len(ev) * 100) - 1
            results[f'EV>{thr}'] = (r, len(ev))

    return results


def run_experiment(name, df, feat_cols, trn, val, oos, alpha, top_k, use_mi=False):
    print(f'\n{"="*60}')
    print(f'実験: {name}  alpha={alpha}  TOP_K={top_k}  MI={use_mi}')
    print(f'{"="*60}')
    print(f'学習: {len(trn):,}行 / val: {len(val):,}行 / OOS: {len(oos):,}行')

    scaler   = StandardScaler()
    poly2    = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    iscaler2 = StandardScaler()

    # 特徴量インデックス選択
    trn_s = trn.sort_values('race_id').reset_index(drop=True)
    X_raw_tr = trn_s[feat_cols].astype(float).fillna(0).values
    X_sc_tr  = scaler.fit_transform(X_raw_tr)
    y_tr_raw = (trn_s['着順_num'] == 1).astype(float).values

    if use_mi:
        print('  相互情報量を計算中...')
        mi = mutual_info_classif(X_sc_tr, y_tr_raw, random_state=42, n_neighbors=5)
        top_idx = np.argsort(mi)[::-1][:top_k]
        top_names = [feat_cols[i] for i in top_idx[:5]]
        print(f'  MI上位5: {top_names}')
    else:
        lgbm_path = os.path.join(MODEL_DIR, 'lambdarank_pace.pkl')
        info_path = os.path.join(MODEL_DIR, 'lambdarank_pace_info.json')
        with open(lgbm_path, 'rb') as f: lgbm_model = pickle.load(f)
        with open(info_path, encoding='utf-8') as f: lgbm_info = json.load(f)
        lgbm_feats = lgbm_info['feat_cols']
        imps = lgbm_model.feature_importance(importance_type='gain')
        feat_set = set(feat_cols)
        ranked = sorted(
            [(lgbm_feats[i], imps[i]) for i in range(len(lgbm_feats)) if lgbm_feats[i] in feat_set],
            key=lambda x: -x[1]
        )
        top_names_full = [f for f, _ in ranked[:top_k]]
        top_idx = np.array([feat_cols.index(f) for f in top_names_full if f in feat_cols])
        print(f'  LGB重要度上位5: {top_names_full[:5]}')

    # 2-way 交互作用
    X_top = X_sc_tr[:, top_idx]
    X_p2  = poly2.fit_transform(X_top)
    X_inter = iscaler2.fit_transform(X_p2[:, len(top_idx):])
    X_tr  = np.hstack([X_sc_tr, X_inter])
    gs_tr = get_group_starts(trn_s['race_id'].values)
    nr_tr = len(gs_tr)
    n_tr  = len(y_tr_raw)
    print(f'  特徴量次元: {X_tr.shape[1]}  (2-way: {X_inter.shape[1]}件)')

    # val 準備
    val_s   = val.sort_values('race_id').reset_index(drop=True)
    X_raw_v = val_s[feat_cols].astype(float).fillna(0).values
    X_sc_v  = scaler.transform(X_raw_v)
    X_p2_v  = poly2.transform(X_sc_v[:, top_idx])
    X_va    = np.hstack([X_sc_v, iscaler2.transform(X_p2_v[:, len(top_idx):])])
    y_va    = (val_s['着順_num'] == 1).astype(float).values
    gs_va   = get_group_starts(val_s['race_id'].values)
    nr_va   = len(gs_va)
    n_va    = len(y_va)

    print('  Adam 最適化中...')
    beta, best_val = train_adam(
        X_tr, y_tr_raw, gs_tr, n_tr, nr_tr,
        X_va, y_va,     gs_va, n_va, nr_va,
        alpha=alpha
    )
    print(f'  最適化完了 best_val={best_val:.4f}')

    results = calibrate_and_eval(val, oos, feat_cols, scaler, poly2, iscaler2, top_idx, beta)
    print(f'  結果:')
    for k, v in results.items():
        if isinstance(v, tuple):
            print(f'    {k}: ROI={v[0]:+.3f}  件数={v[1]}')
        else:
            print(f'    {k}: ROI={v:+.3f}')
    return results


def main():
    print('データ読み込み...')
    df = pd.read_parquet(DATA_FILE)
    df['日付_num'] = pd.to_numeric(df['日付'], errors='coerce')
    df['着順_num'] = pd.to_numeric(df['着順_num'], errors='coerce')
    df = df.dropna(subset=['日付_num', '着順_num'])
    df = df[df['着順_num'] < 99]
    df['race_id'] = (df['日付_num'].astype(int).astype(str) + '_' +
                     df['開催'].astype(str).str.strip() + '_' +
                     df['Ｒ'].astype(str).str.strip())
    df = add_pace_features(df)
    df = add_new_features(df)

    num_cols  = df.select_dtypes(include='number').columns.tolist()
    feat_cols = [c for c in num_cols if c not in EXCLUDE and c not in ODDS_REMOVE]
    print(f'特徴量: {len(feat_cols)}列')

    val = df[(df['日付_num'] >= 210101) & (df['日付_num'] <= 221231)]
    oos = df[df['日付_num'] >= 230101]

    all_results = {}

    # ── 実験1: alpha=2.0 + TOP_K=50 ─────────────────────────────────────
    trn1 = df[(df['日付_num'] >= 130101) & (df['日付_num'] < 210101)]
    all_results['exp1_alpha2_k50'] = run_experiment(
        'alpha=2.0 + TOP_K=50', df, feat_cols, trn1, val, oos,
        alpha=2.0, top_k=50, use_mi=False
    )

    # ── 実験2: 2016スタート (alpha=1.0, TOP_K=35) ─────────────────────────
    trn2 = df[(df['日付_num'] >= 160101) & (df['日付_num'] < 210101)]
    all_results['exp2_2016start'] = run_experiment(
        '2016スタート + alpha=1.0 + TOP_K=35', df, feat_cols, trn2, val, oos,
        alpha=1.0, top_k=35, use_mi=False
    )

    # ── 実験3: 相互情報量ベース (alpha=1.0, TOP_K=35) ────────────────────
    trn3 = df[(df['日付_num'] >= 130101) & (df['日付_num'] < 210101)]
    all_results['exp3_mi'] = run_experiment(
        '相互情報量ベース + alpha=1.0 + TOP_K=35', df, feat_cols, trn3, val, oos,
        alpha=1.0, top_k=35, use_mi=True
    )

    # ── 比較サマリ ─────────────────────────────────────────────────────────
    print('\n' + '='*60)
    print('比較サマリ（現在ベスト: EV>0.03 ROI=-7.8%）')
    print('='*60)
    for exp, res in all_results.items():
        ev03 = res.get('EV>0.03', ('N/A', 0))
        ev02 = res.get('EV>0.02', ('N/A', 0))
        print(f'{exp}:')
        print(f'  rank=1全体={res["rank=1全体"]:+.3f}  EV>0.02={ev02}  EV>0.03={ev03}')


if __name__ == '__main__':
    main()
