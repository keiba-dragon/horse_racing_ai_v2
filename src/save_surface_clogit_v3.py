# coding: utf-8
"""
芝ダ別 条件付きロジット v3
- グローバルlambdarank重要度を使いつつ、TOP_K を表面別にチューニング
- val NLL ベースで TOP_K × alpha グリッドサーチ
- 目標: rank=1全買い OOS ROI >= -10%
"""
import os, sys, json, pickle
import numpy as np
import pandas as pd
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

# グリッドサーチ対象
TOP_K_LIST = [20, 25, 30, 35, 45]
ALPHA_LIST = [0.5, 1.0, 2.0]


def make_race_id(df):
    df['race_id'] = (df['日付_num'].astype(int).astype(str) + '_' +
                     df['開催'].astype(str).str.strip() + '_' +
                     df['Ｒ'].astype(str).str.strip())
    return df


def get_surface(df):
    return df['距離'].astype(str).str.strip().str.extract(r'^([芝ダ])')[0].fillna('不明')


def get_global_top_idx(feat_cols, top_k):
    lgbm_path = os.path.join(MODEL_DIR, 'lambdarank_pace.pkl')
    info_path = os.path.join(MODEL_DIR, 'lambdarank_pace_info.json')
    with open(lgbm_path, 'rb') as f: lgbm = pickle.load(f)
    with open(info_path, encoding='utf-8') as f: info = json.load(f)
    lgbm_feats = info['feat_cols']
    imps = lgbm.feature_importance(importance_type='gain')
    feat_set = set(feat_cols)
    ranked = sorted(
        [(lgbm_feats[i], imps[i]) for i in range(len(lgbm_feats)) if lgbm_feats[i] in feat_set],
        key=lambda x: -x[1]
    )
    top_names = [f for f, _ in ranked[:top_k] if f in feat_cols]
    return np.array([feat_cols.index(f) for f in top_names])


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


def train_one(trn_s, val_s, feat_cols, top_idx, alpha):
    X_tr, y_tr, gs_tr, n_tr, nr_tr, scaler, poly2, iscaler2, _, _ = prepare(
        trn_s, feat_cols,
        scaler=None, poly2=None, inter_scaler2=None, top_idx=top_idx,
        poly3=None, inter_scaler3=None, top_idx3=None, fit=True)
    X_va, y_va, gs_va, n_va, nr_va, *_ = prepare(
        val_s, feat_cols,
        scaler=scaler, poly2=poly2, inter_scaler2=iscaler2, top_idx=top_idx,
        poly3=None, inter_scaler3=None, top_idx3=None, fit=False)

    beta, best_val = train_adam(X_tr, y_tr, gs_tr, nr_tr,
                                X_va, y_va, gs_va, nr_va, alpha)
    pr_v = segment_softmax(X_va @ beta, gs_va, n_va)
    ir = IsotonicRegression(out_of_bounds='clip', increasing=True)
    ir.fit(pr_v, y_va)

    return {
        'coef': beta, 'scaler': scaler, 'poly2': poly2,
        'inter_scaler2': iscaler2, 'top_idx': top_idx,
        'poly3': None, 'inter_scaler3': None, 'top_idx3': None,
        'feat_cols': feat_cols, 'isotonic': ir, 'best_val': best_val,
        'alpha': alpha,
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

    # 表面別 データ分割
    splits = {}
    for surf in ['芝', 'ダ']:
        trn_s = df[(df['日付_num'] >= 130101) & (df['日付_num'] < 210101) & (df['surface'] == surf)].sort_values('race_id').reset_index(drop=True)
        val_s = df[(df['日付_num'] >= 210101) & (df['日付_num'] <= 221231) & (df['surface'] == surf)].sort_values('race_id').reset_index(drop=True)
        oos_s = df[(df['日付_num'] >= 230101) & (df['surface'] == surf)].sort_values('race_id').reset_index(drop=True)
        splits[surf] = (trn_s, val_s, oos_s)

    # ── グリッドサーチ (val NLL を最小化) ──────────────────────────────────
    print('\nグリッドサーチ中 (TOP_K × alpha, val NLL 基準)...')
    best_params = {}
    best_arts_final = {}

    for surf in ['芝', 'ダ']:
        trn_s, val_s, _ = splits[surf]
        best_vl = np.inf
        best_k, best_a, best_art = None, None, None

        for top_k in TOP_K_LIST:
            top_idx = get_global_top_idx(feat_cols, top_k)
            for alpha in ALPHA_LIST:
                art = train_one(trn_s, val_s, feat_cols, top_idx, alpha)
                vl  = art['best_val']
                print(f'  {surf} TOP_K={top_k:2d}  α={alpha:.1f}  val_NLL={vl:.4f}')
                if vl < best_vl:
                    best_vl = vl
                    best_k, best_a, best_art = top_k, alpha, art

        best_params[surf] = (best_k, best_a, best_vl)
        best_arts_final[surf] = best_art
        print(f'  → {surf} ベスト: TOP_K={best_k}  α={best_a}  val_NLL={best_vl:.4f}\n')

    # ── OOS 評価 ──────────────────────────────────────────────────────────
    print('OOS 評価中...')
    all_oos = []
    for surf in ['芝', 'ダ']:
        art   = best_arts_final[surf]
        _, _, oos_s = splits[surf]
        pred  = predict_one(oos_s, art)
        oos_s_sorted = oos_s.sort_values('race_id').reset_index(drop=True)
        pred['odds_num']    = pd.to_numeric(oos_s_sorted['単勝オッズ'], errors='coerce').values
        pred['market_prob'] = 1.0 / pred['odds_num']
        pred['ev_score']    = pred['calib_prob'] - pred['market_prob'] * 0.80
        pred['yr']          = pred['日付_num'] // 10000
        pred['surface']     = surf
        all_oos.append(pred)

    all_pred = pd.concat(all_oos, ignore_index=True)
    all_pred['rank_model'] = all_pred.groupby('race_id')['calib_prob'].rank(
        ascending=False, method='first')

    top1 = all_pred[all_pred['rank_model'] == 1]

    print('\n' + '='*55)
    for surf, (k, a, vl) in best_params.items():
        print(f'{surf}: TOP_K={k}  α={a}  val_NLL={vl:.4f}')
    print('='*55)
    print('\n  rank=1 全体')
    for yr in sorted(top1['yr'].unique()):
        s   = top1[top1['yr'] == yr]
        won = s['着順_num'] == 1
        r   = (s.loc[won, 'odds_num'] * 100).sum() / (len(s) * 100) - 1
        print(f'    20{int(yr):02d}: {len(s):5d}R  win={won.mean():.3f}  ROI={r:+.3f}')
    won = top1['着順_num'] == 1
    total_roi = (top1.loc[won, 'odds_num'] * 100).sum() / (len(top1) * 100) - 1
    print(f'    Total: {len(top1):5d}R  win={won.mean():.3f}  ROI={total_roi:+.3f}')

    print('\n--- EV フィルタ ---')
    for thr in [0.00, 0.02, 0.03, 0.05]:
        ev = all_pred[(all_pred['rank_model'] == 1) & (all_pred['ev_score'] > thr)]
        if len(ev) >= 200:
            won = ev['着順_num'] == 1
            r   = (ev.loc[won, 'odds_num'] * 100).sum() / (len(ev) * 100) - 1
            print(f'  EV>{thr:.2f}: {len(ev):5d}件  win={won.mean():.3f}  ROI={r:+.3f}')

    print('\n--- 表面別 ---')
    for surf in ['芝', 'ダ']:
        sub = top1[top1['surface'] == surf]
        won = sub['着順_num'] == 1
        r   = (sub.loc[won, 'odds_num'] * 100).sum() / (len(sub) * 100) - 1
        print(f'  {surf}: {len(sub):5d}R  win={won.mean():.3f}  ROI={r:+.3f}')

    print(f'\n合算 ROI: {total_roi:+.3f}')
    mark = ' ← 目標達成(-10%)!' if total_roi >= -0.10 else f'  (あと{abs(-0.10 - total_roi):.3f})'
    print(mark)
    print('[比較] v1(global TOP_K=35): -0.132  v2(surface lambdarank): -0.137')

    out_pkl = os.path.join(MODEL_DIR, 'surface_clogit_v3.pkl')
    with open(out_pkl, 'wb') as f:
        pickle.dump({
            'artifacts': best_arts_final,
            'feat_cols': feat_cols,
            'best_params': {k: v[:2] for k, v in best_params.items()},
            'total_oos_roi': total_roi,
        }, f)
    print(f'保存完了: {out_pkl}')


if __name__ == '__main__':
    main()
