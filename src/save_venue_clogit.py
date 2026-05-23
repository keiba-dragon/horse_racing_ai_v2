# coding: utf-8
"""
会場別×芝ダ別 条件付きロジット
- 会場グループ: 東/中/阪/京/ローカル(名新福小函札)
- 各グループ × 表面(芝/ダ) で計10サブモデル
- ベースライン: surface_clogit.pkl (-13.2%) からの改善を目標
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
ALPHA       = 1.0
TOP_K       = 35

VENUE_GROUPS = {
    '東京': ['東'],
    '中山': ['中'],
    '阪神': ['阪'],
    '京都': ['京'],
    'ローカル': ['名', '新', '福', '小', '函', '札'],
}


def make_race_id(df):
    df['race_id'] = (df['日付_num'].astype(int).astype(str) + '_' +
                     df['開催'].astype(str).str.strip() + '_' +
                     df['Ｒ'].astype(str).str.strip())
    return df


def get_surface(df):
    return df['距離'].astype(str).str.strip().str.extract(r'^([芝ダ])')[0].fillna('不明')


def get_venue(df):
    return df['開催'].astype(str).str.strip().str[1:-1]


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


def train_adam(X_tr, y_tr, gs_tr, nr_tr, X_va, y_va, gs_va, nr_va):
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
        residuals = y_tr - probs
        grad = (-(X_tr.T @ residuals) + 2 * ALPHA * beta) / nr_tr

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


def train_submodel(trn_s, val_s, feat_cols, top_idx):
    X_tr, y_tr, gs_tr, n_tr, nr_tr, scaler, poly2, iscaler2, _, _ = prepare(
        trn_s, feat_cols,
        scaler=None, poly2=None, inter_scaler2=None, top_idx=top_idx,
        poly3=None, inter_scaler3=None, top_idx3=None, fit=True)
    X_va, y_va, gs_va, n_va, nr_va, *_ = prepare(
        val_s, feat_cols,
        scaler=scaler, poly2=poly2, inter_scaler2=iscaler2, top_idx=top_idx,
        poly3=None, inter_scaler3=None, top_idx3=None, fit=False)

    beta, best_val = train_adam(X_tr, y_tr, gs_tr, nr_tr, X_va, y_va, gs_va, nr_va)

    pr_v = segment_softmax(X_va @ beta, gs_va, n_va)
    ir = IsotonicRegression(out_of_bounds='clip', increasing=True)
    ir.fit(pr_v, y_va)

    return {
        'coef': beta, 'scaler': scaler, 'poly2': poly2,
        'inter_scaler2': iscaler2, 'top_idx': top_idx,
        'poly3': None, 'inter_scaler3': None, 'top_idx3': None,
        'feat_cols': feat_cols, 'isotonic': ir, 'best_val': best_val,
    }


def predict_submodel(oos_s, art):
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
    df['venue']   = get_venue(df)
    df = df[df['surface'].isin(['芝', 'ダ'])].copy()

    num_cols  = df.select_dtypes(include='number').columns.tolist()
    feat_cols = [c for c in num_cols if c not in EXCLUDE and c not in ODDS_REMOVE]
    print(f'特徴量: {len(feat_cols)}列')

    top_idx = get_top_idx(feat_cols)

    # 会場グループ列を付与
    venue_map = {}
    for grp, venues in VENUE_GROUPS.items():
        for v in venues:
            venue_map[v] = grp
    df['venue_grp'] = df['venue'].map(venue_map).fillna('ローカル')

    all_oos  = []
    artifacts = {}

    for grp_name, venues in VENUE_GROUPS.items():
        df_grp = df[df['venue_grp'] == grp_name].copy()

        for surf in ['芝', 'ダ']:
            key = f'{grp_name}_{surf}'
            df_s = df_grp[df_grp['surface'] == surf].copy()

            trn_s = df_s[(df_s['日付_num'] >= 130101) & (df_s['日付_num'] < 210101)].sort_values('race_id').reset_index(drop=True)
            val_s = df_s[(df_s['日付_num'] >= 210101) & (df_s['日付_num'] <= 221231)].sort_values('race_id').reset_index(drop=True)
            oos_s = df_s[df_s['日付_num'] >= 230101].sort_values('race_id').reset_index(drop=True)

            nr_trn = trn_s['race_id'].nunique()
            nr_val = val_s['race_id'].nunique()
            nr_oos = oos_s['race_id'].nunique()
            print(f'\n{key}: 学習={nr_trn:,}R  val={nr_val:,}R  OOS={nr_oos:,}R')

            if nr_trn < 300 or nr_val < 100 or nr_oos < 50:
                print(f'  → データ不足のためスキップ')
                continue

            art = train_submodel(trn_s, val_s, feat_cols, top_idx)
            print(f'  best_val={art["best_val"]:.4f}')

            pred = predict_submodel(oos_s, art)
            pred['odds_num']    = pd.to_numeric(oos_s.sort_values('race_id').reset_index(drop=True)['単勝オッズ'], errors='coerce').values
            pred['market_prob_ev'] = 1.0 / pred['odds_num']
            pred['ev_score']    = pred['calib_prob'] - pred['market_prob_ev'] * 0.80
            pred['yr']          = pred['日付_num'] // 10000
            pred['venue_grp']   = grp_name
            pred['surface']     = surf

            pred['rank_model'] = pred.groupby('race_id')['calib_prob'].rank(ascending=False, method='first')
            top1 = pred[pred['rank_model'] == 1]
            won  = top1['着順_num'] == 1
            r    = (top1.loc[won, 'odds_num'] * 100).sum() / (len(top1) * 100) - 1
            print(f'  OOS ROI={r:+.3f}  ({len(top1)}R  win={won.mean():.3f})')

            all_oos.append(pred)
            artifacts[key] = art

    print(f'\n{"="*55}')
    print('全体 OOS 合算 (会場別×芝ダ別モデル)')
    print('='*55)

    all_pred = pd.concat(all_oos, ignore_index=True)
    all_pred['rank_model'] = all_pred.groupby('race_id')['calib_prob'].rank(ascending=False, method='first')

    top1_all = all_pred[all_pred['rank_model'] == 1]
    for yr in sorted(top1_all['yr'].unique()):
        s   = top1_all[top1_all['yr'] == yr]
        won = s['着順_num'] == 1
        r   = (s.loc[won, 'odds_num'] * 100).sum() / (len(s) * 100) - 1
        print(f'  20{int(yr):02d}: {len(s):5d}R  win={won.mean():.3f}  ROI={r:+.3f}')

    won = top1_all['着順_num'] == 1
    total_roi = (top1_all.loc[won, 'odds_num'] * 100).sum() / (len(top1_all) * 100) - 1
    print(f'  Total: {len(top1_all):5d}R  win={won.mean():.3f}  ROI={total_roi:+.3f}')

    print('\n--- EV フィルタ ---')
    for thr in [0.00, 0.02, 0.03, 0.05]:
        ev = all_pred[(all_pred['rank_model'] == 1) & (all_pred['ev_score'] > thr)]
        if len(ev) >= 200:
            won = ev['着順_num'] == 1
            r   = (ev.loc[won, 'odds_num'] * 100).sum() / (len(ev) * 100) - 1
            print(f'  EV>{thr:.2f}: {len(ev):5d}件  win={won.mean():.3f}  ROI={r:+.3f}')

    print('\n--- 会場グループ別 ---')
    for grp in VENUE_GROUPS:
        sub = top1_all[top1_all['venue_grp'] == grp]
        if len(sub) == 0:
            continue
        won = sub['着順_num'] == 1
        r   = (sub.loc[won, 'odds_num'] * 100).sum() / (len(sub) * 100) - 1
        print(f'  {grp}: {len(sub):5d}R  win={won.mean():.3f}  ROI={r:+.3f}')

    print(f'\n合算 ROI: {total_roi:+.3f}')
    mark = ' ← 目標達成(-10%)!' if total_roi >= -0.10 else f'  (あと{abs(-0.10 - total_roi):.3f})'
    print(mark)
    print('[比較] surface_clogit v1 (ベスト): -0.132')

    out_pkl = os.path.join(MODEL_DIR, 'venue_clogit.pkl')
    with open(out_pkl, 'wb') as f:
        pickle.dump({'artifacts': artifacts, 'feat_cols': feat_cols, 'total_oos_roi': total_roi}, f)
    print(f'保存完了: {out_pkl}')


if __name__ == '__main__':
    main()
