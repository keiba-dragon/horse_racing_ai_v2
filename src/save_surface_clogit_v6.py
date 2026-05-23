# coding: utf-8
"""
芝ダ別 条件付きロジット v6
- グローバル surface clogit (v1 と同じ学習)
- isotonic calibration を会場別に実施（会場ごとのバイアスを補正）
- 京都など val=0 の会場はグローバル isotonic にフォールバック
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
MIN_VAL_RACES_FOR_VENUE_CALIB = 150  # 会場別 isotonic に必要な最低 val レース数


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


def fit_venue_isotonics(raw_val, y_va, val_df, global_ir):
    """会場別 isotonic を fit。データ不足の会場はグローバルを返す。"""
    venue_ir = {}
    for venue in val_df['venue'].unique():
        mask = (val_df['venue'] == val_df['venue']).values  # 全選択
        mask = (val_df['venue'].values == venue)
        n_races = val_df.loc[mask, 'race_id'].nunique() if 'race_id' in val_df.columns else mask.sum() // 10
        if mask.sum() >= MIN_VAL_RACES_FOR_VENUE_CALIB * 10:  # 行数で判定
            ir_v = IsotonicRegression(out_of_bounds='clip', increasing=True)
            ir_v.fit(raw_val[mask], y_va[mask])
            venue_ir[venue] = ir_v
        else:
            venue_ir[venue] = global_ir  # フォールバック
    return venue_ir


def predict_with_venue_calib(X_oo, beta, gs_oo, n_oo, oos_df, venue_ir, global_ir):
    raw = segment_softmax(X_oo @ beta, gs_oo, n_oo)
    calib = np.zeros_like(raw)
    for i, venue in enumerate(oos_df['venue'].values):
        ir = venue_ir.get(venue, global_ir)
        calib[i] = ir.predict([raw[i]])[0]
    return raw, calib


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

    all_oos = []
    artifacts = {}

    for surf in ['芝', 'ダ']:
        print(f'\n{"="*50}')
        print(f'馬場面: {surf}')

        trn_s = df[(df['日付_num'] >= 130101) & (df['日付_num'] < 210101) & (df['surface'] == surf)].sort_values('race_id').reset_index(drop=True)
        val_s = df[(df['日付_num'] >= 210101) & (df['日付_num'] <= 221231) & (df['surface'] == surf)].sort_values('race_id').reset_index(drop=True)
        oos_s = df[(df['日付_num'] >= 230101) & (df['surface'] == surf)].sort_values('race_id').reset_index(drop=True)

        print(f'  学習: {trn_s["race_id"].nunique():,}R  val: {val_s["race_id"].nunique():,}R  OOS: {oos_s["race_id"].nunique():,}R')
        print('  val 会場別レース数:')
        for v, cnt in val_s.groupby('venue')['race_id'].nunique().items():
            print(f'    {v}: {cnt:,}R')

        X_tr, y_tr, gs_tr, n_tr, nr_tr, scaler, poly2, iscaler2, _, _ = prepare(
            trn_s, feat_cols,
            scaler=None, poly2=None, inter_scaler2=None, top_idx=top_idx,
            poly3=None, inter_scaler3=None, top_idx3=None, fit=True)
        X_va, y_va, gs_va, n_va, nr_va, *_ = prepare(
            val_s, feat_cols,
            scaler=scaler, poly2=poly2, inter_scaler2=iscaler2, top_idx=top_idx,
            poly3=None, inter_scaler3=None, top_idx3=None, fit=False)

        print('  Adam 最適化中...')
        beta, best_val = train_adam(X_tr, y_tr, gs_tr, nr_tr, X_va, y_va, gs_va, nr_va)
        print(f'  best_val={best_val:.4f}')

        # グローバル isotonic
        raw_va = segment_softmax(X_va @ beta, gs_va, n_va)
        global_ir = IsotonicRegression(out_of_bounds='clip', increasing=True)
        global_ir.fit(raw_va, y_va)

        # 会場別 isotonic
        venue_ir = {}
        for venue in val_s['venue'].unique():
            mask_v = (val_s['venue'] == venue).values
            n_v_races = val_s.loc[val_s['venue'] == venue, 'race_id'].nunique()
            if n_v_races >= MIN_VAL_RACES_FOR_VENUE_CALIB:
                ir_v = IsotonicRegression(out_of_bounds='clip', increasing=True)
                ir_v.fit(raw_va[mask_v], y_va[mask_v])
                venue_ir[venue] = ir_v
                print(f'    会場別 isotonic fit: {venue} ({n_v_races}R)')
            else:
                venue_ir[venue] = global_ir
                print(f'    フォールバック (グローバル): {venue} ({n_v_races}R)')

        # OOS 予測
        X_oo, y_oo, gs_oo, n_oo, *_ = prepare(
            oos_s, feat_cols,
            scaler=scaler, poly2=poly2, inter_scaler2=iscaler2, top_idx=top_idx,
            poly3=None, inter_scaler3=None, top_idx3=None, fit=False)
        raw_oo = segment_softmax(X_oo @ beta, gs_oo, n_oo)

        # 会場別 isotonic を適用
        calib_oo = np.zeros_like(raw_oo)
        for i, venue in enumerate(oos_s['venue'].values):
            ir = venue_ir.get(venue, global_ir)
            calib_oo[i] = ir.predict([raw_oo[i]])[0]

        oos_out = oos_s.copy()
        oos_out['model_prob']  = raw_oo
        oos_out['calib_prob']  = calib_oo
        oos_out['odds_num']    = pd.to_numeric(oos_s['単勝オッズ'], errors='coerce').values
        oos_out['market_prob_ev'] = 1.0 / oos_out['odds_num']
        oos_out['ev_score']    = oos_out['calib_prob'] - oos_out['market_prob_ev'] * 0.80
        oos_out['yr']          = oos_out['日付_num'] // 10000
        oos_out['surface']     = surf

        oos_out['rank_model'] = oos_out.groupby('race_id')['calib_prob'].rank(ascending=False, method='first')
        top1 = oos_out[oos_out['rank_model'] == 1]
        won  = top1['着順_num'] == 1
        r    = (top1.loc[won, 'odds_num'] * 100).sum() / (len(top1) * 100) - 1
        print(f'  OOS ROI={r:+.3f}  ({len(top1)}R  win={won.mean():.3f})')

        all_oos.append(oos_out)
        artifacts[surf] = {
            'coef': beta, 'scaler': scaler, 'poly2': poly2,
            'inter_scaler2': iscaler2, 'top_idx': top_idx,
            'poly3': None, 'inter_scaler3': None, 'top_idx3': None,
            'feat_cols': feat_cols, 'global_isotonic': global_ir,
            'venue_isotonic': venue_ir,
        }

    print(f'\n{"="*50}')
    print('芝+ダ合算 OOS (v6: 会場別 isotonic calibration)')
    print('='*50)

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

    print(f'\n合算 ROI: {total_roi:+.3f}')
    mark = ' ← 目標達成(-10%)!' if total_roi >= -0.10 else f'  (あと{abs(-0.10 - total_roi):.3f})'
    print(mark)
    print('[比較] v1 (グローバル isotonic): -0.132')

    out_pkl = os.path.join(MODEL_DIR, 'surface_clogit_v6.pkl')
    with open(out_pkl, 'wb') as f:
        pickle.dump({'artifacts': artifacts, 'feat_cols': feat_cols, 'total_oos_roi': total_roi}, f)
    print(f'保存完了: {out_pkl}')


if __name__ == '__main__':
    main()
