# coding: utf-8
"""
芝ダ別 条件付きロジット v7
- 前走単勝オッズ / 2走前単勝オッズ を特徴量に追加
- 過去オッズはリークなし（現在レースのオッズは依然除外）
- 仮説: 市場の過去の評価が win 予測に追加情報を与える
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

# 現在レースのオッズ（除外）以外の過去オッズを追加
PAST_ODDS_COLS = [
    '前走単勝オッズ', '1走前_単勝オッズ',
    '2走前_単勝オッズ', '3走前_単勝オッズ',
]


def make_race_id(df):
    df['race_id'] = (df['日付_num'].astype(int).astype(str) + '_' +
                     df['開催'].astype(str).str.strip() + '_' +
                     df['Ｒ'].astype(str).str.strip())
    return df


def get_surface(df):
    return df['距離'].astype(str).str.strip().str.extract(r'^([芝ダ])')[0].fillna('不明')


def get_top_idx(feat_cols_base):
    """lambdarank 重要度で TOP_K を選択（past odds は TOP_K 対象外）"""
    with open(os.path.join(MODEL_DIR, 'lambdarank_pace.pkl'), 'rb') as f:
        lgbm = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'lambdarank_pace_info.json'), encoding='utf-8') as f:
        info = json.load(f)
    lgbm_feats = info['feat_cols']
    imps = lgbm.feature_importance(importance_type='gain')
    feat_set = set(feat_cols_base)
    ranked = sorted(
        [(lgbm_feats[i], imps[i]) for i in range(len(lgbm_feats)) if lgbm_feats[i] in feat_set],
        key=lambda x: -x[1]
    )
    top_names = [f for f, _ in ranked[:TOP_K] if f in feat_cols_base]
    return np.array([feat_cols_base.index(f) for f in top_names])


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

    # 過去オッズを対数変換して追加（外れ値対策）
    for col in PAST_ODDS_COLS:
        if col in df.columns:
            log_col = f'log_{col}'
            df[log_col] = np.log1p(pd.to_numeric(df[col], errors='coerce').fillna(0).clip(lower=0))

    num_cols  = df.select_dtypes(include='number').columns.tolist()
    # 現在レースのオッズは除外、過去オッズの log 変換は追加
    feat_cols_base = [c for c in num_cols
                      if c not in EXCLUDE and c not in ODDS_REMOVE
                      and not c.startswith('log_')]
    log_past_odds = [f'log_{c}' for c in PAST_ODDS_COLS if f'log_{c}' in df.columns]
    feat_cols = feat_cols_base + log_past_odds

    print(f'特徴量: {len(feat_cols)}列 (うち過去オッズ: {len(log_past_odds)}列)')

    # top_idx は base 列のみで選択（past odds は交互作用対象外）
    top_idx = get_top_idx(feat_cols_base)

    all_oos = []
    artifacts = {}

    for surf in ['芝', 'ダ']:
        print(f'\n{"="*50}')
        print(f'馬場面: {surf}')

        trn_s = df[(df['日付_num'] >= 130101) & (df['日付_num'] < 210101) & (df['surface'] == surf)].sort_values('race_id').reset_index(drop=True)
        val_s = df[(df['日付_num'] >= 210101) & (df['日付_num'] <= 221231) & (df['surface'] == surf)].sort_values('race_id').reset_index(drop=True)
        oos_s = df[(df['日付_num'] >= 230101) & (df['surface'] == surf)].sort_values('race_id').reset_index(drop=True)

        print(f'  学習: {trn_s["race_id"].nunique():,}R  val: {val_s["race_id"].nunique():,}R  OOS: {oos_s["race_id"].nunique():,}R')

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

        pr_v = segment_softmax(X_va @ beta, gs_va, n_va)
        ir = IsotonicRegression(out_of_bounds='clip', increasing=True)
        ir.fit(pr_v, y_va)

        # val ROI (ハイブリッド ranking factor=0 と factor=0.15 の比較)
        val_out = val_s.copy()
        val_out['calib_prob'] = ir.predict(pr_v)
        val_out['odds_num']   = pd.to_numeric(val_s['単勝オッズ'], errors='coerce').values
        val_out['market_prob'] = 1.0 / val_out['odds_num'].clip(lower=1.0)
        val_out['rank_pure']  = val_out.groupby('race_id')['calib_prob'].rank(ascending=False, method='first')
        top1_v = val_out[val_out['rank_pure'] == 1]
        won_v  = top1_v['着順_num'] == 1
        val_roi = (top1_v.loc[won_v, 'odds_num'] * 100).sum() / (len(top1_v) * 100) - 1
        print(f'  val ROI (rank=1)={val_roi:+.3f}')

        X_oo, y_oo, gs_oo, n_oo, *_ = prepare(
            oos_s, feat_cols,
            scaler=scaler, poly2=poly2, inter_scaler2=iscaler2, top_idx=top_idx,
            poly3=None, inter_scaler3=None, top_idx3=None, fit=False)
        raw   = segment_softmax(X_oo @ beta, gs_oo, n_oo)
        calib = ir.predict(raw)

        oos_out = oos_s.copy()
        oos_out['model_prob']  = raw
        oos_out['calib_prob']  = calib
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
            'feat_cols': feat_cols, 'isotonic': ir,
        }

    print(f'\n{"="*50}')
    print('芝+ダ合算 OOS (v7: 過去オッズ特徴量追加)')
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
    print('[比較] v1 (ベスト): -0.132')

    out_pkl = os.path.join(MODEL_DIR, 'surface_clogit_v7.pkl')
    with open(out_pkl, 'wb') as f:
        pickle.dump({'artifacts': artifacts, 'feat_cols': feat_cols, 'total_oos_roi': total_roi}, f)
    print(f'保存完了: {out_pkl}')


if __name__ == '__main__':
    main()
