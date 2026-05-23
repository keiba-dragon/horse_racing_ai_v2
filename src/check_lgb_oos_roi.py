# coding: utf-8
"""
lambdarank モデル単独の OOS ROI 確認 + clogit との ensemble テスト
"""
import sys, os, json, pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import save_lambdarank_pace as _slp
from save_lambdarank_pace import add_pace_features, EXCLUDE, ODDS_REMOVE
from save_conditional_logit import add_new_features, segment_softmax, prepare

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, 'data', 'processed', 'all_venues_features.parquet')
MODEL_DIR = os.path.join(BASE_DIR, 'models')


def main():
    # ── データ準備 ────────────────────────────────────────────────────────────
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
    df['surface'] = df['距離'].astype(str).str.strip().str.extract(r'^([芝ダ])')[0].fillna('不明')
    df = df[df['surface'].isin(['芝', 'ダ'])].copy()

    oos = df[df['日付_num'] >= 230101].sort_values('race_id').reset_index(drop=True)

    # ── lambdarank モデル ────────────────────────────────────────────────────
    with open(os.path.join(MODEL_DIR, 'lambdarank_pace.pkl'), 'rb') as f:
        lgbm = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'lambdarank_pace_info.json'), encoding='utf-8') as f:
        info = json.load(f)
    lgbm_feat_cols = info['feat_cols']

    X_lgb = oos[lgbm_feat_cols].astype(float).fillna(0).values
    lgb_scores = lgbm.predict(X_lgb)  # raw LGB scores

    # race ごとに softmax して確率化
    lgb_prob = np.zeros_like(lgb_scores)
    for race_id, grp in oos.groupby('race_id'):
        idx = grp.index
        scores = lgb_scores[idx]
        ex = np.exp(scores - scores.max())
        lgb_prob[idx] = ex / ex.sum()

    oos['lgb_prob'] = lgb_prob
    oos['lgb_score'] = lgb_scores

    # ── clogit モデル ────────────────────────────────────────────────────────
    with open(os.path.join(MODEL_DIR, 'surface_clogit.pkl'), 'rb') as f:
        pkg = pickle.load(f)
    artifacts = pkg['artifacts']
    feat_cols = pkg['feat_cols']

    clogit_probs = np.zeros(len(oos))
    for surf in ['芝', 'ダ']:
        art   = artifacts[surf]
        mask  = (oos['surface'] == surf).values
        oos_s = oos[mask].sort_values('race_id').reset_index(drop=True)
        X_oo, y_oo, gs_oo, n_oo, *_ = prepare(
            oos_s, art['feat_cols'],
            scaler=art['scaler'], poly2=art['poly2'], inter_scaler2=art['inter_scaler2'],
            top_idx=art['top_idx'],
            poly3=None, inter_scaler3=None, top_idx3=None, fit=False)
        raw = segment_softmax(X_oo @ art['coef'], gs_oo, n_oo)
        calib = art['isotonic'].predict(raw)
        # OOS行に戻す
        orig_idx = oos[mask].index
        clogit_probs[orig_idx] = calib

    oos['clogit_prob'] = clogit_probs
    oos['odds_num']    = pd.to_numeric(oos['単勝オッズ'], errors='coerce')
    oos['yr'] = oos['日付_num'] // 10000

    print('=== 単独モデル OOS ROI ===')
    for col, label in [('clogit_prob', 'clogit v1'), ('lgb_prob', 'lambdarank')]:
        oos['rank_'] = oos.groupby('race_id')[col].rank(ascending=False, method='first')
        top1 = oos[oos['rank_'] == 1]
        won  = top1['着順_num'] == 1
        r    = (top1.loc[won, 'odds_num'] * 100).sum() / (len(top1) * 100) - 1
        print(f'  {label}: {len(top1):,}R  win={won.mean():.3f}  ROI={r:+.3f}')

    print('\n=== ensemble: alpha × clogit + (1-alpha) × lgb_prob ===')
    print(f'{"alpha":>7} {"ROI":>8}  年別 2023/2024/2025/2026')
    print('-'*60)
    for alpha in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]:
        oos['ens_score'] = alpha * oos['clogit_prob'] + (1-alpha) * oos['lgb_prob']
        oos['rank_ens']  = oos.groupby('race_id')['ens_score'].rank(ascending=False, method='first')
        top1 = oos[oos['rank_ens'] == 1]
        won  = top1['着順_num'] == 1
        r    = (top1.loc[won, 'odds_num'] * 100).sum() / (len(top1) * 100) - 1
        yr_rois = []
        for yr in [23, 24, 25, 26]:
            s = top1[top1['yr'] == yr]
            w = s['着順_num'] == 1
            rv = (s.loc[w, 'odds_num'] * 100).sum() / (len(s) * 100) - 1
            yr_rois.append(f'{rv:+.3f}')
        print(f'{alpha:>7.1f} {r:>+8.3f}  {" / ".join(yr_rois)}')

    # val 期間で alpha を検証
    val = df[(df['日付_num'] >= 210101) & (df['日付_num'] <= 221231)].sort_values('race_id').reset_index(drop=True)
    X_lgb_v = val[lgbm_feat_cols].astype(float).fillna(0).values
    lgb_s_v  = lgbm.predict(X_lgb_v)
    lgb_p_v  = np.zeros_like(lgb_s_v)
    for rid, grp in val.groupby('race_id'):
        idx = grp.index; s = lgb_s_v[idx]; ex = np.exp(s - s.max())
        lgb_p_v[idx] = ex / ex.sum()
    val['lgb_prob'] = lgb_p_v
    val['odds_num'] = pd.to_numeric(val['単勝オッズ'], errors='coerce')

    clogit_v = np.zeros(len(val))
    for surf in ['芝', 'ダ']:
        art  = artifacts[surf]
        mask = (val['surface'] == surf).values
        val_s = val[mask].sort_values('race_id').reset_index(drop=True)
        X_va, y_va, gs_va, n_va, *_ = prepare(
            val_s, art['feat_cols'],
            scaler=art['scaler'], poly2=art['poly2'], inter_scaler2=art['inter_scaler2'],
            top_idx=art['top_idx'],
            poly3=None, inter_scaler3=None, top_idx3=None, fit=False)
        raw  = segment_softmax(X_va @ art['coef'], gs_va, n_va)
        calib = art['isotonic'].predict(raw)
        clogit_v[val[mask].index] = calib
    val['clogit_prob'] = clogit_v

    print('\n=== val (2021-2022) での alpha 検証 ===')
    for alpha in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]:
        val['ens'] = alpha * val['clogit_prob'] + (1-alpha) * val['lgb_prob']
        val['rk']  = val.groupby('race_id')['ens'].rank(ascending=False, method='first')
        top1 = val[val['rk'] == 1]
        won  = top1['着順_num'] == 1
        r    = (top1.loc[won, 'odds_num'] * 100).sum() / (len(top1) * 100) - 1
        print(f'  alpha={alpha:.1f}: {len(top1):,}R  win={won.mean():.3f}  ROI={r:+.3f}')


if __name__ == '__main__':
    main()
