# coding: utf-8
"""
ストラタ別 条件付きロジットモデル
  芝_短距離 / 芝_中距離 / 芝_長距離 / ダ_短距離 / ダ_中距離以上
各ストラタで独立モデルを学習・評価し、OOS 合算 ROI を計算する。

目標: rank=1全買い OOS ROI >= -5%
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
    add_new_features, segment_softmax, neg_log_lik_and_grad, get_group_starts, prepare
)

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, 'data', 'processed', 'all_venues_features.parquet')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# ハイパーパラメータ
LR        = 0.001
N_EPOCHS  = 800
PATIENCE  = 100
CHECK_EVERY = 10
ALPHA     = 1.0
TOP_K     = 35


def get_top_idx(feat_cols):
    """lambdarank gain重要度でTOP_K特徴量のインデックスを返す"""
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
    top_names = [f for f, _ in ranked[:TOP_K] if f in feat_cols]
    return np.array([feat_cols.index(f) for f in top_names])


def assign_strata(df):
    dist_str = df['距離'].astype(str).str.strip()
    surface  = dist_str.str.extract(r'^([芝ダ])')[0].fillna('不明')
    dist_m   = pd.to_numeric(dist_str.str.extract(r'(\d+)')[0], errors='coerce')
    df = df.copy()
    df['_surface'] = surface
    df['_dist_m']  = dist_m

    def label(row):
        s, d = row['_surface'], row['_dist_m']
        if s == '芝':
            if d <= 1400: return '芝_短距離'
            elif d <= 2000: return '芝_中距離'
            else: return '芝_長距離'
        elif s == 'ダ':
            if d <= 1400: return 'ダ_短距離'
            else: return 'ダ_中距離以上'
        return '不明'

    df['strata'] = df.apply(label, axis=1)
    return df


def train_adam(X_tr, y_tr, gs_tr, nr_tr,
               X_va, y_va, gs_va, nr_va, alpha):
    d = X_tr.shape[1]
    beta = np.zeros(d)
    m, v = np.zeros(d), np.zeros(d)
    b1, b2, eps = 0.9, 0.999, 1e-8
    t = 0
    best_val, best_beta, no_improve = np.inf, beta.copy(), 0
    n_tr, n_va = len(y_tr), len(y_va)

    for epoch in range(1, N_EPOCHS + 1):
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
            if no_improve >= PATIENCE // CHECK_EVERY:
                break

    return best_beta, best_val


def train_stratum(trn_s, val_s, feat_cols, top_idx):
    """1ストラタのモデルを学習してartifactを返す"""
    # prepare は fit=True 時に scaler/poly2/iscaler2 を内部生成して返す
    X_tr, y_tr, gs_tr, n_tr, nr_tr, scaler, poly2, iscaler2, _, _ = prepare(
        trn_s, feat_cols,
        scaler=None, poly2=None, inter_scaler2=None, top_idx=top_idx,
        poly3=None, inter_scaler3=None, top_idx3=None, fit=True
    )
    X_va, y_va, gs_va, n_va, nr_va, *_ = prepare(
        val_s, feat_cols,
        scaler=scaler, poly2=poly2, inter_scaler2=iscaler2, top_idx=top_idx,
        poly3=None, inter_scaler3=None, top_idx3=None, fit=False
    )

    beta, best_val = train_adam(
        X_tr, y_tr, gs_tr, nr_tr,
        X_va, y_va, gs_va, nr_va,
        alpha=ALPHA
    )

    # isotonic calibration
    pr_v = segment_softmax(X_va @ beta, gs_va, n_va)
    ir = IsotonicRegression(out_of_bounds='clip', increasing=True)
    ir.fit(pr_v, y_va)

    return {
        'coef': beta,
        'scaler': scaler,
        'poly2': poly2,
        'inter_scaler2': iscaler2,
        'top_idx': top_idx,
        'poly3': None, 'inter_scaler3': None, 'top_idx3': None,
        'feat_cols': feat_cols,
        'isotonic': ir,
        'best_val': best_val,
    }


def predict_stratum(oos_s, artifact):
    """ストラタのOOSに予測確率を付与して返す"""
    feat_cols = artifact['feat_cols']
    beta      = artifact['coef']
    ir        = artifact['isotonic']

    X_oo, y_oo, gs_oo, n_oo, *_ = prepare(
        oos_s, feat_cols,
        scaler=artifact['scaler'],
        poly2=artifact['poly2'],
        inter_scaler2=artifact['inter_scaler2'],
        top_idx=artifact['top_idx'],
        poly3=None, inter_scaler3=None, top_idx3=None,
        fit=False
    )
    raw  = segment_softmax(X_oo @ beta, gs_oo, n_oo)
    calib = ir.predict(raw)

    oos_out = oos_s.sort_values('race_id').reset_index(drop=True).copy()
    oos_out['model_prob']  = raw
    oos_out['calib_prob']  = calib
    oos_out['y']           = y_oo
    return oos_out


def roi_table(d, label=''):
    rows = []
    for yr in sorted(d['yr'].unique()):
        s   = d[d['yr'] == yr]
        won = s['着順_num'] == 1
        r   = (s.loc[won, 'odds_num'] * 100).sum() / (len(s) * 100) - 1
        rows.append((yr, len(s), won.mean(), r))
    won = d['着順_num'] == 1
    total_r = (d.loc[won, 'odds_num'] * 100).sum() / (len(d) * 100) - 1

    print(f'  {label}')
    for yr, n, wr, r in rows:
        print(f'    20{int(yr):02d}: {n:5d}R  win={wr:.3f}  ROI={r:+.3f}')
    print(f'    Total: {len(d):5d}R  win={won.mean():.3f}  ROI={total_r:+.3f}')
    return total_r


def main():
    print('データ読み込み中...')
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
    df = assign_strata(df)

    num_cols  = df.select_dtypes(include='number').columns.tolist()
    feat_cols = [c for c in num_cols if c not in EXCLUDE and c not in ODDS_REMOVE]
    print(f'特徴量: {len(feat_cols)}列')

    top_idx = get_top_idx(feat_cols)
    print(f'TOP_K={TOP_K} インタラクション特徴量インデックス取得完了')

    strata_list = ['芝_短距離', '芝_中距離', '芝_長距離', 'ダ_短距離', 'ダ_中距離以上']

    all_oos_preds = []
    artifacts = {}

    for st in strata_list:
        print(f'\n{"="*55}')
        print(f'ストラタ: {st}')
        print(f'{"="*55}')

        trn_s = df[(df['日付_num'] >= 130101) & (df['日付_num'] < 210101) & (df['strata'] == st)].copy()
        val_s = df[(df['日付_num'] >= 210101) & (df['日付_num'] <= 221231) & (df['strata'] == st)].copy()
        oos_s = df[(df['日付_num'] >= 230101) & (df['strata'] == st)].copy()

        trn_races = trn_s['race_id'].nunique()
        val_races = val_s['race_id'].nunique()
        oos_races = oos_s['race_id'].nunique()
        print(f'  学習: {len(trn_s):,}行 ({trn_races:,}R)  val: {len(val_s):,}行 ({val_races:,}R)  OOS: {len(oos_s):,}行 ({oos_races:,}R)')

        # race_id でソート必須（clogit の group 計算のため）
        trn_s = trn_s.sort_values('race_id').reset_index(drop=True)
        val_s = val_s.sort_values('race_id').reset_index(drop=True)
        oos_s = oos_s.sort_values('race_id').reset_index(drop=True)

        print('  Adam 最適化中...')
        art = train_stratum(trn_s, val_s, feat_cols, top_idx)
        print(f'  best_val={art["best_val"]:.4f}')

        oos_pred = predict_stratum(oos_s, art)
        oos_pred['odds_num']    = pd.to_numeric(oos_s['単勝オッズ'], errors='coerce').values
        oos_pred['market_prob'] = 1.0 / oos_pred['odds_num']
        oos_pred['ev_score']    = oos_pred['calib_prob'] - oos_pred['market_prob'] * 0.80
        oos_pred['yr']          = oos_pred['日付_num'] // 10000
        oos_pred['strata']      = st

        all_oos_preds.append(oos_pred)
        artifacts[st] = art

        # ストラタ別OOS ROI
        oos_pred['rank_model'] = oos_pred.groupby('race_id')['calib_prob'].rank(
            ascending=False, method='first')
        top1 = oos_pred[oos_pred['rank_model'] == 1]
        won = top1['着順_num'] == 1
        r = (top1.loc[won, 'odds_num'] * 100).sum() / (len(top1) * 100) - 1
        print(f'  OOS ROI={r:+.3f}  ({len(top1)}R  win={won.mean():.3f})')

    # ---- 合算評価 ----
    print(f'\n{"="*55}')
    print('全ストラタ合算 OOS 評価')
    print(f'{"="*55}')

    all_pred = pd.concat(all_oos_preds, ignore_index=True)
    all_pred['rank_model'] = all_pred.groupby('race_id')['calib_prob'].rank(
        ascending=False, method='first')

    top1_all = all_pred[all_pred['rank_model'] == 1]
    total_roi = roi_table(top1_all, 'rank=1 全体（ストラタ別モデル）')

    print('\n--- EV フィルタ ---')
    for thr in [0.00, 0.01, 0.02, 0.03, 0.05]:
        ev = all_pred[(all_pred['rank_model'] == 1) & (all_pred['ev_score'] > thr)]
        if len(ev) >= 200:
            won = ev['着順_num'] == 1
            r   = (ev.loc[won, 'odds_num'] * 100).sum() / (len(ev) * 100) - 1
            print(f'  EV>{thr:.2f}: {len(ev):5d}件  win={won.mean():.3f}  ROI={r:+.3f}')

    print('\n--- ストラタ別サマリ ---')
    for st in strata_list:
        sub = top1_all[top1_all['strata'] == st]
        won = sub['着順_num'] == 1
        r   = (sub.loc[won, 'odds_num'] * 100).sum() / (len(sub) * 100) - 1
        print(f'  {st:20s}: {len(sub):5d}R  win={won.mean():.3f}  ROI={r:+.3f}')

    print(f'\n合算 ROI: {total_roi:+.3f}')
    mark = ' ← 目標達成!' if total_roi >= -0.05 else f'  (目標まであと{abs(-0.05 - total_roi):.3f})'
    print(mark)

    # 保存
    out_pkl  = os.path.join(MODEL_DIR, 'strata_clogit.pkl')
    out_json = os.path.join(MODEL_DIR, 'strata_clogit_info.json')
    with open(out_pkl, 'wb') as f:
        pickle.dump({'artifacts': artifacts, 'feat_cols': feat_cols, 'total_oos_roi': total_roi}, f)

    info = {
        'strata': strata_list,
        'feat_cols': feat_cols,
        'n_features': len(feat_cols),
        'total_oos_roi': total_roi,
    }
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f'\n保存完了: {out_pkl}')


if __name__ == '__main__':
    main()
