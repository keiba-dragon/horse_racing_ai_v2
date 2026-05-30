# coding: utf-8
"""
Conditional Logistic Regression + 交互作用特徴量

P(horse i wins race r) = exp(Xᵢβ) / Σⱼ∈r exp(Xⱼβ)

修正点:
  - 損失・勾配をレース数で正規化（勾配ノルムが大きすぎてABNORMAL終了した問題を修正）
  - lambdarankの上位重要特徴量でdegree-2交互作用を生成

出力:
  models/conditional_logit.pkl
  models/conditional_logit_info.json
"""
import sys, io, os, json, pickle
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from save_lambdarank_pace import add_pace_features, EXCLUDE, ODDS_REMOVE, PACE_COLS

NEW_FEATURE_COLS = [
    '休養日数', '斤量変化', '距離変化_m', 'クラス変化', '近走着順トレンド',
    # 以下は相関が高くノイズになったためコメントアウト
    # '出走月', '枠番比率', '短期休養フラグ', '長期休養フラグ',
    # '近走最良着順', '連続top3', '前走相対着順', '近5走_相対着順平均',
]

# レース内平均+オッズ調整で補完する列と方向 (+1=高い方が良い, -1=低い方が良い, 0=調整なし)
RACE_IMPUTE_COLS = {
    '近5走_タイム指数平均':      +1,
    '近5走_タイム指数_max':      +1,
    '近5走_タイム指数_min':      +1,
    '近5走_タイム指数_range':     0,
    '近5走_タイム指数_std':       0,
    '近5走_平均着順':            -1,
    '近5走_複勝率':              +1,
    '近5走_クラス調整_平均着順': -1,
    '近5走_クラス補正スコア':    +1,
    '近3走_勝率':                +1,
    '近3走_平均着順':            -1,
    '近3走_複勝率':              +1,
    '近10走_勝率':               +1,
    '近10走_平均着順':           -1,
    '近10走_複勝率':             +1,
    '1走前_タイム指数':          +1,
    '1走前_着順_num':            -1,
    '近走着順トレンド':           0,
}


def apply_race_impute(df, alpha=0.0):
    """
    NaN を同レース内の非NaN馬の平均で埋め、alpha>0 のときオッズ相対差で調整する。
    非NaN 値は変更しない。race_id 列が必要。
    """
    df = df.copy()
    mp = 1.0 / np.clip(pd.to_numeric(df['単勝オッズ'], errors='coerce').values, 1.0, None)
    df['_mprob'] = mp

    for col, sign in RACE_IMPUTE_COLS.items():
        if col not in df.columns:
            continue
        nan_mask = df[col].isna()
        if nan_mask.sum() == 0:
            continue

        race_mean = df.groupby('race_id')[col].transform('mean')
        imputed   = race_mean.copy()

        if alpha > 0 and sign != 0:
            race_mean_mp = df.groupby('race_id')['_mprob'].transform('mean')
            race_std_mp  = df.groupby('race_id')['_mprob'].transform('std').fillna(0.01).clip(lower=0.01)
            odds_z       = (df['_mprob'] - race_mean_mp) / race_std_mp
            race_std_col = df.groupby('race_id')[col].transform('std').fillna(df[col].std())
            imputed      = imputed + sign * alpha * odds_z * race_std_col

        df.loc[nan_mask, col] = imputed[nan_mask]

    df = df.drop(columns=['_mprob'])
    return df


def _col(df, name, default=np.nan):
    """列を数値で取得。なければdefault埋め。"""
    if name in df.columns:
        return pd.to_numeric(df[name], errors='coerce')
    return pd.Series(default, index=df.index)


def add_new_features(df: pd.DataFrame) -> pd.DataFrame:
    """API事前情報のみを使った派生特徴量を追加。リークなし。"""
    df = df.copy()

    # ── 1. 休養日数 ────────────────────────────────────────────────────────
    def yymmdd_to_dt(series):
        return pd.to_datetime(
            pd.to_numeric(series, errors='coerce').fillna(0).astype(int)
              .astype(str).str.zfill(6),
            format='%y%m%d', errors='coerce'
        )
    if '日付' in df.columns and '前走日付' in df.columns:
        curr_dt = yymmdd_to_dt(df['日付'])
        prev_dt = yymmdd_to_dt(df['前走日付'])
        df['休養日数'] = (curr_dt - prev_dt).dt.days.clip(0, 365)
    else:
        df['休養日数'] = np.nan

    # ── 2. 斤量変化 ─────────────────────────────────────────────────────────
    df['斤量変化'] = _col(df, '斤量') - _col(df, '前走斤量')

    # ── 3. 距離変化 ─────────────────────────────────────────────────────────
    dist_now  = _col(df, '今回_距離_m') if '今回_距離_m' in df.columns else _col(df, '距離')
    df['距離変化_m'] = dist_now - _col(df, '1走前_距離')

    # ── 4. クラス変化 ────────────────────────────────────────────────────────
    df['クラス変化'] = _col(df, 'クラス_rank') - _col(df, '1走前_クラス_rank')

    # ── 5. 近走着順トレンド（線形傾き, 正=上昇形） ───────────────────────────
    finish_cols = [c for c in ['1走前_着順_num', '2走前_着順_num', '3走前_着順_num',
                                '4走前_着順_num', '5走前_着順_num'] if c in df.columns]
    if len(finish_cols) >= 3:
        Y = np.column_stack([pd.to_numeric(df[c], errors='coerce').values for c in finish_cols])
        x_c = np.arange(Y.shape[1], dtype=float)
        x_c -= x_c.mean()
        x_var = np.dot(x_c, x_c)
        valid = ~np.isnan(Y)
        n_valid = valid.sum(axis=1)
        row_means = np.where(n_valid > 0, np.nansum(Y, axis=1) / np.maximum(n_valid, 1), 8.0)
        Y_f = Y.copy()
        ri, ci = np.where(~valid)
        Y_f[ri, ci] = row_means[ri]
        slopes = (Y_f - Y_f.mean(axis=1, keepdims=True)) @ x_c / x_var
        slopes[n_valid < 2] = np.nan
        df['近走着順トレンド'] = slopes
    else:
        df['近走着順トレンド'] = np.nan

    return df

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE  = os.path.join(BASE_DIR, 'data', 'processed', 'all_venues_features.parquet')
MODEL_DIR  = os.path.join(BASE_DIR, 'models')

ALPHA    = 1.0    # L2正則化
TOP_K    = 35     # 2-way交互作用の対象特徴量数（50はval_loss悪化）
TOP_K3   = 10     # 3-way交互作用の対象特徴量数
LR       = 0.001  # Adam学習率
N_EPOCHS = 800    # Adamエポック数（早期停止あり）
PATIENCE = 100    # 早期停止のpatience


# ── セグメントsoftmax（ベクトル化） ────────────────────────────────────────
def segment_softmax(scores, group_starts, n):
    group_sizes = np.diff(np.append(group_starts, n))
    group_max   = np.maximum.reduceat(scores, group_starts)
    elem_max    = np.repeat(group_max, group_sizes)
    exp_s       = np.exp(scores - elem_max)
    group_sum   = np.add.reduceat(exp_s, group_starts)
    elem_sum    = np.repeat(group_sum, group_sizes)
    return exp_s / elem_sum


def neg_log_lik_and_grad(beta, X, y, group_starts, n, n_races):
    """損失・勾配をレース数で割って正規化。勾配のスケールをO(1)に保つ。"""
    scores = X @ beta
    probs  = segment_softmax(scores, group_starts, n)
    log_lik = np.sum(y * np.log(np.clip(probs, 1e-15, 1.0)))
    residuals = y - probs
    loss = (-log_lik + ALPHA * np.sum(beta ** 2)) / n_races
    grad = (-(X.T @ residuals) + 2 * ALPHA * beta) / n_races
    return loss, grad


# ── データ準備 ────────────────────────────────────────────────────────────
def get_group_starts(race_ids):
    _, idx = np.unique(race_ids, return_index=True)
    return np.sort(idx)


def prepare(df, feat_cols,
            scaler=None,
            poly2=None, inter_scaler2=None, top_idx=None,
            poly3=None, inter_scaler3=None, top_idx3=None,
            fit=False):
    df = df.sort_values('race_id').reset_index(drop=True)
    X_raw = df[feat_cols].astype(float).fillna(0).values

    if fit:
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X_raw)
    else:
        X_sc = scaler.transform(X_raw)

    parts = [X_sc]

    # ── 2-way 交互作用 (top_idx) ─────────────────────────────────────────
    if top_idx is not None:
        X_top2 = X_sc[:, top_idx]
        if fit:
            poly2 = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
            X_p2 = poly2.fit_transform(X_top2)
        else:
            X_p2 = poly2.transform(X_top2)
        X_inter2 = X_p2[:, len(top_idx):]          # 単独項を除く
        if fit:
            inter_scaler2 = StandardScaler()
            X_inter2 = inter_scaler2.fit_transform(X_inter2)
        else:
            X_inter2 = inter_scaler2.transform(X_inter2)
        parts.append(X_inter2)

    # ── 3-way 交互作用 (top_idx3, degree-3のみ) ──────────────────────────
    if top_idx3 is not None:
        X_top3 = X_sc[:, top_idx3]
        if fit:
            poly3 = PolynomialFeatures(degree=3, interaction_only=True, include_bias=False)
            X_p3 = poly3.fit_transform(X_top3)
        else:
            X_p3 = poly3.transform(X_top3)
        # degree-3 の列だけ抽出
        mask3 = poly3.powers_.sum(axis=1) == 3
        X_inter3 = X_p3[:, mask3]
        if fit:
            inter_scaler3 = StandardScaler()
            X_inter3 = inter_scaler3.fit_transform(X_inter3)
        else:
            X_inter3 = inter_scaler3.transform(X_inter3)
        parts.append(X_inter3)

    X = np.hstack(parts)
    y = (df['着順_num'] == 1).astype(float).values
    gs = get_group_starts(df['race_id'].values)
    n_races = len(gs)
    return X, y, gs, len(y), n_races, scaler, poly2, inter_scaler2, poly3, inter_scaler3


# ── ROI表示 ───────────────────────────────────────────────────────────────
def roi_table(d, label):
    print(f'\n=== {label} ===')
    for yr in sorted(d['yr'].unique()):
        sub = d[d['yr'] == yr]
        won = sub['着順_num'] == 1
        r   = (sub.loc[won, 'odds_num'] * 100).sum() / (len(sub) * 100) - 1
        print(f'  20{yr:02d}: {len(sub):5d}R  win={won.mean():.3f}  ROI={r:+.3f}')
    won = d['着順_num'] == 1
    r   = (d.loc[won, 'odds_num'] * 100).sum() / (len(d) * 100) - 1
    print(f'  Total: {len(d):5d}R  win={won.mean():.3f}  ROI={r:+.3f}')


def main():
    import argparse
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument('--out-dir', default=MODEL_DIR,
                    help='モデル保存先ディレクトリ（デフォルト: models/）')
    ap.add_argument('--data-file', default=None,
                    help='入力parquetパスの上書き（実験用。未指定なら通常パス）')
    ap.add_argument('--race-impute-alpha', type=float, default=None,
                    help='レース内平均+オッズ調整 NaN補完のα（未指定=補完なし）')
    args, _ = ap.parse_known_args()
    out_dir    = args.out_dir
    data_file  = args.data_file or DATA_FILE
    model_path = os.path.join(out_dir, 'conditional_logit.pkl')
    info_path  = os.path.join(out_dir, 'conditional_logit_info.json')
    os.makedirs(out_dir, exist_ok=True)
    if out_dir != MODEL_DIR:
        print(f'[実験モード] 保存先: {out_dir}')
    if args.race_impute_alpha is not None:
        print(f'[実験モード] NaN補完: レース内平均+オッズ調整 α={args.race_impute_alpha}')

    print(f'データ読み込み: {data_file}')
    df = pd.read_parquet(data_file)
    df['日付_num'] = pd.to_numeric(df['日付'], errors='coerce')
    df['着順_num'] = pd.to_numeric(df['着順_num'], errors='coerce')
    df = df.dropna(subset=['日付_num', '着順_num'])
    df = df[df['着順_num'] < 99]
    df['race_id'] = (df['日付_num'].astype(int).astype(str) + '_' +
                     df['開催'].astype(str).str.strip() + '_' +
                     df['Ｒ'].astype(str).str.strip())
    # 開催列がない行（results_supplement由来）は race_id が 'nan' になるため除外
    before = len(df)
    df = df[df['開催'].notna()].copy()
    if len(df) < before:
        print(f'  開催NaN行を除外: {before - len(df):,}行')

    print('展開予想特徴量を追加中...')
    df = add_pace_features(df)

    print('新規特徴量を追加中（休養日数・斤量変化・距離変化・クラス変化・近走トレンド）...')
    df = add_new_features(df)
    added = [c for c in NEW_FEATURE_COLS if c in df.columns]
    print(f'  追加: {added}')

    if args.race_impute_alpha is not None:
        df = apply_race_impute(df, alpha=args.race_impute_alpha)
        n_filled = sum(df[c].isna().sum() for c in RACE_IMPUTE_COLS if c in df.columns)
        print(f'レース内平均NaN補完適用 (α={args.race_impute_alpha}) → 残NaN: {n_filled:,}')

    num_cols  = df.select_dtypes(include='number').columns.tolist()
    feat_cols = [c for c in num_cols if c not in EXCLUDE and c not in ODDS_REMOVE]
    print(f'基本特徴量: {len(feat_cols)}列')

    # ── lambdarankの重要度で交互作用対象を選ぶ ────────────────────────────
    lgbm_info_path = os.path.join(MODEL_DIR, 'lambdarank_pace_info.json')
    top_idx = None
    if os.path.exists(lgbm_info_path):
        lgbm_model_path = os.path.join(MODEL_DIR, 'lambdarank_pace.pkl')
        with open(lgbm_model_path, 'rb') as f:
            lgbm_model = pickle.load(f)
        with open(lgbm_info_path, encoding='utf-8') as f:
            lgbm_info = json.load(f)
        lgbm_feats = lgbm_info['feat_cols']
        importances = lgbm_model.feature_importance(importance_type='gain')
        # feat_cols と lgbm_feats の共通列でソート
        feat_set = set(feat_cols)
        ranked = sorted(
            [(lgbm_feats[i], importances[i]) for i in range(len(lgbm_feats)) if lgbm_feats[i] in feat_set],
            key=lambda x: -x[1]
        )
        top_names  = [f for f, _ in ranked[:TOP_K]]
        top_idx    = np.array([feat_cols.index(f) for f in top_names if f in feat_cols])
        n_inter2 = len(top_idx) * (len(top_idx) - 1) // 2
        print(f'2-way対象 Top-{len(top_idx)}: {top_names[:5]}... ({n_inter2}件)')
        # 3-way は val_loss 悪化のため無効化
        top_idx3 = None
    else:
        top_idx3 = None
        print('lambdarank_pace モデルが見つからない → 交互作用なし')

    trn = df[(df['日付_num'] >= 130101) & (df['日付_num'] <  220101)]
    val = df[(df['日付_num'] >= 220101) & (df['日付_num'] <= 221231)]
    print(f'学習: {len(trn):,}行 / valid: {len(val):,}行')

    X_tr, y_tr, gs_tr, n_tr, nr_tr, scaler, poly2, iscaler2, poly3, iscaler3 = prepare(
        trn, feat_cols, top_idx=top_idx, top_idx3=top_idx3, fit=True)
    X_va, y_va, gs_va, n_va, nr_va, *_ = prepare(
        val, feat_cols, scaler=scaler,
        poly2=poly2, inter_scaler2=iscaler2, top_idx=top_idx,
        poly3=poly3, inter_scaler3=iscaler3, top_idx3=top_idx3)

    print(f'拡張後の特徴量次元: {X_tr.shape[1]}')
    print(f'勾配ノルム確認（beta=0）: ', end='')
    probs0   = np.repeat(
        1.0 / np.diff(np.append(gs_tr, n_tr)),
        np.diff(np.append(gs_tr, n_tr))
    )
    grad0_norm = np.linalg.norm(-(X_tr.T @ (y_tr - probs0)) / nr_tr)
    print(f'{grad0_norm:.2f}')

    print(f'\nAdam最適化開始 (lr={LR}, max_epochs={N_EPOCHS}, alpha={ALPHA})...')
    d     = X_tr.shape[1]
    beta  = np.zeros(d)
    m     = np.zeros(d)
    v     = np.zeros(d)
    b1, b2, eps_adam = 0.9, 0.999, 1e-8
    t            = 0
    best_val     = np.inf
    best_beta    = beta.copy()
    no_improve   = 0
    CHECK_EVERY  = 10

    # epoch=0 のベースライン（beta=0）
    vl0, _ = neg_log_lik_and_grad(beta, X_va, y_va, gs_va, n_va, nr_va)
    print(f'  epoch=   0  val_loss={vl0:.4f}  (beta=0 ベースライン)')

    for epoch in range(1, N_EPOCHS + 1):
        loss, grad = neg_log_lik_and_grad(beta, X_tr, y_tr, gs_tr, n_tr, nr_tr)
        t += 1
        m = b1 * m + (1 - b1) * grad
        v = b2 * v + (1 - b2) * grad ** 2
        m_hat = m / (1 - b1 ** t)
        v_hat = v / (1 - b2 ** t)
        beta -= LR * m_hat / (np.sqrt(v_hat) + eps_adam)

        if epoch % CHECK_EVERY == 0:
            vl, _ = neg_log_lik_and_grad(beta, X_va, y_va, gs_va, n_va, nr_va)
            marker = ''
            if vl < best_val:
                best_val  = vl
                best_beta = beta.copy()
                no_improve = 0
                marker = ' ← best'
            else:
                no_improve += 1

            if epoch % 50 == 0 or marker:
                print(f'  epoch={epoch:4d}  tr_loss={loss:.4f}  val_loss={vl:.4f}{marker}')

            if no_improve >= PATIENCE // CHECK_EVERY:
                print(f'  早期停止: epoch={epoch} (patience={PATIENCE})')
                break

    print(f'最適化完了 (best val={best_val:.4f})')
    beta = best_beta

    # ── OOS評価 ───────────────────────────────────────────────────────────
    print('\n=== OOS評価 (2023+) ===')
    oos = df[df['日付_num'] >= 230101].copy()
    X_oos, y_oos, gs_oos, n_oos, nr_oos, *_ = prepare(
        oos, feat_cols, scaler=scaler,
        poly2=poly2, inter_scaler2=iscaler2, top_idx=top_idx,
        poly3=poly3, inter_scaler3=iscaler3, top_idx3=top_idx3)

    scores_oos = X_oos @ beta
    probs_oos  = segment_softmax(scores_oos, gs_oos, n_oos)

    oos = oos.sort_values('race_id').reset_index(drop=True)
    oos['model_prob']  = probs_oos
    oos['rank_model']  = oos.groupby('race_id')['model_prob'].rank(ascending=False, method='first')
    oos['pop_num']     = pd.to_numeric(oos['人気'], errors='coerce')
    oos['odds_num']    = pd.to_numeric(oos['単勝オッズ'], errors='coerce')
    oos['market_prob'] = 1.0 / oos['odds_num']
    oos['ev_score']    = oos['model_prob'] - oos['market_prob'] * 0.8
    oos['yr']          = oos['日付_num'] // 10000

    top1 = oos[oos['rank_model'] == 1]
    roi_table(top1, 'rank=1 全体')
    roi_table(top1[top1['pop_num'] >= 2], 'rank=1 × 2番人気以下')

    # EV フィルタ効果
    for thr in [0.0, 0.01, 0.02, 0.03]:
        ev_top1 = oos[(oos['rank_model'] == 1) & (oos['ev_score'] > thr)]
        if len(ev_top1) >= 100:
            won = ev_top1['着順_num'] == 1
            r   = (ev_top1.loc[won, 'odds_num'] * 100).sum() / (len(ev_top1) * 100) - 1
            print(f'rank=1 × EV>{thr:.2f}: {len(ev_top1)}件  ROI={r:+.3f}')

    # キャリブレーション
    print('\n=== キャリブレーション ===')
    oos['prob_bin'] = pd.qcut(oos['model_prob'], 10, labels=False, duplicates='drop')
    cal = oos.groupby('prob_bin').agg(
        pred_prob=('model_prob', 'mean'),
        actual_win=('着順_num', lambda x: (x == 1).mean()),
        n=('model_prob', 'count'),
    )
    for _, row in cal.iterrows():
        print(f'  pred={row.pred_prob:.3f}  actual={row.actual_win:.3f}  n={int(row.n):6d}')

    # ── 保存 ───────────────────────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    artifact = {
        'scaler':        scaler,
        'poly2':         poly2,
        'inter_scaler2': iscaler2,
        'top_idx':       top_idx,
        'poly3':         poly3,
        'inter_scaler3': iscaler3,
        'top_idx3':      top_idx3,
        'coef':          beta,
        'feat_cols':     feat_cols,
    }
    with open(model_path, 'wb') as f:
        pickle.dump(artifact, f)

    # 係数絶対値でtop20（feature名は省略でインデックスのみ）
    abs_c   = np.abs(beta)
    top20   = np.argsort(abs_c)[::-1][:20]
    n_orig  = len(feat_cols)
    top_feats_info = []
    for i in top20:
        name = feat_cols[i] if i < n_orig else f'inter_{i - n_orig}'
        top_feats_info.append((name, float(beta[i])))

    info = {
        'feat_cols':    feat_cols,
        'n_features':   len(feat_cols),
        'n_interactions': int(X_tr.shape[1]) - len(feat_cols),
        'total_dim':    int(X_tr.shape[1]),
        'top_k':        TOP_K,
        'alpha':        ALPHA,
        'top_features': top_feats_info,
        'train_range':  [130101, 201231],
    }
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f'\n保存完了: {model_path}  (特徴量次元={X_tr.shape[1]})')


if __name__ == '__main__':
    main()
