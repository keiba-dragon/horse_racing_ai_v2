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
    '休養日数', '斤量変化', '距離変化_m', '近走着順トレンド',
    # タイム × 内容 交互作用
    '前走_タイム×PCI', '前走_上り×RPCI', '前走_タイム×4角', '前走_4角×上り',
    'タイム傾き×距離変化',
    '近5_タイム×RPCI', '近5_上り×RPCI', '近5_タイム×4角', '近5_4角×上り',
    # B: 前走着差タイム × 距離変化
    '前走着差×距離変化', '前走着差×クラス補正',
    # C: 年齢トレンド
    '年齢_peak_diff', '年齢_peak_abs',
]

# v2 lean: greedy feature search で収束した21特徴量（exp_v302_best）
V2_LEAN_FEATURES = [
    # 前走スピード（絶対値・上がり・着差の3角度）
    '1走前_タイム指数',
    '1走前_上り3F',
    '前走着差タイム',
    # ペース文脈
    '1走前_RPCI',
    # 中期実力（平均＋ポテンシャル）
    '近5走_タイム指数平均',
    '近5走_タイム指数_max',
    # 調子トレンド
    'タイム指数_近3走_slope',
    # 着順（クラス補正版）
    '1走前_クラス調整着順',
    '近5走_クラス調整_平均着順',
    # 枠
    '馬番',
    'コース枠_r200_勝率',
    # 騎手
    '騎手コース_r100_勝率',
    '騎手変更',
    # 調教師
    '調教師コース_r100_勝率',
    # コース×脚質
    'コース脚質_r200_勝率',
    # 前走脚質
    '1走前_脚質_num',
    # 血統（greedy searchで追加）
    '種牡馬_勝率',
    '種牡馬_ダ_勝率',
    # 負担重量（greedy searchで追加）
    '斤量',
    # 芝ダ転向シグナル（greedy searchで追加）
    '芝ダ転向',
    # 前走間隔（greedy searchで追加）
    '間隔',
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

    # ── 6. タイム × 内容 交互作用特徴量 ──────────────────────────────────────
    # 前走
    t1 = _col(df, '1走前_タイム指数')
    p1 = _col(df, '1走前_PCI')
    r1 = _col(df, '1走前_RPCI')
    u1 = _col(df, '1走前_上り3F')
    k1 = _col(df, '1走前_4角')
    df['前走_タイム×PCI']  = t1 * p1
    df['前走_上り×RPCI']   = u1 * r1
    df['前走_タイム×4角']  = t1 * k1
    df['前走_4角×上り']    = k1 * u1
    # トレンド × 距離変化
    df['タイム傾き×距離変化'] = _col(df, 'タイム指数_近3走_slope') * _col(df, '距離変化_m')
    # 近5走版
    t5 = _col(df, '近5走_タイム指数平均')
    r5 = _col(df, '近5走_RPCI平均')
    u5 = _col(df, '近5走_上り3F平均')
    k5 = _col(df, '近5走_平均4角位置')
    df['近5_タイム×RPCI']  = t5 * r5
    df['近5_上り×RPCI']    = u5 * r5
    df['近5_タイム×4角']   = t5 * k5
    df['近5_4角×上り']     = k5 * u5

    # ── 7. B: 前走着差タイム × 距離変化 / クラス変化 ────────────────────────────
    atd = _col(df, '前走着差タイム')
    df['前走着差×距離変化']  = atd * _col(df, '距離変化_m')
    df['前走着差×クラス補正'] = _col(df, '前走着差タイム_クラス補正') * _col(df, '距離変化_m')

    # ── 8. C: 年齢トレンド（ピーク4歳からの距離） ───────────────────────────────
    age = pd.to_numeric(df.get('年齢', np.nan), errors='coerce')
    df['年齢_peak_diff'] = age - 4.0          # 正=ピーク超え, 負=ピーク前
    df['年齢_peak_abs']  = (age - 4.0).abs()  # ピークからの距離（大きいほど不利）

    # ── 9. 間隔フラグ（パーケットに入っていないため実行時に計算）──────────────────
    if '間隔_長_flag' not in df.columns or df['間隔_長_flag'].isna().all():
        interval = pd.to_numeric(df.get('間隔', pd.Series(np.nan, index=df.index)), errors='coerce')
        df['間隔_長_flag'] = (interval >= 60).astype(float)

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


def neg_log_lik_fukusho_and_grad(beta, X, y_top3, group_starts, n, n_races):
    """
    複勝版損失・勾配。
    P(複勝) = sum_{i in top3} softmax_i
    L = -mean_race log P(複勝) + L2
    """
    scores = X @ beta
    probs  = segment_softmax(scores, group_starts, n)
    group_sizes = np.diff(np.append(group_starts, n))

    # レースごとの P(top3) = sum P_i for i in top3
    top3_sum = np.add.reduceat(probs * y_top3, group_starts)
    log_lik  = np.sum(np.log(np.clip(top3_sum, 1e-15, 1.0)))

    # 勾配: residual_i = y_i * P_i / P_top3(race) - P_i
    top3_per_horse = np.repeat(top3_sum, group_sizes)
    residuals = y_top3 * probs / np.clip(top3_per_horse, 1e-15, 1.0) - probs

    loss = (-log_lik + ALPHA * np.sum(beta ** 2)) / n_races
    grad = -(X.T @ residuals) / n_races + 2 * ALPHA * beta / n_races
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
    ap.add_argument('--feat-include', default=None,
                    help='カンマ区切りキーワード。いずれかを含む特徴量だけ使用（実験用）')
    ap.add_argument('--lean', action='store_true',
                    help='v2 lean モード: 前走タイム・着順・枠・騎手・コース適正・脚質のみ')
    ap.add_argument('--min-career', type=int, default=0,
                    help='キャリア最小戦数フィルタ（例: 5 → 5走前_着順_numが非NaNの馬のみ）')
    ap.add_argument('--dist-split', action='store_true',
                    help='距離帯分割: 短距離(≤1400m)/中距離(1401-2000m)/長距離(>2000m) × 芝/ダ = 6モデル')
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

    if args.lean:
        print('[lean モード] 前走タイム・着順・枠・騎手・コース適正・脚質のみ使用')
        feat_cols = [c for c in V2_LEAN_FEATURES if c in df.columns]
        missing_lean = [c for c in V2_LEAN_FEATURES if c not in df.columns]
        if missing_lean:
            print(f'  [警告] lean特徴量に欠損列: {missing_lean}')
        print(f'  lean特徴量: {len(feat_cols)}列 → {feat_cols}')
        top_idx  = None
        top_idx3 = None
    else:
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
        if args.feat_include:
            keywords = [k.strip() for k in args.feat_include.split(',')]
            feat_cols = [c for c in feat_cols if any(k in c for k in keywords)]
            print(f'--feat-include フィルタ ({keywords}): {len(feat_cols)}列')
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
            feat_set = set(feat_cols)
            ranked = sorted(
                [(lgbm_feats[i], importances[i]) for i in range(len(lgbm_feats)) if lgbm_feats[i] in feat_set],
                key=lambda x: -x[1]
            )
            top_names  = [f for f, _ in ranked[:TOP_K]]
            top_idx    = np.array([feat_cols.index(f) for f in top_names if f in feat_cols])
            n_inter2 = len(top_idx) * (len(top_idx) - 1) // 2
            print(f'2-way対象 Top-{len(top_idx)}: {top_names[:5]}... ({n_inter2}件)')
            top_idx3 = None
        else:
            top_idx3 = None
            print('lambdarank_pace モデルが見つからない → 交互作用なし')

    # surface列付与・不明行除外（障害は距離先頭が '障' のため自動除外）
    df['surface'] = df['距離'].astype(str).str.strip().str.extract(r'^([芝ダ])')[0].fillna('不明')
    df = df[df['surface'].isin(['芝', 'ダ'])].copy()
    # 新馬（クラス_rank=1）除外
    if 'クラス_rank' in df.columns:
        before_shinma = len(df)
        df = df[df['クラス_rank'] != 1.0].copy()
        print(f'  新馬除外: {before_shinma - len(df):,}行')
    # キャリア戦数フィルタ
    if args.min_career > 0:
        career_col = f'{args.min_career}走前_着順_num'
        if career_col in df.columns:
            before_c = len(df)
            df = df[df[career_col].notna()].copy()
            print(f'  キャリア{args.min_career}戦以上フィルタ: {before_c - len(df):,}行除外 → {len(df):,}行')
        else:
            print(f'  [警告] {career_col} が見つからないのでフィルタをスキップ')
    print(f'有効データ: {len(df):,}行 (芝: {(df["surface"]=="芝").sum():,}行, ダ: {(df["surface"]=="ダ").sum():,}行)')

    if args.dist_split:
        df['dist_m'] = pd.to_numeric(
            df['距離'].astype(str).str.extract(r'(\d+)')[0], errors='coerce')
        # 芝: 短距離(≤1400) / 中距離(1401-2000) / 長距離(>2000)
        # ダ: 短距離(≤1400) / 中長距離(>1400) ← 長距離はサンプル少なく中と統合
        dm = df['dist_m']
        shi = df['surface'] == '芝'
        da  = df['surface'] == 'ダ'
        df['dist_band'] = ''
        df.loc[shi & (dm <= 1400),              'dist_band'] = '短距離'
        df.loc[shi & (dm > 1400) & (dm <= 2000),'dist_band'] = '中距離'
        df.loc[shi & (dm > 2000),               'dist_band'] = '長距離'
        df.loc[da  & (dm <= 1400),              'dist_band'] = '短距離'
        df.loc[da  & (dm > 1400),               'dist_band'] = '中長距離'
        print(f'距離帯分布:\n{df.groupby(["surface", "dist_band"], observed=True).size().to_string()}')
        segments = [('芝', '短距離'), ('芝', '中距離'), ('芝', '長距離'),
                    ('ダ', '短距離'), ('ダ', '中長距離')]
    else:
        segments = [(s, None) for s in ['芝', 'ダ']]

    all_artifacts  = {}
    all_top_feats  = {}
    last_X_tr_dim  = None
    all_oos_top1   = []

    for surf, dist_band in segments:
        seg_key   = f'{surf}_{dist_band}' if dist_band is not None else surf
        print(f'\n{"="*60}')
        print(f'  {seg_key} モデル学習')
        print(f'{"="*60}')
        if dist_band is not None:
            df_s = df[(df['surface'] == surf) & (df['dist_band'] == dist_band)].copy()
        else:
            df_s = df[df['surface'] == surf].copy()

        trn = df_s[(df_s['日付_num'] >= 130101) & (df_s['日付_num'] <  220101)]
        val = df_s[(df_s['日付_num'] >= 220101) & (df_s['日付_num'] <= 221231)]
        print(f'学習: {len(trn):,}行 / valid: {len(val):,}行')

        X_tr, y_tr, gs_tr, n_tr, nr_tr, scaler, poly2, iscaler2, poly3, iscaler3 = prepare(
            trn, feat_cols, top_idx=top_idx, top_idx3=top_idx3, fit=True)
        X_va, y_va, gs_va, n_va, nr_va, *_ = prepare(
            val, feat_cols, scaler=scaler,
            poly2=poly2, inter_scaler2=iscaler2, top_idx=top_idx,
            poly3=poly3, inter_scaler3=iscaler3, top_idx3=top_idx3)

        print(f'拡張後の特徴量次元: {X_tr.shape[1]}')
        print(f'勾配ノルム確認（beta=0）: ', end='')
        probs0 = np.repeat(
            1.0 / np.diff(np.append(gs_tr, n_tr)),
            np.diff(np.append(gs_tr, n_tr))
        )
        grad0_norm = np.linalg.norm(-(X_tr.T @ (y_tr - probs0)) / nr_tr)
        print(f'{grad0_norm:.2f}')

        print(f'\nAdam最適化開始 (lr={LR}, max_epochs={N_EPOCHS}, alpha={ALPHA})...')
        d    = X_tr.shape[1]
        beta = np.zeros(d)
        m    = np.zeros(d)
        v    = np.zeros(d)
        b1, b2, eps_adam = 0.9, 0.999, 1e-8
        t           = 0
        best_val    = np.inf
        best_beta   = beta.copy()
        no_improve  = 0
        CHECK_EVERY = 10

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
        last_X_tr_dim = X_tr.shape[1]

        # ── OOS評価 ───────────────────────────────────────────────────────
        print(f'\n=== OOS評価 ({seg_key}, 2023+) ===')
        oos_s = df_s[df_s['日付_num'] >= 230101].copy()
        X_oos, y_oos, gs_oos, n_oos, nr_oos, *_ = prepare(
            oos_s, feat_cols, scaler=scaler,
            poly2=poly2, inter_scaler2=iscaler2, top_idx=top_idx,
            poly3=poly3, inter_scaler3=iscaler3, top_idx3=top_idx3)

        scores_oos = X_oos @ beta
        probs_oos  = segment_softmax(scores_oos, gs_oos, n_oos)

        oos_s = oos_s.sort_values('race_id').reset_index(drop=True)
        oos_s['model_prob'] = probs_oos
        oos_s['rank_model'] = oos_s.groupby('race_id')['model_prob'].rank(ascending=False, method='first')
        oos_s['pop_num']    = pd.to_numeric(oos_s['人気'], errors='coerce')
        oos_s['odds_num']   = pd.to_numeric(oos_s['単勝オッズ'], errors='coerce')
        oos_s['ev_score']   = oos_s['model_prob'] - (1.0 / oos_s['odds_num']) * 0.8
        oos_s['yr']         = oos_s['日付_num'] // 10000

        top1 = oos_s[oos_s['rank_model'] == 1]
        all_oos_top1.append(top1)
        roi_table(top1, f'{seg_key} rank=1 全体')
        roi_table(top1[top1['pop_num'] >= 2], f'{seg_key} rank=1 × 2番人気以下')

        for thr in [0.0, 0.01, 0.02, 0.03]:
            ev_top1 = oos_s[(oos_s['rank_model'] == 1) & (oos_s['ev_score'] > thr)]
            if len(ev_top1) >= 100:
                won = ev_top1['着順_num'] == 1
                r   = (ev_top1.loc[won, 'odds_num'] * 100).sum() / (len(ev_top1) * 100) - 1
                print(f'{seg_key} rank=1 × EV>{thr:.2f}: {len(ev_top1)}件  ROI={r:+.3f}')

        # キャリブレーション
        print(f'\n=== キャリブレーション ({seg_key}) ===')
        oos_s['prob_bin'] = pd.qcut(oos_s['model_prob'], 10, labels=False, duplicates='drop')
        cal = oos_s.groupby('prob_bin').agg(
            pred_prob=('model_prob', 'mean'),
            actual_win=('着順_num', lambda x: (x == 1).mean()),
            n=('model_prob', 'count'),
        )
        for _, row in cal.iterrows():
            print(f'  pred={row.pred_prob:.3f}  actual={row.actual_win:.3f}  n={int(row.n):6d}')

        # artifact構築
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
        all_artifacts[seg_key] = artifact

        abs_c = np.abs(beta)
        top20 = np.argsort(abs_c)[::-1][:20]
        n_orig = len(feat_cols)
        top_feats_info = []
        for i in top20:
            name = feat_cols[i] if i < n_orig else f'inter_{i - n_orig}'
            top_feats_info.append((name, float(beta[i])))
        all_top_feats[seg_key] = top_feats_info
        print(f'{seg_key} Top-5特徴量: {top_feats_info[:5]}')

    # ── 全セグメント合計ROI ────────────────────────────────────────────────
    if len(all_oos_top1) > 1:
        combined = pd.concat(all_oos_top1, ignore_index=True)
        roi_table(combined, '全セグメント合計 rank=1')

    # ── 保存 ───────────────────────────────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)
    pkg = {
        'artifacts': all_artifacts,
        'feat_cols': feat_cols,
    }
    with open(model_path, 'wb') as f:
        pickle.dump(pkg, f)

    info = {
        'feat_cols':     feat_cols,
        'n_features':    len(feat_cols),
        'total_dim':     last_X_tr_dim,
        'top_k':         TOP_K,
        'alpha':         ALPHA,
        'top_features':  all_top_feats,
        'train_range':   [130101, 221231],
        'surface_split': True,
        'dist_split':    args.dist_split,
    }
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f'\n保存完了: {model_path}  (surface分割モデル, 特徴量次元={last_X_tr_dim})')


if __name__ == '__main__':
    main()
