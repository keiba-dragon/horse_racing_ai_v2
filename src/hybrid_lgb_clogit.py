# coding: utf-8
"""
hybrid_lgb_clogit.py - clogit(21F) + LightGBM ハイブリッドスコアリング
clogit: レース内相対比較に強い、レース定数特徴は使えない
LightGBM: 非線形OK、レース定数特徴も使える（距離・馬場・クラス等）
combined_score = (1-alpha)*clogit_prob + alpha*lgb_prob
alpha を val(2022)で最適化してOOSで評価
"""
import sys, os, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from save_conditional_logit import (
    prepare, segment_softmax, BASE_DIR, DATA_FILE, LR, N_EPOCHS, PATIENCE
)
from save_v3 import add_computed_features, calc_roi

try:
    import lightgbm as lgb
    print(f'LightGBM version: {lgb.__version__}')
except ImportError:
    print('LightGBM not installed. Run: pip install lightgbm')
    sys.exit(1)

# clogit の21F特徴
CLOGIT_FEATS = [
    '近5走_クラス調整_平均着順', '近5走_タイム指数_max', '1走前_タイム指数', '前走着差タイム',
    '騎手コース_r100_勝率', '1走前_クラス調整着順', '調教師コース_r100_勝率',
    '1走前_RPCI', '1走前_上3F地点差', '斤量', '種牡馬_勝率',
    '間隔_長_flag', '1走前_脚質_num', '騎手変更', '馬番',
    '近3走_複勝率', '調教師_r200_複勝率',
    'ブリンカー変更', '2走前_クラス差', '4走前_クラス差', '1走前_馬場状態',
]

# LGB追加特徴（clogitが使えない/使いにくいもの）
LGB_EXTRA = [
    # レース定数系（clogitでは消えるが LGBMは使える）
    'dist_m',           # 距離
    '今回_馬場_num',     # 今日の馬場状態
    'クラス_rank',       # レースのクラス
    # 年齢・体重
    '年齢',
    '馬体重',
    '馬体重増減',
    # 脚質・ポジション
    '近5走_平均4角位置',
    '近走_先行率',       # 新規計算
    # 馬場適性
    '同馬場_平均着順_近5走',
    '良馬場_平均着順_近5走',
    # 血統追加
    '母父馬_勝率',
    '産地_勝率',
    '種牡馬_ダ_勝率',
    # 馬個人成績
    '馬コース_r20_勝率',
    '馬距離_勝率',
    # その他
    '近走連続入着数',
    '格上経験数_近5走',
    '間隔',
    '芝ダ転向',
    '距離変化_前走',
    '騎手距離_r100_勝率',
    '騎手調教師_r100_勝率',
    '2走前_タイム指数',
    '3走前_タイム指数',
    '近5走_タイム指数_min',
]

LGB_FEATS = CLOGIT_FEATS + [f for f in LGB_EXTRA if f not in CLOGIT_FEATS]


def load_segment():
    df = pd.read_parquet(DATA_FILE)
    df['日付_num'] = pd.to_numeric(df['日付'], errors='coerce')
    df['着順_num'] = pd.to_numeric(df['着順_num'], errors='coerce')
    df = df.dropna(subset=['日付_num', '着順_num'])
    df = df[df['着順_num'] < 99]
    df['race_id'] = (df['日付_num'].astype(int).astype(str) + '_' +
                     df['開催'].astype(str).str.strip() + '_' +
                     df['Ｒ'].astype(str).str.strip())
    df = df[df['開催'].notna()].copy()
    df['surface'] = (df['距離'].astype(str).str.strip()
                      .str.extract(r'^([芝ダ])')[0].fillna('不明'))
    dm = pd.to_numeric(df['距離'].astype(str).str.extract(r'(\d+)')[0], errors='coerce')
    df = df[(df['surface'] == 'ダ') & (dm > 1400)].copy()
    df['dist_m'] = dm[df.index]
    if 'クラス_rank' in df.columns:
        df = df[df['クラス_rank'] != 1.0].copy()
    df = add_computed_features(df)

    baba_map = {'良': 0, '稍重': 1, '重': 2, '不良': 3}
    for col in df.columns:
        if '馬場状態' in col:
            df[col] = df[col].map(baba_map)

    if '年齢' in df.columns:
        df['年齢'] = pd.to_numeric(df['年齢'], errors='coerce')
    if '馬体重' in df.columns:
        df['馬体重'] = pd.to_numeric(df['馬体重'], errors='coerce')
    if '馬体重増減' in df.columns:
        df['馬体重増減'] = pd.to_numeric(df['馬体重増減'], errors='coerce')
    if '間隔' in df.columns:
        df['間隔'] = pd.to_numeric(df['間隔'], errors='coerce')
    if 'クラス_rank' in df.columns:
        df['クラス_rank'] = pd.to_numeric(df['クラス_rank'], errors='coerce')
    if '今回_馬場_num' in df.columns:
        df['今回_馬場_num'] = pd.to_numeric(df['今回_馬場_num'], errors='coerce')

    # 近走先行率（近3走での逃/先割合）
    senten_cols = [f'{i}走前_脚質_num' for i in range(1, 4) if f'{i}走前_脚質_num' in df.columns]
    if senten_cols:
        flags = pd.DataFrame({
            c: (df[c] <= 1).astype(float).where(df[c].notna(), np.nan)
            for c in senten_cols
        })
        df['近走_先行率'] = flags.mean(axis=1)

    df['y'] = (df['着順_num'] == 1).astype(int)
    return df


def _loss_grad(beta, X, y, gs, n, nr):
    probs = segment_softmax(X @ beta, gs, n)
    res   = y - probs
    loss  = -np.sum(y * np.log(np.clip(probs, 1e-15, 1.0))) / nr
    grad  = -(X.T @ res) / nr
    return loss, grad


def adam_fit(X_tr, y_tr, gs_tr, n_tr, nr_tr, X_va, y_va, gs_va, n_va, nr_va):
    d = X_tr.shape[1]
    beta, m, v = np.zeros(d), np.zeros(d), np.zeros(d)
    b1, b2, eps = 0.9, 0.999, 1e-8
    t, best_val, best_beta, no_imp = 0, np.inf, np.zeros(d), 0
    for epoch in range(1, N_EPOCHS + 1):
        _, grad = _loss_grad(beta, X_tr, y_tr, gs_tr, n_tr, nr_tr)
        t += 1
        m = b1*m + (1-b1)*grad
        v = b2*v + (1-b2)*grad**2
        beta -= LR * (m/(1-b1**t)) / (np.sqrt(v/(1-b2**t)) + eps)
        if epoch % 10 == 0:
            vl, _ = _loss_grad(beta, X_va, y_va, gs_va, n_va, nr_va)
            if vl < best_val:
                best_val, best_beta, no_imp = vl, beta.copy(), 0
            else:
                no_imp += 1
            if no_imp >= PATIENCE // 10:
                break
    return best_beta, best_val


def get_clogit_probs(df_score, beta, scaler, valid_feats):
    """データフレームに clogit 確率列を付与して返す"""
    valid_p = [c for c in valid_feats if c in df_score.columns]
    X_p, _, gs_p, n_p, *_ = prepare(df_score, valid_p, scaler=scaler,
                                      top_idx=None, top_idx3=None)
    scored = df_score.sort_values('race_id').reset_index(drop=True)
    scored['clogit_prob'] = segment_softmax(X_p @ beta, gs_p, n_p)
    return scored


def eval_combined(scored, alpha):
    """alpha で混合してROI計算"""
    sc = scored.copy()
    sc['combined'] = (1 - alpha) * sc['clogit_prob'] + alpha * sc['lgb_prob']
    sc['rank'] = sc.groupby('race_id')['combined'].rank(ascending=False, method='first')
    top1 = sc[sc['rank'] == 1]
    roi, wins = calc_roi(top1)
    return roi, len(top1), wins


def main():
    df = load_segment()
    df_trn = df[(df['日付_num'] >= 130101) & (df['日付_num'] < 220101)]
    df_val = df[(df['日付_num'] >= 220101) & (df['日付_num'] <= 221231)]
    oos    = df[df['日付_num'] >= 230101]
    oos_parts = {
        '2324': oos[oos['日付_num'] < 250101],
        '2025': oos[(oos['日付_num'] >= 250101) & (oos['日付_num'] < 260101)],
        '2026': oos[oos['日付_num'] >= 260101],
    }

    # ===== 1. clogit 訓練 =====
    print('=== clogit (21F) 訓練 ===')
    valid_c = [c for c in CLOGIT_FEATS
               if c in df_trn.columns and df_trn[c].isna().mean() <= 0.65]
    X_tr, y_tr, gs_tr, n_tr, nr_tr, scaler, *_ = prepare(
        df_trn, valid_c, top_idx=None, top_idx3=None, fit=True)
    X_va, y_va, gs_va, n_va, nr_va, *_ = prepare(
        df_val, valid_c, scaler=scaler, top_idx=None, top_idx3=None)
    beta, val_nll = adam_fit(X_tr, y_tr, gs_tr, n_tr, nr_tr,
                              X_va, y_va, gs_va, n_va, nr_va)
    print(f'valNLL={val_nll:.5f}  有効特徴:{len(valid_c)}本')

    # clogit確率を各データに付与
    val_scored  = get_clogit_probs(df_val.copy(), beta, scaler, valid_c)
    oos_scored  = {p: get_clogit_probs(df_p.copy(), beta, scaler, valid_c)
                   for p, df_p in oos_parts.items() if len(df_p) > 0}

    # clogit単体ROI
    print('\nclogit 単体 OOS:')
    for p, sc in oos_scored.items():
        sc['rank'] = sc.groupby('race_id')['clogit_prob'].rank(ascending=False, method='first')
        top1 = sc[sc['rank'] == 1]
        roi, wins = calc_roi(top1)
        print(f'  {p}: {len(top1)}R  ROI={roi:+.4f}  勝率={wins/len(top1):.1%}')

    r25_c, n25_c, _ = oos_scored['2025'].groupby('race_id')['clogit_prob'].max(), 0, 0
    # 25+26合算
    top1_25 = oos_scored['2025'][oos_scored['2025'].groupby('race_id')['clogit_prob'].transform('rank', ascending=False, method='first') == 1]
    top1_26 = oos_scored['2026'][oos_scored['2026'].groupby('race_id')['clogit_prob'].transform('rank', ascending=False, method='first') == 1]
    r25_c, _ = calc_roi(top1_25); r26_c, _ = calc_roi(top1_26)
    n25_c = len(top1_25); n26_c = len(top1_26)
    comb_c = (r25_c*n25_c + r26_c*n26_c) / (n25_c + n26_c)
    print(f'  25+26合算: {comb_c:+.4f}')

    # ===== 2. LightGBM 訓練 =====
    print('\n=== LightGBM 訓練 ===')
    valid_l = [c for c in LGB_FEATS
               if c in df_trn.columns and df_trn[c].isna().mean() <= 0.80]
    print(f'LGB特徴数: {len(valid_l)}本')

    X_lgb_tr = df_trn[valid_l].values
    y_lgb_tr = df_trn['y'].values
    X_lgb_va = df_val[valid_l].values
    y_lgb_va = df_val['y'].values

    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1,
    )
    model.fit(
        X_lgb_tr, y_lgb_tr,
        eval_set=[(X_lgb_va, y_lgb_va)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
    )
    print(f'best iteration: {model.best_iteration_}')

    # LGB確率 → レース内softmax
    def add_lgb_prob(scored_df):
        X_ = scored_df[valid_l].values
        raw = model.predict_proba(X_)[:, 1]
        scored_df = scored_df.copy()
        scored_df['lgb_raw'] = raw
        # レース内 softmax
        def race_softmax(grp):
            e = np.exp(grp - grp.max())
            return e / e.sum()
        scored_df['lgb_prob'] = scored_df.groupby('race_id')['lgb_raw'].transform(race_softmax)
        return scored_df

    val_scored  = add_lgb_prob(val_scored)
    for p in oos_scored:
        oos_scored[p] = add_lgb_prob(oos_scored[p])

    # LGB単体ROI
    print('\nLightGBM 単体 OOS (alpha=1.0):')
    for p, sc in oos_scored.items():
        roi, n, wins = eval_combined(sc, alpha=1.0)
        print(f'  {p}: {n}R  ROI={roi:+.4f}  勝率={wins/n:.1%}')
    r25_l, n25_l, _ = eval_combined(oos_scored['2025'], 1.0)
    r26_l, n26_l, _ = eval_combined(oos_scored['2026'], 1.0)
    comb_l = (r25_l*n25_l + r26_l*n26_l) / (n25_l + n26_l)
    print(f'  25+26合算: {comb_l:+.4f}')

    # ===== 3. alpha チューニング (val=2022) =====
    print('\n=== alpha チューニング (val=2022) ===')
    best_alpha, best_val_roi = 0.0, -999
    for alpha in np.arange(0.0, 1.05, 0.05):
        roi, n, _ = eval_combined(val_scored, alpha)
        if roi > best_val_roi:
            best_val_roi, best_alpha = roi, alpha
        print(f'  alpha={alpha:.2f}  val_ROI={roi:+.4f}')

    print(f'\n最適 alpha={best_alpha:.2f}  val_ROI={best_val_roi:+.4f}')

    # ===== 4. OOS評価 =====
    print('\n=== OOS評価（最適alpha）===')
    for p, sc in oos_scored.items():
        roi, n, wins = eval_combined(sc, best_alpha)
        print(f'  {p}: {n}R  ROI={roi:+.4f}  勝率={wins/n:.1%}')

    r25_h, n25_h, _ = eval_combined(oos_scored['2025'], best_alpha)
    r26_h, n26_h, _ = eval_combined(oos_scored['2026'], best_alpha)
    comb_h = (r25_h*n25_h + r26_h*n26_h) / (n25_h + n26_h)
    print(f'  25+26合算: {comb_h:+.4f}')

    # ===== サマリ =====
    print(f'\n{"="*55}')
    print(f'{"モデル":<20} {"2025":>8} {"2026":>8} {"25+26":>8}')
    print(f'{"clogit 21F":<20} {r25_c:>+8.2%} {r26_c:>+8.2%} {comb_c:>+8.2%}')
    print(f'{"LightGBM 単体":<20} {r25_l:>+8.2%} {r26_l:>+8.2%} {comb_l:>+8.2%}')
    print(f'{"ハイブリッド":<20} {r25_h:>+8.2%} {r26_h:>+8.2%} {comb_h:>+8.2%}')
    print(f'(alpha={best_alpha:.2f}: clogit×{1-best_alpha:.0%} + LGB×{best_alpha:.0%})')

    # LGB重要特徴
    print('\nLGB 重要特徴 Top15:')
    imp = pd.Series(model.feature_importances_, index=valid_l).sort_values(ascending=False)
    for feat, score in imp.head(15).items():
        print(f'  {feat:<35} {score:>6.0f}')


if __name__ == '__main__':
    main()
