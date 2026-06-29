# coding: utf-8
"""
tune_lgb_hybrid.py - LightGBM チューニング × clogit ハイブリッド探索
objective: binary vs lambdarank
params: num_leaves x min_child_samples グリッド
各設定で val ROI 最良の alpha を探索し OOS で評価
"""
import sys, os, warnings, itertools
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from save_conditional_logit import (
    prepare, segment_softmax, BASE_DIR, DATA_FILE, LR, N_EPOCHS, PATIENCE
)
from save_v3 import add_computed_features, calc_roi
import lightgbm as lgb

CLOGIT_FEATS = [
    '近5走_クラス調整_平均着順', '近5走_タイム指数_max', '1走前_タイム指数', '前走着差タイム',
    '騎手コース_r100_勝率', '1走前_クラス調整着順', '調教師コース_r100_勝率',
    '1走前_RPCI', '1走前_上3F地点差', '斤量', '種牡馬_勝率',
    '間隔_長_flag', '1走前_脚質_num', '騎手変更', '馬番',
    '近3走_複勝率', '調教師_r200_複勝率',
    'ブリンカー変更', '2走前_クラス差', '4走前_クラス差', '1走前_馬場状態',
]

# LGB 追加特徴（レース定数含む）
LGB_EXTRA = [
    'dist_m', '今回_馬場_num', 'クラス_rank',
    '年齢', '馬体重', '馬体重増減',
    '近5走_平均4角位置', '近5走_RPCI平均',
    '同馬場_平均着順_近5走', '良馬場_平均着順_近5走',
    '母父馬_勝率', '産地_勝率', '種牡馬_ダ_勝率',
    '馬コース_r20_勝率', '馬距離_勝率',
    '近走連続入着数', '間隔', '距離変化_前走',
    '騎手距離_r100_勝率', '騎手調教師_r100_勝率',
    '2走前_タイム指数', '3走前_タイム指数',
    '近5走_タイム指数_min', '近3走_複勝率',
]


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
    for col in ['年齢', '馬体重', '馬体重増減', '間隔', 'クラス_rank', '今回_馬場_num']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df['y'] = (df['着順_num'] == 1).astype(int)
    return df


def _loss_grad(beta, X, y, gs, n, nr):
    probs = segment_softmax(X @ beta, gs, n)
    loss  = -np.sum(y * np.log(np.clip(probs, 1e-15, 1.0))) / nr
    grad  = -(X.T @ (y - probs)) / nr
    return loss, grad


def adam_fit(X_tr, y_tr, gs_tr, n_tr, nr_tr, X_va, y_va, gs_va, n_va, nr_va):
    d = X_tr.shape[1]
    beta, m, v = np.zeros(d), np.zeros(d), np.zeros(d)
    b1, b2, eps = 0.9, 0.999, 1e-8
    t, best_val, best_beta, no_imp = 0, np.inf, np.zeros(d), 0
    for epoch in range(1, N_EPOCHS + 1):
        _, grad = _loss_grad(beta, X_tr, y_tr, gs_tr, n_tr, nr_tr)
        t += 1
        m = b1*m + (1-b1)*grad; v = b2*v + (1-b2)*grad**2
        beta -= LR * (m/(1-b1**t)) / (np.sqrt(v/(1-b2**t)) + eps)
        if epoch % 10 == 0:
            vl, _ = _loss_grad(beta, X_va, y_va, gs_va, n_va, nr_va)
            if vl < best_val: best_val, best_beta, no_imp = vl, beta.copy(), 0
            else: no_imp += 1
            if no_imp >= PATIENCE // 10: break
    return best_beta


def get_clogit_probs(df_in, beta, scaler, valid_c):
    valid_p = [c for c in valid_c if c in df_in.columns]
    X_p, _, gs_p, n_p, *_ = prepare(df_in, valid_p, scaler=scaler,
                                      top_idx=None, top_idx3=None)
    sc = df_in.sort_values('race_id').reset_index(drop=True)
    sc['clogit_prob'] = segment_softmax(X_p @ beta, gs_p, n_p)
    return sc


def race_softmax_col(df_in, raw_col, out_col):
    df_in = df_in.copy()
    def _sm(g):
        e = np.exp(g - g.max())
        return e / e.sum()
    df_in[out_col] = df_in.groupby('race_id')[raw_col].transform(_sm)
    return df_in


def eval_hybrid(scored, alpha):
    sc = scored.copy()
    if alpha == 0.0 or 'lgb_prob' not in sc.columns:
        sc['score'] = sc['clogit_prob']
    else:
        sc['score'] = (1 - alpha) * sc['clogit_prob'] + alpha * sc['lgb_prob']
    sc['rank'] = sc.groupby('race_id')['score'].rank(ascending=False, method='first')
    top1 = sc[sc['rank'] == 1]
    roi, wins = calc_roi(top1)
    return roi, len(top1)


def best_alpha_on_val(val_sc):
    best_a, best_r = 0.0, -999
    for a in np.arange(0.0, 1.05, 0.05):
        r, _ = eval_hybrid(val_sc, a)
        if r > best_r:
            best_r, best_a = r, a
    return best_a, best_r


def comb_roi(oos_scored, alpha):
    r25, n25 = eval_hybrid(oos_scored['2025'], alpha)
    r26, n26 = eval_hybrid(oos_scored['2026'], alpha)
    return (r25*n25 + r26*n26) / (n25+n26), r25, r26


def main():
    df = load_segment()
    df_trn = df[(df['日付_num'] >= 130101) & (df['日付_num'] < 220101)]
    df_val = df[(df['日付_num'] >= 220101) & (df['日付_num'] <= 221231)]
    oos    = df[df['日付_num'] >= 230101]
    oos_parts = {
        '2325': oos[oos['日付_num'] < 250101],
        '2025': oos[(oos['日付_num'] >= 250101) & (oos['日付_num'] < 260101)],
        '2026': oos[oos['日付_num'] >= 260101],
    }

    # ===== clogit 訓練（共通） =====
    valid_c = [c for c in CLOGIT_FEATS
               if c in df_trn.columns and df_trn[c].isna().mean() <= 0.65]
    X_tr, y_tr, gs_tr, n_tr, nr_tr, scaler, *_ = prepare(
        df_trn, valid_c, top_idx=None, top_idx3=None, fit=True)
    X_va, y_va, gs_va, n_va, nr_va, *_ = prepare(
        df_val, valid_c, scaler=scaler, top_idx=None, top_idx3=None)
    beta = adam_fit(X_tr, y_tr, gs_tr, n_tr, nr_tr, X_va, y_va, gs_va, n_va, nr_va)

    val_sc   = get_clogit_probs(df_val.copy(), beta, scaler, valid_c)
    oos_sc   = {p: get_clogit_probs(df_p.copy(), beta, scaler, valid_c)
                for p, df_p in oos_parts.items() if len(df_p) > 0}

    # clogit ベースライン
    c_comb, c25, c26 = comb_roi(oos_sc, 0.0)
    print(f'clogit 21F ベース: 2025:{c25:+.4f}  2026:{c26:+.4f}  25+26:{c_comb:+.4f}')
    print()

    # ===== LGB 特徴セット =====
    feats_21   = [c for c in CLOGIT_FEATS
                  if c in df_trn.columns and df_trn[c].isna().mean() <= 0.65]
    feats_full = [c for c in CLOGIT_FEATS + LGB_EXTRA
                  if c in df_trn.columns and df_trn[c].isna().mean() <= 0.80
                  and c not in ['y']]
    feats_full = list(dict.fromkeys(feats_full))  # dedup

    # LambdaRank 用グループ
    def make_groups(df_in, feats):
        return df_in.groupby('race_id', sort=False).size().values

    # ===== グリッドサーチ =====
    GRID = {
        'objective':        ['binary', 'lambdarank'],
        'num_leaves':       [15, 31, 63],
        'min_child_samples':[50, 150, 300],
        'feature_set':      ['21F', 'full'],
    }

    results = []
    total = 2 * 3 * 3 * 2
    run = 0

    for obj, nl, mcs, fset in itertools.product(
        GRID['objective'], GRID['num_leaves'],
        GRID['min_child_samples'], GRID['feature_set']
    ):
        run += 1
        feats = feats_21 if fset == '21F' else feats_full
        tag = f'{obj[:3]}_L{nl}_M{mcs}_{fset}'

        X_lgb_tr = df_trn[feats].values
        X_lgb_va = df_val[feats].values
        y_lgb_tr = df_trn['y'].values
        y_lgb_va = df_val['y'].values

        params = dict(
            n_estimators=1000, learning_rate=0.05,
            num_leaves=nl, min_child_samples=mcs,
            subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, random_state=42, verbose=-1,
        )

        if obj == 'binary':
            model = lgb.LGBMClassifier(**params)
            model.fit(X_lgb_tr, y_lgb_tr,
                      eval_set=[(X_lgb_va, y_lgb_va)],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
            pred_fn = lambda X: model.predict_proba(X)[:, 1]
        else:  # lambdarank
            model = lgb.LGBMRanker(
                **{k: v for k, v in params.items()},
                objective='lambdarank',
                lambdarank_truncation_level=8,
            )
            g_tr = make_groups(df_trn.sort_values('race_id'), feats)
            g_va = make_groups(df_val.sort_values('race_id'), feats)
            model.fit(
                df_trn.sort_values('race_id')[feats].values,
                df_trn.sort_values('race_id')['y'].values,
                group=g_tr,
                eval_set=[(df_val.sort_values('race_id')[feats].values,
                           df_val.sort_values('race_id')['y'].values)],
                eval_group=[g_va],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
            pred_fn = lambda X: model.predict(X)

        # val に LGB スコアを付与
        val_lgb = val_sc.copy()
        val_lgb['lgb_raw'] = pred_fn(val_lgb[feats].values)
        val_lgb = race_softmax_col(val_lgb, 'lgb_raw', 'lgb_prob')

        # OOS に LGB スコアを付与
        oos_lgb = {}
        for p, sc in oos_sc.items():
            tmp = sc.copy()
            tmp['lgb_raw'] = pred_fn(tmp[feats].values)
            oos_lgb[p] = race_softmax_col(tmp, 'lgb_raw', 'lgb_prob')

        # val で最適 alpha
        best_a, best_val_r = best_alpha_on_val(val_lgb)

        # OOS
        h_comb, h25, h26 = comb_roi(oos_lgb, best_a)
        lgb_only, *_ = comb_roi(oos_lgb, 1.0)

        beat = '★' if h_comb > c_comb else ' '
        print(f'[{run:02d}/{total}] {beat} {tag:<28} '
              f'best_a={best_a:.2f}  val={best_val_r:+.4f}  '
              f'OOS 25+26={h_comb:+.4f}  (LGB単={lgb_only:+.4f})')

        results.append({
            'tag': tag, 'obj': obj, 'num_leaves': nl, 'mcs': mcs,
            'fset': fset, 'best_alpha': best_a,
            'val_roi': best_val_r, 'oos_comb': h_comb,
            'oos_25': h25, 'oos_26': h26, 'lgb_only': lgb_only,
            'n_iter': model.best_iteration_,
        })

    # ===== サマリ =====
    print(f'\n{"="*65}')
    print(f'clogit 21F ベース: 25+26={c_comb:+.4f}')
    df_res = pd.DataFrame(results).sort_values('oos_comb', ascending=False)
    print('\nOOS 25+26 上位10件:')
    print(f'  {"設定":<28} {"α":>5}  {"val":>8}  {"OOS25+26":>9}  {"2025":>8}  {"2026":>8}')
    for _, row in df_res.head(10).iterrows():
        beat = '★' if row['oos_comb'] > c_comb else ' '
        print(f'  {beat}{row["tag"]:<27} {row["best_alpha"]:>5.2f}  '
              f'{row["val_roi"]:>+8.4f}  {row["oos_comb"]:>+9.4f}  '
              f'{row["oos_25"]:>+8.4f}  {row["oos_26"]:>+8.4f}')

    best = df_res.iloc[0]
    print(f'\n最良設定: {best["tag"]}')
    print(f'  alpha={best["best_alpha"]:.2f}  val={best["val_roi"]:+.4f}  OOS={best["oos_comb"]:+.4f}')
    print(f'  2025:{best["oos_25"]:+.4f}  2026:{best["oos_26"]:+.4f}')
    if best['oos_comb'] > c_comb:
        print(f'  → clogit比 {best["oos_comb"]-c_comb:+.4f}pp 改善 !')
    else:
        print(f'  → clogit比 {best["oos_comb"]-c_comb:+.4f}pp（改善なし）')


if __name__ == '__main__':
    main()
