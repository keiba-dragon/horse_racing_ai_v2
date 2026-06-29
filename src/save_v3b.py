# coding: utf-8
"""
save_v3b.py - v3の改良版
  ・LGB特徴量拡充（+近3走_平均着順, 2走前/3走前_タイム指数, ダ中長に展開フィット等）
  ・セグメント別最適ブレンド比
      芝_短距離:  clogit=0.9
      芝_中距離:  clogit=0.9
      芝_長距離:  clogit=0.9
      ダ_短距離:  clogit=1.0  (純clogit = v303と同等)
      ダ_中長距離: clogit=0.2  (LGB重視)
出力: models/v3b/{seg_key}/clogit.pkl + lgbm.pkl
"""
import sys, os, pickle, json, time
import numpy as np
import pandas as pd
import lightgbm as lgb

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from save_conditional_logit import (
    prepare, segment_softmax, get_group_starts,
    BASE_DIR, DATA_FILE, LR, N_EPOCHS, PATIENCE,
)
from sklearn.isotonic import IsotonicRegression
from save_v3 import (
    add_computed_features, race_normalize, calc_roi,
    CLOGIT_FEATS, _loss_grad, adam_opt,
)

OUT_DIR = os.path.join(BASE_DIR, 'models', 'v3b')
os.makedirs(OUT_DIR, exist_ok=True)

SEGMENTS = [('芝', '短距離'), ('芝', '中距離'), ('芝', '長距離'),
            ('ダ', '短距離'), ('ダ', '中長距離')]

# セグメント別ブレンド比（clogit側の重み）
BLEND_W = {
    '芝_短距離':  0.9,
    '芝_中距離':  0.9,
    '芝_長距離':  0.9,
    'ダ_短距離':  1.0,
    'ダ_中長距離': 0.2,
}

# ── LightGBM 特徴量 ──────────────────────────────────────────────────────────
# v3から追加: 近3走_平均着順, 2走前_タイム指数, 3走前_タイム指数
LGBM_BASE_FEATS = [
    # スピード・タイム系
    '1走前_タイム指数',
    '2走前_タイム指数',
    '3走前_タイム指数',
    '近5走_タイム指数平均',
    '近5走_タイム指数_max',
    '前走着差タイム',
    '1走前_上り3F',
    # 着順・クラス
    '1走前_クラス調整着順',
    '近5走_クラス調整_平均着順',
    '近3走_平均着順',
    '近3走_複勝率',
    '近10走_勝率',
    # ペース・脚質
    '1走前_RPCI',
    '1走前_脚質_num',
    # 騎手・調教師
    '騎手コース_r100_勝率',
    '調教師コース_r100_勝率',
    # コース適性
    'コース脚質_r200_勝率',
    'コース枠_r200_勝率',
    # 血統
    '種牡馬_勝率',
    '種牡馬_ダ_勝率',
    # 負担・枠・体重
    '斤量',
    '馬番',
    '間隔',
    '馬体重',
    '馬体重増減',
    # クラス・出走頭数
    'クラス_rank',
    '出走頭数',
    # 計算特徴量
    '間隔_長_flag',
    '間隔_短_flag',
    '血統_ダ優位度',
]

LGBM_SEG_EXTRA = {
    '芝_短距離':  ['芝ダ転向', '1走前_3角', 'タイム指数_近3走_slope', '騎手変更'],
    '芝_中距離':  ['タイム指数_近3走_slope', '騎手変更', '芝ダ転向'],
    '芝_長距離':  ['タイム指数_近3走_slope', '騎手変更', '1走前_上3F地点差'],
    'ダ_短距離':  ['展開フィット_v2', '乗替り_近走不振', '騎手変更',
                  'タイム指数_近3走_slope', '芝ダ転向'],
    'ダ_中長距離': ['展開フィット_v2', '乗替り_近走不振', '騎手変更',
                  'タイム指数_近3走_slope', '1走前_上3F地点差'],
}


def _lgbm_feats(seg_key):
    seen, out = set(), list(LGBM_BASE_FEATS)
    seen.update(out)
    for f in LGBM_SEG_EXTRA.get(seg_key, []):
        if f not in seen:
            out.append(f)
            seen.add(f)
    return out


LGBM_PARAMS = {
    'objective':         'binary',
    'metric':            'binary_logloss',
    'n_estimators':      2000,
    'learning_rate':     0.03,
    'num_leaves':        31,
    'min_child_samples': 50,
    'subsample':         0.8,
    'colsample_bytree':  0.8,
    'reg_alpha':         0.1,
    'reg_lambda':        1.0,
    'random_state':      42,
    'n_jobs':            -1,
    'verbose':           -1,
}


def load_data():
    print(f'読み込み: {DATA_FILE}')
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
    df = df[df['surface'].isin(['芝', 'ダ'])].copy()
    if 'クラス_rank' in df.columns:
        df = df[df['クラス_rank'] != 1.0].copy()
    dm = pd.to_numeric(df['距離'].astype(str).str.extract(r'(\d+)')[0], errors='coerce')
    df['dist_m'] = dm
    shi, da = df['surface'] == '芝', df['surface'] == 'ダ'
    df['dist_band'] = ''
    df.loc[shi & (dm <= 1400),               'dist_band'] = '短距離'
    df.loc[shi & (dm > 1400) & (dm <= 2000), 'dist_band'] = '中距離'
    df.loc[shi & (dm > 2000),                'dist_band'] = '長距離'
    df.loc[da  & (dm <= 1400),               'dist_band'] = '短距離'
    df.loc[da  & (dm > 1400),                'dist_band'] = '中長距離'
    df = add_computed_features(df)
    print(f'有効行: {len(df):,}')
    return df


def score_segment(df_eval, beta, scaler, iso, clogit_feats,
                  lgbm_model, lgbm_valid_feats, clogit_w):
    valid_c = [c for c in clogit_feats if c in df_eval.columns]
    X_c, _, gs, n_total, *_ = prepare(df_eval, valid_c, scaler=scaler,
                                       top_idx=None, top_idx3=None)
    raw_p    = segment_softmax(X_c @ beta, gs, n_total)
    calib_p  = iso.predict(raw_p)
    clogit_p = race_normalize(calib_p, gs, n_total)

    lgbm_w = 1.0 - clogit_w
    if lgbm_w > 0:
        df_s = df_eval.sort_values('race_id').reset_index(drop=True)
        valid_l = [c for c in lgbm_valid_feats if c in df_s.columns]
        X_l = df_s[valid_l].astype(float).fillna(0).values
        lgbm_raw = lgbm_model.predict_proba(X_l)[:, 1]
        lgbm_p   = race_normalize(lgbm_raw, gs, n_total)
        final_p  = clogit_w * clogit_p + lgbm_w * lgbm_p
    else:
        df_s    = df_eval.sort_values('race_id').reset_index(drop=True)
        lgbm_p  = np.zeros(n_total)
        final_p = clogit_p

    df_s = df_eval.sort_values('race_id').reset_index(drop=True)
    df_s['clogit_prob'] = clogit_p
    df_s['lgbm_prob']   = lgbm_p
    df_s['final_prob']  = final_p
    return df_s


def main():
    t_start = time.time()
    df = load_data()

    all_oos = {k: [] for k in ['2324', '2025', '2026']}

    for surf, dist_band in SEGMENTS:
        seg_key  = f'{surf}_{dist_band}'
        seg_dir  = os.path.join(OUT_DIR, seg_key)
        os.makedirs(seg_dir, exist_ok=True)
        clogit_w = BLEND_W[seg_key]

        df_s = df[(df['surface'] == surf) & (df['dist_band'] == dist_band)].copy()
        trn  = df_s[(df_s['日付_num'] >= 130101) & (df_s['日付_num'] < 220101)]
        val  = df_s[(df_s['日付_num'] >= 220101) & (df_s['日付_num'] <= 221231)]
        oos  = df_s[df_s['日付_num'] >= 230101].copy()
        parts = {
            '2324': oos[oos['日付_num'] < 250101],
            '2025': oos[(oos['日付_num'] >= 250101) & (oos['日付_num'] < 260101)],
            '2026': oos[oos['日付_num'] >= 260101],
        }

        if len(trn) < 300 or len(val) < 30:
            print(f'[{seg_key}] データ不足スキップ')
            continue

        cf_all      = CLOGIT_FEATS[seg_key]
        lf_all      = _lgbm_feats(seg_key)
        clogit_valid = [c for c in cf_all if c in df_s.columns and df_s[c].isna().mean() <= 0.65]
        lgbm_valid   = [c for c in lf_all if c in df_s.columns and df_s[c].isna().mean() <= 0.65]

        print(f'\n[{seg_key}] clogit_w={clogit_w}  '
              f'clogit:{len(clogit_valid)}特徴  lgbm:{len(lgbm_valid)}特徴  '
              f'trn:{len(trn):,} val:{len(val):,}')

        t1 = time.time()

        # clogit 訓練
        X_tr, y_tr, gs_tr, n_tr, nr_tr, scaler, *_ = prepare(
            trn, clogit_valid, top_idx=None, top_idx3=None, fit=True)
        X_va, y_va, gs_va, n_va, nr_va, *_ = prepare(
            val, clogit_valid, scaler=scaler, top_idx=None, top_idx3=None)
        beta = adam_opt(X_tr, y_tr, gs_tr, n_tr, nr_tr,
                        X_va, y_va, gs_va, n_va, nr_va)
        val_probs = segment_softmax(X_va @ beta, gs_va, n_va)
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(val_probs, y_va)

        # LightGBM 訓練（clogit_w=1.0 のセグメントでもモデルは保存しておく）
        X_lgbm_tr = trn.sort_values('race_id')[lgbm_valid].astype(float).fillna(0)
        y_lgbm_tr = (trn.sort_values('race_id')['着順_num'] == 1).astype(int)
        X_lgbm_va = val.sort_values('race_id')[lgbm_valid].astype(float).fillna(0)
        y_lgbm_va = (val.sort_values('race_id')['着順_num'] == 1).astype(int)

        lgbm_model = lgb.LGBMClassifier(**LGBM_PARAMS)
        lgbm_model.fit(
            X_lgbm_tr, y_lgbm_tr,
            eval_set=[(X_lgbm_va, y_lgbm_va)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )

        print(f'  訓練完了 {time.time()-t1:.0f}s  lgbm_iter={lgbm_model.best_iteration_}')

        # OOS 評価
        for period, df_p in parts.items():
            if len(df_p) == 0:
                print(f'  {period}: データなし')
                continue
            scored = score_segment(df_p, beta, scaler, iso, clogit_valid,
                                   lgbm_model, lgbm_valid, clogit_w)
            scored['rank_model'] = scored.groupby('race_id')['final_prob'].rank(
                ascending=False, method='first')
            top1 = scored[scored['rank_model'] == 1].copy()
            roi, wins = calc_roi(top1)
            nR = len(top1)
            print(f'  {period}: {nR}R  ROI={roi:+.4f}  勝率={wins/nR:.1%}')
            all_oos[period].append(top1)

        # 保存
        with open(os.path.join(seg_dir, 'clogit.pkl'), 'wb') as f:
            pickle.dump({'beta': beta, 'scaler': scaler, 'iso': iso,
                         'feat_cols': clogit_valid, 'seg_key': seg_key}, f)
        with open(os.path.join(seg_dir, 'lgbm.pkl'), 'wb') as f:
            pickle.dump({'model': lgbm_model, 'feat_cols': lgbm_valid,
                         'seg_key': seg_key}, f)
        print(f'  保存完了: {seg_dir}/')

    print('\n=== v3b 全体 ===')
    for period, tops in all_oos.items():
        if not tops:
            continue
        combined = pd.concat(tops, ignore_index=True)
        roi, wins = calc_roi(combined)
        n = len(combined)
        print(f'{period}: {n}R  ROI={roi:+.4f}  勝率={wins/n:.1%}')

    meta = {
        'weights': BLEND_W,
        'segments': [f'{s}_{d}' for s, d in SEGMENTS],
        'lgbm_params': LGBM_PARAMS,
    }
    with open(os.path.join(OUT_DIR, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f'\n総処理時間: {time.time()-t_start:.0f}s')
    print(f'モデル保存先: {OUT_DIR}')


if __name__ == '__main__':
    main()
