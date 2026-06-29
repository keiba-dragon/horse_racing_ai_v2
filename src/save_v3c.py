# coding: utf-8
"""
save_v3c.py - clogit + LGBMRanker (LambdaMART) ハイブリッド v3c
  ・LGBMClassifier → LGBMRanker (lambdarank, NDCG@1最適化)
  ・着順ラベル: 1着=3, 2着=2, 3着=1, 4着以下=0
  ・セグメント別ブレンド比はv3bと同一（芝0.9, ダ短1.0, ダ中長0.2）
  ・評価指標: 2025+2026 合算ROI を主指標

出力: models/v3c/{seg_key}/clogit.pkl + lgbm.pkl
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
from save_v3b import LGBM_BASE_FEATS, LGBM_SEG_EXTRA, _lgbm_feats, load_data

OUT_DIR = os.path.join(BASE_DIR, 'models', 'v3c')
os.makedirs(OUT_DIR, exist_ok=True)

SEGMENTS = [('芝', '短距離'), ('芝', '中距離'), ('芝', '長距離'),
            ('ダ', '短距離'), ('ダ', '中長距離')]

BLEND_W = {
    '芝_短距離':  0.9,
    '芝_中距離':  0.9,
    '芝_長距離':  0.9,
    'ダ_短距離':  1.0,
    'ダ_中長距離': 0.2,
}

LGBM_PARAMS = {
    'objective':         'lambdarank',
    'metric':            'ndcg',
    'ndcg_eval_at':      [1],
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


def make_rank_label(着順_num):
    """1着=3, 2着=2, 3着=1, 4着以下=0"""
    arr = np.clip(4 - 着順_num.values.astype(int), 0, 3)
    return arr.astype(int)


def get_lgb_groups(df_sorted):
    """race_idソート済みDFからLGB用グループサイズ配列を返す"""
    return df_sorted.groupby('race_id', sort=False)['race_id'].count().values


def score_segment(df_eval, beta, scaler, iso, clogit_feats,
                  lgbm_model, lgbm_valid_feats, clogit_w):
    valid_c = [c for c in clogit_feats if c in df_eval.columns]
    X_c, _, gs, n_total, *_ = prepare(df_eval, valid_c, scaler=scaler,
                                       top_idx=None, top_idx3=None)
    raw_p    = segment_softmax(X_c @ beta, gs, n_total)
    calib_p  = iso.predict(raw_p)
    clogit_p = race_normalize(calib_p, gs, n_total)

    lgbm_w = 1.0 - clogit_w
    df_s = df_eval.sort_values('race_id').reset_index(drop=True)
    if lgbm_w > 0:
        valid_l  = [c for c in lgbm_valid_feats if c in df_s.columns]
        X_l      = df_s[valid_l].astype(float).fillna(0).values
        lgbm_raw = lgbm_model.predict(X_l)          # ranker → score (大きいほど上位)
        lgbm_p   = race_normalize(np.exp(lgbm_raw - lgbm_raw.max()), gs, n_total)
        final_p  = clogit_w * clogit_p + lgbm_w * lgbm_p
    else:
        lgbm_p  = np.zeros(n_total)
        final_p = clogit_p

    df_s['clogit_prob'] = clogit_p
    df_s['lgbm_score']  = lgbm_p
    df_s['final_prob']  = final_p
    return df_s


def main():
    t_start = time.time()
    df = load_data()

    all_oos   = {k: [] for k in ['2324', '2025', '2026']}
    seg_rows  = []

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

        cf_all       = CLOGIT_FEATS[seg_key]
        lf_all       = _lgbm_feats(seg_key)
        clogit_valid = [c for c in cf_all if c in df_s.columns and df_s[c].isna().mean() <= 0.65]
        lgbm_valid   = [c for c in lf_all if c in df_s.columns and df_s[c].isna().mean() <= 0.65]

        print(f'\n[{seg_key}] clogit_w={clogit_w}  '
              f'clogit:{len(clogit_valid)}特徴  lgbm:{len(lgbm_valid)}特徴  '
              f'trn:{len(trn):,} val:{len(val):,}')

        t1 = time.time()

        # ── clogit 訓練 ─────────────────────────────────────────────────────
        X_tr, y_tr, gs_tr, n_tr, nr_tr, scaler, *_ = prepare(
            trn, clogit_valid, top_idx=None, top_idx3=None, fit=True)
        X_va, y_va, gs_va, n_va, nr_va, *_ = prepare(
            val, clogit_valid, scaler=scaler, top_idx=None, top_idx3=None)
        beta = adam_opt(X_tr, y_tr, gs_tr, n_tr, nr_tr,
                        X_va, y_va, gs_va, n_va, nr_va)
        val_probs = segment_softmax(X_va @ beta, gs_va, n_va)
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(val_probs, y_va)

        # ── LGBMRanker 訓練 ─────────────────────────────────────────────────
        trn_s  = trn.sort_values('race_id')
        val_s  = val.sort_values('race_id')
        X_lgtr = trn_s[lgbm_valid].astype(float).fillna(0)
        y_lgtr = make_rank_label(trn_s['着順_num'])
        g_lgtr = get_lgb_groups(trn_s)
        X_lgva = val_s[lgbm_valid].astype(float).fillna(0)
        y_lgva = make_rank_label(val_s['着順_num'])
        g_lgva = get_lgb_groups(val_s)

        lgbm_model = lgb.LGBMRanker(**LGBM_PARAMS)
        lgbm_model.fit(
            X_lgtr, y_lgtr,
            group=g_lgtr,
            eval_set=[(X_lgva, y_lgva)],
            eval_group=[g_lgva],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )

        print(f'  訓練完了 {time.time()-t1:.0f}s  lgbm_iter={lgbm_model.best_iteration_}')

        # ── OOS 評価 ────────────────────────────────────────────────────────
        seg_roi = {}
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
            seg_roi[period] = roi
            nR = len(top1)
            print(f'  {period}: {nR}R  ROI={roi:+.4f}  勝率={wins/nR:.1%}')
            all_oos[period].append(top1)
        seg_rows.append({'seg': seg_key, **seg_roi})

        # ── 保存 ────────────────────────────────────────────────────────────
        with open(os.path.join(seg_dir, 'clogit.pkl'), 'wb') as f:
            pickle.dump({'beta': beta, 'scaler': scaler, 'iso': iso,
                         'feat_cols': clogit_valid, 'seg_key': seg_key}, f)
        with open(os.path.join(seg_dir, 'lgbm.pkl'), 'wb') as f:
            pickle.dump({'model': lgbm_model, 'feat_cols': lgbm_valid,
                         'seg_key': seg_key, 'type': 'ranker'}, f)
        print(f'  保存完了: {seg_dir}/')

    # ── 全体集計（主指標: 2025+2026合算） ────────────────────────────────────
    print('\n=== v3c 全体 ===')
    for period, tops in all_oos.items():
        if not tops:
            continue
        combined = pd.concat(tops, ignore_index=True)
        roi, wins = calc_roi(combined)
        n = len(combined)
        print(f'{period}: {n}R  ROI={roi:+.4f}  勝率={wins/n:.1%}')

    # 2025+2026 合算
    tops_recent = all_oos['2025'] + all_oos['2026']
    if tops_recent:
        combined_r = pd.concat(tops_recent, ignore_index=True)
        roi_r, wins_r = calc_roi(combined_r)
        n_r = len(combined_r)
        print(f'2025+2026合算: {n_r}R  ROI={roi_r:+.4f}  勝率={wins_r/n_r:.1%}  ← 主指標')

    meta = {
        'weights': BLEND_W,
        'segments': [f'{s}_{d}' for s, d in SEGMENTS],
        'lgbm_params': LGBM_PARAMS,
        'lgbm_type': 'ranker',
    }
    with open(os.path.join(OUT_DIR, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f'\n総処理時間: {time.time()-t_start:.0f}s')
    print(f'モデル保存先: {OUT_DIR}')


if __name__ == '__main__':
    main()
