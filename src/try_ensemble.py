# coding: utf-8
"""
try_ensemble.py - ブートストラップバギングアンサンブル
  各モデルは訓練レースの80%ランダムサンプルで学習
  B=10-20 モデルの確率平均でROI改善を目指す
  また距離サブセグメント分割 (ダ1500-1700 vs ダ1800+) も試験
"""
import sys, os
import numpy as np
import pandas as pd

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from save_conditional_logit import (
    prepare, segment_softmax, BASE_DIR, DATA_FILE, LR, N_EPOCHS, PATIENCE
)
from save_v3 import add_computed_features, calc_roi

BASE_24 = [
    '近5走_クラス調整_平均着順', '近5走_タイム指数_max', '1走前_タイム指数', '前走着差タイム',
    '騎手コース_r100_勝率', '1走前_クラス調整着順', '調教師コース_r100_勝率',
    '1走前_RPCI', '1走前_上3F地点差', '斤量', '種牡馬_勝率',
    '間隔_長_flag', '1走前_脚質_num', '騎手変更', '馬番',
    '近3走_複勝率', '調教師_r200_複勝率',
    'ブリンカー変更', '2走前_クラス差', '4走前_クラス差', '1走前_馬場状態',
    '性別_num', '所属_num', 'キャリア_浅い',
]
L2 = 0.006


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
    return df


def _loss_grad(beta, X, y, gs, n, nr, l2=0.0):
    probs = segment_softmax(X @ beta, gs, n)
    loss = -np.sum(y * np.log(np.clip(probs, 1e-15, 1.0))) / nr + l2 * np.dot(beta, beta)
    grad = -(X.T @ (y - probs)) / nr + 2 * l2 * beta
    return loss, grad


def adam_fit(X_tr, y_tr, gs_tr, n_tr, nr_tr, X_va, y_va, gs_va, n_va, nr_va, l2=0.0):
    d = X_tr.shape[1]
    beta, m, v = np.zeros(d), np.zeros(d), np.zeros(d)
    b1, b2, eps = 0.9, 0.999, 1e-8
    t, best_val, best_beta, no_imp = 0, np.inf, np.zeros(d), 0
    for epoch in range(1, N_EPOCHS + 1):
        _, grad = _loss_grad(beta, X_tr, y_tr, gs_tr, n_tr, nr_tr, l2)
        t += 1
        m = b1*m + (1-b1)*grad
        v = b2*v + (1-b2)*grad**2
        beta -= LR * (m/(1-b1**t)) / (np.sqrt(v/(1-b2**t)) + eps)
        if epoch % 10 == 0:
            vl, _ = _loss_grad(beta, X_va, y_va, gs_va, n_va, nr_va, l2=0.0)
            if vl < best_val:
                best_val, best_beta, no_imp = vl, beta.copy(), 0
            else:
                no_imp += 1
            if no_imp >= PATIENCE // 10:
                break
    return best_beta


def fit_single(df_trn, df_val, feats, l2=0.0):
    valid = [c for c in feats if c in df_trn.columns and df_trn[c].isna().mean() <= 0.65]
    X_tr, y_tr, gs_tr, n_tr, nr_tr, scaler, *_ = prepare(
        df_trn, valid, top_idx=None, top_idx3=None, fit=True)
    X_va, y_va, gs_va, n_va, nr_va, *_ = prepare(
        df_val, valid, scaler=scaler, top_idx=None, top_idx3=None)
    beta = adam_fit(X_tr, y_tr, gs_tr, n_tr, nr_tr,
                    X_va, y_va, gs_va, n_va, nr_va, l2=l2)
    return beta, valid, scaler


def score_df(df_p, beta, valid, scaler):
    valid_p = [c for c in valid if c in df_p.columns]
    X_p, _, gs_p, n_p, *_ = prepare(df_p, valid_p, scaler=scaler,
                                      top_idx=None, top_idx3=None)
    scored = df_p.sort_values('race_id').reset_index(drop=True)
    if X_p.shape[1] != len(beta):
        beta_use = beta[:X_p.shape[1]]
    else:
        beta_use = beta
    scored['prob'] = segment_softmax(X_p @ beta_use, gs_p, n_p)
    return scored


def calc_ensemble_roi(oos_parts, betas_valids_scalers, df_val):
    oos_roi = {}
    for period, df_p in oos_parts.items():
        if len(df_p) == 0:
            continue
        # 各モデルのprobを平均
        prob_sum = None
        count = 0
        base_df = df_p.sort_values('race_id').reset_index(drop=True)
        for beta, valid, scaler in betas_valids_scalers:
            s = score_df(df_p, beta, valid, scaler)
            if prob_sum is None:
                prob_sum = s['prob'].values.copy()
            else:
                prob_sum = prob_sum + s['prob'].values
            count += 1
        avg_prob = prob_sum / count
        base_df['prob'] = avg_prob
        base_df['rank'] = base_df.groupby('race_id')['prob'].rank(ascending=False, method='first')
        top1 = base_df[base_df['rank'] == 1]
        roi, wins = calc_roi(top1)
        oos_roi[period] = (roi, len(top1), wins)
    return oos_roi


def show(label, roi, primary_keys=('2324', '2025', '2026')):
    parts = []
    ns = []
    for k in primary_keys:
        r, n, _ = roi.get(k, (0, 1, 0))
        parts.append(f'{k}:{r:+.2%}')
        ns.append((r, n))
    if len(ns) >= 2:
        comb = (ns[-2][0]*ns[-2][1] + ns[-1][0]*ns[-1][1]) / (ns[-2][1] + ns[-1][1])
    else:
        comb = ns[0][0]
    print(f'  {label:<40} {" ".join(parts)}  25+26:{comb:+.2%}')
    return comb


def main():
    df = load_segment()
    df_trn = df[(df['日付_num'] >= 130101) & (df['日付_num'] < 220101)]
    df_val = df[(df['日付_num'] >= 220101) & (df['日付_num'] <= 221231)]
    oos = df[df['日付_num'] >= 230101]
    oos_parts = {
        '2324': oos[oos['日付_num'] < 250101],
        '2025': oos[(oos['日付_num'] >= 250101) & (oos['日付_num'] < 260101)],
        '2026': oos[oos['日付_num'] >= 260101],
    }

    print('=== アンサンブル実験 ===')
    print()

    # ベースライン (B=1, 全データ)
    beta0, valid0, sc0 = fit_single(df_trn, df_val, BASE_24, l2=L2)
    bvs_single = [(beta0, valid0, sc0)]
    roi0 = calc_ensemble_roi(oos_parts, bvs_single, df_val)
    c0 = show('ベース (B=1, 全データ)', roi0)

    # ブートストラップバギング
    print()
    races_trn = df_trn['race_id'].unique()
    rng = np.random.default_rng(42)

    for B, frac in [(5, 0.8), (10, 0.8), (20, 0.8), (10, 0.7)]:
        betas_valids_scalers = []
        for b in range(B):
            n_sample = int(len(races_trn) * frac)
            sampled_races = rng.choice(races_trn, size=n_sample, replace=True)
            df_boot = df_trn[df_trn['race_id'].isin(sampled_races)]
            try:
                beta_b, valid_b, sc_b = fit_single(df_boot, df_val, BASE_24, l2=L2)
                betas_valids_scalers.append((beta_b, valid_b, sc_b))
            except Exception:
                pass
        roi = calc_ensemble_roi(oos_parts, betas_valids_scalers, df_val)
        c = show(f'バギング B={B} frac={frac:.0%}', roi)
        print(f'    Δ={c-c0:+.2%}')

    # 距離サブセグメント分割
    print()
    print('=== 距離サブセグメント分割 ===')
    # 距離分布確認
    dist_counts = df[df['日付_num'] >= 230101]['dist_m'].value_counts().sort_index()
    print('OOS期間の距離分布:')
    for d, cnt in dist_counts.items():
        print(f'  ダ{d:.0f}: {cnt}頭')

    print()
    # ダ1500-1700 vs ダ1800+ 分割
    for cut in [1800, 1700]:
        df_short = df[df['dist_m'] < cut].copy()
        df_long  = df[df['dist_m'] >= cut].copy()

        print(f'切り口: ダ<{cut} vs ダ≥{cut}')
        if len(df_short) < 1000 or len(df_long) < 1000:
            print('  データ不足でスキップ')
            continue

        # 短距離モデル
        trn_s = df_short[(df_short['日付_num'] >= 130101) & (df_short['日付_num'] < 220101)]
        val_s = df_short[(df_short['日付_num'] >= 220101) & (df_short['日付_num'] <= 221231)]
        oos_s = df_short[df_short['日付_num'] >= 230101]

        trn_l = df_long[(df_long['日付_num'] >= 130101) & (df_long['日付_num'] < 220101)]
        val_l = df_long[(df_long['日付_num'] >= 220101) & (df_long['日付_num'] <= 221231)]
        oos_l = df_long[df_long['日付_num'] >= 230101]

        if len(trn_s) < 500 or len(trn_l) < 500:
            print('  訓練データ不足でスキップ')
            continue

        try:
            beta_s, valid_s, sc_s = fit_single(trn_s, val_s, BASE_24, l2=L2)
            beta_l, valid_l, sc_l = fit_single(trn_l, val_l, BASE_24, l2=L2)
        except Exception as e:
            print(f'  エラー: {e}')
            continue

        # 各サブセグメントのOOS評価
        for label_suffix, df_oos_s, df_oos_l in [
            ('2324', oos_s[oos_s['日付_num'] < 250101], oos_l[oos_l['日付_num'] < 250101]),
            ('2025', oos_s[(oos_s['日付_num']>=250101)&(oos_s['日付_num']<260101)],
                     oos_l[(oos_l['日付_num']>=250101)&(oos_l['日付_num']<260101)]),
            ('2026', oos_s[oos_s['日付_num']>=260101], oos_l[oos_l['日付_num']>=260101]),
        ]:
            rois = []
            ns_wins = []
            for (df_p, beta, valid, sc) in [(df_oos_s, beta_s, valid_s, sc_s),
                                             (df_oos_l, beta_l, valid_l, sc_l)]:
                if len(df_p) == 0:
                    continue
                s = score_df(df_p, beta, valid, sc)
                s['rank'] = s.groupby('race_id')['prob'].rank(ascending=False, method='first')
                top1 = s[s['rank'] == 1]
                roi_v, wins = calc_roi(top1)
                rois.append((roi_v, len(top1), wins))

            if rois:
                # 結合ROI
                total_n = sum(r[1] for r in rois)
                total_wins = sum(r[2] for r in rois)
                total_ret = sum(r[0]*r[1] for r in rois)
                combined_roi = total_ret / total_n
                print(f'  {label_suffix}: n={total_n}R  ROI={combined_roi:+.2%}  勝率={total_wins/total_n:.1%}')


if __name__ == '__main__':
    main()
