# coding: utf-8
"""
try_horse_venue.py - 馬の会場・クラス特化勝率テスト (訓練データから計算)
  - 馬の当会場勝率 (当該馬が同じ開催地で走った時の勝率)
  - 馬の全期間勝率 (訓練データ内のダート中長距離での勝率)
  - 馬の当会場出走数 (経験数)
  - 前走着差タイム vs 近5走平均の比 (相対着差)
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
BASE_25 = BASE_24 + ['タイム指数_近5走_slope']
L2 = 0.006


def compute_horse_stats(df_trn, df_all, min_runs=3):
    """訓練データから馬別の会場・全体勝率を計算 (min_runs未満は全体平均で補完)"""
    # 全体勝率
    horse_overall = df_trn.groupby('馬名S').apply(
        lambda x: pd.Series({
            '馬_全体勝率': (x['着順_num'] == 1).mean(),
            '馬_全体出走数': len(x),
        })
    ).reset_index()

    global_win_rate = horse_overall.loc[
        horse_overall['馬_全体出走数'] >= min_runs, '馬_全体勝率'
    ].mean()

    horse_overall.loc[horse_overall['馬_全体出走数'] < min_runs, '馬_全体勝率'] = global_win_rate
    horse_map = horse_overall.set_index('馬名S')['馬_全体勝率'].to_dict()

    # 会場別勝率
    horse_venue = df_trn.groupby(['馬名S', '開催']).apply(
        lambda x: pd.Series({
            '馬_会場勝率': (x['着順_num'] == 1).mean(),
            '馬_会場出走数': len(x),
        })
    ).reset_index()

    global_venue_wr = horse_venue.loc[
        horse_venue['馬_会場出走数'] >= min_runs, '馬_会場勝率'
    ].mean()

    # 少ない場合はグローバル平均
    horse_venue.loc[horse_venue['馬_会場出走数'] < min_runs, '馬_会場勝率'] = global_venue_wr
    venue_map = horse_venue.set_index(['馬名S', '開催'])['馬_会場勝率'].to_dict()

    df_all = df_all.copy()
    df_all['馬_全体勝率'] = df_all['馬名S'].map(horse_map).fillna(global_win_rate)
    df_all['馬_会場勝率'] = df_all.apply(
        lambda r: venue_map.get((r['馬名S'], r['開催']), horse_map.get(r['馬名S'], global_win_rate)),
        axis=1
    )
    return df_all


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

    # 相対着差タイム (前走着差 ÷ 近5走平均着差)
    diff1 = pd.to_numeric(df.get('前走着差タイム', pd.Series(np.nan, index=df.index)), errors='coerce')
    diff_avg = pd.to_numeric(df.get('近5走_着差タイム平均', pd.Series(np.nan, index=df.index)), errors='coerce')
    df['着差相対比'] = diff1 / (diff_avg.abs() + 0.1)  # 前走着差 / 平均(+ε for stability)

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


def evaluate(df_trn, df_val, oos_parts, feats, l2=0.0):
    valid = [c for c in feats if c in df_trn.columns and df_trn[c].isna().mean() <= 0.65]
    X_tr, y_tr, gs_tr, n_tr, nr_tr, scaler, *_ = prepare(
        df_trn, valid, top_idx=None, top_idx3=None, fit=True)
    X_va, y_va, gs_va, n_va, nr_va, *_ = prepare(
        df_val, valid, scaler=scaler, top_idx=None, top_idx3=None)
    beta = adam_fit(X_tr, y_tr, gs_tr, n_tr, nr_tr,
                    X_va, y_va, gs_va, n_va, nr_va, l2=l2)
    oos_roi = {}
    for period, df_p in oos_parts.items():
        if len(df_p) == 0:
            continue
        valid_p = [c for c in valid if c in df_p.columns]
        X_p, _, gs_p, n_p, *_ = prepare(df_p, valid_p, scaler=scaler,
                                          top_idx=None, top_idx3=None)
        scored = df_p.sort_values('race_id').reset_index(drop=True)
        scored['prob'] = segment_softmax(X_p @ beta, gs_p, n_p)
        scored['rank'] = scored.groupby('race_id')['prob'].rank(ascending=False, method='first')
        top1 = scored[scored['rank'] == 1]
        roi, wins = calc_roi(top1)
        oos_roi[period] = (roi, len(top1), wins)
    return oos_roi


def show(label, roi, c0=None):
    r24 = roi.get('2324', (0, 1, 0))[0]
    r25, n25, _ = roi.get('2025', (0, 1, 0))
    r26, n26, _ = roi.get('2026', (0, 1, 0))
    comb = (r25*n25 + r26*n26) / (n25+n26)
    delta = f'  Δ={comb-c0:+.2%}' if c0 is not None else ''
    mark = ' ★' if c0 is not None and comb > c0 else ''
    print(f'  {label:<50} 2324:{r24:+.2%}  2025:{r25:+.2%}  2026:{r26:+.2%}  25+26:{comb:+.2%}{delta}{mark}')
    return comb


def main():
    df = load_segment()
    df_trn = df[(df['日付_num'] >= 130101) & (df['日付_num'] < 220101)]

    df = compute_horse_stats(df_trn, df)
    df_trn = df[(df['日付_num'] >= 130101) & (df['日付_num'] < 220101)]
    df_val = df[(df['日付_num'] >= 220101) & (df['日付_num'] <= 221231)]
    oos = df[df['日付_num'] >= 230101]
    oos_parts = {
        '2324': oos[oos['日付_num'] < 250101],
        '2025': oos[(oos['日付_num'] >= 250101) & (oos['日付_num'] < 260101)],
        '2026': oos[oos['日付_num'] >= 260101],
    }

    CANDIDATES = ['馬_全体勝率', '馬_会場勝率', '着差相対比']

    for c in CANDIDATES:
        if c in df_trn.columns:
            print(f'{c}: NaN={df_trn[c].isna().mean():.1%}  mean={df_trn[c].mean():.4f}')
    print()

    print('=== 馬別会場・勝率テスト (BASE_25 + L2=0.006) ===')
    print()

    roi0 = evaluate(df_trn, df_val, oos_parts, BASE_25, l2=L2)
    c0 = show('BASE_25 ベース', roi0)
    print()

    results = []
    for c in CANDIDATES:
        if c not in df_trn.columns:
            print(f'  ※ 列なし: {c}')
            continue
        nan_frac = df_trn[c].isna().mean()
        if nan_frac > 0.65:
            print(f'  スキップ(NaN={nan_frac:.0%}): {c}')
            continue
        roi = evaluate(df_trn, df_val, oos_parts, BASE_25 + [c], l2=L2)
        comb = show(f'+{c}', roi, c0)
        results.append((comb - c0, c))

    positives = sorted([(d, c) for d, c in results if d > 0], reverse=True)
    print()
    if not positives:
        print('有望なし')
    else:
        print('★ ポジティブ:')
        for d, c in positives:
            print(f'  {c}: Δ={d:+.2%}')


if __name__ == '__main__':
    main()
