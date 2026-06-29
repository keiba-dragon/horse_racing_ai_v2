# coding: utf-8
"""
try_horse_venue2.py - 馬_会場勝率 の時間リーク対処版
  前スクリプトの問題: 訓練データ全体から馬_会場勝率を計算していた
  → 2013年のレースに2021年勝利が混入 (時間的リーク)

  修正: 各レースの時点より前のデータのみを使ってexpanding計算
  ただし OOS (2023+) には 2013-2021 全体を使うので問題なし

  具体的に:
    - 訓練内の正しい版: 各レースより前の同会場出走のみで累積計算
    - OOS用: 2013-2021 全体から計算 (これはリークなし)
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


def compute_venue_rate_proper(df_full, trn_end=220101, min_runs=3):
    """
    時間リーク対処版:
    - train/val (< trn_end): 各レース時点より前の同会場の過去レースのみで計算
    - OOS (>= trn_end): 訓練データ全体 (< trn_end) から計算
    """
    # 元の行位置を保存
    df = df_full.copy()
    df['__pos__'] = np.arange(len(df))

    trn_mask_full = df['日付_num'] < trn_end
    global_avg = (df.loc[trn_mask_full, '着順_num'] == 1).mean()

    # OOS 用: 訓練データ全体から馬×会場勝率を計算
    trn = df[trn_mask_full].copy()
    horse_venue_trn = trn.groupby(['馬名S', '開催']).apply(
        lambda x: pd.Series({'wins': (x['着順_num'] == 1).sum(), 'runs': len(x)})
    ).reset_index()
    horse_venue_trn['oos_rate'] = horse_venue_trn['wins'] / horse_venue_trn['runs']
    horse_venue_trn.loc[horse_venue_trn['runs'] < min_runs, 'oos_rate'] = global_avg
    oos_map = horse_venue_trn.set_index(['馬名S', '開催'])['oos_rate'].to_dict()

    # train/val 用: 馬×会場でソートして累積計算 (過去のみ)
    df_sorted = df.sort_values(['馬名S', '開催', '日付_num']).copy()
    df_sorted['is_win'] = (df_sorted['着順_num'] == 1).astype(int)
    df_sorted['cum_wins'] = df_sorted.groupby(['馬名S', '開催'])['is_win'].cumsum().shift(1).fillna(0)
    df_sorted['cum_runs'] = df_sorted.groupby(['馬名S', '開催']).cumcount()

    df_sorted['馬_会場勝率_proper'] = np.where(
        df_sorted['cum_runs'] < min_runs,
        global_avg,
        df_sorted['cum_wins'] / df_sorted['cum_runs'].clip(lower=1)
    )

    # OOS 部分だけ OOS map で上書き
    oos_mask = df_sorted['日付_num'] >= trn_end
    df_sorted.loc[oos_mask, '馬_会場勝率_proper'] = df_sorted.loc[oos_mask].apply(
        lambda r: oos_map.get((r['馬名S'], r['開催']), global_avg), axis=1
    )

    # 元の行順に戻す
    rate_array = np.full(len(df), global_avg)
    rate_array[df_sorted['__pos__'].values] = df_sorted['馬_会場勝率_proper'].values

    result = df_full.copy()
    result['馬_会場勝率_proper'] = rate_array
    return result


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
    print(f'  {label:<55} 2324:{r24:+.2%}  2025:{r25:+.2%}  2026:{r26:+.2%}  25+26:{comb:+.2%}{delta}{mark}')
    return comb


def main():
    print('データ読み込み・時間リーク対処版計算中...')
    df = load_segment()

    # 時間リーク対処版を計算
    df = compute_venue_rate_proper(df, trn_end=220101, min_runs=3)

    df_trn = df[(df['日付_num'] >= 130101) & (df['日付_num'] < 220101)]
    df_val = df[(df['日付_num'] >= 220101) & (df['日付_num'] <= 221231)]
    oos = df[df['日付_num'] >= 230101]
    oos_parts = {
        '2324': oos[oos['日付_num'] < 250101],
        '2025': oos[(oos['日付_num'] >= 250101) & (oos['日付_num'] < 260101)],
        '2026': oos[oos['日付_num'] >= 260101],
    }

    c = '馬_会場勝率_proper'
    print(f'{c}: NaN={df_trn[c].isna().mean():.1%}  mean={df_trn[c].mean():.4f}')
    print(f'  train内の値分布: min={df_trn[c].min():.3f} max={df_trn[c].max():.3f}')
    print()

    print('=== 馬_会場勝率 (時間リーク対処版) テスト ===')
    print()

    roi0 = evaluate(df_trn, df_val, oos_parts, BASE_25, l2=L2)
    c0 = show('BASE_25 ベース', roi0)
    print()

    roi1 = evaluate(df_trn, df_val, oos_parts, BASE_25 + [c], l2=L2)
    c1 = show(f'+{c}', roi1, c0)
    print()

    if c1 > c0:
        print(f'★ 採用 → 25+26: {c1:+.2%} (Δ={c1-c0:+.2%})')
        print(f'21F基準から: {c1-(-0.197):+.2%}')
    else:
        print(f'棄却 (Δ={c1-c0:+.2%})')
        print('前回の +3.63% は訓練データ内のリークによる偽陽性だった可能性あり')


if __name__ == '__main__':
    main()
