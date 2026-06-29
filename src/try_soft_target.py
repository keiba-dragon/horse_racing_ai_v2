# coding: utf-8
"""
try_soft_target.py - soft/ordinal target でclogit学習
  通常: y=1 for 1着のみ
  案A: y ∝ (3,2,1) for 1~3着, 残り0  → 3倍のシグナル
  案B: y ∝ (1/着順) normalized  → 順位に反比例する重み
  案C: Plackett-Luce approximation: 1着にy=1, 2着にy=0.5, 3着にy=0.25 (diminishing)
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


def compute_soft_y(df_sorted, gs, n_total, target='win'):
    """race_idでソート済みdfから soft y を計算
    gs: group_starts (get_group_starts の出力 — 各レースの開始インデックス)
    """
    y = np.zeros(len(df_sorted))
    ends = np.append(gs[1:], n_total)
    for start, end in zip(gs, ends):
        chunk = df_sorted.iloc[start:end]
        ranks = chunk['着順_num'].values

        if target == 'win':
            w = (ranks == 1).astype(float)
        elif target == 'linear321':
            w = np.where(ranks == 1, 3.0,
                np.where(ranks == 2, 2.0,
                np.where(ranks == 3, 1.0, 0.0)))
        elif target == 'inv_rank':
            w = np.where(ranks <= 3, 1.0/ranks, 0.0)
        elif target == 'geometric':
            w = np.where(ranks == 1, 1.0,
                np.where(ranks == 2, 0.5,
                np.where(ranks == 3, 0.25, 0.0)))
        elif target == 'top3_equal':
            w = np.where(ranks <= 3, 1.0, 0.0)
        else:
            w = (ranks == 1).astype(float)

        s = w.sum()
        if s > 0:
            w = w / s
        y[start:end] = w
    return y


def _loss_grad(beta, X, y, gs, n, nr, l2=0.0):
    probs = segment_softmax(X @ beta, gs, n)
    loss = -np.sum(y * np.log(np.clip(probs, 1e-15, 1.0))) / nr + l2 * np.dot(beta, beta)
    grad = -(X.T @ (y - probs)) / nr + 2 * l2 * beta
    return loss, grad


def adam_fit(X_tr, y_tr, gs_tr, n_tr, nr_tr, X_va, y_va_win, gs_va, n_va, nr_va, l2=0.0):
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
            # 検証は win-target で評価（本番と同じ）
            vl, _ = _loss_grad(beta, X_va, y_va_win, gs_va, n_va, nr_va, l2=0.0)
            if vl < best_val:
                best_val, best_beta, no_imp = vl, beta.copy(), 0
            else:
                no_imp += 1
            if no_imp >= PATIENCE // 10:
                break
    return best_beta


def run(df_trn, df_val, oos_parts, feats, l2=0.0, train_target='win'):
    valid = [c for c in feats if c in df_trn.columns and df_trn[c].isna().mean() <= 0.65]

    # prepare() sorts by race_id internally
    df_trn_s = df_trn.sort_values('race_id').reset_index(drop=True)
    df_val_s  = df_val.sort_values('race_id').reset_index(drop=True)

    X_tr, y_tr_win, gs_tr, n_tr, nr_tr, scaler, *_ = prepare(
        df_trn_s, valid, top_idx=None, top_idx3=None, fit=True)
    X_va, y_va_win, gs_va, n_va, nr_va, *_ = prepare(
        df_val_s, valid, scaler=scaler, top_idx=None, top_idx3=None)

    # 訓練用yをtarget種別で計算 (gs_tr=group_starts, n_tr=total_horses)
    y_tr = compute_soft_y(df_trn_s, gs_tr, n_tr, target=train_target)

    beta = adam_fit(X_tr, y_tr, gs_tr, n_tr, nr_tr,
                    X_va, y_va_win, gs_va, n_va, nr_va, l2=l2)

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


def show(label, roi):
    r24 = roi.get('2324', (0, 1, 0))[0]
    r25, n25, _ = roi.get('2025', (0, 1, 0))
    r26, n26, _ = roi.get('2026', (0, 1, 0))
    comb = (r25*n25 + r26*n26) / (n25+n26)
    print(f'  {label:<38} 2324:{r24:+.2%}  2025:{r25:+.2%}  2026:{r26:+.2%}  25+26:{comb:+.2%}')
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

    print('=== Soft Target 比較 (BASE_24, L2=0.006) ===')
    print()

    c0 = None
    for target, label in [
        ('win',        'ベース (y=1着のみ)'),
        ('linear321',  'soft top3: y∝(3,2,1)/6'),
        ('inv_rank',   'soft top3: y∝1/着順'),
        ('geometric',  'soft top3: y=(1,0.5,0.25)'),
        ('top3_equal', 'soft top3: 均等1/3'),
    ]:
        roi = run(df_trn, df_val, oos_parts, BASE_24, l2=L2, train_target=target)
        c = show(label, roi)
        if c0 is None:
            c0 = c
        else:
            print(f'    Δ={c-c0:+.2%}')

    print()
    print(f'21F基準(-19.70%) から最良: {max(c0, c0):+.2%}')


if __name__ == '__main__':
    main()
