# coding: utf-8
"""
try_within_race_rank.py - レース内相対ランク特徴量
  「このレース内で最良TIの馬か」などの非線形閾値効果を捉える
  同レース内での各特徴量の順位(rank)を新特徴量として追加
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

# レース内ランクを作る特徴量（値が大きい方が良い場合は ascending=False）
# (元特徴量名, 新特徴量名, ascending=True→小さい方が良い/False→大きい方が良い)
RANK_FEATS = [
    ('近5走_タイム指数_max', 'rank_TI_max', False),      # 大きい方が良い
    ('1走前_タイム指数', 'rank_prev_TI', False),
    ('近5走_クラス調整_平均着順', 'rank_class_avg', True), # 小さい方が良い（着順）
    ('騎手コース_r100_勝率', 'rank_jockey', False),
    ('前走着差タイム', 'rank_chakusa', True),             # 小さい方が良い
    ('1走前_RPCI', 'rank_RPCI', False),
    ('調教師コース_r100_勝率', 'rank_trainer', False),
    ('種牡馬_勝率', 'rank_sire', False),
    ('近3走_複勝率', 'rank_place_rate', False),
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

    # レース内ランク特徴量を計算
    for src, dst, asc in RANK_FEATS:
        if src in df.columns:
            df[dst] = df.groupby('race_id')[src].rank(ascending=asc, method='average')
            # 正規化: ランクを0-1に変換（1位=0, 最下位=1 or その逆）
            field_size = df.groupby('race_id')['race_id'].transform('count')
            df[dst] = df[dst] / field_size

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

    rank_cols = [dst for _, dst, _ in RANK_FEATS]

    def show(label, roi):
        r24 = roi.get('2324', (0, 1, 0))[0]
        r25, n25, _ = roi.get('2025', (0, 1, 0))
        r26, n26, _ = roi.get('2026', (0, 1, 0))
        c = (r25*n25 + r26*n26) / (n25+n26)
        print(f'  {label:<40} 2324:{r24:+.2%}  2025:{r25:+.2%}  2026:{r26:+.2%}  25+26:{c:+.2%}')
        return c

    r0 = evaluate(df_trn, df_val, oos_parts, BASE_24, l2=L2)
    c0 = show('BASE_24 (ベース)', r0)

    # 全レース内ランク特徴量を追加
    r1 = evaluate(df_trn, df_val, oos_parts, BASE_24 + rank_cols, l2=L2)
    c1 = show('BASE_24 + 全レース内ランク', r1)
    print(f'    Δ25+26={c1-c0:+.2%}')

    print()
    print('個別ランク特徴量テスト:')
    best_c = c0; best_col = None
    for _, dst, _ in RANK_FEATS:
        r = evaluate(df_trn, df_val, oos_parts, BASE_24 + [dst], l2=L2)
        c = show(f'  +{dst}', r)
        d = c - c0
        print(f'    Δ25+26={d:+.2%}')
        if c > best_c:
            best_c = c; best_col = dst

    print()
    if best_col:
        print(f'最良: +{best_col} → {best_c:+.2%} (Δ={best_c-c0:+.2%})')
    else:
        print('有望なし')


if __name__ == '__main__':
    main()
