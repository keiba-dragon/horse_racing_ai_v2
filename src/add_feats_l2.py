# coding: utf-8
"""
add_feats_l2.py - L2=0.006 環境で特徴量を再探索
  正則化があると過学習しにくいため、以前弾かれた特徴量が有効になる可能性あり
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

# 以前テストして微マイナス〜neutral だったもの + まだ未試験のもの
CANDIDATES = [
    # 以前 neutral だったもの
    '相手レベル_実力差', '相手レベル_平均着順',
    '着順_近5走_std', '近5走_タイム指数_std',
    '近5走_タイム指数_range', '近5走_RPCI平均',
    # 以前小マイナスだったもの（L2で変わるかも）
    '内外枠', '斤量変化', '距離変化_前走',
    '3走前_クラス差', '5走前_クラス差',
    '同距離帯_平均着順_近5走', '2走前_タイム指数',
    '2走前_RPCI', '2走前_上3F地点差',
    'コース展開マッチ', '展開フィット_v2', '展開_コース_脚質フィット',
    '騎手コース距離_r100_勝率', '騎手調教師_r100_勝率',
    '母父馬_勝率', '生産者_勝率', '産地_勝率',
    '休み明けフラグ', '連闘フラグ', '近走連続入着数',
    '近3走_勝率', '格上経験数_近5走', '最大クラス差_近5走',
    '近5走_平均相対着順', '近5走_クラス補正スコア',
    # 以前中程度マイナスだったもの
    '年齢', '芝ダ転向', 'キャリア',
    '馬体重増減', '近3走_体重増減合計',
    '同会場_複勝率_近5走', '同馬場_平均着順_近5走',
    '2走前_脚質_num', '2走前_着順_num',
    # 前回第2ラウンドで未試験のもの
    '前走人気', '1走前_PCI',
    '近5走_平均4角位置', '前走_4角位置',
    '近5走_着差タイム平均',
    '騎手コース_r100_複勝率', '調教師コース_r100_複勝率',
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
    return df


def _loss_grad(beta, X, y, gs, n, nr, l2=0.0):
    probs = segment_softmax(X @ beta, gs, n)
    res   = y - probs
    loss  = -np.sum(y * np.log(np.clip(probs, 1e-15, 1.0))) / nr + l2 * np.dot(beta, beta)
    grad  = -(X.T @ res) / nr + 2 * l2 * beta
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
    return best_beta, best_val


def evaluate(df_trn, df_val, oos_parts, feats, l2=0.0):
    valid = [c for c in feats if c in df_trn.columns and df_trn[c].isna().mean() <= 0.65]
    if not valid:
        return None, {}
    X_tr, y_tr, gs_tr, n_tr, nr_tr, scaler, *_ = prepare(
        df_trn, valid, top_idx=None, top_idx3=None, fit=True)
    X_va, y_va, gs_va, n_va, nr_va, *_ = prepare(
        df_val, valid, scaler=scaler, top_idx=None, top_idx3=None)
    beta, _ = adam_fit(X_tr, y_tr, gs_tr, n_tr, nr_tr,
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
        scored['rank'] = scored.groupby('race_id')['prob'].rank(
            ascending=False, method='first')
        top1 = scored[scored['rank'] == 1]
        roi, _ = calc_roi(top1)
        oos_roi[period] = (roi, len(top1))
    return beta, oos_roi


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

    _, roi0 = evaluate(df_trn, df_val, oos_parts, BASE_24, l2=L2)
    r25_0, n25 = roi0['2025']
    r26_0, n26 = roi0['2026']
    comb0 = (r25_0*n25 + r26_0*n26) / (n25+n26)
    print(f'ベース24F (L2={L2}): 2025:{r25_0:+.2%}  2026:{r26_0:+.2%}  25+26:{comb0:+.2%}')
    print()

    results = []
    for cand in CANDIDATES:
        if cand in BASE_24 or cand not in df.columns:
            continue
        nan_trn = df_trn[cand].isna().mean() if cand in df_trn.columns else 1.0
        if nan_trn > 0.65:
            continue
        _, roi = evaluate(df_trn, df_val, oos_parts, BASE_24 + [cand], l2=L2)
        r25 = roi.get('2025', (0, 0))[0]
        r26 = roi.get('2026', (0, 0))[0]
        n25_t = roi.get('2025', (0, 1))[1]
        n26_t = roi.get('2026', (0, 1))[1]
        comb = (r25*n25_t + r26*n26_t) / (n25_t+n26_t) if n25_t+n26_t > 0 else 0
        d25 = r25 - r25_0; d26 = r26 - r26_0; d_comb = comb - comb0
        both = (d25 > 0 and d26 > 0)
        results.append((cand, d25, d26, d_comb, both, nan_trn))
        sym = '✓' if d_comb > 0.003 else ('✗' if d_comb < -0.003 else '~')
        agree = '[両✓]' if both else ''
        print(f'{sym} {cand:<35} Δ25+26:{d_comb:>+7.2%}  Δ25:{d25:>+7.2%}  Δ26:{d26:>+7.2%}  {agree}')

    results.sort(key=lambda x: -x[3])
    print()
    print('=== ランキング ===')
    for cand, d25, d26, d_comb, both, _ in results[:15]:
        sym = '✓' if d_comb > 0.003 else ('✗' if d_comb < -0.003 else '~')
        agree = '[両✓]' if both else ''
        print(f'{sym} {cand:<33} Δ25+26:{d_comb:>+7.2%}  {agree}')

    promising = [(c, d, b) for c, _, _, d, b, _ in results if d > 0.003]
    print()
    if promising:
        print(f'【有望 (L2={L2})】:')
        for c, d, b in promising:
            print(f'  ✓ {c}: {d:+.2%}{"  [両✓]" if b else ""}')
    else:
        print('有望なし')


if __name__ == '__main__':
    main()
