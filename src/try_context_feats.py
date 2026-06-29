# coding: utf-8
"""
try_context_feats.py - 状況適合系特徴量テスト
  馬場・距離帯・馬場状態別の過去成績、クラス補正スコアなど
  BASE_25 = BASE_24 + タイム指数_近5走_slope でテスト
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

CANDIDATES = [
    # 状況適合
    '同距離帯_平均着順_近5走',      # 同距離帯での成績 (距離専門家)
    '同馬場_平均着順_近5走',        # 同馬場状態での成績 (馬場適性)
    '良馬場_平均着順_近5走',        # 良馬場限定の成績
    '芝ダ一致_平均着順_近5走',      # 同表面(ダート)での成績
    '道悪_平均着順_近5走',         # 道悪時の成績 (NaN多い可能性)
    # クラス補正系
    '近5走_クラス補正スコア',       # クラス補正の別指標
    '近5走_着差タイム_クラス補正平均', # クラス補正後の着差タイム平均
    '近5走_着差タイム平均',         # 着差タイム平均(クラス補正なし)
    '近5走_平均相対着順',          # 相対着順(フィールドサイズ補正?)
    # 安定性/経験系
    '着順_近5走_std',            # 着順のばらつき
    '近3走_体重増減合計',          # 直近3走の体重変化トレンド
    '格上経験数_近5走',           # 上のクラスで走った回数
    '最大クラス差_近5走',          # 格上経験の大きさ
    '同会場_出走数_近5走',         # 同会場の経験数
    # 騎手系
    '騎手馬場_r100_勝率',         # 騎手×馬場状態の勝率
    '騎手馬場_r100_複勝率',
    # 血統系
    '血統_ダ優位度',             # ダート血統の優位度
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
    print(f'  {label:<45} 2324:{r24:+.2%}  2025:{r25:+.2%}  2026:{r26:+.2%}  25+26:{comb:+.2%}{delta}{mark}')
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

    print('=== 状況適合系特徴テスト (BASE_25 + L2=0.006) ===')
    print()

    roi0 = evaluate(df_trn, df_val, oos_parts, BASE_25, l2=L2)
    c0 = show('BASE_25 ベース', roi0)
    print()

    avail = [c for c in CANDIDATES if c in df.columns]
    not_avail = [c for c in CANDIDATES if c not in df.columns]
    if not_avail:
        print(f'  ※ 列なし: {not_avail}')
    print()

    results = []
    for c in avail:
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
        return

    print('★ ポジティブ:')
    for d, c in positives:
        print(f'  {c}: Δ={d:+.2%}')

    print()
    print('ステップワイズ:')
    current_feats = BASE_25.copy()
    current_c = c0
    added = []
    for d, c in positives:
        trial_feats = current_feats + [c]
        roi = evaluate(df_trn, df_val, oos_parts, trial_feats, l2=L2)
        comb = show(f'累計', roi, current_c)
        if comb > current_c:
            current_feats = trial_feats
            current_c = comb
            added.append(c)
            print(f'    +{c} 採用 → {current_c:+.2%}')
        else:
            print(f'    +{c} 棄却')

    print()
    print(f'最終: {len(current_feats)}F → {current_c:+.2%} (Δ={current_c-c0:+.2%})')
    print(f'21F基準 から: {current_c-(-0.197):+.2%}')


if __name__ == '__main__':
    main()
