# coding: utf-8
"""
try_rank_transform.py - ランク変換でclogit
  各特徴量を訓練データ内でパーセンタイルランクに変換してからclogitを学習
  外れ値への頑健性向上、特徴量の非線形効果も部分的に捉える
"""
import sys, os
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

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


def run_with_transform(df_trn, df_val, oos_parts, feats, l2=0.0, use_quantile=False):
    valid = [c for c in feats if c in df_trn.columns and df_trn[c].isna().mean() <= 0.65]

    if use_quantile:
        # QuantileTransformer: 訓練データで均一分布にランク変換
        df_trn_s = df_trn.sort_values('race_id').reset_index(drop=True)
        X_raw_tr = df_trn_s[valid].astype(float).fillna(0).values
        qt = QuantileTransformer(n_quantiles=1000, output_distribution='uniform', random_state=42)
        qt.fit(X_raw_tr)

        # 変換後のデータでprepareを呼ぶため、dfに書き戻す
        df_trn_qt = df_trn_s.copy()
        df_val_qt = df_val.sort_values('race_id').reset_index(drop=True).copy()

        X_tr_qt = qt.transform(df_trn_s[valid].astype(float).fillna(0).values)
        X_va_qt = qt.transform(df_val_qt[valid].astype(float).fillna(0).values)

        for i, c in enumerate(valid):
            df_trn_qt[c] = X_tr_qt[:, i]
            df_val_qt[c] = X_va_qt[:, i]

        X_tr, y_tr, gs_tr, n_tr, nr_tr, scaler, *_ = prepare(
            df_trn_qt, valid, top_idx=None, top_idx3=None, fit=True)
        X_va, y_va, gs_va, n_va, nr_va, *_ = prepare(
            df_val_qt, valid, scaler=scaler, top_idx=None, top_idx3=None)
    else:
        X_tr, y_tr, gs_tr, n_tr, nr_tr, scaler, *_ = prepare(
            df_trn, valid, top_idx=None, top_idx3=None, fit=True)
        X_va, y_va, gs_va, n_va, nr_va, *_ = prepare(
            df_val, valid, scaler=scaler, top_idx=None, top_idx3=None)
        qt = None

    beta = adam_fit(X_tr, y_tr, gs_tr, n_tr, nr_tr,
                    X_va, y_va, gs_va, n_va, nr_va, l2=l2)

    oos_roi = {}
    for period, df_p in oos_parts.items():
        if len(df_p) == 0:
            continue
        valid_p = [c for c in valid if c in df_p.columns]
        if use_quantile and qt is not None:
            df_p_qt = df_p.sort_values('race_id').reset_index(drop=True).copy()
            X_p_qt = qt.transform(df_p_qt[valid_p].astype(float).fillna(0).values)
            for i, c in enumerate(valid_p):
                df_p_qt[c] = X_p_qt[:, i]
            X_p, _, gs_p, n_p, *_ = prepare(df_p_qt, valid_p, scaler=scaler,
                                              top_idx=None, top_idx3=None)
            scored = df_p_qt
        else:
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

    def show(label, roi):
        r24 = roi.get('2324', (0, 1, 0))[0]
        r25, n25, _ = roi.get('2025', (0, 1, 0))
        r26, n26, _ = roi.get('2026', (0, 1, 0))
        comb = (r25*n25 + r26*n26) / (n25+n26)
        print(f'  {label:<35} 2324:{r24:+.2%}  2025:{r25:+.2%}  2026:{r26:+.2%}  25+26:{comb:+.2%}')
        return comb

    print('=== ランク変換テスト ===')
    r1 = run_with_transform(df_trn, df_val, oos_parts, BASE_24, l2=L2, use_quantile=False)
    c1 = show('標準スケール (ベース)', r1)

    r2 = run_with_transform(df_trn, df_val, oos_parts, BASE_24, l2=L2, use_quantile=True)
    c2 = show('QuantileTransform→均一分布', r2)

    # Winsorization (1-99パーセンタイルでクリップ)
    # QuantileTransformerのoutput_distribution='normal'も試す
    from sklearn.preprocessing import QuantileTransformer as QT
    df2 = df.copy()
    df_trn2 = df2[(df2['日付_num'] >= 130101) & (df2['日付_num'] < 220101)]
    valid = [c for c in BASE_24 if c in df_trn2.columns and df_trn2[c].isna().mean() <= 0.65]
    for c in valid:
        lo = df_trn2[c].quantile(0.01)
        hi = df_trn2[c].quantile(0.99)
        df2[c] = df2[c].clip(lower=lo, upper=hi)

    df_trn3 = df2[(df2['日付_num'] >= 130101) & (df2['日付_num'] < 220101)]
    df_val3 = df2[(df2['日付_num'] >= 220101) & (df2['日付_num'] <= 221231)]
    oos3 = df2[df2['日付_num'] >= 230101]
    oos_parts3 = {
        '2324': oos3[oos3['日付_num'] < 250101],
        '2025': oos3[(oos3['日付_num'] >= 250101) & (oos3['日付_num'] < 260101)],
        '2026': oos3[oos3['日付_num'] >= 260101],
    }
    r3 = run_with_transform(df_trn3, df_val3, oos_parts3, BASE_24, l2=L2, use_quantile=False)
    c3 = show('1-99パーセンタイルWinsorize', r3)

    print()
    print(f'ベース (標準): {c1:+.2%}')
    print(f'QuantileTransform: {c2:+.2%}  Δ={c2-c1:+.2%}')
    print(f'Winsorize: {c3:+.2%}  Δ={c3-c1:+.2%}')
    print(f'21F基準(-19.70%) から最良: {max(c1,c2,c3)-(-0.197):+.2%}')


if __name__ == '__main__':
    main()
