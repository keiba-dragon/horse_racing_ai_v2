# coding: utf-8
"""
baseline_da_short.py - ダート短距離 (<=1400m) の現状確認
  現行 roi_model.pkl (BASE_25, ダート中長距離で訓練) を短距離に当てたときのROI
  + 短距離専用に BASE_25 を訓練したときの比較
"""
import sys, os, pickle
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from save_conditional_logit import (
    prepare, segment_softmax, BASE_DIR, DATA_FILE, LR, N_EPOCHS, PATIENCE
)
from save_v3 import add_computed_features, calc_roi

MODEL_DIR = os.path.join(BASE_DIR, 'models')

BASE_25 = [
    '近5走_クラス調整_平均着順', '近5走_タイム指数_max', '1走前_タイム指数', '前走着差タイム',
    '騎手コース_r100_勝率', '1走前_クラス調整着順', '調教師コース_r100_勝率',
    '1走前_RPCI', '1走前_上3F地点差', '斤量', '種牡馬_勝率',
    '間隔_長_flag', '1走前_脚質_num', '騎手変更', '馬番',
    '近3走_複勝率', '調教師_r200_複勝率',
    'ブリンカー変更', '2走前_クラス差', '4走前_クラス差', '1走前_馬場状態',
    '性別_num', '所属_num', 'キャリア_浅い', 'タイム指数_近5走_slope',
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
    df = df[(df['surface'] == 'ダ') & (dm <= 1400)].copy()
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
    comb = (r25 * n25 + r26 * n26) / (n25 + n26) if n25 + n26 > 0 else 0
    delta = f'  Δ={comb-c0:+.2%}' if c0 is not None else ''
    mark  = ' ★' if c0 is not None and comb > c0 else ''
    print(f'  {label:<50} 2324:{r24:+.2%}  2025:{r25:+.2%}  2026:{r26:+.2%}  25+26:{comb:+.2%}{delta}{mark}')
    return comb


def main():
    print('=== ダート短距離 (<=1400m) ベースライン確認 ===')
    print()

    print('データ読み込み中...')
    df = load_segment()
    df_trn = df[(df['日付_num'] >= 130101) & (df['日付_num'] < 220101)]
    df_val = df[(df['日付_num'] >= 220101) & (df['日付_num'] <= 221231)]
    oos    = df[df['日付_num'] >= 230101]
    oos_parts = {
        '2324': oos[oos['日付_num'] < 250101],
        '2025': oos[(oos['日付_num'] >= 250101) & (oos['日付_num'] < 260101)],
        '2026': oos[oos['日付_num'] >= 260101],
    }
    print(f'train: {len(df_trn):,}行  val: {len(df_val):,}行  OOS: {len(oos):,}行')

    # ── A: 現行 roi_model.pkl (BASE_25 で中長距離訓練) をそのまま当てる ──
    print()
    print('--- A: 現行 roi_model.pkl (BASE_25, 中長距離訓練) を短距離に適用 ---')
    with open(os.path.join(MODEL_DIR, 'roi_model.pkl'), 'rb') as f:
        pkg = pickle.load(f)
    art   = pkg['artifacts']['ダ']
    beta  = art['coef']
    scaler_a = art['scaler']
    feats_a  = art['feat_cols']

    oos_roi_a = {}
    for period, df_p in oos_parts.items():
        if len(df_p) == 0:
            continue
        valid_p = [c for c in feats_a if c in df_p.columns]
        X_p, _, gs_p, n_p, *_ = prepare(df_p, valid_p, scaler=scaler_a,
                                          top_idx=None, top_idx3=None)
        scored = df_p.sort_values('race_id').reset_index(drop=True)
        scored['prob'] = segment_softmax(X_p @ beta, gs_p, n_p)
        scored['rank'] = scored.groupby('race_id')['prob'].rank(ascending=False, method='first')
        top1 = scored[scored['rank'] == 1]
        roi, wins = calc_roi(top1)
        oos_roi_a[period] = (roi, len(top1), wins)
    c_a = show('A: 中長距離BASE_25をそのまま適用 (現状)', oos_roi_a)

    # ── B: 短距離データで BASE_25 を再訓練 ──────────────────────────────────
    print()
    print('--- B: 短距離データで BASE_25 を再訓練 ---')
    oos_roi_b = evaluate(df_trn, df_val, oos_parts, BASE_25, l2=L2)
    c_b = show('B: BASE_25 短距離再訓練', oos_roi_b, c_a)

    # ── C: 旧 final_model_pre_v1.pkl (320特徴, 全ダート訓練) ─────────────────
    pre_pkl = os.path.join(MODEL_DIR, 'final_model_pre_v1.pkl')
    if os.path.exists(pre_pkl):
        print()
        print('--- C: 旧モデル (320特徴, 全ダート訓練) を短距離に適用 ---')
        with open(pre_pkl, 'rb') as f:
            pkg_old = pickle.load(f)
        art_old   = pkg_old['artifacts']['ダ']
        beta_old  = art_old['coef']
        scaler_old = art_old['scaler']
        feats_old  = art_old['feat_cols']

        oos_roi_c = {}
        for period, df_p in oos_parts.items():
            if len(df_p) == 0:
                continue
            # 旧モデルの特徴量を全て補完 (欠損列はNaN→0)
            df_p2 = df_p.copy()
            for fc in feats_old:
                if fc not in df_p2.columns:
                    df_p2[fc] = np.nan
            try:
                X_p, _, gs_p, n_p, *_ = prepare(df_p2, feats_old, scaler=scaler_old,
                                                  poly2=art_old.get('poly2'),
                                                  inter_scaler2=art_old.get('inter_scaler2'),
                                                  top_idx=art_old['top_idx'],
                                                  poly3=art_old.get('poly3'),
                                                  inter_scaler3=art_old.get('inter_scaler3'),
                                                  top_idx3=art_old.get('top_idx3'))
                scored = df_p2.sort_values('race_id').reset_index(drop=True)
                scored['prob'] = segment_softmax(X_p @ beta_old, gs_p, n_p)
                scored['rank'] = scored.groupby('race_id')['prob'].rank(ascending=False, method='first')
                top1 = scored[scored['rank'] == 1]
                roi, wins = calc_roi(top1)
                oos_roi_c[period] = (roi, len(top1), wins)
            except Exception as e:
                print(f'    エラー ({period}): {e}')
                continue
        c_c = show('C: 旧モデル320特徴 (全ダート訓練)', oos_roi_c, c_a)

    print()
    print('=== まとめ ===')
    print(f'  A (現状・中長距離モデル転用): {c_a:+.2%}')
    print(f'  B (BASE_25 短距離再訓練):     {c_b:+.2%}  Δ={c_b-c_a:+.2%}')
    if os.path.exists(pre_pkl):
        print(f'  C (旧320特徴 全ダート):       {c_c:+.2%}  Δ={c_c-c_a:+.2%}')
    print()
    print('→ 最良をベースに特徴量探索へ')


if __name__ == '__main__':
    main()
