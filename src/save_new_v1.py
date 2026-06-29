# coding: utf-8
"""
save_new_v1.py - New v1 モデル保存
  BASE_25 (25特徴, L2=0.006) ダート中長距離 特化モデルを roi_model.pkl の
  'ダ' artifact として保存。'芝' artifact は既存を維持。
"""
import sys, os, pickle, shutil
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
                print(f'  早期停止: epoch={epoch}')
                break
    return best_beta


def main():
    print('=== New v1 モデル保存 (BASE_25, ダート中長距離) ===')
    print()
    print('データ読み込み中...')
    df = load_segment()

    valid_feats = [c for c in BASE_25 if c in df.columns and df[c].isna().mean() <= 0.65]
    missing = [c for c in BASE_25 if c not in df.columns]
    if missing:
        print(f'WARNING: 欠損列: {missing}')
    print(f'使用特徴量: {len(valid_feats)}個')

    df_trn = df[(df['日付_num'] >= 130101) & (df['日付_num'] < 220101)]
    df_val = df[(df['日付_num'] >= 220101) & (df['日付_num'] <= 221231)]
    oos = df[df['日付_num'] >= 230101]
    oos_parts = {
        '2324': oos[oos['日付_num'] < 250101],
        '2025': oos[(oos['日付_num'] >= 250101) & (oos['日付_num'] < 260101)],
        '2026': oos[oos['日付_num'] >= 260101],
    }

    print(f'train: {len(df_trn):,}行, val: {len(df_val):,}行, OOS: {len(oos):,}行')
    print()

    print('学習中 (L2=0.006, Adam)...')
    X_tr, y_tr, gs_tr, n_tr, nr_tr, scaler, *_ = prepare(
        df_trn, valid_feats, top_idx=None, top_idx3=None, fit=True)
    X_va, y_va, gs_va, n_va, nr_va, *_ = prepare(
        df_val, valid_feats, scaler=scaler, top_idx=None, top_idx3=None)
    beta = adam_fit(X_tr, y_tr, gs_tr, n_tr, nr_tr,
                    X_va, y_va, gs_va, n_va, nr_va, l2=L2)
    print('学習完了')

    print('Isotonic calibration (val 2022)...')
    val_sorted = df_val.sort_values('race_id').reset_index(drop=True)
    raw_val = segment_softmax(X_va @ beta, gs_va, n_va)
    y_val = (val_sorted['着順_num'] == 1).astype(float).values
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(raw_val, y_val)
    print('Isotonic fit完了')
    print()

    print('=== OOS ROI 確認 ===')
    n25, n26 = 0, 0
    r25, r26 = 0.0, 0.0
    for period, df_p in oos_parts.items():
        if len(df_p) == 0:
            continue
        valid_p = [c for c in valid_feats if c in df_p.columns]
        X_p, _, gs_p, n_p, *_ = prepare(df_p, valid_p, scaler=scaler,
                                          top_idx=None, top_idx3=None)
        scored = df_p.sort_values('race_id').reset_index(drop=True)
        scored['prob'] = segment_softmax(X_p @ beta, gs_p, n_p)
        scored['rank'] = scored.groupby('race_id')['prob'].rank(ascending=False, method='first')
        top1 = scored[scored['rank'] == 1]
        roi, wins = calc_roi(top1)
        print(f'  {period}: ROI={roi:+.2%}  ({len(top1)}R, {wins}勝)')
        if period == '2025':
            r25, n25 = roi, len(top1)
        elif period == '2026':
            r26, n26 = roi, len(top1)
    if n25 + n26 > 0:
        comb = (r25 * n25 + r26 * n26) / (n25 + n26)
        print(f'  25+26合算: ROI={comb:+.2%}')
    print()

    final_pkl = os.path.join(MODEL_DIR, 'roi_model.pkl')
    backup_path = os.path.join(MODEL_DIR, 'final_model_pre_v1.pkl')
    print(f'既存モデル読み込み中: {final_pkl}')
    with open(final_pkl, 'rb') as f:
        existing_pkg = pickle.load(f)

    shutil.copy2(final_pkl, backup_path)
    print(f'バックアップ保存: {backup_path}')

    da_art = {
        'scaler':        scaler,
        'poly2':         None,
        'inter_scaler2': None,
        'top_idx':       None,
        'poly3':         None,
        'inter_scaler3': None,
        'top_idx3':      None,
        'coef':          beta,
        'feat_cols':     valid_feats,
        'isotonic':      iso,
    }

    new_artifacts = dict(existing_pkg['artifacts'])
    new_artifacts['ダ'] = da_art

    new_pkg = {
        'artifacts':    new_artifacts,
        'feat_cols':    valid_feats,
        'factor_maiden': existing_pkg.get('factor_maiden', 0.00),
        'factor_other':  existing_pkg.get('factor_other',  0.16),
        'total_oos_roi': comb if n25 + n26 > 0 else -0.1718,
        'note':          'New v1: BASE_25 (25特徴, L2=0.006) ダート中長距離',
        'version':       'new_v1',
    }

    with open(final_pkl, 'wb') as f:
        pickle.dump(new_pkg, f)
    print(f'保存完了: {final_pkl}')
    print()
    print('=== New v1 完了 ===')
    print('ダ artifact: BASE_25 (25特徴, L2=0.006, isotonic calibrated)')
    print('芝 artifact: 既存モデルから継承')


if __name__ == '__main__':
    main()
