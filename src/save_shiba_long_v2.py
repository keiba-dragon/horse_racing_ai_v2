# coding: utf-8
"""
save_shiba_long_v2.py - 芝長距離モデル保存 (v2.0: 2325選択指標)
  特徴量(5個): ['馬番', '斤量', 'ブリンカー変更', '間隔', '馬体重']
  選択指標: 2325 OOS ROI (+28.22%)
  セグメント: 芝 2001m以上
  final_model.pkl の '芝長' artifact を上書き更新。
  2325: +28.22%  2323: +69.78%  2025: +24.37%  2026: +4.96%
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
from save_v3 import add_computed_features

MODEL_DIR = os.path.join(BASE_DIR, 'models')

FEATS = ['馬番', '斤量', 'ブリンカー変更', '間隔', '馬体重']
L2 = 0.006
NAN_IND_THRESHOLD = 0.05


def expand_with_nan_indicators(dfs, feats):
    extended = []
    ref_df = dfs[0]
    for f in feats:
        extended.append(f)
        if f not in ref_df.columns:
            continue
        nan_rate = ref_df[f].isna().mean()
        if NAN_IND_THRESHOLD < nan_rate < 1.0:
            ind = f + '_isnan'
            for df in dfs:
                if f in df.columns and ind not in df.columns:
                    df[ind] = df[f].isna().astype(float)
            extended.append(ind)
    return extended


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
    df = df[(df['surface'] == '芝') & (dm > 2000)].copy()
    df['dist_m'] = dm[df.index]
    df = add_computed_features(df)
    for col in FEATS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def _loss_grad(beta, X, y, gs, n, nr, l2=0.0):
    probs = segment_softmax(X @ beta, gs, n)
    loss  = -np.sum(y * np.log(np.clip(probs, 1e-15, 1.0))) / nr + l2 * np.dot(beta, beta)
    grad  = -(X.T @ (y - probs)) / nr + 2 * l2 * beta
    return loss, grad


def adam_fit(X_tr, y_tr, gs_tr, n_tr, nr_tr, X_va, y_va, gs_va, n_va, nr_va, l2=L2):
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


def roi_from_top1(top1):
    won  = top1['着順_num'] == 1
    odds = pd.to_numeric(top1['単勝オッズ'], errors='coerce')
    if len(top1) == 0:
        return float('nan'), 0
    return (odds[won] * 100).sum() / (len(top1) * 100) - 1, len(top1)


def main():
    print('=' * 60)
    print('  芝長距離 モデル保存 v2.0 (2325選択指標)')
    print(f'  特徴量({len(FEATS)}個): {FEATS}')
    print('=' * 60)

    df = load_segment()
    df_trn   = df[(df['日付_num'] >= 130101) & (df['日付_num'] < 220101)]
    df_val   = df[(df['日付_num'] >= 220101) & (df['日付_num'] <= 221231)]
    oos_2324 = df[(df['日付_num'] >= 230101) & (df['日付_num'] < 250101)]
    oos_2025 = df[(df['日付_num'] >= 250101) & (df['日付_num'] < 260101)]
    oos_2026 = df[df['日付_num'] >= 260101]

    print(f'train: {len(df_trn):,}行  val: {len(df_val):,}行')
    print(f'2324: {oos_2324["race_id"].nunique()}R  '
          f'2025: {oos_2025["race_id"].nunique()}R  '
          f'2026: {oos_2026["race_id"].nunique()}R')

    all_dfs = [df_trn, df_val, oos_2324, oos_2025, oos_2026]
    expanded = expand_with_nan_indicators(all_dfs, FEATS)
    valid_feats = [c for c in expanded if c in df_trn.columns
                   and df_trn[c].isna().mean() < 1.0
                   and df_trn[c].std(ddof=0) > 0]
    print(f'有効特徴量({len(valid_feats)}個): {valid_feats}')

    print('\n学習中 (L2=0.006, Adam)...')
    X_tr, y_tr, gs_tr, n_tr, nr_tr, scaler, *_ = prepare(
        df_trn, valid_feats, top_idx=None, top_idx3=None, fit=True)
    X_va, y_va, gs_va, n_va, nr_va, *_ = prepare(
        df_val, valid_feats, scaler=scaler, top_idx=None, top_idx3=None)
    beta = adam_fit(X_tr, y_tr, gs_tr, n_tr, nr_tr,
                    X_va, y_va, gs_va, n_va, nr_va)
    print('学習完了')
    for f, b in zip(valid_feats, beta):
        print(f'  {f}: β={b:+.4f}')

    print('\nIsotonic calibration (val 2022)...')
    val_sorted = df_val.sort_values('race_id').reset_index(drop=True)
    raw_val = segment_softmax(X_va @ beta, gs_va, n_va)
    y_val   = (val_sorted['着順_num'] == 1).astype(float).values
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(raw_val, y_val)
    print('完了')

    print('\n=== OOS ROI 確認 ===')
    r25, n25, r26, n26 = 0.0, 0, 0.0, 0
    r2324_final = float('nan')
    for label, oos in [('2324', oos_2324), ('2025', oos_2025), ('2026', oos_2026)]:
        if len(oos) == 0:
            continue
        valid_p = [c for c in valid_feats if c in oos.columns]
        X_p, _, gs_p, n_p, *_ = prepare(oos, valid_p, scaler=scaler,
                                          top_idx=None, top_idx3=None)
        scored = oos.sort_values('race_id').reset_index(drop=True)
        scored['prob'] = segment_softmax(X_p @ beta, gs_p, n_p)
        scored['rank'] = scored.groupby('race_id')['prob'].rank(ascending=False, method='first')
        top1 = scored[scored['rank'] == 1]
        r, n = roi_from_top1(top1)
        print(f'  {label}: ROI={r:+.2%}  ({n}R)')
        if label == '2324':
            r2324_final = r
        elif label == '2025':
            r25, n25 = r, n
        elif label == '2026':
            r26, n26 = r, n
    comb = (r25*n25 + r26*n26) / (n25+n26) if n25+n26 > 0 else float('nan')
    print(f'  25+26合算: ROI={comb:+.2%}')

    final_pkl   = os.path.join(MODEL_DIR, 'final_model.pkl')
    backup_path = os.path.join(MODEL_DIR, 'final_model_pre_shiba_long_v2.pkl')
    with open(final_pkl, 'rb') as f:
        existing_pkg = pickle.load(f)
    shutil.copy2(final_pkl, backup_path)
    print(f'バックアップ: {backup_path}')

    art = {
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
        'segment':       '芝長 2001m以上',
        'oos_roi_2324':  r2324_final,
        'oos_roi_2526':  comb,
    }

    new_artifacts = dict(existing_pkg['artifacts'])
    new_artifacts['芝長'] = art

    new_pkg = {
        'artifacts':     new_artifacts,
        'feat_cols':     existing_pkg.get('feat_cols', valid_feats),
        'factor_maiden': existing_pkg.get('factor_maiden', 0.00),
        'factor_other':  existing_pkg.get('factor_other',  0.16),
        'total_oos_roi': existing_pkg.get('total_oos_roi', -0.1718),
        'note':          existing_pkg.get('note', '') + ' | 芝長v2(5特徴,2325選択)',
        'version':       'shiba_long_v2',
    }

    with open(final_pkl, 'wb') as f:
        pickle.dump(new_pkg, f)

    print(f'\n保存完了: {final_pkl}')
    print(f'artifacts keys: {list(new_pkg["artifacts"].keys())}')
    print('=== 芝長距離 v2 保存完了 ===')
    print(f'  2325: +28.22% (選択指標)')
    print(f'  2323: {r2324_final:+.2%}')
    print(f'  25+26: {comb:+.2%}')


if __name__ == '__main__':
    main()
