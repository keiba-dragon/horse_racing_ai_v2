# coding: utf-8
"""
save_shiba_short_nv1.py - 芝短距離 nv1 モデル保存
  特徴量: 1走前_3角, 芝ダ転向, 距離変化_前走, 1走前_脚質_num (4特徴, L2=0.006)
  セグメント: 芝 1200m+1400m (千直1000m除外)
  roi_model.pkl に '芝短' artifact として追加。
  '芝' (中長距離用旧320特徴) / 'ダ' (ダ中長距離BASE_25) は維持。
  25+26 OOS ROI: +14.85%  ベースライン比: +40.39pp (旧-26.54%)
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

FEATS = ['1走前_3角', '芝ダ転向', '距離変化_前走', '1走前_脚質_num']
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
    df = df[(df['surface'] == '芝') & (dm >= 1200) & (dm <= 1400)].copy()
    df['dist_m'] = dm[df.index]
    df = add_computed_features(df)
    baba_map = {'良': 0, '稍重': 1, '重': 2, '不良': 3}
    for col in df.columns:
        if '馬場状態' in col:
            df[col] = df[col].map(baba_map)
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
    print('  芝短距離 nv1 モデル保存')
    print(f'  特徴量: {FEATS}')
    print(f'  セグメント: 芝 1200m+1400m')
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

    valid_feats = [c for c in FEATS if c in df_trn.columns and
                   df_trn[c].isna().mean() <= 0.65]
    print(f'有効特徴量: {valid_feats}')

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
        if label == '2025':
            r25, n25 = r, n
        elif label == '2026':
            r26, n26 = r, n
    comb = (r25*n25 + r26*n26) / (n25+n26) if n25+n26 > 0 else float('nan')
    print(f'  25+26合算: ROI={comb:+.2%}')

    # roi_model.pkl に '芝短' artifact として追加
    final_pkl  = os.path.join(MODEL_DIR, 'roi_model.pkl')
    backup_path = os.path.join(MODEL_DIR, 'final_model_pre_shiba_short_nv1.pkl')
    print(f'\n既存モデル読み込み中: {final_pkl}')
    with open(final_pkl, 'rb') as f:
        existing_pkg = pickle.load(f)

    shutil.copy2(final_pkl, backup_path)
    print(f'バックアップ: {backup_path}')

    shiba_short_art = {
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
        'segment':       '芝短 1200m+1400m',
        'oos_roi_2526':  comb,
    }

    new_artifacts = dict(existing_pkg['artifacts'])
    new_artifacts['芝短'] = shiba_short_art

    new_pkg = {
        'artifacts':     new_artifacts,
        'feat_cols':     existing_pkg.get('feat_cols', valid_feats),
        'factor_maiden': existing_pkg.get('factor_maiden', 0.00),
        'factor_other':  existing_pkg.get('factor_other',  0.16),
        'total_oos_roi': existing_pkg.get('total_oos_roi', -0.1718),
        'note':          existing_pkg.get('note', '') + ' | 芝短nv1追加',
        'version':       'shiba_short_nv1',
    }

    with open(final_pkl, 'wb') as f:
        pickle.dump(new_pkg, f)

    print(f'\n保存完了: {final_pkl}')
    print(f'artifacts keys: {list(new_pkg["artifacts"].keys())}')
    print()
    print('=== 芝短距離 nv1 保存完了 ===')
    print(f'  芝短 artifact: nv1 (4特徴, L2=0.006, isotonic calibrated)')
    print(f'  25+26 OOS ROI: {comb:+.2%}  (ベースライン比 +40.39pp)')
    print(f'  ※ 予測コード (06_predict_from_card.py) の距離分岐は')
    print(f'    5セグメント全揃い後に一括更新予定')


if __name__ == '__main__':
    main()
