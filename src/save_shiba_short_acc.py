# coding: utf-8
"""
save_shiba_short_acc.py - 芝短距離 的中率最大化モデル保存

  v1ベスト30特徴  acc_2325=26.38%  ランダム比3.901x
  セグメント: 芝 & ≤1400m & クラス_rank≠1.0  1番人気28.69%
  保存先: models/hitrate_model.pkl  (roi_model.pkl 上書き禁止)
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
from save_v3 import add_computed_features

MODEL_DIR = os.path.join(BASE_DIR, 'models')

FEATS = [
    '馬番', '斤量', '芝ダ一致_平均着順_近5走', '1走前_タイム指数',
    '近5走_クラス調整_平均着順', '馬コース_r20_勝率', '騎手コース_r100_勝率',
    '近10走_勝率', '馬体重', '近5走_上り3F平均', '芝ダ転向',
    '近3走_体重増減合計', '相手レベル_平均着順', '道悪_平均着順_近5走',
    'タイム指数_近3走_slope', '馬体重増減', 'コース馬場_r200_勝率',
    'ブリンカー変更', '3走前_クラス差', '近5走_上り3F_std',
    '騎手コース距離_r100_勝率', '2走前_クラス差', '着順_近3走_slope',
    '近3走_複勝率', '馬距離_勝率', '1走前_3角', '1走前_馬場状態',
    'コース枠_r200_勝率', '前走着差タイム', '馬_r20_勝率',
]
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
    df['クラス_rank'] = pd.to_numeric(df['クラス_rank'], errors='coerce')
    df = df[(df['surface'] == '芝') & (dm <= 1400) & (df['クラス_rank'] != 1.0)].copy()
    df['dist_m'] = dm[df.index]
    df = add_computed_features(df)

    if '今回_会場' in df.columns and '1走前_開催' in df.columns:
        prev_venue = df['1走前_開催'].astype(str).str[1]
        df['輸送有無'] = (df['今回_会場'].astype(str) != prev_venue).astype(float)
        df.loc[df['1走前_開催'].isna(), '輸送有無'] = float('nan')

    baba_map = {'良': 0, '稍重': 1, '重': 2, '不良': 3}
    for col in df.columns:
        if '馬場状態' in col and col != '馬場状態':
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


def acc_from_top1(scored):
    top1 = scored[scored['rank'] == 1]
    if len(top1) == 0:
        return float('nan'), 0
    return (top1['着順_num'] == 1).mean(), len(top1)


def roi_from_top1(top1):
    won  = top1['着順_num'] == 1
    odds = pd.to_numeric(top1['単勝オッズ'], errors='coerce')
    if len(top1) == 0:
        return float('nan'), 0
    return (odds[won] * 100).sum() / (len(top1) * 100) - 1, len(top1)


def main():
    print('=' * 70)
    print('  芝短距離 的中率モデル保存 (v1: 30特徴, acc_2325=26.38%)')
    print(f'  特徴量({len(FEATS)}個)')
    print('  保存先: models/hitrate_model.pkl  (roi_model.pklは変更しない)')
    print('=' * 70)

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
    print(f'有効特徴量({len(valid_feats)}個)')

    print('\n学習中 (L2=0.006, Adam)...')
    X_tr, y_tr, gs_tr, n_tr, nr_tr, scaler, *_ = prepare(
        df_trn, valid_feats, top_idx=None, top_idx3=None, fit=True)
    X_va, y_va, gs_va, n_va, nr_va, *_ = prepare(
        df_val, valid_feats, scaler=scaler, top_idx=None, top_idx3=None)
    beta = adam_fit(X_tr, y_tr, gs_tr, n_tr, nr_tr,
                    X_va, y_va, gs_va, n_va, nr_va)
    print('学習完了')

    print('\nIsotonic calibration...')
    val_sorted = df_val.sort_values('race_id').reset_index(drop=True)
    raw_val = segment_softmax(X_va @ beta, gs_va, n_va)
    y_val   = (val_sorted['着順_num'] == 1).astype(float).values
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(raw_val, y_val)

    print('\n=== OOS 的中率・ROI 確認 ===')
    results = {}
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
        acc, n_r = acc_from_top1(scored)
        roi, _ = roi_from_top1(top1)
        results[label] = (acc, roi, n_r)
        print(f'  {label}: acc={acc:.2%}  ROI={roi:+.2%}  ({n_r}R)')

    n25, n26 = results.get('2025', (0,0,0))[2], results.get('2026', (0,0,0))[2]
    a25, a26 = results.get('2025', (0,0,0))[0], results.get('2026', (0,0,0))[0]
    r25, r26 = results.get('2025', (0,0,0))[1], results.get('2026', (0,0,0))[1]
    a2324 = results.get('2324', (0,0,0))[0]
    n2324 = results.get('2324', (0,0,0))[2]
    acc_2325 = (a2324*n2324 + a25*n25) / (n2324+n25) if (n2324+n25) > 0 else float('nan')
    acc_2526 = (a25*n25 + a26*n26) / (n25+n26) if (n25+n26) > 0 else float('nan')
    roi_2526 = (r25*n25 + r26*n26) / (n25+n26) if (n25+n26) > 0 else float('nan')
    print(f'  acc_2325={acc_2325:.4f}  25+26_acc={acc_2526:.4f}  25+26_ROI={roi_2526:+.2%}')

    acc_pkg = {
        'segment': '芝短 ≤1400m クラス_rank≠1.0',
        'scaler': scaler,
        'coef': beta,
        'feat_cols': valid_feats,
        'isotonic': iso,
        'acc_2325': acc_2325,
        'acc_2526': acc_2526,
        'oos_roi_2526': roi_2526,
        'version': 'shiba_short_acc_v1',
        'note': f'v1: 30特徴 acc_2325={acc_2325:.4f} ランダム比3.901x 1番人気28.69%',
    }

    acc_pkl = os.path.join(MODEL_DIR, 'hitrate_model.pkl')
    if os.path.exists(acc_pkl):
        existing = pickle.load(open(acc_pkl, 'rb'))
        existing['芝短'] = acc_pkg
        with open(acc_pkl, 'wb') as f:
            pickle.dump(existing, f)
        print(f'\n更新: {acc_pkl}  (芝短を追加)')
    else:
        with open(acc_pkl, 'wb') as f:
            pickle.dump({'芝短': acc_pkg}, f)
        print(f'\n新規作成: {acc_pkl}')

    print(f'=== 保存完了 ===')
    print(f'  acc_2325 = {acc_2325:.4f} ({acc_2325:.2%})')
    print(f'  25+26 acc = {acc_2526:.4f} ({acc_2526:.2%})')


if __name__ == '__main__':
    main()
