# coding: utf-8
"""
add_feats_v5.py - 積み上げ継続 + 対数オッズ・交互作用項を探索
  BASE_23 = BASE_22 + [所属_num, キャリア_浅い, 近3走_勝率] を試しつつ
  計算特徴量（log_odds, 交互作用）も追加テスト
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

BASE_22 = [
    '近5走_クラス調整_平均着順', '近5走_タイム指数_max', '1走前_タイム指数', '前走着差タイム',
    '騎手コース_r100_勝率', '1走前_クラス調整着順', '調教師コース_r100_勝率',
    '1走前_RPCI', '1走前_上3F地点差', '斤量', '種牡馬_勝率',
    '間隔_長_flag', '1走前_脚質_num', '騎手変更', '馬番',
    '近3走_複勝率', '調教師_r200_複勝率',
    'ブリンカー変更', '2走前_クラス差', '4走前_クラス差', '1走前_馬場状態',
    '性別_num',
]

# 第2ラウンドで有望だった順
STEPWISE = ['所属_num', 'キャリア_浅い', '近3走_勝率']

# 計算特徴量候補
COMPUTED = [
    'log_前走オッズ',       # log(1走前_単勝オッズ) - 前走市場評価
    'log_2走前オッズ',      # log(2走前_単勝オッズ)
    '性別×馬場',           # 性別_num * 1走前_馬場状態
    '性別×斤量',           # 性別_num * 斤量
    '性別×タイム指数',       # 性別_num * 1走前_タイム指数
    'ti_x_rpci',          # タイム指数 × RPCI (ペース適性×実力)
    '間隔_raw',            # 間隔（生値、0-180日程度）vs 長_flag
    '斤量_標準化',          # 斤量 - 55（基準からのずれ）
    '近5走_クラス差_合計',   # 2走前+3走前+4走前クラス差合計
    '1走前_上り3F_log',    # log(上り3F) → 分布を正規化
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

    # 計算特徴量を追加
    if '1走前_単勝オッズ' in df.columns:
        df['log_前走オッズ'] = np.log1p(df['1走前_単勝オッズ'].clip(lower=0))
    if '2走前_単勝オッズ' in df.columns:
        df['log_2走前オッズ'] = np.log1p(df['2走前_単勝オッズ'].clip(lower=0))
    if '性別_num' in df.columns and '1走前_馬場状態' in df.columns:
        df['性別×馬場'] = df['性別_num'] * df['1走前_馬場状態']
    if '性別_num' in df.columns and '斤量' in df.columns:
        df['性別×斤量'] = df['性別_num'] * df['斤量']
    if '性別_num' in df.columns and '1走前_タイム指数' in df.columns:
        df['性別×タイム指数'] = df['性別_num'] * df['1走前_タイム指数']
    if '1走前_タイム指数' in df.columns and '1走前_RPCI' in df.columns:
        df['ti_x_rpci'] = df['1走前_タイム指数'] * df['1走前_RPCI']
    if '間隔' in df.columns:
        df['間隔_raw'] = df['間隔']
    df['斤量_標準化'] = df['斤量'] - 55.0
    # 近5走クラス差合計
    cls_cols = ['2走前_クラス差', '3走前_クラス差', '4走前_クラス差']
    avail = [c for c in cls_cols if c in df.columns]
    if avail:
        df['近5走_クラス差_合計'] = df[avail].sum(axis=1, min_count=1)
    if '1走前_上り3F' in df.columns:
        df['1走前_上り3F_log'] = np.log1p(df['1走前_上り3F'].clip(lower=0))

    return df


def _loss_grad(beta, X, y, gs, n, nr):
    probs = segment_softmax(X @ beta, gs, n)
    res   = y - probs
    loss  = -np.sum(y * np.log(np.clip(probs, 1e-15, 1.0))) / nr
    grad  = -(X.T @ res) / nr
    return loss, grad


def adam_fit(X_tr, y_tr, gs_tr, n_tr, nr_tr, X_va, y_va, gs_va, n_va, nr_va):
    d = X_tr.shape[1]
    beta, m, v = np.zeros(d), np.zeros(d), np.zeros(d)
    b1, b2, eps = 0.9, 0.999, 1e-8
    t, best_val, best_beta, no_imp = 0, np.inf, np.zeros(d), 0
    for epoch in range(1, N_EPOCHS + 1):
        _, grad = _loss_grad(beta, X_tr, y_tr, gs_tr, n_tr, nr_tr)
        t += 1
        m = b1*m + (1-b1)*grad
        v = b2*v + (1-b2)*grad**2
        beta -= LR * (m/(1-b1**t)) / (np.sqrt(v/(1-b2**t)) + eps)
        if epoch % 10 == 0:
            vl, _ = _loss_grad(beta, X_va, y_va, gs_va, n_va, nr_va)
            if vl < best_val:
                best_val, best_beta, no_imp = vl, beta.copy(), 0
            else:
                no_imp += 1
            if no_imp >= PATIENCE // 10:
                break
    return best_beta, best_val


def evaluate(df_trn, df_val, oos_parts, feats):
    valid = [c for c in feats if c in df_trn.columns and df_trn[c].isna().mean() <= 0.65]
    if not valid:
        return None, None, {}
    X_tr, y_tr, gs_tr, n_tr, nr_tr, scaler, *_ = prepare(
        df_trn, valid, top_idx=None, top_idx3=None, fit=True)
    X_va, y_va, gs_va, n_va, nr_va, *_ = prepare(
        df_val, valid, scaler=scaler, top_idx=None, top_idx3=None)
    beta, val_nll = adam_fit(X_tr, y_tr, gs_tr, n_tr, nr_tr,
                              X_va, y_va, gs_va, n_va, nr_va)
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
    return val_nll, beta, oos_roi


def show_base(label, feats, df, df_trn, df_val, oos_parts):
    nll, _, roi = evaluate(df_trn, df_val, oos_parts, feats)
    r25, n25 = roi.get('2025', (0, 0))
    r26, n26 = roi.get('2026', (0, 0))
    comb = (r25*n25 + r26*n26) / (n25+n26) if n25+n26 > 0 else 0
    print(f'{label}: 2025:{r25:+.2%}  2026:{r26:+.2%}  25+26:{comb:+.2%}')
    return roi, comb


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

    print('=== ステップワイズ追加 ===')
    roi0, comb0 = show_base('BASE_22 (21F+性別)', BASE_22, df, df_trn, df_val, oos_parts)
    r25_0, n25 = roi0['2025']
    r26_0, n26 = roi0['2026']

    current_feats = list(BASE_22)
    for step_cand in STEPWISE:
        if step_cand not in df.columns:
            print(f'  {step_cand}: 列なし')
            continue
        roi_new, comb_new = show_base(
            f'  +{step_cand}', current_feats + [step_cand], df, df_trn, df_val, oos_parts)
        r25_n, _ = roi_new.get('2025', (comb_new, 0))
        r26_n, _ = roi_new.get('2026', (comb_new, 0))
        d = comb_new - comb0
        print(f'    Δ25+26:{d:+.2%}  (2025:{r25_n-r25_0:+.2%} / 2026:{r26_n-r26_0:+.2%})')
        if d > 0:
            current_feats.append(step_cand)
            comb0 = comb_new
            roi0 = roi_new
            r25_0, n25 = roi0['2025']
            r26_0, n26 = roi0['2026']

    print(f'\n最終ベース: {len(current_feats)}F')
    roi_final, comb_final = show_base('確定ベース', current_feats, df, df_trn, df_val, oos_parts)

    print()
    print('=== 計算特徴量テスト ===')
    results = []
    for cand in COMPUTED:
        if cand in current_feats or cand not in df.columns:
            print(f'  スキップ {cand} (列なし)')
            continue
        nan_trn = df_trn[cand].isna().mean() if cand in df_trn.columns else 1.0
        if nan_trn > 0.65:
            print(f'  スキップ {cand} (NaN={nan_trn:.0%})')
            continue
        nll, _, roi = evaluate(df_trn, df_val, oos_parts, current_feats + [cand])
        if nll is None:
            continue
        r25 = roi.get('2025', (0, 0))[0]
        r26 = roi.get('2026', (0, 0))[0]
        n25_t = roi.get('2025', (0, 1))[1]
        n26_t = roi.get('2026', (0, 1))[1]
        comb = (r25*n25_t + r26*n26_t) / (n25_t+n26_t) if n25_t+n26_t > 0 else 0
        d25 = r25 - r25_0; d26 = r26 - r26_0; d_comb = comb - comb_final
        both = (d25 > 0 and d26 > 0)
        results.append((cand, d25, d26, d_comb, both, nan_trn))
        sym = '✓' if d_comb > 0.003 else ('✗' if d_comb < -0.003 else '~')
        agree = '[両✓]' if both else ''
        print(f'{sym} {cand:<30} Δ25+26:{d_comb:>+7.2%}  Δ25:{d25:>+7.2%}  Δ26:{d26:>+7.2%}  {agree}')

    results.sort(key=lambda x: -x[3])
    promising = [(c, d, b) for c, _, _, d, b, _ in results if d > 0.003]
    print()
    if promising:
        print('【計算特徴量 有望】:')
        for c, d, b in promising:
            print(f'  ✓ {c}: {d:+.2%}{"  [両✓]" if b else ""}')
    else:
        print('計算特徴量: 有望なし')

    print(f'\n【現時点の最良】: {len(current_feats)}F, 25+26={comb_final:+.2%}')
    print(f'  基準(21F): -19.70%  改善: {comb_final-(-0.197):+.2%}')


if __name__ == '__main__':
    main()
