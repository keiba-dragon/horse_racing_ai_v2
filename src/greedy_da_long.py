# coding: utf-8
"""
greedy_da_long.py - ダ中長距離 greedy forward feature selection
  選択指標: 2323 OOS ROI (2023-24合算)
  セグメント: ダ & >1400m & クラス_rank≠1.0
  市場相関 < 0.25 の特徴量プールから greedy forward selection
  最大 MAX_FEATS 個まで追加。改善が MIN_IMPROVE 未満で停止。
"""
import sys, os, io, pickle, time
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from save_conditional_logit import (
    prepare, segment_softmax, BASE_DIR, DATA_FILE, LR, N_EPOCHS, PATIENCE
)
from save_v3 import add_computed_features

MAX_FEATS   = 14
MIN_IMPROVE = 0.001   # 0.1%以上改善しないと追加しない
L2          = 0.006

BABA_MAP = {'良': 0, '稍': 1, '稍重': 1, '重': 2, '不': 3, '不良': 3}

# 初期シードセット（市場低相関で使いやすいもの）
SEED_FEATS = [
    'コース枠_r200_勝率',
    '馬距離_勝率',
    'レース内_逃げ馬数',
    '今回_馬場_num',
    '内外枠',
    '斤量',
    '馬体重増減',
    '相手レベル_実力差',
    'PCI3',
    'ブリンカー変更',
    'キャリア_浅い',
    '格上経験数_近5走',
    '近5走_タイム指数_range',
    '1走前_クラス差',
    '間隔_長_flag',
    'クラス_rank',
    '着順_加速度',
    'タイム指数_加速度',
    '馬番',
    '性別_num',
    '月',
    '季節',
    'レース内_先行馬数',
    'レース内_脚質std',
    '前走馬番',
    '馬体重',
    '1走前_馬番',
    '1走前_脚質_num',
    '1走前_上り差',
    '1走前_RPCI',
    '2走前_クラス差',
    '3走前_クラス差',
    '同会場_平均着順_近5走',
    '近5走_上り3F平均',
    '騎手コース_r100_勝率',
    '騎手変更',
    '所属_num',
    '種牡馬_勝率',
    '種牡馬_複勝率',
    '馬コース_r200_複勝率',
    'コース枠_r200_複勝率',
    '同距離_勝率',
    '同馬場_勝率',
    '近3走_上り3F平均',
    '1走前_間隔',
    '2走前_間隔',
    '3走前_斤量',
    '2走前_斤量',
    '1走前_着順_num',
    '2走前_着順_num',
    '前走_上り3F',
    '相手レベル_平均着順',
    'タイム指数_近5走_slope',
]


def load_segment():
    print('データ読み込み中...')
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
    kr = pd.to_numeric(df.get('クラス_rank', pd.Series(dtype=float)), errors='coerce')
    df = df[(df['surface'] == 'ダ') & (dm > 1400) & (kr != 1.0)].copy()
    df['dist_m'] = dm[df.index]
    df = add_computed_features(df)

    # 間隔_長_flag (動的生成)
    interval = pd.to_numeric(df['間隔'], errors='coerce')
    df['間隔_長_flag'] = (interval >= 60).astype(float)

    # 馬場状態エンコード
    for col in df.columns:
        if '馬場状態' in col:
            enc = df[col].map(BABA_MAP)
            if enc.notna().any():
                df[col] = enc

    print(f'  全行数: {len(df):,}  レース数: {df["race_id"].nunique():,}')
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


def roi_from_top1_raw(df_oos, feats, scaler, beta):
    """raw softmax でランク付けして 2323 ROI を返す"""
    valid = [c for c in feats if c in df_oos.columns]
    if not valid:
        return float('nan'), 0
    X_p, _, gs_p, n_p, *_ = prepare(df_oos, valid, scaler=scaler,
                                      top_idx=None, top_idx3=None)
    scored = df_oos.sort_values('race_id').reset_index(drop=True)
    probs  = segment_softmax(X_p @ beta, gs_p, n_p)
    rank   = pd.Series(probs).groupby(
        scored['race_id'].values).rank(ascending=False, method='first')
    top1 = scored[rank.values == 1]
    won  = top1['着順_num'] == 1
    odds = pd.to_numeric(top1['単勝オッズ'], errors='coerce')
    if len(top1) == 0:
        return float('nan'), 0
    return (odds[won] * 100).sum() / (len(top1) * 100) - 1, len(top1)


def eval_feats(feats, df_trn, df_val, df_2324):
    """特徴量セットを評価して 2323 ROI を返す"""
    valid = [c for c in feats if c in df_trn.columns and
             df_trn[c].isna().mean() <= 0.65]
    if not valid:
        return float('nan'), None, None

    X_tr, y_tr, gs_tr, n_tr, nr_tr, scaler, *_ = prepare(
        df_trn, valid, top_idx=None, top_idx3=None, fit=True)
    X_va, y_va, gs_va, n_va, nr_va, *_ = prepare(
        df_val, valid, scaler=scaler, top_idx=None, top_idx3=None)
    beta = adam_fit(X_tr, y_tr, gs_tr, n_tr, nr_tr,
                    X_va, y_va, gs_va, n_va, nr_va)

    roi, n = roi_from_top1_raw(df_2324, valid, scaler, beta)
    return roi, scaler, beta


def main():
    t_start = time.time()
    df = load_segment()

    df_trn   = df[(df['日付_num'] >= 130101) & (df['日付_num'] < 220101)]
    df_val   = df[(df['日付_num'] >= 220101) & (df['日付_num'] <= 221231)]
    df_2324  = df[(df['日付_num'] >= 230101) & (df['日付_num'] <= 241231)]
    df_2025  = df[(df['日付_num'] >= 250101) & (df['日付_num'] <= 251231)]
    df_2026  = df[df['日付_num'] >= 260101]

    print(f'train: {len(df_trn):,}  val: {len(df_val):,}'
          f'  2324: {df_2324["race_id"].nunique()}R'
          f'  2025: {df_2025["race_id"].nunique()}R'
          f'  2026: {df_2026["race_id"].nunique()}R')

    # 使用可能な候補のみに絞る（存在確認 + NaN率）
    pool = []
    for c in SEED_FEATS:
        if c not in df.columns:
            continue
        nan_r = df_trn[c].isna().mean()
        if nan_r > 0.65:
            continue
        pool.append(c)
    print(f'\n候補プール: {len(pool)}個')

    # ── Greedy Forward Selection ─────────────────────────────────────────
    selected = []
    best_roi = float('-inf')

    print('\n' + '='*72)
    print('  Greedy Forward Selection (指標: 2323 OOS ROI)')
    print('='*72)

    for step in range(MAX_FEATS):
        candidates = [c for c in pool if c not in selected]
        if not candidates:
            break

        step_results = []
        print(f'\nStep {step+1}: {len(candidates)}個を評価中...')

        for i, cand in enumerate(candidates):
            trial = selected + [cand]
            roi, _, _ = eval_feats(trial, df_trn, df_val, df_2324)
            step_results.append((cand, roi))
            if (i+1) % 10 == 0:
                print(f'  {i+1}/{len(candidates)} 完了...')

        step_results.sort(key=lambda x: -x[1] if not np.isnan(x[1]) else float('-inf'))
        best_cand, best_cand_roi = step_results[0]

        improve = best_cand_roi - best_roi

        # 上位5個表示
        print(f'  上位5候補:')
        for c, r in step_results[:5]:
            mark = '★' if c == best_cand else '  '
            print(f'  {mark} {c:<45} 2323ROI={r:+.3%}  (改善={r-best_roi:+.3%})')

        if improve < MIN_IMPROVE:
            print(f'\n→ 改善 {improve:+.3%} < 閾値 {MIN_IMPROVE:.1%} → 終了')
            break

        selected.append(best_cand)
        best_roi = best_cand_roi
        print(f'\n→ [{step+1}] 追加: {best_cand}  (2323ROI={best_roi:+.3%})')

    # ── 最終モデルの詳細評価 ──────────────────────────────────────────
    print('\n' + '='*72)
    print(f'  最終特徴量セット ({len(selected)}個): {selected}')
    print('='*72)

    valid = [c for c in selected if c in df_trn.columns]
    X_tr, y_tr, gs_tr, n_tr, nr_tr, scaler, *_ = prepare(
        df_trn, valid, top_idx=None, top_idx3=None, fit=True)
    X_va, y_va, gs_va, n_va, nr_va, *_ = prepare(
        df_val, valid, scaler=scaler, top_idx=None, top_idx3=None)
    beta = adam_fit(X_tr, y_tr, gs_tr, n_tr, nr_tr,
                    X_va, y_va, gs_va, n_va, nr_va)

    print('\n係数:')
    for f, b in zip(valid, beta):
        print(f'  {f:<45} β={b:+.4f}')

    iso = IsotonicRegression(out_of_bounds='clip')
    val_sorted = df_val.sort_values('race_id').reset_index(drop=True)
    raw_val = segment_softmax(X_va @ beta, gs_va, n_va)
    y_val   = (val_sorted['着順_num'] == 1).astype(float).values
    iso.fit(raw_val, y_val)

    print('\n=== OOS ROI ===')
    r23, n23 = roi_from_top1_raw(df_2324, valid, scaler, beta)
    r25, n25 = roi_from_top1_raw(df_2025, valid, scaler, beta)
    r26, n26 = roi_from_top1_raw(df_2026, valid, scaler, beta)
    print(f'  2323: {r23:+.3%} ({n23}R)')
    print(f'  2025: {r25:+.3%} ({n25}R)')
    print(f'  2026: {r26:+.3%} ({n26}R)')
    comb = (r25*n25 + r26*n26)/(n25+n26) if n25+n26>0 else float('nan')
    print(f'  25+26合算: {comb:+.3%}')
    print(f'\n所要時間: {(time.time()-t_start)/60:.1f}分')
    print('\n[BEST_FEATS]', selected)


if __name__ == '__main__':
    main()
