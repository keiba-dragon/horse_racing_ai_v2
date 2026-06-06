# coding: utf-8
"""
report_shiba_nv1_no1000.py - 芝短距離 nv1 再評価（1000m除外: 1200m+1400mのみ）
特徴量: 1走前_3角, 芝ダ転向, 距離変化_前走, 1走前_脚質_num
"""
import sys, os
import numpy as np
import pandas as pd

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from save_conditional_logit import (
    prepare, segment_softmax, BASE_DIR, DATA_FILE, LR, N_EPOCHS, PATIENCE
)
from save_v3 import add_computed_features

L2 = 0.006
FEATS = ['1走前_3角', '芝ダ転向', '距離変化_前走', '1走前_脚質_num']


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
    # 1000m除外: 1200m + 1400mのみ
    df = df[(df['surface'] == '芝') & (dm >= 1200) & (dm <= 1400)].copy()
    df['dist_m'] = dm[df.index]
    df = add_computed_features(df)
    baba_map = {'良': 0, '稍重': 1, '重': 2, '不良': 3}
    for col in df.columns:
        if '馬場状態' in col:
            df[col] = df[col].map(baba_map)
    for col in FEATS:
        if col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception:
                df[col] = np.nan
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


def score_oos(df, feats, scaler, beta):
    valid_p = [c for c in feats if c in df.columns]
    X_p, _, gs_p, n_p, *_ = prepare(df, valid_p, scaler=scaler,
                                      top_idx=None, top_idx3=None)
    scored = df.sort_values('race_id').reset_index(drop=True)
    scored['prob'] = segment_softmax(X_p @ beta, gs_p, n_p)
    scored['rank'] = scored.groupby('race_id')['prob'].rank(ascending=False, method='first')
    return scored


def main():
    print("=" * 65)
    print("  芝短距離 nv1 再評価（1000m除外: 1200m+1400mのみ）")
    print(f"  特徴量: {FEATS}")
    print("=" * 65)

    df = load_segment()
    df_trn   = df[(df['日付_num'] >= 130101) & (df['日付_num'] < 220101)]
    df_val   = df[(df['日付_num'] >= 220101) & (df['日付_num'] <= 221231)]
    oos_2324 = df[(df['日付_num'] >= 230101) & (df['日付_num'] < 250101)]
    oos_2025 = df[(df['日付_num'] >= 250101) & (df['日付_num'] < 260101)]
    oos_2026 = df[df['日付_num'] >= 260101]

    print(f"\nデータ件数:")
    print(f"  train: {len(df_trn):,}行  val: {len(df_val):,}行")
    print(f"  2324:  {len(oos_2324):,}行({oos_2324['race_id'].nunique()}R)  "
          f"2025: {len(oos_2025):,}行({oos_2025['race_id'].nunique()}R)  "
          f"2026: {len(oos_2026):,}行({oos_2026['race_id'].nunique()}R)")

    # 距離構成確認
    print(f"\n距離構成 (全データ):")
    dist_counts = df['dist_m'].value_counts().sort_index()
    for d, n in dist_counts.items():
        print(f"  {int(d)}m: {n:,}行")

    # 学習
    valid = [c for c in FEATS if c in df_trn.columns and df_trn[c].isna().mean() <= 0.65]
    print(f"\n学習中... ({len(valid)}特徴)")
    X_tr, y_tr, gs_tr, n_tr, nr_tr, scaler, *_ = prepare(
        df_trn, valid, top_idx=None, top_idx3=None, fit=True)
    X_va, y_va, gs_va, n_va, nr_va, *_ = prepare(
        df_val, valid, scaler=scaler, top_idx=None, top_idx3=None)
    beta = adam_fit(X_tr, y_tr, gs_tr, n_tr, nr_tr,
                    X_va, y_va, gs_va, n_va, nr_va)

    print(f"\n係数 (β):")
    for f, b in zip(valid, beta):
        print(f"  {f}: {b:+.4f}")

    # OOS評価
    all_oos = [('2324', oos_2324), ('2025', oos_2025), ('2026', oos_2026)]
    oos_results = {}
    for name, oos in all_oos:
        if len(oos) == 0:
            continue
        scored = score_oos(oos, valid, scaler, beta)
        top1 = scored[scored['rank'] == 1]
        r, n = roi_from_top1(top1)
        oos_results[name] = (r, n, scored, top1)

    print(f"\n{'─'*65}")
    print("  OOS ROI 年度別")
    print(f"{'─'*65}")
    r2324, n2324 = oos_results.get('2324', (float('nan'), 0))[:2]
    r25,   n25   = oos_results.get('2025', (float('nan'), 0))[:2]
    r26,   n26   = oos_results.get('2026', (float('nan'), 0))[:2]
    print(f"  2324: {r2324*100:.2f}%  ({n2324}R)")
    print(f"  2025: {r25*100:.2f}%  ({n25}R)")
    print(f"  2026: {r26*100:.2f}%  ({n26}R)")
    if n25 + n26 > 0:
        rcomb = (r25*n25 + r26*n26) / (n25+n26)
        print(f"  25+26: {rcomb*100:.2f}%  ({n25+n26}R)")
        print(f"  1000m除外前比較: {rcomb*100:.2f}% vs +14.34% (1000m込み)")

    # 距離別 ROI (2324)
    print(f"\n{'─'*65}")
    print("  距離別 ROI (2324)")
    print(f"{'─'*65}")
    if '2324' in oos_results:
        top1_2324 = oos_results['2324'][3]
        for dist in [1200, 1400]:
            sub = top1_2324[top1_2324['dist_m'] == dist]
            if len(sub) > 0:
                r, n = roi_from_top1(sub)
                print(f"  {dist}m: {r*100:.2f}%  ({n}R)")

    # 2026月次
    print(f"\n{'─'*65}")
    print("  2026 月次ROI")
    print(f"{'─'*65}")
    if '2026' in oos_results:
        top1_2026 = oos_results['2026'][3].copy()
        top1_2026['month'] = (top1_2026['日付_num'].astype(int) // 100) % 100
        for m in sorted(top1_2026['month'].unique()):
            sub = top1_2026[top1_2026['month'] == m]
            r, n = roi_from_top1(sub)
            print(f"  2026-{m:02d}: {r*100:.2f}%  ({n}R)")

    # クラス_rank別 ROI (2324)
    print(f"\n{'─'*65}")
    print("  クラス_rank別 ROI (2324)")
    print(f"{'─'*65}")
    if '2324' in oos_results:
        top1_2324 = oos_results['2324'][3]
        if 'クラス_rank' in top1_2324.columns:
            cr = pd.to_numeric(top1_2324['クラス_rank'], errors='coerce')
            for cls in sorted(cr.dropna().unique()):
                sub = top1_2324[cr == cls]
                if len(sub) >= 10:
                    r, n = roi_from_top1(sub)
                    print(f"  クラス_rank={cls:.0f}: {r*100:.2f}%  ({n}R)")


if __name__ == '__main__':
    main()
