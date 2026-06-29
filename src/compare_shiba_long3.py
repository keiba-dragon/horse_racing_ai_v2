# coding: utf-8
"""
compare_shiba_long3.py - 芝長距離（芝2001m以上）Round 3
* R1/R2: 全18セットがベースライン(-16.90%)以下
* より単純/異なる特徴でラストチャレンジ
* 「旧芝artifactを維持」判断のため実際のfilteredデータでの旧モデルROIも確認
"""
import sys, os, time, pickle
import numpy as np
import pandas as pd

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from save_conditional_logit import prepare, segment_softmax, BASE_DIR, DATA_FILE, LR, N_EPOCHS, PATIENCE
from save_v3 import add_computed_features

L2 = 0.006
MODEL_DIR = os.path.join(BASE_DIR, 'models')

ALL_FEATS = [
    '1走前_3角', '1走前_脚質_num', '前走着差タイム',
    '1走前_クラス調整着順', '1走前_クラス差', '2走前_クラス差',
    '馬距離_勝率', '騎手コース_r100_勝率', '調教師コース_r100_勝率',
    '間隔', '近5走_クラス調整_平均着順',
    '年齢', '休み明けフラグ', '連闘フラグ',
    '性別_num', '斤量', '芝ダ転向', '距離変化_前走',
]

SETS = {
    'T: 年齢+脚質+馬距離':          ['年齢', '1走前_脚質_num', '馬距離_勝率'],
    'U: 騎手+前走成績':             ['騎手コース_r100_勝率', '1走前_クラス調整着順'],
    'V: 調教師+騎手+脚質':          ['調教師コース_r100_勝率', '騎手コース_r100_勝率',
                                    '1走前_脚質_num'],
    'W: 休み明け+馬距離+脚質':       ['休み明けフラグ', '馬距離_勝率', '1走前_脚質_num'],
    'X: 前走+距離変化+クラス差':      ['前走着差タイム', '距離変化_前走', '1走前_クラス差'],
    'Y: 芝ダ転向+距離変化+脚質':     ['芝ダ転向', '距離変化_前走', '1走前_脚質_num'],
    'Z: 性別+馬距離+騎手':           ['性別_num', '馬距離_勝率', '騎手コース_r100_勝率'],
    'AA: 5走クラス補正+騎手':         ['近5走_クラス調整_平均着順', '騎手コース_r100_勝率'],
    'AB: クラス差のみ':              ['1走前_クラス差'],
    'AC: 前走着差のみ':              ['前走着差タイム'],
}


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
    if 'クラス_rank' in df.columns:
        cr = pd.to_numeric(df['クラス_rank'], errors='coerce')
        df = df[(df['surface'] == '芝') & (dm > 2000) & cr.notna()].copy()
    else:
        df = df[(df['surface'] == '芝') & (dm > 2000)].copy()
    df['dist_m'] = dm[df.index]
    df = add_computed_features(df)
    baba_map = {'良': 0, '稍重': 1, '重': 2, '不良': 3}
    for col in df.columns:
        if '馬場状態' in col:
            df[col] = df[col].map(baba_map)
    for col in ALL_FEATS:
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


def comb2526(r25, n25, r26, n26):
    if n25 + n26 == 0:
        return 0.0
    return (r25 * n25 + r26 * n26) / (n25 + n26)


def evaluate_set(df_trn, df_val, oos_2324, oos_2025, oos_2026, feats):
    valid = [c for c in feats if c in df_trn.columns and df_trn[c].isna().mean() <= 0.65]
    if not valid:
        return {k: (float('nan'), 0) for k in ['2324', '2025', '2026']}, valid, None
    X_tr, y_tr, gs_tr, n_tr, nr_tr, scaler, *_ = prepare(
        df_trn, valid, top_idx=None, top_idx3=None, fit=True)
    X_va, y_va, gs_va, n_va, nr_va, *_ = prepare(
        df_val, valid, scaler=scaler, top_idx=None, top_idx3=None)
    beta = adam_fit(X_tr, y_tr, gs_tr, n_tr, npr_tr := nr_tr, X_va, y_va, gs_va, n_va, nr_va)
    results = {}
    for label, oos in [('2324', oos_2324), ('2025', oos_2025), ('2026', oos_2026)]:
        if len(oos) == 0:
            results[label] = (float('nan'), 0)
            continue
        valid_p = [c for c in valid if c in oos.columns]
        X_p, _, gs_p, n_p, *_ = prepare(oos, valid_p, scaler=scaler,
                                          top_idx=None, top_idx3=None)
        scored = oos.sort_values('race_id').reset_index(drop=True)
        scored['prob'] = segment_softmax(X_p @ beta, gs_p, n_p)
        scored['rank'] = scored.groupby('race_id')['prob'].rank(ascending=False, method='first')
        top1 = scored[scored['rank'] == 1]
        results[label] = roi_from_top1(top1)
    return results, valid, beta


def baseline_old_model(df):
    """旧芝artifactで filteredデータ上のROIを確認"""
    with open(os.path.join(MODEL_DIR, 'roi_model.pkl'), 'rb') as f:
        pkg = pickle.load(f)
    art = pkg['artifacts']['芝']
    beta, scaler, feats = art['coef'], art['scaler'], art['feat_cols']
    poly2, inter_scaler2, top_idx = art.get('poly2'), art.get('inter_scaler2'), art.get('top_idx')
    poly3, inter_scaler3, top_idx3 = art.get('poly3'), art.get('inter_scaler3'), art.get('top_idx3')

    missing = [c for c in feats if c not in df.columns]
    for c in missing:
        df[c] = np.nan

    oos_2324 = df[(df['日付_num'] >= 230101) & (df['日付_num'] < 250101)]
    oos_2025 = df[(df['日付_num'] >= 250101) & (df['日付_num'] < 260101)]
    oos_2026 = df[df['日付_num'] >= 260101]

    results = {}
    for label, oos in [('2324', oos_2324), ('2025', oos_2025), ('2026', oos_2026)]:
        if len(oos) == 0:
            results[label] = (float('nan'), 0)
            continue
        X_p, _, gs_p, n_p, *_ = prepare(
            oos, feats, scaler=scaler,
            poly2=poly2, inter_scaler2=inter_scaler2, top_idx=top_idx,
            poly3=poly3, inter_scaler3=inter_scaler3, top_idx3=top_idx3)
        scored = oos.sort_values('race_id').reset_index(drop=True)
        scored['prob'] = segment_softmax(X_p @ beta, gs_p, n_p)
        scored['rank'] = scored.groupby('race_id')['prob'].rank(ascending=False, method='first')
        top1 = scored[scored['rank'] == 1]
        results[label] = roi_from_top1(top1)
    return results


def main():
    t0 = time.time()
    print("=" * 72)
    print("  芝長距離（芝2001m以上）Round 3")
    print("  R1/R2全18セットがbaseline(-16.90%)以下")
    print("  ※ 2324は参考のみ。25+26で優劣を判断する")
    print("=" * 72)

    df = load_segment()
    df_trn   = df[(df['日付_num'] >= 130101) & (df['日付_num'] < 220101)]
    df_val   = df[(df['日付_num'] >= 220101) & (df['日付_num'] <= 221231)]
    oos_2324 = df[(df['日付_num'] >= 230101) & (df['日付_num'] < 250101)]
    oos_2025 = df[(df['日付_num'] >= 250101) & (df['日付_num'] < 260101)]
    oos_2026 = df[df['日付_num'] >= 260101]

    print(f"\ntrain:{len(df_trn):,}行({df_trn['race_id'].nunique()}R)  "
          f"val:{len(df_val):,}行({df_val['race_id'].nunique()}R)")
    print(f"2324:{oos_2324['race_id'].nunique()}R  "
          f"2025:{oos_2025['race_id'].nunique()}R  "
          f"2026:{oos_2026['race_id'].nunique()}R")

    # 旧モデルでの実際ROI確認
    print("\n旧芝artifact(filteredデータ)での確認...")
    bl_res = baseline_old_model(df)
    r_bl25, n_bl25 = bl_res.get('2025', (float('nan'), 0))
    r_bl26, n_bl26 = bl_res.get('2026', (float('nan'), 0))
    r_bl2324, n_bl2324 = bl_res.get('2324', (float('nan'), 0))
    bl_comb = comb2526(r_bl25, n_bl25, r_bl26, n_bl26)
    print(f"  旧芝: 2324={r_bl2324*100:+.2f}% | 2025={r_bl25*100:+.2f}% | "
          f"2026={r_bl26*100:+.2f}% | 25+26={bl_comb*100:+.2f}%")
    print(f"  ← これが filtered の真のbaseline")

    print(f"\n{'='*72}")
    print(f"  {'セット':28s}  {'2324':>8}  {'2025':>8}  {'2026':>8}  {'25+26':>8}  特徴数")
    print(f"  {'-'*68}")

    best_comb, best_name = -999.0, None
    for name, feats in SETS.items():
        t1 = time.time()
        res, valid, beta = evaluate_set(df_trn, df_val, oos_2324, oos_2025, oos_2026, feats)
        r2324, n2324 = res['2324']
        r25, n25     = res['2025']
        r26, n26     = res['2026']
        rcomb = comb2526(r25, n25, r26, n26)
        marker = ' ←best' if rcomb > best_comb else ''
        if rcomb > best_comb:
            best_comb, best_name = rcomb, name
        print(f"  {name:28s}  {r2324*100:+7.2f}%  {r25*100:+7.2f}%  {r26*100:+7.2f}%  "
              f"{rcomb*100:+7.2f}%  {len(valid)}個  ({int(time.time()-t1)}s){marker}")
        if beta is not None:
            for f, b in zip(valid, beta):
                print(f"      β {f}: {b:+.4f}")

    print(f"\n{'='*72}")
    print(f"  Round3 ベスト: {best_name}  25+26={best_comb*100:.2f}%")
    print(f"  旧芝filtered比: {(best_comb - bl_comb)*100:+.2f}pp")
    print(f"  総時間: {int(time.time()-t0)}s")


if __name__ == '__main__':
    main()
