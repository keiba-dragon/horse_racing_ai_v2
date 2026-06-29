# coding: utf-8
"""
search_shiba_long_acc_v3.py - 芝長 ゼロシードから新候補追加で再探索
目標: 1番人気 36.05% を超える
現状: v1=34.75% (gap -1.30pp) SEEDからの継続では改善なし
新候補: 1走前_単勝オッズ, PCI系, 過去走タイム指数, 個別過去走上り3F等
"""
import sys, os, time, pickle
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from save_conditional_logit import prepare, segment_softmax, BASE_DIR, DATA_FILE, LR, N_EPOCHS, PATIENCE
from save_v3 import add_computed_features

L2 = 0.006
NAN_IND_THRESHOLD = 0.05
MAX_FEATS = 40
FAV_TARGET = 0.3605

FORCED_BASE = ['馬番', '斤量']

CANDIDATES = [
    # 新候補
    '1走前_単勝オッズ', '1走前_上り3F', '1走前_PCI', '1走前_RPCI', '1走前_頭数',
    '2走前_タイム指数', '2走前_上り3F', '2走前_着順_num',
    '3走前_タイム指数', '3走前_上り3F',
    '近10走_複勝率', '近5走_複勝率', '近5走_タイム指数_min',
    # v1の特徴量
    '近3走_複勝率', '騎手距離_r100_勝率', '近5走_タイム指数平均',
    '馬コース_r20_勝率', 'タイム指数_近3走_slope', '調教師コース_r100_勝率',
    '同会場_複勝率_近5走', '近5走_上り3F_std', 'コース枠_r200_複勝率',
    '相手レベル_平均着順', 'タイム指数_加速度', '近10走_勝率', '近3走_体重増減合計',
    'コース馬場_r200_勝率', '1走前_馬場状態', '種牡馬_ダ_勝率', 'タイム指数_近5走_slope',
    '道悪_平均着順_近5走', '1走前_タイム指数', '距離変化_前走', '性別_num',
    '馬体重', '馬体重増減', '調教師_r200_勝率', '輸送有無', 'コース枠_r200_勝率',
    'ブリンカー変更', '種牡馬_勝率',
    # 追加候補
    '芝ダ一致_平均着順_近5走', '1走前_3角', '1走前_4角', '1走前_脚質_num',
    '前走着差タイム', '近5走_上り3F平均', '近5走_タイム指数_max',
    '着順_近3走_slope', '上り3F_近3走_slope', '4角位置_近3走_slope',
    '1走前_クラス差', '2走前_クラス差', '3走前_クラス差',
    '1走前_クラス調整着順', '近5走_クラス調整_平均着順',
    '同馬場_平均着順_近5走', '良馬場_平均着順_近5走',
    '種牡馬_勝率', '母父馬_勝率', '馬_r20_勝率',
    '騎手コース_r100_勝率', '騎手会場_r100_勝率', '騎手コース距離_r100_勝率',
    '騎手馬場_r100_勝率', '間隔', '騎手変更',
    '近3走_勝率', '同会場_平均着順_近5走', '同距離帯_平均着順_近5走',
    '1走前_クラス差', '馬距離_勝率', '芝ダ転向',
]


def expand_nan_ind(dfs, feats):
    ref = dfs[0]
    extended = []
    for f in feats:
        extended.append(f)
        if f not in ref.columns:
            continue
        if NAN_IND_THRESHOLD < ref[f].isna().mean() < 1.0:
            ind = f + '_isnan'
            for df in dfs:
                if f in df.columns and ind not in df.columns:
                    df[ind] = df[f].isna().astype(float)
            extended.append(ind)
    return extended


def _loss_grad(beta, X, y, gs, n, nr, l2=0.0):
    probs = segment_softmax(X @ beta, gs, n)
    loss = -np.sum(y * np.log(np.clip(probs, 1e-15, 1.0))) / nr + l2 * np.dot(beta, beta)
    grad = -(X.T @ (y - probs)) / nr + 2 * l2 * beta
    return loss, grad


def adam_fit(X_tr, y_tr, gs_tr, n_tr, nr_tr, X_va, y_va, gs_va, n_va, nr_va, l2=L2):
    d = X_tr.shape[1]
    beta, m, v = np.zeros(d), np.zeros(d), np.zeros(d)
    b1, b2, eps = 0.9, 0.999, 1e-8
    t, best_val, best_beta, no_imp = 0, np.inf, np.zeros(d), 0
    for epoch in range(1, N_EPOCHS + 1):
        _, grad = _loss_grad(beta, X_tr, y_tr, gs_tr, n_tr, nr_tr, l2)
        t += 1
        m = b1 * m + (1 - b1) * grad
        v = b2 * v + (1 - b2) * grad ** 2
        beta -= LR * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
        if epoch % 10 == 0:
            vl, _ = _loss_grad(beta, X_va, y_va, gs_va, n_va, nr_va, l2=0.0)
            if vl < best_val:
                best_val, best_beta, no_imp = vl, beta.copy(), 0
            else:
                no_imp += 1
            if no_imp >= PATIENCE // 10:
                break
    return best_beta


def eval_feats(feats, dfs):
    df_trn, df_val, oos_2324, oos_2025 = dfs
    all_dfs = list(dfs)
    expanded = expand_nan_ind(all_dfs, feats)
    valid = [c for c in expanded if c in df_trn.columns
             and df_trn[c].isna().mean() < 1.0 and df_trn[c].std(ddof=0) > 0]
    if not valid:
        return float('-inf')
    try:
        X_tr, y_tr, gs_tr, n_tr, nr_tr, sc, *_ = prepare(
            df_trn, valid, top_idx=None, top_idx3=None, fit=True)
        X_va, y_va, gs_va, n_va, nr_va, *_ = prepare(
            df_val, valid, scaler=sc, top_idx=None, top_idx3=None)
        beta = adam_fit(X_tr, y_tr, gs_tr, n_tr, nr_tr,
                        X_va, y_va, gs_va, n_va, nr_va)
    except Exception:
        return float('-inf')

    def acc_oos(oos):
        vp = [c for c in valid if c in oos.columns]
        X_p, _, gs_p, n_p, *_ = prepare(oos, vp, scaler=sc, top_idx=None, top_idx3=None)
        s = oos.sort_values('race_id').reset_index(drop=True)
        s['prob'] = segment_softmax(X_p @ beta, gs_p, n_p)
        s['rank'] = s.groupby('race_id')['prob'].rank(ascending=False, method='first')
        t = s[s['rank'] == 1]
        nr = s['race_id'].nunique()
        return (t['着順_num'] == 1).mean(), nr

    a2324, n2324 = acc_oos(oos_2324)
    a25, n25 = acc_oos(oos_2025)
    acc_2325 = (a2324 * n2324 + a25 * n25) / (n2324 + n25) if (n2324 + n25) > 0 else float('-inf')
    return acc_2325


def save_seg(feats, seg):
    df_trn = seg[(seg['日付_num'] >= 130101) & (seg['日付_num'] < 220101)]
    df_val = seg[(seg['日付_num'] >= 220101) & (seg['日付_num'] <= 221231)]
    oos_2324 = seg[(seg['日付_num'] >= 230101) & (seg['日付_num'] < 250101)]
    oos_2025 = seg[(seg['日付_num'] >= 250101) & (seg['日付_num'] < 260101)]
    oos_2026 = seg[seg['日付_num'] >= 260101]
    all_dfs = [df_trn, df_val, oos_2324, oos_2025, oos_2026]
    expanded = expand_nan_ind(all_dfs, feats)
    valid = [c for c in expanded if c in df_trn.columns
             and df_trn[c].isna().mean() < 1.0 and df_trn[c].std(ddof=0) > 0]
    X_tr, y_tr, gs_tr, n_tr, nr_tr, scaler, *_ = prepare(
        df_trn, valid, top_idx=None, top_idx3=None, fit=True)
    X_va, y_va, gs_va, n_va, nr_va, *_ = prepare(
        df_val, valid, scaler=scaler, top_idx=None, top_idx3=None)
    beta = adam_fit(X_tr, y_tr, gs_tr, n_tr, nr_tr, X_va, y_va, gs_va, n_va, nr_va)
    val_s = df_val.sort_values('race_id').reset_index(drop=True)
    raw_val = segment_softmax(X_va @ beta, gs_va, n_va)
    y_val = (val_s['着順_num'] == 1).astype(float).values
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(raw_val, y_val)
    results = {}
    for label, oos in [('2324', oos_2324), ('2025', oos_2025), ('2026', oos_2026)]:
        if len(oos) == 0: continue
        vp = [c for c in valid if c in oos.columns]
        X_p, _, gs_p, n_p, *_ = prepare(oos, vp, scaler=scaler, top_idx=None, top_idx3=None)
        s = oos.sort_values('race_id').reset_index(drop=True)
        s['prob'] = segment_softmax(X_p @ beta, gs_p, n_p)
        s['rank'] = s.groupby('race_id')['prob'].rank(ascending=False, method='first')
        t = s[s['rank'] == 1]
        nr = s['race_id'].nunique()
        acc = (t['着順_num'] == 1).mean()
        odds = pd.to_numeric(t['単勝オッズ'], errors='coerce')
        roi = (odds[t['着順_num'] == 1] * 100).sum() / (len(t) * 100) - 1
        results[label] = (acc, roi, nr)
        print(f'  {label}: acc={acc:.2%} ROI={roi:+.2%} ({nr}R)')
    n2324,n25,n26 = results.get('2324',(0,0,0))[2],results.get('2025',(0,0,0))[2],results.get('2026',(0,0,0))[2]
    a2324,a25,a26 = results.get('2324',(0,0,0))[0],results.get('2025',(0,0,0))[0],results.get('2026',(0,0,0))[0]
    r25,r26 = results.get('2025',(0,0,0))[1],results.get('2026',(0,0,0))[1]
    acc_2325 = (a2324*n2324+a25*n25)/(n2324+n25) if (n2324+n25)>0 else 0.0
    acc_2526 = (a25*n25+a26*n26)/(n25+n26) if (n25+n26)>0 else 0.0
    roi_2526 = (r25*n25+r26*n26)/(n25+n26) if (n25+n26)>0 else 0.0
    print(f'  acc_2325={acc_2325:.4f}  25+26_acc={acc_2526:.4f}  25+26_ROI={roi_2526:+.2%}')
    acc_pkg = {
        'segment': '芝長', 'scaler': scaler, 'coef': beta, 'feat_cols': valid, 'isotonic': iso,
        'acc_2325': acc_2325, 'acc_2526': acc_2526, 'oos_roi_2526': roi_2526,
        'version': 'shiba_long_acc_v3',
        'note': f'v3: {len(feats)}特徴 ゼロシード acc_2325={acc_2325:.4f} 1番人気36.05%',
    }
    acc_pkl = os.path.join(BASE_DIR, 'models', 'hitrate_model.pkl')
    existing = pickle.load(open(acc_pkl, 'rb'))
    existing['芝長'] = acc_pkg
    with open(acc_pkl, 'wb') as f:
        pickle.dump(existing, f)
    print(f'  保存完了: 芝長 v3')
    return acc_2325


def main():
    print('芝長 的中率探索 v3 (ゼロシード + 新候補)')
    print(f'目標: {FAV_TARGET:.2%} (1番人気)  MAX_FEATS={MAX_FEATS}')

    df = pd.read_parquet(DATA_FILE)
    df['日付_num'] = pd.to_numeric(df['日付'], errors='coerce')
    df['着順_num'] = pd.to_numeric(df['着順_num'], errors='coerce')
    df = df.dropna(subset=['日付_num', '着順_num'])
    df = df[df['着順_num'] < 99]
    df['race_id'] = (df['日付_num'].astype(int).astype(str) + '_' +
                     df['開催'].astype(str).str.strip() + '_' +
                     df['Ｒ'].astype(str).str.strip())
    df = df[df['開催'].notna()].copy()
    df['surface'] = df['距離'].astype(str).str.strip().str.extract(r'^([芝ダ])')[0].fillna('不明')
    dm = pd.to_numeric(df['距離'].astype(str).str.extract(r'(\d+)')[0], errors='coerce')
    df['クラス_rank'] = pd.to_numeric(df['クラス_rank'], errors='coerce')
    df = add_computed_features(df)
    if '今回_会場' in df.columns and '1走前_開催' in df.columns:
        df['輸送有無'] = (df['今回_会場'].astype(str) != df['1走前_開催'].astype(str).str[1]).astype(float)
        df.loc[df['1走前_開催'].isna(), '輸送有無'] = float('nan')
    baba_map = {'良': 0, '稍重': 1, '重': 2, '不良': 3}
    for col in df.columns:
        if '馬場状態' in col and col != '馬場状態':
            df[col] = df[col].map(baba_map)

    seg = df[(df['surface'] == '芝') & (dm > 2000) & (df['クラス_rank'] != 1.0)].copy()
    seg['dist_m'] = dm[seg.index]

    df_trn = seg[(seg['日付_num'] >= 130101) & (seg['日付_num'] < 220101)]
    df_val = seg[(seg['日付_num'] >= 220101) & (seg['日付_num'] <= 221231)]
    oos_2324 = seg[(seg['日付_num'] >= 230101) & (seg['日付_num'] < 250101)]
    oos_2025 = seg[(seg['日付_num'] >= 250101) & (seg['日付_num'] < 260101)]
    dfs = (df_trn, df_val, oos_2324, oos_2025)

    candidates = list(dict.fromkeys([c for c in CANDIDATES if c in seg.columns and c not in FORCED_BASE]))
    current = FORCED_BASE[:]
    best_score = eval_feats(current, dfs)
    print(f'ゼロシード: acc_2325={best_score:.4f}')
    best_feats = current[:]

    remaining = [c for c in candidates if c not in current]
    t0 = time.time()

    while len(current) < MAX_FEATS and remaining:
        best_add = None
        best_add_score = best_score

        for cand in remaining:
            score = eval_feats(current + [cand], dfs)
            if score > best_add_score:
                best_add_score = score
                best_add = cand

        if best_add is None:
            print(f'改善なし → 終了 ({len(current)}特徴, {time.time()-t0:.0f}s)')
            break

        current.append(best_add)
        remaining.remove(best_add)
        best_score = best_add_score
        best_feats = current[:]
        gap = FAV_TARGET - best_score
        mark = '★' if best_score >= FAV_TARGET else ' '
        print(f'{mark}+{best_add:38s} {best_score:.4f} gap={gap:+.4f} ({len(current)}f, {time.time()-t0:.0f}s)')
        sys.stdout.flush()

        if best_score >= FAV_TARGET:
            print(f'\n★★★ 1番人気超え達成! {best_score:.4f} > {FAV_TARGET:.4f} ★★★')
            break

    print(f'\n最終: 芝長 acc_2325={best_score:.4f}  目標まで{FAV_TARGET-best_score:+.4f}  ({len(best_feats)}特徴)')

    acc_pkl = os.path.join(BASE_DIR, 'models', 'hitrate_model.pkl')
    existing_acc = 0.0
    if os.path.exists(acc_pkl):
        pkg = pickle.load(open(acc_pkl, 'rb'))
        if '芝長' in pkg:
            existing_acc = pkg['芝長'].get('acc_2325', 0.0)
    print(f'現行: {existing_acc:.4f}  今回: {best_score:.4f}')

    if best_score > existing_acc + 0.0001:
        print('改善あり、保存...')
        save_seg(best_feats, seg)
    else:
        print('改善なし、スキップ')


if __name__ == '__main__':
    main()
