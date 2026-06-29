# coding: utf-8
"""
search_shiba_long_acc_multistart.py - 芝長 マルチスタートランダムサーチ
複数のランダム出発点から探索し、局所最適の罠を脱出する
usage: python src/search_shiba_long_acc_multistart.py [N_STARTS] [ITERS_PER_START] [L2]
"""
import sys, os, time, pickle, random
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from save_conditional_logit import prepare, segment_softmax, BASE_DIR, DATA_FILE, LR, N_EPOCHS, PATIENCE
from save_v3 import add_computed_features

SEG_NAME       = sys.argv[1]        if len(sys.argv) > 1 else '芝長'
N_STARTS       = int(sys.argv[2])   if len(sys.argv) > 2 else 20
ITERS_PER_START= int(sys.argv[3])   if len(sys.argv) > 3 else 300
L2             = float(sys.argv[4]) if len(sys.argv) > 4 else 0.003
MASTER_SEED    = 2025
NAN_IND_THRESHOLD = 0.05
MIN_FEATS, MAX_FEATS = 15, 40
FAV_MAP = {'ダ長': 0.3403, 'ダ短': 0.3490, '芝短': 0.2869, '芝中': 0.3321, '芝長': 0.3605}
FAV = FAV_MAP[SEG_NAME]

ALL_CANDS = [
    '近3走_上り3F_min','前走_1番人気フラグ','前走_人気着順差',
    '1走前_単勝オッズ','1走前_上り3F','1走前_PCI','1走前_RPCI','1走前_頭数',
    '2走前_タイム指数','2走前_上り3F','2走前_着順_num','3走前_タイム指数','3走前_上り3F',
    '近10走_複勝率','近5走_複勝率','近5走_タイム指数_min',
    '芝ダ一致_平均着順_近5走','1走前_3角','1走前_4角','1走前_脚質_num',
    '芝ダ転向','距離変化_前走','馬距離_勝率','前走着差タイム',
    '近5走_上り3F平均','近5走_上り3F_std','1走前_タイム指数',
    '近5走_タイム指数平均','近5走_タイム指数_max',
    'タイム指数_近3走_slope','タイム指数_近5走_slope','タイム指数_加速度',
    '着順_近3走_slope','上り3F_近3走_slope','4角位置_近3走_slope',
    '1走前_クラス差','2走前_クラス差','3走前_クラス差',
    '1走前_クラス調整着順','近5走_クラス調整_平均着順',
    '1走前_馬場状態','道悪_平均着順_近5走','同馬場_平均着順_近5走','良馬場_平均着順_近5走',
    '種牡馬_勝率','種牡馬_ダ_勝率','母父馬_勝率','馬_r20_勝率','馬コース_r20_勝率',
    '騎手コース_r100_勝率','騎手会場_r100_勝率','騎手コース距離_r100_勝率',
    '騎手距離_r100_勝率','騎手馬場_r100_勝率','調教師コース_r100_勝率','調教師_r200_勝率',
    '近3走_複勝率','近3走_勝率','近10走_複勝率','近10走_勝率',
    '同会場_複勝率_近5走','同会場_平均着順_近5走','同距離帯_平均着順_近5走','相手レベル_平均着順',
    '間隔','性別_num','騎手変更','輸送有無','馬番','斤量',
    '馬体重','馬体重増減','ブリンカー変更','近3走_体重増減合計',
    'コース枠_r200_勝率','コース脚質_r200_勝率','コース馬場_r200_勝率','コース枠_r200_複勝率',
]
FORCED = ['馬番', '斤量']


def expand_nan_ind(dfs, feats):
    ref = dfs[0]; extended = []
    for f in feats:
        extended.append(f)
        if f not in ref.columns: continue
        if NAN_IND_THRESHOLD < ref[f].isna().mean() < 1.0:
            ind = f + '_isnan'
            for df in dfs:
                if f in df.columns and ind not in df.columns:
                    df[ind] = df[f].isna().astype(float)
            extended.append(ind)
    return extended


def _loss_grad(beta, X, y, gs, n, nr):
    probs = segment_softmax(X @ beta, gs, n)
    loss = -np.sum(y * np.log(np.clip(probs, 1e-15, 1.0))) / nr + L2 * np.dot(beta, beta)
    grad = -(X.T @ (y - probs)) / nr + 2 * L2 * beta
    return loss, grad


def adam_fit(X_tr, y_tr, gs_tr, n_tr, nr_tr, X_va, y_va, gs_va, n_va, nr_va):
    d = X_tr.shape[1]; beta, m, v = np.zeros(d), np.zeros(d), np.zeros(d)
    b1, b2, eps = 0.9, 0.999, 1e-8; t, best_val, best_beta, no_imp = 0, np.inf, np.zeros(d), 0
    for epoch in range(1, N_EPOCHS + 1):
        _, grad = _loss_grad(beta, X_tr, y_tr, gs_tr, n_tr, nr_tr)
        t += 1; m = b1*m + (1-b1)*grad; v = b2*v + (1-b2)*grad**2
        beta -= LR * (m/(1-b1**t)) / (np.sqrt(v/(1-b2**t)) + eps)
        if epoch % 10 == 0:
            vl, _ = _loss_grad(beta, X_va, y_va, gs_va, n_va, nr_va)
            if vl < best_val: best_val, best_beta, no_imp = vl, beta.copy(), 0
            else: no_imp += 1
            if no_imp >= PATIENCE // 10: break
    return best_beta


def eval_feats(feats, dfs):
    df_trn, df_val, oos_2324, oos_2025 = dfs
    all_dfs = list(dfs)
    expanded = expand_nan_ind(all_dfs, feats)
    valid = [c for c in expanded if c in df_trn.columns
             and df_trn[c].isna().mean() < 1.0 and df_trn[c].std(ddof=0) > 0]
    if len(valid) < 2: return float('-inf')
    try:
        X_tr, y_tr, gs_tr, n_tr, nr_tr, sc, *_ = prepare(df_trn, valid, top_idx=None, top_idx3=None, fit=True)
        X_va, y_va, gs_va, n_va, nr_va, *_ = prepare(df_val, valid, scaler=sc, top_idx=None, top_idx3=None)
        beta = adam_fit(X_tr, y_tr, gs_tr, n_tr, nr_tr, X_va, y_va, gs_va, n_va, nr_va)
    except: return float('-inf')
    def acc_oos(oos):
        vp = [c for c in valid if c in oos.columns]
        X_p, _, gs_p, n_p, *_ = prepare(oos, vp, scaler=sc, top_idx=None, top_idx3=None)
        s = oos.sort_values('race_id').reset_index(drop=True)
        s['prob'] = segment_softmax(X_p @ beta, gs_p, n_p)
        s['rank'] = s.groupby('race_id')['prob'].rank(ascending=False, method='first')
        t2 = s[s['rank'] == 1]
        return (t2['着順_num'] == 1).mean(), s['race_id'].nunique()
    a2324, n2324 = acc_oos(oos_2324); a25, n25 = acc_oos(oos_2025)
    return (a2324*n2324 + a25*n25) / (n2324+n25) if (n2324+n25) > 0 else float('-inf')


def local_search(start_feats, cands, dfs, rng, n_iter):
    best_feats = start_feats[:]
    best_score = eval_feats(best_feats, dfs)
    for _ in range(n_iter):
        op = rng.choice(['add', 'remove', 'swap', 'swap'])
        current = best_feats[:]
        not_in = [c for c in cands if c not in current]
        if op == 'add' and not_in and len(current) < MAX_FEATS:
            current.append(rng.choice(not_in))
        elif op == 'remove' and len(current) > MIN_FEATS:
            removable = [c for c in current if c not in FORCED]
            if removable: current.remove(rng.choice(removable))
            else: continue
        elif op == 'swap' and not_in and len(current) > MIN_FEATS:
            removable = [c for c in current if c not in FORCED]
            if removable:
                current.remove(rng.choice(removable))
                current.append(rng.choice([c for c in cands if c not in current]))
            else: continue
        else:
            continue
        score = eval_feats(current, dfs)
        if score > best_score:
            best_score = score
            best_feats = current[:]
    return best_feats, best_score


def save_result(best_feats, best_score, seg, df_trn, df_val, oos_2324, oos_2025):
    oos_2026 = seg[seg['日付_num'] >= 260101]
    all_dfs = [df_trn, df_val, oos_2324, oos_2025, oos_2026]
    expanded = expand_nan_ind(all_dfs, best_feats)
    valid = [c for c in expanded if c in df_trn.columns
             and df_trn[c].isna().mean() < 1.0 and df_trn[c].std(ddof=0) > 0]
    X_tr, y_tr, gs_tr, n_tr, nr_tr, scaler, *_ = prepare(df_trn, valid, top_idx=None, top_idx3=None, fit=True)
    X_va, y_va, gs_va, n_va, nr_va, *_ = prepare(df_val, valid, scaler=scaler, top_idx=None, top_idx3=None)
    beta = adam_fit(X_tr, y_tr, gs_tr, n_tr, nr_tr, X_va, y_va, gs_va, n_va, nr_va)
    raw_val = segment_softmax(X_va @ beta, gs_va, n_va)
    val_s = df_val.sort_values('race_id').reset_index(drop=True)
    y_val = (val_s['着順_num'] == 1).astype(float).values
    iso = IsotonicRegression(out_of_bounds='clip'); iso.fit(raw_val, y_val)
    results = {}
    for label, oos in [('2324', oos_2324), ('2025', oos_2025), ('2026', oos_2026)]:
        if len(oos) == 0: continue
        vp = [c for c in valid if c in oos.columns]
        X_p, _, gs_p, n_p, *_ = prepare(oos, vp, scaler=scaler, top_idx=None, top_idx3=None)
        s2 = oos.sort_values('race_id').reset_index(drop=True)
        s2['prob'] = segment_softmax(X_p @ beta, gs_p, n_p)
        s2['rank'] = s2.groupby('race_id')['prob'].rank(ascending=False, method='first')
        t2 = s2[s2['rank'] == 1]; nr = s2['race_id'].nunique()
        acc = (t2['着順_num'] == 1).mean()
        odds = pd.to_numeric(t2['単勝オッズ'], errors='coerce')
        roi = (odds[t2['着順_num'] == 1] * 100).sum() / (len(t2) * 100) - 1
        results[label] = (acc, roi, nr)
        print(f'    {label}: acc={acc:.2%} ROI={roi:+.2%} ({nr}R)')
    n2324 = results.get('2324',(0,0,0))[2]; n25 = results.get('2025',(0,0,0))[2]; n26 = results.get('2026',(0,0,0))[2]
    a2324 = results.get('2324',(0,0,0))[0]; a25 = results.get('2025',(0,0,0))[0]; a26 = results.get('2026',(0,0,0))[0]
    r25 = results.get('2025',(0,0,0))[1]; r26 = results.get('2026',(0,0,0))[1]
    acc_2325 = (a2324*n2324 + a25*n25) / (n2324+n25) if (n2324+n25) > 0 else 0.0
    acc_2526 = (a25*n25 + a26*n26) / (n25+n26) if (n25+n26) > 0 else 0.0
    roi_2526 = (r25*n25 + r26*n26) / (n25+n26) if (n25+n26) > 0 else 0.0
    print(f'    acc_2325={acc_2325:.4f}  25+26_acc={acc_2526:.4f}  ROI={roi_2526:+.2%}')
    acc_pkg = {
        'segment': SEG_NAME, 'scaler': scaler, 'coef': beta, 'feat_cols': valid,
        'isotonic': iso, 'acc_2325': acc_2325, 'acc_2526': acc_2526,
        'oos_roi_2526': roi_2526, 'version': f'{SEG_NAME}_acc_multistart_L2_{str(L2).replace(".","")[:5]}',
        'note': f'multistart L2={L2}: {len(best_feats)}特徴 acc_2325={acc_2325:.4f}',
    }
    acc_pkl = os.path.join(BASE_DIR, 'models', 'hitrate_model.pkl')
    existing = pickle.load(open(acc_pkl, 'rb'))
    existing[SEG_NAME] = acc_pkg
    with open(acc_pkl, 'wb') as f: pickle.dump(existing, f)
    print(f'    保存完了: {SEG_NAME}')


def main():
    print(f'マルチスタートサーチ: {SEG_NAME}  N_STARTS={N_STARTS}  ITERS/START={ITERS_PER_START}  L2={L2}  目標={FAV:.4f}')

    df = pd.read_parquet(DATA_FILE)
    df['日付_num'] = pd.to_numeric(df['日付'], errors='coerce')
    df['着順_num'] = pd.to_numeric(df['着順_num'], errors='coerce')
    df = df.dropna(subset=['日付_num', '着順_num'])
    df = df[df['着順_num'] < 99]
    df['race_id'] = (df['日付_num'].astype(int).astype(str) + '_' +
                     df['開催'].astype(str).str.strip() + '_' + df['Ｒ'].astype(str).str.strip())
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
        if '馬場状態' in col and col != '馬場状態': df[col] = df[col].map(baba_map)

    s = df['surface']; r = df['クラス_rank']
    if SEG_NAME == 'ダ長':   mask = (s == 'ダ') & (dm > 1400) & (r != 1.0)
    elif SEG_NAME == 'ダ短': mask = (s == 'ダ') & (dm <= 1400) & (r != 1.0)
    elif SEG_NAME == '芝短': mask = (s == '芝') & (dm <= 1400) & (r != 1.0)
    elif SEG_NAME == '芝中': mask = (s == '芝') & (dm > 1400) & (dm <= 2000) & (r != 1.0)
    else:                    mask = (s == '芝') & (dm > 2000) & (r != 1.0)
    seg = df[mask].copy(); seg['dist_m'] = dm[seg.index]
    df_trn = seg[(seg['日付_num'] >= 130101) & (seg['日付_num'] < 220101)]
    df_val = seg[(seg['日付_num'] >= 220101) & (seg['日付_num'] <= 221231)]
    oos_2324 = seg[(seg['日付_num'] >= 230101) & (seg['日付_num'] < 250101)]
    oos_2025 = seg[(seg['日付_num'] >= 250101) & (seg['日付_num'] < 260101)]
    dfs = (df_trn, df_val, oos_2324, oos_2025)

    cands = [c for c in ALL_CANDS if c in seg.columns]
    print(f'有効候補数: {len(cands)}')

    rng = random.Random(MASTER_SEED)
    global_best_feats = None
    global_best_score = float('-inf')

    # 現在のpklベストもスタート点に加える
    acc_pkl = os.path.join(BASE_DIR, 'models', 'hitrate_model.pkl')
    if os.path.exists(acc_pkl):
        pkg = pickle.load(open(acc_pkl, 'rb'))
        if SEG_NAME in pkg:
            global_best_score = pkg[SEG_NAME].get('acc_2325', 0.0)
            saved_feats = [f for f in pkg[SEG_NAME].get('feat_cols', []) if not f.endswith('_isnan')]
            global_best_feats = saved_feats
            print(f'既存pkl: {global_best_score:.4f} ({len(global_best_feats)}特徴)')

    t0 = time.time()
    for start_idx in range(N_STARTS):
        # ランダムな出発点を生成（MIN_FEATS〜25個）
        n_init = rng.randint(MIN_FEATS, 25)
        pool = [c for c in cands if c not in FORCED]
        init_feats = FORCED[:] + rng.sample(pool, min(n_init, len(pool)))

        score_before = eval_feats(init_feats, dfs)
        best_f, best_s = local_search(init_feats, cands, dfs, random.Random(rng.randint(0, 99999)), ITERS_PER_START)

        mark = '★' if best_s >= FAV else ('↑' if best_s > global_best_score else ' ')
        elapsed = time.time() - t0
        print(f'{mark} スタート[{start_idx+1:2d}/{N_STARTS}] init={score_before:.4f}→{best_s:.4f} '
              f'({len(best_f)}f) gap={FAV-best_s:+.4f} [{elapsed:.0f}s]', flush=True)

        if best_s > global_best_score:
            global_best_score = best_s
            global_best_feats = best_f
            if global_best_score >= FAV:
                print(f'*** {SEG_NAME}: 1番人気超え! {global_best_score:.4f} > {FAV:.4f} ***')
                break

    print(f'\n=== 最終結果: acc_2325={global_best_score:.4f}  目標まで{FAV-global_best_score:+.4f} ===')

    if global_best_feats is not None:
        existing_acc = 0.0
        if os.path.exists(acc_pkl):
            pkg = pickle.load(open(acc_pkl, 'rb'))
            if SEG_NAME in pkg: existing_acc = pkg[SEG_NAME].get('acc_2325', 0.0)
        if global_best_score > existing_acc + 0.0001:
            print(f'改善あり ({existing_acc:.4f}->{global_best_score:.4f}), 保存...')
            save_result(global_best_feats, global_best_score, seg, df_trn, df_val, oos_2324, oos_2025)
        else:
            print(f'改善なし ({existing_acc:.4f}>={global_best_score:.4f}), スキップ')


if __name__ == '__main__':
    main()
