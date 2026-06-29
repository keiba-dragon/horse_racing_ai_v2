# coding: utf-8
"""
search_fukusho_random.py - 複勝版 ランダムサーチ
- 目的変数: 3着以内 (複勝)
- 損失: neg_log_lik_fukusho (P(top3)の最大化)
- 選択指標: acc_fukusho_2325 (2325合算複勝的中率)
- 保存先: accuracy_fukusho_model.pkl
usage: python src/search_fukusho_random.py 芝長 300
"""
import sys, os, time, pickle, random
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from save_conditional_logit import (prepare, segment_softmax,
                                    neg_log_lik_fukusho_and_grad,
                                    BASE_DIR, DATA_FILE, LR, N_EPOCHS, PATIENCE)
from save_v3 import add_computed_features

SEG_NAME = sys.argv[1] if len(sys.argv) > 1 else '芝長'
N_ITER   = int(sys.argv[2]) if len(sys.argv) > 2 else 300

L2 = 0.006
NAN_IND_THRESHOLD = 0.05
SEED_RNG = 42
MIN_FEATS = 10
MAX_FEATS = 50

# 1番人気の複勝的中率（目標値）
FAV_FUKUSHO = {'ダ長': 0.615, 'ダ短': 0.620, '芝短': 0.570, '芝中': 0.625, '芝長': 0.660}

# SEEDは単勝版と同じ（同じ特徴量空間から探索）
SEEDS = {
    'ダ長': ['馬番','斤量','近3走_複勝率','騎手コース_r100_勝率','1走前_クラス調整着順',
             '近5走_タイム指数_max','馬距離_勝率','種牡馬_勝率','タイム指数_加速度',
             '近5走_タイム指数平均','近5走_上り3F平均','近5走_上り3F_std','1走前_クラス差',
             'ブリンカー変更','1走前_3角','間隔','距離変化_前走','前走着差タイム','騎手変更',
             '輸送有無','コース枠_r200_勝率','馬体重増減','コース脚質_r200_勝率','1走前_馬場状態',
             '1走前_脚質_num','近10走_勝率','2走前_クラス差'],
    'ダ短': ['馬番','斤量','芝ダ一致_平均着順_近5走','1走前_タイム指数',
             '近5走_クラス調整_平均着順','馬コース_r20_勝率',
             '近3走_体重増減合計','性別_num','1走前_クラス差','コース枠_r200_複勝率',
             'コース枠_r200_勝率','3走前_クラス差','近10走_勝率','ブリンカー変更',
             '2走前_クラス差','相手レベル_平均着順','種牡馬_ダ_勝率','近5走_上り3F平均',
             '近5走_タイム指数_max','1走前_馬場状態','間隔','近5走_タイム指数平均'],
    '芝短': ['馬番','斤量','芝ダ一致_平均着順_近5走','1走前_タイム指数',
             '近5走_クラス調整_平均着順','馬コース_r20_勝率','騎手コース_r100_勝率',
             '近10走_勝率','馬体重','近5走_上り3F平均','芝ダ転向',
             '近3走_体重増減合計','相手レベル_平均着順',
             'タイム指数_近3走_slope','馬体重増減','コース馬場_r200_勝率',
             'ブリンカー変更','3走前_クラス差','近5走_上り3F_std',
             '馬距離_勝率','コース枠_r200_勝率','前走着差タイム'],
    '芝中': ['馬番','斤量','芝ダ一致_平均着順_近5走','騎手距離_r100_勝率',
             '1走前_タイム指数','1走前_クラス調整着順','馬コース_r20_勝率',
             '馬体重','前走着差タイム','近5走_上り3F平均',
             '近5走_タイム指数平均','1走前_クラス差',
             'コース枠_r200_勝率','性別_num','1走前_馬場状態',
             'ブリンカー変更','芝ダ転向','間隔','相手レベル_平均着順','近5走_上り3F_std'],
    '芝長': ['近3走_複勝率','騎手距離_r100_勝率','近5走_タイム指数平均',
             '馬コース_r20_勝率','タイム指数_近3走_slope','調教師コース_r100_勝率',
             '同会場_複勝率_近5走','近5走_上り3F_std','コース枠_r200_複勝率',
             '相手レベル_平均着順','タイム指数_加速度','近10走_勝率','近3走_体重増減合計',
             'コース馬場_r200_勝率','1走前_馬場状態','タイム指数_近5走_slope',
             '1走前_タイム指数','距離変化_前走','性別_num',
             '馬体重','馬体重増減','コース枠_r200_勝率','ブリンカー変更','種牡馬_勝率'],
}

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
    '種牡馬_芝_複勝率','種牡馬_ダ_複勝率','馬_r20_複勝率',
    '騎手_r200_複勝率','調教師_r200_複勝率',
]

VERSION_MAP = {
    'ダ長': 'da_long_fukusho',
    'ダ短': 'da_short_fukusho',
    '芝短': 'shiba_short_fukusho',
    '芝中': 'shiba_mid_fukusho',
    '芝長': 'shiba_long_fukusho',
}


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


def get_y_fukusho(df_sorted):
    """3着以内フラグ"""
    return (df_sorted['着順_num'] <= 3).astype(float).values


def _loss_grad(beta, X, y, gs, n, nr):
    return neg_log_lik_fukusho_and_grad(beta, X, y, gs, n, nr)


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
        X_tr, _, gs_tr, n_tr, nr_tr, sc, *_ = prepare(df_trn, valid, top_idx=None, top_idx3=None, fit=True)
        y_tr = get_y_fukusho(df_trn.sort_values('race_id').reset_index(drop=True))
        X_va, _, gs_va, n_va, nr_va, *_ = prepare(df_val, valid, scaler=sc, top_idx=None, top_idx3=None)
        y_va = get_y_fukusho(df_val.sort_values('race_id').reset_index(drop=True))
        beta = adam_fit(X_tr, y_tr, gs_tr, n_tr, nr_tr, X_va, y_va, gs_va, n_va, nr_va)
    except Exception as e:
        return float('-inf')

    def acc_fukusho_oos(oos):
        vp = [c for c in valid if c in oos.columns]
        X_p, _, gs_p, n_p, *_ = prepare(oos, vp, scaler=sc, top_idx=None, top_idx3=None)
        s = oos.sort_values('race_id').reset_index(drop=True)
        s['prob'] = segment_softmax(X_p @ beta, gs_p, n_p)
        s['rank'] = s.groupby('race_id')['prob'].rank(ascending=False, method='first')
        top1 = s[s['rank'] == 1]; nr = s['race_id'].nunique()
        # 複勝的中率 = モデル1位の馬が3着以内に入った割合
        acc = (top1['着順_num'] <= 3).mean()
        return acc, nr

    a2324, n2324 = acc_fukusho_oos(oos_2324)
    a25, n25     = acc_fukusho_oos(oos_2025)
    return (a2324*n2324 + a25*n25) / (n2324+n25) if (n2324+n25) > 0 else float('-inf')


def save_seg(name, feats, seg, version_name):
    df_trn = seg[(seg['日付_num']>=130101)&(seg['日付_num']<220101)]
    df_val = seg[(seg['日付_num']>=220101)&(seg['日付_num']<=221231)]
    oos_2324 = seg[(seg['日付_num']>=230101)&(seg['日付_num']<250101)]
    oos_2025 = seg[(seg['日付_num']>=250101)&(seg['日付_num']<260101)]
    oos_2026 = seg[seg['日付_num']>=260101]
    all_dfs = [df_trn, df_val, oos_2324, oos_2025, oos_2026]
    expanded = expand_nan_ind(all_dfs, feats)
    valid = [c for c in expanded if c in df_trn.columns
             and df_trn[c].isna().mean() < 1.0 and df_trn[c].std(ddof=0) > 0]
    X_tr, _, gs_tr, n_tr, nr_tr, scaler, *_ = prepare(df_trn, valid, top_idx=None, top_idx3=None, fit=True)
    y_tr = get_y_fukusho(df_trn.sort_values('race_id').reset_index(drop=True))
    X_va, _, gs_va, n_va, nr_va, *_ = prepare(df_val, valid, scaler=scaler, top_idx=None, top_idx3=None)
    y_va = get_y_fukusho(df_val.sort_values('race_id').reset_index(drop=True))
    beta = adam_fit(X_tr, y_tr, gs_tr, n_tr, nr_tr, X_va, y_va, gs_va, n_va, nr_va)

    # isotonic calibration（複勝確率でキャリブ）
    val_s = df_val.sort_values('race_id').reset_index(drop=True)
    raw_val = segment_softmax(X_va @ beta, gs_va, n_va)
    y_val_fukusho = get_y_fukusho(val_s)
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(raw_val, y_val_fukusho)

    results = {}
    for label, oos in [('2324', oos_2324), ('2025', oos_2025), ('2026', oos_2026)]:
        if len(oos) == 0: continue
        vp = [c for c in valid if c in oos.columns]
        X_p, _, gs_p, n_p, *_ = prepare(oos, vp, scaler=scaler, top_idx=None, top_idx3=None)
        s = oos.sort_values('race_id').reset_index(drop=True)
        s['prob'] = segment_softmax(X_p @ beta, gs_p, n_p)
        s['prob_calib'] = iso.predict(s['prob'].values)
        s['rank'] = s.groupby('race_id')['prob'].rank(ascending=False, method='first')
        top1 = s[s['rank'] == 1]; nr = s['race_id'].nunique()

        # 複勝的中率
        acc_f = (top1['着順_num'] <= 3).mean()
        # 複勝ROI（払い戻し: 複勝配当/100）
        top1_hit = top1[top1['着順_num'] <= 3]
        fukusho_odds = pd.to_numeric(top1_hit['複勝配当'], errors='coerce') / 100
        payout = fukusho_odds.sum()
        roi_f = (payout - nr) / nr
        # 推定EV（P_calib × 単勝オッズ × 0.32）
        top1['推定複勝EV'] = (
            pd.to_numeric(top1['prob_calib'], errors='coerce') *
            pd.to_numeric(top1['単勝オッズ'], errors='coerce') * 0.32
        )
        results[label] = (acc_f, roi_f, nr)
        print(f'  {label}: 複勝的中率={acc_f:.2%} 複勝ROI={roi_f:+.2%} ({nr}R)')

    n2324 = results.get('2324', (0, 0, 0))[2]
    n25   = results.get('2025', (0, 0, 0))[2]
    n26   = results.get('2026', (0, 0, 0))[2]
    a2324 = results.get('2324', (0, 0, 0))[0]
    a25   = results.get('2025', (0, 0, 0))[0]
    a26   = results.get('2026', (0, 0, 0))[0]
    r25   = results.get('2025', (0, 0, 0))[1]
    r26   = results.get('2026', (0, 0, 0))[1]
    acc_2325 = (a2324*n2324 + a25*n25) / (n2324+n25) if (n2324+n25) > 0 else 0.0
    acc_2526 = (a25*n25 + a26*n26) / (n25+n26) if (n25+n26) > 0 else 0.0
    roi_2526 = (r25*n25 + r26*n26) / (n25+n26) if (n25+n26) > 0 else 0.0
    fav = FAV_FUKUSHO.get(name, 0.6)
    print(f'  acc_fukusho_2325={acc_2325:.4f}  25+26複勝的中率={acc_2526:.4f}  '
          f'25+26複勝ROI={roi_2526:+.2%}  (1番人気目標≈{fav:.2%})')

    pkg = {
        'segment': name,
        'scaler': scaler,
        'coef': beta,
        'feat_cols': valid,
        'isotonic': iso,
        'mode': 'fukusho',
        'acc_2325': acc_2325,
        'acc_2526': acc_2526,
        'roi_2526': roi_2526,
        'version': version_name,
        'note': f'{version_name}: {len(feats)}特徴 acc_fukusho_2325={acc_2325:.4f} 1番人気目標≈{fav:.2%}',
    }
    # accuracy_fukusho_model.pkl に保存（hitrate_model.pkl は上書きしない）
    fukusho_pkl = os.path.join(BASE_DIR, 'models', 'accuracy_fukusho_model.pkl')
    if os.path.exists(fukusho_pkl):
        existing = pickle.load(open(fukusho_pkl, 'rb'))
    else:
        existing = {}
    if name not in existing or acc_2325 > existing[name].get('acc_2325', 0):
        existing[name] = pkg
        with open(fukusho_pkl, 'wb') as f:
            pickle.dump(existing, f)
        print(f'  保存: {name} -> accuracy_fukusho_model.pkl ({version_name})')
    else:
        print(f'  スキップ: {name} 既存スコア{existing[name]["acc_2325"]:.4f} >= 新スコア{acc_2325:.4f}')
    return acc_2325


def main():
    rng = random.Random(SEED_RNG)
    name = SEG_NAME
    print(f'複勝版ランダムサーチ: {name}  N_ITER={N_ITER}  L2={L2}')

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
        if '馬場状態' in col and col != '馬場状態':
            df[col] = df[col].map(baba_map)

    s = df['surface']; r = df['クラス_rank']
    if name == 'ダ長':   mask = (s=='ダ')&(dm>1400) &(r!=1.0)
    elif name == 'ダ短': mask = (s=='ダ')&(dm<=1400)&(r!=1.0)
    elif name == '芝短': mask = (s=='芝')&(dm<=1400)&(r!=1.0)
    elif name == '芝中': mask = (s=='芝')&(dm>1400) &(dm<=2000)&(r!=1.0)
    elif name == '芝長': mask = (s=='芝')&(dm>2000) &(r!=1.0)
    else: raise ValueError(f'Unknown segment: {name}')
    seg = df[mask].copy(); seg['dist_m'] = dm[seg.index]

    # 1番人気複勝的中率を計算して目標表示
    seg_oos = seg[(seg['日付_num']>=230101)&(seg['日付_num']<260101)]
    seg_oos_sorted = seg_oos.sort_values('race_id').reset_index(drop=True)
    seg_oos_sorted['単勝オッズ_num'] = pd.to_numeric(seg_oos_sorted['単勝オッズ'], errors='coerce')
    seg_oos_sorted['rank_odds'] = seg_oos_sorted.groupby('race_id')['単勝オッズ_num'].rank(method='first', ascending=True)
    fav1 = seg_oos_sorted[seg_oos_sorted['rank_odds']==1]
    fav_acc = (fav1['着順_num'] <= 3).mean()
    fav_n   = seg_oos_sorted['race_id'].nunique()
    print(f'1番人気複勝的中率 (2325 OOS): {fav_acc:.2%} ({fav_n}R)')
    FAV_FUKUSHO[name] = fav_acc

    df_trn  = seg[(seg['日付_num']>=130101)&(seg['日付_num']<220101)]
    df_val  = seg[(seg['日付_num']>=220101)&(seg['日付_num']<=221231)]
    oos_2324 = seg[(seg['日付_num']>=230101)&(seg['日付_num']<250101)]
    oos_2025 = seg[(seg['日付_num']>=250101)&(seg['日付_num']<260101)]
    dfs = (df_trn, df_val, oos_2324, oos_2025)

    cands = [c for c in ALL_CANDS if c in seg.columns]
    seed_feats = [f for f in SEEDS[name] if f in seg.columns]

    best_feats = seed_feats[:]
    best_score = eval_feats(best_feats, dfs)
    print(f'SEED score: {best_score:.4f} ({len(best_feats)}特徴)')
    print(f'候補数: {len(cands)}')

    t0 = time.time()
    for i in range(N_ITER):
        op = rng.choice(['add', 'remove', 'swap', 'swap'])
        current = best_feats[:]
        not_in = [c for c in cands if c not in current]

        if op == 'add' and not_in and len(current) < MAX_FEATS:
            current.append(rng.choice(not_in))
        elif op == 'remove' and len(current) > MIN_FEATS:
            base = [f for f in current if not f.endswith('_isnan')]
            if base: current.remove(rng.choice(base))
        elif op == 'swap' and not_in and len(current) > MIN_FEATS:
            base = [f for f in current if not f.endswith('_isnan')]
            if base:
                current.remove(rng.choice(base))
                current.append(rng.choice(not_in))

        score = eval_feats(current, dfs)
        if score > best_score:
            best_score = score; best_feats = current
            elapsed = time.time() - t0
            print(f'  it={i+1:4d} {score:.4f} ← 改善 {elapsed:.0f}s  {len(best_feats)}特徴', flush=True)

    print(f'\n最終: {name} acc_fukusho_2325={best_score:.4f}  目標={fav_acc:.4f} ({N_ITER}it)')
    if best_score >= eval_feats(seed_feats, dfs):
        save_seg(name, best_feats, seg, VERSION_MAP[name])
    else:
        print('改善なし、スキップ')


if __name__ == '__main__':
    main()
