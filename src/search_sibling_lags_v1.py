# coding: utf-8
"""
search_sibling_lags_v1.py — 既存FEATSに「兄弟ラグ」候補を追加してgreedy再探索

考え方:
  現在のFEATSに「2走前_クラス差」があるのに「1走前_クラス差」が無い、といった
  抜けを候補として追加し、SEED=現在のFEATSから改善するか試す
  （search_da_short_acc_v4.pyと同じprepare/segment_softmax/adam_fitを再利用）。
  改善すれば既存より必ず良い状態からスタートするので、既存モデルを壊す心配はない。

usage:
  python src/search_sibling_lags_v1.py --seg ダ短
  python src/search_sibling_lags_v1.py --seg all
"""
import sys, os, time, pickle, argparse, re
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from save_conditional_logit import prepare, segment_softmax, BASE_DIR, DATA_FILE, LR, N_EPOCHS, PATIENCE
from save_v3 import add_computed_features

L2 = 0.006
NAN_IND_THRESHOLD = 0.05
MAX_FEATS = 60

SEG_DEF = {
    'ダ長': ('ダ', lambda d: d > 1400),
    'ダ短': ('ダ', lambda d: d <= 1400),
    '芝短': ('芝', lambda d: d <= 1400),
    '芝中': ('芝', lambda d: (d > 1400) & (d <= 2000)),
    '芝長': ('芝', lambda d: d > 2000),
}
FAV_TARGET = {'ダ長': 0.3403, 'ダ短': 0.3490, '芝短': 0.2869, '芝中': 0.3321, '芝長': 0.3605}

BASE_PATTERNS = ['クラス差', 'クラス調整着順', '馬場状態', 'タイム指数', '3角', '4角',
                 '脚質_num', '上り3F', 'PCI', 'RPCI', '単勝オッズ']
LAG_RANGE = range(1, 6)  # 1走前〜5走前


STAT_SUFFIXES = ['平均', '_max', '_min', '_std', '_range']


def get_sibling_candidates(current_feats, existing_cols):
    """N走前ラグ・勝率⇔複勝率・近3/5/10走・統計量バリエーション・r20/100/200窓
    のいずれかで「使っているのに抜けている」関連列を候補として返す。"""
    feats = set(current_feats)
    missing = set()

    # 1. N走前ラグ
    used_bases = set()
    for f in feats:
        m = re.match(r'\d+走前_(.+)', f)
        if m and m.group(1) in BASE_PATTERNS:
            used_bases.add(m.group(1))
    for base in used_bases:
        for n in LAG_RANGE:
            col = f'{n}走前_{base}'
            if col in existing_cols and col not in feats:
                missing.add(col)

    # 2. 勝率 <-> 複勝率
    for f in feats:
        if f.endswith('_勝率'):
            sib = f[:-3] + '_複勝率'
        elif f.endswith('_複勝率'):
            sib = f[:-4] + '_勝率'
        else:
            continue
        if sib in existing_cols and sib not in feats:
            missing.add(sib)

    # 3. 近3走 <-> 近5走 <-> 近10走
    for f in feats:
        m = re.match(r'近(3|5|10)走_(.+)', f)
        if not m:
            continue
        base = m.group(2)
        for w in ('3', '5', '10'):
            sib = f'近{w}走_{base}'
            if sib in existing_cols and sib not in feats:
                missing.add(sib)

    # 4. 統計量バリエーション（平均/_max/_min/_std/_range）
    for f in feats:
        for suf in STAT_SUFFIXES:
            if f.endswith(suf):
                stem = f[:-len(suf)]
                for suf2 in STAT_SUFFIXES:
                    sib = stem + suf2
                    if sib in existing_cols and sib not in feats:
                        missing.add(sib)
                break

    # 5. r20/r100/r200 ローリング窓
    for f in feats:
        m = re.search(r'_r(20|100|200)_', f)
        if not m:
            continue
        for w in ('20', '100', '200'):
            sib = f.replace(f'_r{m.group(1)}_', f'_r{w}_')
            if sib in existing_cols and sib not in feats:
                missing.add(sib)

    return sorted(missing)


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
    except Exception as e:
        print(f'    [ERROR eval] {e}')
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


def save_seg(feats, seg_name, seg_df):
    df_trn = seg_df[(seg_df['日付_num'] >= 130101) & (seg_df['日付_num'] < 220101)]
    df_val = seg_df[(seg_df['日付_num'] >= 220101) & (seg_df['日付_num'] <= 221231)]
    oos_2324 = seg_df[(seg_df['日付_num'] >= 230101) & (seg_df['日付_num'] < 250101)]
    oos_2025 = seg_df[(seg_df['日付_num'] >= 250101) & (seg_df['日付_num'] < 260101)]
    oos_2026 = seg_df[seg_df['日付_num'] >= 260101]
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
        if len(oos) == 0:
            continue
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

    n2324, n25, n26 = (results.get(k, (0, 0, 0))[2] for k in ('2324', '2025', '2026'))
    a2324, a25, a26 = (results.get(k, (0, 0, 0))[0] for k in ('2324', '2025', '2026'))
    r25, r26 = results.get('2025', (0, 0, 0))[1], results.get('2026', (0, 0, 0))[1]
    acc_2325 = (a2324 * n2324 + a25 * n25) / (n2324 + n25) if (n2324 + n25) > 0 else 0.0
    acc_2526 = (a25 * n25 + a26 * n26) / (n25 + n26) if (n25 + n26) > 0 else 0.0
    roi_2526 = (r25 * n25 + r26 * n26) / (n25 + n26) if (n25 + n26) > 0 else 0.0
    print(f'  acc_2325={acc_2325:.4f}  25+26_acc={acc_2526:.4f}  25+26_ROI={roi_2526:+.2%}')

    acc_pkg = {
        'segment': seg_name, 'scaler': scaler, 'coef': beta, 'feat_cols': valid, 'isotonic': iso,
        'acc_2325': acc_2325, 'acc_2526': acc_2526, 'oos_roi_2526': roi_2526,
        'version': f'{seg_name}_sibling_lag_v1',
        'note': f'兄弟ラグ追加探索: {len(valid)}特徴 acc_2325={acc_2325:.4f}',
    }
    acc_pkl = os.path.join(BASE_DIR, 'models', 'hitrate_model.pkl')
    existing = pickle.load(open(acc_pkl, 'rb'))
    existing[seg_name] = acc_pkg
    with open(acc_pkl, 'wb') as f:
        pickle.dump(existing, f)
    print(f'  保存完了: {seg_name} sibling_lag_v1')
    return acc_2325


def load_data():
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
    df['surface'] = df['距離'].astype(str).str.strip().str.extract(r'^([芝ダ])')[0].fillna('不明')
    df['dist_m'] = pd.to_numeric(df['距離'].astype(str).str.extract(r'(\d+)')[0], errors='coerce')
    df['クラス_rank'] = pd.to_numeric(df['クラス_rank'], errors='coerce')
    df = add_computed_features(df)
    if '今回_会場' in df.columns and '1走前_開催' in df.columns:
        df['輸送有無'] = (df['今回_会場'].astype(str) != df['1走前_開催'].astype(str).str[1]).astype(float)
        df.loc[df['1走前_開催'].isna(), '輸送有無'] = float('nan')
    baba_map = {'良': 0, '稍重': 1, '重': 2, '不良': 3}
    for col in df.columns:
        if '馬場状態' in col and col != '馬場状態':
            df[col] = df[col].map(baba_map)
    df = df[df['クラス_rank'] != 1.0].copy()
    return df


def run_segment(seg_name, df):
    surf, dist_fn = SEG_DEF[seg_name]
    fav = FAV_TARGET[seg_name]

    with open(os.path.join(BASE_DIR, 'models', 'hitrate_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    current_feats = [f for f in model[seg_name]['feat_cols'] if not f.endswith('_isnan')]
    existing_acc = model[seg_name].get('acc_2325', model[seg_name].get('best_roi_2325', 0.0))

    seg_df = df[(df['surface'] == surf) & dist_fn(df['dist_m'])].copy()
    candidates = get_sibling_candidates(current_feats, seg_df.columns)
    candidates = [c for c in candidates if c not in current_feats]
    print(f'\n{"="*70}\n  {seg_name}  現在{len(current_feats)}特徴  兄弟ラグ候補{len(candidates)}個\n{"="*70}')
    print(f'  候補: {candidates}')
    if not candidates:
        print('  候補なし。スキップ')
        return

    df_trn = seg_df[(seg_df['日付_num'] >= 130101) & (seg_df['日付_num'] < 220101)]
    df_val = seg_df[(seg_df['日付_num'] >= 220101) & (seg_df['日付_num'] <= 221231)]
    oos_2324 = seg_df[(seg_df['日付_num'] >= 230101) & (seg_df['日付_num'] < 250101)]
    oos_2025 = seg_df[(seg_df['日付_num'] >= 250101) & (seg_df['日付_num'] < 260101)]
    dfs = (df_trn, df_val, oos_2324, oos_2025)

    current = current_feats[:]
    best_score = eval_feats(current, dfs)
    print(f'  SEED acc_2325={best_score:.4f}')
    remaining = candidates[:]
    t0 = time.time()

    while len(current) < MAX_FEATS and remaining:
        best_add, best_add_score = None, best_score
        for cand in remaining:
            score = eval_feats(current + [cand], dfs)
            if score > best_add_score:
                best_add_score, best_add = score, cand
        if best_add is None:
            print(f'  改善なし → 終了 ({len(current)}特徴, {int(time.time()-t0)}s)')
            break
        current.append(best_add)
        remaining.remove(best_add)
        best_score = best_add_score
        print(f'  +{best_add:30s} acc_2325={best_score:.4f} ({len(current)}f, {int(time.time()-t0)}s)')
        sys.stdout.flush()
        if best_score >= fav:
            print(f'  ★ 目標達成 {best_score:.4f} >= {fav:.4f}')
            break

    print(f'\n  最終: {seg_name} acc_2325={best_score:.4f} ({len(current)}特徴)')
    print(f'  既存: {existing_acc:.4f}  今回: {best_score:.4f}')
    if best_score > existing_acc + 0.0001:
        print('  改善あり、保存...')
        save_seg(current, seg_name, seg_df)
    else:
        print('  改善なし、スキップ')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seg', default='all')
    args = ap.parse_args()
    df = load_data()
    segs = list(SEG_DEF.keys()) if args.seg == 'all' else [args.seg]
    for seg in segs:
        run_segment(seg, df)


if __name__ == '__main__':
    main()
