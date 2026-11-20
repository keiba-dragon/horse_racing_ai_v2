# coding: utf-8
"""
nm_baseline.py - 新モデル(別名で作る。models/hitrate_model.pkl には一切触れない)の基準線 v0

目的: 「体重系・騎手系の特徴を外すと、どれだけ落ちるか」を、同じ手順で学習して測る。
  - 特徴の選択はしない(2023-25の成績で選ぶと、検証期間が汚れて実戦より良く見えるため)
  - 学習=2013-2021 / 早期停止=2022 / 評価=2023・2024・2025・2026(いっさい選択に使わない)
  - 比べるもの: ALL(今のモデルの特徴の合算) と NEW(ALL から体重系・騎手系を除いたもの)
  - 指標: 本命の勝率/単勝回収/平均着順、予想順位と着順の平均のずれ、1番人気との比較

  python src/nm_baseline.py            # 5セグメント全部
  python src/nm_baseline.py 芝中       # 1セグメント
"""
import sys, os, time, pickle
import numpy as np
import pandas as pd

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from save_conditional_logit import prepare, segment_softmax, BASE_DIR, DATA_FILE
from save_v3 import add_computed_features
from search_shiba_mid_2325_v3 import adam_fit, expand_with_nan_indicators

SEGS = {
    'ダ長': ('ダ', 1400, 99999), 'ダ短': ('ダ', 0, 1400),
    '芝短': ('芝', 0, 1400), '芝中': ('芝', 1400, 2000), '芝長': ('芝', 2000, 99999),
}
PERIODS = [('2023', 230101, 240101), ('2024', 240101, 250101), ('2025', 250101, 260101), ('2026', 260101, 999999)]
L2 = 0.006


def is_removed(f: str) -> bool:
    """体重系・騎手系(ユーザー指示 2026-11-20)。"""
    return ('体重' in f) or f.startswith('騎手')


def all_features() -> list:
    m = pickle.load(open(os.path.join(BASE_DIR, 'models', 'hitrate_model.pkl'), 'rb'))
    feats = []
    for seg in SEGS:
        for f in m[seg]['feat_cols']:
            if not f.endswith('_isnan') and f not in feats:
                feats.append(f)
    return feats


def load_segment(seg: str) -> pd.DataFrame:
    surf, lo, hi = SEGS[seg]
    df = pd.read_parquet(DATA_FILE)
    df['日付_num'] = pd.to_numeric(df['日付'], errors='coerce')
    df['着順_num'] = pd.to_numeric(df['着順_num'], errors='coerce')
    df = df.dropna(subset=['日付_num', '着順_num'])
    df = df[df['着順_num'] < 99]
    df['race_id'] = (df['日付_num'].astype(int).astype(str) + '_' + df['開催'].astype(str).str.strip() + '_' + df['Ｒ'].astype(str).str.strip())
    df = df[df['開催'].notna()].copy()
    df['surface'] = df['距離'].astype(str).str.strip().str.extract(r'^([芝ダ])')[0].fillna('不明')
    dm = pd.to_numeric(df['距離'].astype(str).str.extract(r'(\d+)')[0], errors='coerce')
    df['クラス_rank'] = pd.to_numeric(df['クラス_rank'], errors='coerce')
    df = df[(df['surface'] == surf) & (dm > lo) & (dm <= hi) & (df['クラス_rank'] != 1.0)].copy()   # 新馬除外
    df = add_computed_features(df)
    baba = {'良': 0, '稍重': 1, '重': 2, '不良': 3}
    for col in df.columns:
        if '馬場状態' in col and col != '馬場状態':
            df[col] = df[col].map(baba)
    return df


def to_num(df, feats):
    for c in feats:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')


def fit_eval(df, feats):
    trn = df[(df['日付_num'] >= 130101) & (df['日付_num'] < 220101)].copy()
    val = df[(df['日付_num'] >= 220101) & (df['日付_num'] < 230101)].copy()
    tests = {n: df[(df['日付_num'] >= a) & (df['日付_num'] < b)].copy() for n, a, b in PERIODS}
    expanded = expand_with_nan_indicators([trn, val, *tests.values()], feats)
    valid = [c for c in expanded if c in trn.columns and trn[c].isna().mean() < 1.0 and trn[c].std(ddof=0) > 0]
    X_tr, y_tr, gs_tr, n_tr, nr_tr, scaler, *_ = prepare(trn, valid, top_idx=None, top_idx3=None, fit=True)
    X_va, y_va, gs_va, n_va, nr_va, *_ = prepare(val, valid, scaler=scaler, top_idx=None, top_idx3=None)
    beta = adam_fit(X_tr, y_tr, gs_tr, n_tr, nr_tr, X_va, y_va, gs_va, n_va, nr_va, l2=L2)
    out = {}
    for name, te in tests.items():
        if len(te) == 0:
            continue
        X, y, gs, n, nr, *_ = prepare(te, valid, scaler=scaler, top_idx=None, top_idx3=None)
        s = te.sort_values('race_id').reset_index(drop=True)
        s['prob'] = segment_softmax(X @ beta, gs, n)
        s['pr'] = s.groupby('race_id')['prob'].rank(ascending=False, method='first')
        s['odds'] = pd.to_numeric(s['単勝オッズ'], errors='coerce')
        s['orank'] = s.groupby('race_id')['odds'].rank(method='first')
        nll = -np.log(np.clip(s.loc[s['着順_num'] == 1, 'prob'], 1e-15, 1)).mean()
        t = s[s.pr == 1]; f = s[s.orank == 1]
        out[name] = dict(races=len(t), nll=nll,
                         win=(t['着順_num'] == 1).mean(), roi=(t.loc[t['着順_num'] == 1, 'odds'].sum() / len(t)) - 1,
                         fin=t['着順_num'].mean(), gap=(s.pr - s['着順_num']).abs().mean(),
                         fav_win=(f['着順_num'] == 1).mean(), fav_gap=(s.orank - s['着順_num']).abs().mean())
    return out, valid, beta, scaler


def main():
    segs = sys.argv[1:] or list(SEGS)
    allf = all_features()
    newf = [f for f in allf if not is_removed(f)]
    print(f'特徴(ALL)={len(allf)}個  特徴(NEW=体重・騎手を除く)={len(newf)}個')
    print('除外:', [f for f in allf if is_removed(f)])
    res = {}
    for seg in segs:
        t0 = time.time()
        df = load_segment(seg)
        to_num(df, allf)
        print(f'\n■ {seg}: {len(df):,}行 {df.race_id.nunique():,}R  ({time.time()-t0:.0f}s)', flush=True)
        for label, feats in [('ALL', allf), ('NEW', newf)]:
            t1 = time.time()
            out, valid, beta, scaler = fit_eval(df, feats)
            res[(seg, label)] = out
            print(f'  [{label}] {len(valid)}次元 ({time.time()-t1:.0f}s)')
            print(f'     {"期間":6} {"R数":>5} {"NLL":>6} {"勝率":>6} {"単勝回収":>8} {"本命平均着":>8} {"順位ずれ":>7} | {"1人気勝率":>8} {"人気ずれ":>7}')
            for p, r in out.items():
                print(f'     {p:6} {r["races"]:5d} {r["nll"]:6.3f} {r["win"]*100:5.1f}% {r["roi"]*100:+7.1f}% {r["fin"]:8.2f} {r["gap"]:7.2f} | {r["fav_win"]*100:7.1f}% {r["fav_gap"]:7.2f}')
    # まとめ(2023-2025 を R数で加重)
    print('\n=== まとめ: 2023-2025 (R数で加重平均) ===')
    print(f'{"セグ":4} {"版":3} {"勝率":>6} {"単勝回収":>8} {"順位ずれ":>7} {"NLL":>6} | {"1人気勝率":>8} {"人気ずれ":>7}')
    for seg in segs:
        for label in ('ALL', 'NEW'):
            o = res[(seg, label)]
            ps = [p for p in ('2023', '2024', '2025') if p in o]
            w = np.array([o[p]['races'] for p in ps], float)
            g = lambda k: float(np.average([o[p][k] for p in ps], weights=w))
            print(f'{seg:4} {label:3} {g("win")*100:5.1f}% {g("roi")*100:+7.1f}% {g("gap"):7.2f} {g("nll"):6.3f} | {g("fav_win")*100:7.1f}% {g("fav_gap"):7.2f}')


if __name__ == '__main__':
    main()
