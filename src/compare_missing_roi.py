# coding: utf-8
import pickle, pandas as pd, numpy as np, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, 'src')

from save_conditional_logit import add_new_features, segment_softmax, prepare
from save_lambdarank_pace import add_pace_features

DATA_FILE = 'data/processed/all_venues_features.parquet'
print('データ読み込み中...')
df = pd.read_parquet(DATA_FILE)
df['日付_num'] = pd.to_numeric(df['日付'], errors='coerce')
df['着順_num'] = pd.to_numeric(df['着順_num'], errors='coerce')
df = df.dropna(subset=['日付_num', '着順_num'])
df = df[df['着順_num'] < 99]
df['race_id'] = (df['日付_num'].astype(int).astype(str) + '_' +
                 df['開催'].astype(str).str.strip() + '_' +
                 df['Ｒ'].astype(str).str.strip())
df = add_pace_features(df)
df = add_new_features(df)

oos = df[df['日付_num'] >= 230101].copy()
oos['odds_num'] = pd.to_numeric(oos['単勝オッズ'], errors='coerce')
oos['yr'] = (oos['日付_num'] // 10000).astype(int)
print(f'OOS: {len(oos):,}行 / {oos["race_id"].nunique():,}レース')

with open('models/final_model.pkl', 'rb') as f:
    pkg = pickle.load(f)

MISSING_AT_PRED = [
    '前走走破タイム_sec', '今回_surface', '今回_距離_m',
    'レース印１', '前走レース印１', '斤量', '休養日数', '距離変化_m'
]
MISSING_AFTER_ALIAS = [
    'レース印１', '前走レース印１', '斤量', '休養日数', '距離変化_m'
]


def roi_eval(oos, pkg, zero_cols=None, label=''):
    # prepare はrace_idソートするので、surface別に処理してから元indexで戻す
    all_scores = pd.Series(np.nan, index=oos.index)

    for surf in ['芝', 'ダ']:
        art = pkg['artifacts'][surf]
        mask = oos['距離'].astype(str).str.strip().str[:1] == surf
        sub = oos[mask].copy()
        if len(sub) == 0:
            continue
        s2 = sub.copy()
        if zero_cols:
            for c in zero_cols:
                if c in s2.columns:
                    s2[c] = np.nan
        for c in art['feat_cols']:
            if c not in s2.columns:
                s2[c] = np.nan
        try:
            # prepare内部でrace_idソートされる → ソート後の順序でscoreが返る
            s2_sorted = s2.sort_values('race_id').reset_index()  # 元indexを保存
            orig_idx = s2_sorted['index'].values
            s2_for_prep = s2_sorted.drop(columns=['index'])

            X, _, gs, n, *_ = prepare(
                s2_for_prep, art['feat_cols'],
                scaler=art['scaler'], poly2=art['poly2'],
                inter_scaler2=art['inter_scaler2'], top_idx=art['top_idx'],
                poly3=None, inter_scaler3=None, top_idx3=None, fit=False)
            raw = segment_softmax(X @ art['coef'], gs, n)
            calib = art['isotonic'].predict(raw)
            odds = pd.to_numeric(s2_for_prep['単勝オッズ'], errors='coerce').values
            mprob = 1.0 / np.clip(odds, 1.0, None)
            cls = pd.to_numeric(
                s2_for_prep.get('クラス_rank', pd.Series([0]*len(s2_for_prep))),
                errors='coerce').fillna(0).values
            factor = np.where(cls == 2, pkg['factor_maiden'], pkg['factor_other'])
            score = calib - factor * mprob
            # ソート前の元indexに戻してセット
            all_scores.loc[orig_idx] = score
        except Exception as e:
            print(f'  ERROR {surf}: {e}')

    oos2 = oos.copy()
    oos2['_score'] = all_scores
    oos2 = oos2[oos2['_score'].notna()]
    oos2['_rank'] = oos2.groupby('race_id')['_score'].rank(
        ascending=False, method='first')
    top1 = oos2[oos2['_rank'] == 1]
    won = top1['着順_num'] == 1
    roi = (top1.loc[won, 'odds_num'] * 100).sum() / (len(top1) * 100) - 1
    print(f'{label}: {len(top1)}R  win={won.mean():.3f}  ROI={roi:+.4f}')
    return roi


print()
r_full  = roi_eval(oos, pkg, zero_cols=None,                label='① フル（全特徴量あり）        ')
r_cur   = roi_eval(oos, pkg, zero_cols=MISSING_AT_PRED,     label='② 現状（重要欠損も0埋め）     ')
r_alias = roi_eval(oos, pkg, zero_cols=MISSING_AFTER_ALIAS, label='③ エイリアス補完後（残り0埋め）')

print()
print(f'① → ② 劣化: {r_cur   - r_full:+.4f}  ({(r_cur   - r_full)*100:+.2f}%pt)')
print(f'② → ③ 改善: {r_alias - r_cur :+.4f}  ({(r_alias - r_cur )*100:+.2f}%pt)')
print(f'① → ③ 残差: {r_alias - r_full:+.4f}  ({(r_alias - r_full)*100:+.2f}%pt)')
