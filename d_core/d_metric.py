# coding: utf-8
"""D指標 - 計算モジュール
D = sub_cs × sub_ri ÷ (cur_r × sub_r)  ※ cur_r×sub_r は 0.25 でクリップ
"""
import re, json, pickle, os
import pandas as pd
import numpy as np
from config import MODEL_DIR, SUBMODEL_DIR, RANKER_DIR, SUB_RANKER_DIR, PARQUET_PATH


def _extract_venue(k):
    m = re.search(r'\d+([^\d]+)', str(k))
    return m.group(1) if m else str(k)

def _dist_band(dist_str):
    m = re.search(r'\d+', str(dist_str))
    if not m: return None
    d = int(m.group())
    if d <= 1400:   return '短距離'
    elif d <= 1800: return 'マイル'
    elif d <= 2200: return '中距離'
    else:           return '長距離'

def _cls_group(r):
    try: r = int(float(r))
    except: return '3勝以上'
    return {1:'新馬', 2:'未勝利', 3:'1勝', 4:'2勝'}.get(r, '3勝以上')


def load_models():
    """cur/sub モデルを読み込んで返す"""
    with open(os.path.join(MODEL_DIR, 'model_info.json'), encoding='utf-8') as f:
        cur_info = json.load(f)
    with open(os.path.join(SUBMODEL_DIR, 'submodel_info.json'), encoding='utf-8') as f:
        sub_info = json.load(f)
    return cur_info, sub_info


def calc_d(df: pd.DataFrame, cur_info: dict, sub_info: dict) -> pd.DataFrame:
    """
    parquetから読み込んだDataFrameにD指標を計算して追加する。
    追加列: cur_cs, sub_cs, cur_ri, sub_ri, cur_r, sub_r, D
    """
    df = df.copy()
    cur_features = cur_info['features']
    sub_features = sub_info['features']

    all_feats = list(set(cur_features + sub_features))
    for col in all_feats:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).replace({'nan': '', 'None': ''}), errors='coerce')

    df['会場']       = df['開催'].apply(_extract_venue)
    df['cur_key']    = df['会場'] + '_' + df['距離'].astype(str)
    df['_dist_band'] = df['距離'].apply(_dist_band)
    mask = (df['芝・ダ'] == 'ダ') & (df['_dist_band'].isin(['中距離', '長距離']))
    df.loc[mask, '_dist_band'] = '中長距離'
    df['_cls_group'] = df['クラス_rank'].apply(_cls_group)
    df['sub_key']    = df['芝・ダ'].astype(str) + '_' + df['_dist_band'].astype(str) + '_' + df['_cls_group'].astype(str)
    df['race_key']   = df['_dnum'].astype(str) + '_' + df['開催'].astype(str) + '_' + df['Ｒ'].astype(str)

    for col in ['cur_prob', 'sub_prob', 'cur_cs', 'sub_cs', 'cur_ri', 'sub_ri',
                'cur_r', 'sub_r', '_cur_score', '_sub_score']:
        df[col] = np.nan

    cur_feats_avail = [c for c in cur_features if c in df.columns]
    sub_feats_avail = [c for c in sub_features if c in df.columns]

    # cur 勝率モデル
    for ck in df['cur_key'].dropna().unique():
        wf = os.path.join(MODEL_DIR, f'lgb_{ck}_win.pkl')
        if not os.path.exists(wf): continue
        idx = df[df['cur_key'] == ck].index
        with open(wf, 'rb') as f: wm = pickle.load(f)
        try:
            prob = wm.predict_proba(df.loc[idx, cur_feats_avail].values)[:, 1]
            df.loc[idx, 'cur_prob'] = prob
            st  = cur_info['models'].get(ck, {}).get('stats', {})
            w_m = st.get('win_mean', np.nanmean(prob))
            w_s = st.get('win_std',  np.nanstd(prob))
            df.loc[idx, 'cur_cs'] = 50 + 10 * (prob - w_m) / (w_s if w_s > 0 else 1)
        except: pass

    # cur ランカー
    for ck in df['cur_key'].dropna().unique():
        rf = os.path.join(RANKER_DIR, f'ranker_{ck}.pkl')
        if not os.path.exists(rf): continue
        idx = df[df['cur_key'] == ck].index
        if df.loc[idx, 'cur_prob'].isna().all(): continue
        with open(rf, 'rb') as f: rm = pickle.load(f)
        try: df.loc[idx, '_cur_score'] = rm.predict(df.loc[idx, cur_feats_avail].values)
        except: pass

    df['cur_r'] = df.groupby('race_key')['_cur_score'].rank(ascending=False, method='min')
    gm = df.groupby('race_key')['cur_prob'].transform('mean')
    gs = df.groupby('race_key')['cur_prob'].transform('std')
    df['cur_ri'] = 50 + 10 * (df['cur_prob'] - gm) / gs.clip(lower=1e-6)

    # sub 勝率モデル
    for sk in df['sub_key'].dropna().unique():
        wf = os.path.join(SUBMODEL_DIR, f'sub_{sk}_win.pkl')
        if not os.path.exists(wf): continue
        idx = df[df['sub_key'] == sk].index
        with open(wf, 'rb') as f: wm = pickle.load(f)
        try:
            prob = wm.predict_proba(df.loc[idx, sub_feats_avail].values)[:, 1]
            df.loc[idx, 'sub_prob'] = prob
            st  = sub_info['models'].get(sk, {}).get('stats', {})
            w_m = st.get('win_mean', np.nanmean(prob))
            w_s = st.get('win_std',  np.nanstd(prob))
            df.loc[idx, 'sub_cs'] = 50 + 10 * (prob - w_m) / (w_s if w_s > 0 else 1)
        except: pass

    # sub ランカー
    for sk in df['sub_key'].dropna().unique():
        rf = os.path.join(SUB_RANKER_DIR, f'class_ranker_{sk}.pkl')
        if not os.path.exists(rf): continue
        idx = df[df['sub_key'] == sk].index
        if df.loc[idx, 'sub_prob'].isna().all(): continue
        with open(rf, 'rb') as f: rm = pickle.load(f)
        try: df.loc[idx, '_sub_score'] = rm.predict(df.loc[idx, sub_feats_avail].values)
        except: pass

    df['sub_r'] = df.groupby('race_key')['_sub_score'].rank(ascending=False, method='min')
    gm = df.groupby('race_key')['sub_prob'].transform('mean')
    gs = df.groupby('race_key')['sub_prob'].transform('std')
    df['sub_ri'] = 50 + 10 * (df['sub_prob'] - gm) / gs.clip(lower=1e-6)

    # D指標
    prod_r = (df['cur_r'] * df['sub_r']).clip(lower=0.25)
    df['D'] = df['sub_cs'] * df['sub_ri'] / prod_r

    return df


def add_gap(df: pd.DataFrame, rk_col: str = 'race_key') -> pd.DataFrame:
    """D指標のS1/S2ギャップをレース単位で計算して D1位馬に付与"""
    def _gap(g):
        g2 = g.sort_values('D', ascending=False).reset_index(drop=True)
        d1 = g2.iloc[0]['D'] if len(g2) >= 1 else np.nan
        d2 = g2.iloc[1]['D'] if len(g2) >= 2 else np.nan
        return pd.Series({
            'gap_ratio': d1 / d2 if pd.notna(d2) and d2 > 0 else np.nan,
            'gap_pct':   (d1 - d2) / d2 * 100 if pd.notna(d2) and d2 > 0 else np.nan,
        })

    gap_df = df.groupby(rk_col).apply(_gap)
    df['D_rank']  = df.groupby(rk_col)['D'].rank(ascending=False, method='min')
    df['D_mean']  = df.groupby(rk_col)['D'].transform('mean').clip(lower=1)
    df['D_pct']   = (df['D'] - df['D_mean']) / df['D_mean'] * 100
    df['_n_qual'] = df.groupby(rk_col)['D_pct'].transform(lambda x: (x > 200).sum())

    top1 = df[df['D_rank'] == 1].copy()
    top1 = top1.merge(gap_df, left_on=rk_col, right_index=True, how='left')
    return top1
