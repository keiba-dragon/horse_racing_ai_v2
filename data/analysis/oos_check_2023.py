# coding: utf-8
"""OOS確認: 2023 / 2025 / 2026 のD指標ROI比較
再学習後（TRAIN_START=240101）に実行して、2023がインサンプスではないか確認する。
"""
import sys, io, re, json, pickle, os, warnings
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
import pandas as pd, numpy as np

MODEL_DIR = 'models'

def extract_venue(k):
    m = re.search(r'\d+([^\d]+)', str(k)); return m.group(1) if m else str(k)
def get_distance_band(d):
    m = re.search(r'\d+', str(d))
    if not m: return None
    d = int(m.group())
    return '短距離' if d<=1400 else 'マイル' if d<=1800 else '中距離' if d<=2200 else '長距離'
def get_class_group(r):
    try: r=int(float(r))
    except: return '3勝以上'
    return {1:'新馬',2:'未勝利',3:'1勝',4:'2勝'}.get(r,'3勝以上')

with open(os.path.join(MODEL_DIR,'model_info.json'),encoding='utf-8') as f: cur_info=json.load(f)
with open(os.path.join(MODEL_DIR,'submodel','submodel_info.json'),encoding='utf-8') as f: sub_info=json.load(f)
cur_features=cur_info['features']; sub_features=sub_info['features']
cur_models_meta=cur_info['models']; sub_models_meta=sub_info['models']

df=pd.read_parquet('data/processed/all_venues_features.parquet')
dnum_col='日付_num' if '日付_num' in df.columns else '日付'
df['_dnum']=pd.to_numeric(df[dnum_col],errors='coerce')
df=df[df['_dnum']>=230101].reset_index(drop=True)  # 2023年全体を対象
for col in list(set(cur_features+sub_features)):
    if col in df.columns:
        df[col]=pd.to_numeric(df[col].astype(str).replace({'nan':'','None':''}),errors='coerce')
df['会場']=df['開催'].apply(extract_venue)
df['cur_key']=df['会場']+'_'+df['距離'].astype(str)
df['_dist_band']=df['距離'].apply(get_distance_band)
mask=(df['芝・ダ']=='ダ')&(df['_dist_band'].isin(['中距離','長距離']))
df.loc[mask,'_dist_band']='中長距離'
df['_cls_group']=df['クラス_rank'].apply(get_class_group)
df['sub_key']=df['芝・ダ'].astype(str)+'_'+df['_dist_band'].astype(str)+'_'+df['_cls_group'].astype(str)
df['race_key']=df['_dnum'].astype(str)+'_'+df['開催'].astype(str)+'_'+df['Ｒ'].astype(str)
for col in ['cur_prob','sub_prob','cur_cs','sub_cs','cur_ri','sub_ri','cur_r','sub_r','_cur_sc','_sub_sc']:
    df[col]=np.nan

cur_feats_avail=[c for c in cur_features if c in df.columns]
for ck in df['cur_key'].dropna().unique():
    wf=os.path.join(MODEL_DIR,f'lgb_{ck}_win.pkl')
    if not os.path.exists(wf): continue
    idx=df[df['cur_key']==ck].index
    with open(wf,'rb') as f: wm=pickle.load(f)
    try:
        prob=wm.predict_proba(df.loc[idx,cur_feats_avail].values)[:,1]; df.loc[idx,'cur_prob']=prob
        st=cur_models_meta.get(ck,{}).get('stats',{}); w_m=st.get('win_mean',np.nanmean(prob)); w_s=st.get('win_std',np.nanstd(prob))
        df.loc[idx,'cur_cs']=50+10*(prob-w_m)/(w_s if w_s>0 else 1)
    except: pass
for ck in df['cur_key'].dropna().unique():
    rf=os.path.join(MODEL_DIR,'ranker',f'ranker_{ck}.pkl')
    if not os.path.exists(rf): continue
    idx=df[df['cur_key']==ck].index
    if df.loc[idx,'cur_prob'].isna().all(): continue
    with open(rf,'rb') as f: rm=pickle.load(f)
    try: df.loc[idx,'_cur_sc']=rm.predict(df.loc[idx,cur_feats_avail].values)
    except: pass
df['cur_r']=df.groupby('race_key')['_cur_sc'].rank(ascending=False,method='min')
gm=df.groupby('race_key')['cur_prob'].transform('mean'); gs_s=df.groupby('race_key')['cur_prob'].transform('std')
df['cur_ri']=50+10*(df['cur_prob']-gm)/gs_s.clip(lower=1e-6)

sub_feats_avail=[c for c in sub_features if c in df.columns]
for sk in df['sub_key'].dropna().unique():
    wf=os.path.join(MODEL_DIR,'submodel',f'sub_{sk}_win.pkl')
    if not os.path.exists(wf): continue
    idx=df[df['sub_key']==sk].index
    with open(wf,'rb') as f: wm=pickle.load(f)
    try:
        prob=wm.predict_proba(df.loc[idx,sub_feats_avail].values)[:,1]; df.loc[idx,'sub_prob']=prob
        st=sub_models_meta.get(sk,{}).get('stats',{}); w_m=st.get('win_mean',np.nanmean(prob)); w_s=st.get('win_std',np.nanstd(prob))
        df.loc[idx,'sub_cs']=50+10*(prob-w_m)/(w_s if w_s>0 else 1)
    except: pass
for sk in df['sub_key'].dropna().unique():
    rf=os.path.join(MODEL_DIR,'submodel_ranker',f'class_ranker_{sk}.pkl')
    if not os.path.exists(rf): continue
    idx=df[df['sub_key']==sk].index
    if df.loc[idx,'sub_prob'].isna().all(): continue
    with open(rf,'rb') as f: rm=pickle.load(f)
    try: df.loc[idx,'_sub_sc']=rm.predict(df.loc[idx,sub_feats_avail].values)
    except: pass
df['sub_r']=df.groupby('race_key')['_sub_sc'].rank(ascending=False,method='min')
gm=df.groupby('race_key')['sub_prob'].transform('mean'); gs_s=df.groupby('race_key')['sub_prob'].transform('std')
df['sub_ri']=50+10*(df['sub_prob']-gm)/gs_s.clip(lower=1e-6)

prod_r=(df['cur_r']*df['sub_r']).clip(lower=0.25)
df['D']=df['sub_cs']*df['sub_ri']/prod_r
df['D_rank']=df.groupby('race_key')['D'].rank(ascending=False,method='min')
df['D_mean']=df.groupby('race_key')['D'].transform('mean').clip(lower=1)
df['D_pct']=(df['D']-df['D_mean'])/df['D_mean']*100
df['単勝オッズ']=pd.to_numeric(df['単勝オッズ'],errors='coerce')

with open('data/raw/2023年～の結果.csv','rb') as f: raw=f.read()
res=pd.read_csv(pd.io.common.BytesIO(raw),encoding='cp932')
res.columns=res.columns.str.strip()
def zen(s):
    if pd.isna(s): return np.nan
    s=str(s).strip().translate(str.maketrans('０１２３４５６７８９','0123456789'))
    m=re.search(r'\d+',s); return int(m.group()) if m else np.nan
res['着_num']=res['着順'].apply(zen)
res['_dnum']=pd.to_numeric(res['日付'],errors='coerce').astype('Int64')
res['_venue']=res['開催'].apply(extract_venue)
res['_R']=pd.to_numeric(res['Ｒ'],errors='coerce')
res['_tan']=pd.to_numeric(res['単勝配当'],errors='coerce')
tan_race=(res[res['着_num']==1].groupby(['_dnum','_venue','_R'])['_tan'].first()
          .reset_index().rename(columns={'_tan':'_tan_race'}))
res=res.merge(tan_race,on=['_dnum','_venue','_R'],how='left')
res['_fuku_use']=pd.to_numeric(res['複勝配当'],errors='coerce').where(res['着_num']<=3)

df['_dnum_k']=df['_dnum'].astype(int)
df['_venue_k']=df['会場'].astype(str)
df['_R_k']=pd.to_numeric(df['Ｒ'],errors='coerce')
res_sub=res[['_dnum','_venue','_R','馬名S','着_num','_tan_race','_fuku_use']].copy()
res_sub['_dnum']=res_sub['_dnum'].astype(int)
merged=df.merge(res_sub,left_on=['_dnum_k','_venue_k','_R_k','馬名S'],
                right_on=['_dnum','_venue','_R','馬名S'],how='inner')

m23 = merged[merged['_dnum_k'].between(230101,231231)].copy()
m25 = merged[merged['_dnum_k'].between(250101,251231)].copy()
m26 = merged[merged['_dnum_k']>=260101].copy()
print(f"2023: {len(m23)}頭 / {m23['race_key'].nunique()}レース  ← 新モデルのOOS")
print(f"2025: {len(m25)}頭 / {m25['race_key'].nunique()}レース")
print(f"2026: {len(m26)}頭 / {m26['race_key'].nunique()}レース")

def roi_tan(sub):
    if len(sub)<5: return np.nan, np.nan, np.nan
    sub_t=sub.dropna(subset=['_tan_race'])
    if len(sub_t)==0: return np.nan,np.nan,np.nan
    roi=(sub_t[sub_t['着_num']==1]['_tan_race'].sum()/100-len(sub_t))/len(sub_t)
    wr=(sub['着_num']==1).mean()
    return roi, wr, len(sub)

def roi_fuku(sub):
    if len(sub)<5: return np.nan, np.nan, np.nan
    hits=sub[sub['着_num']<=3].dropna(subset=['_fuku_use'])
    roi=(hits['_fuku_use'].sum()/100-len(sub))/len(sub)
    pr=(sub['着_num']<=3).mean()
    return roi, pr, len(sub)

print("\n" + "="*70)
print("【D指標 ◎○▲ 単勝ROI（2023/2025/2026比較）】")
print("2023 = 新モデルOOS（学習に未使用）/ 2025,2026 = 従来通りOOS")
print("="*70)

# 現行の閾値をそのまま使って3年比較
rules = [
    ('◎ D1位 od>8 D%>200', lambda d: d[(d['D_rank']==1)&(d['単勝オッズ']>8)&(d['D_pct']>200)]),
    ('○ D2位 od>6 D%>100', lambda d: d[(d['D_rank']==2)&(d['単勝オッズ']>6)&(d['D_pct']>100)]),
    ('▲ D3位 od>5 D%>30',  lambda d: d[(d['D_rank']==3)&(d['単勝オッズ']>5)&(d['D_pct']>30)]),
    ('D1位 全オッズ',        lambda d: d[d['D_rank']==1]),
]

hdr = f"  {'条件':<22}  {'N23':>5}  {'ROI23':>8}  {'N25':>5}  {'ROI25':>8}  {'N26':>5}  {'ROI26':>8}"
print(hdr)
print("  " + "-"*68)
for label, filt in rules:
    r23, w23, n23 = roi_tan(filt(m23))
    r25, w25, n25 = roi_tan(filt(m25))
    r26, w26, n26 = roi_tan(filt(m26))
    n23s = f'{n23:>5}' if not np.isnan(n23) else '    -'
    n25s = f'{n25:>5}' if not np.isnan(n25) else '    -'
    n26s = f'{n26:>5}' if not np.isnan(n26) else '    -'
    r23s = f'{r23:>+8.1%}' if not np.isnan(r23) else '       -'
    r25s = f'{r25:>+8.1%}' if not np.isnan(r25) else '       -'
    r26s = f'{r26:>+8.1%}' if not np.isnan(r26) else '       -'
    print(f"  {label:<22}  {n23s}  {r23s}  {n25s}  {r25s}  {n26s}  {r26s}")

print()
print("="*70)
print("【複勝ROI（2023/2025/2026比較）】")
print("="*70)
print(hdr)
print("  " + "-"*68)
for label, filt in rules:
    r23, w23, n23 = roi_fuku(filt(m23))
    r25, w25, n25 = roi_fuku(filt(m25))
    r26, w26, n26 = roi_fuku(filt(m26))
    n23s = f'{n23:>5}' if not np.isnan(n23) else '    -'
    n25s = f'{n25:>5}' if not np.isnan(n25) else '    -'
    n26s = f'{n26:>5}' if not np.isnan(n26) else '    -'
    r23s = f'{r23:>+8.1%}' if not np.isnan(r23) else '       -'
    r25s = f'{r25:>+8.1%}' if not np.isnan(r25) else '       -'
    r26s = f'{r26:>+8.1%}' if not np.isnan(r26) else '       -'
    print(f"  {label:<22}  {n23s}  {r23s}  {n25s}  {r25s}  {n26s}  {r26s}")
