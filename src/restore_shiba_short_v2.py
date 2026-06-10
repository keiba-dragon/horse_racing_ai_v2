# coding: utf-8
"""shiba_short_acc_v2を再計算して accuracy_model.pkl を修復する"""
import os, sys, pickle
import numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from save_conditional_logit import prepare, segment_softmax, BASE_DIR, DATA_FILE, LR, N_EPOCHS, PATIENCE
from save_v3 import add_computed_features

L2 = 0.006
NAN_IND_THRESHOLD = 0.05

FEATS = ['馬番','斤量','芝ダ一致_平均着順_近5走','1走前_タイム指数',
         '近5走_クラス調整_平均着順','馬コース_r20_勝率','騎手コース_r100_勝率',
         '近10走_勝率','馬体重','近5走_上り3F平均','芝ダ転向',
         '近3走_体重増減合計','相手レベル_平均着順','道悪_平均着順_近5走',
         'タイム指数_近3走_slope','馬体重増減','コース馬場_r200_勝率',
         'ブリンカー変更','3走前_クラス差','近5走_上り3F_std',
         '騎手コース距離_r100_勝率','2走前_クラス差','着順_近3走_slope',
         '近3走_複勝率','馬距離_勝率','1走前_3角','1走前_馬場状態',
         'コース枠_r200_勝率','前走着差タイム','馬_r20_勝率','2走前_着順_num']

def expand_nan_ind(dfs, feats):
    ref = dfs[0]; extended = []
    for f in feats:
        extended.append(f)
        if f not in ref.columns: continue
        if NAN_IND_THRESHOLD < ref[f].isna().mean() < 1.0:
            ind = f+'_isnan'
            for df in dfs:
                if f in df.columns and ind not in df.columns:
                    df[ind] = df[f].isna().astype(float)
            extended.append(ind)
    return extended

def _loss_grad(beta, X, y, gs, n, nr):
    probs = segment_softmax(X@beta, gs, n)
    loss = -np.sum(y*np.log(np.clip(probs,1e-15,1.0)))/nr + L2*np.dot(beta,beta)
    grad = -(X.T@(y-probs))/nr + 2*L2*beta
    return loss, grad

def adam_fit(X_tr, y_tr, gs_tr, n_tr, nr_tr, X_va, y_va, gs_va, n_va, nr_va):
    d = X_tr.shape[1]; beta,m,v = np.zeros(d),np.zeros(d),np.zeros(d)
    b1,b2,eps = 0.9,0.999,1e-8; t,best_val,best_beta,no_imp = 0,np.inf,np.zeros(d),0
    for epoch in range(1,N_EPOCHS+1):
        _,grad = _loss_grad(beta,X_tr,y_tr,gs_tr,n_tr,nr_tr)
        t+=1; m=b1*m+(1-b1)*grad; v=b2*v+(1-b2)*grad**2
        beta -= LR*(m/(1-b1**t))/(np.sqrt(v/(1-b2**t))+eps)
        if epoch%10==0:
            vl,_ = _loss_grad(beta,X_va,y_va,gs_va,n_va,nr_va)
            if vl<best_val: best_val,best_beta,no_imp=vl,beta.copy(),0
            else: no_imp+=1
            if no_imp>=PATIENCE//10: break
    return best_beta

df=pd.read_parquet(DATA_FILE)
df['日付_num']=pd.to_numeric(df['日付'],errors='coerce')
df['着順_num']=pd.to_numeric(df['着順_num'],errors='coerce')
df=df.dropna(subset=['日付_num','着順_num']); df=df[df['着順_num']<99]
df['race_id']=(df['日付_num'].astype(int).astype(str)+'_'+
               df['開催'].astype(str).str.strip()+'_'+df['Ｒ'].astype(str).str.strip())
df=df[df['開催'].notna()].copy()
df['surface']=df['距離'].astype(str).str.strip().str.extract(r'^([芝ダ])')[0].fillna('不明')
dm=pd.to_numeric(df['距離'].astype(str).str.extract(r'(\d+)')[0],errors='coerce')
df['クラス_rank']=pd.to_numeric(df['クラス_rank'],errors='coerce')
df=add_computed_features(df)
baba_map={'良':0,'稍重':1,'重':2,'不良':3}
for col in df.columns:
    if '馬場状態' in col and col!='馬場状態': df[col]=df[col].map(baba_map)

seg=df[(df['surface']=='芝')&(dm<=1400)&(df['クラス_rank']!=1.0)].copy()
seg['dist_m']=dm[seg.index]

df_trn=seg[(seg['日付_num']>=130101)&(seg['日付_num']<220101)]
df_val=seg[(seg['日付_num']>=220101)&(seg['日付_num']<=221231)]
oos_2324=seg[(seg['日付_num']>=230101)&(seg['日付_num']<250101)]
oos_2025=seg[(seg['日付_num']>=250101)&(seg['日付_num']<260101)]
oos_2026=seg[seg['日付_num']>=260101]

all_dfs=[df_trn,df_val,oos_2324,oos_2025,oos_2026]
expanded=expand_nan_ind(all_dfs, FEATS)
valid=[c for c in expanded if c in df_trn.columns
       and df_trn[c].isna().mean()<1.0 and df_trn[c].std(ddof=0)>0]
print(f'有効特徴量: {len(valid)}')

X_tr,y_tr,gs_tr,n_tr,nr_tr,scaler,*_=prepare(df_trn,valid,top_idx=None,top_idx3=None,fit=True)
X_va,y_va,gs_va,n_va,nr_va,*_=prepare(df_val,valid,scaler=scaler,top_idx=None,top_idx3=None)
beta=adam_fit(X_tr,y_tr,gs_tr,n_tr,nr_tr,X_va,y_va,gs_va,n_va,nr_va)
val_s=df_val.sort_values('race_id').reset_index(drop=True)
raw_val=segment_softmax(X_va@beta,gs_va,n_va)
y_val=(val_s['着順_num']==1).astype(float).values
iso=IsotonicRegression(out_of_bounds='clip'); iso.fit(raw_val,y_val)

results={}
for label,oos in [('2324',oos_2324),('2025',oos_2025),('2026',oos_2026)]:
    if len(oos)==0: continue
    vp=[c for c in valid if c in oos.columns]
    X_p,_,gs_p,n_p,*_=prepare(oos,vp,scaler=scaler,top_idx=None,top_idx3=None)
    s=oos.sort_values('race_id').reset_index(drop=True)
    s['prob']=segment_softmax(X_p@beta,gs_p,n_p)
    s['rank']=s.groupby('race_id')['prob'].rank(ascending=False,method='first')
    t=s[s['rank']==1]; nr=s['race_id'].nunique()
    acc=(t['着順_num']==1).mean()
    odds=pd.to_numeric(t['単勝オッズ'],errors='coerce')
    roi=(odds[t['着順_num']==1]*100).sum()/(len(t)*100)-1
    results[label]=(acc,roi,nr)
    print(f'{label}: acc={acc:.2%} ROI={roi:+.2%} ({nr}R)')

n2324=results.get('2324',(0,0,0))[2]; n25=results.get('2025',(0,0,0))[2]; n26=results.get('2026',(0,0,0))[2]
a2324=results.get('2324',(0,0,0))[0]; a25=results.get('2025',(0,0,0))[0]; a26=results.get('2026',(0,0,0))[0]
r25=results.get('2025',(0,0,0))[1]; r26=results.get('2026',(0,0,0))[1]
acc_2325=(a2324*n2324+a25*n25)/(n2324+n25)
acc_2526=(a25*n25+a26*n26)/(n25+n26) if (n25+n26)>0 else 0.0
roi_2526=(r25*n25+r26*n26)/(n25+n26) if (n25+n26)>0 else 0.0
print(f'acc_2325={acc_2325:.4f}  25+26_acc={acc_2526:.4f}  ROI={roi_2526:+.2%}')

pkg={'segment':'芝短','scaler':scaler,'coef':beta,'feat_cols':valid,'isotonic':iso,
     'acc_2325':acc_2325,'acc_2526':acc_2526,'oos_roi_2526':roi_2526,
     'version':'shiba_short_acc_v2','note':f'shiba_short_acc_v2 restored: acc_2325={acc_2325:.4f}'}
acc_pkl=os.path.join(BASE_DIR,'models','accuracy_model.pkl')
m=pickle.load(open(acc_pkl,'rb')); m['芝短']=pkg
with open(acc_pkl,'wb') as f: pickle.dump(m,f)
print('芝短 shiba_short_acc_v2 を修復しました')
