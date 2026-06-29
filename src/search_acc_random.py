# coding: utf-8
"""
search_acc_random.py - ランダムサーチ (greedy 局所最適を脱出)
usage: python search_acc_random.py ダ長 500
  セグメント名とイテレーション数を指定（省略時: 芝長 300）
戦略: SEEDから始め、特徴を1つ追加/削除/入れ替えをランダムに試し
      改善したら更新（ランダム局所探索）
"""
import sys, os, time, pickle, random
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from save_conditional_logit import prepare, segment_softmax, BASE_DIR, DATA_FILE, LR, N_EPOCHS, PATIENCE
from save_v3 import add_computed_features

SEG_NAME = sys.argv[1] if len(sys.argv) > 1 else '芝長'
N_ITER   = int(sys.argv[2]) if len(sys.argv) > 2 else 300
L2 = 0.006
NAN_IND_THRESHOLD = 0.05
SEED_RNG = int(sys.argv[3]) if len(sys.argv) > 3 else 42
MIN_FEATS = 10
MAX_FEATS = 50

SEEDS = {
    'ダ長': ['馬番','斤量','近3走_複勝率','騎手コース_r100_勝率','1走前_クラス調整着順',
             '近5走_タイム指数_max','馬距離_勝率','種牡馬_勝率','タイム指数_加速度',
             '近5走_タイム指数平均','近5走_上り3F平均','近5走_上り3F_std','1走前_クラス差',
             'ブリンカー変更','1走前_3角','間隔','距離変化_前走','前走着差タイム','騎手変更',
             '輸送有無','コース枠_r200_勝率','馬体重増減','コース脚質_r200_勝率','1走前_馬場状態',
             '1走前_脚質_num','近10走_勝率','2走前_クラス差','道悪_平均着順_近5走',
             'コース馬場_r200_勝率','種牡馬_ダ_勝率','近3走_勝率','コース枠_r200_複勝率',
             '馬_r20_勝率','性別_num','騎手会場_r100_勝率','相手レベル_平均着順','1走前_4角'],
    'ダ短': ['馬番','斤量','芝ダ一致_平均着順_近5走','1走前_タイム指数',
             '近5走_クラス調整_平均着順','輸送有無','馬コース_r20_勝率',
             '近3走_体重増減合計','性別_num','1走前_クラス差','コース枠_r200_勝率',
             '近10走_勝率','調教師_r200_勝率','ブリンカー変更','相手レベル_平均着順',
             'コース馬場_r200_勝率','種牡馬_ダ_勝率','近5走_上り3F平均','近5走_タイム指数_max',
             '1走前_馬場状態','タイム指数_近5走_slope','上り3F_近3走_slope','近3走_勝率',
             'タイム指数_加速度','1走前_クラス調整着順','距離変化_前走','間隔',
             '騎手馬場_r100_勝率','近5走_タイム指数平均','馬体重増減','近5走_上り3F_std',
             '着順_近3走_slope','馬距離_勝率','母父馬_勝率','前走着差タイム','騎手変更',
             '1走前_頭数','1走前_上り3F'],
    '芝短': ['馬番','斤量','1走前_馬場状態','近10走_複勝率','種牡馬_ダ_勝率',
             '馬体重','近10走_勝率','3走前_上り3F','騎手会場_r100_勝率','馬距離_勝率',
             '1走前_タイム指数','1走前_単勝オッズ','1走前_PCI','騎手コース_r100_勝率',
             '相手レベル_平均着順','前走着差タイム','近5走_クラス調整_平均着順',
             '1走前_3角','1走前_頭数','馬_r20_勝率','1走前_クラス調整着順',
             '同距離帯_平均着順_近5走','1走前_4角','近5走_上り3F平均','間隔',
             '3走前_タイム指数','騎手変更','同会場_複勝率_近5走','芝ダ転向','コース枠_r200_勝率'],
    '芝中': ['馬番','斤量','芝ダ一致_平均着順_近5走','騎手距離_r100_勝率',
             '1走前_タイム指数','1走前_クラス調整着順','馬コース_r20_勝率',
             '種牡馬_ダ_勝率','馬体重','前走着差タイム','近5走_上り3F平均',
             '近5走_タイム指数平均','1走前_クラス差','同馬場_平均着順_近5走',
             'コース枠_r200_勝率','輸送有無','性別_num','1走前_馬場状態',
             'ブリンカー変更','コース馬場_r200_勝率','芝ダ転向','近3走_体重増減合計',
             '間隔','相手レベル_平均着順','タイム指数_加速度','母父馬_勝率',
             '近3走_勝率','騎手変更','1走前_4角','着順_近3走_slope',
             '上り3F_近3走_slope','1走前_3角','近5走_上り3F_std',
             '近5走_タイム指数_max','タイム指数_近5走_slope','1走前_単勝オッズ','近5走_タイム指数_min'],
    '芝長': ['近3走_複勝率','騎手距離_r100_勝率','近5走_タイム指数平均',
             '馬コース_r20_勝率','タイム指数_近3走_slope','調教師コース_r100_勝率',
             '同会場_複勝率_近5走','近5走_上り3F_std','コース枠_r200_複勝率',
             '相手レベル_平均着順','タイム指数_加速度','近10走_勝率','近3走_体重増減合計',
             'コース馬場_r200_勝率','種牡馬_ダ_勝率','タイム指数_近5走_slope',
             '道悪_平均着順_近5走','1走前_タイム指数','距離変化_前走','性別_num',
             '馬体重','馬体重増減','調教師_r200_勝率','輸送有無','コース枠_r200_勝率',
             '種牡馬_勝率','馬番','斤量','近5走_タイム指数_min'],
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
]

FAV = {'ダ長':0.3403,'ダ短':0.3490,'芝短':0.2869,'芝中':0.3321,'芝長':0.3605}
VERSION_MAP = {
    'ダ長': 'da_long_acc_rand',
    'ダ短': 'da_short_acc_rand',
    '芝短': 'shiba_short_acc_rand',
    '芝中': 'shiba_mid_acc_rand',
    '芝長': 'shiba_long_acc_rand',
}


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


def eval_feats(feats, dfs):
    df_trn,df_val,oos_2324,oos_2025 = dfs
    all_dfs = list(dfs)
    expanded = expand_nan_ind(all_dfs, feats)
    valid = [c for c in expanded if c in df_trn.columns
             and df_trn[c].isna().mean()<1.0 and df_trn[c].std(ddof=0)>0]
    if len(valid)<2: return float('-inf')
    try:
        X_tr,y_tr,gs_tr,n_tr,nr_tr,sc,*_ = prepare(df_trn,valid,top_idx=None,top_idx3=None,fit=True)
        X_va,y_va,gs_va,n_va,nr_va,*_ = prepare(df_val,valid,scaler=sc,top_idx=None,top_idx3=None)
        beta = adam_fit(X_tr,y_tr,gs_tr,n_tr,nr_tr,X_va,y_va,gs_va,n_va,nr_va)
    except: return float('-inf')
    def acc_oos(oos):
        vp = [c for c in valid if c in oos.columns]
        X_p,_,gs_p,n_p,*_ = prepare(oos,vp,scaler=sc,top_idx=None,top_idx3=None)
        s = oos.sort_values('race_id').reset_index(drop=True)
        s['prob'] = segment_softmax(X_p@beta,gs_p,n_p)
        s['rank'] = s.groupby('race_id')['prob'].rank(ascending=False,method='first')
        t = s[s['rank']==1]; nr = s['race_id'].nunique()
        return (t['着順_num']==1).mean(), nr
    a2324,n2324 = acc_oos(oos_2324); a25,n25 = acc_oos(oos_2025)
    return (a2324*n2324+a25*n25)/(n2324+n25) if (n2324+n25)>0 else float('-inf')


def save_seg(name, feats, seg, version_name):
    df_trn=seg[(seg['日付_num']>=130101)&(seg['日付_num']<220101)]
    df_val=seg[(seg['日付_num']>=220101)&(seg['日付_num']<=221231)]
    oos_2324=seg[(seg['日付_num']>=230101)&(seg['日付_num']<250101)]
    oos_2025=seg[(seg['日付_num']>=250101)&(seg['日付_num']<260101)]
    oos_2026=seg[seg['日付_num']>=260101]
    all_dfs=[df_trn,df_val,oos_2324,oos_2025,oos_2026]
    expanded = expand_nan_ind(all_dfs, feats)
    valid = [c for c in expanded if c in df_trn.columns
             and df_trn[c].isna().mean()<1.0 and df_trn[c].std(ddof=0)>0]
    X_tr,y_tr,gs_tr,n_tr,nr_tr,scaler,*_ = prepare(df_trn,valid,top_idx=None,top_idx3=None,fit=True)
    X_va,y_va,gs_va,n_va,nr_va,*_ = prepare(df_val,valid,scaler=scaler,top_idx=None,top_idx3=None)
    beta = adam_fit(X_tr,y_tr,gs_tr,n_tr,nr_tr,X_va,y_va,gs_va,n_va,nr_va)
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
        print(f'  {label}: acc={acc:.2%} ROI={roi:+.2%} ({nr}R)')
    n2324,n25,n26=results.get('2324',(0,0,0))[2],results.get('2025',(0,0,0))[2],results.get('2026',(0,0,0))[2]
    a2324,a25,a26=results.get('2324',(0,0,0))[0],results.get('2025',(0,0,0))[0],results.get('2026',(0,0,0))[0]
    r25,r26=results.get('2025',(0,0,0))[1],results.get('2026',(0,0,0))[1]
    acc_2325=(a2324*n2324+a25*n25)/(n2324+n25) if (n2324+n25)>0 else 0.0
    acc_2526=(a25*n25+a26*n26)/(n25+n26) if (n25+n26)>0 else 0.0
    roi_2526=(r25*n25+r26*n26)/(n25+n26) if (n25+n26)>0 else 0.0
    print(f'  acc_2325={acc_2325:.4f}  25+26_acc={acc_2526:.4f}  ROI={roi_2526:+.2%}')
    acc_pkg={
        'segment':name,'scaler':scaler,'coef':beta,'feat_cols':valid,'isotonic':iso,
        'acc_2325':acc_2325,'acc_2526':acc_2526,'oos_roi_2526':roi_2526,
        'version':version_name,
        'note':f'{version_name}: {len(feats)}特徴 acc_2325={acc_2325:.4f} 1番人気{FAV[name]:.2%}',
    }
    acc_pkl=os.path.join(BASE_DIR,'models','hitrate_model.pkl')
    existing=pickle.load(open(acc_pkl,'rb'))
    existing[name]=acc_pkg
    with open(acc_pkl,'wb') as f: pickle.dump(existing,f)
    print(f'  保存: {name} {version_name}')
    return acc_2325


def main():
    rng = random.Random(SEED_RNG)
    print(f'ランダムサーチ: {SEG_NAME}  N_ITER={N_ITER}  L2={L2}')

    df=pd.read_parquet(DATA_FILE)
    df['日付_num']=pd.to_numeric(df['日付'],errors='coerce')
    df['着順_num']=pd.to_numeric(df['着順_num'],errors='coerce')
    df=df.dropna(subset=['日付_num','着順_num'])
    df=df[df['着順_num']<99]
    df['race_id']=(df['日付_num'].astype(int).astype(str)+'_'+
                   df['開催'].astype(str).str.strip()+'_'+df['Ｒ'].astype(str).str.strip())
    df=df[df['開催'].notna()].copy()
    df['surface']=df['距離'].astype(str).str.strip().str.extract(r'^([芝ダ])')[0].fillna('不明')
    dm=pd.to_numeric(df['距離'].astype(str).str.extract(r'(\d+)')[0],errors='coerce')
    df['クラス_rank']=pd.to_numeric(df['クラス_rank'],errors='coerce')
    df=add_computed_features(df)
    if '今回_会場' in df.columns and '1走前_開催' in df.columns:
        df['輸送有無']=(df['今回_会場'].astype(str)!=df['1走前_開催'].astype(str).str[1]).astype(float)
        df.loc[df['1走前_開催'].isna(),'輸送有無']=float('nan')
    baba_map={'良':0,'稍重':1,'重':2,'不良':3}
    for col in df.columns:
        if '馬場状態' in col and col!='馬場状態': df[col]=df[col].map(baba_map)

    s=df['surface']; r=df['クラス_rank']
    name = SEG_NAME
    if name=='ダ長': mask=(s=='ダ')&(dm>1400)&(r!=1.0)
    elif name=='ダ短': mask=(s=='ダ')&(dm<=1400)&(r!=1.0)
    elif name=='芝短': mask=(s=='芝')&(dm<=1400)&(r!=1.0)
    elif name=='芝中': mask=(s=='芝')&(dm>1400)&(dm<=2000)&(r!=1.0)
    elif name=='芝長': mask=(s=='芝')&(dm>2000)&(r!=1.0)
    seg=df[mask].copy(); seg['dist_m']=dm[seg.index]

    df_trn=seg[(seg['日付_num']>=130101)&(seg['日付_num']<220101)]
    df_val=seg[(seg['日付_num']>=220101)&(seg['日付_num']<=221231)]
    oos_2324=seg[(seg['日付_num']>=230101)&(seg['日付_num']<250101)]
    oos_2025=seg[(seg['日付_num']>=250101)&(seg['日付_num']<260101)]
    dfs=(df_trn,df_val,oos_2324,oos_2025)

    # 利用可能候補
    cands = [c for c in ALL_CANDS if c in seg.columns]
    seed_feats = [f for f in SEEDS[name] if f in seg.columns]

    best_feats = seed_feats[:]
    best_score = eval_feats(best_feats, dfs)
    print(f'SEED score: {best_score:.4f} ({len(best_feats)}特徴)')
    print(f'候補数: {len(cands)}  目標: {FAV[name]:.4f}')

    t0 = time.time()
    for i in range(N_ITER):
        # ランダムに変化を選ぶ: add / remove / swap
        op = rng.choice(['add','remove','swap','swap'])
        current = best_feats[:]
        not_in = [c for c in cands if c not in current]

        if op == 'add' and not_in and len(current) < MAX_FEATS:
            current.append(rng.choice(not_in))
        elif op == 'remove' and len(current) > MIN_FEATS:
            current.remove(rng.choice(current))
        elif op == 'swap' and not_in and len(current) > MIN_FEATS:
            current.remove(rng.choice(current))
            current.append(rng.choice([c for c in cands if c not in current]))
        else:
            continue  # 無効なオペレーションはスキップ

        score = eval_feats(current, dfs)
        if score > best_score:
            best_score = score
            best_feats = current[:]
            gap = FAV[name] - best_score
            mark = '★' if best_score >= FAV[name] else ' '
            print(f'{mark}[{i+1:4d}/{N_ITER}] op={op:6s} {best_score:.4f} gap={gap:+.4f} '
                  f'({len(best_feats)}f, {time.time()-t0:.0f}s)')
            sys.stdout.flush()
            if best_score >= FAV[name]:
                print(f'*** {name}: 1番人気超え! {best_score:.4f} > {FAV[name]:.4f} ***')
                break

    print(f'\n最終: {name} acc_2325={best_score:.4f}  目標まで{FAV[name]-best_score:+.4f} ({N_ITER}it, {time.time()-t0:.0f}s)')

    acc_pkl=os.path.join(BASE_DIR,'models','hitrate_model.pkl')
    existing_acc=0.0
    if os.path.exists(acc_pkl):
        pkg=pickle.load(open(acc_pkl,'rb'))
        if name in pkg: existing_acc=pkg[name].get('acc_2325',0.0)
    if best_score>existing_acc+0.0001:
        print(f'改善あり ({existing_acc:.4f}->{best_score:.4f}), 保存...')
        save_seg(name,best_feats,seg,VERSION_MAP[name])
    else:
        print(f'改善なし ({existing_acc:.4f}>={best_score:.4f}), スキップ')


if __name__=='__main__':
    main()
