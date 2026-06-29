# coding: utf-8
"""
芝中・芝長の accuracy_model を元の特徴量セットで再訓練して復元する
"""
import sys, os, pickle
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from save_conditional_logit import prepare, segment_softmax, BASE_DIR, DATA_FILE, LR, N_EPOCHS, PATIENCE
from save_v3 import add_computed_features

L2 = 0.003
NAN_IND_THRESHOLD = 0.05

# 元の特徴量（体重除外前の accuracy_model feat_cols）
ORIG_FEATS = {
    '芝中': [
        '馬番', '斤量', '1走前_単勝オッズ', '種牡馬_勝率', '馬体重増減',
        '近5走_クラス調整_平均着順', '1走前_クラス差', '3走前_クラス差', '1走前_上り3F',
        '騎手馬場_r100_勝率', '4角位置_近3走_slope', '2走前_クラス差', '1走前_馬場状態',
        '騎手会場_r100_勝率', '2走前_タイム指数', '同会場_平均着順_近5走', '近10走_複勝率',
        '着順_近3走_slope', '前走_人気着順差', '1走前_PCI', '近3走_複勝率',
        'コース枠_r200_複勝率', '近10走_勝率', '馬体重', '良馬場_平均着順_近5走',
        '1走前_RPCI', '1走前_頭数', '同馬場_平均着順_近5走', '種牡馬_ダ_勝率',
        '前走着差タイム', '2走前_上り3F', '性別_num', '馬コース_r20_勝率',
        '馬距離_勝率', 'タイム指数_近5走_slope', '母父馬_勝率', '芝ダ転向',
    ],
    '芝長': [
        '近3走_複勝率', '騎手距離_r100_勝率', '近5走_タイム指数平均', '馬コース_r20_勝率',
        '調教師コース_r100_勝率', '同会場_複勝率_近5走', '近5走_上り3F_std',
        'コース枠_r200_複勝率', '相手レベル_平均着順', 'タイム指数_加速度', '近10走_勝率',
        'コース馬場_r200_勝率', '種牡馬_ダ_勝率', 'タイム指数_近5走_slope',
        '道悪_平均着順_近5走', '1走前_タイム指数', '距離変化_前走', '性別_num',
        '馬体重', '馬体重増減', '調教師_r200_勝率', '輸送有無',
        'コース枠_r200_勝率', '馬番', '斤量', '近5走_タイム指数_min',
    ],
}

FAV_MAP = {'芝中': 0.3321, '芝長': 0.3605}


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


def adam_fit(X_tr, y_tr, gs_tr, n_tr, nr_tr, X_va, y_va, gs_va, n_va, nr_va):
    d = X_tr.shape[1]; beta, m, v = np.zeros(d), np.zeros(d), np.zeros(d)
    b1, b2, eps = 0.9, 0.999, 1e-8; t, best_val, best_beta, no_imp = 0, np.inf, np.zeros(d), 0
    for epoch in range(1, N_EPOCHS + 1):
        probs = segment_softmax(X_tr @ beta, gs_tr, n_tr)
        loss = -np.sum(y_tr * np.log(np.clip(probs, 1e-15, 1.0))) / nr_tr + L2 * np.dot(beta, beta)
        grad = -(X_tr.T @ (y_tr - probs)) / nr_tr + 2 * L2 * beta
        t += 1; m = b1*m+(1-b1)*grad; v = b2*v+(1-b2)*grad**2
        beta -= LR*(m/(1-b1**t))/(np.sqrt(v/(1-b2**t))+eps)
        if epoch % 10 == 0:
            p2 = segment_softmax(X_va @ beta, gs_va, n_va)
            vl = -np.sum(y_va*np.log(np.clip(p2,1e-15,1.0)))/nr_va + L2*np.dot(beta,beta)
            if vl < best_val: best_val, best_beta, no_imp = vl, beta.copy(), 0
            else: no_imp += 1
            if no_imp >= PATIENCE // 10: break
    return best_beta


def main():
    df = pd.read_parquet(DATA_FILE)
    df['日付_num'] = pd.to_numeric(df['日付'], errors='coerce')
    df['着順_num'] = pd.to_numeric(df['着順_num'], errors='coerce')
    df = df.dropna(subset=['日付_num','着順_num'])
    df = df[df['着順_num'] < 99]
    df['race_id'] = (df['日付_num'].astype(int).astype(str)+'_'+
                     df['開催'].astype(str).str.strip()+'_'+df['Ｒ'].astype(str).str.strip())
    df = df[df['開催'].notna()].copy()
    df['surface'] = df['距離'].astype(str).str.strip().str.extract(r'^([芝ダ])')[0].fillna('不明')
    dm = pd.to_numeric(df['距離'].astype(str).str.extract(r'(\d+)')[0], errors='coerce')
    df['クラス_rank'] = pd.to_numeric(df['クラス_rank'], errors='coerce')
    df = add_computed_features(df)
    if '今回_会場' in df.columns and '1走前_開催' in df.columns:
        df['輸送有無'] = (df['今回_会場'].astype(str) != df['1走前_開催'].astype(str).str[1]).astype(float)
        df.loc[df['1走前_開催'].isna(),'輸送有無'] = float('nan')
    baba_map = {'良':0,'稍重':1,'重':2,'不良':3}
    for col in df.columns:
        if '馬場状態' in col and col != '馬場状態': df[col] = df[col].map(baba_map)

    acc_pkl = os.path.join(BASE_DIR, 'models', 'hitrate_model.pkl')
    existing = pickle.load(open(acc_pkl, 'rb'))

    SEG_MASKS = {
        '芝中': (df['surface']=='芝') & (dm>1400) & (dm<=2000) & (df['クラス_rank']!=1.0),
        '芝長': (df['surface']=='芝') & (dm>2000) & (df['クラス_rank']!=1.0),
    }

    for seg, mask in SEG_MASKS.items():
        print(f'\n=== {seg} 復元 ===')
        seg_df = df[mask].copy()
        seg_df['dist_m'] = dm[seg_df.index]
        tr   = seg_df[(seg_df['日付_num']>=130101)&(seg_df['日付_num']<220101)]
        va   = seg_df[(seg_df['日付_num']>=220101)&(seg_df['日付_num']<=221231)]
        o2324 = seg_df[(seg_df['日付_num']>=230101)&(seg_df['日付_num']<250101)]
        o2025 = seg_df[(seg_df['日付_num']>=250101)&(seg_df['日付_num']<260101)]
        o2026 = seg_df[seg_df['日付_num']>=260101]

        feats = ORIG_FEATS[seg]
        all_dfs = [tr, va, o2324, o2025, o2026]
        expanded = expand_nan_ind(all_dfs, feats)
        valid = [c for c in expanded if c in tr.columns
                 and tr[c].isna().mean() < 1.0 and tr[c].std(ddof=0) > 0]
        print(f'有効特徴数: {len(valid)}')

        X_tr,y_tr,gs_tr,n_tr,nr_tr,scaler,*_ = prepare(tr, valid, top_idx=None, top_idx3=None, fit=True)
        X_va,y_va,gs_va,n_va,nr_va,*_ = prepare(va, valid, scaler=scaler, top_idx=None, top_idx3=None)
        beta = adam_fit(X_tr,y_tr,gs_tr,n_tr,nr_tr,X_va,y_va,gs_va,n_va,nr_va)

        raw_val = segment_softmax(X_va @ beta, gs_va, n_va)
        val_s = va.sort_values('race_id').reset_index(drop=True)
        y_val = (val_s['着順_num']==1).astype(float).values
        iso = IsotonicRegression(out_of_bounds='clip'); iso.fit(raw_val, y_val)

        results = {}
        for label, oos in [('2324',o2324),('2025',o2025),('2026',o2026)]:
            if len(oos)==0: continue
            vp = [c for c in valid if c in oos.columns]
            X_p,_,gs_p,n_p,*_ = prepare(oos, vp, scaler=scaler, top_idx=None, top_idx3=None)
            s2 = oos.sort_values('race_id').reset_index(drop=True)
            s2['prob'] = segment_softmax(X_p @ beta, gs_p, n_p)
            s2['rank'] = s2.groupby('race_id')['prob'].rank(ascending=False, method='first')
            t2 = s2[s2['rank']==1]; nr = s2['race_id'].nunique()
            acc = (t2['着順_num']==1).mean()
            odds = pd.to_numeric(t2['単勝オッズ'], errors='coerce')
            roi = (odds[t2['着順_num']==1]*100).sum()/(len(t2)*100)-1
            results[label] = (acc, roi, nr)
            print(f'  {label}: acc={acc:.2%} ROI={roi:+.2%} ({nr}R)')

        n2324=results.get('2324',(0,0,0))[2]; n25=results.get('2025',(0,0,0))[2]; n26=results.get('2026',(0,0,0))[2]
        a2324=results.get('2324',(0,0,0))[0]; a25=results.get('2025',(0,0,0))[0]; a26=results.get('2026',(0,0,0))[0]
        r25=results.get('2025',(0,0,0))[1]; r26=results.get('2026',(0,0,0))[1]
        acc_2325 = (a2324*n2324+a25*n25)/(n2324+n25) if (n2324+n25)>0 else 0.0
        acc_2526 = (a25*n25+a26*n26)/(n25+n26) if (n25+n26)>0 else 0.0
        roi_2526 = (r25*n25+r26*n26)/(n25+n26) if (n25+n26)>0 else 0.0
        print(f'  acc_2325={acc_2325:.4f}  25+26_acc={acc_2526:.4f}  ROI={roi_2526:+.2%}')

        acc_pkg = {
            'segment': seg, 'scaler': scaler, 'coef': beta, 'feat_cols': valid,
            'isotonic': iso, 'acc_2325': acc_2325, 'acc_2526': acc_2526,
            'oos_roi_2526': roi_2526,
            'version': f'{seg}_acc_restored',
            'note': f'復元: 元特徴量(体重込み) {len(feats)}特徴 acc_2325={acc_2325:.4f}',
        }
        existing2 = pickle.load(open(acc_pkl,'rb'))
        existing2[seg] = acc_pkg
        with open(acc_pkl,'wb') as f: pickle.dump(existing2, f)
        print(f'  保存完了: {seg}')


if __name__ == '__main__':
    main()
