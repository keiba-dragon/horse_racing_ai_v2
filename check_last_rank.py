# coding: utf-8
"""
check_last_rank.py
  単勝モデル vs 複勝モデル の 3着以内率 をモデルランク4～18で比較
"""
import sys, os, pickle, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from save_v3 import add_computed_features
from save_conditional_logit import segment_softmax, neg_log_lik_fukusho_and_grad

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, 'data', 'processed', 'all_venues_features.parquet')
MODEL     = pickle.load(open(os.path.join(BASE_DIR, 'models', 'accuracy_model.pkl'), 'rb'))
ORDER     = ['芝長', '芝中', '芝短', 'ダ長', 'ダ短']

L2 = 0.006; LR = 0.001; N_EPOCHS = 600; PATIENCE = 80

def adam_train(X_tr, y_tr, gs_tr, X_va, y_va, gs_va, nr_tr, nr_va, loss_fn):
    beta = np.zeros(X_tr.shape[1])
    m = v = np.zeros_like(beta)
    b1, b2, eps = 0.9, 0.999, 1e-8
    best_loss, best_beta, wait = np.inf, beta.copy(), 0
    for ep in range(1, N_EPOCHS + 1):
        loss, grad = loss_fn(beta, X_tr, y_tr, gs_tr, len(X_tr), nr_tr)
        m = b1*m+(1-b1)*grad; v = b2*v+(1-b2)*grad**2
        mh = m/(1-b1**ep); vh = v/(1-b2**ep)
        beta -= LR * mh / (np.sqrt(vh) + eps)
        pv = segment_softmax(X_va @ beta, gs_va, len(X_va))
        vl = -np.sum(y_va * np.log(np.clip(pv,1e-15,1))) / nr_va
        if vl < best_loss - 1e-7: best_loss, best_beta, wait = vl, beta.copy(), 0
        else:
            wait += 1
            if wait >= PATIENCE: break
    return best_beta

def get_group_starts(race_ids):
    _, idx = np.unique(race_ids, return_index=True)
    return np.sort(idx)

def seg_key(surf, dist_m):
    if pd.isna(dist_m): return None
    s = str(surf).strip()
    if s == '芝': return '芝短' if dist_m<=1400 else ('芝中' if dist_m<=2000 else '芝長')
    elif s == 'ダ': return 'ダ短' if dist_m<=1400 else 'ダ長'
    return None

print('データ読み込み中...')
df = pd.read_parquet(DATA_FILE)
df['日付_num'] = pd.to_numeric(df['日付'], errors='coerce')
df['着順_num'] = pd.to_numeric(df['着順_num'], errors='coerce')
df = df.dropna(subset=['日付_num','着順_num'])
df = df[df['着順_num'] < 99]
df['race_id'] = (df['日付_num'].astype(int).astype(str)+'_'+
                 df['開催'].astype(str).str.strip()+'_'+df['Ｒ'].astype(str).str.strip())
df = df[df['開催'].notna()].copy()
df['_surf']   = df['距離'].astype(str).str.strip().str.extract(r'^([芝ダ])')[0].fillna('不明')
df['_dist_m'] = pd.to_numeric(df['距離'].astype(str).str.extract(r'(\d+)')[0], errors='coerce')
df['クラス_rank'] = pd.to_numeric(df['クラス_rank'], errors='coerce')
df = df[df['クラス_rank'] != 1.0].copy()
df['seg_key'] = [seg_key(s,d) for s,d in zip(df['_surf'],df['_dist_m'])]
df = df[df['seg_key'].notna()].copy()
df['dist_m'] = df['_dist_m']
df = add_computed_features(df)
bm = {'良':0,'稍重':1,'重':2,'不良':3}
for c in df.columns:
    if '馬場状態' in c and c!='馬場状態': df[c]=df[c].map(bm)
print('完了\n')

PERIODS = {'train':(130101,220101), 'val':(220101,230101), 'oos':(230101,990101)}

def build_X(grp, feat_cols, scaler):
    X = np.zeros((len(grp), len(feat_cols)), dtype=float)
    for j,f in enumerate(feat_cols):
        if f.endswith('_isnan'):
            base=f[:-6]; X[:,j]=(~grp[base].notna()).astype(float).values if base in grp.columns else 1.0
        else:
            col = grp[f] if f in grp.columns else pd.Series(np.nan,index=grp.index)
            X[:,j] = pd.to_numeric(col,errors='coerce').fillna(0).values
    return scaler.transform(X)

def win_loss(beta,X,y,gs,n,nr):
    p=segment_softmax(X@beta,gs,n)
    return -np.sum(y*np.log(np.clip(p,1e-15,1)))/nr+L2*np.dot(beta,beta), \
           -(X.T@(y-p))/nr+2*L2*beta

def compute_p3_by_rank(beta, g, X):
    g = g.copy()
    g['_score'] = X @ beta
    g['_rank']  = g.groupby('race_id')['_score'].rank(ascending=False, method='first').astype(int)
    result = {}
    for r in range(1, 19):
        sub = g[g['_rank'] == r]
        n   = len(sub)
        if n < 10:
            result[r] = (n, float('nan'))
        else:
            result[r] = (n, float((sub['着順_num'] <= 3).mean()))
    return result

for seg in ORDER:
    art = MODEL.get(seg)
    if art is None: continue
    feat_cols, scaler = art['feat_cols'], art['scaler']
    seg_df = df[df['seg_key']==seg].sort_values('race_id').reset_index(drop=True)

    def split(p):
        d0,d1=PERIODS[p]
        g=seg_df[(seg_df['日付_num']>=d0)&(seg_df['日付_num']<d1)].copy().sort_values('race_id').reset_index(drop=True)
        X=build_X(g,feat_cols,scaler); gs=get_group_starts(g['race_id'].values); nr=g['race_id'].nunique()
        return X,gs,nr,(g['着順_num']==1).astype(float).values,(g['着順_num']<=3).astype(float).values,g

    X_tr,gs_tr,nr_tr,yw_tr,yp_tr,_    = split('train')
    X_va,gs_va,nr_va,yw_va,yp_va,_    = split('val')
    X_oo,gs_oo,nr_oo,yw_oo,yp_oo,g_oo = split('oos')

    r_win = compute_p3_by_rank(art['coef'], g_oo, X_oo)
    bf    = adam_train(X_tr,yp_tr,gs_tr,X_va,yw_va,gs_va,nr_tr,nr_va,neg_log_lik_fukusho_and_grad)
    r_fuk = compute_p3_by_rank(bf, g_oo, X_oo)

    print(f'{"─"*56}')
    print(f'【{seg}】  3着以内率 (OOS 2023–)  rank4～18')
    print(f'{"rank":<6}  {"頭数":>5}  {"単勝":>7}  {"複勝ロス":>8}  {"差":>6}')
    print(f'{"─"*56}')
    for r in range(4, 19):
        n_w, p_w = r_win.get(r, (0, float('nan')))
        n_f, p_f = r_fuk.get(r, (0, float('nan')))
        if n_w < 10: continue
        if p_w != p_w or p_f != p_f:
            print(f'rank{r:<3}  {n_w:>5}  {"−":>7}  {"−":>8}  {"−":>6}')
            continue
        diff = p_f - p_w
        sign = '+' if diff >= 0 else ''
        marker = ' 複勝↑' if diff > 0.005 else (' 単勝↑' if diff < -0.005 else '')
        print(f'rank{r:<3}  {n_w:>5}  {p_w*100:6.1f}%  {p_f*100:7.1f}%  {sign}{diff*100:.1f}%{marker}')
    print()

print('完了')
