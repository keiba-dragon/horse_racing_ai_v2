"""
diff（偏差値の差）の有効性検証
・diffなし vs diffあり でROIがどう変わるか
・閾値を0〜20で変えたときのROI推移
・cur_diff の追加効果
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd, numpy as np, os, pickle, json, re, time

base_dir  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_dir = os.path.join(base_dir, 'models_2025')

def get_distance_band(dist):
    m = re.search(r'\d+', str(dist))
    if not m: return None
    d = int(m.group())
    if d <= 1400: return '短距離'
    elif d <= 1800: return 'マイル'
    elif d <= 2200: return '中距離'
    return '中長距離'

def get_class_group(r):
    try: r = int(float(r))
    except: return '3勝以上'
    if r == 1: return '新馬'
    elif r == 2: return '未勝利'
    elif r == 3: return '1勝'
    elif r == 4: return '2勝'
    return '3勝以上'

def extract_venue(kaikai):
    m = re.search(r'\d+([^\d]+)', str(kaikai))
    return m.group(1) if m else str(kaikai)

print("モデル読み込み中...")
with open(os.path.join(model_dir, 'model_info.json'), encoding='utf-8') as f:
    cur_info = json.load(f)
cur_features = cur_info['features']; cur_models = cur_info['models']
with open(os.path.join(model_dir, 'ranker', 'ranker_info.json'), encoding='utf-8') as f:
    cur_rankers = json.load(f).get('rankers', {})
with open(os.path.join(model_dir, 'submodel', 'submodel_info.json'), encoding='utf-8') as f:
    sub_info = json.load(f)
sub_features = sub_info['features']; sub_models = sub_info['models']
with open(os.path.join(model_dir, 'submodel_ranker', 'class_ranker_info.json'), encoding='utf-8') as f:
    sub_rankers = json.load(f).get('rankers', {})
all_feats = list(set(cur_features + sub_features))

test_path = os.path.join(base_dir, 'data', 'processed', 'all_venues_features_2026test.csv')
df = pd.read_csv(test_path, low_memory=False)
df['着順_num']   = pd.to_numeric(df['着順_num'],   errors='coerce')
df['単勝配当']   = pd.to_numeric(df['単勝配当'],   errors='coerce')
df['単勝オッズ'] = pd.to_numeric(df['単勝オッズ'], errors='coerce')
df = df.dropna(subset=['着順_num', '単勝配当'])
df['target_win'] = (df['着順_num'] == 1).astype(int)
df['会場']       = df['開催'].apply(extract_venue)
df['_surface']   = df['芝・ダ'].astype(str).str.strip()
df['_dist_band'] = df['距離'].apply(get_distance_band)
mask_da = (df['_surface'] == 'ダ') & (df['_dist_band'].isin(['中距離', '長距離']))
df.loc[mask_da, '_dist_band'] = '中長距離'
df['_cls_group'] = df['クラス_rank'].apply(get_class_group) if 'クラス_rank' in df.columns else '3勝以上'
df['cur_key'] = df['会場'] + '_' + df['距離'].astype(str)
df['sub_key'] = df['_surface'] + '_' + df['_dist_band'] + '_' + df['_cls_group']
for col in all_feats:
    if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
print(f"データ: {len(df):,}頭  期間: {int(df['日付'].min())}〜{int(df['日付'].max())}")

# モデルキャッシュ
cur_mc={}; cur_rc={}; sub_mc={}; sub_rc={}
for ck in df['cur_key'].dropna().unique():
    if ck in cur_models:
        p = os.path.join(model_dir, cur_models[ck]['win'])
        if os.path.exists(p):
            with open(p,'rb') as f2: m=pickle.load(f2)
            cur_mc[ck]=(m, m.booster_.feature_name())
    if ck in cur_rankers:
        p = os.path.join(model_dir, 'ranker', cur_rankers[ck])
        if os.path.exists(p):
            with open(p,'rb') as f2: cur_rc[ck]=pickle.load(f2)
for sk in df['sub_key'].dropna().unique():
    if sk in sub_models:
        p = os.path.join(model_dir, 'submodel', sub_models[sk]['win'])
        if os.path.exists(p):
            with open(p,'rb') as f2: m=pickle.load(f2)
            sub_mc[sk]=(m, m.booster_.feature_name())
    if sk in sub_rankers:
        p = os.path.join(model_dir, 'submodel_ranker', sub_rankers[sk])
        if os.path.exists(p):
            with open(p,'rb') as f2: sub_rc[sk]=pickle.load(f2)
print(f"モデルキャッシュ: cur={len(cur_mc)} sub={len(sub_mc)}")

# 予測
t0=time.time(); all_rows=[]
race_keys = [c for c in ['開催','Ｒ'] if c in df.columns]
for gk, idx in df.groupby(race_keys, sort=False).groups.items():
    sub = df.loc[idx].copy()
    ck = sub['cur_key'].iloc[0]
    sk = sub['sub_key'].iloc[0]
    sub['cur_rank']=np.nan; sub['cur_diff']=np.nan; sub['cur_score']=np.nan
    sub['sub_rank']=np.nan; sub['sub_diff']=np.nan; sub['sub_score']=np.nan

    if ck in cur_mc:
        m, wf = cur_mc[ck]
        for c in wf:
            if c not in sub.columns: sub[c]=np.nan
        prob = m.predict_proba(sub[wf])[:,1]
        st=cur_models[ck].get('stats',{}); wm=st.get('win_mean',prob.mean()); ws=st.get('win_std',prob.std())
        cs = 50+10*(prob-wm)/(ws if ws>0 else 1)
        sub['cur_score']=cs
        rm=prob.mean(); rs=prob.std()
        sub['cur_diff']=50+10*(prob-rm)/(rs if rs>0 else 1)-cs
        if ck in cur_rc:
            sc=cur_rc[ck].predict(sub[cur_features])
            sub['cur_rank']=pd.Series(sc,index=sub.index).rank(ascending=False,method='min').astype(int)

    if sk in sub_mc:
        m, wf = sub_mc[sk]
        for c in wf:
            if c not in sub.columns: sub[c]=np.nan
        prob = m.predict_proba(sub[wf])[:,1]
        st=sub_models[sk].get('stats',{}); wm=st.get('win_mean',prob.mean()); ws=st.get('win_std',prob.std())
        cs = 50+10*(prob-wm)/(ws if ws>0 else 1)
        sub['sub_score']=cs
        rm=prob.mean(); rs=prob.std()
        sub['sub_diff']=50+10*(prob-rm)/(rs if rs>0 else 1)-cs
        if sk in sub_rc:
            sc=sub_rc[sk].predict(sub[wf])
            sub['sub_rank']=pd.Series(sc,index=sub.index).rank(ascending=False,method='min').astype(int)

    all_rows.append(sub)

result = pd.concat(all_rows, ignore_index=True)
print(f"予測完了: {len(result):,}頭 ({time.time()-t0:.0f}秒)\n")

# ── 分析 ──
cr  = pd.to_numeric(result['cur_rank'],   errors='coerce')
sr  = pd.to_numeric(result['sub_rank'],   errors='coerce')
cd  = pd.to_numeric(result['cur_diff'],   errors='coerce')
sd  = pd.to_numeric(result['sub_diff'],   errors='coerce')
od  = pd.to_numeric(result['単勝オッズ'], errors='coerce')
win = pd.to_numeric(result['target_win'], errors='coerce')
pay = pd.to_numeric(result['単勝配当'],   errors='coerce')

both_r1  = (cr == 1) & (sr == 1)
odds_ok5 = od.isna() | (od >= 5)

def roi_stat(mask):
    m = mask & win.notna()
    n = int(m.sum())
    if n == 0: return float('nan'), 0, 0
    wins = int((m & (win==1)).sum())
    ret  = pay[m & (win==1)].sum()
    roi  = (ret - n*100) / (n*100) * 100
    return roi, n, wins

print(f"{'='*65}")
print("  ① diffフィルターの有無（両Rnk=1 & odds≥5）")
print(f"{'条件':<32} {'頭数':>5} {'的中':>5} {'的中率':>7} {'ROI':>8}")
print('-'*65)
for label, mask in [
    ('diffなし',                    both_r1 & odds_ok5),
    ('sub_diff≥5',                  both_r1 & odds_ok5 & (sd>=5)),
    ('sub_diff≥10',                 both_r1 & odds_ok5 & (sd>=10)),
    ('sub_diff≥15',                 both_r1 & odds_ok5 & (sd>=15)),
    ('sub_diff≥20',                 both_r1 & odds_ok5 & (sd>=20)),
]:
    r, n, w = roi_stat(mask)
    wr = w/n if n else 0
    print(f"{label:<32} {n:>5} {w:>5} {wr:>7.1%} {r:>+7.1f}%")

print(f"\n{'='*65}")
print("  ② cur_diffの追加効果（both_r1 & odds≥5 & sub_diff≥10）")
print(f"{'条件':<32} {'頭数':>5} {'的中':>5} {'的中率':>7} {'ROI':>8}")
print('-'*65)
base = both_r1 & odds_ok5 & (sd>=10)
for label, mask in [
    ('cur_diffなし（base）',          base),
    ('+ cur_diff≥0',                  base & (cd>=0)),
    ('+ cur_diff≥5',                  base & (cd>=5)),
    ('+ cur_diff≥10',                 base & (cd>=10)),
    ('+ cur_diff≥15',                 base & (cd>=15)),
]:
    r, n, w = roi_stat(mask)
    wr = w/n if n else 0
    print(f"{label:<32} {n:>5} {w:>5} {wr:>7.1%} {r:>+7.1f}%")

print(f"\n{'='*65}")
print("  ③ sub_diff閾値スキャン（both_r1 & odds≥5）")
print(f"{'閾値':<10} {'頭数':>6} {'的中率':>7} {'ROI':>8}  {'傾向'}")
print('-'*65)
for th in range(0, 26, 2):
    r, n, w = roi_stat(both_r1 & odds_ok5 & (sd>=th))
    if n == 0: continue
    wr = w/n if n else 0
    bar = '▮' * max(0, int((r+50)/10))
    print(f"≥{th:<8} {n:>6} {wr:>7.1%} {r:>+7.1f}%  {bar}")

print(f"\n{'='*65}")
print("  ④ ランク条件なし vs ランク条件あり（sub_diff≥10 & odds≥5）")
print(f"{'条件':<32} {'頭数':>5} {'的中':>5} {'的中率':>7} {'ROI':>8}")
print('-'*65)
for label, mask in [
    ('sub_rank=1のみ',                (sr==1) & odds_ok5 & (sd>=10)),
    ('both_rank=1',                   both_r1 & odds_ok5 & (sd>=10)),
    ('sub_rank≤2',                    (sr<=2) & odds_ok5 & (sd>=10)),
    ('cur_rank≤2 & sub_rank≤2',       (cr<=2)&(sr<=2) & odds_ok5 & (sd>=10)),
]:
    r, n, w = roi_stat(mask)
    wr = w/n if n else 0
    print(f"{label:<32} {n:>5} {w:>5} {wr:>7.1%} {r:>+7.1f}%")
