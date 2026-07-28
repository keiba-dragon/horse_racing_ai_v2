# coding: utf-8
"""
後入れデータ（馬体重増減・騎手統計）パッチ前後の的中率・ROI比較
上書きなし・評価のみ

使い方:
  python src/eval_patched_vs_raw.py
"""
import sys, io, os, re, time, urllib.request
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pickle
import pandas as pd
import numpy as np

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, 'data', 'processed', 'all_venues_features.parquet')

CACHE_MAP = {
    '05/30': ('出馬表形式05月30日_api.cache.pkl', '20260530'),
    '05/31': ('出馬表形式05月31日_api.cache.pkl', '20260531'),
    '06/07': ('出馬表形式06月7日_api.cache.pkl',  '20260607'),
    '06/13': ('出馬表形式06月13日_api.cache.pkl', '20260613'),
    '06/14': ('出馬表形式06月14日_api.cache.pkl', '20260614'),
    '06/20': ('出馬表形式06月20日_api.cache.pkl', '20260620'),
    '06/21': ('出馬表形式06月21日_api.cache.pkl', '20260621'),
    '06/27': ('出馬表形式06月27日_api.cache.pkl', '20260627'),
    '06/28': ('出馬表形式06月28日_api.cache.pkl', '20260628'),
}

VENUE_LETTER_TO_CODE = {
    '東':'05','中':'06','中京':'07','名':'07','京':'08','阪':'09',
    '新':'04','福':'03','函':'02','札':'01','小':'10',
}

hitrate_model = pickle.load(open(os.path.join(BASE_DIR, 'models', 'hitrate_model.pkl'), 'rb'))
Z_MARU = 1.2; Z_SANKAKU = 1.5

# ── 騎手統計補完（make_newspaper.pyのpatch_jockey_statsを簡略化） ──
_MARK_RE = re.compile(r'[☆▲△▼○●◎◇◆★]')
_DOT_RE  = re.compile(r'[．.]')
def _norm(s): return _MARK_RE.sub('', str(s)).strip()
def _norm_jkn(s):
    import unicodedata
    return _DOT_RE.sub('', unicodedata.normalize('NFKC', _norm(s)))

JOCKEY_STAT_COLS = [
    '騎手コース_r100_勝率','騎手馬場_r100_勝率','騎手距離_r100_勝率',
    '騎手会場_r100_勝率','騎手_r200_勝率','騎手_r200_複勝率',
    '騎手コース_r100_複勝率','騎手馬場_r100_複勝率','騎手距離_r100_複勝率',
]

def patch_jockey(result_df, card_df):
    target = [c for c in JOCKEY_STAT_COLS if c in result_df.columns]
    if not target: return result_df
    horse_jkn = {}
    if card_df is not None and '馬名S' in card_df.columns and '騎手' in card_df.columns:
        for _, r in card_df.drop_duplicates('馬名S').iterrows():
            horse_jkn[str(r['馬名S'])] = _norm(r.get('騎手',''))
    result_df = result_df.copy()
    result_df['_jkn'] = result_df['馬名S'].map(horse_jkn).fillna('') if '馬名S' in result_df.columns else ''
    today_shorts = set(result_df['_jkn'].dropna()) - {'','nan'}
    if not today_shorts:
        result_df.drop(columns=['_jkn'], errors='ignore', inplace=True)
        return result_df
    try:
        base_cols = ['騎手','今回_コース種別','日付_num']
        avail = []
        for c in target:
            try: pd.read_parquet(DATA_FILE, columns=[c]); avail.append(c)
            except: pass
        pq = pd.read_parquet(DATA_FILE, columns=base_cols + avail)
    except Exception as e:
        print(f'  [WARN] parquet読込失敗: {e}')
        result_df.drop(columns=['_jkn'], errors='ignore', inplace=True)
        return result_df
    pq['_jkn_full'] = pq['騎手'].apply(_norm)
    all_names = pq['_jkn_full'].value_counts()
    short_to_full = {}
    for s in today_shorts:
        cands = {fn: cnt for fn, cnt in all_names.items()
                 if (fn.startswith(s) or s.startswith(fn[:2])) and len(fn) >= len(s)}
        if cands:
            short_to_full[s] = max(cands, key=cands.get)
    n_match = 0
    for s, fn in short_to_full.items():
        rows_df = result_df[result_df['_jkn'] == s]
        if rows_df.empty: continue
        jkn_rows = pq[pq['_jkn_full'] == fn].sort_values('日付_num', ascending=False)
        if jkn_rows.empty: continue
        n_match += 1
        for col in avail:
            mask = rows_df.index[result_df.loc[rows_df.index, col].isna()]
            if len(mask) == 0: continue
            src = jkn_rows[col].dropna()
            if src.empty: continue
            result_df.loc[mask, col] = src.iloc[0]
    result_df.drop(columns=['_jkn'], errors='ignore', inplace=True)
    return result_df


def fetch_weights_from_result(race_ids):
    """result.html から馬体重・増減を取得（レース確定後）"""
    row_pat   = re.compile(r'<tr[^>]*class="[^"]*HorseList[^"]*"[^>]*>(.*?)</tr>', re.DOTALL)
    uma_pat   = re.compile(r'class="Num Txt_C"[^>]*>(.*?)</td>', re.DOTALL)
    wt_pat    = re.compile(r'(\d{3})\(([+\-]?\d+)\)')
    weights   = {}
    for rid in race_ids:
        vc = rid[4:6]; rn = int(rid[10:12])
        try:
            req = urllib.request.Request(
                f'https://race.netkeiba.com/race/result.html?race_id={rid}',
                headers={'User-Agent':'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=10) as r:
                h = r.read().decode('euc-jp', errors='replace')
            if 'Result_Num' not in h: continue
            race_wt = {}
            for m in row_pat.finditer(h):
                row = m.group(1)
                um = uma_pat.search(row)
                wm = wt_pat.search(row)
                if um and wm:
                    us = re.sub(r'<[^>]+>','',um.group(1)).strip()
                    if us.isdigit():
                        race_wt[us.zfill(2)] = (int(wm.group(1)), int(wm.group(2)))
            if race_wt: weights[(vc,rn)] = race_wt
            time.sleep(0.08)
        except: pass
    return weights


def get_race_ids(full_date):
    req = urllib.request.Request(
        f'https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={full_date}',
        headers={'User-Agent':'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=12) as r:
        html = r.read().decode('euc-jp', errors='replace')
    return list(dict.fromkeys(re.findall(r'race_id=(\d{12})', html)))


def get_results(race_ids):
    row_pat   = re.compile(r'<tr[^>]*class="[^"]*HorseList[^"]*"[^>]*>(.*?)</tr>', re.DOTALL)
    jyuni_pat = re.compile(r'class="Result_Num"[^>]*>(.*?)</td>', re.DOTALL)
    uma_pat   = re.compile(r'class="Num Txt_C"[^>]*>(.*?)</td>', re.DOTALL)
    race_results = {}
    for rid in race_ids:
        vc = rid[4:6]; rn = int(rid[10:12])
        try:
            req = urllib.request.Request(
                f'https://race.netkeiba.com/race/result.html?race_id={rid}',
                headers={'User-Agent':'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=10) as r:
                h = r.read().decode('euc-jp', errors='replace')
            if 'Result_Num' not in h: continue
            order = {}
            for m in row_pat.finditer(h):
                row = m.group(1)
                jm = jyuni_pat.search(row); um = uma_pat.search(row)
                if um:
                    us = re.sub(r'<[^>]+>','',um.group(1)).strip()
                    if us.isdigit() and jm:
                        rs = re.sub(r'<[^>]+>','',jm.group(1)).strip()
                        if rs.isdigit(): order[us.zfill(2)] = int(rs)
            tansho = 0
            tm = re.search(r'<tr class="Tansho"[^>]*>(.*?)</tr>', h, re.DOTALL)
            if tm:
                pm = re.search(r'class="Payout"[^>]*>(.*?)</td>', tm.group(1), re.DOTALL)
                if pm:
                    pv = re.sub(r'[^\d]','',pm.group(1))
                    if pv: tansho = int(pv)
            if order: race_results[(vc,rn)] = {'order':order,'tansho':tansho}
            time.sleep(0.08)
        except: pass
    return race_results


def get_seg_key(surf, dist_m):
    if pd.isna(dist_m): return None
    surf = str(surf).strip()
    if surf == '芝':
        if dist_m <= 1400: return '芝短'
        elif dist_m <= 2000: return '芝中'
        else: return '芝長'
    elif surf == 'ダ':
        return 'ダ短' if dist_m <= 1400 else 'ダ長'
    return None


def score_result(result, patched=False):
    """result DataFrameをhitrate_modelでスコアリングして買い目リスト返却"""
    race_keys = [c for c in ['開催','Ｒ','レース名','距離','芝・ダ'] if c in result.columns]
    result_r = result.reset_index(drop=True)
    for k in race_keys: result_r[k] = result_r[k].astype(str)

    buys = []
    for gk, grp in result_r.groupby(race_keys, sort=False):
        grp = grp.copy()
        if isinstance(gk, tuple):
            kaikai = str(gk[0]); r_num = str(gk[1])
            kyori_raw = str(gk[3]) if len(gk)>3 else ''
            shiba_da  = str(gk[4]) if len(gk)>4 else ''
        else:
            kaikai = str(gk); r_num = kyori_raw = shiba_da = ''
        m2 = re.search(r'(\d+)', kyori_raw)
        dist_m = pd.to_numeric(m2.group() if m2 else '', errors='coerce')
        surf   = str(shiba_da).strip() if shiba_da and shiba_da!='nan' else str(kyori_raw)[:1]
        seg    = get_seg_key(surf, dist_m)
        if seg is None or seg not in hitrate_model: continue
        art = hitrate_model[seg]
        rows_arr = []
        for _, row in grp.iterrows():
            fv = []
            for f in art['feat_cols']:
                if f.endswith('_isnan'):
                    fv.append(1.0 if pd.isna(row.get(f[:-6])) else 0.0)
                else:
                    v = row.get(f, np.nan)
                    try: fv.append(float(v) if not pd.isna(v) else 0.0)
                    except: fv.append(0.0)
            rows_arr.append(fv)
        X = np.array(rows_arr, dtype=float)
        try:
            X_sc = art['scaler'].transform(X)
            scores = X_sc @ art['coef']
        except: continue
        _std = float(np.std(scores, ddof=0))
        z = (scores - scores.mean()) / _std if _std > 0 else np.zeros(len(scores))
        best_idx = int(np.argmax(scores))
        best_z   = z[best_idx]
        best_row = list(grp.iterrows())[best_idx][1]
        if best_z >= Z_SANKAKU: continue
        mark = '◎' if best_z < Z_MARU else '○'
        uma_ban = str(int(float(best_row.get('馬番',0)))).zfill(2) if '馬番' in grp.columns else '00'
        vm = re.search(r'[^\d]+', str(kaikai))
        vletter = vm.group().strip() if vm else ''
        vcode = VENUE_LETTER_TO_CODE.get(vletter,'')
        buys.append({
            'kaikai': kaikai, 'r_num': int(r_num), 'seg': seg,
            '馬名': best_row.get('馬名S','?'), '印': mark, 'z': round(best_z,2),
            'uma_ban': uma_ban, 'vcode': vcode, 'vletter': vletter,
        })
    return buys


# ── メイン ──
all_raw = []; all_patched = []
print(f"{'日付':>6}  {'RAW':^18}  {'PATCHED':^18}")
print(f"{'':>6}  {'買/的/ROI':^18}  {'買/的/ROI':^18}")
print('-' * 50)

for label, (cache_file, full_date) in CACHE_MAP.items():
    cache_path = os.path.join(BASE_DIR, 'data', 'raw', 'cache', cache_file)
    try:
        with open(cache_path,'rb') as f: cache = pickle.load(f)
    except: print(f'{label}: キャッシュなし'); continue

    result   = cache['result'].copy()
    card_df  = cache.get('card_df', pd.DataFrame())

    # レースID・結果取得
    race_ids     = get_race_ids(full_date)
    race_results = get_results(race_ids)
    wt_map       = fetch_weights_from_result(race_ids)

    def apply_patch(res):
        res = patch_jockey(res, card_df)
        # 馬体重・馬体重増減を result.html から補完
        bango_col = 'dc_馬番' if 'dc_馬番' in res.columns else ('馬番' if '馬番' in res.columns else None)
        if bango_col and '開催' in res.columns and 'Ｒ' in res.columns:
            for idx, row in res.iterrows():
                try:
                    uma_s = str(int(float(row.get(bango_col,0)))).zfill(2)
                    vm2 = re.search(r'[^\d]+', str(row.get('開催','')))
                    vl  = vm2.group().strip() if vm2 else ''
                    vc  = VENUE_LETTER_TO_CODE.get(vl,'')
                    rn  = int(row.get('Ｒ',0))
                    entry = wt_map.get((vc,rn),{}).get(uma_s)
                    if entry:
                        wt, wt_diff = entry
                        if pd.isna(row.get('馬体重')): res.at[idx,'馬体重'] = wt
                        if pd.isna(row.get('馬体重増減')): res.at[idx,'馬体重増減'] = wt_diff
                except: pass
        return res

    def eval_buys(buys, label_str):
        rows = []
        for b in buys:
            res = race_results.get((b['vcode'], b['r_num']))
            actual = res['order'].get(b['uma_ban']) if res else None
            tansho = res['tansho'] if res else 0
            hit = (actual==1) if actual is not None else None
            rows.append({**b, '着順':actual or '-', '単勝':tansho, '的中':'◯' if hit else ('✗' if hit is False else '-')})
        return rows

    # RAW スコア
    buys_raw = score_result(result, patched=False)
    rows_raw = eval_buys(buys_raw, label)

    # PATCHED スコア
    result_p = apply_patch(result.copy())
    buys_pat = score_result(result_p, patched=True)
    rows_pat = eval_buys(buys_pat, label)

    def summarize(rows):
        if not rows: return 0,0,0,float('nan')
        n = len(rows); nh = sum(1 for r in rows if r['的中']=='◯')
        paid = sum(r['単勝'] for r in rows if r['的中']=='◯' and isinstance(r['単勝'],int))
        roi = (paid - n*100)/(n*100)*100 if n else 0
        return n, nh, paid, roi

    nr,hr,_,rr = summarize(rows_raw)
    np_,hp,_,rp = summarize(rows_pat)
    print(f'{label}  {nr:2}R {hr:2}的中 {rr:+5.0f}%  {np_:2}R {hp:2}的中 {rp:+5.0f}%')
    all_raw.extend(rows_raw)
    all_patched.extend(rows_pat)

print('=' * 50)
def tot(rows):
    n=len(rows); nh=sum(1 for r in rows if r['的中']=='◯')
    paid=sum(r['単勝'] for r in rows if r['的中']=='◯' and isinstance(r['単勝'],int))
    roi=(paid-n*100)/(n*100)*100 if n else 0
    return n,nh,roi
nr,hr,rr=tot(all_raw); np_,hp,rp=tot(all_patched)
print(f'合計    {nr:2}R {hr:2}的中 {rr:+5.0f}%  {np_:2}R {hp:2}的中 {rp:+5.0f}%')

# セグメント別（パッチ済み）
print('\n--- PATCHED セグメント別 ---')
df = pd.DataFrame(all_patched)
if not df.empty:
    for seg, sg in df.groupby('seg'):
        sn=len(sg); sh=(sg['的中']=='◯').sum()
        sp=sg[sg['的中']=='◯']['単勝'].apply(lambda x: x if isinstance(x,int) else 0).sum()
        sr=(sp-sn*100)/(sn*100)*100
        print(f'  {seg}: {sn}R / {sh}的中 ({sh/sn*100:.0f}%) / ROI {sr:+.1f}%')
