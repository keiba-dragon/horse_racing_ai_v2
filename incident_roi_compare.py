"""05-30 修正前（git HEAD）vs 修正後 ROI 比較"""
import pickle, pandas as pd, sys, ssl, gzip, re, time
import urllib.request
from bs4 import BeautifulSoup

sys.stdout.reconfigure(encoding='utf-8')

with open('data/raw/cache/20260530_old.cache.pkl', 'rb') as f:
    old_cache = pickle.load(f)
with open('data/raw/cache/20260530_new.cache.pkl', 'rb') as f:
    new_cache = pickle.load(f)

old_df = old_cache['result'].copy()
new_df = new_cache['result'].copy()

old_df['_rank'] = old_df.groupby('場 R')['clogit_calib'].rank(method='first', ascending=False)
new_df['_rank'] = new_df.groupby('場 R')['clogit_calib'].rank(method='first', ascending=False)

top1_old = old_df[old_df['_rank'] == 1][['場 R','馬名S','clogit_calib']].rename(columns={'馬名S':'horse_old','clogit_calib':'calib_old'})
top1_new = new_df[new_df['_rank'] == 1][['場 R','馬名S','clogit_calib']].rename(columns={'馬名S':'horse_new','clogit_calib':'calib_new'})

top1 = top1_old.merge(top1_new, on='場 R')
changed = top1[top1['horse_old'] != top1['horse_new']]
print(f"予測1位が変わったレース: {len(changed)}/{len(top1)}R")
print(changed[['場 R','horse_old','calib_old','horse_new','calib_new']].to_string(index=False))
print()

# Yahoo結果取得
venue_keys = {'08': [3, 11], '05': [2, 11]}
venue_map = {'08': '京', '05': '東'}
HEADERS = {'User-Agent': 'Mozilla/5.0','Referer': 'https://sports.yahoo.co.jp/keiba/'}
SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE

def _get(url):
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=10, context=SSL_CTX) as res:
            raw = res.read()
        try: return gzip.decompress(raw).decode('utf-8', errors='replace')
        except: return raw.decode('utf-8', errors='replace')
    except: return None

def parse_result(html):
    soup = BeautifulSoup(html, 'html.parser')
    horses, tansho = {}, None
    for t in soup.find_all('table'):
        for row in t.find_all('tr'):
            cells = row.find_all(['td','th'])
            if len(cells) >= 9:
                try:
                    rank = int(cells[0].get_text(strip=True))
                    uma = re.split(r'[牡牝セ]\d|せん\d', cells[3].get_text(strip=True))[0].strip()
                    if uma: horses[uma] = rank
                except: pass
            elif len(cells) == 3 and tansho is None:
                txt = cells[1].get_text(strip=True)
                if '円' in txt:
                    try: tansho = int(re.sub(r'[^\d]','',txt)) / 100.0
                    except: pass
    return horses, tansho

results = []
for vc, (kai, nichi) in venue_keys.items():
    vs = venue_map[vc]
    for rn in range(1, 13):
        key = f'26{vc}{kai:02d}{nichi:02d}{rn:02d}'
        html = _get(f'https://sports.yahoo.co.jp/keiba/race/result/{key}')
        if not html or len(html) < 2000: continue
        horses, tansho = parse_result(html)
        if not horses: continue
        race_pat = f'{vs}{rn}$'
        ro = top1_old[top1_old['場 R'].str.contains(race_pat)]
        rn2 = top1_new[top1_new['場 R'].str.contains(race_pat)]
        if ro.empty or rn2.empty: continue
        ho, hn = ro.iloc[0]['horse_old'], rn2.iloc[0]['horse_new']
        results.append({
            'race': f'{vs}{rn}',
            'horse_old': ho, 'hit_old': horses.get(ho)==1,
            'horse_new': hn, 'hit_new': horses.get(hn)==1,
            'tansho': tansho, 'changed': ho != hn,
        })
        time.sleep(0.15)

df = pd.DataFrame(results)
n = len(df)

def roi(df, hit_col):
    hits = df[hit_col].sum()
    ret = df[df[hit_col]]['tansho'].fillna(0).sum()
    return hits, (ret - n) / n * 100, ret

h_old, roi_old, ret_old = roi(df, 'hit_old')
h_new, roi_new, ret_new = roi(df, 'hit_new')

print(f"{'':10} {'修正前':>8} {'修正後':>8}")
print(f"{'対象R':10} {n:>8} {n:>8}")
print(f"{'的中':10} {h_old:>8} {h_new:>8}")
print(f"{'的中率':10} {h_old/n*100:>7.1f}% {h_new/n*100:>7.1f}%")
print(f"{'総回収':10} {ret_old:>8.1f} {ret_new:>8.1f}")
print(f"{'ROI':10} {roi_old:>+7.1f}% {roi_new:>+7.1f}%")
print()

print("=== 詳細（変わったレース） ===")
ch = df[df['changed']]
print(ch[['race','horse_old','hit_old','horse_new','hit_new','tansho']].to_string(index=False))
