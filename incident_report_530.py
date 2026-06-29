"""
インシデントレポート: 05-30 修正前後の予測ズレ分析
- old = 20260530_new.cache.pkl の clogit_calib_old（近N走修正済み、クラス調整未修正）
  ※ 完全修正前の最後のキャッシュ
- new = 20260530_new.cache.pkl の clogit_calib_new（完全修正）
  ※ check_diff.py で取得済みの両値を再ロード
"""
import pickle, pandas as pd, sys, json, ssl, gzip, re, time
import urllib.request
from bs4 import BeautifulSoup

sys.stdout.reconfigure(encoding='utf-8')

# 旧 = 近N走修正済み＋クラス調整未修正（before this session）
with open('data/raw/cache/20260530.cache.pkl', 'rb') as f:
    new_cache = pickle.load(f)
# new_cache は現在 fully-fixed 版
# old_partial は取得不可（上書き済み）→ gitから再実行した予測を使う

# 現在持っているのは完全修正版のみなので、
# Yahoo結果から修正後ROIを計算して、修正前との差分はdiff出力から推定
new_df = new_cache['result'].copy()
new_df['_pred_rank'] = new_df.groupby('場 R')['clogit_calib'].rank(method='first', ascending=False)
top1_new = new_df[new_df['_pred_rank'] == 1][['場 R','馬名S','clogit_calib']].copy()

# Yahoo結果取得
venue_keys = {'08': [3, 11], '05': [2, 11]}
year2 = '26'
venue_map = {'08': '京', '05': '東'}

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'text/html,application/xhtml+xml',
    'Accept-Language': 'ja,en-US;q=0.7',
    'Referer': 'https://sports.yahoo.co.jp/keiba/',
}
SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE

def _get(url):
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=10, context=SSL_CTX) as res:
            raw = res.read()
        try:
            return gzip.decompress(raw).decode('utf-8', errors='replace')
        except Exception:
            return raw.decode('utf-8', errors='replace')
    except:
        return None

def parse_result(html):
    soup = BeautifulSoup(html, 'html.parser')
    horses = {}
    tansho_odds = None
    for t in soup.find_all('table'):
        for row in t.find_all('tr'):
            cells = row.find_all(['td', 'th'])
            if len(cells) >= 9:
                try:
                    chakujun = int(cells[0].get_text(strip=True))
                    uma_raw = cells[3].get_text(strip=True)
                    uma = re.split(r'[牡牝セ]\d|せん\d', uma_raw)[0].strip()
                    if uma:
                        horses[uma] = chakujun
                except:
                    pass
            elif len(cells) == 3 and tansho_odds is None:
                txt = cells[1].get_text(strip=True)
                if '円' in txt:
                    try:
                        tansho_odds = int(re.sub(r'[^\d]', '', txt)) / 100.0
                    except:
                        pass
    return horses, tansho_odds

# 結果収集
rows = []
for venue_code, (kai, nichi) in venue_keys.items():
    vs = venue_map[venue_code]
    for rn in range(1, 13):
        key = f'{year2}{venue_code}{kai:02d}{nichi:02d}{rn:02d}'
        html = _get(f'https://sports.yahoo.co.jp/keiba/race/result/{key}')
        if not html or len(html) < 2000:
            continue
        horses, tansho = parse_result(html)
        if not horses:
            continue
        r = top1_new[top1_new['場 R'].str.contains(f'{vs}{rn}$')]
        if r.empty:
            continue
        horse = r.iloc[0]['馬名S']
        calib = r.iloc[0]['clogit_calib']
        actual = horses.get(horse)
        rows.append({
            'race': f'{vs}{rn}', 'horse': horse,
            'calib_new': calib, 'actual_rank': actual,
            'tansho': tansho, 'hit': actual == 1,
        })
        time.sleep(0.15)

df = pd.DataFrame(rows)
n = len(df)
hits = df['hit'].sum()
ret = df[df['hit']]['tansho'].fillna(0).sum()
roi = (ret - n) / n * 100

print("=== 05-30 修正後 ROI ===")
print(f"対象: {n}R  的中: {hits}R ({hits/n*100:.1f}%)  ROI: {roi:+.1f}%")
print()
print(df[['race','horse','calib_new','actual_rank','tansho','hit']].to_string(index=False))
