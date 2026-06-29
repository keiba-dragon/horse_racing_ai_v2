"""Yahoo競馬の結果ページから着順・単勝オッズを取得してROIを計算"""
import sys, ssl, gzip, re, json, pickle, time
import urllib.request
import pandas as pd
from bs4 import BeautifulSoup

sys.stdout.reconfigure(encoding='utf-8')

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
    except Exception as e:
        print(f'  fetch失敗: {url}: {e}')
        return None

def parse_result(html, debug=False):
    soup = BeautifulSoup(html, 'html.parser')
    tables = soup.find_all('table')

    horses = {}   # 馬名 → 着順
    tansho_odds = None

    for t in tables:
        rows = t.find_all('tr')
        if not rows:
            continue
        # メイン結果テーブル: 9列以上
        for row in rows:
            cells = row.find_all(['td', 'th'])
            if len(cells) >= 9:
                chakujun_raw = cells[0].get_text(strip=True)
                uma_raw = cells[3].get_text(strip=True)
                uma_name = re.split(r'[牡牝セ]\d|せん\d', uma_raw)[0].strip()
                try:
                    chakujun = int(chakujun_raw)
                except ValueError:
                    continue
                if uma_name:
                    horses[uma_name] = chakujun

    # 単勝オッズ: 払戻テーブル（3列）の最初の数値行
    for t in tables:
        rows = t.find_all('tr')
        for row in rows:
            cells = row.find_all(['td', 'th'])
            if len(cells) == 3:
                payout_text = cells[1].get_text(strip=True)
                if '円' in payout_text:
                    try:
                        yen = int(re.sub(r'[^\d]', '', payout_text))
                        tansho_odds = yen / 100.0
                        if debug:
                            print(f'  単勝払戻: {yen}円 → {tansho_odds}倍')
                        break
                    except Exception:
                        pass
        if tansho_odds:
            break

    return horses, tansho_odds


# venue_keys for 05-30
venue_keys = {'08': [3, 11], '05': [2, 11]}
year2 = '26'

# 予測キャッシュ（修正後）
with open('data/raw/cache/出馬表形式05月30日_api.cache.pkl', 'rb') as f:
    cache = pickle.load(f)
pred_df = cache['result'].copy()

pred_df['_pred_rank'] = pred_df.groupby('場 R')['clogit_calib'].rank(method='first', ascending=False)
top1_pred = pred_df[pred_df['_pred_rank'] == 1][['場 R','馬名S','clogit_calib']].copy()

venue_map = {'08': '京', '05': '東'}
results = []

for venue_code, (kai, nichi) in venue_keys.items():
    venue_short = venue_map.get(venue_code, venue_code)
    for race_num in range(1, 13):
        key = f'{year2}{venue_code}{kai:02d}{nichi:02d}{race_num:02d}'
        url = f'https://sports.yahoo.co.jp/keiba/race/result/{key}'
        html = _get(url)
        if not html or len(html) < 2000:
            continue
        dbg = (race_num == 2 and venue_code == '08')  # 京2でデバッグ
        horses, tansho = parse_result(html, debug=dbg)
        if not horses:
            continue
        row = top1_pred[top1_pred['場 R'].str.contains(f'{venue_short}{race_num}$')]
        if row.empty:
            continue
        pred_horse = row.iloc[0]['馬名S']
        actual_rank = horses.get(pred_horse, None)
        results.append({
            'race': f'{venue_short}{race_num}',
            'pred_horse': pred_horse,
            'actual_rank': actual_rank,
            'tansho_odds': tansho,
            'hit': actual_rank == 1,
        })
        status = f'HIT {tansho}倍' if actual_rank == 1 else f'{actual_rank}着'
        print(f'  {venue_short}{race_num}: {pred_horse} → {status}')
        time.sleep(0.2)

print()
if results:
    df = pd.DataFrame(results)
    n = len(df)
    hits = df['hit'].sum()
    total_return = df[df['hit']]['tansho_odds'].fillna(0).sum()
    roi = (total_return - n) / n * 100
    print(f"=== 05-30 修正後 ROI ===")
    print(f"対象: {n}R, 的中: {hits}R ({hits/n*100:.1f}%)")
    print(f"回収: {total_return:.1f} / 投資: {n}")
    print(f"ROI: {roi:+.1f}%")
    print()
    print(df[['race','pred_horse','actual_rank','tansho_odds','hit']].to_string(index=False))
