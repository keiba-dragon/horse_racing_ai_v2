# coding: utf-8
"""
出馬表CSV + Yahoo オッズ → 競馬新聞HTML 生成
モデルなし版（ヒューリスティックスコア）
使い方:
  python src/gen_newspaper_from_csv.py --date 20260524 --csv 出馬表形式5月24日.csv
"""
import sys, io, os, json, re, time, argparse, unicodedata, urllib.request
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DIR = os.path.join(BASE_DIR, 'docs')

HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
MARKS = {1: '◎', 2: '○', 3: '▲', 4: '△', 5: '×'}
MARK_COLOR = {'◎': '#cc0000', '○': '#0055cc', '▲': '#007700', '△': '#e06000', '×': '#777777'}
VENUE_ORDER = ['京', '新', '東', '阪', '中', '中京', '福', '函', '札', '小']
VENUE_FULL = {'京': '京都', '新': '新潟', '東': '東京', '阪': '阪神', '中': '中山'}
VENUE_TO_CODE = {'東': '05', '中': '06', '中京': '07', '京': '08', '阪': '09',
                 '新': '04', '福': '03', '函': '02', '札': '01', '小': '10'}
SURFACE_EMO = {'芝': '🌿', 'ダ': '🏜️'}


# ── ヒューリスティックスコア ──────────────────────────────────────

def parse_chakujun(s):
    if pd.isna(s):
        return 10.0
    s = str(s).strip()
    kanji = {'１': 1, '２': 2, '３': 3, '４': 4, '５': 5, '６': 6, '７': 7,
             '８': 8, '９': 9, '10': 10, '11': 11, '12': 12, '13': 13,
             '14': 14, '15': 15, '16': 16, '17': 17, '18': 18}
    for k, v in kanji.items():
        if s.startswith(k):
            return float(v)
    if s in ('中止', '除外', '失格', '取消'):
        return 15.0
    try:
        return float(s)
    except Exception:
        return 10.0


def compute_score(row, surf):
    prev_fin = parse_chakujun(row.get('前着順'))
    s_prev = (18 - prev_fin) / 17

    if surf == '芝':
        w1, w2, w3 = float(row.get('芝1', 0) or 0), float(row.get('芝2', 0) or 0), float(row.get('芝3', 0) or 0)
    else:
        w1, w2, w3 = float(row.get('ダ1', 0) or 0), float(row.get('ダ2', 0) or 0), float(row.get('ダ3', 0) or 0)
    career = float(row.get('キャリア(最新)', 1) or 1)
    s_surf = (w1 * 3 + w2 * 2 + w3) / max(career, 1)

    all1 = float(row.get('全1', 0) or 0)
    s_win = min(all1 / max(career, 1), 0.8)

    prev_pop = row.get('前人気')
    if pd.notna(prev_pop) and float(prev_pop) > 0:
        s_form = (float(prev_pop) - prev_fin) / max(float(prev_pop), 1)
    else:
        s_form = 0.0

    chg = row.get('前馬体重増減', 0)
    s_wt = -abs(float(chg)) * 0.005 if pd.notna(chg) else 0.0

    kin = row.get('馬齢斤量差', 0)
    s_kin = float(kin) * -0.03 if pd.notna(kin) else 0.0

    s_jky = -0.05 if str(row.get('乗替', '')).strip() == '替' else 0.0

    return round(s_prev * 0.35 + s_surf * 0.25 + s_win * 0.20 + s_form * 0.10 + s_wt + s_kin + s_jky, 4)


# ── Yahoo オッズ取得 ──────────────────────────────────────────────

def fetch_race_ids(date_str):
    url = f'https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={date_str}'
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=8) as res:
            html = res.read().decode('utf-8', errors='replace')
        ids = sorted(set(re.findall(r'race_id=(\d{12})', html)))
        return ids
    except Exception as e:
        print(f'[WARN] race_ids取得失敗: {e}', file=sys.stderr)
        return []


def fetch_yahoo_odds(race_ids):
    sess = requests.Session()
    sess.headers.update(HEADERS)
    all_odds = {}
    for rid in race_ids:
        key = f'{rid[2:4]}{rid[4:6]}{rid[6:8]}{rid[8:10]}{rid[10:12]}'
        url = f'https://sports.yahoo.co.jp/keiba/race/denma/{key}'
        try:
            r = sess.get(url, timeout=6, verify=False)
            if r.status_code == 200 and len(r.text) > 3000:
                soup = BeautifulSoup(r.text, 'html.parser')
                for row in soup.select('table tr')[1:]:
                    tds = row.find_all('td')
                    if len(tds) < 8:
                        continue
                    uma = re.split(r'[牡牝セ]\d|せん\d', tds[2].get_text(strip=True))[0].strip()
                    if not uma:
                        continue
                    m = re.search(r'\(([0-9]+\.[0-9]+)\)', tds[7].get_text(strip=True))
                    if m:
                        try:
                            all_odds[uma] = float(m.group(1))
                        except Exception:
                            pass
        except Exception:
            pass
        time.sleep(0.12)
    return all_odds


def build_odds(df, date_str):
    # CSVからオッズ列があれば取得
    csv_odds = {}
    if '単オッズ' in df.columns and '馬名S' in df.columns:
        for _, row in df[df['単オッズ'].notna()].iterrows():
            csv_odds[str(row['馬名S']).strip()] = float(row['単オッズ'])

    print(f'CSVオッズ: {len(csv_odds)}頭', file=sys.stderr)

    race_ids = fetch_race_ids(date_str)
    print(f'race_ids: {len(race_ids)}R', file=sys.stderr)

    yahoo = fetch_yahoo_odds(race_ids)
    print(f'Yahooオッズ: {len(yahoo)}頭', file=sys.stderr)

    merged = {**yahoo, **csv_odds}  # CSV優先
    print(f'合計: {len(merged)}頭', file=sys.stderr)
    return merged, race_ids


# ── HTML 生成 ──────────────────────────────────────────────────────

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: 'Hiragino Kaku Gothic Pro', 'Meiryo', sans-serif;
  font-size: 11px;
  background: #f5f0e8;
  color: #1a1a1a;
}
.page-header {
  background: #1a1a2e;
  color: #fff;
  padding: 10px 16px;
  display: flex;
  align-items: baseline;
  gap: 16px;
  flex-wrap: wrap;
}
.page-title { font-size: 22px; font-weight: bold; letter-spacing: 2px; color: #f0d060; }
.page-sub { font-size: 12px; color: #aaa; }
.page-meta { margin-left: auto; font-size: 11px; color: #ccc; }
.legend-bar {
  background: #2d2d44; color: #eee; padding: 5px 16px;
  display: flex; gap: 16px; flex-wrap: wrap; font-size: 11px;
}
.legend-bar span { white-space: nowrap; }
.leg-mark { font-weight: bold; }
.venue-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
  gap: 12px; padding: 12px; max-width: 1400px; margin: 0 auto;
}
.venue-col { display: flex; flex-direction: column; gap: 8px; }
.venue-title {
  background: #1a1a2e; color: #f0d060; font-size: 15px; font-weight: bold;
  padding: 6px 12px; border-radius: 4px 4px 0 0; letter-spacing: 4px;
}
.race-card {
  background: #fff; border: 1px solid #c8b89a;
  border-radius: 4px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}
.race-header {
  background: #3a3a5c; color: #fff; padding: 5px 10px;
  display: flex; align-items: center; gap: 8px;
}
.race-label { font-size: 15px; font-weight: bold; color: #f0d060; min-width: 28px; }
.course { font-size: 12px; color: #cce0ff; }
.n-horses { font-size: 11px; color: #aaa; margin-left: auto; }
.badge-g1 { background: #cc0000; color: #fff; font-size: 10px; font-weight: bold; padding: 1px 5px; border-radius: 3px; }
.race-table { width: 100%; border-collapse: collapse; }
.race-table th {
  background: #eae4d8; color: #555; font-size: 10px; padding: 3px 4px;
  text-align: center; border-bottom: 1px solid #c8b89a;
}
.race-table td { padding: 3px 4px; border-bottom: 1px solid #ede8df; vertical-align: middle; }
.race-table tr:last-child td { border-bottom: none; }
.row-top { background: #fff8f0; }
.race-table tr:hover { background: #f5f0e8; }
.mark { text-align: center; font-size: 14px; font-weight: bold; width: 22px; }
.umaban { text-align: center; color: #666; width: 24px; }
.horse { font-weight: 500; max-width: 110px; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }
.pop { text-align: right; color: #555; width: 28px; }
.odds { text-align: right; font-weight: bold; color: #1a3a6e; width: 42px; }
.score { text-align: right; color: #555; width: 44px; font-size: 10px; }
a.race-link { color: #f0d060; text-decoration: none; }
a.race-link:hover { text-decoration: underline; }
a.horse-link { color: inherit; text-decoration: none; }
a.horse-link:hover { text-decoration: underline; color: #0055cc; }
.page-footer { text-align: center; padding: 12px; color: #888; font-size: 10px; border-top: 1px solid #c8b89a; margin-top: 8px; }
.no-odds { color: #bbb; }
@media (max-width: 700px) { .venue-grid { grid-template-columns: 1fr; } .horse { max-width: 80px; } }
"""


def horse_netkeiba_url(name):
    from urllib.parse import quote
    return f'https://db.netkeiba.com/?pid=horse_list&word={quote(name)}'


def race_netkeiba_url(race_id):
    return f'https://race.netkeiba.com/race/newspaper.html?race_id={race_id}&rf=shutuba_submenu'


def gen_race_card(race_id, race_label, course_str, n_horses, rows_sorted, odds_dict):
    rid_link = race_netkeiba_url(race_id) if race_id else '#'
    grade = ''
    if 'G1' in course_str or 'オークス' in course_str or '優駿牝馬' in course_str:
        grade = '<span class="badge-g1">G1</span>'

    html = f'''
<div class="race-card">
  <div class="race-header">
    <span class="race-label"><a href="{rid_link}" target="_blank" class="race-link">{race_label}</a></span>
    <span class="course">{course_str}</span>
    <span class="n-horses">{n_horses}頭</span>
    {grade}
  </div>
  <table class="race-table">
    <thead>
      <tr><th>印</th><th>馬番</th><th>馬名</th><th>人気</th><th>オッズ</th><th>score</th></tr>
    </thead>
    <tbody>
'''
    for i, row in enumerate(rows_sorted[:5]):
        rank = i + 1
        mark = MARKS.get(rank, '')
        color = MARK_COLOR.get(mark, '#777')
        horse = str(row.get('馬名S', '')).strip()
        umaban_raw = row.get('馬番', '')
        try:
            umaban = int(float(str(umaban_raw))) if pd.notna(umaban_raw) and str(umaban_raw).strip() not in ('', '未', 'nan') else ''
        except Exception:
            umaban = str(umaban_raw).strip()

        odds_val = odds_dict.get(horse)
        if odds_val is not None:
            odds_html = f'{odds_val:.1f}'
            # rank by odds within race
            sorted_odds = sorted([v for v in [odds_dict.get(str(r.get('馬名S', '')).strip()) for r in rows_sorted] if v is not None])
            pop = sorted_odds.index(odds_val) + 1 if odds_val in sorted_odds else '-'
        else:
            odds_html = '<span class="no-odds">---</span>'
            pop = '-'

        sc = row.get('_score', 0)
        row_cls = ' class="row-top"' if rank == 1 else ''
        hurl = horse_netkeiba_url(horse)
        html += f'''      <tr{row_cls}>
        <td class="mark" style="color:{color}">{mark}</td>
        <td class="umaban">{umaban}</td>
        <td class="horse"><a href="{hurl}" target="_blank" class="horse-link">{horse}</a></td>
        <td class="pop">{pop}</td>
        <td class="odds">{odds_html}</td>
        <td class="score">{sc:.3f}</td>
      </tr>
'''
    html += '    </tbody>\n  </table>\n</div>\n'
    return html


def generate(date_str, csv_path):
    df = pd.read_csv(csv_path, encoding='cp932')

    # ヒューリスティックスコア
    df['_score'] = df.apply(lambda r: compute_score(r, str(r.get('芝ダ', '')).strip()), axis=1)

    odds_dict, race_ids = build_odds(df, date_str)

    # race_id → (venue_short, rnum)
    rid_map = {}
    for rid in race_ids:
        vc = rid[4:6]
        rnum = int(rid[10:12])
        short = next((s for s, c in VENUE_TO_CODE.items() if c == vc), None)
        if short:
            rid_map[(short, rnum)] = rid

    dt = datetime.strptime(date_str, '%Y%m%d')
    date_jp = f'{dt.year}.{dt.month:02d}.{dt.day:02d}'
    youbi = ['月', '火', '水', '木', '金', '土', '日'][dt.weekday()]
    date_label = f'{date_jp}（{youbi}）'

    races = df.drop_duplicates('場 R').sort_values('発走時刻')[['場 R', '場所', 'Ｒ', '芝ダ', '距離', '頭数', '発走時刻', 'レース名']].values

    # 会場別まとめ
    venue_map = {}
    for race_info in races:
        race_id_col, venue_full, rnum, surf, dist, hdnum, jikan, race_name = race_info
        venue_s = str(race_id_col).rstrip('0123456789')
        venue_map.setdefault(venue_s, []).append(race_info)

    n_races = len(races)
    n_horses = len(df)
    venues_str = '・'.join([VENUE_FULL.get(v, v) for v in VENUE_ORDER if v in venue_map])

    html = f'''<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>競馬新聞 {date_label}</title>
<style>{CSS}</style>
</head>
<body>
<div class="page-header">
  <div class="page-title">競馬新聞</div>
  <div class="page-sub">{date_label}</div>
  <div class="page-meta">
    {venues_str} &nbsp;|&nbsp;
    {n_races}レース {n_horses}頭 &nbsp;|&nbsp;
    ヒューリスティックモデル（指数なし）
  </div>
</div>
<div class="legend-bar">
  <span><span class="leg-mark" style="color:#f0d060">◎</span> 1位</span>
  <span><span class="leg-mark" style="color:#88ccff">○</span> 2位</span>
  <span><span class="leg-mark" style="color:#88ff88">▲</span> 3位</span>
  <span><span class="leg-mark" style="color:#ffaa55">△</span> 4位</span>
  <span><span class="leg-mark" style="color:#aaa">×</span> 5位</span>
  <span style="margin-left:8px">|</span>
  <span>score = 前走着順(35%) + 芝ダ適性(25%) + 通算勝率(20%) + 前走人気超過(10%) + 体重・斤量補正</span>
  <span style="color:#aaa">オッズ: Yahoo競馬 + CSV取得（---は未発売）</span>
</div>
<div class="venue-grid">
'''

    for venue_s in VENUE_ORDER:
        if venue_s not in venue_map:
            continue
        venue_label = VENUE_FULL.get(venue_s, venue_s)
        html += f'  <div class="venue-col">\n    <div class="venue-title">{venue_label}</div>\n'

        for race_info in venue_map[venue_s]:
            race_id_col, venue_full, rnum, surf, dist, hdnum, jikan, race_name = race_info
            try:
                rnum_int = int(float(str(rnum)))
            except Exception:
                rnum_int = 0
            rid = rid_map.get((venue_s, rnum_int), '')
            surf_str = str(surf).strip()
            emo = SURFACE_EMO.get(surf_str, surf_str)
            try:
                dist_int = int(float(str(dist)))
            except Exception:
                dist_int = 0
            course_str = f'{emo}{surf_str}{dist_int}m'
            if race_name and str(race_name).strip():
                course_str += f' {str(race_name).strip()}'
            try:
                n = int(float(str(hdnum)))
            except Exception:
                n = '?'

            grp = df[df['場 R'] == race_id_col].copy()
            grp['_score'] = grp.apply(lambda r: compute_score(r, surf_str), axis=1)
            rows_sorted = grp.sort_values('_score', ascending=False).to_dict('records')

            html += gen_race_card(rid, f'{rnum_int}R', course_str, n, rows_sorted, odds_dict)

        html += '  </div>\n'

    html += f'''</div>
<div class="page-footer">
  生成: {datetime.now().strftime("%Y-%m-%d %H:%M")} &nbsp;|&nbsp;
  clogit競馬新聞 (ヒューリスティックモデル) &nbsp;|&nbsp;
  オッズ未取得レースは発売開始後に再取得してください
</div>
</body>
</html>'''

    out_path = os.path.join(DOCS_DIR, f'newspaper_{date_str}.html')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'出力: {out_path}')
    return out_path


if __name__ == '__main__':
    import urllib3
    urllib3.disable_warnings()
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', default=datetime.now().strftime('%Y%m%d'))
    ap.add_argument('--csv',  default=None)
    args = ap.parse_args()

    date_str = args.date
    dt = datetime.strptime(date_str, '%Y%m%d')
    csv_default = f'出馬表形式{dt.month}月{dt.day}日.csv'
    csv_path = args.csv or csv_default

    if not os.path.exists(csv_path):
        csv_path = os.path.join(BASE_DIR, csv_default)
    if not os.path.exists(csv_path):
        print(f'[ERROR] CSV not found: {csv_path}', file=sys.stderr)
        sys.exit(1)

    generate(date_str, csv_path)
