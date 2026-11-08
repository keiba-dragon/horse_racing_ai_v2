"""mark_stats.py - 新聞(docs/newspaper_*.html)の結果欄から、◎○▲(各レース1位の印)の成績を集計する。
8/15版以降が対象。期間別・週別・▲のセグメント別を出力（回収率は100円買った場合の倍率）。
  python src/mark_stats.py  （出力が長いのでファイルへリダイレクト推奨）
"""
import re, glob, os, html, datetime as dt
import pandas as pd

strip = lambda x: html.unescape(re.sub(r'<[^>]+>', '', x)).strip()
rows = []
for f in sorted(glob.glob(r'C:\horse_racing_ai_v2\docs\newspaper_2026????.html')):
    d = os.path.basename(f)[10:18]
    if d < '20260815':
        continue
    s = open(f, encoding='utf-8').read()
    blocks = s.split('<div class="race-block">')[1:]
    for b in blocks:
        venue = strip(re.search(r'class="race-venue">(.*?)</span>', b).group(1))
        rnum = strip(re.search(r'class="race-rnum">(.*?)</span>', b).group(1))
        seg = strip(re.search(r'class="race-seg"[^>]*>(.*?)</span>', b).group(1))
        ban = re.search(r'class="result-banner">(.*?)</div>', b, re.S)
        if not ban:
            continue
        bt = strip(ban.group(1))
        m = re.search(r'単勝\s*(\d+)番\s*¥([\d,]+)', bt)
        if not m:
            continue
        win_no, win_pay = int(m.group(1)), int(m.group(2).replace(',', ''))
        fuku = {int(a): int(c.replace(',', '')) for a, c in re.findall(r'(\d+)番\s*¥([\d,]+)', bt.split('複勝')[1])} if '複勝' in bt else {}
        horses = []
        for tr in re.findall(r'<tr class="[^"]*expandable"[^>]*>(.*?)</tr>', b, re.S):
            tds = re.findall(r'<td[^>]*>(.*?)</td>', tr, re.S)
            if len(tds) < 7:
                continue
            nm = re.match(r'\s*(\d+)\.', strip(tds[3]))
            if not nm:
                continue
            try:
                odds = float(strip(tds[5]))
            except ValueError:
                odds = float('nan')
            horses.append(dict(no=int(nm.group(1)), rank=strip(tds[1]), mark=strip(tds[2]), odds=odds, fin=strip(tds[0])))
        if not horses:
            continue
        top = [h for h in horses if h['mark'] in ('◎', '○', '▲')]
        if not top:
            continue
        t = top[0]
        od = [h for h in horses if h['odds'] == h['odds']]
        fav = min(od, key=lambda h: h['odds']) if od else None
        rows.append(dict(date=d, venue=venue, rnum=rnum, seg=seg, mark=t['mark'], no=t['no'], odds=t['odds'],
                         win=int(t['no'] == win_no), tan=win_pay if t['no'] == win_no else 0,
                         place=int(t['no'] in fuku), fuku=fuku.get(t['no'], 0),
                         fav_win=int(fav['no'] == win_no) if fav else None))
df = pd.DataFrame(rows)
os.makedirs(r'C:\horse_racing_ai_v2\logs', exist_ok=True)
df.to_csv(r'C:\horse_racing_ai_v2\logs\mark_stats.csv', index=False, encoding='utf-8')


def summ(g):
    return pd.Series(dict(n=len(g), win=g.win.mean(), place=g.place.mean(), tan_roi=g.tan.mean() / 100, fuku_roi=g.fuku.mean() / 100,
                          avg_odds=g.odds.mean(), fav_win=g.fav_win.mean()))


pd.set_option('display.width', 200)
print('期間', df.date.min(), df.date.max(), 'races', len(df))
df['dd'] = pd.to_datetime(df.date)
last = df.dd.max()
for label, lo in [('8/15以降(全期間)', df.dd.min()), ('直近8週', last - pd.Timedelta(days=56)), ('直近4週', last - pd.Timedelta(days=28)), ('直近2週', last - pd.Timedelta(days=14))]:
    x = df[df.dd >= lo]
    print('\n===', label, x.date.min(), '-', x.date.max(), 'n=', len(x))
    print(x.groupby('mark').apply(summ, include_groups=False).round(3).to_string())
    b = x[x.mark.isin(['◎', '○'])]
    print('◎○合計', summ(b).round(3).to_dict())
print('\n=== 週別 ◎○ ===')
df['wk'] = df.dd.dt.to_period('W-SUN').astype(str)
b = df[df.mark.isin(['◎', '○'])]
print(b.groupby('wk').apply(summ, include_groups=False).round(3).to_string())
print('\n=== 週別 ▲ ===')
print(df[df.mark == '▲'].groupby('wk').apply(summ, include_groups=False).round(3).to_string())

print('\n=== ▲ セグメント別 ===')
df['seg'] = df.seg.str.extract(r'^(ダ長|ダ短|芝短|芝中|芝長)')
for label, x in [('9/13以降', df[(df.mark == '▲') & (df.dd >= '2026-09-13')]), ('8/15以降', df[df.mark == '▲'])]:
    print(label)
    print(x.groupby('seg').apply(summ, include_groups=False).round(3).to_string())
