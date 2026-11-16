"""mark_stats_full.py - 新聞の印(◎○▲)の成績を、結果をネット競馬から直接取り直して全レースで集計する。
新聞HTMLの結果欄は15時時点で止まっている日があり(9/13〜11/7)、11R・12Rが欠けるため、その補正用。
  python src/mark_stats_full.py 出力.csv   （リポジトリ直下で実行。9/12以降の新聞が対象）
"""
import sys, re, glob, os, pandas as pd, numpy as np
sys.path.insert(0, 'src')
import make_newspaper as mn
import _fetch_and_predict as fp
short2code = {v: k for k, v in fp.VENUE_CODE_TO_SHORT.items()}
strip = lambda s: re.sub(r'<[^>]+>', '', s).strip()
OUT = sys.argv[1]
dates = sorted(os.path.basename(f)[10:18] for f in glob.glob(r'docs\newspaper_2026????.html'))
dates = [d for d in dates if d >= '20260912']
rows = []
for date in dates:
    try:
        res = mn.fetch_race_results(fp.get_race_ids(date))
    except Exception as e:
        print(date, 'ERR', e, flush=True); continue
    h = open(f'docs/newspaper_{date}.html', encoding='utf-8').read()
    for blk in h.split('<div class="race-block">')[1:]:
        v = re.search(r'race-venue">(.*?)</span>', blk).group(1); rn = int(re.search(r'race-rnum">(\d+)R', blk).group(1))
        seg = strip(re.search(r'race-seg"[^>]*>(.*?)</span>', blk).group(1)) if re.search(r'race-seg"', blk) else ''
        rr = res.get((short2code.get(v), rn))
        hs = []
        for tr in re.findall(r'<tr class="[^"]*expandable"[^>]*>(.*?)</tr>', blk, re.S):
            tds = re.findall(r'<td[^>]*>(.*?)</td>', tr, re.S)
            o = 1 if len(tds) >= 7 else 0
            if len(tds) < 6: continue
            m = re.match(r'(\d+)\.', strip(tds[o + 2]))
            if not m: continue
            try: od = float(strip(tds[o + 4]))
            except: od = np.nan
            hs.append(dict(no=int(m.group(1)), mark=strip(tds[o + 1])[:1], odds=od))
        top = [x for x in hs if x['mark'] in ('◎', '○', '▲')]
        if not top: continue
        t = top[0]; k = str(t['no']).zfill(2)
        ods = [x for x in hs if x['odds'] == x['odds']]
        fav = min(ods, key=lambda x: x['odds']) if ods else None
        if rr is None:
            rows.append(dict(date=date, venue=v, rnum=rn, seg=seg, mark=t['mark'], odds=t['odds'], rank=np.nan, tan=0, fuku=0, fav_rank=np.nan, has=0)); continue
        fk = str(fav['no']).zfill(2) if fav else None
        rows.append(dict(date=date, venue=v, rnum=rn, seg=seg, mark=t['mark'], odds=t['odds'], rank=rr['order'].get(k, np.nan),
                         tan=dict(rr['tansho']).get(k, 0), fuku=dict(rr['fukusho']).get(k, 0), fav_rank=rr['order'].get(fk, np.nan) if fk else np.nan, has=1))
    print(date, 'ok', len(rows), flush=True)
    pd.DataFrame(rows).to_csv(OUT, index=False, encoding='utf-8')
