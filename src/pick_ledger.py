"""pick_ledger.py - レース前の予想(各レース1位の印 ◎○▲)を台帳に固定する。

目的: 成績の集計を「レース前の予想だけ」で行う。新聞は更新のたびに再計算されて上書きされるため、
結果欄が出る前の最後の版の予想を、1レース1行で data/processed/pick_ledger.csv に保存する。

ルール
  - 結果欄(result-banner)が無いレースだけ、台帳に書く／上書きする(最新のレース前の版が残る)
  - 結果欄が出たレースは、二度と書き換えない(後出し情報を入れない)
  - first_no / n_versions: 最初に見た版の本命の馬番と、見た版の数(版による予想のブレを測る用)

使い方
  python src/pick_ledger.py backfill   # Gitの履歴(9/12以降の新聞)から台帳を作り直す
  update_from_html(html, source, snap_time)  # make_newspaper.py から新聞を作るたびに呼ぶ
"""
import re, os, sys, html as _html, subprocess, datetime as dt
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER = os.path.join(ROOT, 'data', 'processed', 'pick_ledger.csv')
COLS = ['date', 'venue', 'rnum', 'seg', 'mark', 'no', 'name', 'odds', 'prob', 'src', 'snap_time', 'first_no', 'n_versions']
KEY = ['date', 'venue', 'rnum']
FROM_DATE = '20260912'  # これより前の新聞は結果入りで一括再生成されていて、レース前の版が無い

_strip = lambda s: _html.unescape(re.sub(r'<[^>]+>', '', s)).strip()


def parse_newspaper(html: str, date: str) -> list:
    """新聞HTMLから、レースごとの {key, seg, has_result, pick(本命の印の馬)} を取り出す。"""
    out = []
    for blk in html.split('<div class="race-block">')[1:]:
        m_v = re.search(r'race-venue">(.*?)</span>', blk)
        m_r = re.search(r'race-rnum">(\d+)R', blk)
        if not (m_v and m_r):
            continue
        m_s = re.search(r'race-seg"[^>]*>(.*?)</span>', blk)
        seg = _strip(m_s.group(1)) if m_s else ''
        seg = (re.match(r'^(ダ長|ダ短|芝短|芝中|芝長)', seg) or [None, seg])[1] if seg else ''
        has_result = 'class="result-banner"' in blk
        pick = None
        for tr in re.findall(r'<tr class="[^"]*expandable"[^>]*>(.*?)</tr>', blk, re.S):
            tds = re.findall(r'<td[^>]*>(.*?)</td>', tr, re.S)
            if len(tds) < 6:
                continue
            o = 1 if len(tds) >= 7 else 0
            mark = _strip(tds[o + 1])[:1]
            if mark not in ('◎', '○', '▲'):
                continue
            nm = re.match(r'(\d+)\.(.*?)(?:\s*\d+\(|$)', _strip(tds[o + 2]))
            if not nm:
                continue
            try:
                odds = float(_strip(tds[o + 4]))
            except ValueError:
                odds = float('nan')
            try:
                prob = float(_strip(tds[o + 5]).rstrip('%'))
            except ValueError:
                prob = float('nan')
            pick = dict(mark=mark, no=int(nm.group(1)), name=nm.group(2).replace('▾', '').strip(), odds=odds, prob=prob)
            break
        out.append(dict(date=date, venue=_strip(m_v.group(1)), rnum=int(m_r.group(1)), seg=seg, has_result=has_result, pick=pick))
    return out


def load_ledger() -> pd.DataFrame:
    if os.path.exists(LEDGER):
        return pd.read_csv(LEDGER, dtype={'date': str}, encoding='utf-8')
    return pd.DataFrame(columns=COLS)


def update_ledger(df: pd.DataFrame, races: list, source: str, snap_time: str):
    """races(parse_newspaper の結果)を台帳 df に反映。結果入りのレースは無視。(df, 追加/更新数) を返す。"""
    idx = {(r.date, r.venue, int(r.rnum)): i for i, r in df.iterrows()}
    rows = df.to_dict('records')
    n_changed = 0
    for r in races:
        if r['has_result'] or not r['pick']:
            continue
        k = (r['date'], r['venue'], r['rnum'])
        p = r['pick']
        rec = dict(date=r['date'], venue=r['venue'], rnum=r['rnum'], seg=r['seg'], mark=p['mark'], no=p['no'], name=p['name'],
                   odds=p['odds'], prob=p['prob'], src=source, snap_time=snap_time)
        if k in idx:
            old = rows[idx[k]]
            rec['first_no'] = old['first_no']
            rec['n_versions'] = int(old['n_versions']) + 1
            rows[idx[k]] = rec
        else:
            rec['first_no'] = p['no']
            rec['n_versions'] = 1
            idx[k] = len(rows)
            rows.append(rec)
        n_changed += 1
    return pd.DataFrame(rows, columns=COLS), n_changed


def save_ledger(df: pd.DataFrame):
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
    df = df.sort_values(KEY).reset_index(drop=True)
    df.to_csv(LEDGER, index=False, encoding='utf-8')


def update_from_html(html: str, date: str, source: str = 'live', snap_time: str = None):
    """新聞を作った直後に呼ぶ。結果欄の無いレースの予想だけを台帳に反映する。"""
    if date < FROM_DATE:
        return 0
    snap_time = snap_time or dt.datetime.now().strftime('%Y-%m-%d %H:%M')
    df, n = update_ledger(load_ledger(), parse_newspaper(html, date), source, snap_time)
    if n:
        save_ledger(df)
    return n


def _git(*args) -> bytes:
    return subprocess.run(['git', '-C', ROOT, *args], capture_output=True, check=True).stdout


def backfill():
    """Gitの履歴を古い順にたどり、9/12以降の新聞の各版から台帳を作り直す(既存の台帳は捨てて作り直す)。"""
    log = _git('log', '--reverse', '--since=2026-09-10', '--format=@@%H %ct', '--name-only', '--', 'docs/newspaper_2026*.html').decode('utf-8')
    commits, cur = [], None
    for line in log.splitlines():
        line = line.strip()
        if line.startswith('@@'):
            h, ct = line[2:].split()
            cur = (h, int(ct), [])
            commits.append(cur)
        elif line and cur is not None:
            cur[2].append(line)
    df = pd.DataFrame(columns=COLS)
    n_ver = 0
    for h, ct, files in commits:
        snap = dt.datetime.fromtimestamp(ct).strftime('%Y-%m-%d %H:%M')
        for f in files:
            m = re.search(r'newspaper_(\d{8})\.html$', f)
            if not m or m.group(1) < FROM_DATE:
                continue
            try:
                html = _git('show', f'{h}:{f}').decode('utf-8')
            except subprocess.CalledProcessError:
                continue
            df, n = update_ledger(df, parse_newspaper(html, m.group(1)), h[:7], snap)
            n_ver += 1
    save_ledger(df)
    print(f'版 {n_ver} 件を処理 → 台帳 {len(df)} レース  {df.date.min()}〜{df.date.max()}')
    return df


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'backfill':
        backfill()
    else:
        print(__doc__)
