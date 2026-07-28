# coding: utf-8
"""
make_result_report.py — 日次予想結果レポート生成

usage:
    python src/make_result_report.py              # 今日
    python src/make_result_report.py 20260620     # 日付指定 (YYYYMMDD)

動作:
  - newspaper_YYYYMMDD.html を解析して AI1位馬の的中・回収を集計
  - docs/report_YYYYMMDD.html として保存
  - docs/reports.html インデックスを更新
"""
import sys, os, re
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_newspaper(date_str: str):
    """newspaper HTML から AI1位馬の結果一覧を返す。"""
    path = os.path.join(BASE_DIR, 'docs', f'newspaper_{date_str}.html')
    if not os.path.exists(path):
        print(f'[ERROR] newspaper not found: {path}')
        return []

    with open(path, encoding='utf-8') as f:
        html = f.read()

    # race title を前後の行から拾う
    # row-r1 の直前にある race-header を対応づけるため、セクション単位で処理
    # セクションを race-header で分割
    sections = re.split(r'(?=<[^>]*class="[^"]*race-header[^"]*")', html)

    picks = []
    for sec in sections:
        # レース名
        race_m = re.search(
            r'<[^>]*class="[^"]*race-title[^"]*"[^>]*>(.*?)</',
            sec, re.DOTALL)
        if not race_m:
            # 別パターン: h2/h3 heading
            race_m = re.search(r'<h[23][^>]*>(.*?)</h[23]>', sec, re.DOTALL)
        race_name = re.sub(r'<[^>]+>', '', race_m.group(1)).strip() if race_m else ''

        # AI1位馬の行（印により row-r1 / row-buy / row-maru に分かれる）
        r1_rows = re.findall(
            r'<tr[^>]*class="[^"]*(?:row-r1|row-buy|row-maru)[^"]*"[^>]*>(.*?)</tr>',
            sec, re.DOTALL)
        for row in r1_rows:
            cells = re.findall(r'<td[^>]*>(.*?)</td>', row, re.DOTALL)
            clean = [re.sub(r'<[^>]+>', '', c).strip() for c in cells]

            # 列構成: [着順(結果確定時のみ), AIrank, 印(◎/○/▲/-), 馬名, 騎手, odds, prob]
            if len(clean) == 7:
                jyuni_s, ai_rank_s, mark, horse_raw, jockey, odds_s, prob_s = clean
            elif len(clean) == 6:
                jyuni_s = ''
                ai_rank_s, mark, horse_raw, jockey, odds_s, prob_s = clean
            else:
                continue

            if ai_rank_s != '1':
                continue

            horse = re.sub(r'\s+\d+\([+-]?\d+\).*', '', horse_raw)
            horse = re.sub(r'^\d+\.', '', horse).strip()
            try:
                odds = float(odds_s)
            except Exception:
                odds = 0.0
            jyuni = int(jyuni_s) if jyuni_s.lstrip('-').isdigit() else None

            picks.append({
                'race': race_name, 'horse': horse, 'jockey': jockey,
                'odds': odds, 'mark': mark, 'jyuni': jyuni,
            })
    return picks


def build_html(date_str: str, picks: list) -> str:
    ymd = f'{date_str[:4]}/{date_str[4:6]}/{date_str[6:]}'

    confirmed = [p for p in picks if p['jyuni'] is not None]
    hits      = [p for p in confirmed if p['jyuni'] == 1]
    n_c       = len(confirmed)
    n_h       = len(hits)
    total_return = sum(p['odds'] for p in hits)
    roi = (total_return - n_c) / n_c * 100 if n_c > 0 else float('nan')
    win_rate = n_h / n_c * 100 if n_c > 0 else 0.0

    def roi_color(v):
        if v >= 0:
            return '#1a7a1a'
        elif v >= -20:
            return '#b36000'
        else:
            return '#c0392b'

    MARK_COLOR = {'◎': '#c0392b', '○': '#e67e22', '▲': '#27ae60', '-': '#aaa'}

    def mark_stats(subset):
        conf = [p for p in subset if p['jyuni'] is not None]
        h    = [p for p in conf if p['jyuni'] == 1]
        n_c2, n_h2 = len(conf), len(h)
        ret2 = sum(p['odds'] for p in h)
        roi2 = (ret2 - n_c2) / n_c2 * 100 if n_c2 > 0 else float('nan')
        return n_c2, n_h2, roi2

    mark_rows_html = ''
    for mk in ('◎', '○', '▲'):
        subset = [p for p in picks if p['mark'] == mk]
        if not subset:
            continue
        n_c2, n_h2, roi2 = mark_stats(subset)
        roi2_s = f'{roi2:+.1f}%' if n_c2 > 0 else '—'
        roi2_c = roi_color(roi2) if n_c2 > 0 else '#888'
        mark_rows_html += f'''
        <tr>
          <td style="color:{MARK_COLOR[mk]};font-weight:bold;font-size:16px">{mk}</td>
          <td style="text-align:center">{n_c2}R</td>
          <td style="text-align:center">{n_h2}R</td>
          <td style="text-align:center">{n_h2/n_c2*100 if n_c2 else 0:.1f}%</td>
          <td style="text-align:right;color:{roi2_c};font-weight:bold">{roi2_s}</td>
        </tr>'''

    rows_html = ''
    for p in picks:
        if p['jyuni'] is None:
            result_mark = '<span style="color:#888">未確定</span>'
            bg   = ''
            j_td = '<td style="color:#888">—</td>'
        elif p['jyuni'] == 1:
            result_mark = '◎ <b>的中！</b>'
            bg   = ' style="background:#fff9e6"'
            j_td = f'<td style="font-weight:bold;color:#1a7a1a">1着</td>'
        else:
            result_mark = f'{p["jyuni"]}着'
            bg   = ''
            j_td = f'<td style="color:#555">{p["jyuni"]}着</td>'

        mk = p.get('mark', '-')
        mk_col = MARK_COLOR.get(mk, '#aaa')

        race_short = re.sub(r'^.*?(\d+R)', r'\1', p['race']) if p['race'] else '—'
        rows_html += f'''
        <tr{bg}>
          <td style="color:#555;font-size:13px">{race_short}</td>
          <td style="font-weight:bold">{p["horse"]}</td>
          <td style="color:#555;font-size:13px">{p["jockey"]}</td>
          <td style="text-align:right">{p["odds"]:.1f}倍</td>
          <td style="text-align:center;color:{mk_col};font-weight:bold;font-size:15px">{mk}</td>
          {j_td}
          <td>{result_mark}</td>
        </tr>'''

    roi_c = roi_color(roi) if n_c > 0 else '#888'
    roi_s = f'{roi:+.1f}%' if n_c > 0 else '—'

    return f'''<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>予想結果 {ymd}</title>
<style>
  body {{ font-family: 'Helvetica Neue', Arial, sans-serif; background:#f5f5f5; margin:0; padding:16px; }}
  .card {{ background:#fff; border-radius:10px; box-shadow:0 1px 4px rgba(0,0,0,.12);
           max-width:780px; margin:0 auto; padding:20px 24px; }}
  h1 {{ font-size:20px; margin:0 0 4px; color:#222; }}
  .subtitle {{ color:#888; font-size:13px; margin-bottom:20px; }}
  .summary {{ display:flex; gap:20px; flex-wrap:wrap; margin-bottom:20px; }}
  .stat {{ background:#f8f8f8; border-radius:8px; padding:12px 18px; min-width:120px; }}
  .stat .label {{ font-size:12px; color:#888; margin-bottom:4px; }}
  .stat .value {{ font-size:24px; font-weight:bold; }}
  table {{ width:100%; border-collapse:collapse; font-size:14px; }}
  th {{ background:#f0f0f0; padding:8px 10px; text-align:left; font-size:12px; color:#555;
        border-bottom:2px solid #ddd; }}
  td {{ padding:9px 10px; border-bottom:1px solid #eee; vertical-align:middle; }}
  tr:hover td {{ background:#fafafa; }}
  .nav {{ display:flex; gap:10px; margin-bottom:16px; }}
  .nav a {{ color:#1a7a9a; text-decoration:none; font-size:13px; }}
  .nav a:hover {{ text-decoration:underline; }}
</style>
</head>
<body>
<div class="card">
  <div class="nav">
    <a href="index.html">← トップ</a>
    <a href="newspapers.html">予想新聞一覧</a>
    <a href="reports.html">結果レポート一覧</a>
    <a href="newspaper_{date_str}.html">📰 {ymd} 新聞</a>
  </div>
  <h1>予想結果レポート — {ymd}</h1>
  <div class="subtitle">AI1位指名馬（的中率最大化モデル）の単勝成績</div>

  <div class="summary">
    <div class="stat">
      <div class="label">的中率</div>
      <div class="value">{win_rate:.1f}%</div>
    </div>
    <div class="stat">
      <div class="label">的中 / 確定R</div>
      <div class="value">{n_h} / {n_c}</div>
    </div>
    <div class="stat">
      <div class="label">単勝回収率</div>
      <div class="value" style="color:{roi_c}">{roi_s}</div>
    </div>
    <div class="stat">
      <div class="label">総投資</div>
      <div class="value">{n_c}R</div>
    </div>
  </div>

  <h2 style="font-size:15px;margin:20px 0 8px;color:#333;border-left:4px solid #1a7a9a;padding-left:8px">印別成績</h2>
  <table>
    <tr><th>印</th><th>確定R</th><th>的中</th><th>的中率</th><th>回収率</th></tr>
    {mark_rows_html}
  </table>

  <h2 style="font-size:15px;margin:20px 0 8px;color:#333;border-left:4px solid #1a7a9a;padding-left:8px">AI1位指名馬 全レース</h2>
  <table>
    <tr>
      <th>レース</th><th>指名馬</th><th>騎手</th><th>オッズ</th>
      <th>印</th><th>着順</th><th>結果</th>
    </tr>
    {rows_html}
  </table>

  <div style="margin-top:16px;font-size:12px;color:#aaa;text-align:right">
    生成: {datetime.now().strftime('%Y-%m-%d %H:%M')}
  </div>
</div>
</body>
</html>'''


def update_report_index(out_dir: str):
    """docs/reports.html を再生成（新しい順）。"""
    import glob
    files = sorted(glob.glob(os.path.join(out_dir, 'report_????????.html')), reverse=True)
    rows = ''
    for f in files:
        bn = os.path.basename(f)
        ds = bn.replace('report_', '').replace('.html', '')
        label = f'{ds[:4]}/{ds[4:6]}/{ds[6:]}'
        rows += f'<li><a href="{bn}">📊 {label}</a></li>\n'

    html = f'''<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>予想結果レポート一覧</title>
<style>
  body{{font-family:sans-serif;background:#f5f5f5;padding:20px}}
  .card{{background:#fff;border-radius:10px;box-shadow:0 1px 4px rgba(0,0,0,.1);
         max-width:480px;margin:0 auto;padding:20px 24px}}
  h1{{font-size:18px;margin:0 0 16px}}
  ul{{list-style:none;padding:0;margin:0}}
  li{{border-bottom:1px solid #eee;padding:10px 0}}
  a{{color:#1a7a9a;text-decoration:none;font-size:15px}}
  .nav{{font-size:13px;margin-bottom:14px}}
  .nav a{{color:#888;text-decoration:none}}
</style>
</head>
<body>
<div class="card">
  <div class="nav"><a href="index.html">← トップ</a> | <a href="newspapers.html">予想新聞</a></div>
  <h1>📊 予想結果レポート一覧</h1>
  <ul>{rows}</ul>
</div>
</body>
</html>'''

    out = os.path.join(out_dir, 'reports.html')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'結果インデックス更新: {out}')


def main():
    if len(sys.argv) > 1:
        date_str = sys.argv[1]
    else:
        date_str = datetime.now().strftime('%Y%m%d')

    print(f'=== 結果レポート生成: {date_str} ===')
    picks = parse_newspaper(date_str)
    if not picks:
        print('予測データなし。終了します。')
        return

    confirmed = [p for p in picks if p['jyuni'] is not None]
    hits      = [p for p in confirmed if p['jyuni'] == 1]
    n_c, n_h  = len(confirmed), len(hits)
    total_ret = sum(p['odds'] for p in hits)
    roi = (total_ret - n_c) / n_c * 100 if n_c > 0 else float('nan')

    print(f'予測レース: {len(picks)}R  確定: {n_c}R  的中: {n_h}R')
    print(f'回収率: {roi:+.1f}%  ({total_ret:.1f}円 / {n_c}R)')

    html = build_html(date_str, picks)
    out_dir = os.path.join(BASE_DIR, 'docs')
    out_path = os.path.join(out_dir, f'report_{date_str}.html')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'HTML出力: {out_path}')

    update_report_index(out_dir)

    # Google Drive にコピー
    gdrive = r'G:\マイドライブ\競馬AI\予想レポート'
    if os.path.isdir(gdrive):
        import shutil
        shutil.copy(out_path, os.path.join(gdrive, f'report_{date_str}.html'))
        print(f'Gdrive出力: {os.path.join(gdrive, f"report_{date_str}.html")}')


if __name__ == '__main__':
    main()
