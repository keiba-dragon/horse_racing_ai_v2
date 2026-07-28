# coding: utf-8
"""
make_monthly_report.py — 月次予想結果サマリー生成

usage:
    python src/make_monthly_report.py 202607        # 2026年7月分
    python src/make_monthly_report.py 202607 20260705 20260712   # 対象日付を明示指定

動作:
  - 指定月の newspaper_YYYYMMDD.html を全て解析し、AI1位馬の的中・回収を月合算
  - docs/report_YYYYMM_monthly.html として保存
  - docs/reports.html インデックスに月次レポートも追記
"""
import sys, os, re, glob
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from make_result_report import parse_newspaper  # noqa: E402


def find_month_dates(month_str: str) -> list:
    """month_str = 'YYYYMM' -> docs/newspaper_YYYYMMDD.html から日付一覧を抽出"""
    pattern = os.path.join(BASE_DIR, 'docs', f'newspaper_{month_str}*.html')
    dates = []
    for path in sorted(glob.glob(pattern)):
        bn = os.path.basename(path)
        m = re.match(r'newspaper_(\d{8})\.html$', bn)
        if m:
            dates.append(m.group(1))
    return dates


def roi_color(v):
    if v >= 0:
        return '#1a7a1a'
    elif v >= -20:
        return '#b36000'
    else:
        return '#c0392b'


def build_html(month_str: str, per_date: list) -> str:
    if month_str.endswith('00'):
        ym = f'{month_str[:4]}年 年初来'
    else:
        ym = f'{month_str[:4]}年{int(month_str[4:6])}月'

    total_c = sum(d['n_c'] for d in per_date)
    total_h = sum(d['n_h'] for d in per_date)
    total_ret = sum(d['total_return'] for d in per_date)
    roi = (total_ret - total_c) / total_c * 100 if total_c > 0 else float('nan')
    win_rate = total_h / total_c * 100 if total_c > 0 else 0.0

    date_rows = ''
    for d in per_date:
        ds = d['date']
        label = f'{ds[:4]}/{ds[4:6]}/{ds[6:]}'
        d_roi = d['roi']
        d_roi_s = f'{d_roi:+.1f}%' if d['n_c'] > 0 else '—'
        d_roi_c = roi_color(d_roi) if d['n_c'] > 0 else '#888'
        date_rows += f'''
        <tr>
          <td><a href="newspaper_{ds}.html">{label}</a></td>
          <td style="text-align:center">{d['n_c']}R</td>
          <td style="text-align:center">{d['n_h']}R</td>
          <td style="text-align:center">{d['n_h'] / d['n_c'] * 100 if d['n_c'] else 0:.1f}%</td>
          <td style="text-align:right;color:{d_roi_c};font-weight:bold">{d_roi_s}</td>
          <td><a href="report_{ds}.html">日次詳細</a></td>
        </tr>'''

    # 全ピックの一覧（着順つき）
    all_picks = []
    for d in per_date:
        for p in d['picks']:
            all_picks.append({**p, 'date': d['date']})

    MARK_COLOR = {'◎': '#c0392b', '○': '#e67e22', '▲': '#27ae60', '-': '#aaa'}

    def mark_stats(subset):
        conf = [p for p in subset if p['jyuni'] is not None]
        h    = [p for p in conf if p['jyuni'] == 1]
        n_c2, n_h2 = len(conf), len(h)
        ret2 = sum(p['odds'] for p in h)
        roi2 = (ret2 - n_c2) / n_c2 * 100 if n_c2 > 0 else float('nan')
        return n_c2, n_h2, roi2

    mark_rows = ''
    for mk in ('◎', '○', '▲'):
        subset = [p for p in all_picks if p.get('mark') == mk]
        if not subset:
            continue
        n_c2, n_h2, roi2 = mark_stats(subset)
        roi2_s = f'{roi2:+.1f}%' if n_c2 > 0 else '—'
        roi2_c = roi_color(roi2) if n_c2 > 0 else '#888'
        mark_rows += f'''
        <tr>
          <td style="color:{MARK_COLOR[mk]};font-weight:bold;font-size:16px">{mk}</td>
          <td style="text-align:center">{n_c2}R</td>
          <td style="text-align:center">{n_h2}R</td>
          <td style="text-align:center">{n_h2/n_c2*100 if n_c2 else 0:.1f}%</td>
          <td style="text-align:right;color:{roi2_c};font-weight:bold">{roi2_s}</td>
        </tr>'''

    pick_rows = ''
    for p in all_picks:
        if p['jyuni'] is None:
            result_mark, bg = '<span style="color:#888">未確定</span>', ''
        elif p['jyuni'] == 1:
            result_mark, bg = '◎ <b>的中！</b>', ' style="background:#fff9e6"'
        else:
            result_mark, bg = f'{p["jyuni"]}着', ''
        mk = p.get('mark', '-')
        mk_col = MARK_COLOR.get(mk, '#aaa')
        race_short = re.sub(r'^.*?(\d+R)', r'\1', p['race']) if p['race'] else '—'
        ds = p['date']
        label = f'{ds[4:6]}/{ds[6:]}'
        pick_rows += f'''
        <tr{bg}>
          <td style="color:#555;font-size:13px">{label}</td>
          <td style="color:#555;font-size:13px">{race_short}</td>
          <td style="font-weight:bold">{p['horse']}</td>
          <td style="text-align:right">{p['odds']:.1f}倍</td>
          <td style="text-align:center;color:{mk_col};font-weight:bold">{mk}</td>
          <td>{result_mark}</td>
        </tr>'''

    roi_c = roi_color(roi) if total_c > 0 else '#888'
    roi_s = f'{roi:+.1f}%' if total_c > 0 else '—'

    return f'''<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{ym} 予想結果サマリー</title>
<style>
  body {{ font-family: 'Helvetica Neue', Arial, sans-serif; background:#f5f5f5; margin:0; padding:16px; }}
  .card {{ background:#fff; border-radius:10px; box-shadow:0 1px 4px rgba(0,0,0,.12);
           max-width:820px; margin:0 auto; padding:20px 24px; }}
  h1 {{ font-size:20px; margin:0 0 4px; color:#222; }}
  h2 {{ font-size:15px; margin:24px 0 8px; color:#333; border-left:4px solid #1a7a9a; padding-left:8px; }}
  .subtitle {{ color:#888; font-size:13px; margin-bottom:20px; }}
  .summary {{ display:flex; gap:20px; flex-wrap:wrap; margin-bottom:20px; }}
  .stat {{ background:#f8f8f8; border-radius:8px; padding:12px 18px; min-width:120px; }}
  .stat .label {{ font-size:12px; color:#888; margin-bottom:4px; }}
  .stat .value {{ font-size:24px; font-weight:bold; }}
  table {{ width:100%; border-collapse:collapse; font-size:14px; margin-bottom:8px; }}
  th {{ background:#f0f0f0; padding:8px 10px; text-align:left; font-size:12px; color:#555;
        border-bottom:2px solid #ddd; }}
  td {{ padding:9px 10px; border-bottom:1px solid #eee; vertical-align:middle; }}
  tr:hover td {{ background:#fafafa; }}
  .nav {{ display:flex; gap:10px; margin-bottom:16px; }}
  .nav a {{ color:#1a7a9a; text-decoration:none; font-size:13px; }}
  .nav a:hover {{ text-decoration:underline; }}
  .note {{ color:#aaa; font-size:12px; margin-top:8px; }}
</style>
</head>
<body>
<div class="card">
  <div class="nav">
    <a href="index.html">← トップ</a>
    <a href="newspapers.html">予想新聞一覧</a>
    <a href="reports.html">結果レポート一覧</a>
  </div>
  <h1>{ym} 予想結果サマリー</h1>
  <div class="subtitle">AI1位指名馬（的中率最大化モデル）の単勝成績 — 集計対象 {len(per_date)}日分</div>

  <div class="summary">
    <div class="stat">
      <div class="label">的中率</div>
      <div class="value">{win_rate:.1f}%</div>
    </div>
    <div class="stat">
      <div class="label">的中 / 確定R</div>
      <div class="value">{total_h} / {total_c}</div>
    </div>
    <div class="stat">
      <div class="label">単勝回収率</div>
      <div class="value" style="color:{roi_c}">{roi_s}</div>
    </div>
    <div class="stat">
      <div class="label">総投資</div>
      <div class="value">{total_c}R</div>
    </div>
  </div>

  <h2>日別内訳</h2>
  <table>
    <tr><th>日付</th><th>確定R</th><th>的中</th><th>的中率</th><th>回収率</th><th>詳細</th></tr>
    {date_rows}
  </table>

  <h2>印別成績</h2>
  <table>
    <tr><th>印</th><th>確定R</th><th>的中</th><th>的中率</th><th>回収率</th></tr>
    {mark_rows}
  </table>

  <h2>全指名馬一覧</h2>
  <table>
    <tr><th>日付</th><th>レース</th><th>指名馬</th><th>オッズ</th><th>印</th><th>結果</th></tr>
    {pick_rows}
  </table>

  <div class="note">
    集計対象日: {', '.join(f"{d['date'][:4]}/{d['date'][4:6]}/{d['date'][6:]}" for d in per_date)}<br>
    生成: {datetime.now().strftime('%Y-%m-%d %H:%M')}
  </div>
</div>
</body>
</html>'''


def update_report_index(out_dir: str):
    """docs/reports.html を再生成（日次+月次、新しい順）。"""
    import glob as _glob
    daily = sorted(_glob.glob(os.path.join(out_dir, 'report_????????.html')), reverse=True)
    monthly = sorted(_glob.glob(os.path.join(out_dir, 'report_??????_monthly.html')), reverse=True)

    rows = ''
    for f in monthly:
        bn = os.path.basename(f)
        ds = bn.replace('report_', '').replace('_monthly.html', '')
        label = f'{ds[:4]}年 年初来' if ds.endswith('00') else f'{ds[:4]}/{ds[4:6]} 月次'
        rows += f'<li><a href="{bn}">📆 {label}</a></li>\n'
    for f in daily:
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
    if len(sys.argv) < 2:
        print('usage: python src/make_monthly_report.py YYYYMM [date1 date2 ...]')
        return
    month_str = sys.argv[1]
    if len(sys.argv) > 2:
        dates = sys.argv[2:]
    else:
        dates = find_month_dates(month_str)

    if not dates:
        print(f'{month_str} の newspaper が見つかりません。')
        return

    print(f'=== 月次結果レポート生成: {month_str} (対象: {", ".join(dates)}) ===')

    per_date = []
    for ds in dates:
        picks = parse_newspaper(ds)
        if not picks:
            print(f'[WARN] {ds}: 予測データなし。スキップ')
            continue
        confirmed = [p for p in picks if p['jyuni'] is not None]
        hits = [p for p in confirmed if p['jyuni'] == 1]
        n_c, n_h = len(confirmed), len(hits)
        total_return = sum(p['odds'] for p in hits)
        roi = (total_return - n_c) / n_c * 100 if n_c > 0 else float('nan')
        print(f'  {ds}: 確定{n_c}R 的中{n_h}R 回収率{roi:+.1f}%')
        per_date.append({
            'date': ds, 'picks': picks, 'n_c': n_c, 'n_h': n_h,
            'total_return': total_return, 'roi': roi,
        })

    if not per_date:
        print('集計可能なデータがありません。')
        return

    total_c = sum(d['n_c'] for d in per_date)
    total_h = sum(d['n_h'] for d in per_date)
    total_ret = sum(d['total_return'] for d in per_date)
    roi = (total_ret - total_c) / total_c * 100 if total_c > 0 else float('nan')
    print(f'=== 月合計: 確定{total_c}R 的中{total_h}R 回収率{roi:+.1f}% ({total_ret:.1f}円/{total_c}R) ===')

    html = build_html(month_str, per_date)
    out_dir = os.path.join(BASE_DIR, 'docs')
    out_path = os.path.join(out_dir, f'report_{month_str}_monthly.html')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'HTML出力: {out_path}')

    update_report_index(out_dir)

    gdrive = r'G:\マイドライブ\競馬AI\予想レポート'
    if os.path.isdir(gdrive):
        import shutil
        shutil.copy(out_path, os.path.join(gdrive, f'report_{month_str}_monthly.html'))
        print(f'Gdrive出力: {os.path.join(gdrive, f"report_{month_str}_monthly.html")}')


if __name__ == '__main__':
    main()
