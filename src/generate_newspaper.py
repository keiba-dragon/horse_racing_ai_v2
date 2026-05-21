# coding: utf-8
"""
clogit 競馬新聞 HTML ジェネレータ

使い方:
  python src/generate_newspaper.py [--date 20260517] [--open]

出力:
  data/html/newspaper_YYYYMMDD.html
"""
import sys, io, os, json, pickle, argparse, re, unicodedata, urllib.request
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(BASE_DIR, 'data', 'raw', 'cache')
OUT_DIR   = os.path.join(BASE_DIR, 'data', 'html')

MARKS  = {1: '◎', 2: '○', 3: '▲', 4: '△', 5: '×'}
MARK_COLOR = {
    '◎': '#cc0000', '○': '#0055cc', '▲': '#007700',
    '△': '#e06000', '×': '#777777',
}
VENUE_ORDER = ['京', '新', '東', '阪', '中', '中京', '福', '函', '札', '小']
VENUE_FULL  = {
    '京': '京都', '新': '新潟', '東': '東京', '阪': '阪神',
    '中': '中山', '中京': '中京', '福': '福島', '函': '函館',
    '札': '札幌', '小': '小倉',
}
SURFACE_EMO = {'芝': '🌿', 'ダ': '🏜️'}
VENUE_TO_CODE = {
    '東': '05', '中': '06', '中京': '07', '京': '08', '阪': '09',
    '新': '04', '福': '03', '函': '02', '札': '01', '小': '10',
}


VENUE_FULL_TO_SHORT = {
    '東京': '東', '中山': '中', '中京': '中京', '京都': '京', '阪神': '阪',
    '新潟': '新', '福島': '福', '函館': '函', '札幌': '札', '小倉': '小',
}

def fetch_race_id_map(target_date: str) -> dict:
    """netkeiba race_list_sub から {(venue_short, r_num): race_id} を返す。失敗時は空dict。"""
    url = f'https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={target_date}'
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as res:
            html = res.read().decode('utf-8', errors='replace')
        race_map = {}
        for m in re.finditer(r'race_id=(\d{12})', html):
            rid = m.group(1)
            venue_code = rid[4:6]   # 2桁会場コード
            r_num = int(rid[10:12]) # 2桁レース番号
            # 会場コード → 短縮名
            short = next((s for s, c in VENUE_TO_CODE.items() if c == venue_code), None)
            if short:
                race_map[(short, r_num)] = rid
        return race_map
    except Exception:
        return {}


def load_data(target_date: str):
    cache_path = os.path.join(CACHE_DIR, f'{target_date}.cache.pkl')
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f'キャッシュなし: {cache_path}')
    with open(cache_path, 'rb') as f:
        cached = pickle.load(f)
    df = cached['result'].copy()
    df.columns = df.columns.astype(object)

    odds_path = os.path.join(CACHE_DIR, f'{target_date}.odds.json')
    odds_dict = {}
    if os.path.exists(odds_path):
        with open(odds_path, encoding='utf-8') as f:
            odds_dict = json.load(f)
    return df, odds_dict


def prepare(df: pd.DataFrame, odds_dict: dict) -> pd.DataFrame:
    col_surf = next((c for c in df.columns if c in ('芝・ダ', '苝・ダ')), None)
    col_dist = next((c for c in df.columns if c == '距離'), None)

    df = df.copy()
    df['_horse']   = df['馬名S'].astype(str).str.strip() if '馬名S' in df.columns else ''
    df['_umaban']  = pd.to_numeric(df.get('馬番', pd.Series(dtype=float)), errors='coerce')
    df['_surface'] = df[col_surf].astype(str).str.strip() if col_surf else ''
    df['_dist_raw']= df[col_dist].astype(str).str.strip() if col_dist else ''
    df['_dist_m']  = df['_dist_raw'].str.extract(r'(\d+)')[0].astype(float)

    race_col = next((c for c in df.columns if c in ('場 R', '場R')), None)
    if race_col:
        df['_race'] = df[race_col].astype(str)
    elif '開催' in df.columns:
        df['_race'] = df['開催'].astype(str)
    else:
        df['_race'] = ''
    df['_venue'] = df['_race'].apply(lambda x: re.match(r'^([^\d]+)', x).group(1) if re.match(r'^([^\d]+)', x) else '')
    df['_R']    = pd.to_numeric(df['Ｒ'], errors='coerce') if 'Ｒ' in df.columns else np.nan

    df['_yahoo_odds'] = df['_horse'].map(odds_dict)
    df['_mprob']      = 1.0 / df['_yahoo_odds'].clip(lower=1.0)

    # clogit_calib が保存されている場合は Yahoo オッズで clogit_score を再計算
    if 'clogit_calib' in df.columns and df['clogit_calib'].notna().any():
        factor = df['clogit_factor'].fillna(0.16)
        has_odds = df['_mprob'].notna()
        df.loc[has_odds, 'clogit_score'] = (
            df.loc[has_odds, 'clogit_calib'] - factor[has_odds] * df.loc[has_odds, '_mprob']
        )
        df['clogit_rank'] = df.groupby('_race')['clogit_score'].rank(ascending=False, method='first')
        df['_ev'] = df['clogit_calib'] - df['_mprob'] * 0.80
    else:
        df['_ev'] = df['clogit_score'] - df['_mprob'] * 0.80

    # レース内オッズ人気
    df['_pop'] = df.groupby('_race')['_yahoo_odds'].rank(method='first', ascending=True)

    # 印
    df['_rank'] = df['clogit_rank'].fillna(99).astype(int)
    df['_mark'] = df['_rank'].map(lambda r: MARKS.get(r, ''))

    # gap = rank1のcalib_prob - rank2のcalib_prob（分析で使った定義に合わせる）
    if 'clogit_calib' in df.columns:
        calib = pd.to_numeric(df['clogit_calib'], errors='coerce')
        df['_calib_tmp'] = calib
        df['_gap'] = df.groupby('_race')['_calib_tmp'].transform(
            lambda x: (x.nlargest(2).iloc[0] - x.nlargest(2).iloc[1])
                      if x.dropna().shape[0] >= 2 else 0.0
        )
        df.drop(columns=['_calib_tmp'], inplace=True)
    else:
        df['_gap'] = 0.0

    # クラス判定（買い推奨フィルタ用）
    # _cls_group 列があればそちらを優先、なければレース名から推定
    def _race_class(name):
        n = unicodedata.normalize('NFKC', str(name))
        if '新馬' in n: return '新馬'
        if '未勝利' in n: return '未勝利'
        if '障害' in n: return '障害'
        return '1勝+'
    if '_cls_group' in df.columns:
        def _from_cls_group(v):
            v = str(v)
            if '新馬' in v: return '新馬'
            if '未勝利' in v: return '未勝利'
            if '障害' in v: return '障害'
            return '1勝+'
        df['_class'] = df['_cls_group'].apply(_from_cls_group)
    elif 'レース名' in df.columns:
        df['_class'] = df['レース名'].apply(_race_class)
    else:
        df['_class'] = '1勝+'

    # キャリア列（キャッシュから取得）
    if 'キャリア' in df.columns:
        df['_career'] = pd.to_numeric(df['キャリア'], errors='coerce').fillna(0)
    else:
        df['_career'] = 0.0

    # 買い推奨フラグ: rank=1 かつ gap>=0.15 かつ ev>=0 かつ (1勝+ or 未勝利キャリア5走以上)
    df['_buy'] = (
        (df['_rank'] == 1) &
        (df['_gap'] >= 0.15) &
        (df['_ev'] >= 0.0) &
        (
            (df['_class'] == '1勝+') |
            ((df['_class'] == '未勝利') & (df['_career'] >= 5))
        )
    )

    return df


def ev_label(ev: float) -> str:
    if pd.isna(ev):
        return ''
    if ev > 0.10:
        return f'<span class="ev-high">EV{ev:+.2f}</span>'
    if ev > 0.03:
        return f'<span class="ev-mid">EV{ev:+.2f}</span>'
    if ev > 0:
        return f'<span class="ev-low">EV{ev:+.2f}</span>'
    return f'<span class="ev-neg">EV{ev:+.2f}</span>'


def render_race(race_key: str, grp: pd.DataFrame, target_date: str = '', race_id_map: dict = {}) -> str:
    grp = grp.sort_values('_rank')
    first = grp.iloc[0]
    venue   = first['_venue']
    r_num   = int(first['_R']) if pd.notna(first['_R']) else '?'
    surf    = first['_surface']
    dist    = int(first['_dist_m']) if pd.notna(first['_dist_m']) else '?'
    n       = len(grp)
    emo     = SURFACE_EMO.get(surf, '')

    # netkeiba レースURL（race_id_mapで個別レースに、なければ日付一覧にフォールバック）
    rid = race_id_map.get((venue, r_num)) if isinstance(r_num, int) else None
    if rid:
        race_url = f"https://race.netkeiba.com/race/result.html?race_id={rid}"
    elif target_date:
        race_url = f"https://race.netkeiba.com/top/race_list.html?kaisai_date={target_date}"
    else:
        race_url = ''

    top1 = grp[grp['_rank'] == 1].iloc[0] if (grp['_rank'] == 1).any() else None
    top1_ev  = top1['_ev']  if top1 is not None else np.nan
    top1_gap = top1['_gap'] if top1 is not None else np.nan
    top1_buy = bool(top1['_buy']) if top1 is not None else False

    # ヘッダーバッジ
    header_badges = ''
    if top1_buy:
        gap_s = f"{top1_gap:.2f}" if pd.notna(top1_gap) else '-'
        header_badges += f'<span class="badge-buy">★買い</span> <span class="gap-tag">gap {gap_s}</span>'
    elif pd.notna(top1_ev) and top1_ev > 0.05:
        header_badges += '<span class="badge-ev">EV+</span>'

    rows_html = []
    for _, r in grp.iterrows():
        mark   = r['_mark']
        mc     = MARK_COLOR.get(mark, '#333')
        horse  = r['_horse']
        umaban = int(r['_umaban']) if pd.notna(r['_umaban']) else '-'
        odds   = r['_yahoo_odds']
        pop    = int(r['_pop']) if pd.notna(r['_pop']) else '-'
        ev     = r['_ev']
        calib  = r['clogit_calib'] if 'clogit_calib' in r.index else np.nan
        buy    = bool(r['_buy'])

        odds_s  = f"{odds:.1f}" if pd.notna(odds) else '-'
        calib_s = f"{calib*100:.1f}%" if pd.notna(calib) else '-'
        ev_s    = ev_label(ev)

        if buy:
            row_cls = ' class="row-buy"'
        elif mark == '◎':
            row_cls = ' class="row-top"'
        elif pd.notna(ev) and ev > 0.05:
            row_cls = ' class="row-ev"'
        else:
            row_cls = ''

        # netkeiba 馬名検索リンク
        import urllib.parse
        horse_url = f"https://db.netkeiba.com/?pid=horse_list&word={urllib.parse.quote(horse)}"
        horse_html = f'<a href="{horse_url}" target="_blank" class="horse-link">{horse}</a>'

        # NaNバッジ
        nan_count    = r.get('_nan_count', np.nan)
        nan_total    = r.get('_nan_total', np.nan)
        nan_features = str(r.get('_nan_features', '') or '')
        if pd.notna(nan_count) and pd.notna(nan_total) and nan_total > 0:
            nan_pct = nan_count / nan_total
            # tooltip: 件数 + 項目名（改行区切り、最大20件）
            feat_list = [f for f in nan_features.split(',') if f]
            feat_preview = '&#10;'.join(feat_list[:20])
            if len(feat_list) > 20:
                feat_preview += f'&#10;...他{len(feat_list)-20}件'
            tip = f"{int(nan_count)}/{int(nan_total)}件 NaN&#10;{feat_preview}"
            if nan_pct >= 0.30:
                nan_html = f'<span class="nan-high" title="{tip}">NaN{nan_pct:.0%}</span>'
            elif nan_pct >= 0.10:
                nan_html = f'<span class="nan-mid" title="{tip}">NaN{nan_pct:.0%}</span>'
            else:
                nan_html = ''
        else:
            nan_html = ''

        rows_html.append(f"""
        <tr{row_cls}>
          <td class="mark" style="color:{mc}">{mark}</td>
          <td class="umaban">{umaban}</td>
          <td class="horse">{horse_html} {nan_html}</td>
          <td class="pop">{pop}</td>
          <td class="odds">{odds_s}</td>
          <td class="calib">{calib_s}</td>
          <td class="ev-col">{ev_s}</td>
        </tr>""")

    race_label_html = (f'<a href="{race_url}" target="_blank" class="race-link">{r_num}R</a>'
                       if race_url else f'{r_num}R')

    return f"""
  <div class="race-card">
    <div class="race-header">
      <span class="race-label">{race_label_html}</span>
      <span class="course">{emo}{surf}{dist}m</span>
      <span class="n-horses">{n}頭</span>
      {header_badges}
    </div>
    <table class="race-table">
      <thead>
        <tr>
          <th>印</th><th>馬番</th><th>馬名</th>
          <th>人気</th><th>オッズ</th><th>勝率</th><th>EV</th>
        </tr>
      </thead>
      <tbody>{''.join(rows_html)}
      </tbody>
    </table>
  </div>"""


def generate_html(df: pd.DataFrame, target_date: str) -> str:
    dt = datetime.strptime(target_date, '%Y%m%d')
    date_str = f"{dt.year}.{dt.month:02d}.{dt.day:02d}"
    weekday  = ['月', '火', '水', '木', '金', '土', '日'][dt.weekday()]

    venues_in_data = df['_venue'].unique()
    venue_order    = [v for v in VENUE_ORDER if v in venues_in_data]

    n_races  = df['_race'].nunique()
    n_horses = len(df)
    n_buy    = int(df['_buy'].sum())

    top1 = df[df['_rank'] == 1]

    race_id_map = fetch_race_id_map(target_date)
    print(f'race_id_map: {len(race_id_map)}件取得')

    venue_cols = []
    for venue in venue_order:
        vdf = df[df['_venue'] == venue].copy()
        races = sorted(vdf['_race'].unique(), key=lambda x: int(re.search(r'\d+', x).group()))
        cards = ''.join(render_race(r, vdf[vdf['_race'] == r], target_date, race_id_map) for r in races)
        venue_full = VENUE_FULL.get(venue, venue)
        venue_cols.append(f"""
    <div class="venue-col">
      <div class="venue-title">{venue_full}</div>
      {cards}
    </div>""")

    cols_html = '\n'.join(venue_cols)

    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>clogit競馬新聞 {date_str}</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
  font-family: 'Hiragino Kaku Gothic Pro', 'Meiryo', sans-serif;
  font-size: 11px;
  background: #f5f0e8;
  color: #1a1a1a;
}}
/* ── ヘッダー ── */
.page-header {{
  background: #1a1a2e;
  color: #fff;
  padding: 10px 16px;
  display: flex;
  align-items: baseline;
  gap: 16px;
  flex-wrap: wrap;
}}
.page-title {{
  font-size: 22px;
  font-weight: bold;
  letter-spacing: 2px;
  color: #f0d060;
}}
.page-sub {{
  font-size: 12px;
  color: #aaa;
}}
.page-meta {{
  margin-left: auto;
  font-size: 11px;
  color: #ccc;
}}
/* 凡例バー */
.legend-bar {{
  background: #2d2d44;
  color: #eee;
  padding: 5px 16px;
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
  font-size: 11px;
}}
.legend-bar span {{ white-space: nowrap; }}
.leg-mark {{ font-weight: bold; }}
/* ── 会場グリッド ── */
.venue-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
  gap: 12px;
  padding: 12px;
  max-width: 1400px;
  margin: 0 auto;
}}
.venue-col {{ display: flex; flex-direction: column; gap: 8px; }}
.venue-title {{
  background: #1a1a2e;
  color: #f0d060;
  font-size: 15px;
  font-weight: bold;
  padding: 6px 12px;
  border-radius: 4px 4px 0 0;
  letter-spacing: 4px;
}}
/* ── レースカード ── */
.race-card {{
  background: #fff;
  border: 1px solid #c8b89a;
  border-radius: 4px;
  overflow: hidden;
  box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}}
.race-header {{
  background: #3a3a5c;
  color: #fff;
  padding: 5px 10px;
  display: flex;
  align-items: center;
  gap: 8px;
}}
.race-label {{
  font-size: 15px;
  font-weight: bold;
  color: #f0d060;
  min-width: 28px;
}}
.course {{ font-size: 12px; color: #cce0ff; }}
.n-horses {{ font-size: 11px; color: #aaa; margin-left: auto; }}
.badge-ev {{
  background: #e8a000;
  color: #fff;
  font-size: 10px;
  font-weight: bold;
  padding: 1px 5px;
  border-radius: 3px;
}}
.badge-buy {{
  background: #cc0000;
  color: #fff;
  font-size: 11px;
  font-weight: bold;
  padding: 2px 7px;
  border-radius: 3px;
  letter-spacing: 1px;
}}
.gap-tag {{
  font-size: 10px;
  color: #ffcccc;
}}
/* ── テーブル ── */
.race-table {{
  width: 100%;
  border-collapse: collapse;
}}
.race-table th {{
  background: #eae4d8;
  color: #555;
  font-size: 10px;
  padding: 3px 4px;
  text-align: center;
  border-bottom: 1px solid #c8b89a;
}}
.race-table td {{
  padding: 3px 4px;
  border-bottom: 1px solid #ede8df;
  vertical-align: middle;
}}
.race-table tr:last-child td {{ border-bottom: none; }}
/* 行ハイライト */
.row-buy {{ background: #fff0f0; }}
.row-top {{ background: #fff8f0; }}
.row-ev  {{ background: #f0fff4; }}
.race-table tr:hover {{ background: #f5f0e8; }}
/* 各列 */
.mark   {{ text-align: center; font-size: 14px; font-weight: bold; width: 22px; }}
.umaban {{ text-align: center; color: #666; width: 24px; }}
.horse  {{ font-weight: 500; max-width: 110px; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }}
.pop    {{ text-align: right; color: #555; width: 28px; }}
.odds   {{ text-align: right; font-weight: bold; color: #1a3a6e; width: 38px; }}
.calib  {{ text-align: right; color: #555; width: 44px; font-size: 10px; }}
.ev-col {{ text-align: right; width: 60px; font-size: 10px; }}
/* EV バッジ */
.ev-high {{ color: #fff; background: #cc0000; border-radius: 2px; padding: 0 3px; font-weight: bold; }}
.ev-mid  {{ color: #fff; background: #e08000; border-radius: 2px; padding: 0 3px; font-weight: bold; }}
.ev-low  {{ color: #006000; font-weight: bold; }}
.ev-neg  {{ color: #aaa; }}
/* フッター */
.page-footer {{
  text-align: center;
  padding: 12px;
  color: #888;
  font-size: 10px;
  border-top: 1px solid #c8b89a;
  margin-top: 8px;
}}
/* NaNバッジ */
.nan-high {{
  background: #cc4400;
  color: #fff;
  font-size: 9px;
  padding: 0 3px;
  border-radius: 2px;
  opacity: 0.85;
}}
.nan-mid {{
  background: #888;
  color: #fff;
  font-size: 9px;
  padding: 0 3px;
  border-radius: 2px;
  opacity: 0.75;
}}
/* リンク */
a.race-link {{
  color: #f0d060;
  text-decoration: none;
}}
a.race-link:hover {{ text-decoration: underline; }}
a.horse-link {{
  color: inherit;
  text-decoration: none;
}}
a.horse-link:hover {{
  text-decoration: underline;
  color: #0055cc;
}}
@media (max-width: 700px) {{
  .venue-grid {{ grid-template-columns: 1fr; }}
  .horse {{ max-width: 80px; }}
}}
</style>
</head>
<body>

<div class="page-header">
  <div class="page-title">clogit 競馬新聞</div>
  <div class="page-sub">{date_str}（{weekday}）</div>
  <div class="page-meta">
    {'・'.join(VENUE_FULL.get(v,v) for v in venue_order)} &nbsp;|&nbsp;
    {n_races}レース {n_horses}頭 &nbsp;|&nbsp;
    ★買い推奨: {n_buy}レース
  </div>
</div>

<div class="legend-bar">
  <span><span class="leg-mark" style="color:#f0d060">◎</span> 1位</span>
  <span><span class="leg-mark" style="color:#88ccff">○</span> 2位</span>
  <span><span class="leg-mark" style="color:#88ff88">▲</span> 3位</span>
  <span><span class="leg-mark" style="color:#ffaa55">△</span> 4位</span>
  <span><span class="leg-mark" style="color:#aaa">×</span> 5位</span>
  <span style="margin-left:8px">|</span>
  <span><span style="background:#cc0000;color:#fff;padding:0 4px;border-radius:2px">EV+0.10〜</span> 強推奨</span>
  <span><span style="background:#e08000;color:#fff;padding:0 4px;border-radius:2px">EV+0.03〜</span> 推奨</span>
  <span><span style="background:#cc0000;color:#fff;padding:0 4px;border-radius:2px;font-weight:bold">★買い</span> gap≥0.15 + EV≥0 + 1勝以上（未勝利はキャリア5走以上）</span>
  <span style="color:#aaa">人気・オッズは Yahoo競馬 取得</span>
</div>

<div class="venue-grid">
{cols_html}
</div>

<div class="page-footer">
  clogit model &nbsp;|&nbsp; surface-split conditional logit + isotonic calibration &nbsp;|&nbsp;
  生成: {datetime.now().strftime('%Y-%m-%d %H:%M')}
</div>

</body>
</html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', default=None, help='YYYYMMDD (省略時: 今日)')
    ap.add_argument('--open', action='store_true', help='生成後ブラウザで開く')
    args = ap.parse_args()

    target_date = args.date or datetime.now().strftime('%Y%m%d')
    print(f'生成中: {target_date}')

    df, odds_dict = load_data(target_date)
    df = prepare(df, odds_dict)

    html = generate_html(df, target_date)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f'newspaper_{target_date}.html')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'出力: {out_path}')

    if args.open:
        import subprocess
        subprocess.Popen(['start', '', out_path], shell=True)


if __name__ == '__main__':
    main()
