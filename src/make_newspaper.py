# coding: utf-8
"""
make_newspaper.py v2 — 競馬AI 詳細新聞生成

新設計:
  - 買い目サマリーを冒頭に大きく表示
  - 各レース: 特徴量ヒートマップ（レース内パーセンタイル色分け）+ NaN一覧
"""
import os, sys, re, pickle, argparse
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SEG_LABEL = {
    'ダ長': 'ダ長（ダ>1400m）',
    'ダ短': 'ダ短（ダ≤1400m）',
    '芝短': '芝短（芝≤1400m）',
    '芝中': '芝中（芝1401-2000m）',
    '芝長': '芝長（芝>2000m）',
}
SEG_COLOR = {
    'ダ長': '#2980b9',
    'ダ短': '#1a6fa0',
    '芝短': '#27ae60',
    '芝中': '#16a085',
    '芝長': '#8e44ad',
}

BABA_MAP_INV = {0: '良', 1: '稍重', 2: '重', 3: '不良',
                0.0: '良', 1.0: '稍重', 2.0: '重', 3.0: '不良'}
SEX_MAP = {0: '牡', 1: '牝', 2: 'セン', 0.0: '牡', 1.0: '牝', 2.0: 'セン'}

V_SHORT = {'東京': '東', '中山': '中', '阪神': '阪', '京都': '京',
           '中京': '名', '新潟': '新', '函館': '函', '小倉': '小',
           '札幌': '札', '福島': '福'}
V_FULL  = {'東京': '東京', '中山': '中山', '阪神': '阪神', '京都': '京都',
           '中京': '中京', '新潟': '新潟', '函館': '函館', '小倉': '小倉',
           '札幌': '札幌', '福島': '福島'}


def get_seg_key(surf, dist_m):
    if pd.isna(dist_m):
        return None
    surf = str(surf).strip()
    if surf == '芝':
        if dist_m <= 1400:  return '芝短'
        elif dist_m <= 2000: return '芝中'
        else:               return '芝長'
    elif surf == 'ダ':
        return 'ダ短' if dist_m <= 1400 else 'ダ長'
    return None


def fmt_val(col, val):
    if pd.isna(val):
        return None
    try:
        if '馬場状態' in col and '_isnan' not in col:
            return BABA_MAP_INV.get(int(float(val)), str(val))
        if col == '性別_num':
            return SEX_MAP.get(val, str(val))
        if col in ('ブリンカー変更', '芝ダ転向') or col.endswith('_isnan'):
            return '有' if val == 1 else '-'
        if '勝率' in col:
            return f'{float(val):.1%}'
        if '上り3F' in col and '_isnan' not in col:
            return f'{float(val):.1f}'
        if 'クラス差' in col or '距離変化' in col or col == '間隔':
            return f'{int(round(float(val)))}'
        if col in ('馬番', '斤量', '馬体重'):
            return f'{float(val):.0f}'
        if isinstance(val, float):
            return f'{val:.3f}'
    except Exception:
        pass
    return str(val)


def short_feat(f):
    return (f.replace('コース枠_r200_', 'C枠')
             .replace('馬距離_', '馬距離')
             .replace('種牡馬_', '種牡馬')
             .replace('1走前_', '前走')
             .replace('2走前_', '2前')
             .replace('3走前_', '3前')
             .replace('近5走_', '5走')
             .replace('距離変化_前走', '距離変化')
             .replace('ブリンカー変更', 'BK')
             .replace('_isnan', '[N?]'))


def percentile_color(pct):
    """0-1 のパーセンタイル → CSS RGB (青=低、白=中、橙=高)"""
    if pd.isna(pct):
        return '#f5f5f5'
    if pct < 0.5:
        t = pct * 2
        r, g, b = int(200 + 55 * t), int(210 + 45 * t), 255
    else:
        t = (pct - 0.5) * 2
        r, g, b = 255, int(255 - 105 * t), int(255 - 155 * t)
    return f'rgb({r},{g},{b})'


def make_newspaper(date_str=None):
    # ── キャッシュ探索 ────────────────────────────────────────────
    cache_dir = os.path.join(BASE_DIR, 'data', 'raw', 'cache')
    all_caches = sorted(
        [f for f in os.listdir(cache_dir) if f.endswith('.cache.pkl')],
        key=lambda f: os.path.getmtime(os.path.join(cache_dir, f)),
        reverse=True
    )
    api_caches = [f for f in all_caches if '_api.cache.pkl' in f]
    caches = api_caches if api_caches else all_caches
    if not caches:
        print(f'キャッシュが見つかりません: {cache_dir}')
        return

    cache_file = os.path.join(cache_dir, caches[0])
    print(f'キャッシュ読み込み: {caches[0]}')
    with open(cache_file, 'rb') as f:
        cache = pickle.load(f)

    result   = cache['result']
    card_df  = cache.get('card_df', pd.DataFrame())
    tgt_date = cache.get('target_date', '??')

    # ── モデル読み込み（的中率最大化モデル）────────────────────────
    model_path = os.path.join(BASE_DIR, 'models', 'accuracy_model.pkl')
    acc_model  = pickle.load(open(model_path, 'rb'))
    seg_feats  = {k: v['feat_cols'] for k, v in acc_model.items()}

    # ── カード情報（騎手・オッズ）────────────────────────────────
    card_map = {}
    if not card_df.empty and '馬名S' in card_df.columns:
        for _, cr in card_df.drop_duplicates('馬名S').iterrows():
            card_map[cr['馬名S']] = {
                '騎手':     cr.get('騎手', cr.get('dc_騎手', '')),
                '単勝オッズ': cr.get('単勝オッズ', ''),
            }

    # ── グループ化 ────────────────────────────────────────────────
    race_keys = [c for c in ['開催', 'Ｒ', 'レース名', '距離', '芝・ダ'] if c in result.columns]
    result_reset = result.reset_index(drop=True)
    for k in race_keys:
        result_reset[k] = result_reset[k].astype(str)
    groups = result_reset.groupby(race_keys, sort=False)

    # グループ情報を収集
    race_data = []
    for gk, grp in groups:
        grp = grp.copy()
        if isinstance(gk, tuple):
            kaikai    = str(gk[0]) if len(gk) > 0 else ''
            r_num     = str(gk[1]) if len(gk) > 1 else ''
            race_name = str(gk[2]) if len(gk) > 2 else ''
            kyori_raw = str(gk[3]) if len(gk) > 3 else ''
            shiba_da  = str(gk[4]) if len(gk) > 4 else ''
        else:
            kaikai = str(gk); r_num = race_name = kyori_raw = shiba_da = ''

        m = re.search(r'(\d+)', kyori_raw)
        dist_m = pd.to_numeric(m.group() if m else '', errors='coerce')
        surf   = str(shiba_da).strip() if shiba_da else str(kyori_raw)[:1]
        seg_key = get_seg_key(surf, dist_m)
        feats   = seg_feats.get(seg_key, []) if seg_key else []

        # accuracy_model でスコア計算してランク付け
        if seg_key and seg_key in acc_model:
            art = acc_model[seg_key]
            feat_cols = art['feat_cols']
            scaler    = art['scaler']
            coef      = art['coef']
            rows = []
            for _, row in grp.iterrows():
                fv = []
                for f in feat_cols:
                    if f.endswith('_isnan'):
                        base_f = f[:-6]
                        fv.append(1.0 if pd.isna(row.get(base_f)) else 0.0)
                    else:
                        v = row.get(f, np.nan)
                        try:
                            fv.append(float(v) if not pd.isna(v) else 0.0)
                        except (ValueError, TypeError):
                            fv.append(0.0)
                rows.append(fv)
            X = np.array(rows, dtype=float)
            try:
                scores = scaler.transform(X) @ coef
            except Exception:
                scores = np.zeros(len(grp))
            grp = grp.copy()
            grp['_acc_score'] = scores
            grp['_sort_rank'] = grp['_acc_score'].rank(ascending=False, method='first')
        else:
            grp['_acc_score'] = np.nan
            grp['_sort_rank'] = pd.to_numeric(
                grp['clogit_rank'] if 'clogit_rank' in grp.columns
                else pd.Series(np.nan, index=grp.index), errors='coerce'
            )
        grp = grp.sort_values('_sort_rank', na_position='last')

        race_data.append(dict(
            grp=grp, kaikai=kaikai, r_num=r_num, race_name=race_name,
            kyori_raw=kyori_raw, shiba_da=shiba_da, dist_m=dist_m,
            surf=surf, seg_key=seg_key, feats=feats
        ))

    # ── 日付表示 ────────────────────────────────────────────────
    d_str = str(tgt_date)
    date_disp = f'20{d_str[:2]}/{d_str[2:4]}/{d_str[4:6]}' if len(d_str) == 6 else str(tgt_date)

    # ── CSS ──────────────────────────────────────────────────────
    css = """<style>
  @page { size: A3 landscape; margin: 5mm; }
  * { box-sizing: border-box; }
  body { font-family: 'Yu Gothic', 'Hiragino Sans', 'Meiryo', sans-serif;
         font-size: 11px; background: #eef1f5; margin: 0; padding: 8px; color: #222; }

  h1.page-title { font-size: 18px; margin: 0 0 10px; padding: 10px 18px;
                  background: #1a252f; color: white; border-radius: 8px;
                  display: flex; align-items: center; gap: 10px; }
  h1.page-title .subtitle { font-size: 11px; color: #aaa; font-weight: normal; }

  /* ── 買い目セクション ─────────────────────────── */
  .buy-section { background: white; border-radius: 10px; padding: 14px 18px;
                 margin-bottom: 14px; box-shadow: 0 2px 6px rgba(0,0,0,0.1); }
  .section-title { font-size: 14px; font-weight: bold; margin: 0 0 10px;
                   padding-bottom: 5px; border-bottom: 2px solid currentColor; }
  .section-title.buy  { color: #c0392b; border-color: #c0392b; }
  .section-title.watch { color: #e67e22; border-color: #e67e22; margin-top: 14px; }
  .buy-grid { display: flex; flex-wrap: wrap; gap: 10px; }
  .buy-card { border-radius: 10px; padding: 10px 14px; min-width: 170px;
              position: relative; }
  .buy-card.confirmed { background: #fde8e8; border: 2px solid #c0392b; }
  .buy-card.watch-card { background: #fef9e7; border: 2px solid #e67e22; }
  .card-race { font-size: 9px; color: #777; margin-bottom: 3px; }
  .card-horse { font-size: 15px; font-weight: bold; color: #1a252f; margin-bottom: 2px; }
  .card-meta { font-size: 9px; color: #666; }
  .badge-buy { display: inline-block; background: #c0392b; color: white;
               font-size: 10px; font-weight: bold; padding: 2px 9px;
               border-radius: 10px; margin-top: 5px; }
  .badge-watch { display: inline-block; background: #e67e22; color: white;
                 font-size: 9px; padding: 2px 9px; border-radius: 10px; margin-top: 5px; }
  .seg-chip { color: white; font-size: 8px; padding: 1px 6px;
              border-radius: 4px; vertical-align: middle; }
  .no-signal { color: #aaa; font-style: italic; font-size: 11px; }

  /* ── レースブロック ───────────────────────────── */
  .race-block { background: white; border-radius: 8px; margin-bottom: 10px;
                box-shadow: 0 1px 4px rgba(0,0,0,0.09); overflow: hidden; }
  .race-header { display: flex; align-items: center; gap: 8px; padding: 8px 14px;
                 background: #f7f9fb; border-left: 6px solid #888; flex-wrap: wrap; }
  .race-venue { font-size: 15px; font-weight: bold; color: #222; }
  .race-rnum  { font-size: 13px; font-weight: bold; color: #555; }
  .race-name  { font-size: 13px; font-weight: bold; flex: 1; color: #1a252f; }
  .race-seg   { color: white; font-size: 9px; padding: 2px 9px; border-radius: 12px; }
  .race-dist  { font-size: 10px; color: #888; }
  .n-horses   { font-size: 9px; color: #aaa; }

  /* NaN Alert */
  .nan-alert { padding: 5px 14px; background: #fff8f8;
               border-top: 1px solid #fcc; font-size: 9px; }
  .nan-chip { display: inline-block; margin: 1px 3px; padding: 1px 6px;
              border-radius: 4px; font-weight: bold; }
  .nan-hi  { background: #c0392b; color: white; }
  .nan-mid { background: #e67e22; color: white; }
  .nan-lo  { background: #f9e79f; color: #555; }

  /* Race Table */
  .table-wrap { overflow-x: auto; }
  table.race-table { border-collapse: collapse; width: 100%; font-size: 10px; }
  table.race-table th { background: #2c3e50; color: white; padding: 4px 6px;
                        text-align: center; border: 1px solid #222;
                        white-space: nowrap; font-size: 9px; font-weight: bold; }
  table.race-table td { padding: 3px 5px; border: 1px solid #e0e0e0;
                        text-align: center; white-space: nowrap; }

  /* 行ハイライト */
  .row-buy { outline: 2px solid #c0392b; outline-offset: -2px; }
  .row-buy td { background: #fde8e8 !important; }
  .row-r1 td  { background: #fef5f5 !important; }
  .row-r2 td  { background: #fef9ee !important; }
  .row-r3 td  { background: #f3faf5 !important; }

  /* 特定セル */
  .td-rank  { font-weight: bold; min-width: 32px; }
  .td-horse { text-align: left !important; font-weight: bold; min-width: 95px; font-size: 11px; }
  .td-jky   { font-size: 9px; min-width: 38px; }
  .td-odds  { min-width: 38px; }
  .td-prob  { min-width: 42px; color: #16a085; font-weight: bold; }
  .td-buy   { background: #c0392b !important; color: white !important;
               font-weight: bold; min-width: 32px; }
  .td-watch { background: #e67e22 !important; color: white !important; min-width: 32px; }
  .td-nan   { background: #ffe0e0 !important; color: #c0392b;
               font-weight: bold; font-size: 8px; }
  .td-none  { color: #ccc; }

  .footer { font-size: 8px; color: #aaa; text-align: right; margin-top: 10px; }
</style>"""

    # ═══════════════════════════════════════════════════════════
    # Section 1: 買い目サマリー
    # ═══════════════════════════════════════════════════════════
    buy_cards   = []
    watch_cards = []

    for rd in race_data:
        seg_key = rd['seg_key']
        seg_color = SEG_COLOR.get(seg_key, '#888')
        seg_lbl   = SEG_LABEL.get(seg_key, seg_key or '?')
        venue_full = next((v for k, v in V_FULL.items() if k in rd['kaikai']), rd['kaikai'][:3])

        for _, r in rd['grp'].iterrows():
            c_rank = r.get('clogit_rank')
            c_buy  = bool(r.get('clogit_buy', False))
            horse  = r.get('馬名S', '')

            ci = card_map.get(horse, {})
            ov = ci.get('単勝オッズ', r.get('単勝オッズ', ''))
            odds_s = f'{float(ov):.1f}倍' if ov not in ('', None) and str(ov) not in ('nan', '') else '未発表'
            jockey = str(ci.get('騎手', r.get('dc_騎手', r.get('騎手', '')))).strip()

            try:
                r_int = int(float(c_rank))
            except Exception:
                r_int = None

            chip = f'<span class="seg-chip" style="background:{seg_color}">{seg_lbl}</span>'
            race_lbl = f'{venue_full} {rd["r_num"]}R　{chip}'

            if c_buy:
                buy_cards.append(f'''
<div class="buy-card confirmed">
  <div class="card-race">{race_lbl}</div>
  <div class="card-horse">{horse}</div>
  <div class="card-meta">{jockey}　単勝 {odds_s}</div>
  <span class="badge-buy">◎ 買い</span>
</div>''')
            elif seg_key in ('芝短', '芝長') and r_int == 1:
                watch_cards.append(f'''
<div class="buy-card watch-card">
  <div class="card-race">{race_lbl}</div>
  <div class="card-horse">{horse}</div>
  <div class="card-meta">{jockey}　単勝 {odds_s}</div>
  <span class="badge-watch">◆ 要確認（≥6倍で買い）</span>
</div>''')

    buy_html = '<div class="buy-section">'
    buy_html += '<div class="section-title buy">◎ 本日の買い目</div>'
    if buy_cards:
        buy_html += f'<div class="buy-grid">{"".join(buy_cards)}</div>'
    else:
        buy_html += '<p class="no-signal">買いシグナルなし（オッズ未発表または条件未達）</p>'

    if watch_cards:
        buy_html += '<div class="section-title watch">◆ 要オッズ確認 — 芝短/芝長 ROI1位</div>'
        buy_html += f'<div class="buy-grid">{"".join(watch_cards)}</div>'

    buy_html += '</div>'

    # ═══════════════════════════════════════════════════════════
    # Section 2: レース別詳細（ヒートマップ + NaN一覧）
    # ═══════════════════════════════════════════════════════════
    race_blocks = []

    for rd in race_data:
        grp     = rd['grp']
        seg_key = rd['seg_key']
        feats   = rd['feats']
        dist_m  = rd['dist_m']
        surf    = rd['surf']

        seg_color = SEG_COLOR.get(seg_key, '#888')
        seg_lbl   = SEG_LABEL.get(seg_key, seg_key or '?')
        venue_s   = next((v for k, v in V_SHORT.items() if k in rd['kaikai']), rd['kaikai'][:2])
        dist_str  = f'{int(dist_m)}m' if pd.notna(dist_m) else '?m'

        # 表示特徴量（_isnanは別扱い）
        display_feats = [f for f in feats if not f.endswith('_isnan')]
        isnan_feats   = [f for f in feats if f.endswith('_isnan')]

        # ── ヒートマップ用パーセンタイル ──────────────────────────
        feat_pct = {}
        for f in display_feats:
            if f in grp.columns:
                vals = pd.to_numeric(grp[f], errors='coerce')
                ranked = vals.rank(pct=True, na_option='keep')
                feat_pct[f] = ranked.to_dict()

        # ── NaN集計（レース内） ───────────────────────────────────
        nan_by_feat = {}
        for f in display_feats:
            if f in grp.columns:
                n = grp[f].isna().sum()
                if n > 0:
                    nan_by_feat[f] = n

        # NaN Alert HTML
        nan_alert_html = ''
        if nan_by_feat:
            chips = []
            for f, n in sorted(nan_by_feat.items(), key=lambda x: -x[1]):
                pct = n / len(grp)
                cls = 'nan-hi' if pct > 0.5 else ('nan-mid' if pct > 0.1 else 'nan-lo')
                chips.append(f'<span class="nan-chip {cls}">{f}: {n}/{len(grp)}頭</span>')
            nan_alert_html = f'<div class="nan-alert">⚠ NaN特徴量:　{"　".join(chips)}</div>'

        # テーブルヘッダー
        hdr = ''
        for f in display_feats:
            hdr += f'<th title="{f}" style="min-width:42px">{short_feat(f)}</th>'
        # _isnan 列ヘッダー
        for f in isnan_feats:
            hdr += f'<th style="color:#f39c12;min-width:32px" title="{f}">{short_feat(f)}</th>'

        # 行HTML
        rows = []
        for _, r in grp.iterrows():
            c_rank  = r.get('clogit_rank')
            c_buy   = bool(r.get('clogit_buy', False))
            c_calib = r.get('clogit_calib')
            horse   = r.get('馬名S', '')

            ci = card_map.get(horse, {})
            ov = ci.get('単勝オッズ', r.get('単勝オッズ', ''))
            odds_s  = f'{float(ov):.1f}' if ov not in ('', None) and str(ov) not in ('nan', '') else '-'
            jockey  = str(ci.get('騎手', r.get('dc_騎手', r.get('騎手', '')))).strip()[:5]
            bango   = r.get('dc_馬番', r.get('馬番', ''))
            prob_s  = f'{c_calib:.1%}' if pd.notna(c_calib) else '-'

            try: r_int = int(float(c_rank))
            except: r_int = None

            if c_buy:
                row_cls = 'row-buy'
            elif r_int == 1: row_cls = 'row-r1'
            elif r_int == 2: row_cls = 'row-r2'
            elif r_int == 3: row_cls = 'row-r3'
            else:             row_cls = ''

            if c_buy:
                buy_td = '<td class="td-buy">◎買</td>'
            elif seg_key in ('芝短', '芝長') and r_int == 1:
                buy_td = '<td class="td-watch">待</td>'
            else:
                buy_td = '<td class="td-none">-</td>'

            rank_s = str(r_int) if r_int else '-'

            # 特徴量セル（ヒートマップ）
            feat_tds = ''
            for f in display_feats:
                raw = r.get(f) if f in r.index else None
                if raw is None or (isinstance(raw, float) and np.isnan(raw)):
                    feat_tds += '<td class="td-nan">NaN</td>'
                else:
                    pct = feat_pct.get(f, {}).get(r.name)
                    bg  = percentile_color(pct)
                    disp = fmt_val(f, raw) or str(raw)
                    feat_tds += f'<td style="background:{bg}">{disp}</td>'

            # _isnan セル
            for f in isnan_feats:
                base = f[:-6]
                ival = 1 if (base in r.index and pd.isna(r[base])) else 0
                cls  = 'nan-hi' if ival else 'td-none'
                feat_tds += f'<td class="{cls}">{"NaN" if ival else "0"}</td>'

            rows.append(
                f'<tr class="{row_cls}">'
                f'<td class="td-rank">{rank_s}</td>'
                f'{buy_td}'
                f'<td class="td-horse">{bango}.{horse}</td>'
                f'<td class="td-jky">{jockey}</td>'
                f'<td class="td-odds">{odds_s}</td>'
                f'<td class="td-prob">{prob_s}</td>'
                f'{feat_tds}'
                f'</tr>'
            )

        race_blocks.append(f'''
<div class="race-block">
  <div class="race-header" style="border-left-color:{seg_color}">
    <span class="race-venue">{venue_s}</span>
    <span class="race-rnum">{rd["r_num"]}R</span>
    <span class="race-name">{rd["race_name"]}</span>
    <span class="race-seg" style="background:{seg_color}">{seg_lbl}</span>
    <span class="race-dist">{surf}{dist_str}</span>
    <span class="n-horses">{len(grp)}頭　特徴{len(display_feats)}個</span>
  </div>
  {nan_alert_html}
  <div class="table-wrap">
  <table class="race-table">
    <thead><tr>
      <th>ROI<br>順位</th><th>買い</th>
      <th style="text-align:left">馬名</th>
      <th>騎手</th><th>オッズ</th><th>勝率</th>
      {hdr}
    </tr></thead>
    <tbody>{"".join(rows)}</tbody>
  </table>
  </div>
</div>''')

    # ═══════════════════════════════════════════════════════════
    # HTML組立
    # ═══════════════════════════════════════════════════════════
    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <title>競馬AI新聞 {date_disp}</title>
  {css}
</head>
<body>
  <div style="background:#1a237e;color:#fff;padding:6px 20px;font-size:0.82rem;display:flex;gap:20px;align-items:center">
    <span>🏇 競馬AI v2</span>
    <a href="accuracy_model_report_20260613.html" style="color:#90caf9;text-decoration:none">📊 的中率モデルレポート</a>
    <a href="model_report_20260613.html" style="color:#a5d6a7;text-decoration:none">📈 ROIモデルレポート</a>
  </div>
  <h1 class="page-title">
    競馬AI 予想新聞　{date_disp}
    <span class="subtitle">{len(race_data)}レース / {len(result)}頭　|　セル色=レース内パーセンタイル（青=低・橙=高）</span>
  </h1>

  {buy_html}

  {"".join(race_blocks)}

  <div class="footer">
    make_newspaper.py v2 | 的中率最大化モデル (accuracy_model.pkl) | clogit + isotonic calibration
    | 芝短/芝長のみ オッズ帯フィルタ(≥6倍) 適用
  </div>
</body>
</html>"""

    # ── 出力 ──────────────────────────────────────────────────
    out_dir  = os.path.join(BASE_DIR, 'docs')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'newspaper_{tgt_date}.html')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'HTML出力: {out_path}')

    gdrive = r'G:\マイドライブ\競馬AI\予想レポート'
    if os.path.isdir(gdrive):
        import shutil
        gd_path = os.path.join(gdrive, f'newspaper_{tgt_date}.html')
        shutil.copy2(out_path, gd_path)
        print(f'Gdrive出力: {gd_path}')

    return out_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', help='日付 YYYYMMDD')
    args = parser.parse_args()
    make_newspaper(args.date)
