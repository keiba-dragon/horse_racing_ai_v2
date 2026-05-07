# coding: utf-8
"""週次競馬新聞レポート生成 — 任意日付・全会場対応"""
import sys, io, os, re, json, pickle, argparse, glob
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd, numpy as np
# pandas 1.x pickleとの互換パッチ
try:
    import pandas.core.arrays.string_ as _s
    _orig = _s.StringDtype.__init__
    def _p(self, *a, **k):
        try: _orig(self, *a, **k)
        except TypeError: _orig(self)
    _s.StringDtype.__init__ = _p
except Exception:
    pass

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── 略称 → 正式会場名 ──
VENUE_FULL = {
    '阪': '阪神', '東': '東京', '中': '中山', '名': '中京',
    '京': '京都', '函': '函館', '新': '新潟', '小': '小倉',
    '札': '札幌', '福': '福島',
}

# ── コース特性 JSON 読み込み ──
_cc_path = os.path.join(base_dir, 'data', 'course_characteristics.json')
if os.path.exists(_cc_path):
    with open(_cc_path, encoding='utf-8') as f:
        COURSE_DB = json.load(f)
else:
    COURSE_DB = {}

def get_course_info(venue_abbr, surf, dist):
    """venue_abbr: '阪', surf: '芝'/'ダ', dist: '1600' → dict or None"""
    full = VENUE_FULL.get(venue_abbr, venue_abbr)
    s = '芝' if '芝' in str(surf) else 'ダ'
    m = re.search(r'\d+', str(dist))
    d = m.group() if m else ''
    return COURSE_DB.get(f'{full}_{s}_{d}')

# ── 種牡馬データベース ──
SIRE_DB = {
    'ゴールドアリュール':  {'type':'ダ', 'icon':'💎', 'comment':'ダート中距離の帝王血統。コパノリッキーなど歴代ダート王輩出。'},
    'ホッコータルマエ':    {'type':'ダ', 'icon':'🏋️', 'comment':'ダート長距離向きスタミナ型。渋馬場でパワー全開。'},
    'シニスターミニスター':{'type':'ダ', 'icon':'🦾', 'comment':'米国系ダート中長距離。時計かかる馬場で真価発揮。'},
    'ヘニーヒューズ':      {'type':'ダ', 'icon':'💪', 'comment':'ダートスピード×パワー。短〜中距離で鉄板の砂巧者。'},
    'ドレフォン':          {'type':'ダ芝', 'icon':'🚀', 'comment':'BCスプリント馬。スピード豊富でダート短距離に高適性。'},
    'パイロ':              {'type':'ダ', 'icon':'🔥', 'comment':'ダートスプリント先行型。砂の短距離戦で本領発揮。'},
    'カジノドライブ':      {'type':'ダ', 'icon':'⚡', 'comment':'米国産ダート中距離。産駒はパワー型が多く渋馬場巧者。'},
    'ディープインパクト':  {'type':'芝', 'icon':'👑', 'comment':'史上最高の種牡馬。外回り芝・良馬場での末脚勝負で圧倒的。'},
    'キズナ':              {'type':'芝', 'icon':'💫', 'comment':'ディープ直子。芝マイル〜中距離、外回り向きの末脚タイプ。'},
    'エピファネイア':      {'type':'芝', 'icon':'👑', 'comment':'菊花賞・JC馬。産駒はスタミナ豊富で芝中距離向き。'},
    'ハービンジャー':      {'type':'芝', 'icon':'🏆', 'comment':'欧州最強クラス。産駒はスタミナ型で芝2000m前後に高適性。'},
    'キタサンブラック':    {'type':'芝', 'icon':'🥇', 'comment':'三冠馬。先行力とスタミナ兼備。内回りコースで真価発揮。'},
    'シルバーステート':    {'type':'芝', 'icon':'🌟', 'comment':'ディープ直子。芝マイル〜中距離、外回り向き末脚タイプ。'},
    'モーリス':            {'type':'芝', 'icon':'👑', 'comment':'G1×6勝。産駒は芝マイル〜中距離向き、持続末脚が武器。'},
    'ドゥラメンテ':        {'type':'芝', 'icon':'👑', 'comment':'クラシック二冠。芝中距離向きでスタミナと瞬発力を兼備。'},
    'オルフェーヴル':      {'type':'芝', 'icon':'⚡', 'comment':'三冠馬。芝中長距離で鋭い末脚。直線勝負で真価。'},
    'ルーラーシップ':      {'type':'芝', 'icon':'👑', 'comment':'QE2C馬。産駒は先行してスタミナを活かす芝中長距離向き。'},
    'サートゥルナーリア':  {'type':'芝', 'icon':'⭐', 'comment':'皐月賞馬。スピードとスタミナのバランスが良い芝マイル〜中距離向き。'},
    'スワーヴリチャード':  {'type':'芝', 'icon':'⭐', 'comment':'大阪杯馬。芝中距離向き、阪神芝との相性も◎。'},
    'フランケル':          {'type':'芝', 'icon':'🌟', 'comment':'無敗の欧州最強馬。産駒は芝マイル〜中距離向きで高能力。'},
    'ミッキーアイル':      {'type':'芝', 'icon':'🏃', 'comment':'NHKマイルC馬。芝スプリント〜マイルで末脚発揮。'},
    'イスラボニータ':      {'type':'芝', 'icon':'💫', 'comment':'皐月賞馬。芝マイル〜中距離向き、速いペースにも対応。'},
    'アドマイヤマーズ':    {'type':'芝', 'icon':'⚡', 'comment':'NHKマイルC・香港マイル馬。スプリント〜マイル向きの切れ者。'},
    'リアルスティール':    {'type':'芝', 'icon':'💫', 'comment':'ディープ直子、UAEダービー馬。芝マイル〜中距離の末脚タイプ。'},
}

def get_sire_info(sire):
    if pd.isna(sire): return None
    return SIRE_DB.get(str(sire).strip())

def assign_marks(df):
    """06_predict_from_card.py と同じロジックで _印 を付与"""
    cur_diff = pd.to_numeric(df.get('cur_偏差値の差'), errors='coerce')
    sub_diff = pd.to_numeric(df.get('sub_偏差値の差'), errors='coerce')
    cur_r    = pd.to_numeric(df.get('cur_ランカー順位'), errors='coerce')
    sub_r    = pd.to_numeric(df.get('sub_ランカー順位'), errors='coerce')
    if cur_diff.isna().all() or cur_r.isna().all():
        df['_印'] = ''
        return df

    both_r1  = (cur_r == 1) & (sub_r == 1)
    star     = (cur_r <= 3) & (sub_r <= 3) & ~both_r1
    _odds = pd.to_numeric(df['dc_単勝オッズ'] if 'dc_単勝オッズ' in df.columns else pd.Series(np.nan, index=df.index), errors='coerce')
    if _odds.isna().all() and '単勝オッズ' in df.columns:
        _odds = pd.to_numeric(df['単勝オッズ'], errors='coerce')
    odds_ok3 = _odds.isna() | (_odds >= 3)
    odds_ok5 = _odds.isna() | (_odds >= 5)
    mask_gekiatu = both_r1 & (cur_diff >= 10) & (sub_diff >= 10) & odds_ok5
    mask_maru    = both_r1 & (sub_diff >= 10) & odds_ok3 & ~mask_gekiatu
    mask_diamond = (cur_r <= 2) & (sub_r <= 2) & ~both_r1 & (sub_diff >= 10) & odds_ok5
    mask_hoshi   = star & ~((cur_r <= 2) & (sub_r <= 2)) & odds_ok5 & (sub_diff >= 10)
    def mark_of(i):
        if mask_gekiatu.iloc[i]: return '激熱'
        if mask_maru.iloc[i]:    return '〇'
        if mask_diamond.iloc[i]: return '▲'
        if mask_hoshi.iloc[i]:   return '☆'
        return ''
    df = df.copy()
    df['_印'] = [mark_of(i) for i in range(len(df))]

    # 印候補: 能力条件を満たすがオッズ条件未達の馬
    _odds_known = _odds.notna()
    mask_gekiatu_nomi = both_r1 & (cur_diff >= 10) & (sub_diff >= 10) & _odds_known & (_odds < 5)
    mask_maru_nomi    = both_r1 & (sub_diff >= 10) & ~(cur_diff >= 10) & _odds_known & (_odds < 3)
    mask_diamond_nomi = (cur_r <= 2) & (sub_r <= 2) & ~both_r1 & (sub_diff >= 10) & _odds_known & (_odds < 5)
    mask_hoshi_nomi   = star & ~((cur_r <= 2) & (sub_r <= 2)) & (sub_diff >= 10) & _odds_known & (_odds < 5)
    def nomi_mark_of(i):
        if mask_gekiatu_nomi.iloc[i]: return '激熱※'
        if mask_maru_nomi.iloc[i]:    return '〇※'
        if mask_diamond_nomi.iloc[i]: return '▲※'
        if mask_hoshi_nomi.iloc[i]:   return '☆※'
        return ''
    df['_nomi_印'] = [nomi_mark_of(i) for i in range(len(df))]
    return df

# ── ヘルパー ──
def fmt_odds(v):
    try:
        f = float(v)
        if f > 0: return f'{f:.1f}'
    except: pass
    return '-'

def mark_bg(m):
    return {'◎':'#c0392b','〇':'#27ae60','▲':'#2471a3','△':'#95a5a6','×':'#bbb','注':'#e67e22'}.get(m, 'transparent')

def diff_bar(v, max_v=30):
    if pd.isna(v): return '<span style="color:#555">-</span>'
    v = float(v)
    w = min(abs(v) / max_v * 100, 100)
    color = '#27ae60' if v >= 0 else '#e74c3c'
    sign = '+' if v >= 0 else ''
    return (f'<div style="display:flex;align-items:center;gap:4px">'
            f'<div style="width:{w:.0f}px;height:8px;background:{color};border-radius:3px;min-width:2px;max-width:80px"></div>'
            f'<span style="font-size:11px;color:{color};font-weight:bold">{sign}{v:.1f}</span></div>')

def extract_venue_abbr(kaikai):
    m = re.search(r'([^\d\s]+)', str(kaikai))
    return m.group(1) if m else str(kaikai)

def ai_comment(grp):
    double = pd.DataFrame()
    if 'cur_偏差値の差' in grp.columns and 'sub_偏差値の差' in grp.columns:
        double = grp[
            (grp['cur_偏差値の差'].fillna(-99) >= 15) &
            (grp['sub_偏差値の差'].fillna(-99) >= 15)
        ]
    top1_cur = grp.nsmallest(1,'cur_ランカー順位')['馬名S'].iloc[0] if 'cur_ランカー順位' in grp.columns and len(grp)>0 else ''
    top1_sub = grp.nsmallest(1,'sub_ランカー順位')['馬名S'].iloc[0] if 'sub_ランカー順位' in grp.columns and len(grp)>0 else ''
    if len(double) > 0:
        names = '・'.join(double['馬名S'].tolist())
        return f'<span style="color:#e74c3c;font-weight:bold">🔥 {names} が両モデル高評価</span>'
    if top1_cur == top1_sub and top1_cur:
        return f'<span style="color:#27ae60;font-weight:bold">✓ {top1_cur} が距離・クラス両モデル1位</span>'
    parts = []
    if top1_cur: parts.append(f'距離1位: <b>{top1_cur}</b>')
    if top1_sub and top1_sub != top1_cur: parts.append(f'クラス1位: <b>{top1_sub}</b>')
    return ' / '.join(parts)

# ── レースカード HTML 生成 ──
def render_race_card(r, name, dist, cls, surf, t, grp, venue_abbr):
    surf_label = '芝' if '芝' in str(surf) else 'ダ'
    surf_color = '#27ae60' if surf_label == '芝' else '#e67e22'
    dist_num = re.search(r'\d+', str(dist))
    dist_num = dist_num.group() if dist_num else dist

    course_info = get_course_info(venue_abbr, surf_label, dist_num)
    if course_info:
        course_section = f'''<div class="course-box">
          <div class="course-title">📍 {VENUE_FULL.get(venue_abbr, venue_abbr)} {surf_label}{dist_num}m — コースの特徴</div>
          <div class="course-body">
            <div class="course-shape">🗺️ {course_info.get("特徴","")}</div>
            <div class="course-points-list">
              <div class="course-point-item">🐴 脚質: {course_info.get("脚質","")}</div>
              <div class="course-point-item">🎰 枠: {course_info.get("枠","")}</div>
            </div>
            <div class="course-blood">🧬 血統: {course_info.get("血統","")}</div>
            <div class="course-comment">💬 {course_info.get("コメント","")}</div>
          </div>
        </div>'''
    else:
        course_section = ''

    rows_html = ''
    grp_s = grp.sort_values('馬番', na_position='last') if '馬番' in grp.columns else grp
    for _, h in grp_s.iterrows():
        mark = h.get('_印','') or ''
        try: banum = int(float(h['馬番'])) if pd.notna(h.get('馬番')) else '-'
        except: banum = '-'
        horse  = h.get('馬名S','')
        jockey = h.get('dc_騎手', h.get('騎手',''))
        odds   = fmt_odds(h.get('単勝オッズ', h.get('dc_単勝オッズ','')))
        cur_r  = int(h['cur_ランカー順位']) if 'cur_ランカー順位' in grp.columns and pd.notna(h.get('cur_ランカー順位')) else '-'
        sub_r  = int(h['sub_ランカー順位']) if 'sub_ランカー順位' in grp.columns and pd.notna(h.get('sub_ランカー順位')) else '-'
        cur_d  = h.get('cur_偏差値の差', np.nan)
        sub_d  = h.get('sub_偏差値の差', np.nan)
        _z = h.get('1走前_着順_num') if pd.notna(h.get('1走前_着順_num', float('nan'))) else h.get('前走着順_num')
        zenso  = int(_z) if _z is not None and pd.notna(_z) else '-'
        seibetsu = h.get('性齢', h.get('性', ''))
        kinryo = h.get('斤量','')
        try: kinryo = f"{float(kinryo):.1f}"
        except: pass
        sire_name = str(h.get('種牡馬','')).strip() if not pd.isna(h.get('種牡馬','')) else ''

        cur_score = h.get('cur_コース偏差値', np.nan)
        sub_score = h.get('sub_コース偏差値', np.nan)
        score_vals = [v for v in [cur_score, sub_score] if not pd.isna(v)]
        total_pt = np.mean(score_vals) if score_vals else np.nan

        mark_style = f'background:{mark_bg(mark)};color:white;font-weight:bold;font-size:14px' if mark else ''
        is_both = (not pd.isna(cur_d) and float(cur_d) >= 15) and (not pd.isna(sub_d) and float(sub_d) >= 15) if not (pd.isna(cur_d) or pd.isna(sub_d)) else False
        is_top  = ((not pd.isna(cur_d) and float(cur_d) >= 10) or (not pd.isna(sub_d) and float(sub_d) >= 10)) if not (pd.isna(cur_d) and pd.isna(sub_d)) else False
        row_class = 'row-both' if is_both else ('row-top' if is_top else '')

        cur_r_style = 'color:#e8b400;font-weight:bold' if cur_r != '-' and cur_r <= 3 else 'color:#8b949e'
        sub_r_style = 'color:#e8b400;font-weight:bold' if sub_r != '-' and sub_r <= 3 else 'color:#8b949e'

        if not pd.isna(total_pt):
            pt_color = '#e74c3c' if total_pt >= 65 else ('#f39c12' if total_pt >= 58 else ('#27ae60' if total_pt >= 52 else '#8b949e'))
            total_pt_html = f'<span style="font-weight:bold;font-size:13px;color:{pt_color}">{total_pt:.1f}</span>'
        else:
            total_pt_html = '<span style="color:#555">-</span>'

        sire_info = get_sire_info(sire_name)
        bofuha_name = str(h.get('母父馬','')).strip() if not pd.isna(h.get('母父馬','')) else ''
        if sire_info:
            tc = {'芝':'#27ae60','ダ':'#e67e22','芝ダ':'#3498db'}.get(sire_info['type'], '#888')
            bofuha_html = f' <span style="color:#666;font-size:10px">母父:{bofuha_name}</span>' if bofuha_name and bofuha_name != 'nan' else ''
            sire_cell = (f'<div style="font-size:10px;color:#777;margin-top:2px">'
                         f'{sire_info["icon"]} <span style="color:#aaa">{sire_name}</span>{bofuha_html}'
                         f' <span style="background:{tc}22;color:{tc};border:1px solid {tc}44;border-radius:4px;padding:0 4px;margin-left:3px;font-size:9px">{sire_info["type"]}向き</span>'
                         f'</div>'
                         f'<div style="font-size:10.5px;color:#7a8a9a;margin-top:2px;line-height:1.4">{sire_info["comment"]}</div>')
        else:
            bofuha_html = f' <span style="color:#666;font-size:10px">母父:{bofuha_name}</span>' if bofuha_name and bofuha_name != 'nan' else ''
            sire_cell = f'<div style="font-size:10px;color:#555;margin-top:2px">🐴 {sire_name}{bofuha_html}</div>' if sire_name else ''

        rows_html += f'''<tr class="{row_class}">
          <td style="width:28px;{mark_style}">{mark}</td>
          <td style="width:32px;font-weight:bold;color:#8b949e">{banum}</td>
          <td style="text-align:left"><div style="font-weight:bold;font-size:13px">{horse}</div>{sire_cell}</td>
          <td style="color:#aaa;font-size:12px">{jockey}</td>
          <td style="color:#8b949e;font-size:11px">{seibetsu} {kinryo}kg</td>
          <td style="font-weight:bold;color:#f0a500">{odds}</td>
          <td>{total_pt_html}</td>
          <td style="{cur_r_style};font-size:12px">{cur_r}位</td>
          <td>{diff_bar(cur_d)}</td>
          <td style="{sub_r_style};font-size:12px">{sub_r}位</td>
          <td>{diff_bar(sub_d)}</td>
          <td style="color:#8b949e;font-size:12px">{zenso}</td>
        </tr>'''

    comment = ai_comment(grp)
    race_id = f'v{venue_abbr}r{r}'

    return f'''<div class="race-card" id="{race_id}">
      <div class="race-header">
        <div class="race-number">{r}R</div>
        <div class="race-info">
          <div class="race-name">{name}</div>
          <div class="race-meta">
            <span class="surface" style="background:{surf_color}">{surf_label} {dist_num}m</span>
            <span class="cls-badge">{cls}</span>
            {f'<span class="race-time">🕐 {t}</span>' if t else ''}
          </div>
          <div class="race-comment">{comment}</div>
        </div>
      </div>
      {course_section}
      <div class="table-wrap">
        <table class="horse-table">
          <thead><tr>
            <th>印</th><th>馬番</th><th>馬名 / 父</th><th>騎手</th><th>性齢/斤量</th><th>オッズ</th>
            <th>強さPT<br><span style="font-size:9px;font-weight:normal">50=平均</span></th>
            <th>距離Rnk</th><th>距離diff</th><th>クラスRnk</th><th>クラスdiff</th>
            <th>前走着順</th>
          </tr></thead>
          <tbody>{rows_html}</tbody>
        </table>
      </div>
    </div>'''

# ── 会場セクション HTML 生成 ──
def render_venue_section(venue_abbr, df_venue, card_df, venue_color):
    full_name = VENUE_FULL.get(venue_abbr, venue_abbr)
    time_map = {}
    if card_df is not None and '発走時刻' in card_df.columns:
        cv = card_df[card_df['開催'].astype(str).str.contains(venue_abbr, na=False)]
        for _, row in cv[['Ｒ','発走時刻']].drop_duplicates().iterrows():
            try: time_map[int(float(row['Ｒ']))] = str(row['発走時刻'])
            except: pass

    races_data = []
    for r_num, grp in df_venue.sort_values('Ｒ').groupby('Ｒ'):
        r = int(float(r_num))
        name = grp['レース名'].iloc[0] if 'レース名' in grp.columns else f'{r}R'
        dist = str(grp['距離'].iloc[0]) if '距離' in grp.columns else ''
        cls  = grp['クラス'].iloc[0] if 'クラス' in grp.columns else ''
        surf = grp['芝・ダ'].iloc[0] if '芝・ダ' in grp.columns else ''
        t    = time_map.get(r, '')
        races_data.append((r, name, dist, cls, surf, t, grp))

    # 激熱ピック
    best_picks = []
    for r, name, dist, cls, surf, t, grp in races_data:
        if 'cur_偏差値の差' not in grp.columns: continue
        double = grp[
            (grp['cur_偏差値の差'].fillna(-99) >= 15) &
            (grp['sub_偏差値の差'].fillna(-99) >= 15)
        ] if 'sub_偏差値の差' in grp.columns else pd.DataFrame()
        for _, h in double.iterrows():
            odds = fmt_odds(h.get('単勝オッズ', h.get('dc_単勝オッズ','')))
            best_picks.append((r, name, h['馬名S'], odds,
                               h.get('cur_偏差値の差', np.nan),
                               h.get('sub_偏差値の差', np.nan)))

    picks_html = ''
    if best_picks:
        rows = ''.join([
            f'<tr><td style="font-weight:bold;color:#e8b400">{r}R</td>'
            f'<td style="text-align:left">{rname}</td>'
            f'<td style="font-weight:bold;font-size:14px;color:#fff">{hname}</td>'
            f'<td style="color:#f39c12;font-weight:bold">{odds}</td>'
            f'<td style="color:#2ecc71">+{cd:.1f}</td>'
            f'<td style="color:#3498db">+{sd:.1f}</td></tr>'
            for r, rname, hname, odds, cd, sd in best_picks
        ])
        picks_html = f'''<div class="picks-box">
          <div class="picks-title">🔥 {full_name}の激熱ピック（両モデル偏差値差+15以上）</div>
          <table class="picks-table"><thead><tr><th>R</th><th>レース名</th><th>馬名</th><th>オッズ</th><th>距離diff</th><th>クラスdiff</th></tr></thead>
          <tbody>{rows}</tbody></table>
        </div>'''

    nav_items = ''.join([
        f'<a href="#v{venue_abbr}r{r}" class="nav-item">{r}R</a>'
        for r, *_ in races_data
    ])

    race_cards = '\n'.join([
        render_race_card(r, name, dist, cls, surf, t, grp, venue_abbr)
        for r, name, dist, cls, surf, t, grp in races_data
    ])

    return f'''<div class="venue-section" id="venue-{venue_abbr}">
      <div class="venue-header" style="border-left-color:{venue_color}">
        <span class="venue-name" style="color:{venue_color}">🏟️ {full_name}競馬場</span>
        <span class="venue-count">{len(races_data)}レース</span>
      </div>
      <div class="venue-nav">{nav_items}</div>
      {picks_html}
      {race_cards}
    </div>'''

# ── メイン ──
def main():
    ap = argparse.ArgumentParser(description='週次競馬新聞レポート生成')
    ap.add_argument('--date', default=None, help='日付キー(例:4月12日)。省略時は最新キャッシュ')
    ap.add_argument('--venue', default=None, help='会場絞り込み(例:阪、東)。省略時は全会場')
    ap.add_argument('--cache-file', default=None, help='キャッシュファイルを直接指定')
    args = ap.parse_args()

    cache_dir = os.path.join(base_dir, 'data', 'raw', 'cache')
    if args.cache_file:
        cache_files = [args.cache_file]
    elif args.date:
        # オッズ確認用など suffix 付きも含めて最新を選ぶ
        pattern = os.path.join(cache_dir, f'出馬表形式{args.date}*.cache.pkl')
        cache_files = sorted(glob.glob(pattern))
    else:
        cache_files = sorted(glob.glob(os.path.join(cache_dir, '出馬表形式*.cache.pkl')))

    if not cache_files:
        print(f'キャッシュファイルが見つかりません: {cache_dir}')
        sys.exit(1)

    # ベース名（オッズ確認用suffix除去）のpklを優先
    def base_name_of(p):
        s = os.path.basename(p)
        s = re.sub(r'オッズ確認用.*\.cache\.pkl$', '.cache.pkl', s)
        return s

    # ベース名pklがあればそれを使う（オッズ確認用より優先）
    base_candidates = [f for f in cache_files if 'オッズ確認用' not in f]
    cache_file = base_candidates[-1] if base_candidates else cache_files[-1]
    cache_name = os.path.basename(cache_file)
    print(f'読み込み: {cache_name}')

    with open(cache_file, 'rb') as f:
        cached = pickle.load(f)
    df = cached['result']
    card_df = cached.get('card_df')

    # ArrowBacked/StringDtype列・列インデックスをobjectに変換（pandas互換性）
    if 'string' in str(df.columns.dtype).lower():
        df.columns = df.columns.astype(object)
    for _c in list(df.columns):
        if 'string' in str(df[_c].dtype).lower():
            df[_c] = df[_c].astype(object)
    if card_df is not None:
        if 'string' in str(card_df.columns.dtype).lower():
            card_df.columns = card_df.columns.astype(object)
        for _c in list(card_df.columns):
            if 'string' in str(card_df[_c].dtype).lower():
                card_df[_c] = card_df[_c].astype(object)

    # オッズjsonがあれば適用
    base_stem = re.sub(r'\.cache\.pkl$', '', cache_name)
    odds_path = os.path.join(cache_dir, base_stem + '.odds.json')
    if os.path.exists(odds_path):
        with open(odds_path, encoding='utf-8') as f:
            odds_dict = json.load(f)
        # dfにも適用
        if '馬名S' in df.columns:
            df['単勝オッズ'] = df['馬名S'].map(lambda x: odds_dict.get(str(x), float('nan')))
        if card_df is not None and '馬名S' in card_df.columns:
            card_df = card_df.copy()
            card_df['単勝オッズ'] = card_df['馬名S'].map(lambda x: odds_dict.get(str(x), float('nan')))
        print(f'オッズ適用: {len(odds_dict)}頭')

    # 日付ラベル抽出（オッズ確認用などのsuffixは除去）
    m = re.search(r'出馬表形式(.+?)(?:オッズ確認用)?\.cache\.pkl', cache_name)
    date_label = m.group(1) if m else cache_name

    # 印付与（generate_html と同ロジック）
    df = assign_marks(df)

    # 会場一覧
    df['_venue_abbr'] = df['開催'].astype(str).apply(extract_venue_abbr)
    venues = df['_venue_abbr'].unique().tolist()
    if args.venue:
        venues = [v for v in venues if args.venue in v]

    # 会場ごとのカラーパレット
    VENUE_COLORS = ['#e8b400','#58a6ff','#f85149','#3fb950','#d2a8ff','#ff9800','#26c6da','#ec407a']
    venue_color_map = {v: VENUE_COLORS[i % len(VENUE_COLORS)] for i, v in enumerate(venues)}

    print(f'会場: {venues}')

    # トップナビ（会場別）
    top_nav = ''.join([
        f'<a href="#venue-{v}" class="top-nav-item" style="border-color:{venue_color_map[v]};color:{venue_color_map[v]}">'
        f'{VENUE_FULL.get(v,v)}</a>'
        for v in venues
    ])

    # 全印馬サマリ
    MARK_ORDER = {'激熱': 0, '〇': 1, '▲': 2, '☆': 3}
    MARK_COLOR = {'激熱': '#e74c3c', '〇': '#27ae60', '▲': '#2471a3', '☆': '#f39c12'}
    all_marks = []
    for v in venues:
        dv = df[df['_venue_abbr'] == v]
        for r_num, grp in dv.groupby('Ｒ'):
            for _, h in grp.iterrows():
                mark = h.get('_印', '')
                if not mark: continue
                odds = fmt_odds(h.get('単勝オッズ', h.get('dc_単勝オッズ', '')))
                cur_d = h.get('cur_偏差値の差', np.nan)
                sub_d = h.get('sub_偏差値の差', np.nan)
                cur_r = int(h['cur_ランカー順位']) if 'cur_ランカー順位' in grp.columns and pd.notna(h.get('cur_ランカー順位')) else '-'
                sub_r = int(h['sub_ランカー順位']) if 'sub_ランカー順位' in grp.columns and pd.notna(h.get('sub_ランカー順位')) else '-'
                all_marks.append({
                    'venue': VENUE_FULL.get(v, v),
                    'r': int(float(r_num)),
                    'mark': mark,
                    'horse': h['馬名S'],
                    'odds': odds,
                    'cur_d': cur_d,
                    'sub_d': sub_d,
                    'cur_r': cur_r,
                    'sub_r': sub_r,
                    'sort_key': (MARK_ORDER.get(mark, 9), int(float(r_num))),
                })
    all_marks.sort(key=lambda x: x['sort_key'])

    summary_html = ''
    if all_marks:
        rows = ''
        for m in all_marks:
            mc = MARK_COLOR.get(m['mark'], '#aaa')
            cd = f"+{m['cur_d']:.1f}" if pd.notna(m['cur_d']) else '-'
            sd = f"+{m['sub_d']:.1f}" if pd.notna(m['sub_d']) else '-'
            rows += (
                f'<tr>'
                f'<td style="font-weight:bold;color:{mc};font-size:15px">{m["mark"]}</td>'
                f'<td style="color:#aaa">{m["venue"]}</td>'
                f'<td style="font-weight:bold;color:#e8b400">{m["r"]}R</td>'
                f'<td style="font-weight:bold;color:#fff;font-size:14px">{m["horse"]}</td>'
                f'<td style="color:#f0a500;font-weight:bold">{m["odds"]}</td>'
                f'<td style="color:#8b949e">{m["cur_r"]}位</td>'
                f'<td style="color:#2ecc71">{cd}</td>'
                f'<td style="color:#8b949e">{m["sub_r"]}位</td>'
                f'<td style="color:#3498db">{sd}</td>'
                f'</tr>'
            )
        summary_html = f'''<div class="summary-box">
          <div class="summary-title">📋 本日の全印馬一覧</div>
          <table class="picks-table">
            <thead><tr><th>印</th><th>会場</th><th>R</th><th>馬名</th><th>オッズ</th><th>距離Rnk</th><th>距離diff</th><th>クラスRnk</th><th>クラスdiff</th></tr></thead>
            <tbody>{rows}</tbody>
          </table>
        </div>'''

    # 印候補サマリ (_nomi_印 が存在する場合)
    NOMI_MARK_ORDER = {'激熱※': 0, '〇※': 1, '▲※': 2, '☆※': 3}
    NOMI_MARK_COLOR = {'激熱※': '#c0392b', '〇※': '#1e8449', '▲※': '#1a5276', '☆※': '#b7770d'}
    all_nomi = []
    if '_nomi_印' in df.columns:
        for v in venues:
            dv = df[df['_venue_abbr'] == v]
            for r_num, grp in dv.groupby('Ｒ'):
                for _, h in grp.iterrows():
                    nomi = h.get('_nomi_印', '')
                    if not nomi: continue
                    odds = fmt_odds(h.get('単勝オッズ', h.get('dc_単勝オッズ', '')))
                    cur_d = h.get('cur_偏差値の差', np.nan)
                    sub_d = h.get('sub_偏差値の差', np.nan)
                    cur_r = int(h['cur_ランカー順位']) if 'cur_ランカー順位' in grp.columns and pd.notna(h.get('cur_ランカー順位')) else '-'
                    sub_r = int(h['sub_ランカー順位']) if 'sub_ランカー順位' in grp.columns and pd.notna(h.get('sub_ランカー順位')) else '-'
                    current_mark = h.get('_印', '') or '無印'
                    all_nomi.append({
                        'venue': VENUE_FULL.get(v, v),
                        'r': int(float(r_num)),
                        'nomi': nomi,
                        'current_mark': current_mark,
                        'horse': h['馬名S'],
                        'odds': odds,
                        'cur_d': cur_d,
                        'sub_d': sub_d,
                        'cur_r': cur_r,
                        'sub_r': sub_r,
                        'sort_key': (NOMI_MARK_ORDER.get(nomi, 9), int(float(r_num))),
                    })
    all_nomi.sort(key=lambda x: x['sort_key'])

    nomi_summary_html = ''
    if all_nomi:
        nomi_rows = ''
        for m in all_nomi:
            mc = NOMI_MARK_COLOR.get(m['nomi'], '#aaa')
            cd = f"+{m['cur_d']:.1f}" if pd.notna(m['cur_d']) else '-'
            sd = f"+{m['sub_d']:.1f}" if pd.notna(m['sub_d']) else '-'
            nomi_rows += (
                f'<tr>'
                f'<td style="font-weight:bold;color:{mc};font-size:14px">{m["nomi"]}</td>'
                f'<td style="color:#888;font-size:11px">{m["current_mark"]}</td>'
                f'<td style="color:#aaa">{m["venue"]}</td>'
                f'<td style="font-weight:bold;color:#e8b400">{m["r"]}R</td>'
                f'<td style="font-weight:bold;color:#fff;font-size:14px">{m["horse"]}</td>'
                f'<td style="color:#f0a500;font-weight:bold">{m["odds"]}</td>'
                f'<td style="color:#8b949e">{m["cur_r"]}位</td>'
                f'<td style="color:#2ecc71">{cd}</td>'
                f'<td style="color:#8b949e">{m["sub_r"]}位</td>'
                f'<td style="color:#3498db">{sd}</td>'
                f'</tr>'
            )
        nomi_summary_html = f'''<div class="summary-box" style="border-color:#7d6608;background:linear-gradient(135deg,#1a1500,#2d2500)">
          <div class="summary-title" style="color:#e8b400">🔍 印候補一覧（オッズ条件未達）</div>
          <table class="picks-table">
            <thead><tr><th>能力印</th><th>現印</th><th>会場</th><th>R</th><th>馬名</th><th>オッズ</th><th>距離Rnk</th><th>距離diff</th><th>クラスRnk</th><th>クラスdiff</th></tr></thead>
            <tbody>{nomi_rows}</tbody>
          </table>
        </div>'''

    # 会場セクション生成
    venue_sections = '\n'.join([
        render_venue_section(
            v,
            df[df['_venue_abbr'] == v].copy(),
            card_df,
            venue_color_map[v]
        )
        for v in venues
    ])

    # 開催日表示
    venue_names_str = '・'.join([VENUE_FULL.get(v,v) for v in venues])
    total_races = sum(df[df['_venue_abbr'] == v]['Ｒ'].nunique() for v in venues)

    from datetime import datetime as _dt
    updated_at = _dt.now().strftime('%Y-%m-%d %H:%M:%S')

    html = f'''<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>競馬新聞 {date_label}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Hiragino Sans','Yu Gothic',sans-serif;background:#0d1117;color:#e6edf3;font-size:13px;line-height:1.5}}
.top-bar{{background:linear-gradient(135deg,#1a1f3a 0%,#2d1b4e 100%);padding:20px 16px;border-bottom:4px solid #e8b400}}
.top-bar h1{{font-size:24px;color:#e8b400;letter-spacing:2px}}
.top-bar .date{{font-size:15px;color:#fff;margin-top:6px;font-weight:bold}}
.top-bar .subtitle{{color:#aaa;font-size:12px;margin-top:4px}}
.top-bar .generated-at{{color:#666;font-size:11px;margin-top:4px}}
.top-nav{{position:sticky;top:0;z-index:100;background:#161b22;border-bottom:2px solid #30363d;padding:10px 12px;display:flex;gap:8px;overflow-x:auto;flex-wrap:wrap}}
.top-nav-item{{padding:5px 14px;background:#21262d;border-radius:14px;text-decoration:none;font-size:13px;font-weight:bold;white-space:nowrap;border:1px solid transparent}}
.top-nav-item:hover{{background:#30363d}}

.summary-box{{margin:14px;padding:14px 16px;background:linear-gradient(135deg,#1a0a0a,#2d1010);border:2px solid #c0392b;border-radius:10px}}
.summary-title{{font-size:16px;font-weight:bold;color:#e74c3c;margin-bottom:10px}}
.picks-box{{margin:12px 12px 0;padding:12px 14px;background:linear-gradient(135deg,#1a0a0a,#2d1010);border:1px solid #8b2020;border-radius:8px}}
.picks-title{{font-size:13px;font-weight:bold;color:#e74c3c;margin-bottom:8px}}
.picks-table{{width:100%;border-collapse:collapse}}
.picks-table th{{background:#3d1010;color:#e74c3c;padding:6px 10px;text-align:center;font-size:12px}}
.picks-table td{{padding:7px 10px;text-align:center;border-bottom:1px solid #3d1010}}

.venue-section{{margin:16px 0}}
.venue-header{{margin:0 12px;padding:12px 16px;background:linear-gradient(90deg,#1f2937,#161b22);border-left:5px solid #e8b400;border-radius:6px 6px 0 0;display:flex;align-items:center;gap:12px}}
.venue-name{{font-size:20px;font-weight:900;letter-spacing:1px}}
.venue-count{{font-size:12px;color:#aaa;background:#21262d;padding:2px 8px;border-radius:10px}}
.venue-nav{{margin:0 12px;padding:8px 10px;background:#161b22;border-bottom:1px solid #30363d;display:flex;gap:6px;overflow-x:auto;flex-wrap:wrap}}
.nav-item{{padding:3px 10px;background:#21262d;border-radius:10px;color:#58a6ff;text-decoration:none;font-size:12px;font-weight:bold;white-space:nowrap}}
.nav-item:hover{{background:#30363d}}

.race-card{{margin:12px;background:#161b22;border-radius:10px;border:1px solid #30363d;overflow:hidden}}
.race-header{{padding:12px 14px;background:linear-gradient(90deg,#1f2937,#1a2035);display:flex;align-items:flex-start;gap:12px;flex-wrap:wrap}}
.race-number{{font-size:32px;font-weight:900;color:#e8b400;min-width:52px;line-height:1}}
.race-info{{flex:1;min-width:200px}}
.race-name{{font-size:17px;font-weight:bold;color:#fff}}
.race-meta{{display:flex;gap:8px;margin-top:5px;align-items:center;flex-wrap:wrap}}
.surface{{padding:3px 10px;border-radius:10px;color:white;font-weight:bold;font-size:12px}}
.cls-badge{{padding:3px 10px;background:#30363d;border-radius:10px;color:#aaa;font-size:12px}}
.race-time{{color:#58a6ff;font-size:12px}}
.race-comment{{font-size:12px;margin-top:6px;padding-top:6px;border-top:1px solid rgba(255,255,255,0.06)}}

.course-box{{margin:0;padding:12px 16px;background:#0f1923;border-bottom:1px solid #1c2a3a}}
.course-title{{font-size:13px;font-weight:bold;color:#58a6ff;padding-bottom:6px;border-bottom:2px solid #1a5276;margin-bottom:8px}}
.course-body{{font-size:12px}}
.course-shape{{color:#aaa;margin-bottom:6px}}
.course-points-list{{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:8px}}
.course-point-item{{background:#0d1f2d;border:1px solid #1c3a52;border-radius:6px;padding:4px 10px;color:#c8d8e8;font-size:11.5px}}
.course-blood{{color:#b8860b;background:rgba(184,134,11,0.1);padding:6px 10px;border-radius:6px;border-left:3px solid #b8860b;font-size:11.5px;margin-bottom:4px}}
.course-comment{{color:#888;font-size:11.5px;padding-left:4px}}

.table-wrap{{overflow-x:auto}}
.horse-table{{width:100%;border-collapse:collapse;min-width:650px}}
.horse-table thead tr{{background:#21262d}}
.horse-table th{{padding:6px 8px;text-align:center;color:#8b949e;font-size:11px;border-bottom:1px solid #30363d;white-space:nowrap}}
.horse-table td{{padding:7px 8px;text-align:center;border-bottom:1px solid #1c2128;vertical-align:middle}}
.horse-table tr:last-child td{{border-bottom:none}}
.row-both{{background:rgba(192,57,43,0.18)!important}}
.row-top{{background:rgba(39,174,96,0.07)!important}}
.horse-table tr:hover{{background:rgba(88,166,255,0.05)}}
</style>
</head>
<body>
<div class="top-bar">
  <h1>🏇 競馬新聞 AI予想版</h1>
  <div class="date">{date_label} — {venue_names_str} 計{total_races}レース</div>
  <div class="subtitle">距離モデル × クラスモデル AIスコア完全版　コース特性・血統ワンポイント付き</div>
  <div class="generated-at">🕐 更新日時: {updated_at}</div>
</div>
<nav class="top-nav">{top_nav}</nav>
{summary_html}
{nomi_summary_html}
{venue_sections}
<div style="text-align:center;padding:20px;color:#444;font-size:11px">競馬AI ／ 強さPT=コース偏差値（50=平均）／ diff=偏差値差</div>
</body>
</html>'''

    os.makedirs(os.path.join(base_dir, 'output'), exist_ok=True)
    # ファイル名: weekly_report_YYMMDD.html（docs/ に保存）
    m_date = re.search(r'(\d+)月(\d+)日', date_label)
    if m_date:
        file_date = f'26{int(m_date.group(1)):02d}{int(m_date.group(2)):02d}'
    else:
        file_date = re.sub(r'[\\/:*?"<>|]', '_', date_label)
    docs_dir = os.path.join(base_dir, 'docs')
    os.makedirs(docs_dir, exist_ok=True)
    out = os.path.join(docs_dir, f'weekly_report_{file_date}.html')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'出力完了: {out}')

    # index.html も自動更新
    update_index(docs_dir)
    print(f'index.html 更新完了')


def update_index(docs_dir):
    """docs/ の内容からindex.htmlを再生成"""
    from datetime import datetime as _dt

    # 週次レポート一覧（新しい順）
    weekly_files = sorted(glob.glob(os.path.join(docs_dir, 'weekly_report_*.html')), reverse=True)
    weekly_cards = ''
    for f in weekly_files:
        name = os.path.basename(f)
        m = re.search(r'weekly_report_26(\d{2})(\d{2})\.html', name)
        if m:
            label = f"20{m.group(1)[0:2]}/{m.group(1)[0:2] and m.group(1)[-0:] or ''}"
            mo, dy = m.group(1), m.group(2)
            label = f"2026/{mo}/{dy}"
        else:
            label = name
        weekly_cards += f'''
      <a class="item" href="{name}">
        <span class="item-icon">📰</span>
        <span class="item-label">{label} 週次レポート</span>
        <span class="item-arrow">›</span>
      </a>'''

    # 出馬予測カード（日付ごとに最新1つ、新しい順）
    card_files = sorted(glob.glob(os.path.join(docs_dir, 'card_*.html')), reverse=True)
    seen_dates = set()
    card_cards = ''
    for f in card_files:
        name = os.path.basename(f)
        m = re.search(r'card_(26\d{4})', name)
        if not m:
            continue
        date_key = m.group(1)
        if date_key in seen_dates:
            continue
        seen_dates.add(date_key)
        d = date_key
        label = f"2026/{d[2:4]}/{d[4:6]}"
        card_cards += f'''
      <a class="item" href="{name}">
        <span class="item-icon">🔍</span>
        <span class="item-label">{label} 出馬予測カード</span>
        <span class="item-arrow">›</span>
      </a>'''

    updated_at = _dt.now().strftime('%Y-%m-%d %H:%M')

    html = f'''<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>競馬AI ダッシュボード</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:"Hiragino Kaku Gothic Pro",Meiryo,sans-serif;background:#1a1a2e;color:#e0e0e0;min-height:100vh;padding:24px 16px}}
header{{text-align:center;padding:32px 0 28px}}
header h1{{font-size:1.7em;color:#f0c040;letter-spacing:0.05em}}
header p{{color:#aaa;margin-top:8px;font-size:0.85em}}
header .updated{{color:#555;font-size:0.75em;margin-top:6px}}
.section{{max-width:520px;margin:0 auto 28px}}
.section-title{{font-size:0.8em;font-weight:bold;color:#f0c040;letter-spacing:0.1em;text-transform:uppercase;padding:0 4px 8px;border-bottom:1px solid #2a2a4a;margin-bottom:10px}}
.item{{display:flex;align-items:center;gap:12px;background:#16213e;border-radius:10px;padding:14px 16px;text-decoration:none;color:#e0e0e0;margin-bottom:8px;border:1px solid #2a2a4a;transition:background 0.15s}}
.item:hover{{background:#1a3a5a;border-color:#f0c040}}
.item-icon{{font-size:1.3em;min-width:28px;text-align:center}}
.item-label{{flex:1;font-size:0.95em}}
.item-arrow{{color:#555;font-size:1.2em}}
footer{{text-align:center;color:#555;font-size:0.75em;margin-top:32px}}
footer a{{color:#4a9eff;text-decoration:none}}
</style>
</head>
<body>
<header>
  <h1>🏇 競馬AI ダッシュボード</h1>
  <p>LightGBM + LambdaMART による競馬予測システム</p>
  <div class="updated">更新: {updated_at}</div>
</header>

<div class="section">
  <div class="section-title">週次予想レポート</div>
  {weekly_cards}
</div>

<div class="section">
  <div class="section-title">出馬予測カード</div>
  {card_cards}
</div>

<div class="section">
  <div class="section-title">ROI・成績</div>
  <a class="item" href="actual_bet_roi.html">
    <span class="item-icon">🎫</span>
    <span class="item-label">実際の馬券ROI</span>
    <span class="item-arrow">›</span>
  </a>
  <a class="item" href="daily_roi_2026.html">
    <span class="item-icon">📊</span>
    <span class="item-label">予測ROI（最終オッズ基準）</span>
    <span class="item-arrow">›</span>
  </a>
  <a class="item" href="predict_time_roi_2026.html">
    <span class="item-icon">⏱️</span>
    <span class="item-label">予測ROI（予想時点オッズ基準）</span>
    <span class="item-arrow">›</span>
  </a>
</div>

<footer>
  <p>keiba-dragon / horse_racing_ai &nbsp;|&nbsp; <a href="https://github.com/keiba-dragon/horse_racing_ai">GitHub</a></p>
</footer>
</body>
</html>'''

    index_path = os.path.join(docs_dir, 'index.html')
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(html)


if __name__ == '__main__':
    main()
