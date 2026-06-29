# coding: utf-8
"""
make_newspaper.py v2 — 競馬AI 詳細新聞生成

新設計:
  - 買い目サマリーを冒頭に大きく表示
  - 各レース: 特徴量ヒートマップ（レース内パーセンタイル色分け）+ NaN一覧
"""
import os, sys, re, pickle, argparse, time, urllib.request
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _decode_html(r) -> str:
    raw = r.read()
    ct = r.headers.get('Content-Type', '')
    if 'euc-jp' in ct.lower():
        return raw.decode('euc-jp', errors='replace')
    try:
        return raw.decode('utf-8')
    except UnicodeDecodeError:
        return raw.decode('euc-jp', errors='replace')

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


VENUE_LETTER_TO_CODE = {
    '東': '05', '中': '06', '中京': '07', '名': '07',
    '京': '08', '阪': '09', '新': '04', '福': '03',
    '函': '02', '札': '01', '小': '10',
}


_MARK_RE = re.compile(r'[☆▲△▼○●◎◇◆★]')
_DOT_RE  = re.compile(r'[．.]')
_STABLE_RE = re.compile(r'^(栗東|美浦)')
def _norm_name(s): return _MARK_RE.sub('', str(s)).strip()
def _norm_jkn(s):
    import unicodedata
    return _DOT_RE.sub('', unicodedata.normalize('NFKC', _norm_name(s)))
def _norm_trainer(s):
    """調教師名の栗東/美浦プレフィックスを除去して略称を返す"""
    return _STABLE_RE.sub('', _norm_name(s))

def _extract_venue(kaikai):
    m = re.search(r'\d+([^\d]+)', str(kaikai))
    return m.group(1) if m else str(kaikai)


def patch_jockey_stats(result_df, card_df, data_file):
    """parquetから騎手・調教師の直近統計をresult_dfのNaN列に補完する。"""
    JOCKEY_STAT_COLS = [
        '騎手コース_r100_勝率', '騎手馬場_r100_勝率', '騎手距離_r100_勝率',
        '騎手会場_r100_勝率', '騎手_r200_勝率', '騎手_r200_複勝率',
        '騎手コース_r100_複勝率', '騎手馬場_r100_複勝率', '騎手距離_r100_複勝率',
        '騎手脚質_r100_勝率', '騎手脚質_r100_複勝率', '騎手_平均着順',
    ]
    TRAINER_STAT_COLS = [
        '調教師_r200_勝率', '調教師_r200_複勝率',
        '調教師コース_r100_勝率', '調教師コース_r100_複勝率',
    ]
    target_cols = [c for c in JOCKEY_STAT_COLS if c in result_df.columns]
    trainer_cols = [c for c in TRAINER_STAT_COLS if c in result_df.columns]
    if not target_cols and not trainer_cols:
        return result_df

    # card_dfから 馬名S → 騎手略称 マッピング
    horse_jkn_map = {}
    if card_df is not None and not card_df.empty and '馬名S' in card_df.columns and '騎手' in card_df.columns:
        for _, cr in card_df.drop_duplicates('馬名S').iterrows():
            horse_jkn_map[str(cr['馬名S'])] = _norm_name(cr.get('騎手', ''))

    horse_col = '馬名S' if '馬名S' in result_df.columns else None
    result_df['_jkn_short'] = result_df[horse_col].map(horse_jkn_map).fillna('') if horse_col else ''
    if '開催' in result_df.columns and '芝・ダ' in result_df.columns:
        result_df['_kosu'] = (result_df['開催'].apply(_extract_venue)
                              + '_' + result_df['芝・ダ'].astype(str).str.strip())
    else:
        result_df['_kosu'] = ''

    today_shorts = set(result_df['_jkn_short'].dropna()) - {'', 'nan'}
    if not today_shorts:
        result_df.drop(columns=['_jkn_short', '_kosu'], errors='ignore', inplace=True)
        return result_df

    # parquetに存在する列だけ読み込む
    base_cols = ['騎手', '今回_コース種別', '日付_num']
    valid_stat_cols = []
    for c in target_cols:
        try:
            pd.read_parquet(data_file, columns=[c])
            valid_stat_cols.append(c)
        except Exception:
            pass
    if not valid_stat_cols:
        result_df.drop(columns=['_jkn_short', '_kosu'], errors='ignore', inplace=True)
        return result_df
    try:
        pq = pd.read_parquet(data_file, columns=base_cols + valid_stat_cols)
    except Exception as e:
        print(f'[WARN] 騎手統計parquet読み込み失敗: {e}')
        result_df.drop(columns=['_jkn_short', '_kosu'], errors='ignore', inplace=True)
        return result_df
    target_cols = valid_stat_cols

    pq['_jkn_full'] = pq['騎手'].apply(_norm_name)

    # 略称 → 全名 のマッピング（双方向prefix + ドット除去正規化 + 最多レース数で決定）
    all_pq_fullnames = pq['_jkn_full'].value_counts()
    # 統計が存在する行数（NaN以外）を優先スコアとして事前計算
    pq_stat_counts = (pq[target_cols].notna().any(axis=1)
                      .groupby(pq['_jkn_full']).sum())
    # 正規化済みparquet名→元名 の逆引き（Ｍ．デム→Mデム等）
    pq_norm_map = {_norm_jkn(fn): fn for fn in all_pq_fullnames.index}
    def _char_overlap(short, full):
        """shortの全文字がfullに含まれる割合（略称が全名の部分集合かを測る）"""
        return sum(1 for c in short if c in full) / len(short) if short else 0

    short_to_full = {}
    for short in today_shorts:
        short_n = _norm_jkn(short)
        # 候補1: 双方向prefix（正規化後）
        candidates = {fn: cnt for fn, cnt in all_pq_fullnames.items()
                      if _norm_jkn(fn).startswith(short_n) or short_n.startswith(_norm_jkn(fn))}
        # 候補2: 姓2文字一致 + 全文字包含スコア1.00（略称が全名の部分集合）
        if len(short) >= 2:
            surname = short[:2]
            for fn, cnt in all_pq_fullnames.items():
                if fn.startswith(surname) and _char_overlap(short, fn) >= 1.0:
                    candidates.setdefault(fn, cnt)
        if candidates:
            # 統計データ行数 → 総行数の順で最良候補を選択
            short_to_full[short] = max(
                candidates.items(),
                key=lambda x: (pq_stat_counts.get(x[0], 0), x[1])
            )[0]

    # 全名でフィルタ
    target_fullnames = set(short_to_full.values())
    pq_f = pq[pq['_jkn_full'].isin(target_fullnames)].sort_values('日付_num', ascending=False)
    # 略称→全名の逆引き
    full_to_short = {v: k for k, v in short_to_full.items()}

    # (short, kosu) → {stat: val}  最新行からlookup
    stats_map = {}
    for fullname in target_fullnames:
        short = full_to_short.get(fullname, fullname)
        jrows = pq_f[pq_f['_jkn_full'] == fullname]
        if jrows.empty:
            continue
        for cosu in result_df.loc[result_df['_jkn_short'] == short, '_kosu'].dropna().unique():
            # 優先順: 完全一致 → 同surface → 全行（データがある行が見つかるまで拡大）
            surf = cosu.split('_')[-1] if '_' in cosu else cosu
            for candidate_rows in [
                jrows[jrows['今回_コース種別'] == cosu],
                jrows[jrows['今回_コース種別'].str.endswith('_' + surf, na=False)],
                jrows,
            ]:
                entry = {}
                for _, rw in candidate_rows.iterrows():
                    entry = {c: float(rw[c]) for c in target_cols if pd.notna(rw.get(c))}
                    if entry:
                        break
                if entry:
                    stats_map[(short, cosu)] = entry
                    break

    # NaN補完
    filled = 0
    for idx, row in result_df.iterrows():
        short = row['_jkn_short'] if '_jkn_short' in row else ''
        cosu  = row['_kosu'] if '_kosu' in row else ''
        entry = stats_map.get((short, cosu)) or stats_map.get((short, '')) or {}
        for c, v in entry.items():
            if pd.isna(result_df.at[idx, c]):
                result_df.at[idx, c] = v
                filled += 1

    matched = sum(1 for s in today_shorts if s in short_to_full)
    print(f'騎手統計補完: {filled}セル補完 / 略称マッチ {matched}/{len(today_shorts)}人 / {len(stats_map)}ペア')

    # ── 調教師統計補完 ──────────────────────────────────────────
    if trainer_cols:
        try:
            tr_pq = pd.read_parquet(data_file, columns=['調教師', '今回_コース種別', '日付_num'] + trainer_cols)
            tr_pq['_tr_full'] = tr_pq['調教師'].apply(_norm_name)
            tr_counts = tr_pq['_tr_full'].value_counts()
            tr_stat_counts = tr_pq[trainer_cols].notna().any(axis=1).groupby(tr_pq['_tr_full']).sum()

            # card_dfから 馬名S → 調教師略称（栗東/美浦プレフィックス除去）
            horse_tr_map = {}
            if card_df is not None and '調教師' in card_df.columns:
                for _, cr in card_df.drop_duplicates('馬名S').iterrows():
                    horse_tr_map[str(cr['馬名S'])] = _norm_trainer(cr.get('調教師', ''))
            result_df['_tr_short'] = result_df['馬名S'].map(horse_tr_map).fillna('') if '馬名S' in result_df.columns else ''

            today_tr = set(result_df['_tr_short'].dropna()) - {'', 'nan'}
            tr_short_to_full = {}
            for short in today_tr:
                short_n = _norm_jkn(short)
                cands = {fn: cnt for fn, cnt in tr_counts.items()
                         if _norm_jkn(fn).startswith(short_n) or short_n.startswith(_norm_jkn(fn))}
                if len(short) >= 2:
                    for fn, cnt in tr_counts.items():
                        if fn.startswith(short[:2]) and _char_overlap(short, fn) >= 1.0:
                            cands.setdefault(fn, cnt)
                if cands:
                    tr_short_to_full[short] = max(cands.items(),
                                                   key=lambda x: (tr_stat_counts.get(x[0], 0), x[1]))[0]

            tr_fns = set(tr_short_to_full.values())
            tr_pq_f = tr_pq[tr_pq['_tr_full'].isin(tr_fns)].sort_values('日付_num', ascending=False)
            tr_f2s = {v: k for k, v in tr_short_to_full.items()}
            tr_stats_map = {}
            for fn in tr_fns:
                short = tr_f2s.get(fn, fn)
                jrows = tr_pq_f[tr_pq_f['_tr_full'] == fn]
                if jrows.empty:
                    continue
                for cosu in result_df.loc[result_df['_tr_short'] == short, '_kosu'].dropna().unique():
                    surf = cosu.split('_')[-1] if '_' in cosu else cosu
                    for crow in [jrows[jrows['今回_コース種別'] == cosu],
                                 jrows[jrows['今回_コース種別'].str.endswith('_' + surf, na=False)],
                                 jrows]:
                        entry = {}
                        for _, rw in crow.iterrows():
                            entry = {c: float(rw[c]) for c in trainer_cols if pd.notna(rw.get(c))}
                            if entry:
                                break
                        if entry:
                            tr_stats_map[(short, cosu)] = entry
                            break

            tr_filled = 0
            for idx, row in result_df.iterrows():
                short = row.get('_tr_short', '')
                cosu  = row.get('_kosu', '')
                entry = tr_stats_map.get((short, cosu)) or tr_stats_map.get((short, '')) or {}
                for c, v in entry.items():
                    if c in result_df.columns and pd.isna(result_df.at[idx, c]):
                        result_df.at[idx, c] = v
                        tr_filled += 1
            print(f'調教師統計補完: {tr_filled}セル補完 / {len(tr_stats_map)}ペア')
        except Exception as e:
            print(f'[WARN] 調教師統計補完失敗: {e}')

    result_df.drop(columns=['_jkn_short', '_kosu', '_tr_short'], errors='ignore', inplace=True)
    return result_df


def _get_race_ids(tgt_date: str) -> list:
    """netkeibaからレースIDリスト取得（odds/weight/result で共有）"""
    full_date = ('20' + str(tgt_date)) if len(str(tgt_date)) == 6 else str(tgt_date)
    try:
        req = urllib.request.Request(
            f'https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={full_date}',
            headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as r:
            html = _decode_html(r)
        return list(dict.fromkeys(re.findall(r'race_id=(\d{12})', html)))
    except Exception as e:
        print(f'[WARN] レースID取得失敗 {e}')
        return []


def fetch_live_odds(race_ids: list) -> dict:
    """netkeibaのAPIから単勝オッズ取得。
    Returns: {(venue_code_2char, r_num_int): {umaban_str_02d: odds_float}}
    """
    import json
    odds_by_race = {}   # {(venue_code, r_num_int): {umaban_02d: odds_float}}
    total_horses = 0
    for race_id in race_ids:
        venue_code = race_id[4:6]
        r_num_int  = int(race_id[10:12])
        try:
            api_url = (f'https://race.netkeiba.com/api/api_get_jra_odds.html'
                       f'?race_id={race_id}&type=1&action=init')
            req = urllib.request.Request(api_url, headers={
                'User-Agent': 'Mozilla/5.0',
                'Referer': f'https://race.netkeiba.com/odds/index.html?race_id={race_id}',
            })
            with urllib.request.urlopen(req, timeout=8) as r:
                data = json.loads(r.read().decode('utf-8', errors='replace'))
            raw = data.get('data', {}).get('odds', {}).get('1', {})
            race_odds = {}
            for uma_s, vals in raw.items():
                try:
                    v = float(vals[0])
                    if v > 0:
                        race_odds[uma_s.zfill(2)] = v
                except (ValueError, IndexError, TypeError):
                    pass
            if race_odds:
                odds_by_race[(venue_code, r_num_int)] = race_odds
                total_horses += len(race_odds)
            time.sleep(0.15)
        except Exception:
            continue

    print(f'オッズ取得: {total_horses}頭 / {len(odds_by_race)}R ({len(race_ids)}R中)')
    return odds_by_race


def fetch_horse_weights(race_ids: list) -> dict:
    """shutuba.htmlから馬体重を取得。
    Returns: {(venue_code, r_num_int): {umaban_02d: '500(+2)'}}
    """
    result = {}
    n_filled = 0
    row_pat = re.compile(r'<tr[^>]*class="[^"]*HorseList[^"]*"[^>]*>(.*?)</tr>', re.DOTALL)
    uma_pat = re.compile(r'class="Umaban\d*[^"]*"[^>]*>\s*(\d+)', re.DOTALL)
    wt_pat  = re.compile(r'class="[^"]*Weight[^"]*"[^>]*>(.*?)</td>', re.DOTALL)
    for race_id in race_ids:
        venue_code = race_id[4:6]
        r_num_int  = int(race_id[10:12])
        try:
            req = urllib.request.Request(
                f'https://race.netkeiba.com/race/shutuba.html?race_id={race_id}',
                headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=10) as r:
                html = _decode_html(r)
            race_wt = {}
            for m in row_pat.finditer(html):
                row = m.group(1)
                u_m = uma_pat.search(row)
                w_m = wt_pat.search(row)
                if u_m and w_m:
                    uma_s  = u_m.group(1).zfill(2)
                    wt_raw = re.sub(r'<[^>]+>', '', w_m.group(1)).strip()
                    if re.search(r'\d{3}', wt_raw):
                        race_wt[uma_s] = wt_raw
            if race_wt:
                result[(venue_code, r_num_int)] = race_wt
                n_filled += len(race_wt)
            time.sleep(0.2)
        except Exception:
            continue
    print(f'馬体重取得: {n_filled}頭 / {len(result)}R')
    return result


def fetch_race_results(race_ids: list) -> dict:
    """result.htmlからレース結果（着順・払戻金・オッズ）を取得。未確定レースはスキップ。
    Returns: {(venue_code, r_num_int): {
        'order':   {umaban_02d: actual_rank_int},
        'tansho':  [(umaban_02d, payout_int)],
        'fukusho': [(umaban_02d, payout_int)],
        'odds':    {umaban_02d: odds_float},
    }}
    """
    results = {}
    n_done  = 0
    row_pat   = re.compile(r'<tr[^>]*class="[^"]*HorseList[^"]*"[^>]*>(.*?)</tr>', re.DOTALL)
    jyuni_pat = re.compile(r'class="Result_Num"[^>]*>(.*?)</td>', re.DOTALL)
    uma_pat   = re.compile(r'class="Num Txt_C"[^>]*>(.*?)</td>', re.DOTALL)
    odds_pat  = re.compile(r'class="Odds\s+Txt_C"[^>]*>\s*([\d.]+)', re.DOTALL)
    for race_id in race_ids:
        venue_code = race_id[4:6]
        r_num_int  = int(race_id[10:12])
        try:
            req = urllib.request.Request(
                f'https://race.netkeiba.com/race/result.html?race_id={race_id}',
                headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=10) as r:
                html = _decode_html(r)
            # 確定チェック
            if 'Result_Num' not in html:
                continue
            # 着順・オッズ → 馬番 マッピング
            order = {}
            race_odds = {}
            for m in row_pat.finditer(html):
                row = m.group(1)
                j_m = jyuni_pat.search(row)
                u_m = uma_pat.search(row)
                o_m = odds_pat.search(row)
                if u_m:
                    uma_s2 = re.sub(r'<[^>]+>', '', u_m.group(1)).strip()
                    if uma_s2.isdigit():
                        uma_key = uma_s2.zfill(2)
                        if j_m:
                            rank_s = re.sub(r'<[^>]+>', '', j_m.group(1)).strip()
                            if rank_s.isdigit():
                                order[uma_key] = int(rank_s)
                        if o_m:
                            try:
                                race_odds[uma_key] = float(o_m.group(1))
                            except ValueError:
                                pass
            if not order:
                continue
            # 払戻: Tansho / Fukusho 行
            tansho, fukusho = [], []
            for tr_cls, lst in [('Tansho', tansho), ('Fukusho', fukusho)]:
                for tr_m in re.finditer(f'<tr class="{tr_cls}"[^>]*>(.*?)</tr>', html, re.DOTALL):
                    tr_html = tr_m.group(1)
                    res_m  = re.search(r'class="Result"[^>]*>(.*?)</td>', tr_html, re.DOTALL)
                    pay_m  = re.search(r'class="Payout"[^>]*>(.*?)</td>', tr_html, re.DOTALL)
                    if not (res_m and pay_m):
                        continue
                    uma_nums = re.findall(r'<span>(\d+)</span>', res_m.group(1))
                    pay_vals = [int(re.sub(r'[^\d]', '', v))
                                for v in re.findall(r'(\d[\d,]*円)', pay_m.group(1))
                                if re.sub(r'[^\d]', '', v)]
                    for uma_n, pay_v in zip(uma_nums, pay_vals if pay_vals else [None]):
                        if pay_v:
                            lst.append((uma_n.zfill(2), pay_v))
            results[(venue_code, r_num_int)] = {
                'order': order, 'tansho': tansho, 'fukusho': fukusho, 'odds': race_odds}
            n_done += 1
            time.sleep(0.2)
        except Exception:
            continue
    if n_done:
        print(f'レース結果取得: {n_done}R確定')
    return results


def make_newspaper(date_str=None):
    from datetime import datetime
    generated_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
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

    # タイム指数の異常値を修正（着順_num=0の行由来: 走破タイム=0 → 偏差値500超え）
    _ti_cols = [c for c in result.columns if 'タイム指数' in c and 'slope' not in c and '加速度' not in c]
    for _c in _ti_cols:
        _s = pd.to_numeric(result[_c], errors='coerce')
        result[_c] = _s.where(_s.between(0, 130), other=np.nan)

    # 馬場状態の異常値を修正（JVLinkコード10/50/90等 + 截断文字列'稍'/'不' が混入）
    # _BABA_NUM_MAP: 10=良(芝外), 50=良(芝内), 70=重(芝内), 80=不良(芝内), 90=不明→良 etc.
    _ext_baba_str  = {'良': 0, '稍重': 1, '稍': 1, '重': 2, '不良': 3, '不': 3}
    _ext_baba_num  = {0:0, 1:0, 2:1, 3:2, 4:3, 5:0, 6:1, 7:2, 8:3, 9:0,
                      10:0, 20:1, 30:2, 40:3, 50:0, 60:1, 70:2, 80:3, 90:0}
    for _c in [c for c in result.columns if '馬場状態' in c and '_isnan' not in c]:
        _raw = result[_c]
        _via_str = _raw.map(_ext_baba_str)
        _via_num = pd.to_numeric(_raw, errors='coerce').round().astype('Int64').map(_ext_baba_num)
        _merged = _via_str.combine_first(_via_num)
        _s = pd.to_numeric(_merged, errors='coerce')
        result[_c] = _s.where(_s.between(0, 3), other=np.nan)

    # ── モデル読み込み（的中率最大化モデル）────────────────────────
    model_path = os.path.join(BASE_DIR, 'models', 'hitrate_model.pkl')
    acc_model  = pickle.load(open(model_path, 'rb'))
    seg_feats  = {k: v['feat_cols'] for k, v in acc_model.items()}

    # 印のz-score閾値
    # ◎: z<1.2 (混戦でモデルが優勢) 全OOS +37.8%
    # ○: 1.2<=z<1.5                 全OOS  +7.1%
    # ▲: z>=1.5 (モデルが強く確信=人気馬) 全OOS -4.5% → 買わない
    Z_MARU   = 1.2
    Z_SANKAKU = 1.5

    # ── 騎手会場_r100_勝率: 騎手コース_r100_勝率で代替 ──────────────
    # 予測パイプラインはjockey名列を持たないため parquet照合不可
    # 騎手コース勝率（同コース・同馬場条件）を近似値として使用
    if '騎手会場_r100_勝率' not in result.columns and '騎手コース_r100_勝率' in result.columns:
        result['騎手会場_r100_勝率'] = result['騎手コース_r100_勝率']
        print('騎手会場_r100_勝率 ← 騎手コース_r100_勝率 で代替')

    # ── netkeiba レースID取得 → オッズ・体重・結果を一括フェッチ ─────
    race_ids     = _get_race_ids(tgt_date)
    live_odds    = fetch_live_odds(race_ids)
    horse_weights = fetch_horse_weights(race_ids)
    race_results  = fetch_race_results(race_ids)

    # ── カード情報（騎手・オッズ）────────────────────────────────
    # live_oddsはAPIから取得済みの {(venue_code, r_num_int): {umaban_02d: odds_float}}
    # オッズはレース別馬番ベースで後から注入するため、ここでは騎手のみ保持
    card_map = {}
    if not card_df.empty and '馬名S' in card_df.columns:
        for _, cr in card_df.drop_duplicates('馬名S').iterrows():
            horse = cr['馬名S']
            card_map[horse] = {
                '騎手':     cr.get('騎手', cr.get('dc_騎手', '')),
                '単勝オッズ': cr.get('単勝オッズ', cr.get('単オッズ', '')),
            }

    # ── 騎手統計 NaN 補完 ─────────────────────────────────────────
    data_file = os.path.join(BASE_DIR, 'data', 'processed', 'all_venues_features.parquet')
    result = patch_jockey_stats(result.copy(), card_df, data_file)

    # ── 馬体重 NaN 補完（horse_weightsから）──────────────────────────
    bango_col_r = 'dc_馬番' if 'dc_馬番' in result.columns else ('馬番' if '馬番' in result.columns else None)
    if horse_weights and bango_col_r and '開催' in result.columns and 'Ｒ' in result.columns:
        n_wt_filled = 0
        for idx, row in result.iterrows():
            if pd.notna(row.get('馬体重')):
                continue
            kaikai = str(row.get('開催', ''))
            try:
                r_num_int2 = int(float(row.get('Ｒ', 0)))
                uma_s2 = str(int(float(row.get(bango_col_r, 0)))).zfill(2)
            except (ValueError, TypeError):
                continue
            _vm2 = re.search(r'[^\d]+', kaikai)
            _vcode2 = VENUE_LETTER_TO_CODE.get(_vm2.group().strip() if _vm2 else '', '')
            wt_str2 = horse_weights.get((_vcode2, r_num_int2), {}).get(uma_s2, '')
            m_wt = re.match(r'(\d{3,})\(([+\-]?\d+)\)', str(wt_str2))
            if m_wt:
                result.at[idx, '馬体重'] = int(m_wt.group(1))
                if '馬体重増減' in result.columns:
                    result.at[idx, '馬体重増減'] = int(m_wt.group(2))
                n_wt_filled += 1
        if n_wt_filled:
            print(f'馬体重補完（horse_weights）: {n_wt_filled}頭')

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
                X_sc = scaler.transform(X)
                scores = X_sc @ coef
                contribs_arr = X_sc * coef  # shape: (n_horses, n_feats)
            except Exception:
                X_sc = np.zeros_like(X)
                scores = np.zeros(len(grp))
                contribs_arr = np.zeros_like(X)
            grp = grp.copy()
            grp['_acc_score'] = scores
            grp['_contrib_dict'] = pd.array(
                [{feat_cols[fi]: float(contribs_arr[ri, fi]) for fi in range(len(feat_cols))}
                 for ri in range(len(grp))],
                dtype=object
            )
            grp['_sort_rank'] = grp['_acc_score'].rank(ascending=False, method='first')
            # accuracy_model の isotonic calibration で確率を計算
            _e = np.exp(scores - scores.max())
            _p_raw = _e / _e.sum()
            _iso = art.get('isotonic')
            if _iso is not None:
                _p_calib = np.clip(_iso.predict(_p_raw), 0.001, 0.999)
            else:
                _p_calib = _p_raw
            grp['_acc_prob'] = _p_calib
            # coef 符号から「低い方が有利か」を記録（feat_rank の方向に使う）
            coef_asc_map = {feat_cols[fi]: bool(coef[fi] < 0) for fi in range(len(feat_cols))}
        else:
            grp['_acc_score']    = np.nan
            grp['_sort_rank']    = pd.Series(np.nan, index=grp.index)
            grp['_acc_prob']     = np.nan
            grp['_contrib_dict'] = pd.array([{} for _ in range(len(grp))], dtype=object)
            coef_asc_map         = {}

        # ── per-race z-score → 印を付与 ──────────────────────────────
        # z = (score_rank1 - mean) / std within race
        # 低z（混戦）= 市場も迷ってる = 高オッズ = ROI良
        _sc = grp['_acc_score'].values
        _mean, _std = _sc.mean(), _sc.std(ddof=0)
        grp['_z'] = (_sc - _mean) / _std if _std > 0 else np.zeros(len(_sc))

        def _mark(row):
            if row.get('_sort_rank') != 1:
                return ''
            z = row.get('_z', np.nan)
            if pd.isna(z):   return ''
            if z < Z_MARU:   return '◎'
            if z < Z_SANKAKU: return '○'
            return '▲'
        grp['_mark'] = [_mark(r) for _, r in grp.iterrows()]
        # 表示は馬番順（AI順位は列に保持）
        bango_col = 'dc_馬番' if 'dc_馬番' in grp.columns else ('馬番' if '馬番' in grp.columns else None)
        if bango_col:
            grp = grp.assign(_bango_num=pd.to_numeric(grp[bango_col], errors='coerce')).sort_values('_bango_num', na_position='last').drop(columns=['_bango_num'])
        else:
            grp = grp.sort_values('_sort_rank', na_position='last')

        race_data.append(dict(
            grp=grp, kaikai=kaikai, r_num=r_num, race_name=race_name,
            kyori_raw=kyori_raw, shiba_da=shiba_da, dist_m=dist_m,
            surf=surf, seg_key=seg_key, feats=feats, coef_asc_map=coef_asc_map
        ))

    # ── 日付表示 ────────────────────────────────────────────────
    d_str = str(tgt_date)
    date_disp = f'20{d_str[:2]}/{d_str[2:4]}/{d_str[4:6]}' if len(d_str) == 6 else str(tgt_date)

    # ── CSS ──────────────────────────────────────────────────────
    css = """<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body { font-family: 'Yu Gothic', 'Hiragino Sans', 'Meiryo', sans-serif;
               font-size: 18px; background: #c8d0d8; color: #222; }
  .app { background: #eef1f5; min-height: 100vh; overflow-x: hidden; }

  /* ── トップバー ─────────────────────────────────── */
  .topbar { background: #1a237e; color: #fff; padding: 4px 12px;
            display: flex; gap: 12px; align-items: center; font-size: 16px; }
  .topbar a { color: #90caf9; text-decoration: none; }
  .topbar a:hover { text-decoration: underline; }

  /* ── ページタイトル ──────────────────────────────── */
  .page-title { font-size: 20px; font-weight: bold; padding: 8px 14px;
                background: #1a252f; color: white;
                display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
  .page-title .subtitle { font-size: 13px; color: #aaa; font-weight: normal; }
  .report-btn { background: #1a237e; color: #fff; text-decoration: none;
                font-size: 14px; font-weight: 600; padding: 3px 10px;
                border-radius: 4px; border: 1px solid rgba(255,255,255,.3); }
  .report-btn:hover { opacity: .85; }

  /* ── タブバー ───────────────────────────────────── */
  .tab-bar { display: flex; background: #fff; border-bottom: 2px solid #c8d0d8;
             position: sticky; top: 0; z-index: 50;
             box-shadow: 0 1px 4px rgba(0,0,0,0.08); overflow-x: auto; }
  .tab-btn { padding: 8px 16px; border: none; background: none; cursor: pointer;
             font-size: 16px; font-weight: 600; color: #666; white-space: nowrap;
             border-bottom: 3px solid transparent; margin-bottom: -2px; }
  .tab-btn:hover { color: #1a237e; background: #f0f4ff; }
  .tab-btn.active { color: #1a237e; border-bottom-color: #1a237e; background: #f0f4ff; }
  .tab-btn .cnt { font-size: 13px; color: #aaa; margin-left: 3px; }
  .tab-btn.active .cnt { color: #5c6bc0; }

  /* ── タブコンテンツ ──────────────────────────────── */
  .tab-pane { display: none; padding: 6px 4px 24px; overflow: hidden; max-width: 100%; }
  .tab-pane.active { display: block; }

  /* ── 買い目セクション ────────────────────────────── */
  .buy-section { background: white; border-radius: 8px; padding: 12px 14px;
                 margin-bottom: 10px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }
  .section-title { font-size: 19px; font-weight: bold; margin: 0 0 8px;
                   padding-bottom: 5px; border-bottom: 2px solid currentColor; }
  .section-title.buy   { color: #c0392b; }
  .section-title.watch { color: #e67e22; margin-top: 12px; }
  .buy-grid { display: flex; flex-wrap: wrap; gap: 8px; }
  .buy-card { border-radius: 7px; padding: 8px 12px; min-width: 160px; }
  .buy-card.confirmed  { background: #fde8e8; border: 2px solid #c0392b; }
  .buy-card.watch-card { background: #fef9e7; border: 2px solid #e67e22; }
  .card-race  { font-size: 14px; color: #777; margin-bottom: 2px; }
  .card-horse { font-size: 20px; font-weight: bold; color: #1a252f; margin-bottom: 2px; }
  .card-meta  { font-size: 14px; color: #666; }
  .badge-buy   { display: inline-block; background: #c0392b; color: white;
                 font-size: 15px; font-weight: bold; padding: 2px 10px;
                 border-radius: 8px; margin-top: 4px; }
  .badge-watch { display: inline-block; background: #e67e22; color: white;
                 font-size: 14px; padding: 2px 10px; border-radius: 8px; margin-top: 4px; }
  .seg-chip { color: white; font-size: 12px; padding: 1px 6px;
              border-radius: 3px; vertical-align: middle; }
  .no-signal { color: #aaa; font-style: italic; font-size: 16px; }

  /* ── レースブロック ──────────────────────────────── */
  .race-block { background: white; border-radius: 6px; margin-bottom: 8px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.08); overflow: hidden; }
  .race-header { display: flex; align-items: center; gap: 6px; padding: 6px 12px;
                 background: #f7f9fb; border-left: 5px solid #888; flex-wrap: wrap; }
  .race-venue { font-size: 19px; font-weight: bold; color: #222; }
  .race-rnum  { font-size: 18px; font-weight: bold; color: #555; }
  .race-name  { font-size: 18px; font-weight: bold; flex: 1; color: #1a252f; }
  .race-seg   { color: white; font-size: 13px; padding: 2px 10px; border-radius: 10px; }
  .race-dist  { font-size: 15px; color: #888; }
  .n-horses   { font-size: 13px; color: #aaa; }
  .seg-report-link { margin-left: auto; font-size: 13px; color: #1a237e;
                     text-decoration: none; padding: 2px 8px; border-radius: 3px;
                     border: 1px solid #c5cae9; background: #e8eaf6; white-space: nowrap; }
  .seg-report-link:hover { background: #c5cae9; }

  /* NaN Alert */
  .nan-alert { padding: 4px 12px; background: #fff8f8;
               border-top: 1px solid #fcc; font-size: 13px; }
  .nan-chip { display: inline-block; margin: 1px 3px; padding: 1px 6px;
              border-radius: 3px; font-weight: bold; }
  .nan-hi  { background: #c0392b; color: white; }
  .nan-mid { background: #e67e22; color: white; }
  .nan-lo  { background: #f9e79f; color: #555; }

  /* Race Table */
  .table-wrap { overflow-x: auto; }
  table.race-table { border-collapse: collapse; width: 100%; font-size: 16px; table-layout: fixed; }
  table.race-table th { background: #2c3e50; color: white; padding: 4px 5px;
                        text-align: center; border: 1px solid #222;
                        font-size: 15px; font-weight: bold; overflow: hidden; }
  table.race-table td { padding: 4px 5px; border: 1px solid #e0e0e0;
                        text-align: center; overflow: hidden; }
  .row-buy td  { background: #fde8e8 !important; outline: 2px solid #c0392b; }
  .row-maru td { background: #fff3e0 !important; outline: 2px solid #e67e22; }
  .row-r1 td   { background: #fef5f5 !important; }
  .row-r2 td   { background: #fef9ee !important; }
  .row-r3 td   { background: #f3faf5 !important; }

  .td-rank  { font-weight: bold; width: 32px; min-width: 32px; max-width: 32px; }
  .td-horse { text-align: left !important; font-weight: bold; font-size: 18px;
              overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .td-jky   { font-size: 15px; width: 52px; min-width: 52px; max-width: 52px;
              overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .td-odds  { width: 56px; min-width: 56px; max-width: 56px; }
  .td-prob  { width: 56px; min-width: 56px; max-width: 56px; color: #16a085; font-weight: bold; }
  .td-buy   { background: #c0392b !important; color: white !important; font-weight: bold; width: 36px; min-width: 36px; max-width: 36px; }
  .td-watch { background: #e67e22 !important; color: white !important; width: 36px; }
  .td-nan   { background: #ffe0e0 !important; color: #c0392b; font-weight: bold; font-size: 13px; }
  .td-none  { color: #ccc; }

  /* ── 詳細展開パネル ─────────────────────────────── */
  .detail-row td { padding: 10px 12px; background: #f9f9f9 !important;
                   border: 1px solid #e8e8e8; outline: none; }
  .detail-panel { display: flex; flex-wrap: wrap; gap: 8px; }
  .feat-chip { display: inline-flex; flex-direction: column; align-items: center;
               padding: 8px 12px; border-radius: 6px; font-size: 18px;
               min-width: 80px; border: 1px solid rgba(0,0,0,0.12);
               cursor: default; }
  .feat-name { font-size: 13px; color: rgba(0,0,0,0.55); line-height: 1.2; margin-bottom: 3px; }
  .feat-val  { font-weight: bold; font-size: 20px; line-height: 1.2; }
  .feat-pt   { font-size: 13px; color: rgba(0,0,0,0.45); line-height: 1; margin-top: 3px; }
  .detail-hint { font-size: 14px; color: #bbb; margin-left: 3px; transition: color .15s; }
  tr.expandable:hover td { background: #fafafa; }
  tr.expandable:hover .detail-hint { color: #777; }
  /* ── 馬体重チップ ─── */
  .wt-chip { font-size: 13px; color: #888; font-weight: normal; }
  /* ── 実着順列 ─── */
  .td-jyuni { width: 28px; min-width: 28px; max-width: 28px; font-weight: bold; font-size: 16px; text-align: center; }
  .ar-1 { background: #ffd700 !important; color: #5a3000 !important; }
  .ar-2 { background: #c0c0c0 !important; color: #333 !important; }
  .ar-3 { background: #cd7f32 !important; color: #fff !important; }
  .ar-n { color: #aaa; }
  /* ── 払戻バナー ─── */
  .result-banner { background: #1b5e20; color: #fff; padding: 5px 12px;
                   font-size: 15px; font-weight: bold; }

  /* ── レース内側タブ ─────────────────────────────── */
  .race-tab-bar { display: flex; flex-wrap: wrap; gap: 4px; padding: 8px 10px 0;
                  background: #f0f4f8; border-bottom: 2px solid #d0d8e0; }
  .race-tab-btn { padding: 5px 14px; border: none; background: #e0e8f0;
                  border-radius: 5px 5px 0 0; cursor: pointer;
                  font-size: 16px; font-weight: 600; color: #555; margin-bottom: -2px; }
  .race-tab-btn:hover { background: #d0dcea; color: #1a237e; }
  .race-tab-btn.active { background: #fff; color: #1a237e; border: 2px solid #d0d8e0;
                          border-bottom-color: #fff; }
  .race-tab-body { padding: 6px 6px 12px; background: #f0f4f8; overflow: hidden; }
  .race-tab-pane { display: none; }
  .race-tab-pane.active { display: block; }

  .footer { font-size: 12px; color: #aaa; text-align: right; padding: 6px 10px 16px; }
</style>"""

    # 買い判定パラメータ
    EV_MIN = 1.0   # AIの勝率 × 単勝オッズ ≥ 1.0 (期待値プラス) のみ買い

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

        _vm1 = re.search(r'[^\d]+', rd['kaikai'])
        _vletter1 = _vm1.group().strip() if _vm1 else ''
        _vcode1 = VENUE_LETTER_TO_CODE.get(_vletter1, '')
        try:
            _rn_int1 = int(rd['r_num'])
        except (ValueError, TypeError):
            _rn_int1 = 0
        race_live_odds1 = live_odds.get((_vcode1, _rn_int1), {})

        for _, r in rd['grp'].iterrows():
            sort_rank1 = r.get('_sort_rank')
            horse  = r.get('馬名S', '')
            acc_prob1 = r.get('_acc_prob', np.nan)

            ci = card_map.get(horse, {})
            bango1 = r.get('dc_馬番', r.get('馬番', ''))
            try:
                uma_s1 = str(int(float(bango1))).zfill(2)
            except (ValueError, TypeError):
                uma_s1 = '00'
            ov = race_live_odds1.get(uma_s1) or ci.get('単勝オッズ', r.get('単勝オッズ', ''))
            try:
                odds_num1 = float(ov) if ov not in ('', None) and str(ov) not in ('nan', '') else np.nan
            except (ValueError, TypeError):
                odds_num1 = np.nan
            odds_s = f'{odds_num1:.1f}倍' if pd.notna(odds_num1) else '未発表'
            ev1 = acc_prob1 * odds_num1 if pd.notna(acc_prob1) and pd.notna(odds_num1) else np.nan
            mark1 = r.get('_mark', '')
            c_buy  = (sort_rank1 == 1) and (mark1 in ('◎', '○'))
            jockey = str(ci.get('騎手', r.get('dc_騎手', r.get('騎手', '')))).strip()
            prob_disp = f'{acc_prob1:.1%}' if pd.notna(acc_prob1) else ''
            ev_s = f'EV {ev1:.2f}' if pd.notna(ev1) else ''

            chip = f'<span class="seg-chip" style="background:{seg_color}">{seg_lbl}</span>'
            race_lbl = f'{venue_full} {rd["r_num"]}R　{chip}'

            if c_buy:
                badge_color = '#c0392b' if mark1 == '◎' else '#e67e22'
                buy_cards.append(f'''
<div class="buy-card confirmed">
  <div class="card-race">{race_lbl}</div>
  <div class="card-horse">{horse}</div>
  <div class="card-meta">{jockey}　単勝 {odds_s}　AI {prob_disp}　{ev_s}</div>
  <span class="badge-buy" style="background:{badge_color}">{mark1} 買い</span>
</div>''')

    buy_html = '<div class="buy-section">'
    buy_html += '<div class="section-title buy">◎○ 本日の買い目　<small style="font-weight:normal;font-size:12px">◎混戦穴(z&lt;1.2) ○やや混戦(z&lt;1.5)</small></div>'
    if buy_cards:
        buy_html += f'<div class="buy-grid">{"".join(buy_cards)}</div>'
    else:
        buy_html += '<p class="no-signal">買いシグナルなし（オッズ未発表または条件未達）</p>'

    # watch_cards は accuracy_model 移行後は不使用

    buy_html += '</div>'

    # ═══════════════════════════════════════════════════════════
    # Section 2: レース別詳細（ヒートマップ + NaN一覧）
    # ═══════════════════════════════════════════════════════════
    from collections import defaultdict
    race_groups = defaultdict(list)   # venue_key → [html, ...]
    venue_order = []                  # 登場順の会場キー

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

        # ライブオッズ: kaikai から venue_code を抽出して該当レースを引く
        _vm = re.search(r'[^\d]+', rd['kaikai'])
        _vletter = _vm.group().strip() if _vm else ''
        _vcode = VENUE_LETTER_TO_CODE.get(_vletter, '')
        try:
            _rn_int = int(rd['r_num'])
        except (ValueError, TypeError):
            _rn_int = 0
        race_live_odds = live_odds.get((_vcode, _rn_int), {})
        race_wt_map    = horse_weights.get((_vcode, _rn_int), {})
        race_res       = race_results.get((_vcode, _rn_int))   # None or dict

        # 表示特徴量（_isnanは別扱い）
        display_feats = [f for f in feats if not f.endswith('_isnan')]
        isnan_feats   = [f for f in feats if f.endswith('_isnan')]

        # ── レース内ランク ────────────────────────────────────────
        # coef符号から方向を判断（coef<0 → 低い値が有利 → ascending=True）
        coef_asc_map = rd.get('coef_asc_map', {})
        feat_rank = {}
        n_horses_in_race = len(grp)
        for f in display_feats:
            if f in grp.columns:
                vals = pd.to_numeric(grp[f], errors='coerce')
                asc  = coef_asc_map.get(f, False)
                ranked = vals.rank(ascending=asc, method='min', na_option='keep')
                feat_rank[f] = ranked.to_dict()

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

        # 行HTML（シンプル6列 + クリックで詳細展開）
        rows = []
        vk_safe = rd['kaikai'].replace(' ', '_')
        rn_safe = rd['r_num'].replace(' ', '_')
        for hi, (_, r) in enumerate(grp.iterrows()):
            sort_rank = r.get('_sort_rank', np.nan)
            c_calib = r.get('_acc_prob')      # accuracy_model の確率
            horse   = r.get('馬名S', '')
            acc_score = r.get('_acc_score', np.nan)

            ci    = card_map.get(horse, {})
            bango = r.get('dc_馬番', r.get('馬番', ''))
            # オッズ: APIライブ → card_df → result の優先順
            try:
                uma_s = str(int(float(bango))).zfill(2)
            except (ValueError, TypeError):
                uma_s = '00'
            ov = (race_live_odds.get(uma_s)
                  or (race_res.get('odds', {}).get(uma_s) if race_res else None)
                  or ci.get('単勝オッズ', r.get('単勝オッズ', '')))
            try:
                odds_num = float(ov) if ov not in ('', None) and str(ov) not in ('nan', '') else np.nan
            except (ValueError, TypeError):
                odds_num = np.nan
            odds_s  = f'{odds_num:.1f}' if pd.notna(odds_num) else '-'
            ev = c_calib * odds_num if pd.notna(c_calib) and pd.notna(odds_num) else np.nan
            mark = r.get('_mark', '')
            c_buy = (sort_rank == 1) and (mark in ('◎', '○'))
            jockey  = str(ci.get('騎手', r.get('dc_騎手', r.get('騎手', '')))).strip()[:5]
            prob_s  = f'{c_calib:.1%}' if pd.notna(c_calib) else '-'

            try: rank_i = int(float(sort_rank))
            except: rank_i = None
            ai_rank_s = str(rank_i) if rank_i else '-'

            if mark == '◎':        row_cls = 'row-buy'
            elif mark == '○':      row_cls = 'row-maru'
            elif rank_i == 1:      row_cls = 'row-r1'
            elif rank_i == 2:      row_cls = 'row-r2'
            elif rank_i == 3:      row_cls = 'row-r3'
            else:                  row_cls = ''

            if mark == '◎':
                buy_td = '<td class="td-buy" style="color:#c0392b;font-weight:bold">◎</td>'
            elif mark == '○':
                buy_td = '<td class="td-buy" style="color:#e67e22;font-weight:bold">○</td>'
            elif mark == '▲':
                buy_td = '<td class="td-watch" style="color:#27ae60;font-weight:bold">▲</td>'
            else:
                buy_td = '<td class="td-none">-</td>'

            # 馬体重
            wt_str  = race_wt_map.get(uma_s, '')
            wt_html = f'<span class="wt-chip"> {wt_str}</span>' if wt_str else ''

            # 実際着順（結果確定後）
            actual_rank = race_res['order'].get(uma_s) if race_res else None
            if actual_rank is not None:
                ar_cls = {1: 'ar-1', 2: 'ar-2', 3: 'ar-3'}.get(actual_rank, 'ar-n')
                jyuni_td = f'<td class="td-jyuni {ar_cls}">{actual_rank}</td>'
            elif race_res:
                jyuni_td = '<td class="td-jyuni ar-n">-</td>'
            else:
                jyuni_td = ''
            n_cols = 7 if race_res else 6

            # 詳細パネル（特徴量チップ）
            detail_id = f'det-{vk_safe}-{rn_safe}-{hi}'
            contrib_dict = r.get('_contrib_dict') or {}
            if not isinstance(contrib_dict, dict):
                contrib_dict = {}
            chips = []
            for f in display_feats:
                val     = r.get(f)
                fv      = fmt_val(f, val)
                contrib = contrib_dict.get(f, np.nan)
                rank_v  = feat_rank.get(f, {}).get(r.name, np.nan)
                if fv is None:
                    bg, fc, fv_disp, pt_s = '#f0f0f0', '#aaa', 'NaN', ''
                else:
                    fv_disp = fv
                    feat_rank_s = f'{int(rank_v)}/{n_horses_in_race}位' if not pd.isna(rank_v) else ''
                    if pd.isna(contrib):
                        bg, fc, pt_s = '#f5f5f5', '#222', feat_rank_s
                    elif contrib > 0:
                        intensity = min(abs(contrib) / 0.5, 1.0)
                        bg = f'rgb(255,{int(255-100*intensity)},{int(255-155*intensity)})'
                        fc = '#222'
                        pt_s = f'+{contrib:.2f}　{feat_rank_s}' if feat_rank_s else f'+{contrib:.2f}'
                    else:
                        intensity = min(abs(contrib) / 0.5, 1.0)
                        bg = f'rgb({int(200+55*(1-intensity))},{int(210+45*(1-intensity))},255)'
                        fc = '#222'
                        pt_s = f'{contrib:.2f}　{feat_rank_s}' if feat_rank_s else f'{contrib:.2f}'
                sname = short_feat(f)
                chips.append(
                    f'<span class="feat-chip" style="background:{bg};color:{fc}">'
                    f'<span class="feat-name">{sname}</span>'
                    f'<span class="feat-val">{fv_disp}</span>'
                    f'<span class="feat-pt">{pt_s}</span>'
                    f'</span>'
                )
            detail_html = f'<div class="detail-panel">{"".join(chips)}</div>'

            rows.append(
                f'<tr class="{row_cls} expandable" onclick="toggleDetail(\'{detail_id}\')">'
                f'{jyuni_td}'
                f'<td class="td-rank">{ai_rank_s}</td>'
                f'{buy_td}'
                f'<td class="td-horse">{bango}.{horse}{wt_html}<span class="detail-hint">▾</span></td>'
                f'<td class="td-jky">{jockey}</td>'
                f'<td class="td-odds">{odds_s}</td>'
                f'<td class="td-prob">{prob_s}</td>'
                f'</tr>'
                f'<tr id="{detail_id}" class="detail-row" style="display:none">'
                f'<td colspan="{n_cols}">{detail_html}</td>'
                f'</tr>'
            )

        venue_key = rd['kaikai']
        if venue_key not in venue_order:
            venue_order.append(venue_key)

        acc_report_href = f'accuracy_model_report_20{tgt_date}.html#tab-{seg_key}' if seg_key else '#'

        # 払戻バナー（結果確定後）
        payout_banner = ''
        if race_res:
            parts = []
            for uma_s2, pay in race_res.get('tansho', []):
                parts.append(f'単勝 {int(uma_s2)}番 ¥{pay:,}')
            fk = race_res.get('fukusho', [])
            if fk:
                fk_str = ' / '.join(f'{int(u)}番 ¥{p:,}' for u, p in fk)
                parts.append(f'複勝 {fk_str}')
            if parts:
                payout_banner = f'<div class="result-banner">🏆 {"　".join(parts)}</div>'

        # テーブルヘッダー（結果列は結果確定時のみ）
        result_th = '<th>着</th>' if race_res else ''
        race_groups[venue_key].append((rd['r_num'], rd['race_name'], f'''
<div class="race-block">
  <div class="race-header" style="border-left-color:{seg_color}">
    <span class="race-venue">{venue_s}</span>
    <span class="race-rnum">{rd["r_num"]}R</span>
    <span class="race-name">{rd["race_name"]}</span>
    <span class="race-seg" style="background:{seg_color}">{seg_lbl}</span>
    <span class="race-dist">{surf}{dist_str}</span>
    <span class="n-horses">{len(grp)}頭　特徴{len(display_feats)}個</span>
    <a class="seg-report-link" href="{acc_report_href}" target="_blank">📊 モデル</a>
  </div>
  {payout_banner}
  {nan_alert_html}
  <div class="table-wrap">
  <table class="race-table">
    <thead><tr>
      {result_th}<th>順位</th><th>買い</th>
      <th style="text-align:left">馬名</th>
      <th>騎手</th><th>オッズ</th><th>AI勝率</th>
    </tr></thead>
    <tbody>{"".join(rows)}</tbody>
  </table>
  </div>
</div>'''))

    # ═══════════════════════════════════════════════════════════
    # HTML組立（タブ構成: 外=買い目/会場  内=レース別）
    # ═══════════════════════════════════════════════════════════

    def build_venue_pane(vk, races):
        """races = [(r_num, race_name, html), ...]"""
        vid = f'v-{vk.replace(" ", "_")}'
        # 内側タブボタン
        inner_btns = ''
        inner_panes = ''
        for i, (rnum, rname, rhtml) in enumerate(races):
            rid = f'{vid}-r{rnum}'
            act = 'active' if i == 0 else ''
            has_buy = 'clogit_buy' in rhtml and '◎買' in rhtml
            buy_dot = ' <span style="color:#c0392b;font-weight:bold">●</span>' if has_buy else ''
            inner_btns  += f'<button class="race-tab-btn {act}" onclick="switchRace(\'{vid}\',\'{rid}\',this)">{rnum}R{buy_dot}</button>'
            inner_panes += f'<div id="pane-{rid}" class="race-tab-pane {act}">{rhtml}</div>'
        return f'<div class="race-tab-bar">{inner_btns}</div><div class="race-tab-body">{inner_panes}</div>'

    n_buy = len(buy_cards)
    tab_buttons = f'<button class="tab-btn active" onclick="switchTab(\'buy\', this)">◎ 買い目 <span class="cnt">({n_buy}件)</span></button>'
    tab_panes   = f'<div id="pane-buy" class="tab-pane active">{buy_html}</div>'

    for vk in venue_order:
        races = race_groups[vk]
        venue_full = next((v for k, v in V_FULL.items() if k in vk), vk[:3])
        tab_id = f'v-{vk.replace(" ", "_")}'
        buy_cnt = sum(1 for _, _, h in races if '◎買' in h)
        buy_marker = f' <span style="color:#c0392b;font-size:10px">({buy_cnt}買)</span>' if buy_cnt else ''
        tab_buttons += f'<button class="tab-btn" onclick="switchTab(\'{tab_id}\', this)">{venue_full}{buy_marker} <span class="cnt">({len(races)}R)</span></button>'
        tab_panes   += f'<div id="pane-{tab_id}" class="tab-pane">{build_venue_pane(vk, races)}</div>'

    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>競馬AI新聞 {date_disp}</title>
  {css}
</head>
<body>
<div class="app">
  <div class="page-title">
    <span>🏇 競馬AI 予想新聞　{date_disp}</span>
    <span class="subtitle">{len(race_data)}レース / {len(result)}頭</span>
    <span class="subtitle" style="color:#90caf9">更新: {generated_at}</span>
    <span style="margin-left:auto;display:flex;gap:6px;flex-shrink:1;flex-wrap:wrap;justify-content:flex-end">
      <a class="report-btn" href="accuracy_model_report_20{tgt_date}.html">📊 予想</a>
      <a class="report-btn" href="model_report_20{tgt_date}.html" style="background:#1b5e20">📈 ROI</a>
    </span>
  </div>

  <div class="tab-bar">{tab_buttons}</div>

  {tab_panes}

  <div class="footer">
    ◎z&lt;1.2(混戦穴) ○z&lt;1.5(やや混戦) ▲z&gt;=1.5(人気) | accuracy_model clogit + isotonic
    <br>生成日時: {generated_at}
  </div>
</div>

<script>
function switchTab(id, btn) {{
  document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('pane-' + id).classList.add('active');
  btn.classList.add('active');
}}
function switchRace(vid, rid, btn) {{
  // 同じ会場内のレースタブだけ切替
  const pane = document.getElementById('pane-' + vid);
  pane.querySelectorAll('.race-tab-pane').forEach(p => p.classList.remove('active'));
  pane.querySelectorAll('.race-tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('pane-' + rid).classList.add('active');
  btn.classList.add('active');
}}
function toggleDetail(id) {{
  var el = document.getElementById(id);
  if (!el) return;
  var showing = el.style.display !== 'none';
  el.style.display = showing ? 'none' : '';
  // ▾ ▴ の切替
  var btn = el.previousElementSibling;
  if (btn) {{
    var hint = btn.querySelector('.detail-hint');
    if (hint) hint.textContent = showing ? '▾' : '▴';
  }}
}}
</script>
</body>
</html>"""

    # ── 出力 ──────────────────────────────────────────────────
    out_dir  = os.path.join(BASE_DIR, 'docs')
    os.makedirs(out_dir, exist_ok=True)
    fname_date = f'20{tgt_date}' if len(str(tgt_date)) == 6 else str(tgt_date)
    out_path = os.path.join(out_dir, f'newspaper_{fname_date}.html')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'HTML出力: {out_path}')

    _update_newspaper_index(out_dir)

    gdrive = r'G:\マイドライブ\競馬AI\予想レポート'
    if os.path.isdir(gdrive):
        import shutil
        gd_path = os.path.join(gdrive, f'newspaper_{fname_date}.html')
        shutil.copy2(out_path, gd_path)
        idx_src = os.path.join(out_dir, 'newspapers.html')
        if os.path.exists(idx_src):
            shutil.copy2(idx_src, os.path.join(gdrive, 'newspapers.html'))
        print(f'Gdrive出力: {gd_path}')

    return out_path


def _update_newspaper_index(out_dir: str):
    """docs/ 内の newspaper_*.html をスキャンして newspapers.html を再生成"""
    import glob as _glob
    files = sorted(_glob.glob(os.path.join(out_dir, 'newspaper_????????.html')), reverse=True)

    rows = []
    for fp in files:
        fname = os.path.basename(fp)
        ds = fname[len('newspaper_'):-len('.html')]  # e.g. 20260620
        try:
            dt = pd.Timestamp(ds)
            label = dt.strftime('%Y年%-m月%-d日') if os.name != 'nt' else dt.strftime('%Y年%#m月%#d日')
            weekday = ['月', '火', '水', '木', '金', '土', '日'][dt.weekday()]
            label = f'{label}（{weekday}）'
        except Exception:
            label = ds
        rows.append((ds, label, fname))

    items_html = '\n'.join(
        f'    <li><a href="{fname}">{label}</a></li>'
        for ds, label, fname in rows
    )

    idx_html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>競馬AI 予想新聞 一覧</title>
  <style>
    body {{ font-family: 'Segoe UI','Noto Sans JP',sans-serif; background:#f4f6f9; color:#212121; margin:0; }}
    nav {{ background:#1a1a2e; padding:0 24px; height:52px; display:flex; align-items:center; gap:16px; }}
    .nav-brand {{ color:#fff; font-weight:700; font-size:1rem; }}
    nav a {{ color:#ccc; text-decoration:none; font-size:.875rem; }}
    nav a:hover {{ color:#fff; }}
    .container {{ max-width:600px; margin:40px auto; padding:0 24px; }}
    h1 {{ font-size:1.4rem; margin-bottom:24px; color:#1a1a2e; }}
    ul {{ list-style:none; padding:0; margin:0; }}
    li {{ background:#fff; border-radius:8px; margin-bottom:10px;
          box-shadow:0 2px 6px rgba(0,0,0,.08); }}
    li a {{ display:block; padding:14px 20px; text-decoration:none; color:#1a1a2e;
            font-size:1rem; font-weight:600; border-radius:8px; transition:background .15s; }}
    li a:hover {{ background:#eef2ff; }}
    li:first-child a {{ color:#1565c0; font-size:1.05rem; }}
    .badge {{ display:inline-block; background:#1565c0; color:#fff;
              border-radius:12px; font-size:.72rem; padding:2px 8px; margin-left:8px; }}
  </style>
</head>
<body>
<nav>
  <span class="nav-brand">🏇 競馬AI v2</span>
  <a href="index.html">モデルレポート</a>
  <a href="newspapers.html" style="color:#fff;font-weight:700;">予想新聞</a>
</nav>
<div class="container">
  <h1>予想新聞 一覧</h1>
  <ul>
{items_html}
  </ul>
</div>
</body>
</html>"""

    idx_path = os.path.join(out_dir, 'newspapers.html')
    with open(idx_path, 'w', encoding='utf-8') as f:
        f.write(idx_html)
    print(f'新聞インデックス更新: {idx_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', help='日付 YYYYMMDD')
    args = parser.parse_args()
    make_newspaper(args.date)
