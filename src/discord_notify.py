# coding: utf-8
"""
Discord レース通知デーモン (clogit版)

clogit_rank=1 かつ EV>0 (モデル確率 > 市場確率×0.80) の馬を
発走10分前に Discord に通知する。

使い方:
  python src/discord_notify.py              # 今日の YYYYMMDD.cache.pkl を使用
  python src/discord_notify.py --test       # 即時テスト送信（時刻チェックなし）
  python src/discord_notify.py --ev 0.03   # EVしきい値を指定（デフォルト0.0）

設定:
  config/discord_config.json に webhook_url を記載
"""
import sys, io, os, json, pickle, re, time, argparse
from datetime import datetime, timedelta

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yahoo_odds_fetch import fetch_odds_now, save_odds_json, load_v2_card_df

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR   = os.path.join(BASE_DIR, 'data', 'raw', 'cache')
CONFIG_PATH = os.path.join(BASE_DIR, 'config', 'discord_config.json')

NOTIFY_BEFORE_MIN = 10
END_HOUR          = 17
END_MIN           = 30
ODDS_REFRESH_MIN  = 15

HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

VENUE_NAME = {
    '京': '京都', '東': '東京', '阪': '阪神', '中': '中山',
    '新': '新潟', '福': '福島', '函': '函館', '札': '札幌',
    '小': '小倉', '中京': '中京',
}
SURFACE_EMO = {'芝': '🌿', 'ダ': '🏜️'}

# JRA標準発走時刻 (レース番号 → HH:MM)
# 実際は会場・曜日で変わるが目安として使用
FALLBACK_TIMES = {
    1:  '09:55', 2:  '10:15', 3:  '10:40', 4:  '11:05',
    5:  '11:30', 6:  '12:00', 7:  '12:35', 8:  '13:10',
    9:  '13:50', 10: '14:25', 11: '15:05', 12: '15:45',
}


# ── Discord送信 ─────────────────────────────────────────────────────────

def load_webhook_url() -> str:
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f'設定ファイルが見つかりません: {CONFIG_PATH}')
    with open(CONFIG_PATH, encoding='utf-8') as f:
        cfg = json.load(f)
    url = cfg.get('webhook_url', '').strip()
    if not url or 'YOUR_WEBHOOK' in url:
        raise ValueError('discord_config.json の webhook_url を設定してください')
    return url


def send_discord(webhook_url: str, content: str = '', embeds: list = None):
    payload = {}
    if content:
        payload['content'] = content
    if embeds:
        payload['embeds'] = embeds
    r = requests.post(webhook_url, json=payload, timeout=10, verify=False)
    if r.status_code not in (200, 204):
        print(f'[Discord ERROR] {r.status_code} {r.text[:200]}', file=sys.stderr)
    return r.status_code


# ── キャッシュ読み込み ────────────────────────────────────────────────────

def load_today_cache(target_date: str) -> pd.DataFrame:
    path = os.path.join(CACHE_DIR, f'{target_date}.cache.pkl')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'キャッシュが見つかりません: {path}\n'
            f'先に実行してください: python src/06_predict_from_card.py'
        )
    with open(path, 'rb') as f:
        cached = pickle.load(f)
    df = cached['result']
    df.columns = df.columns.astype(object)
    for col in list(df.columns):
        if 'string' in str(df[col].dtype).lower():
            df[col] = df[col].astype(object)
    return df


# ── 発走時刻スクレイプ ────────────────────────────────────────────────────

def fetch_race_times(venue_keys: dict, year2: str) -> dict:
    """
    sports.yahoo.co.jp から各レースの発走時刻を取得。
    Returns: {(venue_prefix, r_num): 'HH:MM'}
    venue_keys: {'05': [kai, nichi], '08': [kai, nichi], ...}
    """
    from bs4 import BeautifulSoup
    venue_code_to_prefix = {
        '01': '札', '02': '函', '03': '福', '04': '新', '05': '東',
        '06': '中', '07': '中京', '08': '京', '09': '阪', '10': '小',
    }
    times = {}
    for vcode, (kai, nichi) in venue_keys.items():
        prefix = venue_code_to_prefix.get(vcode, vcode)
        for race_num in range(1, 13):
            key = f'{year2}{vcode}{kai:02d}{nichi:02d}{race_num:02d}'
            url = f'https://sports.yahoo.co.jp/keiba/race/denma/{key}'
            try:
                r = requests.get(url, headers=HEADERS, timeout=5, verify=False)
                if r.status_code != 200 or len(r.text) < 3000:
                    continue
                r.encoding = 'utf-8'
                soup = BeautifulSoup(r.text, 'html.parser')
                # 発走時刻は h3 や span などに "HH:MM発走" 形式で出る
                m = re.search(r'(\d{1,2}:\d{2})\s*(?:発走|発)', soup.get_text())
                if m:
                    times[(prefix, race_num)] = m.group(1)
                time.sleep(0.1)
            except Exception:
                pass
    return times


# ── EV計算とフィルタリング ────────────────────────────────────────────────

def compute_ev(df: pd.DataFrame, odds_dict: dict) -> pd.DataFrame:
    df = df.copy()
    df['_yahoo_odds']  = df['_horse'].map(odds_dict)
    df['_market_prob'] = 1.0 / df['_yahoo_odds'].clip(lower=1.0)

    # clogit_calib が保存されている場合は Yahoo オッズで clogit_score を再計算
    # （cache 生成時にオッズ未確定 → score=calib になっていた問題の補正）
    if 'clogit_calib' in df.columns and df['clogit_calib'].notna().any():
        factor = df['clogit_factor'].fillna(0.16)
        has_odds = df['_market_prob'].notna()
        df.loc[has_odds, 'clogit_score'] = (
            df.loc[has_odds, 'clogit_calib'] - factor[has_odds] * df.loc[has_odds, '_market_prob']
        )
        df['clogit_rank'] = df.groupby('_race_key')['clogit_score'].rank(
            ascending=False, method='first'
        )
        df['_ev'] = df['clogit_calib'] - df['_market_prob'] * 0.80
    else:
        df['_ev'] = df['clogit_score'] - df['_market_prob'] * 0.80
    return df


def get_good_picks(df: pd.DataFrame, ev_thr: float = 0.0) -> pd.DataFrame:
    """
    clogit_rank=1 かつ EV > ev_thr の馬を抽出。
    """
    mask = (df['clogit_rank'] == 1) & (df['_ev'] > ev_thr)
    return df[mask].copy()


# ── 準備処理 ─────────────────────────────────────────────────────────────

def prepare_df(df: pd.DataFrame, odds_dict: dict, race_times: dict) -> pd.DataFrame:
    df = df.copy()
    col_race = next((c for c in df.columns if c in ('場 R', '場R')), None)
    col_surf = next((c for c in df.columns if c in ('芝・ダ', '苝・ダ')), None)
    col_dist = next((c for c in df.columns if c in ('距離',)), None)

    def venue_prefix(race_str):
        m = re.match(r'^([^\d]+)', str(race_str))
        return m.group(1) if m else ''

    df['_race_key'] = df[col_race] if col_race else ''
    df['_venue']    = df[col_race].apply(venue_prefix) if col_race else ''
    df['_R']        = pd.to_numeric(df['Ｒ'], errors='coerce') if 'Ｒ' in df.columns else np.nan
    df['_horse']    = df['馬名S'].astype(str).str.strip() if '馬名S' in df.columns else ''
    df['_surface']  = df[col_surf].astype(str).str.strip() if col_surf else ''
    df['_dist']     = df[col_dist].astype(str).str.strip() if col_dist else ''

    # 発走時刻: スクレイプ結果 → なければ標準時刻
    def get_time(row):
        key = (row['_venue'], int(row['_R']) if pd.notna(row['_R']) else 0)
        if key in race_times:
            return race_times[key]
        return FALLBACK_TIMES.get(int(row['_R']) if pd.notna(row['_R']) else 0, '')

    df['_time_str'] = df.apply(get_time, axis=1)

    # EV計算
    df = compute_ev(df, odds_dict)

    # レース内Yahoo人気
    df['_pop_rank'] = (df.groupby('_race_key')['_yahoo_odds']
                       .rank(method='first', ascending=True))
    return df


def parse_race_time(time_str: str) -> datetime | None:
    try:
        h, m = map(int, time_str.split(':'))
        now = datetime.now()
        return now.replace(hour=h, minute=m, second=0, microsecond=0)
    except Exception:
        return None


# ── 通知フォーマット ──────────────────────────────────────────────────────

def build_embed(row: pd.Series, live_odds: float = None, ev_thr: float = 0.0) -> dict:
    surf_emo  = SURFACE_EMO.get(row['_surface'], '🏇')
    venue_full = VENUE_NAME.get(row['_venue'], row['_venue'])
    r_num      = int(row['_R']) if pd.notna(row['_R']) else '?'

    odds  = live_odds if live_odds is not None else row.get('_yahoo_odds')
    odds_str = f"{odds:.1f}倍" if pd.notna(odds) else '-'

    pop = int(row['_pop_rank']) if pd.notna(row.get('_pop_rank')) else '?'
    ev  = row.get('_ev', np.nan)
    ev_str = f"{ev:+.3f}" if pd.notna(ev) else '-'

    # EVレベルで色分け
    if ev > 0.05:
        color = 0x00C853   # 緑 (強いEV)
        ev_tag = '🟢 EV高'
    elif ev > 0.02:
        color = 0xFFB300   # 黄
        ev_tag = '🟡 EV+'
    else:
        color = 0x42A5F5   # 青
        ev_tag = '🔵 EV+'

    dist_clean = re.sub(r'^[芝ダ]', '', row['_dist'])
    title = f"🏇 {venue_full} {r_num}R"
    desc  = (
        f"**{row['_horse']}**\n"
        f"　{surf_emo} {row['_surface']}{dist_clean}  👥 {pop}番人気  💴 {odds_str}\n"
        f"　{ev_tag}  EV={ev_str}  clogit_score={row['clogit_score']:.3f}\n\n"
        f"⚡ clogit1位 × EV>{ev_thr:.2f} → **買い対象**"
    )
    return {
        'title':       title,
        'description': desc,
        'color':       color,
        'footer':      {'text': f"clogit (OOS ROI -12.7%) | 発走: {row['_time_str']}"},
    }


def morning_summary(picks: list[dict], webhook_url: str, ev_thr: float):
    if not picks:
        send_discord(webhook_url, '**🏇 本日のclogit買い対象: なし**')
        return

    high_ev = [p for p in picks if p['ev'] > 0.05]
    mid_ev  = [p for p in picks if 0.02 < p['ev'] <= 0.05]
    low_ev  = [p for p in picks if p['ev'] <= 0.02]

    lines = [f"**🏇 本日のclogit買い対象** (rank=1 × EV>{ev_thr:.2f})  計{len(picks)}件\n"]

    for label, group, emo in [
        ('EV>0.05 (強推奨)', high_ev, '🟢'),
        ('EV>0.02',          mid_ev,  '🟡'),
        ('EV+',              low_ev,  '🔵'),
    ]:
        if not group:
            continue
        lines.append(f'**{emo} {label}:**')
        for p in sorted(group, key=lambda x: x['time_str']):
            odds_s = f"{p['odds']:.1f}倍" if pd.notna(p['odds']) else '-'
            lines.append(
                f"　`{p['time_str']}` **{p['venue']} {p['r']}R**"
                f" │ {p['horse']}  {p['pop']}番人気 {odds_s}  EV={p['ev']:+.3f}"
            )

    send_discord(webhook_url, '\n'.join(lines))
    print(f'[朝サマリ送信] {len(picks)}件')


# ── メインループ ──────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test', action='store_true', help='即時テスト送信')
    ap.add_argument('--ev',   type=float, default=0.0, help='EVしきい値 (デフォルト0.0)')
    args = ap.parse_args()

    print('=== Discord 通知デーモン (clogit版) 起動 ===')

    webhook_url  = load_webhook_url()
    target_date  = datetime.now().strftime('%Y%m%d')
    year2        = target_date[2:4]

    print(f'対象日: {target_date}  EVしきい値: {args.ev}')

    # キャッシュ読み込み
    df = load_today_cache(target_date)
    print(f'キャッシュ: {len(df)}頭  clogit_rank有: {df["clogit_rank"].notna().sum()}頭')

    # Yahoo オッズ取得（既存JSONがあれば使用、なければ取得）
    odds_json = os.path.join(CACHE_DIR, f'{target_date}.odds.json')
    if os.path.exists(odds_json):
        with open(odds_json, encoding='utf-8') as f:
            odds_dict = json.load(f)
        print(f'オッズ: キャッシュから {len(odds_dict)}頭')
    else:
        print('Yahooからオッズ取得中...')
        odds_dict = fetch_odds_now(target_date, card_df=df)
        save_odds_json(target_date, odds_dict)
        print(f'オッズ取得完了: {len(odds_dict)}頭')

    # venue_keys 読み込み（race_times スクレイプに使用）
    vk_path = os.path.join(CACHE_DIR, f'{target_date}.venue_keys.json')
    race_times = {}
    if os.path.exists(vk_path):
        with open(vk_path, encoding='utf-8') as f:
            venue_keys_raw = json.load(f)
        venue_keys = {k: v for k, v in venue_keys_raw.items()}
        print('発走時刻をスクレイプ中...')
        race_times = fetch_race_times(venue_keys, year2)
        print(f'時刻取得: {len(race_times)}件')
    else:
        print('[WARN] venue_keys.json なし → 標準時刻を使用')

    # DataFrame準備
    df = prepare_df(df, odds_dict, race_times)

    # 好条件馬を抽出
    picks_df = get_good_picks(df, ev_thr=args.ev)
    print(f'買い対象: {len(picks_df)}頭 (clogit_rank=1 × EV>{args.ev})')

    # 通知リスト作成
    picks = []
    for _, row in picks_df.iterrows():
        t = parse_race_time(row['_time_str'])
        picks.append({
            'race_key': row['_race_key'],
            'venue':    row['_venue'],
            'r':        int(row['_R']) if pd.notna(row['_R']) else 0,
            'horse':    row['_horse'],
            'pop':      int(row['_pop_rank']) if pd.notna(row.get('_pop_rank')) else '?',
            'odds':     row.get('_yahoo_odds', np.nan),
            'ev':       row['_ev'],
            'time_str': row['_time_str'],
            'race_dt':  t,
            'row':      row,
        })

    # 朝サマリ送信
    morning_summary(picks, webhook_url, args.ev)

    if args.test:
        if picks:
            print('[テストモード] 1件embed送信')
            send_discord(webhook_url, '', embeds=[build_embed(picks[0]['row'], ev_thr=args.ev)])
        print('テスト完了')
        return

    last_refresh = datetime.now()
    notified     = set()

    print(f'待機中... 発走{NOTIFY_BEFORE_MIN}分前に通知  終了: {END_HOUR}:{END_MIN:02d}')
    while True:
        now = datetime.now()
        if now.hour > END_HOUR or (now.hour == END_HOUR and now.minute >= END_MIN):
            print(f'{END_HOUR}:{END_MIN:02d} 到達 → 終了')
            break

        # 定期オッズ更新
        if (now - last_refresh).seconds >= ODDS_REFRESH_MIN * 60:
            print('[オッズ更新]')
            new_odds = fetch_odds_now(target_date, card_df=df)
            if new_odds:
                odds_dict = new_odds
                save_odds_json(target_date, odds_dict)
                # EV再計算して picks の odds を更新
                for p in picks:
                    p['odds'] = odds_dict.get(p['horse'], np.nan)
            last_refresh = now

        for p in picks:
            if p['race_key'] in notified:
                continue
            if p['race_dt'] is None:
                continue
            notify_at = p['race_dt'] - timedelta(minutes=NOTIFY_BEFORE_MIN)
            if now >= notify_at:
                live_odds = odds_dict.get(p['horse'])
                print(f"[通知] {p['venue']} {p['r']}R {p['horse']}  EV={p['ev']:+.3f}  {p['time_str']}")
                send_discord(webhook_url, '', embeds=[build_embed(p['row'], live_odds, ev_thr=args.ev)])
                notified.add(p['race_key'])

        if picks and len(notified) >= len(picks):
            print('全件通知済み → 終了')
            break

        time.sleep(30)


if __name__ == '__main__':
    main()
