# -*- coding: utf-8 -*-
"""
通知・ヘルスステータス共有モジュール。

auto_weekly_update.py / auto_pipeline.py に重複していた _notify_discord() を統合し、
さらに以下を追加する:
  - Discord送信のリトライ（指数バックオフ）でDNS一時失敗等を吸収
  - logs/health_status.json への実行結果記録（Discordが完全に届かなくても
    次回いずれかのパイプライン実行時・新聞生成時に状態を確認できるようにする）

discord_notify.py は import 時に sys.stdout/stderr を再ラップするため、
呼び出し元の _Tee 済みストリームと衝突する既知の問題がある。
そのため送信ロジックはこのモジュールに直接実装し、discord_notify.py はimportしない。

使い方:
  from notify import notify_discord, record_health, load_health, JVLINK_UNAVAILABLE

  ok = run_fetch()  # 戻り値: 'ok' / 'jvlink_unavailable' / 'error'
  record_health('weekly_update', ok)
  notify_discord('...')
"""
import os
import json
import time
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, 'config', 'discord_config.json')
HEALTH_PATH = os.path.join(BASE_DIR, 'logs', 'health_status.json')

# run_fetch() 等が返すステータス種別
STATUS_OK = 'ok'
STATUS_JVLINK_UNAVAILABLE = 'jvlink_unavailable'
STATUS_ERROR = 'error'

# health_statusでこの回数以上「連続でJV-Link取得できていない」場合に新聞へ警告を出す
JVLINK_WARN_THRESHOLD = 2


def notify_discord(content: str, retries: int = 3, backoff_sec: float = 2.0) -> bool:
    """DiscordのwebhookへPOSTする。DNS一時失敗等に備えて指数バックオフでリトライする。
    未設定・全リトライ失敗時はFalseを返す（例外は投げない）。
    """
    try:
        with open(CONFIG_PATH, encoding='utf-8') as f:
            cfg = json.load(f)
    except Exception as e:
        print(f'  [WARN] discord_config.json 読込失敗: {e}')
        return False

    webhook_url = cfg.get('webhook_url', '').strip()
    if not webhook_url or 'YOUR_WEBHOOK' in webhook_url:
        print('  [WARN] discord_config.json の webhook_url 未設定。通知スキップ')
        return False

    import requests
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.post(webhook_url, json={'content': content}, timeout=10, verify=False)
            if r.status_code in (200, 204):
                return True
            last_err = f'{r.status_code} {r.text[:200]}'
        except Exception as e:
            last_err = str(e)
        if attempt < retries:
            time.sleep(backoff_sec * attempt)
    print(f'  [WARN] Discord通知失敗（{retries}回リトライ後）: {last_err}')
    return False


def load_health() -> dict:
    """logs/health_status.json を読み込む。存在しなければ空の初期状態を返す。"""
    if not os.path.exists(HEALTH_PATH):
        return {}
    try:
        with open(HEALTH_PATH, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def record_health(task_name: str, status: str, detail: str = '') -> dict:
    """タスクの実行結果をlogs/health_status.jsonに記録する。
    JV-Link取得(status==STATUS_JVLINK_UNAVAILABLE)が連続した回数を追跡し、
    新聞側で警告バナーを出すかどうかの判断材料にする。

    戻り値: 更新後のhealthレコード全体（呼び出し元でのログ出力用）。
    """
    os.makedirs(os.path.dirname(HEALTH_PATH), exist_ok=True)
    health = load_health()
    entry = health.setdefault(task_name, {
        'consecutive_jvlink_unavailable': 0,
        'consecutive_errors': 0,
    })

    entry['last_run'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    entry['last_status'] = status
    entry['last_detail'] = detail

    if status == STATUS_JVLINK_UNAVAILABLE:
        entry['consecutive_jvlink_unavailable'] = entry.get('consecutive_jvlink_unavailable', 0) + 1
    else:
        entry['consecutive_jvlink_unavailable'] = 0

    if status == STATUS_ERROR:
        entry['consecutive_errors'] = entry.get('consecutive_errors', 0) + 1
    else:
        entry['consecutive_errors'] = 0

    health[task_name] = entry
    with open(HEALTH_PATH, 'w', encoding='utf-8') as f:
        json.dump(health, f, ensure_ascii=False, indent=2)
    return entry


def jvlink_warning_message() -> str:
    """新聞・レポート側で表示する警告文言を返す。問題なければ空文字。"""
    health = load_health()
    entry = health.get('weekly_update', {})
    n = entry.get('consecutive_jvlink_unavailable', 0)
    if n >= JVLINK_WARN_THRESHOLD:
        last_run = entry.get('last_run', '?')
        return (
            f'⚠ JV-Link（ターゲットFrontier）が{n}週連続で取得できていません（最終確認: {last_run}）。'
            f'前走特徴量が古いデータのまま計算されている可能性があります。ターゲットFrontierの起動状態を確認してください。'
        )
    return ''
