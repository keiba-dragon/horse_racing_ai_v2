# coding: utf-8
"""
週末自動パイプライン: netkeiba API経由（TARGET Frontier不要）で
出馬表取得→予測→新聞生成を自動実行する。

タスクスケジューラ登録（setup_scheduler.py）から呼ばれる想定:
  金曜 21:00  python auto_pipeline.py --mode predict
              → 直近の土曜・日曜の出馬表を取得し予想新聞を生成（印候補）
  土日 各時刻 python auto_pipeline.py --mode watch --once
              → 当日の新聞をオッズ・結果で更新（確定版push）

使い方:
  python src/auto_pipeline.py --mode predict
  python src/auto_pipeline.py --mode watch --once
  python src/auto_pipeline.py --mode predict --date 20260808   # 日付を明示指定
"""
import sys, os, subprocess, argparse
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable

FETCH_PREDICT = os.path.join(BASE_DIR, 'src', '_fetch_and_predict.py')
MAKE_NEWSPAPER = os.path.join(BASE_DIR, 'src', 'make_newspaper.py')
MAKE_REPORT = os.path.join(BASE_DIR, 'src', 'make_result_report.py')


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def _notify_discord(content: str):
    """discord_config.json のwebhookへ送信する（未設定なら黙って諦める）。
    discord_notify.pyはimport時にsys.stdoutを再ラップし、Tee済みstdoutと
    衝突して壊れるためimportせず直接実装する（auto_weekly_update.pyと同じ対策）。
    """
    try:
        import json
        import requests
        config_path = os.path.join(BASE_DIR, 'config', 'discord_config.json')
        with open(config_path, encoding='utf-8') as f:
            cfg = json.load(f)
        webhook_url = cfg.get('webhook_url', '').strip()
        if not webhook_url or 'YOUR_WEBHOOK' in webhook_url:
            print('  [WARN] discord_config.json の webhook_url 未設定。通知スキップ')
            return
        r = requests.post(webhook_url, json={'content': content}, timeout=10, verify=False)
        if r.status_code not in (200, 204):
            print(f'  [WARN] Discord通知失敗: {r.status_code} {r.text[:200]}')
    except Exception as e:
        print(f'  [WARN] Discord通知失敗: {e}')


def get_predict_dates(base_date=None):
    """直近（今日以降）の土曜・日曜の日付を返す。"""
    base = base_date or datetime.now()
    days_until_sat = (5 - base.weekday()) % 7
    sat = base + timedelta(days=days_until_sat)
    sun = sat + timedelta(days=1)
    return [sat.strftime('%Y%m%d'), sun.strftime('%Y%m%d')]


def get_watch_dates(base_date=None):
    """当日の日付を返す（土日のみ想定）。"""
    base = base_date or datetime.now()
    return [base.strftime('%Y%m%d')]


def run_step(args, label):
    print(f'\n--- {label} ---', flush=True)
    r = subprocess.run([PY] + args, cwd=BASE_DIR)
    ok = r.returncode == 0
    print(f'  {"OK" if ok else f"FAILED (code={r.returncode})"}')
    return ok


def run_predict(date_str):
    """出馬表取得→予測→新聞生成を1日分実行する。"""
    ok1 = run_step([FETCH_PREDICT, '--date', date_str], f'{date_str} 出馬表取得+予測')
    if not ok1:
        return False
    ok2 = run_step([MAKE_NEWSPAPER, '--date', date_str], f'{date_str} 新聞生成')
    return ok2


def _has_cache_for_date(date_str):
    """指定日付のカードキャッシュが既に存在するか（ファイル名の月日から判定）。"""
    cache_dir = os.path.join(BASE_DIR, 'data', 'raw', 'cache')
    if not os.path.isdir(cache_dir):
        return False
    month, day = int(date_str[4:6]), int(date_str[6:8])
    pat = f'{month:02d}月{day}日_api.cache.pkl'
    return any(f.endswith(pat) for f in os.listdir(cache_dir))


def run_watch(date_str):
    """当日の新聞をオッズ・結果で更新する（カード未取得なら先に取得する）。"""
    if not _has_cache_for_date(date_str):
        ok = run_step([FETCH_PREDICT, '--date', date_str], f'{date_str} 出馬表取得+予測（初回）')
        if not ok:
            return False
    ok_news = run_step([MAKE_NEWSPAPER, '--date', date_str], f'{date_str} 新聞更新')
    if ok_news:
        run_step([MAKE_REPORT, date_str], f'{date_str} 結果レポート更新')
    return ok_news


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['predict', 'watch'], required=True)
    ap.add_argument('--once', action='store_true', help='(watchモード) 1回のみ実行して終了')
    ap.add_argument('--date', help='対象日付を明示指定 YYYYMMDD（省略時は曜日から自動判定）')
    args = ap.parse_args()

    log_dir = os.path.join(BASE_DIR, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(log_dir, f'auto_pipeline_{args.mode}_{ts}.log')
    log_f = open(log_path, 'w', encoding='utf-8')
    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    sys.stdout = _Tee(orig_stdout, log_f)
    sys.stderr = _Tee(orig_stderr, log_f)

    try:
        print('=' * 50)
        print(f'  競馬AI 自動パイプライン [{args.mode}]  {datetime.now().strftime("%Y-%m-%d %H:%M")}')
        print('=' * 50)

        if args.date:
            dates = [args.date]
        elif args.mode == 'predict':
            dates = get_predict_dates()
        else:
            dates = get_watch_dates()

        print(f'対象日付: {dates}')

        results = {}
        for d in dates:
            if args.mode == 'predict':
                results[d] = run_predict(d)
            else:
                results[d] = run_watch(d)

        failed = [d for d, ok in results.items() if not ok]
        if failed:
            _notify_discord(
                f'🚨 競馬AI 自動パイプライン[{args.mode}] 一部失敗\n'
                f'失敗日付: {", ".join(failed)}\nログ: {os.path.basename(log_path)}'
            )
            sys.stdout, sys.stderr = orig_stdout, orig_stderr
            log_f.close()
            sys.exit(1)
        else:
            mode_label = '予想新聞生成' if args.mode == 'predict' else 'オッズ/結果更新'
            _notify_discord(
                f'✅ 競馬AI 自動パイプライン[{args.mode}] 成功\n'
                f'{mode_label}: {", ".join(dates)}\nログ: {os.path.basename(log_path)}'
            )
    except Exception as e:
        print(f'\n[ERROR] 予期しない例外: {e}')
        _notify_discord(f'🚨 競馬AI 自動パイプライン[{args.mode}] 失敗（予期しない例外）\n{e}\nログ: {os.path.basename(log_path)}')
        sys.stdout, sys.stderr = orig_stdout, orig_stderr
        log_f.close()
        sys.exit(1)
    else:
        sys.stdout, sys.stderr = orig_stdout, orig_stderr
        log_f.close()


if __name__ == '__main__':
    main()
