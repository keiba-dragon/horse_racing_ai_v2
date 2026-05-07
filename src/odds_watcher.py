# -*- coding: utf-8 -*-
"""
オッズ自動更新ウォッチャー

レース当日、JV-Link O1（単勝オッズ）を定期ポーリングし、
変化があれば D指標新聞を自動再生成して GitHub Pages に push する。

使い方:
  python src/odds_watcher.py --date 20260503
  python src/odds_watcher.py --date 20260502,20260503
  python src/odds_watcher.py          # 今日の日付を自動
  python src/odds_watcher.py --interval 10  # ポーリング間隔（分）

動作:
  1. jvlink_odds_fetch.py でオッズ取得 → odds.json 更新
  2. 変化があれば newspaper.py で新聞再生成
  3. docs/ にコピー → git commit & push
  17:30 以降は自動終了

ポーリング間隔デフォルト: 10分
"""
import sys, io, os, subprocess, shutil, time, argparse
from datetime import datetime, timedelta

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR      = os.path.join(BASE_DIR, 'src')
NEWSPAPER_PY = os.path.join(BASE_DIR, 'd_core', 'predict', 'newspaper.py')
DOCS_DIR     = os.path.join(BASE_DIR, 'docs')
OUTPUT_DIR   = os.path.join(BASE_DIR, 'd_core', 'predict', 'output')
CACHE_DIR    = os.path.join(BASE_DIR, 'data', 'raw', 'cache')
VENV_PYTHON  = os.path.join(BASE_DIR, '.venv_new', 'Scripts', 'python.exe')

PYTHON = VENV_PYTHON if os.path.exists(VENV_PYTHON) else sys.executable


def run(cmd: list, label: str = '') -> tuple[int, str]:
    """コマンドを実行して (returncode, stdout) を返す"""
    label_s = f'[{label}] ' if label else ''
    print(f"{label_s}実行: {' '.join(cmd)}")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8',
                           errors='replace', cwd=BASE_DIR)
        if r.stdout.strip():
            for line in r.stdout.strip().splitlines()[-8:]:
                print(f"  {line}")
        if r.returncode != 0 and r.stderr.strip():
            for line in r.stderr.strip().splitlines()[-4:]:
                print(f"  [STDERR] {line}", file=sys.stderr)
        return r.returncode, r.stdout
    except Exception as e:
        print(f"  [ERROR] {e}", file=sys.stderr)
        return -1, ''


def fetch_odds(target_date: str, from_date: str) -> bool:
    """オッズを取得・更新。変化があれば True"""
    _, out = run(
        [PYTHON, os.path.join(SRC_DIR, 'jvlink', 'jvlink_odds_fetch.py'),
         '--date', target_date, '--from', from_date],
        label='odds_fetch'
    )
    return 'CHANGED=1' in out


def regenerate_newspaper(target_date: str) -> str | None:
    """
    newspaper.py を実行して最新の HTML パスを返す。
    失敗時は None。
    """
    rc, _ = run([PYTHON, NEWSPAPER_PY, target_date], label='newspaper')
    if rc != 0:
        print(f"[ERROR] newspaper.py 失敗 (rc={rc})")
        return None

    # 最新の出力 HTML を探す
    import glob
    yymmdd = target_date[2:]  # 20260503 → 260503
    pattern = os.path.join(OUTPUT_DIR, f'd_newspaper_{yymmdd}_*.html')
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def copy_to_docs(html_path: str, target_date: str):
    """docs/ にコピー"""
    yymmdd = target_date[2:]
    dest = os.path.join(DOCS_DIR, f'd_newspaper_{yymmdd}.html')
    shutil.copy2(html_path, dest)
    print(f"  → docs にコピー: {os.path.basename(dest)}")
    return dest


def git_commit_push(target_dates: list[str], now_str: str):
    """変更を git commit して push"""
    files = []
    for td in target_dates:
        yymmdd = td[2:]
        f = os.path.join(DOCS_DIR, f'd_newspaper_{yymmdd}.html')
        if os.path.exists(f):
            files.append(f'd_newspaper_{yymmdd}.html')

    if not files:
        return

    # git add
    git_add_targets = [os.path.join('docs', f) for f in files]
    run(['git', 'add'] + git_add_targets, label='git')

    dates_str = ', '.join(td[4:] for td in target_dates)
    msg = f"auto: オッズ更新 {dates_str} [{now_str}]"
    rc, _ = run(['git', 'commit', '-m', msg], label='git')
    if rc != 0:
        print("  コミット不要（変更なし）")
        return

    run(['git', 'push', 'origin', 'main'], label='git')


def one_cycle(target_dates: list[str], from_date: str) -> bool:
    """1ポーリングサイクル。更新があれば True"""
    any_changed = False
    updated_dates = []

    for td in target_dates:
        print(f"\n--- {td} オッズチェック ---")
        changed = fetch_odds(td, from_date)
        if not changed:
            print(f"  変化なし → スキップ")
            continue

        print(f"  変化あり → 新聞再生成")
        html_path = regenerate_newspaper(td)
        if html_path is None:
            continue

        copy_to_docs(html_path, td)
        updated_dates.append(td)
        any_changed = True

    if updated_dates:
        now_str = datetime.now().strftime('%H:%M')
        git_commit_push(updated_dates, now_str)

    return any_changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', default=None,
                    help='対象日 YYYYMMDD（カンマ区切りで複数可）')
    ap.add_argument('--from', dest='from_date', default=None,
                    help='JVOpen 開始日 YYYYMMDD (省略時: 2週前)')
    ap.add_argument('--interval', type=int, default=10,
                    help='ポーリング間隔（分）デフォルト10')
    ap.add_argument('--once', action='store_true',
                    help='1回だけ実行して終了')
    ap.add_argument('--until', default='17:30',
                    help='終了時刻 HH:MM (デフォルト 17:30)')
    args = ap.parse_args()

    today = datetime.now().strftime('%Y%m%d')
    from_date = args.from_date or (datetime.now() - timedelta(days=14)).strftime('%Y%m%d')

    if args.date:
        target_dates = [d.strip() for d in args.date.split(',')]
    else:
        target_dates = [today]

    until_h, until_m = map(int, args.until.split(':'))
    interval_sec = args.interval * 60

    print(f"=== オッズウォッチャー 起動 ===")
    print(f"対象日: {target_dates}")
    print(f"ポーリング間隔: {args.interval}分 / 終了時刻: {args.until}")
    print(f"JVOpen from: {from_date}")
    print()

    while True:
        now = datetime.now()
        if now.hour * 60 + now.minute >= until_h * 60 + until_m:
            print(f"[{now.strftime('%H:%M')}] 終了時刻に達しました。終了。")
            break

        print(f"[{now.strftime('%H:%M:%S')}] ポーリング開始")
        one_cycle(target_dates, from_date)

        if args.once:
            print("--once オプション: 1回で終了")
            break

        next_time = datetime.now() + timedelta(seconds=interval_sec)
        print(f"\n次回: {next_time.strftime('%H:%M:%S')} (約{args.interval}分後)\n")
        time.sleep(interval_sec)


if __name__ == '__main__':
    main()
