# coding: utf-8
"""
weight_watch.py — 指定時刻に新聞を自動再生成・Push

usage:
    python src/weight_watch.py              # 今日の日付・デフォルト時刻(09:30)で実行
    python src/weight_watch.py 260621       # 日付指定 (YYMMDD)
    python src/weight_watch.py 260621 1000  # 日付 + 実行時刻 (HHMM)

動作:
  1. RUN_AT の時刻まで待機
  2. make_newspaper.py を実行（shutuba.html から体重を自動取得してスコア再計算）
  3. docs/ への HTML 出力後、git push で GitHub Pages に公開
"""
import sys, os, time, subprocess
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_AT   = '0930'  # デフォルト実行時刻 (HHMM)


def wait_until(hhmm: str):
    """指定時刻(HHMM)まで待機する。"""
    target_h, target_m = int(hhmm[:2]), int(hhmm[2:])
    while True:
        n = datetime.now()
        if (n.hour, n.minute) >= (target_h, target_m):
            break
        remaining = (target_h * 60 + target_m) - (n.hour * 60 + n.minute)
        print(f'[{now()}] {hhmm[:2]}:{hhmm[2:]} まで待機中... (あと約{remaining}分)', flush=True)
        time.sleep(60)


def run_newspaper():
    """make_newspaper.py を実行して HTML を生成する。"""
    script = os.path.join(BASE_DIR, 'src', 'make_newspaper.py')
    print(f'\n[{now()}] 新聞生成開始...')
    result = subprocess.run(
        [sys.executable, script],
        cwd=BASE_DIR,
        capture_output=False,
    )
    return result.returncode == 0


def git_push():
    """docs/ の変更を commit して push する。"""
    print(f'[{now()}] GitHub Push...')
    cmds = [
        ['git', 'add', 'docs/'],
        ['git', 'commit', '-m', f'feat: 馬体重反映済み新聞 ({now()})'],
        ['git', 'push', 'origin', 'main'],
    ]
    for cmd in cmds:
        r = subprocess.run(cmd, cwd=BASE_DIR, capture_output=True, text=True, encoding='utf-8')
        if r.returncode != 0 and 'nothing to commit' not in r.stdout + r.stderr:
            print(f'[WARN] {" ".join(cmd)}: {r.stderr.strip()}')
            return False
    print(f'[{now()}] Push 完了')
    return True


def now() -> str:
    return datetime.now().strftime('%H:%M:%S')


def main():
    tgt_date = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime('%y%m%d')
    run_at   = sys.argv[2] if len(sys.argv) > 2 else RUN_AT

    print(f'=== weight_watch.py === 対象日: {tgt_date}  実行時刻: {run_at[:2]}:{run_at[2:]}')

    wait_until(run_at)

    print(f'[{now()}] 時刻到達 → 新聞生成開始')
    ok = run_newspaper()
    if ok:
        git_push()
        print(f'[{now()}] 完了')
    else:
        print(f'[{now()}] 新聞生成に失敗しました')


if __name__ == '__main__':
    main()
