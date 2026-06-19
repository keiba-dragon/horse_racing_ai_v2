# coding: utf-8
"""
weight_watch.py — 全レースの発走前に新聞を自動再生成・Push

usage:
    python src/weight_watch.py              # 今日
    python src/weight_watch.py 260621       # 日付指定 (YYMMDD)

動作:
  1. netkeibaから当日の発走時刻一覧を取得
  2. 各レースの BEFORE_MINS 分前に make_newspaper.py を実行
  3. docs/ 更新後 git push で GitHub Pages に公開
"""
import sys, os, re, time, subprocess, urllib.request
from datetime import datetime, timedelta

sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEFORE_MINS = 30   # 発走の何分前に実行するか


def get_race_times(tgt_date: str) -> list[datetime]:
    """netkeibaから当日の発走時刻リストを取得して返す（昇順）。"""
    full_date = ('20' + str(tgt_date)) if len(str(tgt_date)) == 6 else str(tgt_date)
    year = int(full_date[:4])
    month = int(full_date[4:6])
    day = int(full_date[6:8])
    try:
        req = urllib.request.Request(
            f'https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={full_date}',
            headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as r:
            html = r.read().decode('euc-jp', errors='replace')
    except Exception as e:
        print(f'[WARN] レース一覧取得失敗: {e}')
        return []

    # li ブロック単位で race_id と時刻を対応付け
    times = []
    for block in re.split(r'(?=<li[^>]*>)', html):
        if 'myrace_' not in block:
            continue
        t_m = re.search(r'class="RaceList_Itemtime">(\d{2}:\d{2})', block)
        if not t_m:
            continue
        h, m = map(int, t_m.group(1).split(':'))
        times.append(datetime(year, month, day, h, m))

    times = sorted(set(times))
    print(f'発走時刻: {[t.strftime("%H:%M") for t in times]}')
    return times


def run_newspaper():
    script = os.path.join(BASE_DIR, 'src', 'make_newspaper.py')
    print(f'[{now()}] 新聞生成開始...')
    r = subprocess.run([sys.executable, script], cwd=BASE_DIR, capture_output=False)
    return r.returncode == 0


def git_push(label: str):
    print(f'[{now()}] GitHub Push ({label})...')
    cmds = [
        ['git', 'add', 'docs/'],
        ['git', 'commit', '-m', f'feat: 新聞更新 {label} ({now()})'],
        ['git', 'push', 'origin', 'main'],
    ]
    for cmd in cmds:
        r = subprocess.run(cmd, cwd=BASE_DIR, capture_output=True, text=True, encoding='utf-8')
        if r.returncode != 0 and 'nothing to commit' not in (r.stdout + r.stderr):
            print(f'[WARN] {r.stderr.strip()}')
            return False
    print(f'[{now()}] Push 完了')
    return True


def now() -> str:
    return datetime.now().strftime('%H:%M:%S')


def main():
    tgt_date = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime('%y%m%d')
    print(f'=== weight_watch.py === 対象日: {tgt_date}  発走{BEFORE_MINS}分前に実行')

    race_times = get_race_times(tgt_date)
    if not race_times:
        print('発走時刻が取得できませんでした。終了します。')
        return

    # 実行スケジュール = 各発走時刻の BEFORE_MINS 分前（重複排除・過去分スキップ）
    triggers = sorted(set(t - timedelta(minutes=BEFORE_MINS) for t in race_times))
    now_dt   = datetime.now()
    triggers = [t for t in triggers if t > now_dt]

    if not triggers:
        print('本日の実行タイミングがすべて過去です。今すぐ実行します。')
        if run_newspaper():
            git_push('即時')
        return

    print(f'実行予定: {[t.strftime("%H:%M") for t in triggers]}')

    last_run = None
    for trigger in triggers:
        # 待機
        while datetime.now() < trigger:
            remaining = int((trigger - datetime.now()).total_seconds() / 60)
            print(f'[{now()}] 次の実行: {trigger.strftime("%H:%M")} (あと約{remaining}分)', flush=True)
            time.sleep(60)

        # 同じ分に複数トリガーが重なっていたらスキップ
        if last_run and (datetime.now() - last_run).total_seconds() < 120:
            continue

        label = trigger.strftime('%H:%M')
        print(f'[{now()}] {label} トリガー')
        if run_newspaper():
            git_push(label)
            last_run = datetime.now()


if __name__ == '__main__':
    main()
