# -*- coding: utf-8 -*-
"""
祝日などの土日以外の開催日に、1回だけ動くタスクを登録する（setup_scheduler.py は土日のみ想定のため）。

登録するタスク（いずれも1回限り）:
  KeibAI-Extra-{日付}-Predict  前日 17:30  --mode predict --date {日付}（枠順確定後の予想新聞）
  KeibAI-Extra-{日付}-AM       当日 10:00  --mode watch --once --date {日付}
  KeibAI-Extra-{日付}-PM       当日 15:00  同上
  KeibAI-Extra-{日付}-Eve      当日 18:30  同上（夜の確定版）

実行:
  python src/schedule_extra_day.py 20261123            # 登録
  python src/schedule_extra_day.py 20261123 --status   # 確認
  python src/schedule_extra_day.py 20261123 --delete   # 削除（終わったら消してよい）
"""
import sys, argparse
from datetime import datetime, timedelta

from setup_scheduler import PYTHON, SCRIPT, BASE_DIR, ps_run  # stdoutのUTF-8化も読み込み時に済む


def plan(date_str):
    d = datetime.strptime(date_str, '%Y%m%d')
    prev = d - timedelta(days=1)
    base = f'KeibAI-Extra-{date_str}'
    watch = f'--mode watch --once --date {date_str}'
    return [
        (f'{base}-Predict', prev.replace(hour=17, minute=30), f'--mode predict --date {date_str}'),
        (f'{base}-AM',      d.replace(hour=10, minute=0),     watch),
        (f'{base}-PM',      d.replace(hour=15, minute=0),     watch),
        (f'{base}-Eve',     d.replace(hour=18, minute=30),    watch),
    ]


def esc(p):
    return p.replace("'", "''")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('date', help='開催日 YYYYMMDD（例: 20261123）')
    ap.add_argument('--delete', action='store_true')
    ap.add_argument('--status', action='store_true')
    args = ap.parse_args()

    for name, when, targs in plan(args.date):
        if args.delete:
            ps = f"Unregister-ScheduledTask -TaskName '{name}' -Confirm:$false; Write-Output 'deleted: {name}'"
        elif args.status:
            ps = (f"$t = Get-ScheduledTask -TaskName '{name}' -ErrorAction SilentlyContinue; "
                  f"if ($t) {{ $i = $t | Get-ScheduledTaskInfo; "
                  f"Write-Output ('{name}: ' + $t.State + ' next=' + $i.NextRunTime + ' last=' + $i.LastRunTime + ' result=' + $i.LastTaskResult) }} "
                  f"else {{ Write-Output '{name}: 未登録' }}")
        else:
            if when < datetime.now():
                print(f'スキップ(時刻が過ぎている): {name} {when:%m/%d %H:%M}  → 手動で: python src/auto_pipeline.py {targs}')
                continue
            ps = f"""
$a = New-ScheduledTaskAction -Execute '{esc(PYTHON)}' -Argument '"{esc(SCRIPT)}" {targs}' -WorkingDirectory '{esc(BASE_DIR)}'
$t = New-ScheduledTaskTrigger -Once -At '{when:%Y-%m-%d %H:%M}'
$s = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Hours 4) -StartWhenAvailable
Register-ScheduledTask -TaskName '{name}' -Description '競馬AI 臨時開催 {args.date}' -Action $a -Trigger $t -Settings $s -Force | Out-Null
Write-Output 'OK: {name} {when:%m/%d %H:%M}'
"""
        rc, _ = ps_run(ps)
        if rc != 0:
            print(f'  ✗ 失敗: {name}')


if __name__ == '__main__':
    main()
