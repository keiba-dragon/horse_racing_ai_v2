# -*- coding: utf-8 -*-
"""
Windows タスクスケジューラに競馬AI パイプラインを登録する。

作成するタスク:
  KeibAI-Predict   金曜 21:00  predict_weekend + docs push（印候補）
  KeibAI-Watch-Sat 土曜 08:00  オッズポーリング → 確定版 push
  KeibAI-Watch-Sun 日曜 08:00  同上

実行:
  python src/setup_scheduler.py           # 登録
  python src/setup_scheduler.py --delete  # 削除
  python src/setup_scheduler.py --status  # 確認

注意:
  - JV-Link (ターゲットFrontier) が起動している状態でないと失敗します
  - 「ログオン時のみ実行」で登録するため、PC がスリープ中は実行されません
    → スリープ無効推奨（コントロールパネル → 電源オプション）
"""
import sys, io, os, subprocess, argparse

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENV_PYTHON = os.path.join(BASE_DIR, '.venv_new', 'Scripts', 'python.exe')
PYTHON      = VENV_PYTHON if os.path.exists(VENV_PYTHON) else sys.executable
SCRIPT         = os.path.join(BASE_DIR, 'src', 'auto_pipeline.py')
UPDATE_SCRIPT  = os.path.join(BASE_DIR, 'src', 'auto_weekly_update.py')
LOG_DIR     = os.path.join(BASE_DIR, 'logs')

TASKS = [
    {
        'name':  'KeibAI-Predict',
        'desc':  '競馬AI 週末予測（金曜・印候補新聞生成）',
        'day':   'FRI',
        'time':  '21:00',
        'args':  '--mode predict',
    },
    {
        'name':  'KeibAI-Watch-Sat-AM',
        'desc':  '競馬AI 土曜 午前オッズスナップショット',
        'day':   'SAT',
        'time':  '10:00',
        'args':  '--mode watch --once',
    },
    {
        'name':  'KeibAI-Watch-Sat-PM',
        'desc':  '競馬AI 土曜 午後オッズスナップショット',
        'day':   'SAT',
        'time':  '15:00',
        'args':  '--mode watch --once',
    },
    {
        'name':  'KeibAI-Watch-Sun-AM',
        'desc':  '競馬AI 日曜 午前オッズスナップショット',
        'day':   'SUN',
        'time':  '10:00',
        'args':  '--mode watch --once',
    },
    {
        'name':  'KeibAI-Watch-Sun-PM',
        'desc':  '競馬AI 日曜 午後オッズスナップショット',
        'day':   'SUN',
        'time':  '15:00',
        'args':  '--mode watch --once',
    },
]

# 月曜 週次更新タスク（別スクリプト・別定義）
UPDATE_TASKS = [
    {'name': 'KeibAI-Update-Mon', 'desc': '競馬AI 週次自動更新（月曜 JV-Link結果取得）', 'day': 'MON', 'time': '06:00'},
]

# Discord通知デーモン用タスク（別スクリプト・別定義）
NOTIFY_TASKS = [
    {'name': 'KeibAI-Notify-Sat', 'desc': '競馬AI Discord通知 土曜', 'day': 'SAT', 'time': '09:00'},
    {'name': 'KeibAI-Notify-Sun', 'desc': '競馬AI Discord通知 日曜', 'day': 'SUN', 'time': '09:00'},
]


DAY_MAP = {'MON': 'Monday', 'TUE': 'Tuesday', 'WED': 'Wednesday',
           'THU': 'Thursday', 'FRI': 'Friday', 'SAT': 'Saturday', 'SUN': 'Sunday'}


def ps_run(script: str) -> tuple[int, str]:
    r = subprocess.run(
        ['powershell', '-NoProfile', '-NonInteractive', '-Command', script],
        capture_output=True, text=True, encoding='utf-8', errors='replace',
        cwd=BASE_DIR
    )
    out = (r.stdout + r.stderr).strip()
    if out:
        for line in out.splitlines():
            print(f"    {line}")
    return r.returncode, r.stdout.strip()


NOTIFY_SCRIPT = os.path.join(BASE_DIR, 'src', 'discord_notify.py')


def register():
    print("=== タスクスケジューラ 登録 ===\n")
    os.makedirs(LOG_DIR, exist_ok=True)

    # 既存パイプラインタスク
    for t in TASKS:
        log_path  = os.path.join(LOG_DIR, f'{t["name"]}.log').replace('\\', '\\\\')
        py_path   = PYTHON.replace('\\', '\\\\')
        scr_path  = SCRIPT.replace('\\', '\\\\')
        base_path = BASE_DIR.replace('\\', '\\\\')
        day_full  = DAY_MAP[t['day']]

        # PowerShell Register-ScheduledTask（管理者不要）
        ps = f"""
$a = New-ScheduledTaskAction `
    -Execute '{py_path}' `
    -Argument '"{scr_path}" {t["args"]}' `
    -WorkingDirectory '{base_path}'
$t = New-ScheduledTaskTrigger -Weekly -DaysOfWeek {day_full} -At '{t["time"]}'
$s = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Hours 4) -StartWhenAvailable
Register-ScheduledTask -TaskName '{t["name"]}' -Description '{t["desc"]}' `
    -Action $a -Trigger $t -Settings $s -Force | Out-Null
Write-Output "OK: {t['name']}"
"""
        print(f"登録: {t['name']}  ({t['day']} {t['time']})")
        rc, out = ps_run(ps)
        print(f"  {'✓ 登録完了' if rc == 0 else '✗ 登録失敗'}\n")

    # 月曜 週次更新タスク
    upd_scr = UPDATE_SCRIPT.replace('\\', '\\\\')
    for t in UPDATE_TASKS:
        py_path   = PYTHON.replace('\\', '\\\\')
        base_path = BASE_DIR.replace('\\', '\\\\')
        day_full  = DAY_MAP[t['day']]
        ps = f"""
$a = New-ScheduledTaskAction `
    -Execute '{py_path}' `
    -Argument '"{upd_scr}"' `
    -WorkingDirectory '{base_path}'
$t = New-ScheduledTaskTrigger -Weekly -DaysOfWeek {day_full} -At '{t["time"]}'
$s = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Hours 2) -StartWhenAvailable
Register-ScheduledTask -TaskName '{t["name"]}' -Description '{t["desc"]}' `
    -Action $a -Trigger $t -Settings $s -Force | Out-Null
Write-Output "OK: {t['name']}"
"""
        print(f"登録: {t['name']}  ({t['day']} {t['time']})")
        rc, out = ps_run(ps)
        print(f"  {'✓ 登録完了' if rc == 0 else '✗ 登録失敗'}\n")

    # Discord通知デーモンタスク
    notify_scr = NOTIFY_SCRIPT.replace('\\', '\\\\')
    for t in NOTIFY_TASKS:
        log_path  = os.path.join(LOG_DIR, f'{t["name"]}.log').replace('\\', '\\\\')
        py_path   = PYTHON.replace('\\', '\\\\')
        base_path = BASE_DIR.replace('\\', '\\\\')
        day_full  = DAY_MAP[t['day']]
        ps = f"""
$a = New-ScheduledTaskAction `
    -Execute '{py_path}' `
    -Argument '"{notify_scr}"' `
    -WorkingDirectory '{base_path}'
$t = New-ScheduledTaskTrigger -Weekly -DaysOfWeek {day_full} -At '{t["time"]}'
$s = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Hours 9) -StartWhenAvailable
Register-ScheduledTask -TaskName '{t["name"]}' -Description '{t["desc"]}' `
    -Action $a -Trigger $t -Settings $s -Force | Out-Null
Write-Output "OK: {t['name']}"
"""
        print(f"登録: {t['name']}  ({t['day']} {t['time']})")
        rc, out = ps_run(ps)
        print(f"  {'✓ 登録完了' if rc == 0 else '✗ 登録失敗'}\n")


def delete():
    print("=== タスクスケジューラ 削除 ===\n")
    for t in TASKS + UPDATE_TASKS + NOTIFY_TASKS:
        print(f"削除: {t['name']}")
        ps_run(f"Unregister-ScheduledTask -TaskName '{t['name']}' -Confirm:$false -ErrorAction SilentlyContinue")
        print()


def status():
    print("=== タスクスケジューラ 確認 ===\n")
    for t in TASKS + UPDATE_TASKS + NOTIFY_TASKS:
        print(f"── {t['name']} ──")
        rc, out = ps_run(
            f"Get-ScheduledTask -TaskName '{t['name']}' -ErrorAction SilentlyContinue "
            f"| Select-Object TaskName,State,@{{n='NextRun';e={{($_ | Get-ScheduledTaskInfo).NextRunTime}}}} "
            f"| Format-List"
        )
        if not out.strip():
            print("  (未登録)")
        print()


def main():
    ap = argparse.ArgumentParser()
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument('--delete', action='store_true', help='タスクを削除')
    grp.add_argument('--status', action='store_true', help='登録状況を確認')
    args = ap.parse_args()

    if args.delete:
        delete()
    elif args.status:
        status()
    else:
        register()
        print("\n登録済みタスク:")
        status()
        print("\n補足:")
        print(f"  ログ出力先: {LOG_DIR}\\")
        print("  手動実行: schtasks /run /tn KeibAI-Predict")
        print("  注意: PC がスリープ中は実行されません（スリープ無効推奨）")


if __name__ == '__main__':
    main()
