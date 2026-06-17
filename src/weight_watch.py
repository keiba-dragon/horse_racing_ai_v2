# coding: utf-8
"""
weight_watch.py — 馬体重発表を監視して新聞を自動再生成・Push

usage:
    python src/weight_watch.py              # 今日の日付で実行
    python src/weight_watch.py 260621       # 日付指定 (YYMMDD)

動作:
  1. 数分ごとに shutuba.html をポーリング
  2. 体重カバー率が THRESHOLD を超えたら make_newspaper.py を実行
  3. docs/ への HTML 出力後、git push で GitHub Pages に公開
"""
import sys, os, re, time, subprocess, urllib.request
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THRESHOLD  = 0.80   # 80% 以上の馬に体重が入ったら実行
POLL_SECS  = 180    # 3分ごとにチェック
MAX_WAIT_H = 4      # 最大4時間待機


def get_race_ids(tgt_date: str) -> list:
    full_date = ('20' + str(tgt_date)) if len(str(tgt_date)) == 6 else str(tgt_date)
    try:
        req = urllib.request.Request(
            f'https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={full_date}',
            headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as r:
            html = r.read().decode('euc-jp', errors='replace')
        return list(dict.fromkeys(re.findall(r'race_id=(\d{12})', html)))
    except Exception as e:
        print(f'[WARN] レースID取得失敗: {e}')
        return []


def check_weight_coverage(race_ids: list) -> tuple[float, int, int]:
    """shutuba.html をチェックして体重カバー率を返す。
    Returns: (coverage_rate, n_with_weight, n_total)
    """
    row_pat = re.compile(r'<tr[^>]*class="[^"]*HorseList[^"]*"[^>]*>(.*?)</tr>', re.DOTALL)
    uma_pat = re.compile(r'class="Umaban\d*[^"]*"[^>]*>\s*(\d+)', re.DOTALL)
    wt_pat  = re.compile(r'class="[^"]*Weight[^"]*"[^>]*>(.*?)</td>', re.DOTALL)

    n_total = 0
    n_filled = 0
    for race_id in race_ids:
        try:
            req = urllib.request.Request(
                f'https://race.netkeiba.com/race/shutuba.html?race_id={race_id}',
                headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=10) as r:
                html = r.read().decode('euc-jp', errors='replace')
            for m in row_pat.finditer(html):
                row = m.group(1)
                u_m = uma_pat.search(row)
                w_m = wt_pat.search(row)
                if u_m:
                    n_total += 1
                    if w_m:
                        wt_raw = re.sub(r'<[^>]+>', '', w_m.group(1)).strip()
                        if re.search(r'\d{3}', wt_raw):
                            n_filled += 1
            time.sleep(0.2)
        except Exception:
            continue

    rate = n_filled / n_total if n_total > 0 else 0.0
    return rate, n_filled, n_total


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
    print(f'=== weight_watch.py 開始 === 対象日: {tgt_date}')
    print(f'閾値: {THRESHOLD:.0%}  チェック間隔: {POLL_SECS}秒  最大待機: {MAX_WAIT_H}時間')

    # レースID取得
    print(f'[{now()}] レースID取得中...')
    race_ids = get_race_ids(tgt_date)
    if not race_ids:
        print('レースIDが取得できませんでした。終了します。')
        return
    print(f'  {len(race_ids)}レース検出')

    deadline = time.time() + MAX_WAIT_H * 3600
    already_run = False

    while time.time() < deadline:
        rate, n_filled, n_total = check_weight_coverage(race_ids)
        ts = now()
        print(f'[{ts}] 体重カバー率: {n_filled}/{n_total}頭 ({rate:.1%})', end='')

        if rate >= THRESHOLD and not already_run:
            print(f'  → 閾値超え！新聞生成します')
            ok = run_newspaper()
            if ok:
                git_push()
                already_run = True
                print(f'[{now()}] 完了。続けて体重が増えた場合は再実行します（次回閾値: 95%）')
                # 一度実行後は95%で再実行（最終確定版）
                # already_run のまま待機し、rate >= 0.95 で再実行
        elif rate >= 0.95 and already_run:
            print(f'  → ほぼ全頭出揃いました。最終版を生成します')
            ok = run_newspaper()
            if ok:
                git_push()
            print(f'[{now()}] 終了します')
            break
        else:
            print(f'  → 待機中...')

        if already_run and rate >= 0.95:
            break

        time.sleep(POLL_SECS)
    else:
        print(f'[{now()}] 最大待機時間を超えました。終了します。')


if __name__ == '__main__':
    main()
