# coding: utf-8
"""
results_freshness.py - 直近の開催日の結果が results_supplement.csv に取り込み済みかを確認する。

JV-Linkの成績確定は月曜朝6時には間に合わず、着順0埋めの未確定データが返ってくる
（2026-11-05 調査）。新聞(金曜夜)を作る前にこの確認をして、取り込めていなければ
auto_weekly_update.py --retry で取り直し、それでも入らなければ警告を出す。
"""
import os
import subprocess
import sys
from datetime import datetime, timedelta

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, 'data', 'raw', 'results')
CACHE_DIR = os.path.join(BASE_DIR, 'data', 'raw', 'cache')
SUPPLEMENT_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'master', 'results_supplement.csv')
WEEKLY_UPDATE = os.path.join(BASE_DIR, 'src', 'auto_weekly_update.py')

LOOKBACK_DAYS = 14


def _confirmed_in_supplement():
    """supplement内で着順>0（確定）の行がある日付(YYMMDD, int)の集合。"""
    df = pd.read_csv(SUPPLEMENT_PATH, usecols=['日付', '着順'], low_memory=False)
    d = pd.to_numeric(df['日付'], errors='coerce')
    c = pd.to_numeric(df['着順'], errors='coerce')
    return set(d[c.gt(0)].dropna().astype(int).unique())


def _has_card(day):
    """その日の出馬表キャッシュがあるか（開催日だったかの目安）。"""
    if not os.path.isdir(CACHE_DIR):
        return False
    pat = f'{day.month:02d}月{day.day}日_api.cache.pkl'
    return any(f.endswith(pat) for f in os.listdir(CACHE_DIR))


def pending_days(today=None):
    """直近LOOKBACK_DAYS日（今日を除く）で、結果が未取り込みの開催日を返す。
    開催日の判定: 結果ファイルがある日、または出馬表キャッシュがある土日。
    戻り値: ['20261031', ...]（古い順）
    """
    today = (today or datetime.now()).replace(hour=0, minute=0, second=0, microsecond=0)
    confirmed = _confirmed_in_supplement()
    out = []
    for back in range(LOOKBACK_DAYS, 0, -1):
        day = today - timedelta(days=back)
        stem = day.strftime('%Y%m%d')
        date_num = int(stem[2:])
        has_result_file = os.path.exists(os.path.join(RESULTS_DIR, f'{stem}.csv'))
        is_race_day = has_result_file or (day.weekday() >= 5 and _has_card(day))
        if is_race_day and date_num not in confirmed:
            out.append(stem)
    return out


def ensure_results(notify=None, today=None):
    """未取り込みの開催日があれば週次更新を取り直し、それでも残れば警告する。
    notify: 文字列を受け取るDiscord通知関数（省略可）。
    戻り値: 取り直し後も未取り込みの日付リスト（空なら問題なし）
    """
    try:
        pend = pending_days(today)
    except Exception as e:
        print(f'  [WARN] 結果の鮮度チェックに失敗: {e}')
        return []
    if not pend:
        print('  結果の鮮度チェック: 直近の開催日はすべて取り込み済み')
        return []

    print(f'  結果が未取り込みの開催日: {", ".join(pend)} → 週次更新を取り直します')
    subprocess.run([sys.executable, WEEKLY_UPDATE, '--retry'], cwd=BASE_DIR)
    pend = pending_days(today)
    if pend:
        msg = (f'⚠️ 新聞作成前チェック: 次の開催日の結果がまだ取り込めていません: {", ".join(pend)}\n'
               f'これらの日に走った馬は前走情報が欠けた状態で予想されます（新聞は作成します）。')
        print('  ' + msg.replace('\n', '\n  '))
        if notify:
            notify(msg)
    else:
        print('  取り直しで取り込み完了')
    return pend


if __name__ == '__main__':
    print(pending_days())
