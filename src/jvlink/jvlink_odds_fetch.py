# -*- coding: utf-8 -*-
"""
JV-Link 単勝オッズ取得 → odds.json 保存

使い方:
  python src/jvlink/jvlink_odds_fetch.py --date 20260503
  python src/jvlink/jvlink_odds_fetch.py           # 今日の日付

速報系単複枠オッズ（dataspec "0B31"）を JVRTOpen で取得し、
SE レコード（枠順確定後）で馬番→馬名のマッピングを作成して
data/raw/cache/YYYYMMDD.odds.json に保存する。

戻り値（print）:
  CHANGED=1  変化あり
  CHANGED=0  変化なし
"""
import sys, io, os, json, argparse, time
from datetime import datetime, timedelta

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import pythoncom
pythoncom.CoInitialize()
import win32com.client as wc

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE_DIR = os.path.join(BASE_DIR, 'data', 'raw', 'cache')

JYO_NAME = {
    "01": "札", "02": "函", "03": "福", "04": "新", "05": "東",
    "06": "中", "07": "中京", "08": "京", "09": "阪", "10": "小",
}


def fetch_umaban_map(target_date: str, from_date: str) -> dict:
    """
    JVOpen RACE spec (mode=1) から SE レコードを読み、
    (jyo_cd, race_no, umaban) → horse_name マッピングを返す。
    枠順確定後（水〜木）に SE[28:30] に馬番が入る。
    """
    jv = wc.gencache.EnsureDispatch("JVDTLab.JVLink")
    rc = jv.JVInit("UNKNOWN")
    if rc != 0:
        print(f"[ERROR] JVInit失敗: {rc}", file=sys.stderr)
        return {}

    from_dt = from_date + "000000"
    rc, readcnt, dldcnt, lts = jv.JVOpen("RACE", from_dt, 1, 0, 0, "")
    if rc not in (0, 1):
        print(f"[WARN] JVOpen(RACE) rc={rc}", file=sys.stderr)
        jv.JVClose()
        return {}

    buf = " " * 110000
    uma_map = {}  # (jyo_cd, race_no, umaban) → horse_name

    while True:
        try:
            ret, data, sz, fname = jv.JVRead(buf, 110000, "")
        except Exception as e:
            print(f"[WARN] JVRead例外: {e}", file=sys.stderr)
            break
        if ret == 0: break
        if ret == -1: continue
        if ret == -3:
            time.sleep(0.05)
            continue
        if ret < 0: break

        if data[:2] != "SE":
            continue

        kd = data[11:19].strip()
        if kd != target_date:
            continue

        vc = data[19:21].strip()
        rn = data[25:27].strip()
        uma = data[40:58].strip() if len(data) > 58 else ''
        umaban = data[28:30].strip() if len(data) > 30 else ''
        chakujun = data[212:214].strip() if len(data) > 214 else ''

        if not uma or not umaban or chakujun != '00':
            continue

        uma_map[(vc, rn, umaban)] = uma

    jv.JVClose()
    print(f"馬番マップ: {len(uma_map)}頭 (target={target_date})")
    return uma_map


def fetch_o1_odds(uma_map: dict, target_date: str) -> dict:
    """
    JVRTOpen("0B31") で速報系単複オッズを取得し、
    {horse_name: odds_float} を返す（馬名マッチできたもののみ）。

    "0B31" は JVRTOpen 専用の速報系単複枠オッズ dataspec。
    "O1" は JVOpen(蓄積系)用であり JVRTOpen には使えない（rc=-111）。

    O1 レコード固定長レイアウト（JVLink仕様書準拠, 1レコード=1頭分）:
      [0:2]   RecordSpec "O1"
      [2:10]  KaisaiDate YYYYMMDD
      [10:12] JyoCode
      [12:13] KaisaiKai
      [13:14] KaisaiNichime
      [14:16] RaceNum
      [16:18] Umaban
      [18:23] TanshoOdds (×10, 例 "01850" = 18.5倍)
    """
    jv = wc.gencache.EnsureDispatch("JVDTLab.JVLink")
    rc = jv.JVInit("UNKNOWN")
    if rc != 0:
        print(f"[ERROR] JVInit失敗: {rc}", file=sys.stderr)
        return {}

    # "0B31" = 速報系単複枠オッズ。key に日付を渡すことで当日分のみ取得。
    rc = jv.JVRTOpen("0B31", target_date)
    if rc < 0:
        print(f"[WARN] JVRTOpen(0B31) rc={rc} (オッズ未配信 or レース日以外)", file=sys.stderr)
        jv.JVClose()
        return {}

    buf = " " * 110000
    odds_dict = {}  # horse_name → odds_float
    n_matched = n_unmatched = 0

    while True:
        try:
            ret, data, sz, fname = jv.JVRead(buf, 110000, "")
        except Exception as e:
            print(f"[WARN] JVRead例外: {e}", file=sys.stderr)
            break
        if ret == 0: break
        if ret == -1: continue
        if ret == -3: continue
        if ret < 0: break

        if not data.startswith("O1"):
            continue
        if len(data) < 23:
            continue

        # フィールド抽出
        kd      = data[2:10].strip()
        jyo_cd  = data[10:12].strip()
        race_no = data[14:16].strip()
        umaban  = data[16:18].strip()
        tansho  = data[18:23].strip()

        if kd != target_date:
            continue

        try:
            odds_val = int(tansho) / 10.0
        except ValueError:
            continue

        key = (jyo_cd, race_no, umaban)
        horse = uma_map.get(key)
        if horse:
            odds_dict[horse] = odds_val
            n_matched += 1
        else:
            n_unmatched += 1

    jv.JVClose()
    print(f"O1取得: {n_matched}頭マッチ / {n_unmatched}頭未マッチ")
    return odds_dict


def save_odds_json(target_date: str, odds_dict: dict) -> bool:
    """
    odds.json に保存。既存と差分があれば True を返す。
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f'{target_date}.odds.json')

    prev = {}
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            try:
                prev = json.load(f)
            except Exception:
                pass

    if not odds_dict:
        return False

    # 変化検出: 合計オッズの差が1%以上 or 頭数変化
    prev_sum = sum(prev.values())
    new_sum  = sum(odds_dict.values())
    changed = (
        len(odds_dict) != len(prev)
        or abs(new_sum - prev_sum) / max(prev_sum, 1) > 0.01
    )

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(odds_dict, f, ensure_ascii=False, indent=None)
    print(f"odds.json 保存: {len(odds_dict)}頭 → {path}")
    return changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', default=None, help='対象日 YYYYMMDD (省略時: 今日)')
    ap.add_argument('--from', dest='from_date', default=None,
                    help='JVOpen 開始日 YYYYMMDD (省略時: 2週前)')
    args = ap.parse_args()

    today = datetime.now().strftime('%Y%m%d')
    target_date = args.date or today
    from_date   = args.from_date or (datetime.now() - timedelta(days=14)).strftime('%Y%m%d')

    print(f"対象日: {target_date}  (JVOpen from: {from_date})")

    # Step 1: 馬番→馬名マッピング
    uma_map = fetch_umaban_map(target_date, from_date)

    if not uma_map:
        print("馬番マップなし（枠順未確定 or 対象日データなし）")
        print("CHANGED=0")
        return

    # Step 2: O1 オッズ取得
    odds_dict = fetch_o1_odds(uma_map, target_date)

    if not odds_dict:
        print("オッズデータなし（レース日以外 or 未配信）")
        print("CHANGED=0")
        return

    # Step 3: 保存・変化検出
    changed = save_odds_json(target_date, odds_dict)
    print(f"CHANGED={1 if changed else 0}")


if __name__ == '__main__':
    main()
