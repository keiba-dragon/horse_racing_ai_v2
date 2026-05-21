# -*- coding: utf-8 -*-
"""
JV-Link 海外競走成績取得 (BAGO → WH/WE レコード)

使い方:
  python src/fetch_overseas.py --probe --from 20240101        # WH生レコード確認
  python src/fetch_overseas.py --from 20130101 --to 20201231  # 初回全件
  python src/fetch_overseas.py --incremental                   # 未取得分だけ補完

出力:
  data/raw/overseas/YYYYMMDD.csv  (日ごと・1頭1行)

列: 日付, 会場コード, 会場名, レースNo, 距離, 芝ダ, 頭数, 馬番, 馬名, 着順, クラス推定, 走破タイム

※ WH/WE バイト位置は --probe で実測して WH_POS / WE_POS を更新すること。
"""
import sys, io, os, csv, argparse, time, glob as _glob
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import pythoncom
pythoncom.CoInitialize()
import win32com.client as wc

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR  = os.path.join(BASE_DIR, 'data', 'raw', 'overseas')

# 海外競馬場コード → 略名
OVERSEAS_VENUE = {
    "A0": "香港",   # Hong Kong
    "B0": "UAE",    # Dubai
    "C0": "英",     # UK
    "D0": "仏",     # France
    "E0": "愛",     # Ireland
    "F0": "独",     # Germany
    "G0": "伊",     # Italy
    "H0": "米",     # USA
    "I0": "加",     # Canada
    "J0": "豪",     # Australia
    "K0": "NZ",     # New Zealand
    "L0": "シンガポール",
    "M0": "サウジ",
}

# ★ WE7 フィールド位置（未確認: --probe で実測すること）
# WE7 = 海外競走基本情報
WE_POS = {
    'kaisai_year': (11, 15),
    'kaisai_md':   (15, 19),
    'venue_cd':    (19, 21),
    'race_no':     (25, 27),
    'race_name':   (40, 80),   # レース名 (要確認)
    'shusso_tosu': (80, 82),   # 出走頭数 (要確認)
    'distance':    (84, 88),   # 距離 4桁 (要確認)
    'track_cd':    (88, 90),   # コードコード (要確認)
    'grade_cd':    (82, 84),   # グレードコード (要確認: G1=1,G2=2,G3=3,L=4,OP=5)
}

# ★ WH7 フィールド位置（未確認: --probe で実測すること）
# WH7 = 海外出走馬成績
WH_POS = {
    'kaisai_year': (11, 15),
    'kaisai_md':   (15, 19),
    'venue_cd':    (19, 21),
    'race_no':     (25, 27),
    'umaban':      (28, 30),
    'horse_name':  (40, 58),   # 馬名 18バイト (SE7と同じ位置仮定)
    'chakujun':    (212, 214), # 着順 (SE7と同じ位置仮定)
    'soha_time':   (216, 220), # 走破タイム (SE7と同じ位置仮定)
}

CSV_FIELDS = [
    '日付', '会場コード', '会場名', 'レースNo', '距離', '芝ダ', '頭数',
    '馬番', '馬名', '着順', 'クラス推定', '走破タイム',
]

GRADE_MAP = {
    '1': 9,  # G1
    '2': 8,  # G2
    '3': 7,  # G3
    '4': 6,  # Listed
    '5': 6,  # OP相当
}


def _safe_strip(s):
    return s.strip() if isinstance(s, str) else ''


def parse_we(rec):
    try:
        if len(rec) < 100:
            return None
        p = WE_POS
        kaisai_date = rec[p['kaisai_year'][0]:p['kaisai_year'][1]] \
                    + rec[p['kaisai_md'][0]:p['kaisai_md'][1]]
        venue_cd    = _safe_strip(rec[p['venue_cd'][0]:p['venue_cd'][1]])
        race_no     = _safe_strip(rec[p['race_no'][0]:p['race_no'][1]])
        if not kaisai_date.isdigit() or len(kaisai_date) != 8:
            return None
        shusso  = _safe_strip(rec[p['shusso_tosu'][0]:p['shusso_tosu'][1]])
        dist    = _safe_strip(rec[p['distance'][0]:p['distance'][1]])
        track   = _safe_strip(rec[p['track_cd'][0]:p['track_cd'][1]])
        grade   = _safe_strip(rec[p['grade_cd'][0]:p['grade_cd'][1]])

        if track in ('1', '2', '3'):
            surface = '芝'
        elif track in ('4', '5'):
            surface = 'ダ'
        else:
            surface = ''

        return {
            'kaisai_date': kaisai_date,
            'venue_cd':    venue_cd,
            'race_no':     race_no,
            'shusso':      shusso,
            'distance':    dist,
            'surface':     surface,
            'grade_rank':  GRADE_MAP.get(grade, 6),
        }
    except Exception:
        return None


def parse_wh(rec):
    try:
        if len(rec) < 220:
            return None
        p = WH_POS
        kaisai_date = rec[p['kaisai_year'][0]:p['kaisai_year'][1]] \
                    + rec[p['kaisai_md'][0]:p['kaisai_md'][1]]
        venue_cd   = _safe_strip(rec[p['venue_cd'][0]:p['venue_cd'][1]])
        race_no    = _safe_strip(rec[p['race_no'][0]:p['race_no'][1]])
        umaban     = _safe_strip(rec[p['umaban'][0]:p['umaban'][1]])
        horse_name = _safe_strip(rec[p['horse_name'][0]:p['horse_name'][1]])
        chakujun   = _safe_strip(rec[p['chakujun'][0]:p['chakujun'][1]])
        soha_raw   = _safe_strip(rec[p['soha_time'][0]:p['soha_time'][1]])
        soha_time  = soha_raw if soha_raw.isdigit() and int(soha_raw) > 0 else ''

        if not kaisai_date.isdigit() or len(kaisai_date) != 8:
            return None
        if not horse_name:
            return None

        return {
            'kaisai_date': kaisai_date,
            'venue_cd':    venue_cd,
            'race_no':     race_no,
            'umaban':      umaban,
            'horse_name':  horse_name,
            'chakujun':    chakujun,
            'soha_time':   soha_time,
        }
    except Exception:
        return None


# ─────────────────────────────────────────────────────────
# プローブ
# ─────────────────────────────────────────────────────────
def probe_bago(jv, n_we=2, n_wh=3):
    """WE/WHの生バイトを表示してフィールド位置を確認する。"""
    buf  = " " * 110000
    size = 110000
    we_n = wh_n = 0

    while we_n < n_we or wh_n < n_wh:
        ret, data, sz, fname = jv.JVRead(buf, size, "")
        if ret == 0:
            print("[EOF]")
            break
        if ret == -1:
            continue
        if ret == -3:
            time.sleep(0.05)
            continue
        if ret < 0:
            print(f"[JVRead error {ret}]")
            break

        rt = data[:2].strip()

        if rt == "WE" and we_n < n_we:
            print(f"\n{'='*70}")
            print(f"[WE #{we_n+1}]  日付={data[11:19]}  会場={data[19:21]}  レース={data[25:27]}")
            print(f"  レコード長={ret}")
            print(f"{'='*70}")
            _dump_ranges(data, ret, [(0, 120, "ヘッダ〜レース基本情報", 10),
                                      (80, 200, "距離・グレード推定域", 10)])
            we_n += 1

        elif rt == "WH" and wh_n < n_wh:
            print(f"\n{'='*70}")
            print(f"[WH #{wh_n+1}]  日付={data[11:19]}  会場={data[19:21]}  レース={data[25:27]}")
            print(f"  馬番推定[28:30]={repr(data[28:30])}  レコード長={ret}")
            print(f"{'='*70}")
            _dump_ranges(data, ret, [(0,  60, "ヘッダ〜馬名推定域", 10),
                                      (58, 180, "馬名後〜着順推定域", 10),
                                      (175, 280, "着順・タイム推定域", 10)])
            wh_n += 1

    print(f"\nWE: {we_n}件  WH: {wh_n}件")


def _dump_ranges(data, rec_len, ranges):
    for start, end, label, step in ranges:
        end = min(end, rec_len)
        if start >= end:
            continue
        print(f"\n  --- {label} [{start}:{end}] ---")
        for i in range(start, end, step):
            chunk = data[i:min(i+step, end)]
            try:
                encoded = chunk.encode('cp932', errors='replace').hex()
            except Exception:
                encoded = '?'
            print(f"  [{i:03d}:{min(i+step,end):03d}]  {repr(chunk):<40} hex={encoded}")


# ─────────────────────────────────────────────────────────
# メイン取得ロジック
# ─────────────────────────────────────────────────────────
def fetch_range(jv, from_date, to_date, skip_dates=None, setup_mode=1):
    """JVOpen(BAGO) → WE/WH 読み取り → {日付: [rows]} を返す。"""
    from_dt = from_date + "000000"
    rc, readcnt, dldcnt, lts = jv.JVOpen("BAGO", from_dt, setup_mode, 0, 0, "")
    print(f"JVOpen(BAGO): rc={rc} readcnt={readcnt} dldcnt={dldcnt}")
    if rc < 0:
        if rc == -202 and setup_mode == 1:
            print("  rc=-202: SetupMode=2 でリトライ")
            jv.JVClose()
            rc, readcnt, dldcnt, lts = jv.JVOpen("BAGO", from_dt, 2, 0, 0, "")
            print(f"  JVOpen mode=2: rc={rc}")
        if rc < 0:
            print(f"JVOpen(BAGO)失敗 rc={rc}  海外競走データが購読されていない可能性があります。")
            return {}

    buf  = " " * 110000
    size = 110000

    race_info = {}   # (kd, venue_cd, race_no) → WE data
    daily     = defaultdict(list)
    n_we = n_wh = n_skip = 0

    print(f"読み込み中 {from_date}〜{to_date} ...")
    while True:
        try:
            ret, data, sz, fname = jv.JVRead(buf, size, "")
        except Exception as e:
            print(f"JVRead例外: {e}")
            break
        if ret == 0:
            print("EOF")
            break
        if ret == -1:
            continue
        if ret == -3:
            time.sleep(0.05)
            continue
        if ret < 0:
            print(f"JVRead error {ret}")
            break

        rt = data[:2].strip()

        if rt == "WE":
            we = parse_we(data)
            if we and from_date <= we['kaisai_date'] <= to_date:
                key = (we['kaisai_date'], we['venue_cd'], we['race_no'])
                race_info[key] = we
                n_we += 1

        elif rt == "WH":
            wh = parse_wh(data)
            if wh is None:
                continue
            kd = wh['kaisai_date']
            if kd < from_date or kd > to_date:
                continue
            if skip_dates and kd in skip_dates:
                continue

            key = (kd, wh['venue_cd'], wh['race_no'])
            we  = race_info.get(key, {})
            venue_name = OVERSEAS_VENUE.get(wh['venue_cd'], wh['venue_cd'])

            daily[kd].append({
                '日付':      kd,
                '会場コード': wh['venue_cd'],
                '会場名':    venue_name,
                'レースNo':  wh['race_no'],
                '距離':      we.get('distance', ''),
                '芝ダ':      we.get('surface', ''),
                '頭数':      we.get('shusso', ''),
                '馬番':      wh['umaban'],
                '馬名':      wh['horse_name'],
                '着順':      wh['chakujun'],
                'クラス推定': we.get('grade_rank', ''),
                '走破タイム': wh['soha_time'],
            })
            n_wh += 1
        else:
            n_skip += 1

    jv.JVClose()
    print(f"WE: {n_we}件 / WH: {n_wh}件 / その他: {n_skip}件")
    return daily


def save_daily(daily):
    os.makedirs(OUT_DIR, exist_ok=True)
    total = 0
    for kd in sorted(daily.keys()):
        out_path = os.path.join(OUT_DIR, f'{kd}.csv')
        rows = daily[kd]
        with open(out_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        total += len(rows)
    print(f"保存: {len(daily)}日  {total}行 → {OUT_DIR}/")
    return total


# ─────────────────────────────────────────────────────────
# エントリポイント
# ─────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description='JV-Link 海外競走成績フェッチャー')
    ap.add_argument('--from',        dest='from_date', default='20130101')
    ap.add_argument('--to',          dest='to_date',   default='20991231')
    ap.add_argument('--incremental', action='store_true', help='既存日付をスキップ')
    ap.add_argument('--full-setup',  action='store_true', help='SetupMode=2 で強制再取得')
    ap.add_argument('--probe',       action='store_true', help='WE/WH生レコードを表示して終了')
    args = ap.parse_args()

    jv = wc.gencache.EnsureDispatch("JVDTLab.JVLink")
    rc = jv.JVInit("UNKNOWN")
    print(f"JVInit: {rc}")
    if rc != 0:
        print("JVInit失敗。ターゲットFrontierが起動しているか確認してください。")
        sys.exit(1)

    if args.probe:
        from_dt = args.from_date + "000000"
        rc2, readcnt, _, _ = jv.JVOpen("BAGO", from_dt, 1, 0, 0, "")
        if rc2 < 0:
            print(f"JVOpen(BAGO)失敗 rc={rc2}  海外競走データが未購読の可能性があります。")
            sys.exit(1)
        print(f"JVOpen(BAGO) OK: readcnt={readcnt}")
        probe_bago(jv, n_we=3, n_wh=5)
        return

    skip_dates = set()
    if args.incremental:
        existing = _glob.glob(os.path.join(OUT_DIR, '????????.csv'))
        skip_dates = {os.path.basename(f).replace('.csv', '') for f in existing}
        print(f"既存: {len(skip_dates)}日をスキップ")

    setup_mode = 2 if args.full_setup else 1
    daily = fetch_range(jv, args.from_date, args.to_date, skip_dates, setup_mode)

    if not daily:
        print("海外競走データが取れませんでした。")
        return

    save_daily(daily)

    first_date = sorted(daily.keys())[0]
    print(f"\nサンプル ({first_date}) 先頭5行:")
    for r in daily[first_date][:5]:
        print(f"  {r['会場名']}{r['レースNo']}R  馬番{r['馬番']}  {r['馬名']}  "
              f"着={r['着順']}  距離={r['距離']}{r['芝ダ']}  クラス={r['クラス推定']}")


if __name__ == '__main__':
    main()
