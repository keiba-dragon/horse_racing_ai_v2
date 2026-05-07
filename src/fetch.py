# -*- coding: utf-8 -*-
"""
JV-Link 過去レース結果一括取得 (horse_racing_ai_v2)

使い方:
  python src/fetch.py --from 20130101 --to 20201231   # 初回全件
  python src/fetch.py --from 20260101                  # 直近
  python src/fetch.py --incremental                    # 未取得日だけ補完
  python src/fetch.py --probe --from 20260420          # SEレコード生データ確認
  python src/fetch.py --probe-race 20260420 06 11     # 特定レースのSEを詳細表示

出力:
  data/raw/results/YYYYMMDD.csv  (日ごと・1頭1行)

列:
  日付, 会場コード, 会場, レースNo, 距離, 芝ダ, 馬場状態,
  頭数, 馬番, 馬名, 着順, 単勝オッズ,
  斤量(*), 騎手名(*), 馬体重(*), 馬体重変化(*),
  走破タイム(*), 上り3F(*), 1角(*), 2角(*), 3角(*), 4角(*)

(*) = フィールド位置未確定。--probe で確認後 SE_POS_UNVERIFIED を更新すること。
"""
import sys, io, os, csv, argparse, time, glob as _glob
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import pythoncom
pythoncom.CoInitialize()
import win32com.client as wc

# ─────────────────────────────────────────────────────────
# 定数
# ─────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR  = os.path.join(BASE_DIR, 'data', 'raw', 'results')

JYO_NAME = {
    "01": "札", "02": "函", "03": "福", "04": "新", "05": "東",
    "06": "中", "07": "中京", "08": "京", "09": "阪", "10": "小",
}

# SE7 フィールド位置（0ベース Python スライス）
# ★ 確認済み: 2026-03-21 中山7R で着順・オッズ・馬名一致確認
SE_POS_VERIFIED = {
    'kaisai_year': (11, 15),
    'kaisai_md':   (15, 19),
    'venue_cd':    (19, 21),
    'race_no':     (25, 27),
    'umaban':      (28, 30),
    'horse_name':  (40, 58),   # 全角9文字 = 18バイト (SJIS)
    'chakujun':    (212, 214),
    'tan_odds':    (237, 241), # 4桁整数 ÷10 = 倍率
}

# ★ 未確認: --probe-race で実測して更新すること
# JRA-VAN SE7 仕様書 V-19 推定値 (0ベース)
SE_POS_UNVERIFIED = {
    'waku_ban':    (27, 28),   # 枠番 (1桁)
    'jinryo':      (58, 61),   # 斤量 3桁整数 ÷10 = kg (例: "570" = 57.0)
    'jockey_cd':   (61, 66),   # 騎手コード 5桁
    'jockey_name': (66, 80),   # 騎手名 全角7文字 = 14バイト
    'bataiju':     (96, 99),   # 馬体重 3桁 kg
    'bataiju_diff':(99, 102),  # 馬体重変化 符号付3桁
    'soha_time':   (178, 184), # 走破タイム 6桁整数 (1/10秒単位 or 秒×10)
    'last3f':      (184, 187), # 上り3F 3桁整数 ÷10 = 秒
    'corner1':     (188, 190), # 1角通過順 2桁
    'corner2':     (190, 192), # 2角通過順 2桁
    'corner3':     (192, 194), # 3角通過順 2桁
    'corner4':     (194, 196), # 4角通過順 2桁
}

# RA7 フィールド位置（確認済み）
RA_POS = {
    'kaisai_year': (11, 15),
    'kaisai_md':   (15, 19),
    'venue_cd':    (19, 21),
    'race_no':     (25, 27),
    'kyoso_meisho':(40, 60),   # レース名
    'toroku_tosu': (62, 64),   # 登録頭数
    'shusso_tosu': (64, 66),   # 出走頭数
    'track_cd':    (566, 568), # トラックコード 2桁
    'distance':    (558, 562), # 距離 4桁
    'baba_state':  (594, 596), # 馬場状態 (DataKubun=1/2のみ有効)
}

CSV_FIELDS = [
    '日付', '会場コード', '会場', 'レースNo', '距離', '芝ダ', '馬場状態',
    '頭数', '馬番', '馬名', '着順', '単勝オッズ',
    '斤量', '騎手名', '馬体重', '馬体重変化',
    '走破タイム', '上り3F', '1角', '2角', '3角', '4角',
]


# ─────────────────────────────────────────────────────────
# パーサー
# ─────────────────────────────────────────────────────────
def track_to_surface(code_str):
    try:
        c = int(code_str.strip())
    except Exception:
        return ''
    if c == 24 or (51 <= c <= 54): return 'ダ'
    if c >= 55: return '障'
    if 10 <= c <= 29: return '芝'
    return ''


def parse_ra(rec):
    try:
        if len(rec) < 600:
            return None
        p = RA_POS
        kaisai_date = rec[p['kaisai_year'][0]:p['kaisai_year'][1]] \
                    + rec[p['kaisai_md'][0]:p['kaisai_md'][1]]
        venue_cd    = rec[p['venue_cd'][0]:p['venue_cd'][1]].strip()
        race_no     = rec[p['race_no'][0]:p['race_no'][1]].strip()
        distance    = rec[p['distance'][0]:p['distance'][1]].strip()
        track_cd    = rec[p['track_cd'][0]:p['track_cd'][1]].strip()
        shusso      = rec[p['shusso_tosu'][0]:p['shusso_tosu'][1]].strip()
        baba        = rec[p['baba_state'][0]:p['baba_state'][1]].strip()

        if not kaisai_date.isdigit() or len(kaisai_date) != 8:
            return None
        surface = track_to_surface(track_cd)
        if surface == '障':
            return None  # 障害は除外

        return {
            'kaisai_date': kaisai_date,
            'venue_cd':    venue_cd,
            'race_no':     race_no,
            'distance':    distance,
            'surface':     surface,
            'shusso_tosu': shusso,
            'baba_state':  baba,
        }
    except Exception:
        return None


def _safe_int_div(s, divisor):
    s = s.strip() if isinstance(s, str) else ''
    if s.lstrip('-').isdigit():
        return int(s) / divisor
    return ''


def _safe_strip(s):
    return s.strip() if isinstance(s, str) else ''


def parse_se(rec, include_unverified=True):
    try:
        if len(rec) < 242:
            return None
        p = SE_POS_VERIFIED
        kaisai_date = rec[p['kaisai_year'][0]:p['kaisai_year'][1]] \
                    + rec[p['kaisai_md'][0]:p['kaisai_md'][1]]
        venue_cd    = _safe_strip(rec[p['venue_cd'][0]:p['venue_cd'][1]])
        race_no     = _safe_strip(rec[p['race_no'][0]:p['race_no'][1]])
        umaban      = _safe_strip(rec[p['umaban'][0]:p['umaban'][1]])
        horse_name  = _safe_strip(rec[p['horse_name'][0]:p['horse_name'][1]])
        chakujun    = _safe_strip(rec[p['chakujun'][0]:p['chakujun'][1]])

        tan_odds_raw = rec[p['tan_odds'][0]:p['tan_odds'][1]].strip()
        tan_odds = int(tan_odds_raw) / 10.0 if tan_odds_raw.isdigit() and int(tan_odds_raw) > 0 else ''

        if not kaisai_date.isdigit() or len(kaisai_date) != 8:
            return None

        row = {
            'kaisai_date': kaisai_date,
            'venue_cd':    venue_cd,
            'race_no':     race_no,
            'umaban':      umaban,
            'horse_name':  horse_name,
            'chakujun':    chakujun,
            'tan_odds':    tan_odds,
            # unverified fields default empty
            'jinryo': '', 'jockey_name': '',
            'bataiju': '', 'bataiju_diff': '',
            'soha_time': '', 'last3f': '',
            'corner1': '', 'corner2': '', 'corner3': '', 'corner4': '',
        }

        if include_unverified and len(rec) >= 200:
            pu = SE_POS_UNVERIFIED
            row['jinryo']      = _safe_int_div(rec[pu['jinryo'][0]:pu['jinryo'][1]], 10)
            row['jockey_name'] = _safe_strip(rec[pu['jockey_name'][0]:pu['jockey_name'][1]])
            bataiju_raw        = rec[pu['bataiju'][0]:pu['bataiju'][1]].strip()
            row['bataiju']     = int(bataiju_raw) if bataiju_raw.isdigit() else ''
            row['bataiju_diff']= _safe_int_div(rec[pu['bataiju_diff'][0]:pu['bataiju_diff'][1]], 1)
            row['soha_time']   = _safe_int_div(rec[pu['soha_time'][0]:pu['soha_time'][1]], 10)
            row['last3f']      = _safe_int_div(rec[pu['last3f'][0]:pu['last3f'][1]], 10)
            for c, k in [('corner1','1角'),('corner2','2角'),('corner3','3角'),('corner4','4角')]:
                row[c] = _safe_strip(rec[pu[c][0]:pu[c][1]])

        return row
    except Exception:
        return None


# ─────────────────────────────────────────────────────────
# プローブ
# ─────────────────────────────────────────────────────────
def probe_records(jv, n_se=3, n_ra=2, target_venue=None, target_race=None):
    """SEとRAの生バイトを表示してフィールド位置を確認する。"""
    buf  = " " * 110000
    size = 110000
    se_n = ra_n = 0
    needed_se = n_se
    needed_ra = n_ra

    while se_n < needed_se or ra_n < needed_ra:
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

        if rt == "SE" and se_n < needed_se:
            # フィルタ: venue/race指定あり
            v = data[19:21].strip()
            r = data[25:27].strip()
            if target_venue and v != target_venue:
                continue
            if target_race and r != target_race:
                continue

            kd = data[11:19] + ''  # year+mmdd
            print(f"\n{'='*70}")
            print(f"[SE #{se_n+1}]  日付={data[11:19]}  会場={v}  レース={r}  馬番={data[28:30]}  馬名={data[40:58].strip()}")
            print(f"  着順[212:214]={repr(data[212:214])}  オッズ[237:241]={repr(data[237:241])}")
            print(f"{'='*70}")
            print(f"{'オフセット':>10}  値")
            # 主要範囲を詳細表示
            ranges = [(0, 70, "ヘッダ～馬名"), (58, 130, "馬名後～馬体重推定域"),
                      (170, 250, "タイム・コーナー・着順推定域"), (230, 260, "オッズ周辺")]
            shown = set()
            for start, end, label in ranges:
                end = min(end, ret)
                if start >= end: continue
                print(f"\n  --- {label} [{start}:{end}] ---")
                for i in range(start, end, 10):
                    chunk = data[i:min(i+10, end)]
                    if i in shown: continue
                    shown.update(range(i, i+10))
                    try:
                        encoded = chunk.encode('cp932', errors='replace').hex()
                    except Exception:
                        encoded = '?'
                    print(f"  [{i:03d}:{min(i+10,end):03d}]  {repr(chunk):<35} hex={encoded}")
            se_n += 1

        elif rt == "RA" and ra_n < needed_ra:
            v = data[19:21].strip()
            r = data[25:27].strip()
            if target_venue and v != target_venue:
                continue
            if target_race and r != target_race:
                continue

            print(f"\n{'='*70}")
            print(f"[RA #{ra_n+1}]  日付={data[11:19]}  会場={v}  レース={r}")
            print(f"  距離[558:562]={repr(data[558:562])}  コース[566:568]={repr(data[566:568])}")
            print(f"  馬場状態[594:596]={repr(data[594:596])}")
            print(f"{'='*70}")
            # RA: 550-600 range
            for i in range(550, min(610, ret), 10):
                chunk = data[i:min(i+10, ret)]
                try:
                    encoded = chunk.encode('cp932', errors='replace').hex()
                except Exception:
                    encoded = '?'
                print(f"  [{i:03d}:{min(i+10,ret):03d}]  {repr(chunk):<35} hex={encoded}")
            ra_n += 1


# ─────────────────────────────────────────────────────────
# メイン取得ロジック
# ─────────────────────────────────────────────────────────
def fetch_range(jv, from_date, to_date, skip_dates=None):
    """JVOpen(RACE) → SE/RA 読み取り → {日付: [rows]} を返す。"""
    from_dt = from_date + "000000"
    rc, readcnt, dldcnt, lts = jv.JVOpen("RACE", from_dt, 1, 0, 0, "")
    print(f"JVOpen(RACE): rc={rc} readcnt={readcnt} dldcnt={dldcnt}")
    if rc < 0:
        if rc == -1:
            from_dt_prev = (from_date[:6] + "01" + "000000")
            print(f"  rc=-1: 開始日を {from_date[:6]}01 に変更してリトライ")
            jv.JVClose()
            rc, readcnt, dldcnt, lts = jv.JVOpen("RACE", from_dt_prev, 1, 0, 0, "")
            print(f"  JVOpen retry: rc={rc} readcnt={readcnt}")
        if rc < 0:
            raise RuntimeError(f"JVOpen失敗 rc={rc}")

    buf  = " " * 110000
    size = 110000

    race_info = {}   # (kd, venue_cd, race_no) → RA data
    daily     = defaultdict(list)
    n_se = n_ra = n_skip = 0

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

        if rt == "RA":
            ra = parse_ra(data)
            if ra and from_date <= ra['kaisai_date'] <= to_date:
                key = (ra['kaisai_date'], ra['venue_cd'], ra['race_no'])
                race_info[key] = ra
                n_ra += 1

        elif rt == "SE":
            se = parse_se(data, include_unverified=True)
            if se is None:
                continue
            kd = se['kaisai_date']
            if kd < from_date or kd > to_date:
                continue
            if skip_dates and kd in skip_dates:
                continue
            key  = (kd, se['venue_cd'], se['race_no'])
            ra   = race_info.get(key, {})
            venue_name = JYO_NAME.get(se['venue_cd'], se['venue_cd'])

            daily[kd].append({
                '日付':        kd,
                '会場コード':  se['venue_cd'],
                '会場':        venue_name,
                'レースNo':    se['race_no'],
                '距離':        ra.get('distance', ''),
                '芝ダ':        ra.get('surface', ''),
                '馬場状態':    ra.get('baba_state', ''),
                '頭数':        '',  # 後でレースごとに集計
                '馬番':        se['umaban'],
                '馬名':        se['horse_name'],
                '着順':        se['chakujun'],
                '単勝オッズ':  se['tan_odds'],
                '斤量':        se['jinryo'],
                '騎手名':      se['jockey_name'],
                '馬体重':      se['bataiju'],
                '馬体重変化':  se['bataiju_diff'],
                '走破タイム':  se['soha_time'],
                '上り3F':      se['last3f'],
                '1角':         se['corner1'],
                '2角':         se['corner2'],
                '3角':         se['corner3'],
                '4角':         se['corner4'],
            })
            n_se += 1
        else:
            n_skip += 1

    jv.JVClose()
    print(f"SE: {n_se}件 / RA: {n_ra}件 / その他: {n_skip}件")

    # 頭数を補完
    for kd, rows in daily.items():
        from itertools import groupby
        rows_sorted = sorted(rows, key=lambda r: (r['会場コード'], r['レースNo']))
        for _, grp in groupby(rows_sorted, key=lambda r: (r['会場コード'], r['レースNo'])):
            grp_list = list(grp)
            tosu = len(grp_list)
            for row in grp_list:
                row['頭数'] = tosu

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
    ap = argparse.ArgumentParser(description='JV-Link 過去レース結果フェッチャー')
    ap.add_argument('--from',        dest='from_date',  default='20130101')
    ap.add_argument('--to',          dest='to_date',    default='20991231')
    ap.add_argument('--incremental', action='store_true',
                    help='すでに存在する日付をスキップ')
    ap.add_argument('--probe',       action='store_true',
                    help='SE/RAレコード生データを表示して終了')
    ap.add_argument('--probe-race',  nargs=3,
                    metavar=('YYYYMMDD', '会場コード', 'レースNo'),
                    help='特定レースのSEを詳細表示 (例: 20260420 06 11)')
    args = ap.parse_args()

    jv = wc.gencache.EnsureDispatch("JVDTLab.JVLink")
    rc = jv.JVInit("UNKNOWN")
    print(f"JVInit: {rc}")
    if rc != 0:
        print("JVInit失敗。ターゲットFrontierが起動しているか確認してください。")
        sys.exit(1)

    if args.probe:
        from_dt = args.from_date + "000000"
        rc2, _, _, _ = jv.JVOpen("RACE", from_dt, 1, 0, 0, "")
        if rc2 < 0:
            print(f"JVOpen失敗 rc={rc2}")
            sys.exit(1)
        probe_records(jv, n_se=3, n_ra=2)
        return

    if args.probe_race:
        target_date, target_venue, target_race = args.probe_race
        from_dt = target_date + "000000"
        rc2, _, _, _ = jv.JVOpen("RACE", from_dt, 1, 0, 0, "")
        if rc2 < 0:
            print(f"JVOpen失敗 rc={rc2}")
            sys.exit(1)
        probe_records(jv, n_se=20, n_ra=5,
                      target_venue=target_venue.zfill(2),
                      target_race=target_race.zfill(2))
        return

    # 増分モード: 既存ファイルをスキップ
    skip_dates = set()
    if args.incremental:
        existing = _glob.glob(os.path.join(OUT_DIR, '????????.csv'))
        skip_dates = {os.path.basename(f).replace('.csv', '') for f in existing}
        print(f"既存: {len(skip_dates)}日をスキップ")

    daily = fetch_range(jv, args.from_date, args.to_date, skip_dates)

    if not daily:
        print("データが取れませんでした。")
        sys.exit(1)

    save_daily(daily)

    # サンプル表示
    first_date = sorted(daily.keys())[0]
    print(f"\nサンプル ({first_date}) 先頭5行:")
    for r in daily[first_date][:5]:
        print(f"  {r['会場']}{r['レースNo']}R  馬番{r['馬番']}  {r['馬名']}  "
              f"着={r['着順']}  オッズ={r['単勝オッズ']}  "
              f"距離={r['距離']}{r['芝ダ']}  斤量={r['斤量']}  "
              f"馬体重={r['馬体重']}  走破={r['走破タイム']}  上り={r['上り3F']}")

    print("\n--- 未確認フィールド確認 ---")
    print("上記サンプルで 斤量/馬体重/走破タイム/上り3F が正しい値か確認してください。")
    print("違う場合は SE_POS_UNVERIFIED の該当位置を修正して再実行してください。")
    print("詳細確認: python src/fetch.py --probe-race YYYYMMDD 会場コード レースNo")


if __name__ == '__main__':
    main()
