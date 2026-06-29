# -*- coding: utf-8 -*-
"""
JV-Link 出馬表カードビルダー
RA（距離・コース）+ SE（未確定 = 出馬表）から netkeibanote 互換カード CSV を生成する。

使い方:
  python src/jvlink/jvlink_card_builder.py              # 今週末分を自動生成
  python src/jvlink/jvlink_card_builder.py --date 20260503  # 特定日
  python src/jvlink/jvlink_card_builder.py --from 20260501  # 指定日以降の全日付

出力:
  data/raw/cards/jvlink/YYYYMMDD.csv  (日ごとに分割)
  → convert_card_to_base_format() で読み込める netkeibanote 互換形式

注意:
  - 斤量 は SE record Unicode char 174-177（0.1kg単位）から取得
  - 騎手名 は SE record Unicode char 192-200（8文字フィールド）から取得
  - 調教師名 は SE record Unicode char 72-76（馬名直後の4文字フィールド）から取得
  - 単オッズ は JV-Link RACE spec に含まれないため NaN
  - 障害レース (track>=55 or ==52 except ダ) は自動除外
"""
import sys, io, os, csv, argparse, time
from collections import defaultdict
from datetime import datetime, timedelta

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import pythoncom
pythoncom.CoInitialize()
import win32com.client as wc
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

JYO_NAME = {
    "01": "札", "02": "函", "03": "福", "04": "新", "05": "東",
    "06": "中", "07": "中京", "08": "京", "09": "阪", "10": "小",
}

def track_to_surface(code_str):
    """JRA-VAN トラックコード → '芝' / 'ダ' / '障'"""
    try:
        c = int(code_str.strip())
    except Exception:
        return ''
    if c == 24: return 'ダ'
    if c in (51, 52, 53, 54): return 'ダ'
    if c >= 55: return '障'
    if 10 <= c <= 29: return '芝'
    return ''

def kaisai_to_date_s(kaisai_date):
    """'20260503' → '2026.5.3'"""
    try:
        y, m, d = int(kaisai_date[:4]), int(kaisai_date[4:6]), int(kaisai_date[6:8])
        return f"{y}.{m}.{d}"
    except Exception:
        return kaisai_date

def fetch_upcoming_races(from_date_str, to_date_str=None):
    """
    JV-Link から RA + SE(出馬表) を取得し、
    {kaisai_date: {(venue_cd, race_no): {'dist', 'surface', 'horses': [名前...]}}} を返す
    """
    jv = wc.gencache.EnsureDispatch("JVDTLab.JVLink")
    rc = jv.JVInit("UNKNOWN")
    if rc != 0:
        print("JVInit失敗。ターゲットFrontierが起動しているか確認してください。")
        sys.exit(1)

    # レースデータはレース日より前にJVLinkへアップロードされるため、
    # 7日前から取得してレース日フィルタで絞る
    fetch_from = (datetime.strptime(from_date_str, '%Y%m%d') - timedelta(days=7)).strftime('%Y%m%d')
    from_dt = fetch_from + "000000"
    rc, readcnt, dldcnt, lts = jv.JVOpen("RACE", from_dt, 3, 0, 0, "")
    print(f"JVOpen rc={rc} readcnt={readcnt}")
    if rc not in (0, 1):
        print(f"JVOpenエラー rc={rc}")
        jv.JVClose()
        sys.exit(1)

    buf = " " * 110000
    ra_data = {}     # (kaisai_date, venue_cd, race_no) → (dist, track_code)
    se_upcoming = defaultdict(list)  # (kaisai_date, venue_cd, race_no) → [(horse_name, umaban), ...]

    n_ra = n_se_up = n_se_done = 0

    while True:
        try:
            ret, data, sz, fname = jv.JVRead(buf, 110000, "")
        except Exception as e:
            print(f"JVRead例外: {e}")
            break
        if ret == 0:
            print("EOF")
            break
        if ret == -1: continue
        if ret == -3:
            time.sleep(0.05)
            continue
        if ret < 0:
            print(f"JVReadエラー: {ret}")
            break

        rt = data[:2]
        kd = data[11:19].strip()

        # 日付フィルタ
        if kd < from_date_str:
            continue
        if to_date_str and kd > to_date_str:
            continue

        vc = data[19:21].strip()
        rn = data[25:27].strip()
        key = (kd, vc, rn)

        if rt == "RA":
            if len(data) > 568:
                dist = data[558:562].strip()
                track = data[566:568].strip()
                if dist.isdigit() and int(dist) > 0:
                    ra_data[key] = (dist, track)
                    n_ra += 1

        elif rt == "SE":
            chakujun = data[212:214].strip() if len(data) > 214 else '??'
            uma = data[40:58].strip() if len(data) > 58 else ''
            umaban_raw = data[28:30].strip() if len(data) > 30 else ''
            umaban = int(umaban_raw) if umaban_raw.isdigit() and int(umaban_raw) > 0 else None
            # 騎手名 (Unicode char 192-200) / 調教師名 (Unicode char 72-76) / 斤量 (174-177, 単位0.1kg)
            jockey = data[192:200].strip() if len(data) > 200 else ''
            # 調教師名: 馬名フィールド(40-58)の直後 [72:76] (4 Unicode chars = 4 kanji max)
            trainer = data[72:76].strip() if len(data) > 76 else ''
            jinryo_raw = data[174:177].strip() if len(data) > 177 else ''
            try:
                jinryo = int(jinryo_raw) / 10.0 if jinryo_raw.isdigit() and int(jinryo_raw) > 0 else np.nan
            except Exception:
                jinryo = np.nan
            if not uma:
                continue
            if chakujun == '00':
                # 未確定 → 出馬表エントリー
                se_upcoming[key].append((uma, umaban, jockey, jinryo, trainer))
                n_se_up += 1
            else:
                n_se_done += 1

    jv.JVClose()
    print(f"RA: {n_ra}件 / SE(出馬表): {n_se_up}件 / SE(結果済): {n_se_done}件")

    return ra_data, se_upcoming


def build_card_df(ra_data, se_upcoming, exclude_shoegai=True):
    """RA + SE から card DataFrame を構築"""
    rows = []
    for key, (dist, track) in sorted(ra_data.items()):
        kd, vc, rn = key
        surface = track_to_surface(track)

        # 障害レースを除外
        if exclude_shoegai and surface == '障':
            continue

        horses = se_upcoming.get(key, [])
        if not horses:
            continue

        jyo = JYO_NAME.get(vc, vc)
        date_s = kaisai_to_date_s(kd)
        ba_r = f"{jyo}{int(rn)}"

        for horse, umaban, jockey, jinryo, trainer in horses:
            rows.append({
                '日付S':   date_s,
                '場 R':   ba_r,
                '馬名S':  horse,
                '馬番':    umaban,
                '芝ダ':   surface,
                '距離':   int(dist),
                '単オッズ': np.nan,
                '斤量':   jinryo,
                '騎手':   jockey if jockey else np.nan,
                '調教師':  trainer if trainer else np.nan,
            })

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', help='特定日 YYYYMMDD')
    ap.add_argument('--from', dest='from_date', default=None, help='開始日 YYYYMMDD')
    ap.add_argument('--to', dest='to_date', default=None, help='終了日 YYYYMMDD')
    ap.add_argument('--out', default=None, help='出力ディレクトリ')
    args = ap.parse_args()

    today = datetime.now().strftime('%Y%m%d')

    if args.date:
        from_date = args.date
        to_date   = args.date
    elif args.from_date:
        from_date = args.from_date
        to_date   = args.to_date
    else:
        # デフォルト: 今日〜2週間後
        from_date = today
        to_date   = (datetime.now() + timedelta(days=14)).strftime('%Y%m%d')

    print(f"取得期間: {from_date} 〜 {to_date or '制限なし'}")

    ra_data, se_upcoming = fetch_upcoming_races(from_date, to_date)

    # 日付別に分割して保存
    out_dir = args.out or os.path.join(BASE_DIR, 'data', 'raw', 'cards', 'jvlink')
    os.makedirs(out_dir, exist_ok=True)

    # 日付をグループ化
    dates = sorted(set(kd for (kd, _, _) in ra_data.keys()))
    print(f"\n対象日: {dates}")

    all_saved = 0
    for kd in dates:
        ra_day = {k: v for k, v in ra_data.items() if k[0] == kd}
        se_day = {k: v for k, v in se_upcoming.items() if k[0] == kd}

        df = build_card_df(ra_day, se_day)
        if df.empty:
            print(f"{kd}: データなし (RAのみ or 全障害)")
            continue

        out_path = os.path.join(out_dir, f'{kd}.csv')
        df.to_csv(out_path, index=False, encoding='cp932')
        n_races = df['場 R'].nunique()
        n_horses = len(df)
        print(f"{kd}: {n_races}R {n_horses}頭 → {out_path}")
        all_saved += n_horses

        # サマリー表示
        for ba_r, grp in df.groupby('場 R', sort=False):
            surface = grp['芝ダ'].iloc[0]
            dist    = int(grp['距離'].iloc[0])
            n_h     = len(grp)
            print(f"  {ba_r}: {surface}{dist}m {n_h}頭")

    print(f"\n合計 {all_saved}行 → {out_dir}/")


if __name__ == '__main__':
    main()
