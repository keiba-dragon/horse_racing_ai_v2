# -*- coding: utf-8 -*-
"""
SE レコードのバイト位置デバッグ: 調教師名がどのオフセットにあるか確認する
ターゲットFrontierが起動している状態で実行すること
"""
import sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pythoncom
pythoncom.CoInitialize()
import win32com.client as wc

jv = wc.gencache.EnsureDispatch("JVDTLab.JVLink")
rc = jv.JVInit("UNKNOWN")
if rc != 0:
    print(f"JVInit失敗 rc={rc}")
    sys.exit(1)

# 直近7日分
from datetime import datetime, timedelta
from_dt = (datetime.now() - timedelta(days=7)).strftime('%Y%m%d') + "000000"
rc, readcnt, dldcnt, lts = jv.JVOpen("RACE", from_dt, 3, 0, 0, "")
print(f"JVOpen rc={rc} readcnt={readcnt}")

buf = " " * 110000
count = 0

while count < 5:
    try:
        ret, data, sz, fname = jv.JVRead(buf, 110000, "")
    except Exception as e:
        print(f"JVRead例外: {e}")
        break
    if ret == 0:
        break
    if ret in (-1, -3):
        if ret == -3:
            time.sleep(0.05)
        continue
    if ret < 0:
        break

    rt = data[:2]
    if rt != "SE":
        continue

    chakujun = data[212:214].strip() if len(data) > 214 else '??'
    if chakujun != '00':
        continue  # 未確定(出馬表)のみ

    uma     = data[40:58].strip()
    kd      = data[11:19].strip()
    jockey  = data[192:196].strip()

    # バイト位置候補を一括表示
    print(f"\n--- レコード {count+1} ({kd} / {uma}) ---")
    print(f"  騎手名[192:196] = '{jockey}'")
    print(f"  [196:200] = '{data[196:200].strip()}'  (調教師コード候補前半?)")
    print(f"  [196:201] = '{data[196:201].strip()}'  (調教師コード5文字候補)")
    print(f"  [197:201] = '{data[197:201].strip()}'")
    print(f"  [200:204] = '{data[200:204].strip()}'  (調教師名4文字候補?)")
    print(f"  [200:208] = '{data[200:208].strip()}'  (調教師名8文字候補?)")
    print(f"  [201:205] = '{data[201:205].strip()}'  (調教師名4文字候補?)")
    print(f"  [201:209] = '{data[201:209].strip()}'  (調教師名8文字候補 ★現在設定)")
    print(f"  [205:209] = '{data[205:209].strip()}'")
    print(f"  [209:212] = '{data[209:212].strip()}'")
    print(f"  着順[212:214] = '{chakujun}'")
    print(f"  全体[192:214] = '{repr(data[192:214])}'")
    count += 1

jv.JVClose()
print(f"\n{count}件確認完了")
