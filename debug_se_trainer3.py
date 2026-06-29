# -*- coding: utf-8 -*-
"""SE レコード全体を走査して全日本語文字の位置を列挙する"""
import sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pythoncom
pythoncom.CoInitialize()
import win32com.client as wc

jv = wc.gencache.EnsureDispatch("JVDTLab.JVLink")
rc = jv.JVInit("UNKNOWN")
if rc != 0:
    print(f"JVInit失敗 rc={rc}"); sys.exit(1)

from datetime import datetime, timedelta
from_dt = (datetime.now() - timedelta(days=7)).strftime('%Y%m%d') + "000000"
rc, readcnt, dldcnt, lts = jv.JVOpen("RACE", from_dt, 3, 0, 0, "")
print(f"JVOpen rc={rc} readcnt={readcnt}\n")

buf = " " * 110000
count = 0

def is_jp(c):
    return '぀' <= c <= '鿿' or '゠' <= c <= 'ヿ' or '　' == c

while count < 2:
    try:
        ret, data, sz, fname = jv.JVRead(buf, 110000, "")
    except Exception as e:
        print(f"JVRead例外: {e}"); break
    if ret == 0: break
    if ret in (-1, -3):
        if ret == -3: time.sleep(0.05)
        continue
    if ret < 0: break

    rt = data[:2]
    if rt != "SE": continue
    chakujun = data[212:214].strip() if len(data) > 214 else '??'
    if chakujun != '00': continue

    uma    = data[40:58].strip()
    kd     = data[11:19].strip()
    jockey = data[192:196].strip()

    print(f"=== {kd} / {uma} (騎手: {jockey}) | 全長: {len(data)} chars ===")

    # 全体で日本語文字グループを探す
    i = 0
    groups = []
    while i < len(data):
        if is_jp(data[i]):
            j = i
            while j < len(data) and is_jp(data[j]):
                j += 1
            text = data[i:j].strip()
            if text:
                groups.append((i, j, text))
            i = j
        else:
            i += 1

    print("  日本語文字グループ一覧:")
    for start, end, text in groups:
        print(f"    [{start}:{end}] (幅{end-start}) = '{text}'")

    # また全体の repr を50文字ずつ表示
    print("\n  全体repr (50文字ずつ):")
    for chunk_start in range(0, len(data), 50):
        chunk = data[chunk_start:chunk_start+50]
        print(f"    [{chunk_start:3d}] {repr(chunk)}")

    print()
    count += 1

jv.JVClose()
print(f"\n{count}件確認完了")
