# -*- coding: utf-8 -*-
"""SE レコードの広い範囲を走査して日本語人名を探す"""
import sys, io, time, re
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

def find_japanese(s):
    """文字列中の日本語(ひらがな/カタカナ/漢字)の位置をリストで返す"""
    results = []
    i = 0
    while i < len(s):
        c = s[i]
        if '぀' <= c <= '鿿' or '゠' <= c <= 'ヿ':
            # 連続する日本語文字を取得
            j = i
            while j < len(s) and ('぀' <= s[j] <= '鿿' or '゠' <= s[j] <= 'ヿ'):
                j += 1
            results.append((i, j, s[i:j]))
            i = j
        else:
            i += 1
    return results

while count < 3:
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

    print(f"=== {kd} / {uma} (騎手: {jockey}) ===")
    print(f"レコード長: {len(data)} chars")
    print()

    # 180-260 の範囲で日本語文字の場所を検索
    chunk = data[180:280]
    jp_positions = find_japanese(chunk)
    print("  [180:280] 内の日本語文字位置:")
    for start, end, text in jp_positions:
        abs_pos = start + 180
        print(f"    [{abs_pos}:{abs_pos + (end-start)}] = '{text}'")

    print()
    # 各フィールド候補を表示
    print("  フィールド候補:")
    for offset in range(185, 250, 4):
        val = data[offset:offset+4].strip()
        if val:
            print(f"    [{offset}:{offset+4}] = '{val}'  |  ", end='')
        if (offset - 185) % 16 == 15:
            print()
    print()

    # 生データ(repr)
    print(f"  data[185:250] repr:")
    print(f"  {repr(data[185:250])}")
    print()

    count += 1

jv.JVClose()
print(f"\n{count}件確認完了")
