# -*- coding: utf-8 -*-
import sys, re

text = sys.stdin.read()
lines = text.splitlines()
current_se = {}
se_id = None
for line in lines:
    m = re.search(r'\[SE #(\d+)\].*?=(\d{8}).*?=(\d+).*?=(\d+)', line)
    if m:
        se_id = m.group(1)
        ck = re.search(r'\[212:214\]=.(\d+)', line)
        current_se[se_id] = {
            'info': f"SE#{se_id} {m.group(2)} R{m.group(3)} 馬{m.group(4)}",
            'chakujun': ck.group(1) if ck else '?'
        }
    if se_id and '[265:275]' in line:
        val = re.search(r"'([^']+)'", line)
        if val:
            current_se[se_id]['265'] = val.group(1)
    if se_id and '[275:285]' in line:
        val = re.search(r"'([^']+)'", line)
        if val:
            current_se[se_id]['275'] = val.group(1)
    if se_id and '[355:360]' in line:
        val = re.search(r"'([^']+)'", line)
        if val:
            current_se[se_id]['355'] = val.group(1)

print(f"{'SE':20s} {'着順':4s} {'[265:275]':15s} {'[275:285]':15s} {'[355:360]':10s}")
print('-'*75)
for k, v in sorted(current_se.items(), key=lambda x: int(x[0])):
    p265 = v.get('265','?').replace('　','_')
    p275 = v.get('275','?').replace('　','_')
    p355 = v.get('355','?')
    ascii_265 = ''.join(c if c.isdigit() or c in '+-.' else '.' for c in p265)
    print(f"{v['info']:20s} {v['chakujun']:4s} {ascii_265:15s} {p275[:10]:15s} {p355:10s}")
