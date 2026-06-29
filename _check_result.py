# coding: utf-8
import sys, re
sys.stdout.reconfigure(encoding="utf-8")

with open("docs/newspaper_20260620.html", encoding="utf-8") as f:
    html = f.read()

# Find row-r1 (AI 1位馬) rows
r1_rows = re.findall(r'<tr[^>]*class="[^"]*row-r1[^"]*"[^>]*>(.*?)</tr>', html, re.DOTALL)
print(f"AI1位馬の行: {len(r1_rows)}件")

for i, row in enumerate(r1_rows[:36]):
    cells = re.findall(r"<td[^>]*>(.*?)</td>", row, re.DOTALL)
    clean = [re.sub(r"<[^>]+>", "", c).strip() for c in cells]
    print(f"  {clean[:7]}")

# Count td-jyuni (着順 result cells)
jyuni = re.findall(r'<td[^>]*class="[^"]*td-jyuni[^"]*"[^>]*>(.*?)</td>', html, re.DOTALL)
print(f"\n着順セル: {len(jyuni)}件")
print("sample:", [re.sub(r"<[^>]+>","",j).strip() for j in jyuni[:10]])
