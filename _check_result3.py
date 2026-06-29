# coding: utf-8
import sys, re
sys.stdout.reconfigure(encoding="utf-8")

with open("docs/newspaper_20260620.html", encoding="utf-8") as f:
    html = f.read()

r1_rows = re.findall(r'<tr[^>]*class="[^"]*row-r1[^"]*"[^>]*>(.*?)</tr>', html, re.DOTALL)
print(f"row-r1の総数: {len(r1_rows)}")

hits = 0
total = 0
total_return = 0.0
skipped = 0

for row in r1_rows:
    cells = re.findall(r"<td[^>]*>(.*?)</td>", row, re.DOTALL)
    clean = [re.sub(r"<[^>]+>", "", c).strip() for c in cells]

    # 着順あり: 7+セル [着順, AIrank, EV, 馬名, 騎手, odds, prob]
    # 着順なし: 6セル  [AIrank=1, EV, 馬名, 騎手, odds, prob]
    has_result = (len(clean) >= 7 and clean[1] == '1' and clean[0].lstrip('-').isdigit())
    no_result  = (len(clean) >= 6 and clean[0] == '1' and clean[1].startswith('EV'))

    if has_result:
        jyuni = int(clean[0])
        ev = clean[2]
        horse = re.sub(r'\s+\d+\([+-]?\d+\).*', '', clean[3])
        horse = re.sub(r'^\d+\.', '', horse).strip()[:16]
        jockey = clean[4]
        try:
            odds = float(clean[5])
        except:
            odds = 0.0
        total += 1
        if jyuni == 1:
            hits += 1
            total_return += odds
        mark = "◎ 的中！" if jyuni == 1 else f"{jyuni}着"
        print(f"  {jyuni:>2}着 {horse:18} {jockey:6} {odds:.1f}倍 {ev}  {mark}")
    elif no_result:
        horse = re.sub(r'^\d+\.', '', clean[2]).strip()[:16]
        jockey = clean[3]
        try:
            odds = float(clean[4])
        except:
            odds = 0.0
        skipped += 1
        print(f"  --着 {horse:18} {jockey:6} {odds:.1f}倍 {clean[1]}  (結果なし)")
    else:
        print(f"  [不明構造] cells={clean[:5]}")

print(f"\n{'='*60}")
print(f"結果あり: {total}R  結果なし/未確認: {skipped}R")
if total > 0:
    print(f"的中: {hits}/{total} ({hits/total*100:.1f}%)")
    roi = (total_return - total) / total * 100
    print(f"回収率: {total_return:.1f}円 / {total}R = {roi:+.1f}%")
