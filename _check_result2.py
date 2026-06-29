# coding: utf-8
import sys, re
sys.stdout.reconfigure(encoding="utf-8")

with open("docs/newspaper_20260620.html", encoding="utf-8") as f:
    html = f.read()

# Find row-r1 rows - AI rank 1 horses
r1_rows = re.findall(r'<tr[^>]*class="[^"]*row-r1[^"]*"[^>]*>(.*?)</tr>', html, re.DOTALL)

hits = 0
total_with_result = 0
total_return = 0.0

print(f"{'着順':>4} {'馬名':22} {'騎手':6} {'オッズ':>6} {'EV':>5} {'結果'}")
print("-" * 65)

for row in r1_rows:
    cells = re.findall(r"<td[^>]*>(.*?)</td>", row, re.DOTALL)
    clean = [re.sub(r"<[^>]+>", "", c).strip() for c in cells]

    # Determine if result is present (cell[0] is a number > AI rank)
    # Structure with result: [着順, AIrank=1, EV, 馬名, 騎手, odds, prob]
    # Structure without:     [AIrank=1, EV, 馬名, 騎手, odds, prob]

    if len(clean) >= 7 and clean[1] == '1' and clean[0].lstrip('-').isdigit():
        # Has result
        jyuni = int(clean[0])
        ev = clean[2]
        horse = clean[3]
        jockey = clean[4]
        odds_str = clean[5]
        try:
            odds = float(odds_str)
        except:
            odds = 0.0

        total_with_result += 1
        result_str = "◎ 的中！" if jyuni == 1 else f"{jyuni}着"
        if jyuni == 1:
            hits += 1
            total_return += odds

        horse_short = re.sub(r'\s+\d+\([+-]?\d+\).*', '', horse)
        horse_short = re.sub(r'^\d+\.', '', horse_short).strip()[:16]
        print(f"{jyuni:>4}着  {horse_short:22} {jockey:6} {odds:>6.1f}倍  {ev}  {result_str}")
    elif len(clean) >= 6 and clean[0] == '1' and clean[1].startswith('EV'):
        # No result yet - but wait, some of these DO have '1' as 着順 too
        pass

print(f"\n{'='*65}")
print(f"結果確定: {total_with_result}レース")
if total_with_result > 0:
    print(f"的中数:   {hits}/{total_with_result} ({hits/total_with_result*100:.1f}%)")
    roi = (total_return - total_with_result) / total_with_result * 100
    print(f"回収率:   {total_return:.1f}円 / {total_with_result}レース = {roi:+.1f}%")
