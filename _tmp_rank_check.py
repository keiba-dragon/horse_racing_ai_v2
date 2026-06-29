# coding: utf-8
import re
from collections import Counter

with open('docs/newspaper_20260620.html', encoding='utf-8') as f:
    html = f.read()

ranks = re.findall(r'<td class="td-rank">(.*?)</td>', html)
print('順位の分布:', Counter(ranks).most_common(10))
print('サンプル(最初の30件):', ranks[:30])
