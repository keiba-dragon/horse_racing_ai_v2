# coding: utf-8
"""
nm_course_traits.py - 新モデル用: コースの性質をローカルLLM(Gemma/Ollama)に5軸で点数化させる(1回だけ実行して表にする)

  python src/nm_course_traits.py 芝中        # 芝中(芝1401-2000m)の全コース
出力: data/course_traits_llm.json  {コースキー: {先行有利, 内枠有利, 瞬発力, スタミナ, 小回り, 理由}}  各1〜5
LLMが見るのは data/course_characteristics.json のコース説明と、LLM自身のコース知識だけ。
レースの結果は一切渡さない（過去レースの答えを知っている状態にならない）。
"""
import sys, os, re, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(line_buffering=True)
import gemini_eval as ge

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, 'data', 'course_traits_llm.json')
SRC = os.path.join(BASE, 'data', 'course_characteristics.json')
AXES = {
    '先行有利': '1=差し・追込が有利 / 3=互角 / 5=逃げ・先行が圧倒的に有利',
    '内枠有利': '1=外枠が有利 / 3=互角 / 5=内枠が圧倒的に有利',
    '瞬発力': '1=持続力・ロングスパート戦 / 3=標準 / 5=上がり3Fの瞬発力勝負',
    'スタミナ': '1=スタミナ要求が軽い / 3=標準 / 5=タフでスタミナを強く要求',
    '小回り': '1=大箱で直線が長く紛れにくい / 3=標準 / 5=小回り・急坂・直線が短い',
}
SEG_RANGE = {'芝中': ('芝', 1400, 2000), '芝短': ('芝', 0, 1400), '芝長': ('芝', 2000, 99999),
             'ダ長': ('ダ', 1400, 99999), 'ダ短': ('ダ', 0, 1400)}


def courses_for(seg):
    surf, lo, hi = SEG_RANGE[seg]
    src = json.load(open(SRC, encoding='utf-8'))
    out = []
    for k in src:
        v, s, d = k.split('_')
        if s == surf and lo < int(d) <= hi:
            out.append(k)
    return out, src


def prompt(key, info):
    v, s, d = key.split('_')
    axes = '\n'.join(f'  "{a}": 整数1〜5   ({desc})' for a, desc in AXES.items())
    return (f'あなたはJRAのコースに詳しい競馬分析者です。次のコースの性質を、5つの軸で1〜5の整数で採点してください。\n'
            f'対象: {v}競馬場 {s}{d}m\n'
            f'参考情報(コース説明): {json.dumps(info, ensure_ascii=False)}\n\n'
            f'採点は、このコースの一般的・長期的な傾向について、あなたの知識と上の説明にもとづいて行ってください。\n'
            f'特定のレースの結果を思い出して採点してはいけません。\n'
            f'JSONだけを返してください。形式:\n{{\n{axes},\n  "理由": "30字以内"\n}}')


def parse(text):
    m = re.search(r'\{.*\}', text, re.S)
    d = json.loads(m.group(0))
    out = {}
    for a in AXES:
        x = int(round(float(d[a])))
        out[a] = min(5, max(1, x))
    out['理由'] = str(d.get('理由', ''))[:60]
    return out


def main():
    seg = sys.argv[1] if len(sys.argv) > 1 else '芝中'
    cfg = ge.load_config()
    keys, src = courses_for(seg)
    res = json.load(open(OUT, encoding='utf-8')) if os.path.exists(OUT) else {}
    print(f'{seg}: {len(keys)}コース / 既存{len(res)}件 / model={cfg.get("model")}')
    for k in keys:
        if k in res:
            continue
        t = time.time()
        for attempt in range(3):
            try:
                text = ge._call_local(prompt(k, src[k]), cfg['model'], cfg['local_url'])
                res[k] = parse(text)
                break
            except Exception as e:
                print(f'  [retry {attempt+1}] {k}: {e}')
        if k in res:
            json.dump(res, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
            print(f'{k} {time.time()-t:.0f}s {res[k]}')


if __name__ == '__main__':
    main()
