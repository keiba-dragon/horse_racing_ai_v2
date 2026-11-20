# coding: utf-8
"""
nm_eval_traits.py - LLMのコース判定(data/course_traits_llm.json)を「馬の特徴との掛け算」でモデルに足し、
基準線(NEW=体重・騎手なし)と同じ手順・同じ期間で比べる。既存モデルには触らない。

線形モデルは「特徴1つに係数1つ」なので、コースで向きが変わる効果は表せない。
  コースの点数(-1〜+1) × 馬の特徴  という掛け算の列を足すと、向きがコースで変わる効果を表せる。
  (コース単独の列は、同じレース内で全馬同じ値になり、勝敗に効かないので入れない)
  python src/nm_eval_traits.py 芝中
"""
import sys, os, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(line_buffering=True)
import nm_baseline as nb

BASE = nb.BASE_DIR
V = {'東': '東京', '中': '中山', '京': '京都', '阪': '阪神', '名': '中京', '新': '新潟', '福': '福島', '札': '札幌', '函': '函館', '小': '小倉'}


def add_interactions(df, traits):
    n = pd.to_numeric(df['頭数'], errors='coerce')
    dm = pd.to_numeric(df['距離'].astype(str).str.extract(r'(\d+)')[0], errors='coerce')
    venue = df['開催'].astype(str).str.extract(r'([東中京阪名新福札函小])')[0].map(V)
    key = venue + '_芝_' + dm.astype('Int64').astype(str)
    for ax in ['先行有利', '内枠有利', '瞬発力', 'スタミナ', '小回り']:
        df['c_' + ax] = key.map(lambda k, ax=ax: (traits[k][ax] - 3) / 2 if k in traits else 0.0)
    # 馬の特徴(出走前に分かるものだけ)
    pos = []
    for i in (1, 2, 3):
        p = pd.to_numeric(df[f'{i}走前_3角'], errors='coerce') / pd.to_numeric(df[f'{i}走前_頭数'], errors='coerce')
        pos.append(p)
    front = 1 - pd.concat(pos, axis=1).mean(axis=1)                      # 前に行く馬ほど大きい(0〜1)
    df['h_front'] = (front - 0.5).fillna(0.0)
    df['h_draw'] = (0.5 - (pd.to_numeric(df['馬番'], errors='coerce') - 1) / (n - 1)).fillna(0.0)   # 内枠ほど大きい
    last3 = pd.to_numeric(df['近5走_上り3F平均'], errors='coerce')
    df['_l3'] = last3
    df['h_kick'] = (-(last3 - df.groupby('race_id')['_l3'].transform('mean')) / 0.5).clip(-3, 3).fillna(0.0)   # 同じレースの平均より速いほど大きい
    df['h_ext'] = (pd.to_numeric(df['距離変化_前走'], errors='coerce') / 400).clip(-1.5, 1.5).fillna(0.0)    # 前走より距離が延びるほど大きい
    df['X先行×脚質'] = df['c_先行有利'] * df['h_front']
    df['X内枠×枠'] = df['c_内枠有利'] * df['h_draw']
    df['X瞬発×上がり'] = df['c_瞬発力'] * df['h_kick']
    df['X小回り×脚質'] = df['c_小回り'] * df['h_front']
    df['X小回り×枠'] = df['c_小回り'] * df['h_draw']
    df['X坂×延長'] = df['c_スタミナ'] * df['h_ext']
    return ['X先行×脚質', 'X内枠×枠', 'X瞬発×上がり', 'X小回り×脚質', 'X小回り×枠', 'X坂×延長']


def main():
    seg = sys.argv[1] if len(sys.argv) > 1 else '芝中'
    traits = json.load(open(os.path.join(BASE, 'data', 'course_traits_llm.json'), encoding='utf-8'))
    allf = nb.all_features(); newf = [f for f in allf if not nb.is_removed(f)]
    df = nb.load_segment(seg); nb.to_num(df, allf)
    inter = add_interactions(df, traits)
    cover = (df['c_先行有利'] != 0).mean()
    print(f'{seg}: {len(df):,}行 / コース判定あり {cover*100:.1f}% / 追加列={inter}')
    rows = {}
    for label, feats in [('NEW', newf), ('NEW+コース掛け算', newf + inter)]:
        out, valid, beta, _ = nb.fit_eval(df, feats)
        rows[label] = out
        print(f'\n[{label}] {len(valid)}次元')
        for p, r in out.items():
            print(f'  {p} R={r["races"]} NLL={r["nll"]:.4f} 勝率={r["win"]*100:.1f}% 順位ずれ={r["gap"]:.3f} (人気{r["fav_gap"]:.3f}) 本命平均着={r["fin"]:.2f}')
        if label != 'NEW':
            print('  追加列の係数(標準化後):', {f: round(float(b), 3) for f, b in zip(valid, beta) if f in inter})
    print('\n=== 2023-2025(R数加重) ===')
    for label, o in rows.items():
        ps = ['2023', '2024', '2025']; w = np.array([o[p]['races'] for p in ps], float)
        g = lambda k: float(np.average([o[p][k] for p in ps], weights=w))
        print(f'{label:16} NLL={g("nll"):.4f} 勝率={g("win")*100:.2f}% 順位ずれ={g("gap"):.4f} 本命平均着={g("fin"):.3f}')


if __name__ == '__main__':
    main()
