# coding: utf-8
"""
nm_eval_self.py - 「その馬自身の普段と比べた特徴」を足して、基準線(NEW)と同じ手順で比べる。既存モデルには触らない。
同じ値でも、馬によって好材料か悪材料かが違う。馬ごとの平均との差にすると、その馬にとってのプラス/マイナスになる。
  python src/nm_eval_self.py 芝中
"""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(line_buffering=True)
import nm_baseline as nb


def num(df, c):
    return pd.to_numeric(df[c], errors='coerce')


def add_self(df):
    new = {}
    past = lambda pat, ks: pd.concat([num(df, pat.format(k)) for k in ks], axis=1)
    # 前走の指数/上がり/ポジション/間隔/斤量が、その馬の普段(2〜5走前の平均)よりどうだったか
    new['S_指数_前走-普段'] = num(df, '1走前_タイム指数') - past('{}走前_タイム指数', [2, 3, 4, 5]).mean(axis=1)
    new['S_上り_前走-普段'] = past('{}走前_上り3F', [2, 3, 4, 5]).mean(axis=1) - num(df, '1走前_上り3F')   # 速いほど+
    pos = lambda k: num(df, f'{k}走前_3角') / num(df, f'{k}走前_頭数')
    new['S_位置_前走-普段'] = pd.concat([pos(k) for k in [2, 3, 4, 5]], axis=1).mean(axis=1) - pos(1)     # いつもより前に行ったら+
    new['S_間隔_今回-普段'] = num(df, '間隔') - past('{}走前_間隔', [1, 2, 3, 4]).mean(axis=1)
    new['S_斤量_今回-前走'] = num(df, '斤量') - num(df, '1走前_斤量')
    new['S_斤量_今回-普段'] = num(df, '斤量') - past('{}走前_斤量', [1, 2, 3, 4, 5]).mean(axis=1)
    new['S_距離_今回-普段'] = num(df, '距離変化_前走') - 0  # 参考(既存列)。馬ごとの普段との差は下で
    dist_now = pd.to_numeric(df['距離'].astype(str).str.extract(r'(\d+)')[0], errors='coerce')
    new['S_距離_今回-過去平均'] = dist_now - past('{}走前_距離', [1, 2, 3, 4, 5]).mean(axis=1)
    # レース内で標準化(同じレースの他馬との相対にして、極端値を抑える)
    out = []
    for k, v in new.items():
        df[k] = v.clip(v.quantile(0.01), v.quantile(0.99))
        out.append(k)
    return out


def main():
    seg = sys.argv[1] if len(sys.argv) > 1 else '芝中'
    allf = nb.all_features(); newf = [f for f in allf if not nb.is_removed(f)]
    df = nb.load_segment(seg); nb.to_num(df, allf)
    selfs = add_self(df)
    print(f'{seg}: 追加列={selfs}')
    rows = {}
    for label, feats in [('NEW', newf), ('NEW+自己比較', newf + selfs)]:
        out, valid, beta, _ = nb.fit_eval(df, feats)
        rows[label] = out
        print(f'\n[{label}] {len(valid)}次元')
        for p, r in out.items():
            print(f'  {p} R={r["races"]} NLL={r["nll"]:.4f} 勝率={r["win"]*100:.1f}% 順位ずれ={r["gap"]:.3f} (人気{r["fav_gap"]:.3f}) 本命平均着={r["fin"]:.2f}')
        if label != 'NEW':
            print('  追加列の係数:', {f: round(float(b), 3) for f, b in zip(valid, beta) if f in selfs})
    print('\n=== 2023-2025(R数加重) ===')
    for label, o in rows.items():
        ps = ['2023', '2024', '2025']; w = np.array([o[p]['races'] for p in ps], float)
        g = lambda k: float(np.average([o[p][k] for p in ps], weights=w))
        print(f'{label:12} NLL={g("nll"):.4f} 勝率={g("win")*100:.2f}% 順位ずれ={g("gap"):.4f} 本命平均着={g("fin"):.3f}')


if __name__ == '__main__':
    main()
