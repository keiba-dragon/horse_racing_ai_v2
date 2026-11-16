# coding: utf-8
"""
chain_rating.py - 過去レースの「数珠つなぎ」による馬の力比較

競馬の数珠つなぎ（A馬はB馬に先着、B馬はC馬に先着 → A馬>C馬）を数理化したもの。
同じレースに出た馬同士のタイムが橋渡しになり、直接対戦のない馬同士も比べられる。

モデル: log(走破タイム)*100 = レース固有の基準 b_r − 馬の速さ v_h + 誤差
  - b_r が距離・馬場・ペースの違いを吸収する（同じレースの馬同士の差だけが効く）
  - 同じ馬が出た別のレースが「つなぎ目」になって、全馬が1本の物差しに乗る
  - 出走数が少ない馬は0（平均）へ縮める(ridge)。古いレースほど重みを下げる(半減期1年)
  - 芝/ダ別、距離帯別（短≤1400 / 中1401-2000 / 長>2000）に別々に当てはめる
v の単位: 1.0 ≒ 走破タイムの1%（1600mなら約1秒、約5馬身）

使い方: python src/chain_rating.py --date 20261017
"""
import sys, os, re, time, argparse
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

WINDOW_DAYS = 365 * 3
HALF_LIFE_DAYS = 365.0
LAMBDA = 2.0
CLIP_SLOW = 1.06        # 勝ち馬の6%遅れ(1600mで約6秒)より遅い走りは頭打ち（大差・競走中止の歪み防止）
MIN_FINISHERS = 5


def _yymmdd_to_date(v):
    v = int(v)
    return pd.Timestamp(year=2000 + v // 10000, month=(v // 100) % 100, day=v % 100)


def dist_band(d):
    return '短' if d <= 1400 else ('中' if d <= 2000 else '長')


def load_history(asof_yymmdd):
    df = pd.read_parquet(os.path.join(BASE_DIR, 'data', 'processed', 'all_venues_features.parquet'),
                         columns=['日付', '開催', 'Ｒ', '馬名S', '着順_num', '走破タイム_sec', '距離'])
    df['d'] = pd.to_numeric(df['日付'], errors='coerce')
    df['pos'] = pd.to_numeric(df['着順_num'], errors='coerce')
    df['t'] = pd.to_numeric(df['走破タイム_sec'], errors='coerce')
    df = df[(df['d'] < asof_yymmdd) & (df['pos'] > 0) & (df['pos'] < 99) & (df['t'] > 30)].copy()
    df['surface'] = df['距離'].astype(str).str.extract(r'^([芝ダ])')[0]
    df['dist'] = pd.to_numeric(df['距離'].astype(str).str.extract(r'(\d+)')[0], errors='coerce')
    df = df.dropna(subset=['surface', 'dist', 'd'])
    df = df[df['surface'].isin(['芝', 'ダ'])]
    asof = _yymmdd_to_date(asof_yymmdd)
    df['date'] = df['d'].astype(int).map(_yymmdd_to_date)
    df['age'] = (asof - df['date']).dt.days
    df = df[df['age'] <= WINDOW_DAYS].copy()
    df['race_id'] = df['d'].astype(int).astype(str) + '_' + df['開催'].astype(str) + '_' + df['Ｒ'].astype(str)
    # 同一レース・同一馬の重複行を除く
    df = df.drop_duplicates(['race_id', '馬名S'])
    # レース内で勝ち馬比+6%を上限に頭打ち、出走数が少ないレースは除外
    win_t = df.groupby('race_id')['t'].transform('min')
    df['t'] = np.minimum(df['t'], win_t * CLIP_SLOW)
    n = df.groupby('race_id')['t'].transform('size')
    df = df[n >= MIN_FINISHERS].copy()
    df['y'] = 100.0 * np.log(df['t'])
    df['w'] = 0.5 ** (df['age'] / HALF_LIFE_DAYS)
    df['band'] = df['dist'].map(dist_band)
    return df


def fit_ratings(obs, lam=LAMBDA):
    """obs: race_id, 馬名S, y, w → {馬名: (rating, n_runs)}"""
    races = obs['race_id'].astype('category')
    horses = obs['馬名S'].astype('category')
    ri, hi = races.cat.codes.values, horses.cat.codes.values
    n, R, H = len(obs), len(races.cat.categories), len(horses.cat.categories)
    sw = np.sqrt(obs['w'].values)
    rows = np.arange(n)
    A = sp.vstack([
        sp.hstack([sp.csr_matrix((sw, (rows, ri)), shape=(n, R)),
                   sp.csr_matrix((-sw, (rows, hi)), shape=(n, H))]),
        sp.hstack([sp.csr_matrix((H, R)), sp.diags([np.sqrt(lam)] * H)]),
    ]).tocsr()
    b = np.concatenate([sw * obs['y'].values, np.zeros(H)])
    sol = lsqr(A, b, atol=1e-9, btol=1e-9, iter_lim=5000)[0]
    v = sol[R:]
    runs = obs.groupby('馬名S').size()
    names = horses.cat.categories
    return {nm: (float(v[i]), int(runs[nm])) for i, nm in enumerate(names)}


class Graph:
    """馬—レース—馬の2部グラフ。2頭を結ぶ最短の『数珠』を探す。"""
    def __init__(self, obs):
        self.race_horses, self.horse_races, self.info = {}, {}, {}
        for r, h, pos, d, dist in zip(obs['race_id'], obs['馬名S'], obs['pos'], obs['date'], obs['dist']):
            self.race_horses.setdefault(r, []).append(h)
            self.horse_races.setdefault(h, []).append(r)
            self.info[(r, h)] = int(pos)
            self.info[r] = (d, int(dist))

    def path(self, a, b, max_hops=3):
        """a→b の最短連鎖。[(a, race, c), (c, race, b), ...] を返す。無ければ None"""
        if a not in self.horse_races or b not in self.horse_races:
            return None
        frontier = {a: []}
        seen = {a}
        for _ in range(max_hops):
            nxt = {}
            for h, p in frontier.items():
                for r in sorted(self.horse_races[h], key=lambda x: self.info[x][0], reverse=True)[:40]:
                    for c in self.race_horses[r]:
                        if c in seen:
                            continue
                        seen.add(c)
                        np_ = p + [(h, r, c)]
                        if c == b:
                            return np_
                        nxt[c] = np_
            frontier = nxt
            if not frontier:
                break
        return None

    def describe(self, path):
        parts = []
        for h, r, c in path:
            d, dist = self.info[r]
            ph, pc = self.info[(r, h)], self.info[(r, c)]
            parts.append(f"{h}({ph}着)-{c}({pc}着) [{d:%y/%m/%d} {r.split('_')[1]}R{r.split('_')[2]} {dist}m]")
        return ' → '.join(parts)


def fetch_card(date_s):
    import _fetch_and_predict as fp
    races = []
    for rid in fp.get_race_ids(date_s):
        r = fp.parse_shutuba(rid)
        if r and r != fp.UNCONFIRMED:
            races.append(r)
        time.sleep(0.4)
    return races


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', required=True, help='YYYYMMDD')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    asof = int(args.date[2:])

    print(f'履歴を読み込み中（{args.date}より前、直近{WINDOW_DAYS // 365}年）...')
    hist = load_history(asof)
    print(f'  {len(hist):,}頭走 / {hist["race_id"].nunique():,}レース')

    print('出馬表を取得中...')
    races = fetch_card(args.date)
    print(f'  {len(races)}レース')

    fits, graphs, allfit = {}, {}, {}
    for surface in ['芝', 'ダ']:
        sub = hist[hist['surface'] == surface]
        allfit[surface] = fit_ratings(sub)
        for band in ['短', '中', '長']:
            sb = sub[sub['band'] == band]
            fits[(surface, band)] = fit_ratings(sb)
            graphs[(surface, band)] = Graph(sb)
        print(f'  {surface}: 当てはめ完了')

    lines = []
    out_rows = []
    for r in races:
        surface = '芝' if str(r['surface']).startswith('芝') else 'ダ'
        dist = int(r['distance_m'])
        band = dist_band(dist)
        fit, g = fits[(surface, band)], graphs[(surface, band)]
        fa = allfit[surface]
        rows = []
        for h in r['horses']:
            nm = h['馬名S']
            v, n = fit.get(nm, (0.0, 0))
            va, na = fa.get(nm, (0.0, 0))
            rows.append(dict(馬番=h['馬番'], 馬名=nm, v=v, n=n, v全距離=va, n全距離=na))
        rows.sort(key=lambda x: -x['v'])
        top = rows[0]['v'] if rows else 0
        t_sec = dist * (0.0625 if surface == '芝' else 0.0645)
        for rk, x in enumerate(rows, 1):
            x['順位'] = rk
            x['差(馬身)'] = (top - x['v']) / 100.0 * t_sec / 0.2
        lines.append(f"\n■ {r['venue']}{r['r_num']}R {r['class_str']} {surface}{dist}m ({len(rows)}頭)")
        lines.append('  順位 馬番 馬名                  力(%)  差(馬身) 走数  全距離力')
        for x in rows:
            flag = '' if x['n'] >= 3 else ' ※実績少'
            lines.append(f"  {x['順位']:>3}  {x['馬番']:>3}  {x['馬名']:<14}  {x['v']:+6.2f}  {x['差(馬身)']:6.1f}  {x['n']:>3}  {x['v全距離']:+6.2f}{flag}")
            out_rows.append(dict(venue=r['venue'], R=r['r_num'], surface=surface, dist=dist, **x))
        if len(rows) >= 2:
            p = g.path(rows[0]['馬名'], rows[1]['馬名'])
            lines.append('  数珠(1位→2位): ' + (g.describe(p) if p else '3ステップ以内のつながりなし'))
    text = '\n'.join(lines)
    out = args.out or os.path.join(BASE_DIR, 'data', 'processed', f'chain_rating_{args.date}.txt')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(text)
    pd.DataFrame(out_rows).to_csv(out.replace('.txt', '.csv'), index=False, encoding='utf-8-sig')
    print(f'出力: {out}')


if __name__ == '__main__':
    main()
