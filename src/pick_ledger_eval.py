"""pick_ledger_eval.py - 台帳(レース前の予想)を、ネット競馬の確定結果と突き合わせて集計する。

  python src/pick_ledger_eval.py   （リポジトリ直下で実行）
- 結果は data/processed/pick_ledger_results.csv にキャッシュ(確定済みの日は再取得しない)
- 取消・除外(着順なし・オッズなし)は返金扱いで、的中率・回収率の分母から外す
- 単勝/複勝の回収率は100円買った場合の倍率(払戻/100)
"""
import sys, os
sys.path.insert(0, 'src')
import numpy as np, pandas as pd
import make_newspaper as mn
import _fetch_and_predict as fp
import pick_ledger as pl

short2code = {v: k for k, v in fp.VENUE_CODE_TO_SHORT.items()}
CACHE = os.path.join(pl.ROOT, 'data', 'processed', 'pick_ledger_results.csv')


def fetch_results(led: pd.DataFrame) -> pd.DataFrame:
    cache = pd.read_csv(CACHE, dtype={'date': str}) if os.path.exists(CACHE) else pd.DataFrame(
        columns=['date', 'venue', 'rnum', 'no', 'rank', 'odds_res', 'tan', 'fuku', 'scratched'])
    have = set(zip(cache.date, cache.venue, cache.rnum))
    rows = cache.to_dict('records')
    for date in sorted(led.date.unique()):
        need = led[(led.date == date) & ~led.apply(lambda r: (r.date, r.venue, r.rnum) in have, axis=1)]
        if need.empty:
            continue
        res = mn.fetch_race_results(fp.get_race_ids(date))
        for _, r in need.iterrows():
            rr = res.get((short2code.get(r.venue), int(r.rnum)))
            if rr is None:
                continue
            k = str(int(r.no)).zfill(2)
            rank = rr['order'].get(k, np.nan)
            od = rr['odds'].get(k, np.nan)
            rows.append(dict(date=r.date, venue=r.venue, rnum=r.rnum, no=r.no, rank=rank, odds_res=od,
                             tan=dict(rr['tansho']).get(k, 0), fuku=dict(rr['fukusho']).get(k, 0),
                             scratched=int(rank != rank and od != od)))
        print(date, 'ok', flush=True)
        pd.DataFrame(rows).to_csv(CACHE, index=False, encoding='utf-8')
    return pd.DataFrame(rows)


def summ(g):
    return pd.Series(dict(n=len(g), 勝率=g.win.mean(), 複勝率=g.place.mean(), 単勝回収=g.tan.mean() / 100, 複勝回収=g.fuku.mean() / 100))


def main():
    led = pl.load_ledger()
    res = fetch_results(led)
    m = led.merge(res[['date', 'venue', 'rnum', 'rank', 'tan', 'fuku', 'scratched']], on=['date', 'venue', 'rnum'], how='left')
    m['has'] = m['scratched'].notna()
    print(f'台帳 {len(m)}レース / 結果あり {m.has.sum()} / 取消(返金) {int((m.scratched == 1).sum())} / 結果なし {(~m.has).sum()}')
    d = m[m.has & (m.scratched == 0)].copy()
    d['win'] = (d['rank'] == 1).astype(int)
    d['place'] = (d['rank'] <= 3).astype(int)
    d['dd'] = pd.to_datetime(d.date)
    d['wk'] = d.dd.dt.to_period('W-SUN').astype(str)
    pd.set_option('display.width', 200)
    last = d.dd.max()
    print('\n=== 印別(9/13以降 全期間) ===')
    print(d.groupby('mark').apply(summ, include_groups=False).round(3).to_string())
    print('◎○', summ(d[d.mark.isin(['◎', '○'])]).round(3).to_dict())
    print('全体', summ(d).round(3).to_dict())
    for label, days in [('直近8週', 56), ('直近4週', 28), ('直近2週', 14)]:
        x = d[d.dd >= last - pd.Timedelta(days=days)]
        print(f'\n=== {label} {x.date.min()}〜{x.date.max()} ===')
        print(x.groupby('mark').apply(summ, include_groups=False).round(3).to_string())
        print('◎○', summ(x[x.mark.isin(['◎', '○'])]).round(3).to_dict())
    print('\n=== ◎○ 週別 ===')
    print(d[d.mark.isin(['◎', '○'])].groupby('wk').apply(summ, include_groups=False).round(3).to_string())
    print('\n=== ▲ セグメント別 ===')
    print(d[d.mark == '▲'].groupby('seg').apply(summ, include_groups=False).round(3).to_string())
    print('\n=== ◎○ セグメント別 ===')
    print(d[d.mark.isin(['◎', '○'])].groupby('seg').apply(summ, include_groups=False).round(3).to_string())
    d['late'] = d.rnum >= 11
    print('\n=== 11R/12R vs それ以前 ===')
    print(d.groupby('late').apply(summ, include_groups=False).round(3).to_string())
    print('\n=== 版による本命のブレ ===')
    print('最初に見た版と最後の版で本命の馬が違うレース:', f'{(m.first_no != m.no).mean():.1%}')
    m['dow'] = pd.to_datetime(m.date).dt.dayofweek
    print((m.assign(chg=(m.first_no != m.no)).groupby('dow').chg.mean()).round(3).rename({5: '土', 6: '日'}).to_string())
    m.to_csv(os.path.join(pl.ROOT, 'logs', 'pick_ledger_eval.csv'), index=False, encoding='utf-8')


if __name__ == '__main__':
    main()
