# coding: utf-8
"""
ROI実験ログ一覧表示

使い方:
    python data/experiments/show_log.py
    python data/experiments/show_log.py --surface ダ
    python data/experiments/show_log.py --bet 単勝
    python data/experiments/show_log.py --detail   # 全フィールド表示
    python data/experiments/show_log.py --last 5   # 直近5件
"""
import sys, io, json, os, argparse
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'roi_log.jsonl')


def load_log():
    if not os.path.exists(LOG_PATH):
        print('ログなし:', LOG_PATH)
        return []
    records = []
    with open(LOG_PATH, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    return records


def fmt(v, width=8, pct=False):
    if v is None or v == '':
        return ' ' * width
    if pct and isinstance(v, float):
        return f'{v:+.1%}'.rjust(width)
    if isinstance(v, float):
        return f'{v:.3f}'.rjust(width)
    return str(v).rjust(width)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--surface', default=None, help='芝 / ダ でフィルタ')
    ap.add_argument('--bet',     default=None, help='単勝 / 複勝 でフィルタ')
    ap.add_argument('--last',    type=int, default=None, help='直近N件')
    ap.add_argument('--detail',  action='store_true', help='全フィールド表示')
    args = ap.parse_args()

    records = load_log()
    if not records:
        return

    # フィルタ
    if args.surface:
        records = [r for r in records
                   if args.surface in str(r.get('selection', {}).get('surface', ''))]
    if args.bet:
        records = [r for r in records if args.bet in str(r.get('bet_type', ''))]
    if args.last:
        records = records[-args.last:]

    SEP = '=' * 110

    if args.detail:
        for i, r in enumerate(records, 1):
            print(f'\n{SEP}')
            print(f' [{i}] {r["ts"]}  {r["name"]}')
            print(SEP)
            print(f'  仮説      : {r.get("hypothesis", "")}')
            print(f'  学習期間  : {r.get("train_period", "")}')
            print(f'  テスト期間: {r.get("test_period", "")}')
            print(f'  カンニング: {r.get("cheat_risk", "")}')
            print(f'  買い目    : {r.get("bet_type", "")}')
            sel = r.get('selection', {})
            print(f'  選択条件  : {json.dumps(sel, ensure_ascii=False)}')
            res = r.get('results', {})
            roi   = res.get('roi_tan', res.get('roi', None))
            roi_f = res.get('roi_fuku', None)
            print(f'  結果      : N={res.get("N","-")}  勝率={res.get("win_rate","")}'
                  f'  複勝率={res.get("place_rate","")}'
                  f'  単ROI={f"{roi:+.1%}" if roi is not None else "-"}'
                  f'  複ROI={f"{roi_f:+.1%}" if roi_f is not None else "-"}'
                  f'  安定={res.get("min_yr_roi","")}/{res.get("plus_years","")}')
            print(f'  結論      : {r.get("conclusion", "")}')
            print(f'  次のアクション: {r.get("next_action", "")}')
            if r.get('notes'):
                print(f'  備考      : {r["notes"]}')
    else:
        # サマリー表
        print(f'\n{SEP}')
        print(f' ROI実験ログ ({len(records)}件)')
        print(SEP)
        print(f'  {"#":>3}  {"日時":^16}  {"実験名":<28}  {"買い目":^6}'
              f'  {"テスト期間":^12}  {"N":>6}  {"単ROI":>7}  {"安定min":>8}'
              f'  {"プラス":>6}  カンニングリスク')
        print('  ' + '-' * 108)
        for i, r in enumerate(records, 1):
            res  = r.get('results', {})
            sel  = r.get('selection', {})
            roi  = res.get('roi_tan', res.get('roi', None))
            roi_s = f'{roi:+.1%}' if roi is not None else '     -'
            myr  = res.get('min_yr_roi', None)
            myr_s = f'{myr:+.1%}' if isinstance(myr, float) else str(myr or '-')
            cheat = r.get('cheat_risk', '')
            cheat_short = cheat[:20] if cheat else ''
            print(f'  {i:>3}  {r["ts"]:^16}  {r["name"]:<28}  {r.get("bet_type",""):^6}'
                  f'  {r.get("test_period",""):^12}  {res.get("N","-"):>6}'
                  f'  {roi_s:>7}  {myr_s:>8}'
                  f'  {str(res.get("plus_years","-")):>6}  {cheat_short}')
        print()


if __name__ == '__main__':
    main()
