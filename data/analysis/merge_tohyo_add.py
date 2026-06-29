import pandas as pd
import glob, re, sys
sys.stdout.reconfigure(encoding='utf-8')

def parse_amount(val):
    s = str(val).strip()
    if s in ('', 'nan', '-'):
        return 0
    # 全角・半角スラッシュ両対応（フォーメーション: "100／800" → 800が合計）
    if '／' in s or '/' in s:
        parts = re.split(r'[／/]', s)
        for p in reversed(parts):
            p = p.strip().replace(',', '')
            if re.match(r'^\d+$', p):
                return int(p)
        return 0
    try:
        return float(str(s).replace(',', ''))
    except:
        return 0

# 既存データ
existing = pd.read_csv('data/tohyo/all_tohyo.csv', encoding='utf-8-sig')
print(f'既存: {len(existing)}行, 最終日: {existing["日付"].max()}')

# アーカイブから新データ追加
new_dfs = []
for f in sorted(glob.glob('data/tohyo/archive/*_tohyo.csv')):
    for enc in ['cp932', 'utf-8-sig', 'utf-8']:
        try:
            df = pd.read_csv(f, encoding=enc)
            new_dfs.append(df)
            print(f'  読込: {f} ({len(df)}行)')
            break
        except Exception as e:
            pass

if not new_dfs:
    print('新規ファイルなし')
    sys.exit(0)

new_data = pd.concat(new_dfs, ignore_index=True)

# 金額を数値に変換
for col in ['購入金額', '払戻金額', '返還金額']:
    if col in new_data.columns:
        new_data[col] = new_data[col].apply(parse_amount)

# 日付を整数に
new_data['日付'] = new_data['日付'].apply(
    lambda x: int(float(str(x))) if str(x).replace('.', '').isdigit() else None)
new_data = new_data[new_data['日付'].notna()].copy()
new_data['日付'] = new_data['日付'].astype(int)

# 既存の日付と重複しないものだけ追加
existing_dates = set(existing['日付'].unique())
new_dates = set(new_data['日付'].unique())
overlap = existing_dates & new_dates
if overlap:
    print(f'重複日付をスキップ: {sorted(overlap)}')
    new_data = new_data[~new_data['日付'].isin(overlap)]

all_df = pd.concat([existing, new_data], ignore_index=True)
all_df = all_df.sort_values(['日付', '受付番号', '通番']).reset_index(drop=True)
all_df.to_csv('data/tohyo/all_tohyo.csv', index=False, encoding='utf-8-sig')

buy = all_df['購入金額'].sum()
pay = all_df['払戻金額'].sum()
ret = all_df['返還金額'].sum()
net = pay + ret - buy
roi = net / buy * 100 if buy > 0 else 0

print(f'\n保存完了: {len(all_df)}行')
print(f'日付: {sorted(all_df["日付"].unique())}')
print(f'購入合計: {buy:,.0f}円')
print(f'払戻合計: {pay:,.0f}円')
print(f'損益: {net:+,.0f}円 / ROI: {roi:+.1f}%')
