# coding: utf-8
"""新しいCSVをrecent_all.csvに追記"""
import sys, io, os, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd
from datetime import datetime

NEW_CSV    = r'C:\Users\tsuch\Downloads\2026.4.25～2026.05.03(9日間).csv'
RECENT_CSV = r'C:\horse_racing_ai\data\raw\master\recent_all.csv'

# ── 読み込み ──
print('読み込み中...')
recent = pd.read_csv(RECENT_CSV, encoding='cp932', low_memory=False)
new    = pd.read_csv(NEW_CSV,    encoding='cp932', low_memory=False)
print(f'recent_all: {len(recent):,}行  新ファイル: {len(new):,}行')
print(f'新ファイル 日付範囲: {new["日付(yyyy.mm.dd)"].min()} 〜 {new["日付(yyyy.mm.dd)"].max()}')

# ── 欠損列を生成 ──
def parse_date_cols(date_str):
    """'2026. 4.25' → (260425, '2026.4.25', '土')"""
    WEEKDAYS = ['月','火','水','木','金','土','日']
    try:
        s = str(date_str).strip()
        m = re.match(r'(\d{4})\.\s*(\d{1,2})\.\s*(\d{1,2})', s)
        if not m:
            return None, None, None
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        num = (y - 2000) * 10000 + mo * 100 + d
        s_fmt = f'{y}.{mo}.{d}'
        wd = WEEKDAYS[datetime(y, mo, d).weekday()]
        return num, s_fmt, wd
    except:
        return None, None, None

parsed = new['日付(yyyy.mm.dd)'].apply(parse_date_cols)
new['日付']  = parsed.apply(lambda x: x[0])
new['日付S'] = parsed.apply(lambda x: x[1])
new['曜日']  = parsed.apply(lambda x: x[2])

# ── 列を recent_all に揃える ──
# recent_allにあってnewにない列 → NaN
# newにあってrecent_allにない列 → 除外
missing_cols = [c for c in recent.columns if c not in new.columns]
print(f'\nrecent_allにあってnewにない列: {len(missing_cols)}個')
print(missing_cols[:10])

for c in missing_cols:
    new[c] = None

new_aligned = new[recent.columns]  # recent_all と同じ列順に並べる

# ── 重複チェック（既存の日付は追記しない）──
existing_dates = set(recent['日付'].dropna().astype(int).unique())
new_dates      = set(new_aligned['日付'].dropna().astype(int).unique())
overlap        = existing_dates & new_dates
new_only_dates = new_dates - existing_dates

print(f'\n既存日付: {sorted(existing_dates)[-3:]}... (最新3件)')
print(f'新ファイル日付: {sorted(new_dates)}')
print(f'重複日付: {sorted(overlap)}')
print(f'追記対象日付: {sorted(new_only_dates)}')

rows_to_add = new_aligned[new_aligned['日付'].isin(new_only_dates)]
print(f'\n追記行数: {len(rows_to_add):,}行')

if len(rows_to_add) == 0:
    print('追記するデータがありません。')
    sys.exit(0)

# ── バックアップ ──
backup = RECENT_CSV.replace('.csv', '_backup.csv')
recent.to_csv(backup, index=False, encoding='cp932')
print(f'バックアップ: {backup}')

# ── 追記保存 ──
combined = pd.concat([recent, rows_to_add], ignore_index=True)
combined.to_csv(RECENT_CSV, index=False, encoding='cp932')
print(f'\n保存完了: {RECENT_CSV}')
print(f'行数: {len(recent):,} → {len(combined):,} (+{len(rows_to_add):,}行)')
print(f'最新日付: {int(combined["日付"].max())}')
