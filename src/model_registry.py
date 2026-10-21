"""
model_registry.py - 「どの予想を、どのモデルで、いつ出したか」を残す。

- describe(names)        : モデルファイルのハッシュ・更新日時を返す
- log_run(...)           : data/processed/prediction_log.csv に1行追記し、
                           未保存のバージョンは models/versions/ にコピーする
- footer_text(names)     : 新聞フッター用の「使用モデル」文字列
- write_doc()            : MODELS.md の自動生成部分（現行ファイル一覧）を更新する

呼び出し側（予想・新聞の生成）を止めないよう、log_run / footer_text は
失敗しても例外を投げない。
"""
import csv
import hashlib
import os
import shutil
import sys
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
VERSION_DIR = os.path.join(MODEL_DIR, 'versions')
LOG_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'prediction_log.csv')
DOC_PATH = os.path.join(BASE_DIR, 'MODELS.md')

LOG_COLUMNS = ['記録日時', 'スクリプト', '対象日', 'モデル', 'ハッシュ', 'モデル更新日時', 'メモ']

ROLES = {
    'hitrate_model': '的中率モデル（新聞の印・買い目）',
    'roi_model': 'ROIモデル（予想レポートのclogit列）',
}

AUTO_BEGIN = '<!-- AUTO:BEGIN -->'
AUTO_END = '<!-- AUTO:END -->'


def describe(name):
    """models/<name>.pkl のハッシュと更新日時。無ければ None。"""
    path = os.path.join(MODEL_DIR, f'{name}.pkl')
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        sha = hashlib.sha256(f.read()).hexdigest()[:12]
    mtime = datetime.fromtimestamp(os.path.getmtime(path))
    return {'name': name, 'path': path, 'sha': sha,
            'mtime': mtime.strftime('%Y-%m-%d %H:%M'),
            'size': os.path.getsize(path)}


def footer_text(names):
    """例: 使用モデル: hitrate_model[3fa9c2d41b7e / 2026-08-15 更新]"""
    try:
        parts = []
        for n in names:
            d = describe(n)
            if d:
                parts.append(f"{n}[{d['sha']} / {d['mtime'][:10]} 更新]")
        return '使用モデル: ' + ' , '.join(parts) if parts else ''
    except Exception:
        return ''


def _save_version(d):
    """同じハッシュのファイルが無ければ models/versions/ にコピー。"""
    os.makedirs(VERSION_DIR, exist_ok=True)
    dst = os.path.join(VERSION_DIR, f"{d['name']}_{d['sha']}.pkl")
    if not os.path.exists(dst):
        shutil.copy2(d['path'], dst)


def log_run(script, target_date, names, note=''):
    """予想・新聞を出した記録を1行ずつ（モデルごと）追記する。失敗しても無視。"""
    try:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        new_file = not os.path.exists(LOG_PATH)
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(LOG_PATH, 'a', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(LOG_COLUMNS)
            for n in names:
                d = describe(n)
                if d is None:
                    continue
                try:
                    _save_version(d)
                except Exception:
                    pass
                w.writerow([now, script, target_date, n, d['sha'], d['mtime'], note])
    except Exception as e:
        try:
            print(f'[model_registry] 記録に失敗（予想は継続）: {e}')
        except Exception:
            pass


def write_doc():
    """MODELS.md の自動生成部分を更新（手書きの履歴部分は残す）。"""
    rows = ['| モデル | 用途 | ハッシュ | 更新日時 | サイズ |', '|---|---|---|---|---|']
    for n, role in ROLES.items():
        d = describe(n)
        if d:
            rows.append(f"| {n}.pkl | {role} | `{d['sha']}` | {d['mtime']} | {d['size']:,} B |")
        else:
            rows.append(f'| {n}.pkl | {role} | (なし) | | |')
    auto = (f'{AUTO_BEGIN}\n## 現行ファイル（自動生成: `python src/model_registry.py --write-doc`）\n\n'
            + '\n'.join(rows) + f'\n{AUTO_END}')
    if os.path.exists(DOC_PATH):
        text = open(DOC_PATH, encoding='utf-8').read()
        if AUTO_BEGIN in text and AUTO_END in text:
            head = text.split(AUTO_BEGIN)[0]
            tail = text.split(AUTO_END, 1)[1]
            text = head + auto + tail
        else:
            text = text.rstrip() + '\n\n' + auto + '\n'
    else:
        text = '# モデル台帳\n\n' + auto + '\n'
    with open(DOC_PATH, 'w', encoding='utf-8') as f:
        f.write(text)


if __name__ == '__main__':
    if '--write-doc' in sys.argv:
        write_doc()
        print(f'更新: {DOC_PATH}')
    else:
        for n in ROLES:
            print(describe(n))
