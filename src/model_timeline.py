"""
model_timeline.py - 新聞一覧(newspapers.html)に出す「モデル・印ルールの切替履歴」と、
各新聞が「いつ作られたか」の判定。

切替履歴は git のコミット履歴(2026-10-21 調査)から作成。変更したら下の ERAS に1行足す。
"""
import os
import subprocess

# (開始日 YYYYMMDD, 区分, 内容)
ERAS = [
    ('20260613', 'モデル', '新聞を的中率モデル(旧accuracy_model)に切替（6/14に一本化）'),
    ('20260621', '買い目', '買い目を「的中率モデル×ROIモデルの両方が1位の馬」だけに絞る'),
    ('20260623', '印ルール', 'z-score印(◎○▲)を導入。買い目=◎○（オッズ不使用）← 現行の印ルール'),
    ('20260815', 'モデル', 'hitrate_model の特徴量を拡張（8/15版・現行モデル）'),
    ('20260905', '参考表示', 'Gemini参考評価を表示開始（買い目には未反映）'),
    ('20261021', '記録', '新聞の最後に使用モデルの版(ハッシュ)を表示・予想ごとに記録開始'),
]
BACKFILL_NOTE = ('2026年1〜7月の新聞は、7/29にまとめて作り直したもの（その時点のモデルで再生成）。'
                 '当時レースの前に出した予想そのものではない。')


def model_label(ds):
    if ds >= '20260815':
        return '8/15版'
    if ds >= '20260613':
        return '旧版'
    return '旧方式'


def rule_label(ds):
    return 'z印' if ds >= '20260623' else ('旧ルール' if ds >= '20260613' else '旧方式')


def first_commit_dates(base_dir):
    """docs/newspaper_*.html が最初にコミットされた日 {ファイル名: YYYYMMDD}。失敗時は空。"""
    try:
        out = subprocess.run(
            ['git', 'log', '--diff-filter=A', '--name-only', '--format=COMMIT:%ad',
             '--date=format:%Y%m%d', '--', 'docs/newspaper_????????.html'],
            cwd=base_dir, capture_output=True, text=True, encoding='utf-8', timeout=60).stdout
    except Exception:
        return {}
    res, cur = {}, None
    for ln in out.splitlines():
        if ln.startswith('COMMIT:'):
            cur = ln[7:]
        elif ln.strip() and cur:
            res[os.path.basename(ln.strip())] = cur   # 新しい順に並ぶので、最後(最古)が残る
    return res


def made_label(ds, first_commit):
    """(表示, css)。未コミット(=今作ったもの)は当日扱い。"""
    if first_commit is None or first_commit == ds:
        return ('当日公開', 'ok')
    if first_commit < ds:
        return ('前日以前に公開', 'ok')
    return ('後日公開', 'late')


def timeline_html():
    rows = ''.join(
        f'<tr><td>{d[4:6].lstrip("0")}/{d[6:].lstrip("0")}〜</td><td>{k}</td><td>{t}</td></tr>'
        for d, k, t in ERAS)
    return (
        '<div class="tl"><h2>モデル・印ルールの切替履歴</h2>'
        f'<table>{rows}</table>'
        f'<p class="note">{BACKFILL_NOTE}</p>'
        '<p class="note">「当日公開」＝その日のうちにGitHubへ載せた新聞、「後日公開」＝日をまたいで載せた新聞'
        '（後日作成・作り直しの可能性が高い）。9/12より前は手動で公開していたため、公開日と作成日は一致しないことがある。'
        '成績を集計するときは、レース前に公開された版だけを使う。</p></div>')
