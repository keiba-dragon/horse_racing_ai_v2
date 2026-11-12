# -*- coding: utf-8 -*-
"""
Gemini API による生成AI評価レイヤー（v3 Phase 3・シャドウモード）。

統計モデル（hitrate_model, 新聞の◎○▲判定に使用）の上位候補馬について、
Gemini APIにレース状況・近走サマリを渡して「注目馬とその理由・0-100点評価」を取得する。

現段階ではシャドウモード: 新聞に参考表示するのみで、買い目（◎○の選定）には一切反映しない。
CLAUDE.md/メモリの「roi_model提案禁止」方針と同様、既存の的中率選定ロジックを
生成AIスコアで上書き・介入することはしない（8週間のシャドウ検証後に別途判断する）。

失敗時は例外を投げず None を返す（新聞生成を絶対に止めない）。

設定ファイル: config/gemini_config.json （config/gemini_config.json.example参照）
  Gemini API:
  {
    "api_key": "...",
    "model": "gemini-flash-latest",
    "enabled": true
  }
  ローカルLLM（Ollama等のOpenAI互換API。APIキー不要・無料・レート制限なし）:
  {
    "backend": "local",
    "model": "gemma4:e4b",
    "local_url": "http://localhost:11434/v1/chat/completions",
    "include_features": true,
    "enabled": true
  }
  include_features: 候補馬ごとの寄与上位の特徴量とコース特性をプロンプトに含める
                    （localでは既定true、geminiでは既定false＝既存シャドウ検証の入力を変えない）
"""
import os
import json
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, 'config', 'gemini_config.json')
EVAL_LOG_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'ai_eval_log.csv')

DEFAULT_MODEL = 'gemini-3.6-flash'
DEFAULT_LOCAL_MODEL = 'gemma4:e4b'
DEFAULT_LOCAL_URL = 'http://localhost:11434/v1/chat/completions'
LOCAL_TIMEOUT = 180  # CPU推論は1レース数十秒かかる
COURSE_PATH = os.path.join(BASE_DIR, 'data', 'course_characteristics.json')
VENUE_SHORT_TO_FULL = {
    '東': '東京', '中': '中山', '中京': '中京', '京': '京都', '阪': '阪神',
    '新': '新潟', '福': '福島', '函': '函館', '札': '札幌', '小': '小倉',
}
MAX_CANDIDATES = 5  # 統計モデル上位何頭までを評価対象にするか（コスト管理）
API_TIMEOUT = 20

_config_cache = None


def load_config() -> dict:
    """config/gemini_config.json を読む。未設定・不正なら enabled=False を返す。"""
    global _config_cache
    if _config_cache is not None:
        return _config_cache
    if not os.path.exists(CONFIG_PATH):
        _config_cache = {'enabled': False}
        return _config_cache
    try:
        with open(CONFIG_PATH, encoding='utf-8') as f:
            cfg = json.load(f)
    except Exception:
        _config_cache = {'enabled': False}
        return _config_cache
    backend = str(cfg.get('backend', 'gemini')).strip().lower()
    cfg['backend'] = backend
    if backend == 'local':
        cfg.setdefault('enabled', True)
        cfg.setdefault('model', DEFAULT_LOCAL_MODEL)
        cfg.setdefault('local_url', DEFAULT_LOCAL_URL)
        cfg.setdefault('include_features', True)
    else:
        api_key = str(cfg.get('api_key', '')).strip()
        if not api_key or 'YOUR_API_KEY' in api_key:
            cfg['enabled'] = False
        else:
            cfg.setdefault('enabled', True)
        cfg.setdefault('model', DEFAULT_MODEL)
        cfg.setdefault('include_features', False)
    _config_cache = cfg
    return _config_cache


def is_enabled() -> bool:
    return bool(load_config().get('enabled'))


def wants_features() -> bool:
    """候補馬に特徴量(features)を付けるべきか。make_newspaper側で参照する。"""
    return bool(load_config().get('include_features'))


def backend_label() -> str:
    cfg = load_config()
    return f"{cfg.get('backend', 'gemini')}:{cfg.get('model', '')}"


_course_cache = None


def _course_note(race_ctx: dict) -> str:
    """data/course_characteristics.json から該当コースの特徴を引く。なければ空文字。"""
    global _course_cache
    if _course_cache is None:
        try:
            with open(COURSE_PATH, encoding='utf-8') as f:
                _course_cache = json.load(f)
        except Exception:
            _course_cache = {}
    venue = str(race_ctx.get('venue', '')).strip()
    venue = VENUE_SHORT_TO_FULL.get(venue, venue)
    dist = ''.join(ch for ch in str(race_ctx.get('distance_m', '')) if ch.isdigit())
    info = _course_cache.get(f"{venue}_{race_ctx.get('surface', '')}_{dist}")
    if not isinstance(info, dict):
        return ''
    return ' / '.join(f'{k}: {v}' for k, v in info.items() if k in ('特徴', '脚質', '枠'))


def _build_prompt(race_ctx: dict, candidates: list, include_features: bool = False) -> str:
    lines = []
    lines.append(
        f"あなたは日本の競馬（JRA）の分析アシスタントです。"
        f"以下のレース・出走馬データだけを根拠に評価してください。データにない情報は推測しないこと。"
    )
    lines.append(
        f"レース: {race_ctx.get('venue', '?')} {race_ctx.get('r_num', '?')}R "
        f"{race_ctx.get('race_name', '')} "
        f"[{race_ctx.get('surface', '?')}{race_ctx.get('distance_m', '?')} / {race_ctx.get('seg_label', '?')}]"
    )
    if include_features:
        note = _course_note(race_ctx)
        if note:
            lines.append(f"コース特性: {note}")
    lines.append("統計モデル（的中率最適化）上位候補:")
    for c in candidates:
        lines.append(
            f"- {c['horse']} (統計順位{c['stat_rank']}位, 統計予測確率{c.get('prob_pct', '?')}, "
            f"単勝オッズ{c.get('odds', '?')}, 騎手{c.get('jockey', '?')})"
        )
        if include_features and c.get('features'):
            lines.append(f"    根拠となる特徴量（値, モデル寄与 +は加点/-は減点）: {c['features']}")
    if include_features:
        lines.append(
            "理由には統計順位や確率をそのまま繰り返さず、特徴量やコース特性から読み取れる"
            "強み・弱みを書くこと。"
        )
    lines.append(
        "各候補馬について0-100の評価スコア(高いほど推奨)と一言理由をJSONで返してください。"
        '出力形式: {"evaluations": [{"horse": "馬名", "score": 数値, "reason": "20字程度の理由"}], '
        '"note": "レース全体の一言コメント(30字程度)"} '
        "JSON以外の文字列は出力しないこと。"
    )
    return '\n'.join(lines)


def _call_gemini(prompt: str, model: str, api_key: str, retries: int = 3, backoff_sec: float = 3.0) -> str:
    """レース数が多い日は無料枠のレート制限(429)・一時的な過負荷(503)に当たりやすいため、
    Discord通知(notify.py)と同様の指数バックオフでリトライする。"""
    import requests
    url = f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent'
    headers = {'Content-Type': 'application/json', 'X-goog-api-key': api_key}
    payload = {
        'contents': [{'parts': [{'text': prompt}]}],
        'generationConfig': {
            'temperature': 0.2,
            'responseMimeType': 'application/json',
        },
    }
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=API_TIMEOUT)
            if r.status_code in (429, 503):
                last_err = f'{r.status_code} {r.text[:200]}'
            else:
                r.raise_for_status()
                data = r.json()
                return data['candidates'][0]['content']['parts'][0]['text']
        except Exception as e:
            last_err = str(e)
        if attempt < retries:
            time.sleep(backoff_sec * attempt)
    raise RuntimeError(f'Gemini API呼び出し失敗（{retries}回リトライ後）: {last_err}')


def _call_local(prompt: str, model: str, url: str, retries: int = 2) -> str:
    """Ollama / llama.cpp server などOpenAI互換の /v1/chat/completions を呼ぶ。"""
    import requests
    payload = {
        'model': model,
        'messages': [{'role': 'user', 'content': prompt}],
        'temperature': 0.2,
        'max_tokens': 800,
        'response_format': {'type': 'json_object'},
    }
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.post(url, json=payload, timeout=LOCAL_TIMEOUT)
            r.raise_for_status()
            return r.json()['choices'][0]['message']['content']
        except Exception as e:
            last_err = str(e)
    raise RuntimeError(f'ローカルLLM呼び出し失敗（{retries}回）: {last_err}')


def _parse_json(text: str) -> dict:
    """```json ... ``` で囲まれて返ってきても読めるようにする。"""
    t = text.strip()
    if t.startswith('```'):
        t = t.split('\n', 1)[1] if '\n' in t else t
        t = t.rsplit('```', 1)[0]
    return json.loads(t)


def evaluate_race(race_ctx: dict, candidates: list) -> dict:
    """race_ctx: {venue, r_num, race_name, surface, distance_m, seg_label}
    candidates: [{horse, stat_rank, prob_pct, odds, jockey}, ...] (MAX_CANDIDATES件まで自動で絞る)

    戻り値: {horse_name: {'score': float, 'reason': str}, ...} + '_note': str
    取得失敗・未設定時は空dict {} を返す（例外は投げない）。
    """
    cfg = load_config()
    if not cfg.get('enabled'):
        return {}
    if not candidates:
        return {}

    candidates = candidates[:MAX_CANDIDATES]
    prompt = _build_prompt(race_ctx, candidates, include_features=bool(cfg.get('include_features')))

    try:
        if cfg.get('backend') == 'local':
            text = _call_local(prompt, cfg['model'], cfg['local_url'])
        else:
            text = _call_gemini(prompt, cfg.get('model', DEFAULT_MODEL), cfg['api_key'])
        parsed = _parse_json(text)
    except Exception as e:
        print(f'  [WARN] gemini_eval: API呼び出し/解析失敗: {e}')
        return {}

    out = {}
    for ev in parsed.get('evaluations', []):
        name = str(ev.get('horse', '')).strip()
        if not name:
            continue
        try:
            score = float(ev.get('score'))
        except (TypeError, ValueError):
            continue
        out[name] = {'score': score, 'reason': str(ev.get('reason', '')).strip()}
    if out:
        out['_note'] = str(parsed.get('note', '')).strip()
    return out


def log_evaluations(date_num: int, race_ctx: dict, candidates: list, evaluations: dict) -> None:
    """予測時点のGemini評価をai_eval_log.csvに記録する（シャドウ検証の材料）。
    同一 日付+開催+R+馬名 のキーで上書き（同日複数回の新聞再生成に対応）。
    実際の着順はここでは記録しない（結果確定後に別途 reconcile する想定、v3 Phase 3範囲外）。
    """
    if not evaluations:
        return
    import pandas as pd

    rows = []
    for c in candidates:
        name = c['horse']
        ev = evaluations.get(name)
        if ev is None:
            continue
        rows.append({
            '日付': date_num,
            '開催': race_ctx.get('venue', ''),
            'R': race_ctx.get('r_num', ''),
            '馬名S': name,
            '統計順位': c.get('stat_rank'),
            '統計予測確率': c.get('prob_pct'),
            'オッズ': c.get('odds'),
            'AI評価スコア': ev['score'],
            'AI理由': ev.get('reason', ''),
            'AIバックエンド': backend_label(),
            '記録日時': time.strftime('%Y-%m-%d %H:%M:%S'),
        })
    if not rows:
        return
    new_df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(EVAL_LOG_PATH), exist_ok=True)
    if os.path.exists(EVAL_LOG_PATH):
        try:
            existing = pd.read_csv(EVAL_LOG_PATH, encoding='utf-8-sig')
            key_cols = ['日付', '開催', 'R', '馬名S']
            new_keys = set(map(tuple, new_df[key_cols].values.tolist()))
            keep = existing[~existing[key_cols].apply(tuple, axis=1).isin(new_keys)]
            new_df = pd.concat([keep, new_df], ignore_index=True)
        except Exception as e:
            print(f'  [WARN] gemini_eval: ai_eval_log.csv 読込失敗、新規作成します: {e}')
    new_df.to_csv(EVAL_LOG_PATH, index=False, encoding='utf-8-sig')
