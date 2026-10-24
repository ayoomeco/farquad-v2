"""Config loader. Model names, engines and prices come from config/genie.toml (no rebuild to upgrade).
Secrets are NEVER read from this file; only from environment variables."""
from __future__ import annotations
import os, json, pathlib, tomllib, functools

PROJECT = pathlib.Path(__file__).resolve().parent.parent
CONFIG_FILE = pathlib.Path(os.environ.get('JUTSU_CONFIG', PROJECT / 'config' / 'genie.toml'))
CHAR_DIR = pathlib.Path(os.environ.get('JUTSU_CHARACTERS', PROJECT / 'config' / 'characters'))


@functools.lru_cache(maxsize=1)
def load() -> dict:
    with open(CONFIG_FILE, 'rb') as f:
        cfg = tomllib.load(f)
    llm = cfg.setdefault('llm', {})
    llm['url'] = os.environ.get('OLLAMA_URL', llm.get('url', 'http://127.0.0.1:11434')).rstrip('/')
    llm['model'] = os.environ.get('OLLAMA_MODEL', llm.get('model', 'qwen3.5:4b'))
    llm.setdefault('fallback_models', ['llama3.2:3b'])
    q = cfg.setdefault('queue', {})
    if os.environ.get('JUTSU_WORKER') is not None:
        q['worker_enabled'] = os.environ['JUTSU_WORKER'] not in ('0', 'false', 'no')
    return cfg


def reload() -> dict:
    load.cache_clear()
    return load()


def app_name() -> str:
    return load().get('app', {}).get('name', 'Genie Jutsu OS')


def data_root() -> pathlib.Path:
    root = pathlib.Path(os.environ.get('JUTSU_DATA', PROJECT / 'data'))
    root.mkdir(parents=True, exist_ok=True)
    return root


def characters() -> dict[str, dict]:
    out = {}
    if CHAR_DIR.exists():
        for p in sorted(CHAR_DIR.glob('*.json')):
            try:
                d = json.loads(p.read_text())
                out[d['id']] = d
            except Exception:
                continue
    return out


def supabase_settings() -> dict | None:
    """Optional Supabase mirror. Only active when BOTH env vars exist. Values never logged."""
    url = os.environ.get('JUTSU_SUPABASE_URL')
    key = os.environ.get('JUTSU_SUPABASE_KEY')
    if not url or not key:
        return None
    return {'url': url.rstrip('/'), 'key': key, 'table': load().get('store', {}).get('supabase_table', 'genie_jobs')}
