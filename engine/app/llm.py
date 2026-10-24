"""Local LLM client (Ollama). The LLM only proposes text/JSON; it never executes anything."""
from __future__ import annotations
import json, time
import httpx
from . import config

_resolved: dict = {'model': None, 'at': 0.0}


def _installed(names: list[str], model: str) -> bool:
    return any(n == model or n == f'{model}:latest' or (':' not in model and n.split(':')[0] == model) for n in names)


def resolve_model(names: list[str] | None = None) -> str:
    """Configured director model (default qwen3.5:4b) if Ollama has it, else the first installed fallback
    (llama3.2:3b). Re-checked at most every 60 s; if Ollama can't be asked, the configured model is used."""
    c = config.load()['llm']
    want, fallbacks = c['model'], list(c.get('fallback_models') or [])
    if names is None:
        if _resolved['model'] and time.time() - _resolved['at'] < 60:
            return _resolved['model']
        try:
            r = httpx.get(c['url'] + '/api/tags', timeout=2)
            names = [m.get('name', '') for m in r.json().get('models', [])]
        except Exception:
            return want
    pick = next((m for m in [want] + fallbacks if _installed(names, m)), want)
    _resolved.update(model=pick, at=time.time())
    return pick


def _payload(messages, json_mode=False, max_tokens=None, stream=False):
    c = config.load()['llm']
    payload = {'model': resolve_model(), 'messages': messages, 'stream': stream,
               'options': {'temperature': c.get('temperature', 0.45), 'num_ctx': int(c.get('num_ctx', 2048))},
               'keep_alive': c.get('keep_alive', '30m')}
    if payload['model'].startswith(('qwen3', 'qwq', 'deepseek-r1')) and not c.get('think', False):
        payload['think'] = False                         # short Director replies: skip hidden reasoning tokens
    if max_tokens:
        payload['options']['num_predict'] = max_tokens
    if json_mode:
        payload['format'] = 'json'
    return payload


async def stream_chat(messages: list[dict], max_tokens: int | None = None):
    """Yield content chunks as Ollama produces them."""
    c = config.load()['llm']
    async with httpx.AsyncClient(timeout=c.get('timeout_s', 180)) as client:
        async with client.stream('POST', c['url'] + '/api/chat', json=_payload(messages, max_tokens=max_tokens, stream=True)) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line.strip():
                    continue
                d = json.loads(line)
                t = (d.get('message') or {}).get('content', '')
                if t:
                    yield t
                if d.get('done'):
                    break


async def warm(system: str | None = None) -> bool:
    """Load the model and cache the system-prompt prefix so the first real chat is fast."""
    try:
        msgs = ([{'role': 'system', 'content': system}] if system else []) + [{'role': 'user', 'content': 'hi'}]
        await chat(msgs, max_tokens=1)
        return True
    except Exception:
        return False


async def chat(messages: list[dict], json_mode: bool = False, max_tokens: int | None = None) -> str:
    c = config.load()['llm']
    payload = _payload(messages, json_mode, max_tokens)
    async with httpx.AsyncClient(timeout=c.get('timeout_s', 180)) as client:
        r = await client.post(c['url'] + '/api/chat', json=payload)
        r.raise_for_status()
        return r.json()['message']['content']


async def status() -> dict:
    c = config.load()['llm']
    out = {'url': c['url'], 'model': c['model'], 'fallback_models': list(c.get('fallback_models') or []),
           'reachable': False, 'model_installed': False, 'active_model': None}
    try:
        async with httpx.AsyncClient(timeout=2) as client:
            r = await client.get(c['url'] + '/api/tags')
            if r.status_code == 200:
                out['reachable'] = True
                names = [m.get('name', '') for m in r.json().get('models', [])]
                out['model_installed'] = _installed(names, c['model'])
                out['active_model'] = resolve_model(names)
    except Exception:
        pass
    return out
