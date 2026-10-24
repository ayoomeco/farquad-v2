"""Deterministic Genjutsu router (+ optional LLM tiebreak).

Rules are plain, inspectable arithmetic over registry metadata in config/genie.toml ([engines.genjutsu_*]):
  genjutsu_modes, subjects, mode_scores, identity, motion, speed, max_native_seconds, segments, multi_subject,
  native_audio, restyle_strengths. The LLM is consulted ONLY when the top two scores are within `tie_margin`,
  and its answer is accepted only if it names one of those tied candidates. Otherwise the deterministic order wins."""
from __future__ import annotations
import re
from typing import Callable
from .. import config
from .shotspec import ShotSpec

TEXTURED_STYLE = re.compile(r'(oil|paint|photo|real|clay|stop.?motion|watercolou?r|pastel|charcoal|impasto|3d|pixar|unreal|cinematic|film)', re.I)
FLAT_STYLE = re.compile(r'(anime|cel|toon|cartoon|2d|line.?art|comic|manga|ink|flat|vector|pixel|ms paint|disney)', re.I)


def genjutsu_engines() -> dict[str, dict]:
    return {k: v for k, v in (config.load().get('engines') or {}).items()
            if v.get('enabled') and v.get('genjutsu_modes')}


def _settings() -> dict:
    return {'tie_margin': 0.05, **(config.load().get('genjutsu') or {}).get('router', {})}


def score(spec: ShotSpec, name: str, e: dict) -> tuple[float | None, list[str]]:
    """Return (score or None if ineligible, reasons)."""
    why = []
    if spec.mode not in e.get('genjutsu_modes', []):
        return None, [f'does not do {spec.mode}']
    subj = e.get('subjects', ['any'])
    if spec.subject != 'any' and 'any' not in subj and spec.subject not in subj:
        return None, [f'cannot drive subject type {spec.subject!r} (supports {subj})']
    if e.get('requires_character_lora'):
        pack = config.characters().get(spec.character_pack or '') or {}
        if not (pack.get('lora') or {}).get('file'):
            return None, ['needs a character pack with a trained Wan 14B LoRA (none given)']
    if spec.mode == 'restyle' and e.get('requires_style_image') and not spec.refs_by('style'):
        return None, ['needs a style reference image (none given)']
    s = float((e.get('mode_scores') or {}).get(spec.mode, 0.5))
    why.append(f'base {spec.mode} score {s:.2f}')
    w_id, w_mo = {'likeness': (0.25, 0.05), 'motion': (0.05, 0.25), 'balanced': (0.12, 0.12)}[spec.priority]
    s += w_id * float(e.get('identity', 0.5)) + w_mo * float(e.get('motion', 0.5))
    why.append(f'priority={spec.priority}: identity {e.get("identity", 0.5)} motion {e.get("motion", 0.5)}')
    if spec.tier == 'draft':
        s += 0.10 * float(e.get('speed', 0.5)); why.append(f'draft tier favours speed {e.get("speed", 0.5)}')
    if spec.subject_count > 1:
        if e.get('multi_subject'):
            s += 0.10; why.append('handles multiple subjects')
        else:
            s -= 0.20; why.append('single-subject engine, multi-subject shot')
    secs = spec.source_seconds or 0
    nat = float(e.get('max_native_seconds', 15))
    if secs > nat:
        if e.get('segments'):
            s -= 0.03; why.append(f'{secs:.0f}s > native {nat:.0f}s: chained segments ({e["segments"]})')
        else:
            s -= 0.25; why.append(f'{secs:.0f}s > native {nat:.0f}s and no segment chaining')
    if spec.mode == 'character_swap' and spec.keep_background and e.get('in_place_replacement'):
        s += 0.10; why.append('true in-place replacement (keeps source plate/lighting)')
    if spec.mode == 'restyle':
        st = spec.style or ''
        strengths = e.get('restyle_strengths', [])
        if st and TEXTURED_STYLE.search(st) and 'textured' in strengths:
            s += 0.10; why.append('textured/realistic style: engine handles texture')
        if st and TEXTURED_STYLE.search(st) and 'flat_only' in strengths:
            s -= 0.15; why.append('engine is weak on textured styles (ReStyle v0.1 limitation)')
        if st and FLAT_STYLE.search(st) and 'flat' in strengths:
            s += 0.08; why.append('flat/cel style: engine strong here')
        if spec.refs_by('style') and e.get('style_image_ref'):
            s += 0.05; why.append('uses the style reference image directly')
    if spec.character_pack and e.get('wan_lora_compatible'):
        pack = config.characters().get(spec.character_pack) or {}
        if pack.get('lora'):
            s += 0.03; why.append('character pack has a Wan 2.1 LoRA (compatibility UNVERIFIED on this engine)')
    if spec.reentry and e.get('weak_reentry'):
        s -= 0.25; why.append('subject re-enters frame: engine is known to drift on re-entry')
    if spec.partial_swap and spec.mode == 'character_swap':
        if e.get('masked_swap'):
            s += 0.10; why.append('supports masked partial swaps')
        else:
            s -= 0.20; why.append('no masked partial swap')
    if spec.mode == 'upscale':
        hq = float(e.get('hero_quality', 0.5))
        s += (0.25 * hq) if spec.hero else (0.15 * float(e.get('speed', 0.5)))
        why.append(f'hero shot: quality {hq}' if spec.hero else f'bulk upscale: speed {e.get("speed", 0.5)}')
    if e.get('hosted') and not spec.engine:
        s -= 0.30; why.append('hosted paid API: fallback only (never auto-picked over own GPUs)')
    if e.get('verified'):
        s += 0.05; why.append('verified by a measured run')
    return round(s, 4), why


def route(spec: ShotSpec, llm: Callable[[str], str] | None = None) -> dict:
    engs = genjutsu_engines()
    if spec.engine:
        e = engs.get(spec.engine)
        if e and spec.mode in e.get('genjutsu_modes', []):
            return {'engine': spec.engine, 'decided_by': 'explicit', 'ranking': [{'engine': spec.engine, 'score': None, 'why': ['explicit override']}],
                    'rejected': {}, 'tiebreak': None, 'fallback': e.get('fallback')}
    ranked, rejected = [], {}
    for name, e in engs.items():
        sc, why = score(spec, name, e)
        if sc is None:
            rejected[name] = why
        else:
            ranked.append({'engine': name, 'score': sc, 'why': why, 'priority': e.get('priority', 100)})
    ranked.sort(key=lambda r: (-r['score'], r['priority'], r['engine']))
    out = {'engine': ranked[0]['engine'] if ranked else None, 'decided_by': 'rules', 'ranking': ranked,
           'rejected': rejected, 'tiebreak': None, 'fallback': None}
    if not ranked:
        out['decided_by'] = 'none'; return out
    _set_fallback(out, engs)
    margin = float(_settings()['tie_margin'])
    tied = [r for r in ranked if ranked[0]['score'] - r['score'] <= margin]
    if len(tied) > 1 and llm:
        names = [r['engine'] for r in tied]
        prompt = ('You pick ONE video engine for this shot. Reply with the engine name only.\n'
                  f'Shot: mode={spec.mode}, subject={spec.subject}, count={spec.subject_count}, priority={spec.priority}, '
                  f'tier={spec.tier}, style={spec.style!r}, prompt={spec.prompt[:300]!r}\n'
                  + '\n'.join(f'- {r["engine"]}: {engs[r["engine"]].get("label", "")}; ' + '; '.join(r['why']) for r in tied))
        try:
            ans = (llm(prompt) or '').strip()
        except Exception as ex:                    # LLM down -> deterministic answer stands
            out['tiebreak'] = {'candidates': names, 'error': type(ex).__name__, 'accepted': False}
            return out
        pick = next((n for n in names if n in ans), None)
        out['tiebreak'] = {'candidates': names, 'answer': ans[:200], 'accepted': bool(pick)}
        if pick:
            out['engine'] = pick; out['decided_by'] = 'llm_tiebreak'; _set_fallback(out, engs)
    elif len(tied) > 1:
        out['tiebreak'] = {'candidates': [r['engine'] for r in tied], 'accepted': False, 'note': 'no LLM: deterministic order'}
    return out


def _set_fallback(out: dict, engs: dict):
    """Declared fallback (e.g. viggle -> scail2) if it is eligible, else the next-best ranked engine."""
    eligible = [r['engine'] for r in out['ranking'] if r['engine'] != out['engine']]
    decl = (engs.get(out['engine']) or {}).get('fallback')
    out['fallback'] = decl if decl in eligible else (eligible[0] if eligible else None)


def ollama_tiebreaker(prompt: str) -> str:
    """Default LLM tiebreak using the local Ollama model (free). Short, deterministic-ish."""
    import asyncio
    from .. import llm as _llm
    return asyncio.run(_llm.chat([{'role': 'user', 'content': prompt}], max_tokens=12))


# ---------------------------------------------------------------- universal per-subject plan (Oct 14 2026)
UNIVERSAL_ORDER = {
    # cheapest/most-proven first per subject type; stubs stay listed but flagged until verified
    'human':    ['genjutsu_h3_swaplora', 'genjutsu_wan22_animate_replace', 'genjutsu_h3_ref2va', 'genjutsu_viggle', 'genjutsu_mocha', 'genjutsu_wan_animate2'],
    'stylized': ['genjutsu_h3_swaplora', 'genjutsu_mocha', 'genjutsu_wan22_animate_replace', 'genjutsu_depth_only', 'genjutsu_wan_animate2'],
    'creature': ['genjutsu_mocha', 'genjutsu_h3_swaplora', 'genjutsu_depth_only', 'genjutsu_wan_animate2'],
    'object':   ['genjutsu_mocha', 'genjutsu_depth_only', 'genjutsu_h3_objswap'],
}


def universal_plan(subject_type: str, *, occlusion: bool = False, tiny_subject: bool = False, has_subject: bool = True) -> dict:
    """Ordered engine plan for one shot given the identity pack's subject_type. Pure config lookup, no GPU.
    Returns {'action', 'engines': [{engine, enabled, stub, label}], 'skeleton_ok'}."""
    if subject_type not in UNIVERSAL_ORDER:
        raise ValueError(f'unknown subject_type {subject_type!r}')
    if not has_subject:
        return {'action': 'SOURCE_COPY', 'engines': [], 'skeleton_ok': False}
    engs = config.load().get('engines') or {}
    order = list(UNIVERSAL_ORDER[subject_type])
    if occlusion and 'genjutsu_mocha' in order:      # mask-only engine is robust to props/occluders
        order.remove('genjutsu_mocha'); order.insert(0, 'genjutsu_mocha')
    rows = [{'engine': n, 'enabled': bool(engs.get(n, {}).get('enabled')), 'stub': bool(engs.get(n, {}).get('stub')),
             'label': engs.get(n, {}).get('label', '')} for n in order if n in engs]
    return {'action': 'FULL_SHOT_REGEN' if tiny_subject else 'SWAP', 'engines': rows,
            'skeleton_ok': subject_type in ('human', 'stylized')}
