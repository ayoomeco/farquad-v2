"""ShotSpec -> full dry-run plan: route (rules + optional LLM tiebreak) -> recipe graph (validated offline) ->
model requirements (manifest) -> cost quote on the chosen provider. Shared by the jobs API, the Genjutsu engine
plugin and the Phase 1 runner. Never spends money and never contacts a GPU provider."""
from __future__ import annotations
import pathlib
from .. import config
from . import recipes, manifest, router
from .shotspec import ShotSpec


def files_for(spec: ShotSpec) -> dict:
    """Names as they will exist in the worker's ComfyUI input/ dir (the provider uploads them)."""
    name = lambda p: pathlib.PurePath(str(p)).name
    out: dict = {'source': name(spec.source_video) if spec.source_video else None}
    for r in spec.refs:
        out.setdefault(r.role if r.role != 'motion' else 'subject', []).append(name(r.path))
    if out.get('source'):
        stem = out['source'].rsplit('.', 1)[0]
        out['source16'] = stem + '_16fps.mp4'
        if spec.mode == 'restyle' and not (out.get('subject') or out.get('style')):
            # Preset restyle with no image (Higgsfield allows image_urls=[]): anchor H3 on the source's own first
            # frame; the style comes from the preset prompt. A pre-restyled keyframe is better (TODO Phase 2).
            out['subject'] = [stem + '_frame0.png']
            out['derived'] = {stem + '_frame0.png': 'first_frame'}
    return out


def h3_defaults(tier: str) -> dict:
    h = (config.load().get('genjutsu') or {}).get('h3') or {}
    return {'accel': h.get(f'{tier}_accel'), 'vae': h.get(f'{tier}_vae'), 'veda': bool(h.get('veda', False)),
            'vsa': bool(h.get('vsa', False)), 'motion_ref_short_edge': h.get(f'motion_ref_short_edge_{tier}'),
            'prompt_format': h.get('prompt_format')}


def recipe_kwargs(recipe: str, tier: str, gpu: str, overrides: dict | None = None) -> dict:
    kw = {}
    if recipe in ('h3_ref2va', 'h3_cn_object_swap'):
        d = h3_defaults(tier)
        kw = {k: v for k, v in d.items() if v is not None}
        if kw.get('veda') and gpu not in ('5090', 'pro6000'):
            kw['veda'] = False
        if kw.get('vsa') and gpu not in ('5090', 'pro6000', 'h100'):
            kw['vsa'] = False
        if recipe != 'h3_ref2va':                       # only Ref2VA has a reference video / Context-IR prompt
            kw.pop('motion_ref_short_edge', None); kw.pop('prompt_format', None)
    kw.update({k: v for k, v in (overrides or {}).items() if v is not None})
    return kw


def character_kwargs(recipe: str, spec: ShotSpec) -> dict:
    """Optional character pack -> recipe extras (only the Wan VACE recipe consumes a LoRA today)."""
    pack = config.characters().get(spec.character_pack or '') or {}
    if recipe != 'wan_vace_lora' or not pack:
        return {}
    lo = pack.get('lora') or {}
    out = {'lora': lo.get('file'), 'lora_strength': float(lo.get('strength', 1.0))}
    if pack.get('negative'):
        out['negative'] = ('low quality, worst quality, deformed, distorted, disfigured, motion smear, blurry face, watermark, text, '
                           'cartoon, anime, ' + pack['negative'])
    return out


def quote(engine_cfg: dict, spec: ShotSpec, provider: str = 'runpod_pod', gpu: str | None = None) -> dict:
    from .. import providers
    est = (engine_cfg.get('est_gpu_seconds') or {}).get(spec.tier, 300)
    secs = spec.source_seconds or 5.0
    gpu_s = float(est) * max(1.0, secs / 5.0)
    if engine_cfg.get('hosted'):
        rate = (engine_cfg.get('price_usd_per_output_second') or {}).get('768p', 0.08)
        ref_s = secs if engine_cfg.get('video_ref_billed_per_input_second') else 0
        return {'provider': 'minimax_api', 'usd': round((secs + ref_s) * rate, 3), 'basis': f'${rate}/s output + video ref seconds',
                'gpu_seconds': 0}
    prov = providers.get(provider)
    names = {g.get('name'): g.get('id') for g in prov.gpus()}
    return prov.quote(gpu_s, names.get(gpu) if gpu in names else None)


def plan_shot(spec: ShotSpec, provider: str | None = None, gpu: str | None = None, llm=None,
              recipe_overrides: dict | None = None, allow_hosted: bool = False) -> dict:
    g = config.load().get('genjutsu') or {}
    provider = provider or g.get('default_provider', 'runpod_pod')
    gpu = gpu or ('omen' if provider == 'local_omen' else g.get('default_gpu', '5090'))
    r = router.route(spec, llm=llm)
    from .consent import check as consent_check
    out = {'shotspec': spec.model_dump(), 'route': r, 'provider': provider, 'gpu': gpu, 'dry_run': True,
           'consent': consent_check(spec)}
    if not r['engine']:
        out['error'] = 'no engine can do this shot'; return out
    ecfg = router.genjutsu_engines()[r['engine']]
    out['engine'] = r['engine']
    if ecfg.get('hosted'):
        out['quote'] = quote(ecfg, spec, provider, gpu)
        out['note'] = 'hosted API: no graph; requires MINIMAX_API_KEY + approval'
        return out
    rec = ecfg['recipe']
    kw = {**character_kwargs(rec, spec), **recipe_kwargs(rec, spec.tier, gpu, recipe_overrides)}
    try:
        built = recipes.build(rec, spec, files_for(spec), tier=spec.tier, gpu=gpu, **kw)
    except recipes.GraphError as ex:                    # e.g. engine can't take these inputs -> declared/next fallback
        fb = r.get('fallback')
        if fb and router.genjutsu_engines().get(fb, {}).get('hosted') and not allow_hosted:
            fb = None                                   # never fall back to a paid hosted API unless asked
        if not fb or spec.engine:
            raise
        out['fallback_reason'] = f'{r["engine"]}: {ex}'
        nxt = plan_shot(spec.model_copy(update={'engine': fb}), provider, gpu, None, recipe_overrides, allow_hosted)
        nxt['fallback_reason'] = out['fallback_reason']; nxt['route']['first_choice'] = r['engine']
        return nxt
    req = manifest.required_for(built.graph)
    out.update({'recipe': rec, 'recipe_kwargs': kw, 'graph': built.graph, 'outputs': built.outputs, 'uploads': built.uploads,
                'derived_inputs': files_for(spec).get('derived', {}), 'params': built.params, 'notes': built.notes, 'problems': built.problems, 'models': req,
                'quote': quote(ecfg, spec, provider, gpu),
                'post': (['audio_remux'] if spec.keep_audio and spec.mode != 'upscale' else []) +
                        (['trim_to_source'] if spec.mode != 'upscale' else [])})
    return out
