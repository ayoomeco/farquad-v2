"""Recipes: official ComfyUI graph (config/workflows/api/*.api.json, converted from the vendor templates)
+ deterministic bindings for a ShotSpec. Every recipe returns a graph that passes graph.validate() offline.

Inputs are FILE NAMES as they will exist in the worker's ComfyUI input/ dir (uploaded by the provider):
    files = {'source': 'clip_src24.mp4', 'source16': 'clip_src16.mp4', 'subject': [...], 'object': [...], 'style': [...]}
GPU profiles (GPU_PROFILES) change speed patches / file choice only (never the requested mode):
    '5090'    -> NVFP4 pruned H3 trunk + comfy-kitchen INT8 attention + KJ chunked FFN; Veda / VSA allowed
    'pro6000' -> same, no chunking (96 GB)
    'omen'    -> the owner's RTX 2060 12GB (Turing): w4a8 H3, PDD 4 steps, <=0.4 MP, no Veda / sol-attn / kitchen attention
H3 options (Phase 1 review, Oct 7 2026): accel = base20 | turbo4 (draft default) | turbo8 | pdd4 | pdd8 (PDD-Acc node pack) |
hyperflow8; vae = standard | light | x2; veda = True (sm120 only); vsa = True (Ref2VA VSA, experimental); motion reference
video resampled to 544 short edge (384 drafts) via ShotSpec.motion_ref_short_edge."""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from .graph import Graph, GraphError, is_link, validate
from .shotspec import ShotSpec

H3_FPS = 24
WAN_FPS = 16


@dataclass
class Recipe:
    name: str
    template: str
    graph: dict
    outputs: list[str]
    uploads: list[str]
    params: dict
    notes: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)


def _aspect_label(spec: ShotSpec) -> str:
    if not spec.source_size:
        return '16:9 (Widescreen)'
    w, h = spec.source_size
    r = w / h
    table = [(1.0, '1:1 (Square)'), (2 / 3, '2:3 (Portrait Photo)'), (1.5, '3:2 (Photo)'), (0.75, '3:4 (Portrait Standard)'),
             (4 / 3, '4:3 (Standard)'), (9 / 16, '9:16 (Portrait Widescreen)'), (16 / 9, '16:9 (Widescreen)'), (21 / 9, '21:9 (Ultrawide)')]
    return min(table, key=lambda t: abs(math.log(r / t[0])))[1]


def _fit(spec: ShotSpec, long_edge: int, mult: int = 32) -> tuple[int, int]:
    w, h = spec.source_size or (16, 9)
    if w >= h:
        W, H = long_edge, long_edge * h / w
    else:
        W, H = long_edge * w / h, long_edge
    return int(round(W / mult) * mult), int(round(H / mult) * mult)


def _consumers(g: Graph, nid: str, idx: int = 0):
    return [(k, i) for k, n in g.nodes.items() for i, v in n.get('inputs', {}).items() if is_link(v) and v == [nid, idx]]


def insert_after(g: Graph, nid: str, class_type: str, inputs: dict, in_name: str = 'model', idx: int = 0, title=None) -> str:
    """Insert a pass-through node after nid:idx and rewire all of its consumers to the new node."""
    cons = _consumers(g, nid, idx)
    new = g.add(class_type, {in_name: [nid, idx], **inputs}, title=title)
    for k, i in cons:
        g.nodes[k]['inputs'][i] = [new, 0]
    return new


# ---------------------------------------------------------------- H3 speed / decode options (research update Oct 7 2026,
# revised after the Phase 1 review: NVFP4 on Blackwell, comfy-kitchen INT8 attention, small motion ref, PDD via its node pack)
H3_INT8_UNET = 'minimax_h3_ref2va_pruned_int8_convrot.safetensors'      # Comfy-Org pruned int8 convrot (template default)
H3_NVFP4_UNET = 'minimax_h3_ref2va_pruned_nvfp4.safetensors'            # lilcheaty/MiniMax-H3-NVFP4 (12.5 GB, Blackwell only)
# PDD Acc (alibaba-pai, Parallel Decoding Distillation): rank-64 trunk LoRA + 32-interval head bank. Loaded ONLY through
# Jalen-Brunson/ComfyUI-MiniMax-H3-PDD-Acc (MiniMaxH3PDDAccApply; file in models/pdd_acc, aptech0081 ComfyUI conversion).
# NOTE (verified Oct 7): ComfyUI core >= 0.35 (PR #15908) can also read Kijai's `_pruned_comfy` PDD files through a plain
# LoraLoaderModelOnly (the Phase 1 log shows the head bank loading). We still standardise on the node pack: it emits the
# trained block-boundary sigmas, supports nfe 4/6 regrouping officially, refuses off-grid sigmas and checks trunk pairing.
PDD_ACC_FILE = 'minimax_h3_ref2va_pdd_acc_8step_comfyui.safetensors'
PDD_FILE_PATTERNS = ('pdd', 'acc-8step', 'acc_8step', 'acc-4step')
TURBO4_LORA = 'minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors'     # lightx2v Ref2V Turbo v0.1 (544p, shifts 12/3)
TURBO8_LORA = 'minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors'  # lightx2v Ref2V Turbo 8-step v1.0 (768p). The approved swap default is plan/graphs_v5b.py (A08: turbo 8-step v1.1 at 0.8, shift 6/3), not this ladder.
HYPERFLOW_PRUNED = 'custom_node_hyperflow_8step_v1.0_comfyui_pruned.safetensors'
HYPERFLOW_FULL = 'custom_node_hyperflow_8step_v1.0_comfyui.safetensors'
H3_FULL_UNET = 'minimax_h3_ref2va_int8_convrot.safetensors'
LIGHT_VAE = 'minimax_h3_lynnreal_light_vae_int8_convrot.safetensors'
X2_VAE = 'MiniMax-H3-X2-Detail-v1.safetensors'
VEDA_PREDICTOR = 'minimax_h3_t2va_veda_8nfe_600step_preview_fp8.safetensors'
VSA_GATE = 'fasth3_vsa_gate.safetensors'                 # barelymining/ComfyUI-MiniMax-H3-FastVideo (Kablex Ref2VA-VSA node)
KITCHEN_ATTN = 'comfy kitchen attention'                 # ModelAttentionBackend option (core node, INT8 attention)
MOTION_REF_EDGE = {'draft': 384, 'final': 544}          # short edge of reference video 1 (Ref2VA prepends its tokens)

# accel id -> sampling recipe. CFG is always 1 in these templates (BasicGuider). shift = (video, audio) via MiniMaxH3SigmaShift.
H3_ACCEL = {
    'base20':     {'steps': 20, 'lora': None, 'sampler': 'res_multistep', 'label': 'base 20-step (template default)'},
    'turbo4':     {'steps': 4, 'lora': TURBO4_LORA, 'sampler': 'euler', 'shift': (12.0, 3.0),
                   'label': 'lightx2v Turbo Ref2V 4-step v0.1 (544p-trained, shift 12/3, euler, CFG 1)'},
    'turbo8':     {'steps': 8, 'lora': TURBO8_LORA, 'sampler': 'euler', 'shift': (12.0, 3.0),
                   'label': 'lightx2v Turbo Ref2V 8-step v1.0 768p (shift 12/3, euler, CFG 1)'},
    'pdd4':       {'steps': 4, 'lora': None, 'pdd': '4', 'sampler': 'euler', 'shift': (12.0, 3.0),
                   'label': 'PDD-Acc nfe 4 via MiniMaxH3PDDAccApply (official regrouping; draft)'},
    'pdd8':       {'steps': 8, 'lora': None, 'pdd': '8', 'sampler': 'euler', 'shift': (12.0, 3.0),
                   'label': 'PDD-Acc nfe 8 via MiniMaxH3PDDAccApply (final candidate)'},
    'hyperflow8': {'steps': 8, 'lora': None, 'sampler': 'euler', 'hyperflow': True, 'label': 'HyperFlow 8-step (ApplyHyperFlow sigmas)'},
}
DEFAULT_ACCEL = {'draft': 'turbo4', 'final': 'base20'}   # final flips to the ladder/bake-off winner via [genjutsu.h3] final_accel
DEFAULT_VAE = {'draft': 'light', 'final': 'standard'}   # 'standard' | 'light' (LynnReal, faster decode) | 'x2' (X2 Detail, 2x out)

# GPU profiles change speed patches / file choice only, never the requested mode.
#   h3_unet   : default H3 trunk (NVFP4 on Blackwell, int8 convrot elsewhere)
#   kj_attn   : KJ MiniMaxLowVRAMAttention(16) (exact, adds overhead; only when VRAM is tight and no faster attention is set)
#   kj_ffn    : KJ MiniMaxChunkFeedForward(8) (exact; only chunks sequences > 4096 tokens)
#   attention : default ModelAttentionBackend ('kitchen' = comfy-kitchen INT8 attention, ~2x sampler on a 5090)
GPU_PROFILES = {
    '5090':    {'vram_gb': 32, 'blackwell': True, 'h3_unet': H3_NVFP4_UNET, 'kj_attn': False, 'kj_ffn': True,
                'attention': 'kitchen', 'veda_ok': True, 'vsa_ok': True, 'max_mp': 1.0},
    'pro6000': {'vram_gb': 96, 'blackwell': True, 'h3_unet': H3_NVFP4_UNET, 'kj_attn': False, 'kj_ffn': False,
                'attention': 'kitchen', 'veda_ok': True, 'vsa_ok': True, 'max_mp': 1.0},
    'h100':    {'vram_gb': 80, 'blackwell': False, 'h3_unet': H3_INT8_UNET, 'kj_attn': False, 'kj_ffn': False,
                'attention': 'kitchen', 'veda_ok': False, 'vsa_ok': True, 'max_mp': 1.0},   # sm90: Veda not verified
    '4090':    {'vram_gb': 24, 'blackwell': False, 'h3_unet': H3_INT8_UNET, 'kj_attn': True, 'kj_ffn': True,
                'attention': 'kitchen', 'veda_ok': False, 'vsa_ok': True, 'max_mp': 1.0},
    # the owner's HP Omen: RTX 2060 12GB Turing (sm75). fp16 + int8 only (no bf16/fp8/nvfp4 compute, no Sage/Veda/kitchen attn).
    # Launch flags live in the local_omen provider; here: w4a8 H3, 4 steps, ~0.4 MP (480p-class).
    'omen':    {'vram_gb': 12, 'blackwell': False, 'h3_unet': 'minimax_h3_ref2va_pruned_w4a8_mixed.safetensors', 'kj_attn': True,
                'kj_ffn': True, 'attention': None, 'veda_ok': False, 'vsa_ok': False, 'max_mp': 0.4, 'force_accel': 'pdd4'},
}


def gpu_profile(gpu: str) -> dict:
    if gpu not in GPU_PROFILES:
        raise GraphError(f'unknown gpu profile {gpu!r} (have {sorted(GPU_PROFILES)})')
    return GPU_PROFILES[gpu]


def collapse_switch(g: Graph, sw: str, take: bool) -> str:
    """Replace a ComfySwitchNode by the branch it would pick, so unused branches (and their model files) drop out."""
    src = g.get(sw, 'on_true' if take else 'on_false')
    for k, i in _consumers(g, sw, 0):
        g.nodes[k]['inputs'][i] = src
    g.remove([sw])
    return src[0]


def _model_tail(g: Graph, guider: str = '126') -> str:
    return g.get(guider, 'model')[0]


def _veda(g: Graph, guider: str, prof: dict, gpu: str) -> str:
    if not prof['veda_ok']:
        raise GraphError(f'Veda sparse attention is only verified on sm120 (5090 / PRO 6000), not {gpu!r}')
    v = insert_after(g, _model_tail(g, guider), 'VedaSparseAttention',
                     {'predictor': VEDA_PREDICTOR, 'generated_sparsity': '90%', 'reference_sparsity': '90%',
                      'full_attention_layers': '', 'full_attention_steps': '0', 'verbose': False},
                     title='Veda sparse attention (~90% tiles skipped; step 0 dense)')
    return v


def is_pdd_file(name) -> bool:
    n = str(name or '').lower()
    return any(k in n for k in PDD_FILE_PATTERNS)


def assert_no_plain_pdd(nodes: dict):
    """PDD Acc files carry a head bank + trained sigmas: they must go through MiniMaxH3PDDAccApply, never a plain LoRA loader."""
    for nid, n in nodes.items():
        if n.get('class_type') in ('LoraLoaderModelOnly', 'LoraLoader') and is_pdd_file(n.get('inputs', {}).get('lora_name')):
            raise GraphError(f'{nid}: PDD file {n["inputs"]["lora_name"]!r} wired to {n["class_type"]}; use MiniMaxH3PDDAccApply')


def _decode_options(g: Graph, vae: str, decode: str = '122', video: str = '130') -> str:
    """vae='light' -> LynnReal light decoder for VAEDecode only (encode of refs stays on the full VAE);
    vae='x2' -> X2 Detail VAE re-decode at 2x after the normal decode."""
    if vae == 'light':
        lv = g.add('VAELoader', {'vae_name': LIGHT_VAE}, title='LynnReal light H3 VAE (decode only)')
        g.set(decode, 'vae', [lv, 0])
        return 'LynnReal light VAE decode (~1.3x faster decode)'
    if vae == 'x2':
        x2 = g.add('MiniMaxH3X2DetailedUpscale', {'images': [decode, 0], 'detail_vae': X2_VAE, 'detail_strength': 1.0},
                   title='H3 X2 Detail VAE (2x)')
        g.set(video, 'images', [x2, 0])
        return 'X2 Detail VAE: output is 2x the generated resolution'
    if vae != 'standard':
        raise GraphError(f'unknown vae option {vae!r}')
    return 'standard H3 video VAE'


def _h3_common(g: Graph, spec: ShotSpec, tier: str, gpu: str, prompt: str, seconds: float, unet: str = '127',
               accel: str | None = None, vae: str | None = None, veda: bool = False, hyperflow_variant: str = 'pruned',
               megapixels: float | None = None, unet_file: str | None = None, attention: str | None = None,
               kj_attn: bool | None = None, kj_ffn: bool | None = None, vsa: bool = False, vsa_sparsity: float = 0.75) -> dict:
    """Model chain (loader -> sampler), in this order: UNET -> [HyperFlow: must directly follow the loader] -> [turbo LoRA] ->
    SigmaShift -> [PDD Apply] -> KJ chunking -> attention backend (comfy-kitchen INT8) | Veda | VSA -> guider."""
    prof = gpu_profile(gpu)
    accel = prof.get('force_accel') or accel or DEFAULT_ACCEL[tier]
    if accel not in H3_ACCEL:
        raise GraphError(f'unknown H3 accel {accel!r} (have {sorted(H3_ACCEL)})')
    vae = vae or DEFAULT_VAE[tier]
    a = H3_ACCEL[accel]
    mp = megapixels or (0.4 if tier == 'draft' else 1.0)    # 0.9216 = 1280x720 (720p) at 16:9
    if vae == 'x2':
        mp = min(mp, 0.52)                               # 544x960-class -> 1088x1920 out of the X2 decoder
    mp = min(mp, prof['max_mp'])
    notes, patches = [], []
    if vsa and not prof.get('vsa_ok'):
        raise GraphError(f'Ref2VA VSA needs comfy-kitchen sol_attn kernels (not available on {gpu!r})')
    if vsa and veda:
        raise GraphError('VSA and Veda both replace attention: pick one')
    # ---- trunk file
    trunk = unet_file or prof.get('h3_unet') or H3_INT8_UNET
    if a.get('hyperflow') and hyperflow_variant != 'full' and trunk != H3_INT8_UNET:
        trunk = H3_INT8_UNET                             # HyperFlow's curve refit is fitted to the exact Comfy-Org int8 file
        notes.append('HyperFlow pruned: trunk forced to the int8 convrot file its curve refit was fitted on')
    if vsa and trunk != H3_INT8_UNET:
        trunk = H3_INT8_UNET                             # VSA gates were transplanted/validated on the int8 convrot base
        notes.append('VSA: trunk forced to pruned int8 convrot (the base the FastH3 gates were validated on)')
    if trunk == H3_NVFP4_UNET and not prof.get('blackwell'):
        raise GraphError(f'NVFP4 trunk needs a Blackwell GPU (5090 / PRO 6000), not {gpu!r}')
    g.set(unet, 'unet_name', trunk)
    # ---- turbo LoRA (template switch) / steps
    use_lora = bool(a['lora'])
    g.set('146', 'value', use_lora)                      # template switch: LoRA branch vs plain base
    if use_lora:
        g.set('145', 'lora_name', a['lora'])
    g.set('143', 'value', a['steps']).set('144', 'value', a['steps'])
    collapse_switch(g, '141', use_lora); collapse_switch(g, '142', use_lora)
    g.set('123', 'sampler_name', a['sampler'])
    g.set('132', 'value', float(seconds))                # duration -> frames snap inside the template's math node
    g.set('115', 'aspect_ratio', _aspect_label(spec))
    g.set('115', 'megapixels', mp)
    g.set('138', 'value', prompt)
    g.set('129', 'noise_seed', int(spec.seed))
    if a.get('hyperflow'):                               # FIRST patch after the loader, else curve refit is disabled
        full = hyperflow_variant == 'full'
        if full:
            g.set(unet, 'unet_name', H3_FULL_UNET); trunk = H3_FULL_UNET
        hf = insert_after(g, unet, 'ApplyHyperFlow',
                          {'hyperflow_file': HYPERFLOW_FULL if full else HYPERFLOW_PRUNED, 'strength': 1.0, 'lora_mode': 'bypass',
                           'variant': 'full' if full else 'pruned', 'download_if_missing': False, 'verbose': True,
                           'experimental_curve_refit': not full}, title='HyperFlow 8-step')
        g.set('125', 'sigmas', [hf, 1])                  # node's trained 9-point grid replaces BasicScheduler
        patches.append(f'ApplyHyperFlow({"full base, on-recipe" if full else "pruned base + curve refit"}) directly after the loader')
        if not full:
            notes.append('HyperFlow on the PRUNED base is off-recipe (no time_embedder); curve refit recovers most, not all')
    if a.get('shift'):
        sv, sa = a['shift']
        sh = insert_after(g, _model_tail(g), 'MiniMaxH3SigmaShift', {'shift_video': sv, 'shift_audio': sa},
                          title=f'sigma shift {sv:g}/{sa:g} (distill training grid)')
        if '124' in g.nodes:
            g.set('124', 'model', [sh, 0])               # the scheduler must see the shifted model sampling
        patches.append(f'MiniMaxH3SigmaShift({sv:g}/{sa:g})')
    if a.get('pdd'):
        pdd = insert_after(g, _model_tail(g), 'MiniMaxH3PDDAccApply',
                           {'pdd_file': PDD_ACC_FILE, 'nfe': a['pdd'], 'lora_strength': 1.0, 'head_strength': 1.0,
                            'on_off_grid': 'error'}, title=f'PDD Acc (nfe {a["pdd"]}): trunk LoRA + head bank + trained sigmas')
        g.set('125', 'sigmas', [pdd, 1])                 # the Apply node's block-boundary sigmas, never BasicScheduler
        patches.append(f'MiniMaxH3PDDAccApply(nfe={a["pdd"]}, sigmas from node)')
    # ---- attention / memory patches
    attn = attention if attention is not None else prof.get('attention')
    if veda or vsa:
        attn = None                                      # both replace dense attention themselves
    use_kj_attn = prof.get('kj_attn') if kj_attn is None else kj_attn
    if attn == 'kitchen' and kj_attn is None:
        use_kj_attn = False                              # INT8 attention replaces the chunked SDPA call
    use_kj_ffn = prof.get('kj_ffn') if kj_ffn is None else kj_ffn
    if use_kj_attn and not veda and not vsa:
        insert_after(g, _model_tail(g), 'MiniMaxLowVRAMAttention', {'head_chunks': 16}, title='KJ low-VRAM attention (exact)')
        patches.append('MiniMaxLowVRAMAttention(head_chunks=16)')
    if use_kj_ffn:
        insert_after(g, _model_tail(g), 'MiniMaxChunkFeedForward', {'chunks': 8, 'seq_threshold': 4096}, title='KJ chunked FFN (exact)')
        patches.append('MiniMaxChunkFeedForward(chunks=8)')
    if attn == 'kitchen':
        insert_after(g, _model_tail(g), 'ModelAttentionBackend', {'attention': KITCHEN_ATTN},
                     title='comfy-kitchen INT8 attention (falls back to PyTorch if unavailable)')
        patches.append('ModelAttentionBackend(comfy kitchen attention)')
    elif attn not in (None, 'pytorch'):
        raise GraphError(f'unknown attention backend {attn!r}')
    if vsa:
        insert_after(g, _model_tail(g), 'Ref2VAVSAGatePatch', {'gate_file': VSA_GATE, 'sparsity': float(vsa_sparsity)},
                     title=f'Ref2VA VSA gate patch (video sparsity {vsa_sparsity})')
        patches.append(f'Ref2VAVSAGatePatch(sparsity={vsa_sparsity}) [EXPERIMENTAL: QC identity]')
    if veda:
        _veda(g, '126', prof, gpu)
        patches.append('VedaSparseAttention(90%, step 0 dense) [T2VA predictor on R2VA: QC before trusting]')
    dec = _decode_options(g, vae)
    return {'accel': accel, 'accel_label': a['label'], 'steps': a['steps'], 'sampler': a['sampler'], 'lora': a['lora'],
            'pdd_file': PDD_ACC_FILE if a.get('pdd') else None, 'unet': trunk,
            'attention': 'veda' if veda else 'vsa' if vsa else (attn or 'pytorch'),
            'megapixels': mp, 'seconds': seconds, 'aspect': _aspect_label(spec), 'speed_patches': patches,
            'vae': vae, 'decode': dec, 'veda': bool(veda), 'vsa': bool(vsa), 'gpu': gpu, '_notes': notes}


def motion_ref_size(spec: ShotSpec, tier: str, short_edge: int | None = None) -> tuple[int, int] | None:
    """Reference-video conditioning size: resample to `short_edge` (ShotSpec.motion_ref_short_edge, else 544 final / 384
    draft), aspect kept, multiples of 32, never upscaled. None = feed the clip as-is (short_edge 0).
    Priority: the shot's own field > the recipe kwarg ([genjutsu.h3] motion_ref_short_edge_<tier>) > tier default."""
    e = spec.motion_ref_short_edge if spec.motion_ref_short_edge is not None else short_edge
    if e is None:
        e = MOTION_REF_EDGE[tier]
    if not e:
        return None
    w, h = spec.source_size or (1920, 1080)
    s = min(1.0, e / min(w, h))
    fit = lambda v: max(32, min(int(round(v * s / 32)) * 32, int(v // 32) * 32))   # nearest x32, capped: never upscale
    return fit(w), fit(h)


def _h3_seconds(spec: ShotSpec) -> float:
    s = spec.source_seconds or 5.0
    return float(min(15.0, max(5.0, round(s, 2))))


def _add_ref_images(g: Graph, h3_node: str, files: list[str], first_loader: str | None, drop: list[str]):
    files = files[:9]
    for nid in drop:
        g.remove([nid])
    for k in [k for k in list(g.node(h3_node)['inputs']) if k.startswith('ref_images.')]:
        del g.node(h3_node)['inputs'][k]
    for i, f in enumerate(files):
        if i == 0 and first_loader:
            g.set(first_loader, 'image', f); nid = first_loader
        else:
            nid = g.add('LoadImage', {'image': f}, title=f'ref image {i + 1}')
        g.set(h3_node, f'ref_images.ref_image_{i}', [nid, 0])


def h3_ref2va(spec: ShotSpec, files: dict, tier: str = 'draft', gpu: str = '5090', accel: str | None = None,
              vae: str | None = None, veda: bool = False, hyperflow_variant: str = 'pruned', megapixels: float | None = None,
              prompt_override: str | None = None, motion_ref_short_edge: int | None = None, prompt_format: str = 'context_ir',
              description: str | None = None, **speed) -> Recipe:
    """Motion transfer (or full-scene character swap / restyle) with MiniMax H3 Ref2VA: subject image(s) +
    the source clip as REFERENCE VIDEO 1 (motion/timing/camera)."""
    g = Graph.from_template('video_minimax_h3_r2v')
    secs = _h3_seconds(spec)
    imgs = list(files.get('subject') or []) + list(files.get('style') or []) + list(files.get('background') or [])
    if not imgs:
        raise GraphError('H3 Ref2VA needs at least one reference image')
    _add_ref_images(g, '136', imgs, '137', drop=['139'])
    lv = g.add('LoadVideo', {'file': files['source']}, title='source clip (reference video 1)')
    gv = g.add('GetVideoComponents', {'video': [lv, 0]})
    mref = motion_ref_size(spec, tier, motion_ref_short_edge)
    ref_frames = [gv, 0]
    if mref:                                             # Ref2VA prepends the ref video's tokens: a small ref is a big win
        rs = g.add('ImageScale', {'image': [gv, 0], 'upscale_method': 'area', 'width': mref[0], 'height': mref[1], 'crop': 'disabled'},
                   title=f'motion ref -> {mref[0]}x{mref[1]} (24 fps upload)')
        ref_frames = [rs, 0]
    g.set('136', 'ref_videos.ref_video_0', ref_frames)
    if spec.keep_audio:
        g.set('136', 'ref_video_audios.ref_video_audio_0', [gv, 1])
    if prompt_format == 'context_ir':                    # H3 card: Context-IR is critical to output quality
        from .h3_prompt import context_ir
        from .. import config
        pack = config.characters().get(spec.character_pack or '') or {}
        ns, nst, nbg = (len(list(files.get(k) or [])) for k in ('subject', 'style', 'background'))
        prompt = context_ir(spec, ns or len(imgs), secs, description=description, identity_lock=pack.get('identity_lock'),
                            n_style=nst if ns else 0, n_background=nbg if ns else 0)
    else:
        who = spec.prompt or f'the {spec.subject if spec.subject != "any" else "subject"} from the reference image'
        if spec.mode == 'restyle':
            base = f'Re-render reference video 1 in {spec.style or "the style of the reference image"} style. Keep the exact motion, timing, framing and camera of reference video 1.'
        else:
            base = (f'{who} performs exactly the motion, timing, framing and camera movement of reference video 1. '
                    'Keep the face, hair, body shape and outfit of reference image 1. Same duration.')
        prompt = base if not spec.prompt or spec.mode == 'restyle' else f'{base} {spec.prompt}'
    if prompt_override:                                  # full structured H3 prompt (subject_definitions / timeline ...)
        prompt = prompt_override
    params = _h3_common(g, spec, tier, gpu, prompt, secs, accel=accel, vae=vae, veda=veda, hyperflow_variant=hyperflow_variant,
                        megapixels=megapixels, **speed)
    params['motion_ref'] = f'{mref[0]}x{mref[1]} @24 fps' if mref else 'source size'
    params['prompt_format'] = 'override' if prompt_override else prompt_format
    g.set('92', 'filename_prefix', f'genjutsu/h3_ref2va_{tier}_{params["accel"]}')
    g.prune_unreachable()
    return _done('h3_ref2va', 'video_minimax_h3_r2v', g, ['92'], [files['source']] + imgs, params,
                 params.pop('_notes') + ['PDD/HyperFlow/Turbo distills: never stack two distills (character LoRAs are fine)',
                  'Motion ref must be uploaded at 24 fps (provider makes <clip>_src24); the graph only rescales it',
                  'H3 output is 5-15 s; source audio is remuxed afterwards by ffmpeg_local (audio_remux)'])


def h3_cn_object_swap(spec: ShotSpec, files: dict, tier: str = 'draft', gpu: str = '5090', union: str = '2.0',
                      accel: str | None = None, vae: str | None = None, veda: bool = False, **speed) -> Recipe:
    """Object swap: MiniMax H3 Fun ControlNet Union in MASK-INPAINT mode. SAM 3.1 tracks `spec.target`
    through the clip -> mask; the object reference image goes in as ref image 1; everything outside the mask
    stays from the source video."""
    g = Graph.from_template('video_minimax_h3_fun_controlnet_union')
    secs = _h3_seconds(spec)
    g.remove_prefix('700:').remove(['704', '705', '174', '712'])     # drop the pose-control preprocessing path
    del g.node('172')['inputs']['control_video']
    lv = g.add('LoadVideo', {'file': files['source']}, title='source clip')
    gv = g.add('GetVideoComponents', {'video': [lv, 0]})
    sam = g.add('CheckpointLoaderSimple', {'ckpt_name': 'sam3.1_multiplex_fp16.safetensors'}, title='SAM 3.1')
    txt = g.add('CLIPTextEncode', {'clip': [sam, 1], 'text': spec.target}, title='what to replace')
    trk = g.add('SAM3_VideoTrack', {'images': [gv, 0], 'model': [sam, 0], 'conditioning': [txt, 0],
                                    'detection_threshold': 0.5, 'max_objects': max(1, spec.subject_count), 'detect_interval': 1})
    msk = g.add('SAM3_TrackToMask', {'track_data': [trk, 0], 'object_indices': ''})
    grow = g.add('GrowMask', {'mask': [msk, 0], 'expand': 16, 'tapered_corners': True}, title='mask dilation (edges/shadows)')
    g.set('172', 'source_video', [gv, 0]).set('172', 'mask', [grow, 0])
    g.set('173', 'name', 'minimax_h3_fun_controlnet_union_2.0_pruned_int8_convrot.safetensors' if union == '2.0'
          else 'minimax_h3_fun_controlnet_union_pruned_int8_convrot.safetensors')
    objs = list(files.get('object') or [])
    if objs:
        _add_ref_images(g, '136', objs, None, drop=[])
    new = spec.prompt or ('the object in reference image 1' if objs else 'a new object')
    prompt = (f'Replace {spec.target} with {new}. The new object follows the original object\'s position, scale, '
              'motion and occlusion in every frame. Everything outside the masked region stays identical: '
              'people, background, lighting, camera.')
    if (accel or DEFAULT_ACCEL[tier]) == 'hyperflow8':
        raise GraphError('HyperFlow is not validated with Fun ControlNet; use pdd4/pdd8/base20 for object swap')
    speed.pop('motion_ref_short_edge', None)
    params = _h3_common(g, spec, tier, gpu, prompt, secs, accel=accel, vae=vae, veda=veda, **speed)
    params.update({'controlnet_union': union, 'mask': f'SAM3.1 track "{spec.target}" + GrowMask 16px'})
    g.set('707', 'filename_prefix', f'genjutsu/h3_cn_objswap_{tier}')
    g.prune_unreachable()
    return _done('h3_cn_object_swap', 'video_minimax_h3_fun_controlnet_union', g, ['707'], [files['source']] + objs, params,
                 params.pop('_notes') + ['PDD/Turbo Ref2VA distills with Fun ControlNet are UNVERIFIED (trained on plain Ref2VA)',
                  'Union 2.0 file swapped in for the template\'s v1 name: same ModelPatchLoader (UNVERIFIED on GPU; v1 is the fallback)',
                  'Mask-inpaint path (source_video + mask) replaces the template\'s pose-control path'])


def h3_length(seconds: float, cap_frames: int = 362) -> int:
    """H3 frame count: 24 fps snapped up to 17k+5 (template math node), capped (362 = 15 s)."""
    f = max(5, round(seconds * H3_FPS))
    f += (5 - f % 17) % 17
    while f > cap_frames:
        f -= 17
    return f


VIGGLE_SIGMAS = {3: '40', 5: '41', 7: '43'}               # template CustomSigmas nodes (DMD: 3 passes = "4 steps")


def viggle_animate(spec: ShotSpec, files: dict, tier: str = 'draft', gpu: str = '5090', passes: int = 3,
                   veda: bool = False) -> Recipe:
    """Person / character swap with Viggle-Animate (MiniMax-H3 ref2va 33B fine-tune, DMD2 distilled).
    Inputs: the driving clip + ONE still of the new character (ideally the clip's first frame repainted with the
    user's subject, see notes). No text encoder, no pose, no mask: motion, camera, timing, background and lighting
    come from the clip, identity from the still. Weak spot: identity drifts when the subject leaves and re-enters
    frame or poses far from the still -> router falls back to SCAIL-2 for those and for multi-character shots."""
    prof = gpu_profile(gpu)
    g = Graph.from_template('viggle_animate_h3')
    subj = list(files.get('repainted_frame') or []) + list(files.get('subject') or [])
    if not subj:
        raise GraphError('Viggle-Animate needs a still of the new character (subject image or repainted first frame)')
    if passes not in VIGGLE_SIGMAS:
        raise GraphError(f'viggle passes must be one of {sorted(VIGGLE_SIGMAS)}')
    secs = spec.source_seconds or 5.0
    length = h3_length(secs)                             # provider conforms the clip to exactly `length` frames (pads tail)
    mp = min(0.4 if tier == 'draft' else 0.9, prof['max_mp'])   # tested canvas 0.4-0.98 MP
    # template node 26 is JWImageResizeToMegapixels (comfyui-various, not installed) -> core equivalent
    g.nodes['26'] = {'class_type': 'ImageScaleToTotalPixels', '_meta': {'title': f'scale clip to {mp} MP'},
                     'inputs': {'image': ['3', 0], 'upscale_method': 'lanczos', 'megapixels': mp, 'resolution_steps': 32}}
    g.set('3', 'video', files['source']).set('3', 'force_rate', H3_FPS).set('3', 'frame_load_cap', length)
    g.set('3', 'skip_first_frames', 0).set('3', 'select_every_nth', 1)
    g.set('5', 'image', subj[0])
    g.set('1', 'length', length).set('1', 'width', 0).set('1', 'height', 0)
    g.set('16', 'lora_name', 'viggle_animate_dmd_lora_r64.safetensors')
    g.set('12', 'sigmas', [VIGGLE_SIGMAS[passes], 0])
    g.set('10', 'noise_seed', int(spec.seed))
    g.set('11', 'model', ['14', 0])                      # drop the KJ preview override (needs taeh3, preview only)
    g.remove(['25', '29', '30'])
    patches = []
    if veda or gpu == 'omen':                            # Veda must not stack with core sparse attention; Turing: no sol-attn
        g.set('14', 'model', g.get('46', 'model')); g.remove(['46'])
        patches.append('core BlockSparseAttention removed')
    else:
        patches.append('core BlockSparseAttention sol-attn (template default)')
    if prof['kj_ffn']:
        insert_after(g, '13', 'MiniMaxChunkFeedForward', {'chunks': 8, 'seq_threshold': 4096}, title='KJ chunked FFN (exact)')
        patches.append('MiniMaxChunkFeedForward(chunks=8)')
    if veda:
        _veda(g, '11', prof, gpu)
        patches.append('VedaSparseAttention(90%) [UNVERIFIED on Viggle]')
    if spec.keep_audio:
        g.set('21', 'audio', ['3', 2])
    else:
        g.node('21')['inputs'].pop('audio', None)
    g.set('21', 'filename_prefix', f'genjutsu/viggle_{tier}')
    g.prune_unreachable()
    notes = ['Best input: the clip\'s first frame repainted with the new character (Qwen-Image Edit or any editor) as '
             'files["repainted_frame"]; a plain reference still also works but locks pose/framing less tightly',
             'Identity drifts on re-entry / extreme poses: QC (ArcFace) gates it; SCAIL-2 is the fallback']
    if secs * H3_FPS > length:
        notes.append(f'source {secs:.1f}s > {length} frames: needs the chunked-sampler workflow (Phase 2) or scene split')
    return _done('viggle_animate', 'viggle_animate_h3', g, ['21'], [files['source'], subj[0]],
                 {'frames': length, 'fps': H3_FPS, 'megapixels': mp, 'passes': passes, 'cfg': 1.0, 'shift': 3.0,
                  'sampler': 'euler', 'lora': 'viggle_animate_dmd_lora_r64', 'text_encoder': None,
                  'speed_patches': patches, 'veda': bool(veda), 'gpu': gpu}, notes)


def x2_upscale(spec: ShotSpec, files: dict, tier: str = 'final', gpu: str = '5090') -> Recipe:
    """Cheap 2x path (e.g. 544x960 -> 1088x1920) with the H3 X2 Detail VAE, before (or instead of) SeedVR2.
    Works on any clip: re-encodes each frame with the H3 VAE encoder and decodes at 2x. Keeps the clip's audio."""
    g = Graph({})
    lv = g.add('LoadVideo', {'file': files['source']}, title='video to upscale')
    gv = g.add('GetVideoComponents', {'video': [lv, 0]})
    x2 = g.add('MiniMaxH3X2DetailedUpscale', {'images': [gv, 0], 'detail_vae': X2_VAE, 'detail_strength': 1.0},
               title='H3 X2 Detail VAE (2x)')
    cv = g.add('CreateVideo', {'images': [x2, 0], 'audio': [gv, 1], 'fps': [gv, 2]})
    sv = g.add('SaveVideo', {'video': [cv, 0], 'filename_prefix': f'genjutsu/x2_{tier}', 'format': 'auto', 'codec': 'auto'})
    return _done('x2_upscale', '(built in code)', g, [sv], [files['source']], {'scale': 2, 'detail_strength': 1.0},
                 ['Per-frame encode/decode (no temporal model): check for shimmer on fine texture; SeedVR2 for hero shots'])


def flashvsr_upscale(spec: ShotSpec, files: dict, tier: str = 'final', gpu: str = '5090', src_long_edge: int | None = None) -> Recipe:
    """Bulk upscaler (final sweep Oct 7): FlashVSR v1.1 one-step streaming DMD. 10 s 720p -> 1080p ~41 s on a 5090
    vs SeedVR2 ~312 s. SeedVR2 stays for hero shots / faces. Keeps the clip's audio."""
    prof = gpu_profile(gpu)
    scale = 2 if spec.max_resolution / float(src_long_edge or 1344) <= 2.0 else 4
    g = Graph({})
    lv = g.add('LoadVideo', {'file': files['source']}, title='video to upscale')
    gv = g.add('GetVideoComponents', {'video': [lv, 0]})
    small = prof['vram_gb'] <= 32                         # Phase 1: 'tiny' + untiled DiT OOMed on the 5090 (26.5 GB already held)
    up = g.add('FlashVSRNode', {'frames': [gv, 0], 'model': 'FlashVSR-v1.1', 'mode': 'tiny-long' if small else 'tiny',
                                'scale': scale, 'tiled_vae': prof['vram_gb'] < 48, 'tiled_dit': small,
                                'unload_dit': small, 'seed': int(spec.seed)}, title=f'FlashVSR v1.1 x{scale}')
    cv = g.add('CreateVideo', {'images': [up, 0], 'audio': [gv, 1], 'fps': [gv, 2]})
    sv = g.add('SaveVideo', {'video': [cv, 0], 'filename_prefix': f'genjutsu/flashvsr_{tier}', 'format': 'auto', 'codec': 'auto'})
    return _done('flashvsr_upscale', '(built in code)', g, [sv], [files['source']],
                 {'scale': scale, 'target_long_edge': spec.max_resolution, 'mode': g.get(up, 'mode')},
                 ['Output is source x scale; a final ffmpeg scale pass lands it exactly on max_resolution',
                  'Weights must be pre-staged in ComfyUI/models/FlashVSR-v1.1 (start.sh symlink); the node otherwise calls '
                  'snapshot_download at run time (Phase 1: HF 429 rate limit)', 'Run it first on a fresh ComfyUI or after /free'])


def scail2(spec: ShotSpec, files: dict, tier: str = 'draft', gpu: str = '5090') -> Recipe:
    """SCAIL-2: character_swap = replace_mode (keeps the source plate + lighting); motion_transfer = animate mode
    (reference character performs the driving motion in the reference image's world). Any subject incl. animals."""
    g = Graph.from_template('video_wan21_scail2_character_replacement_int8')
    subj = list(files.get('subject') or [])
    if not subj:
        raise GraphError('SCAIL-2 needs a subject reference image')
    src = files.get('source16') or files['source']
    g.set('155', 'file', src).set('30', 'image', subj[0])
    W, H = _fit(spec, 672 if tier == 'draft' else 896)
    frames = int(round((spec.source_seconds or 5.0) * WAN_FPS))
    segs = 1 if frames <= 89 else 2                       # <=0.5 s over one 81-frame segment: trim instead of a 2nd segment
    who = {'human': 'person', 'animal': 'animal', 'creature': 'creature', 'object': 'object'}.get(spec.subject, 'person')
    for seg in ('213', '262') if segs == 2 else ('213',):
        ids = {'213': ('178', '179', '191', '212', '3', '203', '19', '172'), '262': ('236', '237', '242', '261', '258', '247', '227', '255')}[seg]
        w_id, h_id, drv, ref, pos, rep, smp, dist = (f'{seg}:{x}' for x in ids)
        g.set(w_id, 'value', W).set(h_id, 'value', H)
        g.set(drv, 'text', spec.target or who)           # SAM3 prompt: WHO in the source gets replaced/driven
        g.set(ref, 'text', who)                           # SAM3 prompt on the reference image
        g.set(pos, 'text', spec.prompt or f'the {who} from the reference image, natural lighting, consistent identity')
        g.set(rep, 'value', spec.mode == 'character_swap')
        g.set(smp, 'noise_seed', int(spec.seed))
        g.set(dist, 'value', True)                        # lightx2v 6-step distill + DPO LoRA
    if segs == 1:
        g.remove_prefix('262:').remove(['269', '270', '271', '263', '264', '265'])
        outs = ['202']
    else:
        outs = ['271']
    g.set(outs[0], 'filename_prefix', f'genjutsu/scail2_{spec.mode}_{tier}')
    g.unlink_dangling_optional(); g.prune_unreachable()
    params = {'width': W, 'height': H, 'fps': WAN_FPS, 'frames': min(frames, 81 if segs == 1 else 157), 'segments': segs,
              'steps': 6, 'distill_lora': 'lightx2v I2V 480p @0.8', 'dpo_lora': True,
              'replace_mode': spec.mode == 'character_swap'}
    notes = ['SCAIL-2 runs at 16 fps (Wan native); final can be frame-interpolated or left at 16 fps']
    if frames > 157:
        notes.append(f'{frames} frames > 2 segments: needs the character_replacement_scail_2_extend chain (Phase 2)')
    return _done('scail2', 'video_wan21_scail2_character_replacement_int8', g, outs, [src, subj[0]], params, notes)


def wan_animate2(spec: ShotSpec, files: dict, tier: str = 'draft', gpu: str = '5090') -> Recipe:
    g = Graph.from_template('video_wan_animate2_distilled')
    subj = list(files.get('subject') or [])
    if not subj:
        raise GraphError('Wan-Animate-2 needs a subject reference image')
    src = files.get('source16') or files['source']
    g.set('189', 'image', subj[0]).set('240', 'file', src)
    W, H = _fit(spec, 832 if tier == 'final' else 640, mult=16)
    g.set('261:243', 'resize_type.width', W).set('261:243', 'resize_type.height', H)
    g.set('604', 'value', spec.prompt or 'the character performs the driving motion exactly')
    g.set('605', 'value', f'Character appearance description: the {spec.subject if spec.subject != "any" else "character"} from the reference image.')
    g.set('261:19', 'noise_seed', int(spec.seed))
    g.remove(['292']); g.prune_unreachable()
    g.set('246', 'filename_prefix', f'genjutsu/wan_animate2_{tier}')
    return _done('wan_animate2', 'video_wan_animate2_distilled', g, ['246'], [src, subj[0]],
                 {'width': W, 'height': H, 'steps': 10, 'cache': 'WanAnimate2Cache gpu/int8', 'context_windows': True}, [])


LTX23_RENAMES = {
    'LTX2/ltx-2.3-22b-dev_transformer_only_fp8_scaled.safetensors': 'ltx-2.3-22b-dev_transformer_only_fp8_scaled.safetensors',
    'ltx2/LTX23_audio_vae_bf16.safetensors': 'LTX23_audio_vae_bf16.safetensors',
    'ltx2/LTX23_video_vae_bf16.safetensors': 'LTX23_video_vae_bf16.safetensors',
    'ltx2/taeltx2_3.safetensors': 'taeltx2_3.safetensors',
    'LTX2/ltx-2.3_text_projection_bf16.safetensors': 'ltx-2.3_text_projection_bf16.safetensors',
    'LTX2/ltx-2.3-22b-distilled-1.1_lora-dynamic_fro09_avg_rank_111_bf16.safetensors': 'ltx-2.3-22b-distilled-1.1_lora-dynamic_fro09_avg_rank_111_bf16.safetensors',
    'LTX2/2.3/852654_LTX2.3_22B_Style-Transfer_IC-Lora/lora_weights_step_08000.safetensors': '852654_LTX2.3-22B_ReStyle_IC-LoRA_8000_v0.1.safetensors',
}


def ltx_restyle(spec: ShotSpec, files: dict, tier: str = 'draft', gpu: str = '5090', sage: bool = False) -> Recipe:
    """Restyle with Cseti's LTX-2.3 ReStyle IC-LoRA: style reference image becomes frame 0 of the IC guide,
    the source clip carries content + motion. 8 steps (distill LoRA 0.4). Source audio remuxed afterwards."""
    g = Graph.from_template('ltx2.3-restyle-ic-lora.cseti')
    styles = list(files.get('style') or [])
    if not styles:
        raise GraphError('LTX ReStyle needs a style reference image (generate one from the style name first)')
    g.rename_models(LTX23_RENAMES)
    attn_note = 'SageAttention (PathchSageAttentionKJ + LTX2 memory-efficient Sage patch)'
    if not sage:                                          # Phase 1: no sageattention wheel on the pod -> both patches crash
        src = g.get('4995', 'model')
        kit = g.add('ModelAttentionBackend', {'model': src, 'attention': KITCHEN_ATTN}, title='comfy-kitchen INT8 attention (replaces Sage)')
        g.set('4996', 'model', [kit, 0])
        g.set('4998', 'model', g.get('4997', 'model'))
        g.remove(['4995', '4997'])
        attn_note = 'comfy-kitchen INT8 attention (Sage patches removed; PyTorch fallback if unavailable)'
    g.set('5327', 'video', files['source']).set('5347', 'image', styles[0])
    for nid in ('5331', '5332'):
        g.set(nid, 'upscale_method', 'lanczos')           # template uses NVIDIA RTX VSR (extra SDK); lanczos = no extra dep
    W, H = _fit(spec, 768 if tier == 'draft' else 1024)
    fps = 24
    frames = int(round((spec.source_seconds or 5.0) * fps))
    frames = max(9, min(frames, 241)) // 8 * 8 + 1        # LTX wants 8k+1 frames
    g.set('5093', 'value', W).set('5094', 'value', H).set('4979', 'value', frames).set('4978', 'value', float(fps))
    style = spec.style or 'the reference image'
    g.set('5270', 'text1', f'Make it {style} style. ' + (spec.prompt or ''))
    g.set('5002', 'seed', int(spec.seed))
    g.remove(['5344', '5342', '5343', '5361']); g.prune_unreachable()
    g.set('5217', 'filename_prefix', f'genjutsu/ltx_restyle_{tier}')
    return _done('ltx_restyle', 'ltx2.3-restyle-ic-lora.cseti', g, ['5217'], [files['source'], styles[0]],
                 {'width': W, 'height': H, 'frames': frames, 'fps': fps, 'steps': 8, 'cfg': 1.0,
                  'iclora': 'Cseti ReStyle v0.1 (step 8000)', 'distill_lora': 0.4, 'attention': attn_note},
                 ['ReStyle v0.1 is strong on flat/cel/line-art styles, weak on textured ones (router knows this)',
                  'README tip: CFG 1.1-2.0 or the non-distilled model improves hard styles (Phase 2 A/B)'])


def seedvr2_upscale(spec: ShotSpec, files: dict, tier: str = 'final', gpu: str = '5090', src_long_edge: int | None = None) -> Recipe:
    g = Graph.from_template('upscale_video_seedvr2_3b_int8')
    lv = g.add('LoadVideo', {'file': files['source']}, title='video to upscale')
    g.set('125:117', 'video', [lv, 0]).set('125:118', 'on_false', [lv, 0])
    mult = round(max(1.0, spec.max_resolution / float(src_long_edge or 1344)), 3)
    g.set('125:111', 'resize_type.multiplier', mult)
    g.set('125:123', 'value', True)                       # temporal chunking (VRAM-safe)
    sv = g.add('SaveVideo', {'video': ['125:116', 0], 'filename_prefix': f'genjutsu/seedvr2_{tier}', 'format': 'auto', 'codec': 'auto'})
    return _done('seedvr2_upscale', 'upscale_video_seedvr2_3b_int8', g, [sv], [files['source']],
                 {'multiplier': mult, 'target_long_edge': spec.max_resolution, 'temporal_chunking': True}, [])


RECIPES = {'h3_ref2va': h3_ref2va, 'h3_cn_object_swap': h3_cn_object_swap, 'viggle_animate': viggle_animate,
           'scail2': scail2, 'wan_animate2': wan_animate2, 'ltx_restyle': ltx_restyle, 'seedvr2_upscale': seedvr2_upscale,
           'x2_upscale': x2_upscale, 'flashvsr_upscale': flashvsr_upscale}


# ---------------------------------------------------------------- Wan 2.1 VACE 14B + character LoRA (Character pack head-to-head)
WAN_VACE_UNET = 'wan2.1_vace_14B_fp16.safetensors'        # cast to fp8_e4m3fn at load (UNETLoader weight_dtype)
WAN_VACE_TE = 'umt5_xxl_fp8_e4m3fn_scaled.safetensors'
WAN_VACE_VAE = 'wan_2.1_vae.safetensors'
# lightx2v Wan2.1 T2V 14B step+CFG distill LoRA (Kijai/WanVideo_comfy, Lightx2v/ folder, rank 32): 4-6 steps at CFG 1.
WAN_DISTILL_LORA = 'lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank32_bf16.safetensors'
WAN_NEG = ('low quality, worst quality, deformed, distorted, disfigured, motion smear, blurry face, beard, mustache, glasses, male face, '
           'wrong face, different person, source actor face, skinny, thin, obese, extra fingers, watermark, text, cartoon, anime')


def wan_frames(seconds: float) -> int:
    """Real-time frame count at 16 fps, snapped UP to 4k+1 (VACE latent grid)."""
    n = max(1, math.ceil(seconds * WAN_FPS))
    return 4 * math.ceil((n - 1) / 4) + 1


WAN_MIN_WINDOW = 9                                       # shortest VACE window (frames) worth a separate sampler pass


def wan_windows(T: int, chunk: int = 81, overlap: int = 16, cuts=None) -> list[dict]:
    """Chained VACE window plan: [{'start', 'keep', 'length', 'at_cut'}]. A window's output frames start at `start` (its first
    `keep` frames re-generate the previous window's tail as mask-0 keep frames). Without cuts this is the original chain
    (81-frame windows, 16-frame overlap). With cuts (shot starts, 16 fps frame idx) a window that cannot reach T ends on the
    LAST cut in its range and the next window starts fresh at that cut (keep 0), so every join hides inside a real cut."""
    cuts = sorted({int(c) for c in (cuts or []) if 0 < int(c) < T})
    plan, pos, fresh, at_cut = [], 0, True, False
    while pos < T:
        keep = 0 if fresh else overlap
        start = pos - keep
        if T - start <= chunk:                               # the rest fits: one window to the end (cuts inside are fine)
            L = 4 * math.ceil((T - start - 1) / 4) + 1
            plan.append({'start': start, 'keep': keep, 'length': L, 'at_cut': at_cut}); break
        cand = [c for c in cuts if pos + WAN_MIN_WINDOW <= c <= start + chunk]
        if cand:
            end = max(cand)
            L = 4 * math.ceil((end - start - 1) / 4) + 1
            plan.append({'start': start, 'keep': keep, 'length': L, 'at_cut': at_cut})
            pos, fresh, at_cut = end, True, True
        else:
            plan.append({'start': start, 'keep': keep, 'length': chunk, 'at_cut': at_cut})
            pos, fresh, at_cut = start + chunk, False, False
    return plan


def wan_vace_lora(spec: ShotSpec, files: dict, tier: str = 'final', gpu: str = '5090', lora: str | None = None,
                  lora_strength: float = 1.0, control: str = 'softedge', strength: float = 0.95, steps: int = 20, cfg: float = 6.0,
                  shift: float = 8.0, long_edge: int = 1280, chunk: int = 81, overlap: int = 16, prompt_override: str | None = None,
                  negative: str | None = None, distill: str | None = None, distill_strength: float = 0.8,
                  sampler: str | None = None, ref_mode: str = 'first', attention: str = 'kitchen',
                  cuts: list[int] | None = None) -> Recipe:
    """Motion transfer / person swap with Wan 2.1 VACE 14B + an optional character LoRA (the v3f recipe that shipped for
    Character: softedge control @0.95, seed 42, 20 steps, cfg 6, uni_pc/simple, shift 8, LoRA @1.0), at REAL speed:
    seconds x 16 fps frames, generated as chained 81-frame windows. Window i>0 gets the previous window's last `overlap`
    frames as VACE keep-frames (mask 0) followed by softedge control (mask 1). Softedge (PiDiNet) is computed in-graph.
    NOTE: WanVaceToVideo only uses reference_image[:1]; extra refs are listed for parity but only the first conditions.
    distill='lightx2v' stacks the lightx2v Wan2.1 14B step/CFG distill LoRA (strength 0.5-1.0, default 0.8) with the character
    LoRA and switches to 6 steps final / 4 draft, CFG 1, lcm/simple (explicit steps/cfg/sampler still win when passed).
    ref_mode='grid' uses files['subject_grid'] (a 2x2 composite of the subject refs, see refgrid.py) so VACE's single
    reference slot carries several views; EXPERIMENTAL (VACE may copy the grid layout), A/B it against 'first'.
    attention='kitchen' (default) inserts core ModelAttentionBackend(comfy kitchen attention) = INT8 attention; Phase 1 ran
    VACE on plain PyTorch attention (83.6 s/step at 720p x 81 frames) because global Sage was off. 'pytorch' = no patch.
    cuts = source shot-cut positions as 16 fps frame indices (scene split): window joins are moved onto cuts and a window that
    starts at a cut starts fresh (no keep-overlap from the previous shot), so joins hide inside real cuts."""
    if attention not in ('kitchen', 'pytorch'):
        raise GraphError(f'unknown Wan attention {attention!r}')
    if distill not in (None, 'lightx2v'):
        raise GraphError(f'unknown Wan distill {distill!r}')
    if distill:
        steps = steps if steps != 20 else (6 if tier == 'final' else 4)
        cfg = cfg if cfg != 6.0 else 1.0
        sampler = sampler or 'lcm'
    sampler = sampler or 'uni_pc'
    if ref_mode not in ('first', 'grid'):
        raise GraphError(f'unknown ref_mode {ref_mode!r}')
    if ref_mode == 'grid' and not files.get('subject_grid'):
        raise GraphError("ref_mode='grid' needs files['subject_grid'] (build it with app.genjutsu.refgrid.composite_grid)")
    subj = list(files.get('subject_grid') or []) if ref_mode == 'grid' else list(files.get('subject') or [])
    if not subj:
        raise GraphError('Wan VACE needs a reference image of the new subject')
    src = files.get('source16') or files['source']
    T = wan_frames(spec.source_seconds or 5.0)
    W, H = _fit(spec, long_edge, 16)
    g = Graph({})
    unet = g.add('UNETLoader', {'unet_name': WAN_VACE_UNET, 'weight_dtype': 'fp8_e4m3fn'}, title='Wan 2.1 VACE 14B (fp16 file, fp8 at load)')
    model = unet
    if distill:
        model = g.add('LoraLoaderModelOnly', {'model': [model, 0], 'lora_name': WAN_DISTILL_LORA, 'strength_model': float(distill_strength)},
                      title=f'lightx2v step/CFG distill LoRA @{distill_strength}')
    if lora:
        model = g.add('LoraLoaderModelOnly', {'model': [model, 0], 'lora_name': lora, 'strength_model': float(lora_strength)},
                      title='character LoRA')
    if attention == 'kitchen':
        model = g.add('ModelAttentionBackend', {'model': [model, 0], 'attention': KITCHEN_ATTN}, title='comfy-kitchen INT8 attention')
    ms = g.add('ModelSamplingSD3', {'model': [model, 0], 'shift': float(shift)})
    te = g.add('CLIPLoader', {'clip_name': WAN_VACE_TE, 'type': 'wan', 'device': 'default'})
    vae = g.add('VAELoader', {'vae_name': WAN_VACE_VAE})
    pos_txt = prompt_override or spec.prompt or 'the person from the reference image, same pose, motion and camera framing as the control video'
    pos = g.add('CLIPTextEncode', {'text': pos_txt, 'clip': [te, 0]}, title='positive')
    neg = g.add('CLIPTextEncode', {'text': negative or WAN_NEG, 'clip': [te, 0]}, title='negative')
    lv = g.add('LoadVideo', {'file': src}, title=f'source @16 fps ({T} frames)')
    gv = g.add('GetVideoComponents', {'video': [lv, 0]})
    frames = g.add('ImageFromBatch', {'image': [gv, 0], 'batch_index': 0, 'length': T})
    if control == 'softedge':
        ctrl = g.add('PiDiNetPreprocessor', {'image': [frames, 0], 'safe': 'enable', 'resolution': (min(W, H) // 64) * 64},
                     title='softedge control (PiDiNet)')
    else:
        raise GraphError(f'unknown control {control!r}')
    ref = g.add('LoadImage', {'image': subj[0]}, title='reference image (VACE uses the first only)')
    plan = wan_windows(T, chunk, overlap, cuts)
    windows, decs = [], []
    for i, w in enumerate(plan):
        start, keep, L = w['start'], w['keep'], w['length']
        if not keep:
            cv = g.add('ImageFromBatch', {'image': [ctrl, 0], 'batch_index': start, 'length': L},
                       title=f'window {i + 1} control' + (' (fresh at a cut)' if i and w.get('at_cut') else ''))
            masks = None
        else:
            pw = plan[i - 1]
            new = L - keep
            keepi = g.add('ImageFromBatch', {'image': [decs[-1], 0], 'batch_index': start - pw['start'], 'length': keep})
            nxt = g.add('ImageFromBatch', {'image': [ctrl, 0], 'batch_index': start + keep, 'length': new})
            cv = g.add('ImageBatch', {'image1': [keepi, 0], 'image2': [nxt, 0]}, title=f'window {i + 1} control')
            black = g.add('EmptyImage', {'width': W, 'height': H, 'batch_size': keep, 'color': 0})
            white = g.add('EmptyImage', {'width': W, 'height': H, 'batch_size': new, 'color': 16777215})
            mimg = g.add('ImageBatch', {'image1': [black, 0], 'image2': [white, 0]})
            masks = g.add('ImageToMask', {'image': [mimg, 0], 'channel': 'red'}, title='0 = keep overlap, 1 = generate')
        vin = {'positive': [pos, 0], 'negative': [neg, 0], 'vae': [vae, 0], 'control_video': [cv, 0], 'reference_image': [ref, 0],
               'width': W, 'height': H, 'length': L, 'batch_size': 1, 'strength': float(strength)}
        if masks:
            vin['control_masks'] = [masks, 0]
        wv = g.add('WanVaceToVideo', vin, title=f'VACE window {i + 1} ({L}f)')
        ks = g.add('KSampler', {'model': [ms, 0], 'positive': [wv, 0], 'negative': [wv, 1], 'latent_image': [wv, 2], 'seed': int(spec.seed),
                                'steps': int(steps), 'cfg': float(cfg), 'sampler_name': sampler, 'scheduler': 'simple', 'denoise': 1.0})
        tr = g.add('TrimVideoLatent', {'samples': [ks, 0], 'trim_amount': [wv, 3]})
        dec = g.add('VAEDecode', {'samples': [tr, 0], 'vae': [vae, 0]})
        decs.append(dec); windows.append((dec, L))
    out = None
    for i, (dec, L) in enumerate(windows):
        last = i == len(windows) - 1
        part = dec if last else g.add('ImageFromBatch', {'image': [dec, 0], 'batch_index': 0, 'length': plan[i + 1]['start'] - plan[i]['start']})
        out = part if out is None else g.add('ImageBatch', {'image1': [out, 0], 'image2': [part, 0]})
    out = g.add('ImageFromBatch', {'image': [out, 0], 'batch_index': 0, 'length': T}, title=f'trim to {T} frames')
    cvid = {'images': [out, 0], 'fps': float(WAN_FPS)}
    if spec.keep_audio:
        cvid['audio'] = [gv, 1]
    vid = g.add('CreateVideo', cvid)
    sv = g.add('SaveVideo', {'video': [vid, 0], 'filename_prefix': f'genjutsu/wan_vace_{"lora" if lora else "base"}', 'format': 'mp4', 'codec': 'h264'})
    params = {'frames': T, 'fps': WAN_FPS, 'seconds': round(T / WAN_FPS, 3), 'width': W, 'height': H, 'windows': [L for _, L in windows],
              'overlap': overlap, 'steps': steps, 'cfg': cfg, 'shift': shift, 'strength': strength, 'control': control, 'lora': lora,
              'lora_strength': lora_strength, 'seed': spec.seed, 'sampler': f'{sampler}/simple',
              'distill': f'{WAN_DISTILL_LORA} @{distill_strength}' if distill else None, 'ref_mode': ref_mode,
              'attention': attention, 'cuts': list(cuts or []), 'window_plan': plan}
    return _done('wan_vace_lora', '(built in code; v3f recipe)', g, [sv], [src, subj[0]], params,
                 [f'{len(windows)} chained VACE windows {params["windows"]} (starts {[w["start"] for w in plan]}, keep '
                  f'{[w["keep"] for w in plan]}): joins at shot cuts start fresh, other joins use {overlap}-frame keep overlap',
                  'only reference_image[:1] conditions VACE (ComfyUI WanVaceToVideo); identity comes mostly from the LoRA',
                  'PiDiNet softedge in-graph; the v3 control clip that shipped was made offline and its exact detector is not recorded'])


RECIPES['wan_vace_lora'] = wan_vace_lora


def _done(name, template, g: Graph, outputs, uploads, params, notes) -> Recipe:
    nodes = g.to_json()
    assert_no_plain_pdd(nodes)
    probs = validate(nodes)
    return Recipe(name, template, nodes, outputs, sorted(set(uploads)), params, notes, probs)


def build(recipe: str, spec: ShotSpec, files: dict, tier: str = 'draft', gpu: str = '5090', **kw) -> Recipe:
    if recipe not in RECIPES:
        raise GraphError(f'unknown recipe {recipe!r}')
    return RECIPES[recipe](spec, files, tier=tier, gpu=gpu, **kw)
