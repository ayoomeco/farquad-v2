#!/usr/bin/env python3
"""Phase 1 Genjutsu bake-off runner. PREPARED, NOT LAUNCHED (Phase 0 is $0).

    python scripts/phase1_bakeoff.py prep                 # free: make 5 s trims / 16 fps copies / derived frames
    python scripts/phase1_bakeoff.py plan [--with-veda] [--with-x2] [--option A|B]   # free: graphs + time + $ table
    python scripts/phase1_bakeoff.py launch --i-approve-spend [--cap 5]              # SPENDS MONEY (RunPod)
    python scripts/phase1_bakeoff.py plan --ladder        # free: L0-L5 H3 fast-path ladder quote (2 seeds/rung, g8 window)
    python scripts/phase1_bakeoff.py launch --ladder --i-approve-spend --cap 2.5      # SPENDS MONEY (needs the owner's OK)

Live progress: ComfyUI websocket (per-node wall time, per-step timestamps; warm s/step = steps after the first of each bar).
The first run after a model change is flagged cold; ladder rungs are ranked on warm runs (data/bakeoff/phase1/ladder_rank.json).
A run whose live warm s/step drifts > 1.5x the config's baseline is interrupted, ComfyUI is restarted via the pod fix
channel and the run is requeued once. No websocket -> /history polling + comfy.log tqdm (closing line de-duplicated).

What it compares (one 5090, PRO 6000 only with --allow-pro6000), in this order:
  FIRST (never trimmed): the owner's Genjutsu clip, 8 s window t=0..8 s at native speed, 720p, same 5 Character refs:
                       H3 Ref2VA PDD-8 (KJ low-VRAM attention + chunked FFN, NVFP4 TE on cu130)
                       vs Wan 2.1 VACE 14B + Character LoRA step 2500 (v3f recipe, 2 chained windows)
                       then H3 HyperFlow-8; back to back on one ComfyUI (both can't sit in 32 GB at once);
                       a labelled side-by-side data/bakeoff/phase1/g8_compare.mp4 is made locally afterwards.
  The Character LoRA is not on Hugging Face: it is uploaded from the box in 90 MB parts and sha256-checked on the pod.
  H3 motion transfer:  base 20-step  vs  PDD-Acc 8-step  vs  HyperFlow 8-step   (+ PDD 4-step draft)
  person swap:         Viggle-Animate (primary)  vs  SCAIL-2 (fallback)
  object swap:         H3 Fun ControlNet Union 2.0 + SAM 3.1 mask (draft PDD-4, final base-20)
  restyle:             LTX-2.3 ReStyle IC-LoRA (style image)  vs  H3 Ref2VA preset restyle (no style image)
  upscale:             FlashVSR v1.1 720p -> 1080p (bulk path)
  optional:            --with-veda (Veda sparse attention, sm120) and --with-x2 (X2 Detail VAE 2x decode)
Logged per run: wall s, ~s/step, peak VRAM, $ at the live pod rate, free QC checks, output path.

Launch gates (ALL required, any failure = nothing is created):
  1. `launch --i-approve-spend` on an interactive terminal, and the owner types the exact phrase `SPEND <cap>`.
  2. An Approval(confirmed=True, max_usd=cap) with cap <= the provider hard cap ($5, config/genie.toml).
  3. RUNPOD_API_KEY from the environment only (never read from or written to a file), balance >= cap + $1.
Spend control: box-side Watchdog terminates at 92 % of the cap / max seconds / 15 min without progress;
a pod-side dead-man switch (`sleep N; remove pod`) starts at boot; the pod is terminated on exit, error,
Ctrl-C and SIGTERM. Runs execute in priority order; other clips' runs are trimmed (plan) / skipped (live) before the head-to-head is."""
from __future__ import annotations
import argparse, base64, io, json, os, pathlib, re, subprocess, sys, tarfile, time, urllib.error, urllib.request, uuid

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app import config                                                     # noqa: E402
from app.genjutsu import planner, qc                                       # noqa: E402
from app.genjutsu.shotspec import ShotSpec, Ref                            # noqa: E402
from app.providers.base import Approval, ApprovalError, BudgetExceeded     # noqa: E402
sys.path.insert(0, str(ROOT / 'scripts'))
from comfy_progress import (RunTrace, ComfyWS, DriftGuard, dedup_bars, drift_key, model_signature,  # noqa: E402
                            rank as rank_runs)

CLIPS = ROOT / 'data' / 'bakeoff' / 'clips'
OUT = ROOT / 'data' / 'bakeoff' / 'phase1'
BASE_IMAGE = 'runpod/base:1.4.0-cuda1300-ubuntu2404'     # same base as worker/Dockerfile; install.sh runs at boot
# FP_IMAGE: opt-in prebuilt worker image (e.g. ghcr.io/ayoomeco/genie-jutsu-worker@sha256:...) with /opt/jutsu baked: install.sh
# finds its .install-ok marker and skips; models still come from the boot downloader. Default stays BASE_IMAGE (proven path).
IMAGE = os.environ.get('FP_IMAGE') or BASE_IMAGE
# FP_AVOID_MACHINES: comma list of RunPod machine ids; a pod landing on one is terminated at once and creation retried.
AVOID_MACHINES = {m.strip() for m in (os.environ.get('FP_AVOID_MACHINES') or '').split(',') if m.strip()}
RUN_SECONDS = 5.0                                         # every comparison uses a 5 s trim (H3 native window)
SETUP_EST_S = {'boot_and_apt': 240, 'install': 1080, 'models': 1700, 'lora_upload': 60}  # ESTIMATES: torch+ComfyUI+nodes+Sage build; ~164 GB pull
ENGINE_LOAD_EST_S = 120
# $0 placeholder style frame (posterized first frame). A real restyled keyframe (image model / the owner's pick) is a fairer LTX test.
POSTER = ("eq=brightness=0.05:gamma=1.4:saturation=2.0,bilateral=sigmaS=24:sigmaR=0.25,bilateral=sigmaS=24:sigmaR=0.25,"
          "lutrgb=r='floor(val/40)*40+20':g='floor(val/40)*40+20':b='floor(val/40)*40+20',unsharp=7:7:2.5")                                   # first load of each engine's weights from disk
ACCEL_FACTOR = {'base20': 1.0, 'pdd8': 0.45, 'hyperflow8': 0.45, 'pdd4': 0.25}   # vs the 20-step final estimate
OPTIONS = {
    'A': {'label': 'Secure Cloud 5090 + 250 GB network volume (models persist for Phase 2+)', 'cloud': 'SECURE',
          'rate_key': 'usd_per_hr', 'volume_gb': 250, 'container_disk_gb': 40},
    'B': {'label': 'Community Cloud 5090, no volume, 260 GB container disk (nothing persists; cheapest one-off)',
          'cloud': 'COMMUNITY', 'rate_key': 'community_usd_per_hr', 'volume_gb': 0, 'container_disk_gb': 260},
    'L': {'label': 'Community Cloud 5090, no volume, 120 GB container disk (speed ladder: ~62 GB of H3 weights only)',
          'cloud': 'COMMUNITY', 'rate_key': 'community_usd_per_hr', 'volume_gb': 0, 'container_disk_gb': 120},
    'H': {'label': 'Community Cloud 5090, no volume, 200 GB container disk (10 s head-to-head: 108 GB ComfyUI weights + WanGP ~35 GB)',
          'cloud': 'COMMUNITY', 'rate_key': 'community_usd_per_hr', 'volume_gb': 0, 'container_disk_gb': 200},
    'V': {'label': 'Community Cloud 5090, no volume, 180 GB container disk (Oct 13 better build: ~128 GB H3 NVFP4 + Viggle bf16 + swap LoRAs)',
          'cloud': 'COMMUNITY', 'rate_key': 'community_usd_per_hr', 'volume_gb': 0, 'container_disk_gb': int(os.environ.get('FP_DISK_GB', '180'))},
    'R': {'label': 'Community Cloud 5090, no volume, 120 GB container disk (Oct 8 refine: ~58 GB of H3 NVFP4 + PDD weights)',
          'cloud': 'COMMUNITY', 'rate_key': 'community_usd_per_hr', 'volume_gb': 0, 'container_disk_gb': 120},
}

# ---- L0-L5 fast-path ladder (Oct 7 review): same 8 s Genjutsu window, 720p, 5 Character refs, motion ref 544, 2 seeds per rung.
# Order = execution order: NVFP4 rungs back to back, then the int8 ones (one trunk swap); fastest / most decision-relevant first.
# est_s = ESTIMATED warm wall per seed on a 5090 (Phase 1 measured 110.7 s/step for L0-like PDD-8 with a full-res motion ref;
# Veda measured 14.56 vs 43.7 s/step on base20). cold_s = extra for the first run after a model change.
LADDER = [
    ('L3', 'turbo4 (lightx2v Ref2V 4-step) + NVFP4 + kitchen attention', {'accel': 'turbo4'}, 180, 150),
    ('L2', 'PDD-8 (pack) + NVFP4 + comfy-kitchen INT8 attention (KJ attention dropped)', {'accel': 'pdd8'}, 300, 60),
    ('L4', 'PDD-4 (pack, nfe 4) + NVFP4 + kitchen attention', {'accel': 'pdd4'}, 180, 60),
    ('L2v', 'PDD-8 + NVFP4 + Veda sparse attention instead of kitchen', {'accel': 'pdd8', 'veda': True}, 240, 60),
    ('L1', 'PDD-8 + NVFP4 trunk, PyTorch attention + KJ low-VRAM attention/FFN', {'accel': 'pdd8', 'attention': 'pytorch', 'kj_attn': True}, 500, 60),
    ('L5', 'turbo4 + Ref2VA VSA gate 0.75 (forces the int8 trunk)', {'accel': 'turbo4', 'vsa': True, 'vsa_sparsity': 0.75}, 140, 150),
    ('L0', 'Phase-1 config: int8 trunk, PyTorch attention + KJ attn/FFN, PDD-8 (now via the pack), motion ref 544',
     {'accel': 'pdd8', 'unet_file': 'minimax_h3_ref2va_pruned_int8_convrot.safetensors', 'attention': 'pytorch', 'kj_attn': True}, 650, 60),
]
LADDER_SEEDS = (42, 1234)
LADDER_SETUP_S = 720                                      # Phase 1 measured 680 s to ComfyUI-ready with a bigger stage-1 pull
LADDER_LINGER_S = 120
LADDER_STAGE1 = ['h3', 'h3_fast', 'h3_vsa', 'veda']
# Astra brief A1: WanGP v17.17 + MMGP v4 (released Oct 7 2026) as a separate-app rung. REAL (deepbeepmeep/Wan2GP README) and
# headless-capable (`wgp.py --process settings.json --attention sage2 --profile N`), but NOT wired here: it needs its own
# venv (torch 2.10 cu130 + Sage2 build) and its own H3 checkpoint pull. Now wired for --h2h via worker/wangp_install.sh +
# worker/wangp_driver.py (WanGP Python API, settings keys read from the pinned handler source); the ladder only quotes it.
WANGP_ADDON = {'rung': 'L6w', 'what': 'WanGP v17.17 + MMGP v4 (Profile 4, Sage2), H3 Ref2VA PDD-8, same g8 inputs',
               'setup_s': 900 + 300, 'est_s': 2 * 260 + 150}


G8_SECONDS = 8.0                                          # the owner: 8 s window of the Genjutsu clip (t=0..8 s, native speed)
H3_720P_MP = 0.9216                                       # 1280x720
H3_G8_PROMPT = ("<Subject 1> is the character from the reference images. <Video 1> provides the motion. "  # example prompt; write your own per clip
    "Replace the person in <Video 1> with <Subject 1>, following <Video 1> moment by moment.")
WAN_G8_PROMPT = ("the character from the reference images, same body pose, motion and camera framing as the control video, "  # example prompt
    "photorealistic.")


# ----------------------------------------------------------------------------------------------- inputs (free)
def _ff(*args):
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-y', *map(str, args)], check=True)


def prep(log=print) -> dict:
    """Recreate every input from config/phase1_clips.json. Free, local ffmpeg only. Idempotent."""
    cfg = json.loads((ROOT / 'config' / 'phase1_clips.json').read_text())
    CLIPS.mkdir(parents=True, exist_ok=True)
    made = []
    for key, c in cfg['clips'].items():
        dst = CLIPS / c['file']
        if not dst.exists() and pathlib.Path(c['origin']).exists():
            _ff('-ss', c['ss'], '-i', c['origin'], '-t', c['t'], '-c:v', 'libx264', '-crf', 17, '-preset', 'fast', '-c:a', 'aac', dst)
        if not dst.exists():
            raise FileNotFoundError(f'{dst} missing and origin {c["origin"]} not found')
        if key == 'genjutsu8':                                # head-to-head clip: full 8 s window, two engine-native copies
            if not (CLIPS / 'genjutsu8_720p24.mp4').exists():
                _ff('-i', dst, '-vf', 'scale=1280:720:flags=lanczos,fps=24', '-c:v', 'libx264', '-crf', 16, '-preset', 'fast',
                    '-c:a', 'aac', CLIPS / 'genjutsu8_720p24.mp4')
            if not (CLIPS / 'genjutsu8_16fps.mp4').exists():
                _ff('-i', dst, '-vf', 'scale=1280:720:flags=lanczos,fps=16', '-c:v', 'libx264', '-crf', 16, '-preset', 'fast',
                    '-c:a', 'aac', CLIPS / 'genjutsu8_16fps.mp4')
            made.append(dst.name); continue
        five = CLIPS / (dst.stem + '_5s.mp4')
        if not five.exists():
            _ff('-i', dst, '-t', RUN_SECONDS, '-c:v', 'libx264', '-crf', 17, '-preset', 'fast', '-c:a', 'aac', five)
        made.append(five.name)
    r = cfg['refs']
    if not (CLIPS / r['subject']['file']).exists() and pathlib.Path(r['subject']['origin']).exists():
        _ff('-i', r['subject']['origin'], CLIPS / r['subject']['file'])
    if not (CLIPS / r['object']['file']).exists() and pathlib.Path(r['object']['origin']).exists():
        _ff('-ss', r['object']['ss'], '-i', r['object']['origin'], '-frames:v', 1, '-vf', f"crop={r['object']['crop']}", CLIPS / r['object']['file'])
    for img in (config.characters().get(os.environ.get('JUTSU_PACK', 'pack')) or {}).get('reference_images', []):   # optional pack refs (same for both engines)
        if pathlib.Path(img).exists() and not (CLIPS / pathlib.Path(img).name).exists():
            (CLIPS / pathlib.Path(img).name).write_bytes(pathlib.Path(img).read_bytes())
    for f in sorted(CLIPS.glob('*_5s.mp4')):
        s16 = CLIPS / (f.stem + '_16fps.mp4')                 # Wan/SCAIL-2 recipes read <source>_16fps.mp4
        if not s16.exists():
            _ff('-i', f, '-vf', 'fps=16', '-c:v', 'libx264', '-crf', 17, '-an', s16)
        f0 = CLIPS / (f.stem + '_frame0.png')                 # H3 preset-restyle anchor (planner `derived`)
        if not f0.exists():
            _ff('-i', f, '-frames:v', 1, f0)
    scene = CLIPS / 'restyle_scene_5s.mp4'
    style = CLIPS / 'style_ref_cel.png'                       # $0 "repainted first frame" for LTX ReStyle (frame 0 of the IC guide)
    if not style.exists():
        _ff('-i', scene, '-frames:v', 1, '-vf', POSTER, style)
    up = CLIPS / 'person_fullbody_5s_720p.mp4'                # FlashVSR input (720p -> 1080p)
    if not up.exists():
        _ff('-i', CLIPS / 'person_fullbody_5s.mp4', '-vf', 'scale=-2:1280', '-c:v', 'libx264', '-crf', 18, '-c:a', 'copy', up)
    log(f'[prep] inputs ready in {CLIPS}: {sorted(p.name for p in CLIPS.iterdir())}')
    return {'clips': made}


# ----------------------------------------------------------------------------------------------- runs
def _spec(mode, clip, refs=(), **kw) -> ShotSpec:
    return ShotSpec(mode=mode, source_video=str(CLIPS / clip), source_seconds=RUN_SECONDS,
                    refs=[Ref(path=str(CLIPS / p), role=role) for p, role in refs], keep_audio=True, **kw)


def runs(with_veda: bool = False, with_x2: bool = False) -> list[dict]:
    """Priority order = execution order (most decision-relevant first, so a budget stop loses the least)."""
    P, O, S = 'person_fullbody_5s.mp4', 'object_phone_5s.mp4', 'restyle_scene_5s.mp4'
    subj = [('subject_ref_fullbody.png', 'subject')]
    R = []

    def add(rid, question, spec, engine, tier, ov=None, est=None, protect=False):
        R.append({'id': rid, 'question': question, 'spec': spec.model_copy(update={'engine': engine, 'tier': tier}),
                  'tier': tier, 'overrides': ov or {}, 'est_s': est, 'protect': protect})
    # ---- 1. the owner's Genjutsu clip, 8 s @ native speed, 720p: H3 Ref2VA (speed fixes) vs Wan VACE 14B + Character LoRA 2500
    pack = [(pathlib.Path(x).name, 'subject') for x in (config.characters().get(os.environ.get('JUTSU_PACK', 'pack')) or {}).get('reference_images', [])]
    g8 = lambda src, prompt: ShotSpec(mode='character_swap', source_video=str(CLIPS / src), source_seconds=G8_SECONDS, source_size=(1920, 1080),
                                      refs=[Ref(path=str(CLIPS / p), role=r) for p, r in pack], keep_audio=True, prompt=prompt,
                                      character_pack=os.environ.get('JUTSU_PACK', 'pack'), max_resolution=1280, seed=42)
    add('g8_h3_pdd8_720p', 'HEAD-TO-HEAD 8 s: H3 Ref2VA PDD-8 + KJ low-VRAM attn + chunked FFN, NVFP4 TE (cu130), 720p, 5 Character refs',
        g8('genjutsu8_720p24.mp4', ''), 'genjutsu_h3_ref2va', 'final',
        {'accel': 'pdd8', 'vae': 'standard', 'megapixels': H3_720P_MP, 'prompt_override': H3_G8_PROMPT}, est=600, protect=True)
    add('g8_wan_vace_pack2500_720p', 'HEAD-TO-HEAD 8 s: Wan 2.1 VACE 14B + Character LoRA step 2500 (v3f: softedge 0.95, 20 steps, cfg 6), 720p',
        g8('genjutsu8.mp4', WAN_G8_PROMPT), 'genjutsu_wan_vace_lora', 'final',
        {'prompt_override': WAN_G8_PROMPT, 'long_edge': 1280}, est=1840, protect=True)
    add('g8_h3_hyperflow8_720p', 'HEAD-TO-HEAD 8 s: H3 HyperFlow-8 (second few-step variant), 720p, same refs/prompt',
        g8('genjutsu8_720p24.mp4', ''), 'genjutsu_h3_ref2va', 'final',
        {'accel': 'hyperflow8', 'vae': 'standard', 'megapixels': H3_720P_MP, 'prompt_override': H3_G8_PROMPT}, est=600, protect=True)
    # ---- 2. the generic bake-off (trimmed from the END of this list first if the cap gets tight)
    mt = lambda: _spec('motion_transfer', P, subj, prompt='the man from the reference image performs the same moves')
    add('h3_mt_pdd4_draft', 'H3 draft speed/quality (PDD-4, light VAE)', mt(), 'genjutsu_h3_ref2va', 'draft', {'accel': 'pdd4'})
    add('h3_mt_base20_final', 'H3 accel bake-off: baseline', mt(), 'genjutsu_h3_ref2va', 'final', {'accel': 'base20'})
    add('h3_mt_pdd8_final', 'H3 accel bake-off: PDD-Acc 8-step', mt(), 'genjutsu_h3_ref2va', 'final', {'accel': 'pdd8'})
    add('h3_mt_hyperflow8_final', 'H3 accel bake-off: HyperFlow-8 (pruned base, off-recipe)', mt(), 'genjutsu_h3_ref2va', 'final',
        {'accel': 'hyperflow8'})
    sw = lambda: _spec('character_swap', P, subj, prompt='replace the woman with the man from the reference image')
    add('swap_viggle_final', 'person swap: Viggle-Animate (primary)', sw(), 'genjutsu_viggle', 'final')
    add('swap_scail2_final', 'person swap: SCAIL-2 (fallback)', sw(), 'genjutsu_scail2', 'final')
    obj = lambda: _spec('object_swap', O, [('object_ref_phone.png', 'object')], target='phone',
                        prompt='a phone in a black case showing a video call')
    add('objswap_h3cn_pdd4_draft', 'object swap draft (PDD-4 with Fun ControlNet: UNTESTED combo)', obj(), 'genjutsu_h3_objswap', 'draft',
        {'accel': 'pdd4'})
    add('objswap_h3cn_base20_final', 'object swap final (CN Union 2.0, base 20)', obj(), 'genjutsu_h3_objswap', 'final', {'accel': 'base20'})
    add('restyle_ltx_final', 'restyle: LTX ReStyle IC-LoRA w/ style image', _spec('restyle', S, [('style_ref_cel.png', 'style')],
        style='anime cel-shaded'), 'genjutsu_ltx_restyle', 'final')
    add('restyle_h3_preset_draft', 'restyle: H3 preset, no style image (API default path)', _spec('restyle', S, style='anime cel-shaded',
        preset_id='anime-cel'), 'genjutsu_h3_ref2va', 'draft', {'accel': 'pdd4'})
    add('upscale_flashvsr', 'bulk upscale FlashVSR 720p->1080p', _spec('upscale', 'person_fullbody_5s_720p.mp4', max_resolution=1920),
        'genjutsu_flashvsr', 'final')
    if with_x2:
        add('h3_mt_pdd8_x2', 'X2 Detail VAE 2x decode (shimmer check)', mt(), 'genjutsu_h3_ref2va', 'final', {'accel': 'pdd8', 'vae': 'x2'})
    if with_veda:
        add('h3_mt_base20_veda', 'Veda sparse attention on R2VA (UNVERIFIED: T2VA predictor)', mt(), 'genjutsu_h3_ref2va', 'final',
            {'accel': 'base20', 'veda': True})
        add('swap_viggle_veda', 'Veda on Viggle (UNVERIFIED)', sw(), 'genjutsu_viggle', 'final', {'veda': True})
    return R


def ladder_runs(seeds=LADDER_SEEDS) -> list[dict]:
    pack = [(pathlib.Path(x).name, 'subject') for x in (config.characters().get(os.environ.get('JUTSU_PACK', 'pack')) or {}).get('reference_images', [])]
    R = []
    for rung, label, ov, est, cold in LADDER:
        for i, seed in enumerate(seeds):
            spec = ShotSpec(mode='character_swap', source_video=str(CLIPS / 'genjutsu8_720p24.mp4'), source_seconds=G8_SECONDS,
                            source_size=(1280, 720), refs=[Ref(path=str(CLIPS / p), role=r) for p, r in pack], keep_audio=True,
                            prompt='', character_pack=os.environ.get('JUTSU_PACK', 'pack'), max_resolution=1280, seed=seed, motion_ref_short_edge=544)
            R.append({'id': f'ladder_{rung}_s{seed}', 'rung': rung, 'seed': seed, 'question': f'{rung}: {label}',
                      'spec': spec.model_copy(update={'engine': 'genjutsu_h3_ref2va', 'tier': 'final'}), 'tier': 'final',
                      'overrides': {'vae': 'standard', 'megapixels': H3_720P_MP, 'prompt_override': H3_G8_PROMPT, **ov},
                      'est_s': est + (cold if i == 0 else 0), 'protect': rung not in ('L0', 'L1')})
    return R


# ----------------------------------------------------------------------------------------------- Oct 7b: 10 s head-to-head
# the owner's own inputs (any subject, NOT Character): one reference photo + one motion clip window, every engine on the same files.
# H3 rungs = the fixed fast paths (ladder), plus Wan 2.1 VACE + lightx2v distill (no character LoRA) and WanGP/MMGP v4.
# est_s = ESTIMATED warm wall per seed on a 5090 for a 10 s 720p window (ladder 8 s estimates x (10/8)^1.3, 1 ref picture).
H2H_SEEDS = (42, 1234)
H2H_SCALE = (10 / 8) ** 1.3
H2H_RUNGS = [
    ('H3-T4', 'H3 Turbo-4 (lightx2v Ref2V 4-step) + NVFP4 + kitchen attention', {'accel': 'turbo4'}, 180, 150),
    ('H3-P8', 'H3 PDD-8 (PDD-Acc pack, trained sigmas) + NVFP4 + kitchen attention', {'accel': 'pdd8'}, 300, 60),
    ('H3-P4', 'H3 PDD-4 (pack, nfe 4) + NVFP4 + kitchen attention', {'accel': 'pdd4'}, 180, 60),
    ('H3-HF8', 'H3 HyperFlow-8 (int8 trunk, HyperFlow directly after the loader = curve refit on) + kitchen', {'accel': 'hyperflow8'}, 330, 150),
    ('H3-P8v', 'H3 PDD-8 + NVFP4 + Veda sparse attention', {'accel': 'pdd8', 'veda': True}, 240, 60),
    ('H3-T4vsa', 'H3 Turbo-4 + Ref2VA VSA gate 0.75 (int8 trunk)', {'accel': 'turbo4', 'vsa': True, 'vsa_sparsity': 0.75}, 140, 150),
]
H2H_BASELINE = ('H3-L0', 'H3 Phase-1 config (int8, PyTorch attn + KJ, PDD-8): speed control only',
                {'accel': 'pdd8', 'unet_file': 'minimax_h3_ref2va_pruned_int8_convrot.safetensors', 'attention': 'pytorch', 'kj_attn': True}, 650, 60)
# Wan 2.1 VACE 14B + lightx2v T2V distill @0.8, 6 steps CFG 1 lcm, kitchen attention, softedge, cut-aligned windows.
# Phase 1: 83.6 s/step per 81-frame 720p window on PyTorch attention; kitchen INT8 attention assumed ~1.6x -> ~52 s/step.
H2H_WAN = ('WAN-VACE', 'Wan 2.1 VACE 14B + lightx2v distill (6 steps, CFG 1) + kitchen attention, softedge, ref photo, no LoRA',
           {'distill': 'lightx2v', 'attention': 'kitchen', 'long_edge': 1280}, 760, 220)
# WanGP v17.17 / MMGP v4 (separate app, worker/wangp_install.sh + worker/wangp_driver.py): H3 Ref2VA pruned PDD-8, same inputs.
WANGP_MODEL = 'minimax_h3_ref2va_pruned_pdd'
H2H_WANGP = ('WANGP-P4', 'WanGP/MMGP v4 Profile 4 + Sage2: H3 Ref2VA pruned PDD-8, same photo + motion (544 short edge)', 420, 300)
H2H_SETUP_S = 840                                         # stage-1 = H3 bundles (66 GB); Wan VACE (42 GB) pulls during the H3 rungs
H2H_LINGER_S = 120
H2H_STAGE1 = ['h3', 'h3_fast', 'h3_vsa', 'veda', 'h3_hyperflow']
H2H_OUT = ROOT / 'data' / 'bakeoff' / 'h2h'


def scene_cuts(path, start: float = 0.0, seconds: float = 10.0, thr: float = 0.35, merge_s: float = 0.2) -> list[dict]:
    """Shot cuts inside [start, start+seconds] from ffmpeg's scene score (free, local). Returns [{'t' (window-relative s), 'score'}];
    cuts closer than merge_s keep the strongest. Quick-cut action clips also score fast motion ~0.3, hence thr 0.35."""
    r = subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-ss', str(start), '-t', str(seconds), '-i', str(path), '-an', '-vf',
                        "scale=320:-2,select='gte(scene,0)',metadata=print:key=lavfi.scene_score:file=-", '-f', 'null', '-'],
                       capture_output=True, text=True, check=True)
    pts, cur = [], None
    for line in r.stdout.splitlines():
        m = re.search(r'pts_time:([\d.]+)', line)
        if m:
            cur = float(m.group(1))
        m = re.search(r'scene_score=([\d.]+)', line)
        if m and cur is not None and float(m.group(1)) >= thr and cur > 0.05:
            pts.append({'t': round(cur, 3), 'score': round(float(m.group(1)), 3)})
    out = []
    for c in pts:
        if out and c['t'] - out[-1]['t'] < merge_s:
            if c['score'] > out[-1]['score']:
                out[-1] = c
            continue
        out.append(c)
    return out


def load_brief(path) -> dict:
    b = json.loads(pathlib.Path(path).read_text())
    for k in ('motion', 'ref'):
        if b.get(k) and not pathlib.Path(b[k]).is_absolute():
            b[k] = str(ROOT / b[k])
    return b


def prep_h2h(motion, ref, start: float = 0.0, seconds: float = 10.0, log=print) -> dict:
    """Free, local: cut the window and make every engine-native copy into CLIPS (flat, so uploads resolve):
    h2h_motion.mp4 1280x720@24 (+audio; H3 rescales its motion ref to 544 in-graph), h2h_motion_16fps.mp4 (Wan),
    h2h_motion_544.mp4 968x544@24 (WanGP gets the same 544 motion ref), h2h_ref.png (photo, long edge <= 1536, as-is otherwise).
    The window carries a few extra frames so 24 fps -> 241 and 16 fps -> 161 frame grids are fully covered."""
    CLIPS.mkdir(parents=True, exist_ok=True); H2H_OUT.mkdir(parents=True, exist_ok=True)
    t = seconds + 0.08
    vf = 'scale=1280:720:force_original_aspect_ratio=decrease:flags=lanczos,pad=ceil(iw/2)*2:ceil(ih/2)*2'
    _ff('-ss', start, '-i', motion, '-t', t, '-vf', f'{vf},fps=24', '-c:v', 'libx264', '-crf', 16, '-preset', 'fast', '-c:a', 'aac',
        '-b:a', '192k', CLIPS / 'h2h_motion.mp4')
    _ff('-i', CLIPS / 'h2h_motion.mp4', '-vf', 'fps=16', '-c:v', 'libx264', '-crf', 16, '-preset', 'fast', '-c:a', 'copy',
        CLIPS / 'h2h_motion_16fps.mp4')
    _ff('-i', CLIPS / 'h2h_motion.mp4', '-vf', 'scale=-2:544:flags=area', '-c:v', 'libx264', '-crf', 16, '-preset', 'fast',
        '-c:a', 'copy', CLIPS / 'h2h_motion_544.mp4')
    _ff('-i', ref, '-vf', "scale='min(1536,iw)':-2:flags=lanczos", '-frames:v', 1, CLIPS / 'h2h_ref.png')
    cuts = scene_cuts(motion, start, seconds)
    pr = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height,r_frame_rate',
                         '-of', 'json', str(motion)], capture_output=True, text=True)
    info = {'motion': str(motion), 'ref': str(ref), 'start': start, 'seconds': seconds, 'cuts': cuts,
            'source': (json.loads(pr.stdout or '{}').get('streams') or [{}])[0],
            'files': ['h2h_motion.mp4', 'h2h_motion_16fps.mp4', 'h2h_motion_544.mp4', 'h2h_ref.png']}
    (H2H_OUT / 'window.json').write_text(json.dumps(info, indent=1))
    log(f'[h2h] window {start:g}-{start + seconds:g} s, {len(cuts)} cuts: {[c["t"] for c in cuts]}')
    return info


def h2h_h3_prompt(b: dict, seconds: float) -> str:
    """Context-IR (H3 card section order) for a character swap through a fast-cut clip, one face-only picture."""
    shots = b.get('shots') or [[0.0, seconds, True, 'follows <Video 1> moment by moment']]
    with_s = [f'[Shot {i + 1}]' for i, s in enumerate(shots) if s[2]]
    desc = [b.get('style', '')] + [f'[Shot {i + 1}] {s[0]:.1f}-{s[1]:.1f} s: {s[3]}.' for i, s in enumerate(shots)]
    return '\n'.join([
        'subject_definitions:',
        f'<Subject 1> is {b["subject"]}, shown in <Picture 1>. {b.get("ref_note", "")}'.strip(),
        f'<Video 1> is the source video for the editing task. {b["role"][0].upper() + b["role"][1:]} is replaced by <Subject 1> in every '
        f'shot where he appears; none of his identity is kept. <Subject 1> wears {b["wardrobe"]}. {b.get("keep", "")}'.strip(),
        '<Audio 1> is the synchronized audio track of <Video 1>.', '',
        'summary:',
        f'[video editing + audio reuse] The target video is <Video 1> edited so that <Subject 1> plays the main man, following <Video 1> '
        f'cut by cut and moment by moment for {seconds:g} seconds.', '',
        'retention_analysis:',
        f'<Subject 1> (appears in {", ".join(with_s) or "[Shot 1]"}): reference - face, head, hair, skin tone and facial hair come from '
        '<Picture 1>; outfit, props and body motion come from <Video 1>.',
        '<Video 1> (source video editing): fully_preserved - every cut, framing, camera move, set, lighting, villain and timing is kept '
        'while the main man is replaced.',
        '<Audio 1>: fully_preserved - the original audio is kept in sync.', '',
        'detailed_description:', *[d for d in desc if d], '',
        'overall_soundscape:', b.get('soundscape', 'Keep the audio of <Video 1>.')])


def h2h_wan_prompt(b: dict) -> str:
    return (f"{b['subject']}, wearing {b['wardrobe'].replace('holding', 'and holding')}, the main man in a fast-cut comedic horror action "
            "skit inside a wood-panelled house with masked horror villains, same poses, motion, cuts and camera framing as the control "
            "video, photorealistic handheld smartphone footage, natural daylight")


def h2h_runs(b: dict, seeds=H2H_SEEDS, baseline: bool = False, wangp: bool = True, rungs: list[str] | None = None) -> list[dict]:
    secs = float(b.get('seconds', 10.0))
    consent = bool(b.get('likeness_consent'))
    base = dict(mode='character_swap', source_seconds=secs, source_size=(1280, 720), keep_audio=True, max_resolution=1280,
                motion_ref_short_edge=544, likeness_consent=consent)
    ref = [Ref(path=str(CLIPS / 'h2h_ref.png'), role='subject')]
    h3p, wanp = h2h_h3_prompt(b, secs), h2h_wan_prompt(b)
    cuts16 = [int(round(c['t'] * 16)) for c in b.get('cuts') or []]
    R = []
    table = list(H2H_RUNGS) + ([H2H_BASELINE] if baseline else [])
    for rung, label, ov, est, cold in table:
        if rungs and rung not in rungs:
            continue
        for i, seed in enumerate(seeds):
            spec = ShotSpec(source_video=str(CLIPS / 'h2h_motion.mp4'), refs=ref, prompt='', seed=seed, **base)
            R.append({'id': f'h2h_{rung}_s{seed}', 'rung': rung, 'seed': seed, 'question': f'{rung}: {label}',
                      'spec': spec.model_copy(update={'engine': 'genjutsu_h3_ref2va', 'tier': 'final'}), 'tier': 'final',
                      'overrides': {'vae': 'standard', 'megapixels': H3_720P_MP, 'prompt_override': h3p, **ov},
                      'est_s': int(est * H2H_SCALE) + (cold if i == 0 else 0), 'protect': rung in ('H3-P8', 'H3-T4')})
    rung, label, ov, est, cold = H2H_WAN
    if not rungs or rung in rungs:
        for i, seed in enumerate(seeds):
            spec = ShotSpec(source_video=str(CLIPS / 'h2h_motion.mp4'), refs=ref, prompt=wanp, seed=seed, **base)
            R.append({'id': f'h2h_{rung}_s{seed}', 'rung': rung, 'seed': seed, 'question': f'{rung}: {label}',
                      'spec': spec.model_copy(update={'engine': 'genjutsu_wan_vace_lora', 'tier': 'final'}), 'tier': 'final',
                      'overrides': {'prompt_override': wanp, 'cuts': cuts16, **ov}, 'est_s': est + (cold if i == 0 else 0),
                      'protect': True})
    rung, label, est, cold = H2H_WANGP
    if wangp and (not rungs or rung in rungs):
        for i, seed in enumerate(seeds):
            R.append({'id': f'h2h_{rung}_s{seed}', 'rung': rung, 'seed': seed, 'question': f'{rung}: {label}', 'engine': 'wangp',
                      'tier': 'final', 'uploads': ['h2h_ref.png', 'h2h_motion_544.mp4'], 'est_s': est + (cold if i == 0 else 0),
                      'protect': False, 'wangp': {'profile': '4', 'attention': 'sage2', 'settings': wangp_settings(h3p, seed, secs)}})
    return R


def wangp_settings(prompt: str, seed: int, seconds: float, comfy_input: str = '/opt/jutsu/ComfyUI/input') -> dict:
    """WanGP settings for H3 Ref2VA pruned PDD (keys from Wan2GP@6479db3 defaults/minimax_h3_ref2va_pruned_pdd.json and
    models/minimax_h3/minimax_h3_handler.py): `I` = use reference images, `V-U` = one reference video (motion), `K` = its
    soundtrack as an audio reference (the ComfyUI graph passes the same audio as ref_video_audio_0)."""
    from app.genjutsu.recipes import h3_length
    # Oct 7 bug: WanGP's default multi_prompts_gen_type 'PG' turns every prompt LINE into its own generation request, so the
    # multi-line Context-IR prompt became 5 requests and the single-task submit was rejected. One line + 'FG' = one request.
    prompt = ' '.join(x.strip() for x in prompt.splitlines() if x.strip())
    return {'model_type': WANGP_MODEL, 'prompt': prompt, 'multi_prompts_gen_type': 'FG', 'image_refs': [f'{comfy_input}/h2h_ref.png'],
            'video_guide': f'{comfy_input}/h2h_motion_544.mp4', 'video_prompt_type': 'IV-U', 'audio_prompt_type': 'K',
            'resolution': '1280x720', 'video_length': h3_length(seconds), 'force_fps': 24, 'num_inference_steps': 8,
            'seed': int(seed), 'remove_background_images_ref': 0}       # WanGP default 1 would cut the photo out; ComfyUI does not


def est_seconds(run: dict, ecfg: dict) -> int:
    if run.get('est_s'):
        return int(run['est_s'])
    base = float((ecfg.get('est_gpu_seconds') or {}).get(run['tier'], 300)) * RUN_SECONDS / 5.0
    ov = run['overrides']
    if run['tier'] == 'final' and ov.get('accel') in ACCEL_FACTOR:
        base *= ACCEL_FACTOR[ov['accel']]
    if ov.get('vae') == 'x2':
        base += 40
    if ov.get('veda'):
        base *= 0.7
    return int(base)


def build_plan(with_veda=False, with_x2=False, option='B', gpu='5090', write=True, ladder=False, h2h: dict | None = None) -> dict:
    prov_cfg = config.load()['providers']['runpod_pod']
    if ladder and option == 'B':
        option = 'L'
    if h2h is not None and option == 'B':
        option = 'H'
    g = next(x for x in prov_cfg['gpus'] if x['name'] == gpu)
    rate = float(g.get(OPTIONS[option]['rate_key']) or g['usd_per_hr'])
    rate += OPTIONS[option]['container_disk_gb'] * DISK_USD_PER_GB_HR      # what the watchdog charges while running
    out_runs, engines_seen, problems = [], set(), []
    src_runs = h2h['runs'] if h2h is not None else ladder_runs() if ladder else runs(with_veda, with_x2)
    fast = ladder or h2h is not None
    for r in src_runs:
        if r.get('engine') == 'wangp':                         # separate app: no ComfyUI graph, settings JSON for worker/wangp_driver.py
            missing = [u for u in r['uploads'] if not (CLIPS / u).exists()]
            if missing:
                problems.append({'run': r['id'], 'graph_problems': None, 'missing_inputs': missing})
            out_runs.append({'id': r['id'], 'rung': r['rung'], 'seed': r['seed'], 'question': r['question'], 'protect': r['protect'],
                             'engine': 'wangp', 'recipe': WANGP_MODEL, 'tier': 'final', 'recipe_kwargs': None, 'uploads': r['uploads'],
                             'outputs': None, 'params': {'unet': WANGP_MODEL, 'accel': 'pdd8', 'attention': r['wangp']['attention'],
                                                         'profile': r['wangp']['profile']},
                             'est_s': r['est_s'], 'est_usd': round(r['est_s'] / 3600 * rate, 3), 'models_gb': 0, 'graph': None,
                             'wangp': r['wangp'], 'notes': ['WanGP downloads its own H3 files on the first (cold) run']})
            continue
        p = planner.plan_shot(r['spec'], provider='runpod_pod', gpu=gpu, recipe_overrides=r['overrides'])
        ecfg = config.load()['engines'][p['engine']]
        est = est_seconds(r, ecfg) + (0 if fast or p['engine'] in engines_seen else ENGINE_LOAD_EST_S)
        engines_seen.add(p['engine'])
        missing = [u for u in p.get('uploads', []) if not (CLIPS / u).exists()]
        if p.get('problems') or missing:
            problems.append({'run': r['id'], 'graph_problems': p.get('problems'), 'missing_inputs': missing})
        if h2h is not None and (p.get('consent') or {}).get('required') and not (p.get('consent') or {}).get('ok'):
            problems.append({'run': r['id'], 'consent': p.get('consent')})
        out_runs.append({'id': r['id'], 'rung': r.get('rung'), 'seed': r.get('seed'), 'question': r['question'], 'protect': r['protect'], 'engine': p['engine'], 'recipe': p.get('recipe'), 'tier': r['tier'],
                         'recipe_kwargs': p.get('recipe_kwargs'), 'uploads': p.get('uploads', []), 'outputs': p.get('outputs'),
                         'params': p.get('params'), 'est_s': est, 'est_usd': round(est / 3600 * rate, 3),
                         'models_gb': round((p.get('models') or {}).get('bytes', 0) / 1e9, 1), 'graph': p.get('graph'),
                         'notes': p.get('notes', [])})
    if h2h is not None:
        for r in out_runs:
            r['needs_stage2'] = r['engine'] == 'genjutsu_wan_vace_lora'
    setup_s = (H2H_SETUP_S + H2H_LINGER_S if h2h is not None else LADDER_SETUP_S + LADDER_LINGER_S if ladder
               else sum(SETUP_EST_S.values()))
    cap = float(prov_cfg.get('hard_cap_usd', 5.0)) * float(prov_cfg.get('watchdog_safety', 0.92))
    trimmed = []
    while (setup_s + sum(r['est_s'] * 1.3 for r in out_runs)) / 3600 * rate > cap:   # 30 % margin; never trim the head-to-head
        cut = next((r for r in reversed(out_runs) if not r['protect']), None)
        if not cut:
            break
        out_runs.remove(cut); trimmed.append(cut['id'])
    run_s = sum(r['est_s'] for r in out_runs)
    hours = (setup_s + run_s) / 3600
    o = OPTIONS[option]
    vol_usd = o['volume_gb'] * float(prov_cfg.get('volume_usd_per_gb_month', 0.07)) / 730 * max(hours, 1)
    m = json.loads((ROOT / 'config' / 'models.manifest.json').read_text())
    groups = ['h2h'] if h2h is not None else ['ladder'] if ladder else ['phase1'] + (['phase1_veda'] if with_veda else [])
    plan = {'created': time.strftime('%Y-%m-%d %H:%M %Z'), 'option': option, 'option_label': o['label'], 'gpu': g['id'],
            'usd_per_hr': rate, 'hard_cap_usd': float(prov_cfg.get('hard_cap_usd', 5.0)), 'model_groups': groups,
            'models_gb': round(sum(m['group_bytes'][x] for x in groups) / 1e9, 1), 'setup_est_s': SETUP_EST_S,
            'runs_est_s': run_s, 'total_est_hours': round(hours, 2),
            'est_usd': {'pod': round(hours * rate, 2), 'volume_while_running': round(vol_usd, 2),
                        'total': round(hours * rate + vol_usd, 2),
                        'volume_if_kept_30d': round(o['volume_gb'] * float(prov_cfg.get('volume_usd_per_gb_month', 0.07)), 2)},
            'watchdog_stop_after_s': int(float(prov_cfg.get('hard_cap_usd', 5.0)) * float(prov_cfg.get('watchdog_safety', 0.92)) / rate * 3600),
            'trimmed_for_budget': trimmed, 'problems': problems, 'ladder': ladder or h2h is not None, 'h2h': h2h is not None,
            'wangp': any(r.get('engine') == 'wangp' for r in out_runs),
            'h2h_window': {k: h2h.get(k) for k in ('motion', 'ref', 'start', 'seconds', 'cuts')} if h2h is not None else None,
            'stage1_bundles': H2H_STAGE1 if h2h is not None else LADDER_STAGE1 if ladder else None,
            'compare_ids': [x for x in (f'h2h_{k}_s{H2H_SEEDS[-1]}' for k in ('H3-P8', 'H3-T4', 'WAN-VACE', 'WANGP-P4'))
                            if any(r['id'] == x for r in out_runs)] if h2h is not None else None,
            'compare_source': str(CLIPS / 'h2h_motion.mp4') if h2h is not None else None,
            'box_uploads': [{'id': x['id'], 'file': x['file'], 'local_path': x['local_path'], 'sha256': x['sha256'],
                             'size_bytes': x['size_bytes'], 'dest': 'loras'} for x in m['models']
                            if x.get('source') == 'box_upload' and set(x['bundles']) & set().union(*(m['groups'][g] for g in groups))],
            'runs': out_runs}
    if write:
        OUT.mkdir(parents=True, exist_ok=True)
        for r in out_runs:
            d = OUT / r['id']; d.mkdir(exist_ok=True)
            (d / ('wangp_settings.json' if r['engine'] == 'wangp' else 'prompt_api.json')).write_text(
                json.dumps(r['wangp']['settings'] if r['engine'] == 'wangp' else r['graph'], indent=1))
        (OUT / 'plan.json').write_text(json.dumps({**plan, 'runs': [{k: v for k, v in r.items() if k != 'graph'} for r in out_runs]},
                                                  indent=1, default=str))
    return plan


REFINE_OUT = ROOT / 'data' / 'bakeoff' / 'refine'
REFINE_SETUP_S = 600                                      # Oct 7: ComfyUI ready after 523 s with 66 GB stage-1; refine pulls ~58 GB
REFINE_LINGER_S = 120
REFINE_STAGE1 = ['h3', 'h3_fast']
REFINE_COMPARE = ['fixF_depthhead_base20', 'fixG_depthbody_base20', 'fixE_base20_s7']


INPAINT_OUT = ROOT / 'data' / 'bakeoff' / 'inpaint'
BETTER_OUT = ROOT / 'data' / 'bakeoff' / 'better'
FP_OUT = pathlib.Path(os.environ['FP_RUN_DIR']) if os.environ.get('FP_RUN_DIR') else ROOT / 'data' / 'bakeoff' / 'fullperson' / 'run'   # FP_RUN_DIR/FP_GROUP: Oct 13 long-clip run; Oct 13 Character decisive test (scripts/fullperson_pack.py)


def build_refine_plan(cap: float, gpu: str = '5090', write: bool = True, cloud: str = 'COMMUNITY', inpaint: bool = False,
                      better: bool = False, fullperson: bool = False) -> dict:
    """Oct 8 refine round: prebuilt graphs from scripts/h2h_identity_fix.py refine-build (data/bakeoff/refine/<id>/), H3-only
    models, community 5090 (option R). Runs are dropped from the end of the list until the quote (runs x1.3) fits `cap`."""
    import h2h_identity_fix as fx
    prov_cfg = config.load()['providers']['runpod_pod']
    g = next(x for x in prov_cfg['gpus'] if x['name'] == gpu)
    o = dict(OPTIONS['V' if (better or fullperson) else 'R'])
    if gpu != '5090' or cloud != 'COMMUNITY':                  # Oct 8: 5090 out of stock -> PRO 6000 (community, then secure)
        o.update(cloud=cloud, rate_key='community_usd_per_hr' if cloud == 'COMMUNITY' else 'usd_per_hr',
                 label=o['label'].replace('Community Cloud 5090', f"{cloud.title()} Cloud {g['id']}"))
    rate = float(g.get(o['rate_key']) or g['usd_per_hr']) + o['container_disk_gb'] * DISK_USD_PER_GB_HR
    setup_s = REFINE_SETUP_S + REFINE_LINGER_S
    if fullperson:                                        # Oct 13 eve: Character FULL_PERSON H3 vs Viggle (K first, H3/VIG via extra_runs)
        runs = json.loads((FP_OUT / 'fp_runs.json').read_text()); dropped = []
    elif better:                                          # Oct 13: better-build 7-way (scripts/better_build.py build)
        runs = json.loads((BETTER_OUT / 'better_runs.json').read_text()); dropped = []
    elif inpaint:                                           # Oct 8 eve: H3 masked inpaint + per-shot routing (scripts/h3_inpaint.py)
        import h3_inpaint as hi
        runs, dropped = hi.inpaint_runs(json.loads((INPAINT_OUT / 'inpaint_meta.json').read_text()), cap, rate, setup_s, gpu)
    else:
        runs, dropped = fx.refine_runs(cap, rate, setup_s)
    problems = []
    for r in runs:
        if r.get('engine') == 'wangp':                    # Oct 20 piano test: WanGP rung in the fullperson path (no graph)
            r['est_usd'] = round(r['est_s'] / 3600 * rate, 3); continue
        gp = pathlib.Path(r.pop('graph_path'))
        if not gp.exists():
            problems.append({'run': r['id'], 'missing_graph': str(gp)}); r['graph'] = None; continue
        r['graph'] = json.loads(gp.read_text())
        miss = [u for u in r['uploads'] if not (CLIPS / u).exists()]
        if miss:
            problems.append({'run': r['id'], 'missing_inputs': miss})
        if not (better or fullperson) and any(n.get('class_type') == 'MiniMaxH3PDDAccApply' for n in r['graph'].values()) != (r['params']['steps'] in (4, 8)):
            problems.append({'run': r['id'], 'graph_problems': 'PDD node presence does not match the step count'})
        r['est_usd'] = round(r['est_s'] / 3600 * rate, 3)
    run_s = sum(r['est_s'] for r in runs)
    hours = (setup_s + run_s) / 3600
    m = json.loads((ROOT / 'config' / 'models.manifest.json').read_text())
    grp = os.environ.get('FP_GROUP', 'fullperson') if fullperson else 'better' if better else 'refine'
    w = json.loads((H2H_OUT / 'window.json').read_text()) if (H2H_OUT / 'window.json').exists() else {}
    plan = {'created': time.strftime('%Y-%m-%d %H:%M %Z'), 'option': 'R', 'option_label': o['label'], 'gpu': g['id'],
            'gpu_name': gpu, 'cloud': o['cloud'], 'usd_per_hr': rate, 'hard_cap_usd': float(prov_cfg.get('hard_cap_usd', 5.0)), 'model_groups': [grp],
            'models_gb': round(m['group_bytes'][grp] / 1e9, 1), 'setup_est_s': {'boot_install_models': REFINE_SETUP_S},
            'runs_est_s': run_s, 'total_est_hours': round(hours, 2),
            'est_usd': {'pod': round(hours * rate, 2), 'volume_while_running': 0.0, 'total': round(hours * rate, 2),
                        'pessimistic': round((setup_s + run_s * 1.4) / 3600 * rate, 2), 'volume_if_kept_30d': 0.0},
            'watchdog_stop_after_s': int(cap * float(prov_cfg.get('watchdog_safety', 0.92)) / rate * 3600),
            'trimmed_for_budget': dropped, 'problems': problems, 'ladder': False, 'h2h': False, 'refine': True, 'wangp': any(r.get('engine') == 'wangp' for r in runs),
            'inpaint': inpaint,
            'stage1_bundles': [os.environ.get('FP_GROUP', 'fullperson')] if fullperson else REFINE_STAGE1 + (['better_lite'] if better else []), 'better': better,
            'fullperson': fullperson, 'option_key': 'V' if (better or fullperson) else 'R',
            'compare_ids': [] if (inpaint or better or fullperson) else [x for x in REFINE_COMPARE if any(r['id'] == x for r in runs)],
            'compare_source': str(CLIPS / 'h2h_motion.mp4'),
            'audio_source': None if (better or fullperson) else {'path': w.get('motion') or str(ROOT / 'data' / 'bakeoff' / 'inputs-oct7b' / 'motion_src.mp4'),
                             'start': float(w.get('start') or 0.0), 'seconds': float(w.get('seconds') or 10.0) + 0.08},
            'box_uploads': [], 'runs': runs}
    if write:
        dst = FP_OUT if fullperson else BETTER_OUT if better else INPAINT_OUT if inpaint else REFINE_OUT
        dst.mkdir(parents=True, exist_ok=True)
        (dst / 'plan.json').write_text(json.dumps({**plan, 'runs': [{k: v for k, v in r.items() if k != 'graph'} for r in runs]},
                                                         indent=1, default=str))
    return plan


def print_refine_plan(p: dict, cap: float):
    print(f"Oct 8 refine round QUOTE  [{p['option']}] {p['option_label']}")
    print(f"GPU {p['gpu']} @ ${p['usd_per_hr']:.3f}/hr incl. disk   models {p['models_gb']} GB ({'+'.join(p['model_groups'])})   cap ${cap:.2f}")
    print(f"{'run':24} {'steps':>5} {'est s':>6} {'est $':>6}  what")
    for r in p['runs']:
        print(f"{r['id']:24} {r['params']['steps']:>5} {r['est_s']:>6} {r['est_usd']:>6.3f}  {r['question'].split(': ', 1)[-1]}")
    print(f"setup {REFINE_SETUP_S}s + linger {REFINE_LINGER_S}s + runs {p['runs_est_s']}s = {p['total_est_hours']} h -> est "
          f"${p['est_usd']['total']}  (pessimistic, runs x1.4: ${p['est_usd']['pessimistic']})")
    print(f"watchdog stop after {p['watchdog_stop_after_s']}s; dropped for the cap: {p['trimmed_for_budget'] or 'nothing'}; "
          f"problems: {p['problems'] or 'none'}")


def ladder_quote(p: dict) -> list[dict]:
    rows = {}
    for r in p['runs']:
        x = rows.setdefault(r['rung'], {'rung': r['rung'], 'what': r['question'].split(': ', 1)[-1], 'seeds': 0, 'est_s': 0,
                                        'unet': (r.get('params') or {}).get('unet'), 'accel': (r.get('params') or {}).get('accel'),
                                        'attention': (r.get('params') or {}).get('attention')})
        x['seeds'] += 1; x['est_s'] += r['est_s']
    for x in rows.values():
        x['est_min'] = round(x['est_s'] / 60, 1); x['est_usd'] = round(x['est_s'] / 3600 * p['usd_per_hr'], 3)
    return list(rows.values())


def print_plan(p: dict):
    if p.get('ladder'):
        title = '10 s head-to-head + H3 ladder' if p.get('h2h') else 'H3 fast-path ladder'
        print(f"{title} QUOTE (not launched)  [{p['option']}] {p['option_label']}")
        if p.get('h2h_window'):
            w = p['h2h_window']
            print(f"window {w['start']:g}-{float(w['start']) + float(w['seconds']):g} s of {pathlib.Path(w['motion']).name}, ref "
                  f"{pathlib.Path(w['ref']).name}, {len(w['cuts'] or [])} cuts")
        print(f"GPU {p['gpu']} @ ${p['usd_per_hr']:.3f}/hr incl. disk   models {p['models_gb']} GB   seeds/rung {len(LADDER_SEEDS)}")
        print(f"{'rung':5} {'min':>6} {'$':>6}  {'trunk':45} {'accel':7} {'attn':8} what")
        for x in ladder_quote(p):
            print(f"{x['rung']:5} {x['est_min']:>6} {x['est_usd']:>6.3f}  {str(x['unet']):45} {str(x['accel']):7} {str(x['attention']):8} {x['what']}")
        su, li = (H2H_SETUP_S, H2H_LINGER_S) if p.get('h2h') else (LADDER_SETUP_S, LADDER_LINGER_S)
        print(f"setup {su}s + linger {li}s + runs {p['runs_est_s']}s = {p['total_est_hours']} h -> est ${p['est_usd']['total']}")
        pess = (su + li + p['runs_est_s'] * 1.4) / 3600 * p['usd_per_hr']
        print(f"pessimistic (runs x1.4) ${pess:.2f}; cold first seed per rung is timed but excluded from the warm ranking")
        if not p.get('wangp'):
            w = WANGP_ADDON; ws = w['setup_s'] + w['est_s']
            print(f"optional {w['rung']}: {w['what']}: +{ws / 60:.0f} min (install/pull {w['setup_s']}s + 2 seeds) "
                  f"= +${ws / 3600 * p['usd_per_hr']:.2f}  (wired in --h2h)")
        print(f"problems: {p['problems'] or 'none'}")
        return
    print(f"Phase 1 bake-off plan  [{p['option']}] {p['option_label']}")
    print(f"GPU {p['gpu']} @ ${p['usd_per_hr']}/hr   models {p['models_gb']} GB ({'+'.join(p['model_groups'])})   hard cap ${p['hard_cap_usd']}")
    print(f"{'run':30} {'engine':22} {'tier':6} {'est s':>6} {'est $':>6}  question")
    for r in p['runs']:
        print(f"{r['id']:30} {r['engine']:22} {r['tier']:6} {r['est_s']:>6} {r['est_usd']:>6.3f}  {r['question']}")
    s = p['setup_est_s']
    print(f"setup (boot+apt {s['boot_and_apt']}s, install {s['install']}s, model pull {s['models']}s) + runs {p['runs_est_s']}s "
          f"= {p['total_est_hours']} h  ->  est ${p['est_usd']['total']} (pod ${p['est_usd']['pod']}, volume ${p['est_usd']['volume_while_running']})")
    if p['est_usd']['volume_if_kept_30d']:
        print(f"  network volume if KEPT for 30 days: ${p['est_usd']['volume_if_kept_30d']} (outside the pod cap; the runner never creates or deletes volumes; delete it in the console after review)")
    print(f"watchdog hard stop after {p['watchdog_stop_after_s']}s at this rate; problems: {p['problems'] or 'none'}")
    print(f"trimmed for budget: {p['trimmed_for_budget'] or 'nothing'}; box uploads: {[b['file'] for b in p['box_uploads']] or 'none'}")


# ----------------------------------------------------------------------------------------------- live (gated)
G8_IDS = ['g8_h3_pdd8_720p', 'g8_wan_vace_pack2500_720p', 'g8_h3_hyperflow8_720p']
REVIEW = ROOT / 'data' / 'bakeoff' / 'out'                # the owner-facing outputs: side-by-side first, then every render
STATUS_PORT = 8001
STALL_S = int(os.environ.get('JUTSU_STALL_S', '600'))     # stall = no ComfyUI progress event for 10 min (the owner, Oct 17)
DISK_USD_PER_GB_HR = 0.10 / 730                          # container disk while running (added to the watchdog rate)


def estimate_pod_spent(pod: dict, rate: float, clock=time.time) -> float:
    """Wall-clock spend since lastStartedAt/createdAt at `rate` (box-side estimate for --attach-pod)."""
    raw = pod.get('lastStartedAt') or pod.get('createdAt') or ''
    # RunPod REST: "2026-10-08 21:09:09.97 +0000 UTC"
    s = str(raw).replace(' UTC', '').strip()
    t0 = None
    for fmt in ('%Y-%m-%d %H:%M:%S.%f %z', '%Y-%m-%d %H:%M:%S %z', '%Y-%m-%dT%H:%M:%S.%fZ', '%Y-%m-%dT%H:%M:%SZ'):
        try:
            from datetime import datetime
            t0 = datetime.strptime(s.replace('Z', '+0000') if fmt.endswith('Z') else s, fmt).timestamp()
            break
        except Exception:
            continue
    if t0 is None:
        return 0.0
    return max(0.0, (clock() - t0) / 3600.0 * float(rate))



def cloud_try_order(cloud: str, has_volume: bool) -> list[str]:
    """COMMUNITY_FIRST -> ['COMMUNITY','SECURE'] (cheaper first, Secure fallback). Network volumes are Secure-only,
    so with a volume the order collapses to ['SECURE']."""
    c = (cloud or 'SECURE').upper()
    order = ['COMMUNITY', 'SECURE'] if c == 'COMMUNITY_FIRST' else [c]
    return ['SECURE'] if has_volume else order


def cloud_for_attempt(order: list[str], i: int, attempts: int) -> str:
    """First half of the create attempts on order[0], rest on order[-1] (at least 1 each when 2 clouds)."""
    if len(order) == 1:
        return order[0]
    return order[0] if i < max(1, attempts // 2) else order[-1]


def boot_payload(groups: list[str]) -> str:
    """worker scripts + a manifest trimmed to the needed groups, tar.gz + base64 -> env JUTSU_BOOT_B64."""
    m = json.loads((ROOT / 'config' / 'models.manifest.json').read_text())
    want = set().union(*(m['groups'][g] for g in groups))
    m['models'] = [x for x in m['models'] if set(x['bundles']) & want]
    m['groups'] = {g: m['groups'][g] for g in groups}
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz') as t:
        for f in ('install.sh', 'custom_nodes.txt', 'start.sh', 'extra_model_paths.yaml', 'bootstrap_volume.py', 'assemble_uploads.sh',
                  'phase1_boot.sh', 'unchunk_models.sh', 'status_server.py', 'wangp_install.sh', 'wangp_driver.py', 'patches/latentsync_audio_patch.py',
                  'patches/ls_vae_patch.py', 'ensure_a08.sh', 'post_a08.py'):
            t.add(ROOT / 'worker' / f, arcname=f'worker/{f}')
        node = ROOT / 'worker' / 'custom_nodes' / 'h3_exact_lora'
        for p in sorted(node.rglob('*')):
            if not p.is_file() or p.suffix == '.pyc' or '__pycache__' in p.parts:
                continue
            t.add(p, arcname=f"worker/custom_nodes/h3_exact_lora/{p.relative_to(node).as_posix()}")
        data = json.dumps(m, separators=(',', ':')).encode()
        ti = tarfile.TarInfo('config/models.manifest.json'); ti.size = len(data); t.addfile(ti, io.BytesIO(data))
    return base64.b64encode(buf.getvalue()).decode()


def boot_cmd(dead_man_s: int) -> list[str]:
    """Pod start command: dead-man switch first (removes the pod even if the box dies), then worker/phase1_boot.sh."""
    script = (
        # (Oct 15: a pre-boot nvidia-smi CUDA guard stalled two pods before the status server started; removed. The REST
        #  allowedCudaVersions=['13.0'] filter is the CUDA pin.)
        # Dead-man: deadline lives in /tmp/jutsu_deadline (epoch s). Starts at the conservative worst-rate value; the box
        # moves it via pod_cmds once it has read the pod's ACTUAL costPerHr (Oct 16: the fixed $2.49 value removed a
        # $1.69 Community pod at 19/20 steps).
        f'date +%s > /tmp/jutsu_t0; echo $(( $(date +%s) + {int(dead_man_s)} )) > /tmp/jutsu_deadline; '
        '( while [ "$(date +%s)" -lt "$(cat /tmp/jutsu_deadline 2>/dev/null || echo 0)" ]; do sleep 15; done; '
        'echo "[boot] dead-man switch"; runpodctl remove pod "$RUNPOD_POD_ID" || '
        'curl -s -X DELETE -H "Authorization: Bearer $RUNPOD_API_KEY" "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID" ) & '
        'mkdir -p /opt/jutsu-src && echo "$JUTSU_BOOT_B64" | base64 -d | tar xz -C /opt/jutsu-src && cd /opt/jutsu-src && '
        'exec bash worker/phase1_boot.sh')
    return ['bash', '-c', script]


UA = 'genie-jutsu-runner/1.0'                             # RunPod's Cloudflare proxy 403s (error 1010) the default Python-urllib UA


def _urllib(method, url, body=None, headers=None, timeout=60):
    req = urllib.request.Request(url, data=body, method=method, headers={'User-Agent': UA, **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read()


class Comfy:
    """Minimal ComfyUI HTTP client (urllib; `http` injectable for tests)."""

    def __init__(self, base: str, http=None):
        self.base = base.rstrip('/')
        self.http = http or _urllib

    def get_json(self, path):
        st, b = self.http('GET', self.base + path)
        return json.loads(b)

    def ready(self) -> bool:
        try:
            return 'system' in self.get_json('/system_stats')
        except Exception:
            return False

    def upload(self, path: pathlib.Path, subfolder: str = '', data: bytes | None = None, name: str | None = None) -> str:
        bnd = uuid.uuid4().hex
        extra = f'\r\n--{bnd}\r\nContent-Disposition: form-data; name="subfolder"\r\n\r\n{subfolder}' if subfolder else ''
        body = (f'--{bnd}\r\nContent-Disposition: form-data; name="image"; filename="{name or path.name}"\r\n'
                f'Content-Type: application/octet-stream\r\n\r\n').encode() + (data if data is not None else path.read_bytes()) + \
               f'\r\n--{bnd}\r\nContent-Disposition: form-data; name="overwrite"\r\n\r\ntrue{extra}\r\n--{bnd}--\r\n'.encode()
        st, b = self.http('POST', self.base + '/upload/image', body, {'Content-Type': f'multipart/form-data; boundary={bnd}'}, 300)
        return json.loads(b)['name']

    def queue(self, graph: dict, client_id: str = 'jutsu-phase1') -> str:
        st, b = self.http('POST', self.base + '/prompt', json.dumps({'prompt': graph, 'client_id': client_id}).encode(),
                          {'Content-Type': 'application/json'})
        return json.loads(b)['prompt_id']

    def free(self):
        try:
            self.http('POST', self.base + '/free', json.dumps({'unload_models': True, 'free_memory': True}).encode(),
                      {'Content-Type': 'application/json'})
        except Exception:
            pass

    def interrupt(self):
        try:
            self.http('POST', self.base + '/interrupt', b'{}', {'Content-Type': 'application/json'})
        except Exception:
            pass

    def free_and_wait(self, frac: float = 0.85, timeout_s: float = 90, sleep=time.sleep) -> float | None:
        """/free, then wait until >= frac of VRAM is free (Phase 1: FlashVSR OOMed with 26.5 GB still held by the H3 run)."""
        self.free(); t0 = time.time(); free = None
        while time.time() - t0 < timeout_s:
            try:
                tot, free = self.vram()
                if free >= frac * tot:
                    break
            except Exception:
                break
            sleep(3); self.free()
        return free

    def vram(self) -> tuple[float, float]:
        d = self.get_json('/system_stats')['devices'][0]
        return d['vram_total'] / 1e9, d['vram_free'] / 1e9

    def download(self, f: dict, dest: pathlib.Path):
        from urllib.parse import urlencode
        st, b = self.http('GET', self.base + '/view?' + urlencode({k: f.get(k, '') for k in ('filename', 'subfolder', 'type')}), None, None, 600)
        dest.write_bytes(b)


class PodStatus:
    """Client for worker/status_server.py (port 8001): boot stages, model bytes, logs, GPU samples, fix jobs."""

    def __init__(self, base: str, http=None, nonce: str | None = None):
        self.base, self.http, self._nonce = base.rstrip('/'), http or _urllib, nonce

    last_error: str | None = None

    def status(self) -> dict | None:
        try:
            d = json.loads(self.http('GET', self.base + '/status', None, None, 20)[1]); self.last_error = None; return d
        except Exception as e:
            self.last_error = f'{type(e).__name__}: {e}'[:200]
            return None

    def gpu(self, since: float) -> dict:
        try:
            return json.loads(self.http('GET', self.base + f'/gpu?since={since}', None, None, 20)[1])
        except Exception:
            return {}

    def log(self, name: str, frm: int = 0) -> str:
        try:
            return self.http('GET', self.base + f'/log/{name}?from={int(frm)}', None, None, 60)[1].decode('utf-8', 'replace')
        except Exception:
            return ''

    def exec(self, script: str) -> str | None:
        if not self._nonce:
            return None
        st, b = self.http('POST', self.base + '/exec', script.encode(), {'X-Jutsu-Nonce': self._nonce, 'Content-Type': 'text/plain'}, 30)
        return json.loads(b)['job']

    def job(self, jid: str) -> dict:
        return json.loads(self.http('GET', self.base + f'/job/{jid}', None, None, 30)[1])


class Progress:
    """Heartbeat only on REAL progress (stage change, model bytes growing, log growth, GPU busy), so the
    watchdog's 15-min idle rule is a true stall detector rather than 'the box is still polling'."""

    def __init__(self, pod: PodStatus, session, log=print):
        self.pod, self.s, self.log = pod, session, log
        self.sig, self.last, self.stages_seen = None, {}, 0

    def tick(self, extra=None) -> dict:
        st = self.pod.status() or {}
        g = st.get('gpu') or {}
        busy = (g.get('util') or 0) > 10
        sig = (len(st.get('stages', [])), (st.get('models_bytes') or 0) // (64 << 20), sum((st.get('log_bytes') or {}).values()) // 512, extra)
        if st and (sig != self.sig or busy):
            self.s.heartbeat()
        self.sig = sig
        for line in st.get('stages', [])[self.stages_seen:]:
            self.log(f'[pod] {line.split(" ", 1)[-1]}  (${self.s.spent():.2f})')
        self.stages_seen = max(self.stages_seen, len(st.get('stages', [])))
        if st:
            self.last = st
        return st

    def has(self, stage: str) -> bool:
        return any(x.split(' ', 2)[1] == stage for x in self.last.get('stages', []) if ' ' in x)


TQDM = re.compile(r'(\d+)/(\d+) \[(?:[\d:]+)<(?:[\d:?]+),\s*([\d.]+)(s/it|it/s)\]')


def tqdm_rates(text: str) -> list[dict]:
    """Final reading of every ComfyUI sampler progress bar in a log chunk -> [{'steps', 's_per_step'}]. ComfyUI prints each
    closing line twice (same total + elapsed); Phase 1 counted both, doubling sampling_s (s/step itself was right)."""
    return [{'steps': b['steps'], 's_per_step': b['s_per_step']} for b in dedup_bars(text)]


PART_BYTES = 90 * 1024 * 1024                              # < ComfyUI's 100 MB default upload cap


def push_box_uploads(comfy: Comfy, uploads: list[dict], session, log=print, sleep=time.sleep, timeout_s: int = 900) -> list[str]:
    """Character-pack LoRAs that only exist on the box: upload in 90 MB parts + a parts.json; the pod's
    worker/assemble_uploads.sh concatenates, sha256-checks and moves the file into /models/loras, then writes <name>.ok."""
    done = []
    for u in uploads:
        src = pathlib.Path(u['local_path']); n = 0
        with open(src, 'rb') as f:
            while chunk := f.read(PART_BYTES):
                for attempt in range(3):
                    try:
                        comfy.upload(src, 'jutsu_parts', chunk, f"{u['file']}.part{n:03d}"); break
                    except Exception:
                        if attempt == 2:
                            raise
                        sleep(5)
                n += 1; session.heartbeat()
        meta = json.dumps({'name': u['file'], 'parts': n, 'sha256': u['sha256'], 'dest': u.get('dest', 'loras')}).encode()
        comfy.upload(src, 'jutsu_parts', meta, f"{u['file']}.parts.json")
        t0 = time.time()
        while True:
            session.heartbeat()
            try:
                comfy.http('GET', comfy.base + f"/view?filename={u['file']}.ok&subfolder=jutsu_parts&type=input")
                break
            except Exception:
                pass
            if time.time() - t0 > timeout_s or not session.alive():
                raise BudgetExceeded(f"{u['file']} never assembled on the pod (sha256 mismatch or assembler not running)")
            sleep(5)
        log(f"[upload] {u['file']}: {n} parts, sha256 verified on the pod"); done.append(u['file'])
    return done


def compare_side_by_side(results: list[dict], ids: list[str], source: pathlib.Path, out: pathlib.Path) -> str | None:
    """Free, local: source | H3 | Wan ... stacked at 720p height, labelled, source audio."""
    pairs = []
    for i in ids:
        r = next((x for x in reversed(results) if x.get('id') == i and x.get('status') == 'success'), None)
        v = next((o for o in (r or {}).get('outputs', []) if o.endswith('.mp4')), None)
        if v:
            pairs.append((i, v))
    if not pairs:
        return None
    vids = [str(source)] + [v for _, v in pairs]
    labels = ['source'] + [i.replace('g8_', '').replace('h2h_', '').replace('_720p', '') for i, _ in pairs]
    out.parent.mkdir(parents=True, exist_ok=True)
    fl = ';'.join(f"[{i}:v]fps=24,scale=-2:720,setsar=1,drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:"
                  f"text='{labels[i]}':fontsize=30:fontcolor=yellow:box=1:boxcolor=black@0.6:x=12:y=12[v{i}]" for i in range(len(vids)))
    fl += ';' + ''.join(f'[v{i}]' for i in range(len(vids))) + f'hstack=inputs={len(vids)}[out]'
    args = [x for v in vids for x in ('-i', v)] + ['-filter_complex', fl, '-map', '[out]', '-map', '0:a?', '-shortest',
                                                   '-c:v', 'libx264', '-crf', 18, '-preset', 'medium', '-c:a', 'aac', out]
    _ff(*args)
    return str(out)


def downscale_graph(graph: dict, f: float = 0.75) -> dict:
    """OOM retry: same graph at ~f x the linear resolution (H3 megapixels x f^2; Wan/other width/height x f, snapped to 16)."""
    g = json.loads(json.dumps(graph))
    for n in g.values():
        ins = n.get('inputs', {})
        if isinstance(ins.get('megapixels'), (int, float)):
            ins['megapixels'] = round(ins['megapixels'] * f * f, 4)
        for k in ('width', 'height'):
            if isinstance(ins.get(k), int) and ins[k] >= 256:
                ins[k] = max(256, int(ins[k] * f) // 16 * 16)
    return g


def _err_text(rec: dict) -> str:
    for kind, data in (rec.get('status') or {}).get('messages') or []:
        if kind == 'execution_error':
            return f"{data.get('node_type')}: {data.get('exception_type')}: {str(data.get('exception_message'))[:800]}"
    return ''


def _is_oom(text: str) -> bool:
    t = (text or '').lower()
    return 'out of memory' in t or 'outofmemory' in t or 'allocation on device' in t


def graph_sha256(graph: dict) -> str:
    import hashlib
    return hashlib.sha256(json.dumps(graph, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def restart_comfy(pod: PodStatus | None, comfy: Comfy, log=print, sleep=time.sleep, timeout_s: int = 420, session=None) -> bool:
    """Kill ComfyUI through the pod fix channel; phase1_boot.sh's loop starts it again. True once it answers."""
    if not pod:
        return False
    try:
        jid = pod.exec('pkill -f "main.py --listen" || true')
    except Exception as e:
        log(f'[restart] exec failed: {e}'); return False
    if not jid:
        return False
    log(f'[restart] ComfyUI restart requested (job {jid})')
    t0 = time.time(); sleep(8)
    while time.time() - t0 < timeout_s:
        if session:
            session.heartbeat()
        if comfy.ready():
            log(f'[restart] ComfyUI back after {time.time() - t0:.0f}s'); return True
        sleep(5)
    return False


def _wait_ws(comfy, trace, run, session, prog, t0, timeout, guard, ws, poll_hist=60.0, wait_s=10.0):
    """Websocket path: block on the trace, heartbeat on progress, drift check, a /history backstop every poll_hist s."""
    last_hist, key = time.time(), drift_key(run)
    while True:
        if trace.done.wait(wait_s):
            return 'done'
        prog.tick() if prog else session.heartbeat()
        if not session.alive():
            return 'timeout'
        if guard and guard.drifting(key, trace):       # opt-in only (plan drift_guard=true); round 5 killed healthy renders
            return 'drift'
        if (trace.last_event is None and time.time() - t0 > STALL_S) or trace.stalled(STALL_S):
            # Oct 21: ws drops stop progress events -> confirm over HTTP before killing. Running/pending = healthy.
            try:
                h = comfy.get_json(f'/history/{trace.pid}')
                if trace.pid in h and ((h[trace.pid].get('status') or {}).get('completed') is not None or h[trace.pid].get('outputs')):
                    return 'done'
                q = comfy.get_json('/queue'); live = [x[1] for x in (q.get('queue_running') or []) + (q.get('queue_pending') or [])]
                alive = trace.pid in live
            except Exception:
                alive = True                           # can't tell -> don't kill
            if not alive:
                return 'stall'
            if time.time() - t0 > timeout:
                return 'timeout'
            trace.last_event = time.time()             # HTTP says it's still running: count as progress
        if time.time() - last_hist > poll_hist or not ws.connected.is_set():
            last_hist = time.time()
            try:
                h = comfy.get_json(f'/history/{trace.pid}')
                if trace.pid in h and ((h[trace.pid].get('status') or {}).get('completed') is not None or h[trace.pid].get('outputs')):
                    return 'done'
            except Exception:
                pass


def run_one(comfy: Comfy, run: dict, session, rate: float, log=print, poll=5.0, sleep=time.sleep,
            pod: PodStatus | None = None, prog: Progress | None = None, attempt: int = 1, ws: ComfyWS | None = None,
            guard: DriftGuard | None = None, cold: bool | None = None, ws_wait: float = 10.0) -> dict:
    d = OUT / run['id']; d.mkdir(parents=True, exist_ok=True)
    base = {'id': run['id'], 'engine': run.get('engine'), 'attempt': attempt, 'scale': run.get('scale', 1.0),
            'graph_sha256': graph_sha256(run['graph'])}
    for k in ('rung', 'seed'):
        if run.get(k) is not None:
            base[k] = run[k]
    if cold is not None:
        base['cold'] = cold
    try:
        for u in run['uploads']:
            comfy.upload(CLIPS / u)
    except Exception as e:
        return {**base, 'status': 'upload_failed', 'error': f'{type(e).__name__}: {e}'[:500]}
    log_off = ((pod.status() or {}).get('log_bytes') or {}).get('comfy', 0) if pod else 0
    use_ws = bool(ws and ws.connected.is_set())
    trace = RunTrace(None, run['graph'])
    if use_ws:
        ws.trace = trace
    t0 = time.time()
    pid = None
    old = run.get('resume_pid')
    if old:                                                    # resume: the prompt is already queued/running/done on this pod
        try:
            q = comfy.get_json('/queue'); live = [x[1] for x in (q.get('queue_running') or []) + (q.get('queue_pending') or [])]
            if old in live or old in comfy.get_json(f'/history/{old}'):
                pid = old; log(f"[resume] {run['id']}: re-attached to prompt {pid} (not re-queued)")
        except Exception as e:
            log(f'[resume] queue check failed ({e}); re-queueing')
    try:
        if pid is None:
            pid = comfy.queue(run['graph'], ws.client_id if use_ws else 'jutsu-phase1')
            checkpoint(prompt={run['id']: pid}, run=run['id'], phase='queued')
    except urllib.error.HTTPError as e:
        body = e.read().decode('utf-8', 'replace')
        log(f"[run] {run['id']}: REJECTED by ComfyUI ({e.code}): {body[:600]}")
        return {**base, 'status': 'rejected', 'error': body[:3000]}
    except Exception as e:
        return {**base, 'status': 'queue_failed', 'error': f'{type(e).__name__}: {e}'[:500]}
    trace.pid = pid
    total, free_min = 0.0, None
    timeout = max(900, 4 * run['est_s'])
    h = {}
    if use_ws:
        how = _wait_ws(comfy, trace, run, session, prog, t0, timeout, guard, ws, wait_s=ws_wait)
        ws.trace = None
        if how in ('timeout', 'drift', 'stall'):
            comfy.interrupt()
            if how == 'drift':
                live = trace.live_s_per_step()
                log(f"[drift] {run['id']}: live {live}s/step > {guard.factor}x warm baseline {guard.base.get(drift_key(run))}s/step")
                return {**base, 'status': 'drift', 'wall_s': round(time.time() - t0, 1), 'live_s_per_step': live,
                        'baseline_s_per_step': guard.base.get(drift_key(run)), 'trace': trace.summary()}
            return {**base, 'status': ('stall' if how == 'stall' else 'timeout') if session.alive() else 'stopped_by_watchdog', 'wall_s': round(time.time() - t0, 1),
                    'trace': trace.summary()}
        for _ in range(12):                                    # outputs are in /history the moment execution_success fires
            try:
                h = comfy.get_json(f'/history/{pid}')
            except Exception:
                h = {}
            if pid in h:
                break
            sleep(2)
    else:                                                      # fallback: HTTP polling (+ comfy.log tqdm for s/step)
        while True:
            if prog:
                prog.tick()                                    # heartbeat only if the GPU is busy / logs grow
            else:
                session.heartbeat()
            try:
                h = comfy.get_json(f'/history/{pid}')
            except Exception:
                h = {}
            if pid in h and ((h[pid].get('status') or {}).get('completed') is not None or h[pid].get('outputs')
                             or (h[pid].get('status') or {}).get('status_str') == 'error'):
                break
            if time.time() - t0 > timeout or not session.alive():
                comfy.interrupt()
                return {**base, 'status': 'timeout' if session.alive() else 'stopped_by_watchdog', 'wall_s': round(time.time() - t0, 1)}
            if not pod:
                try:
                    total, free = comfy.vram(); free_min = free if free_min is None else min(free_min, free)
                except Exception:
                    pass
            sleep(poll)
    wall = time.time() - t0
    rec = h.get(pid, {})
    msgs = {k: v for k, v in (rec.get('status') or {}).get('messages') or []}
    render_s = None
    if 'execution_start' in msgs and ('execution_success' in msgs or 'execution_error' in msgs):
        end = (msgs.get('execution_success') or msgs.get('execution_error') or {}).get('timestamp')
        if end and msgs['execution_start'].get('timestamp'):
            render_s = round((end - msgs['execution_start']['timestamp']) / 1000, 1)
    files = [f for o in (rec.get('outputs') or {}).values() for k in ('gifs', 'images', 'videos') for f in o.get(k, [])
             if f.get('type', 'output') == 'output']
    saved = []
    for f in files:
        dest = d / f['filename']
        try:
            comfy.download(f, dest); saved.append(str(dest))
        except Exception as e:
            log(f"[run] {run['id']}: download of {f['filename']} failed ({e})")
    if run.get('stitch'):                                    # Oct 21 default: crop-and-stitch back onto the driver
        for s_ in saved:
            if s_.endswith('.mp4'):
                subprocess.run(['bash', str(ROOT / 'bench' / 'stitch.sh'), s_], check=False)
    status = (rec.get('status') or {}).get('status_str') or (trace.status if use_ws else None) or 'unknown'
    if status == 'success' and not any(s.endswith('.mp4') for s in saved) and run.get('engine') != 'image':
        status = 'no_output'
    res = {**base, 'status': status, 'wall_s': round(wall, 1), 'render_s': render_s, 'usd': round(wall / 3600 * rate, 3), 'outputs': saved}
    if status != 'success':
        res['error'] = _err_text(rec) or (json.dumps(trace.error)[:800] if trace.error else '')
    if use_ws:
        ts = trace.summary()
        res['trace'] = ts
        if ts['warm_s_per_step']:
            res['warm_s_per_step'] = ts['warm_s_per_step']
            res['sampling_s'] = ts['sampling_s']
            res['s_per_step'] = round(ts['sampling_s'] / ts['steps'], 2) if ts['steps'] else None
    if pod:
        gp = pod.gpu(t0)
        if gp.get('peak_mib'):
            res['vram_peak_gb'] = round(gp['peak_mib'] / 1024, 1); res['vram_total_gb'] = round((gp.get('total_mib') or 0) / 1024, 1)
            res['gpu_mean_util'] = gp.get('mean_util')
        rates = tqdm_rates(pod.log('comfy', log_off))
        res['sampler_bars'] = rates
        if rates and not res.get('s_per_step'):
            tot_steps = sum(r['steps'] for r in rates)
            res['s_per_step'] = round(sum(r['steps'] * r['s_per_step'] for r in rates) / tot_steps, 2)
            res['sampling_s'] = round(sum(r['steps'] * r['s_per_step'] for r in rates), 1)
    elif free_min is not None:
        res['vram_peak_gb'] = round(total - free_min, 1)
    steps = (run.get('params') or {}).get('steps')
    if steps and not res.get('s_per_step'):
        res['s_per_step_approx'] = round((render_s or wall) / steps, 2)
    vids = [s for s in saved if s.endswith('.mp4')]
    if vids:
        src = CLIPS / run['uploads'][0] if run['uploads'] else None
        try:
            res['qc'] = qc.run(vids[0], str(src) if src else None, learned=False).to_dict()
        except Exception as e:
            res['qc'] = {'error': str(e)[:200]}
    log(f"[run] {run['id']} (try {attempt}, x{res['scale']}{', COLD' if res.get('cold') else ''}): {status} wall {res['wall_s']}s "
        f"render {render_s}s s/step {res.get('s_per_step') or res.get('s_per_step_approx')} warm {res.get('warm_s_per_step')}  "
        f"peak {res.get('vram_peak_gb')} GB  ${res['usd']}" + (f"  ERR {res.get('error', '')[:300]}" if status != 'success' else ''))
    return res


def _exec_wait(pod: PodStatus, script: str, session=None, prog=None, sleep=time.sleep, timeout_s: float = 3500, poll: float = 15) -> dict:
    jid = pod.exec(script)
    if not jid:
        return {'done': True, 'rc': -1, 'out': 'exec channel unavailable'}
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        j = pod.job(jid)
        if j.get('done'):
            return j
        if prog:
            prog.tick()
        if session is not None and not session.alive():
            return {'done': False, 'rc': None, 'out': 'pod gone'}
        sleep(poll)
    return {'done': False, 'rc': None, 'out': 'timeout'}


def run_wangp_batch(pod: PodStatus, comfy: Comfy, batch: list[dict], session, prog=None, log=print, sleep=time.sleep,
                    install_wait_s: float = 1800) -> list[dict]:
    """All WanGP runs in ONE driver call (one MMGP session: first job = cold, later ones warm). Waits for the side install
    (stage wangp_done), frees ComfyUI's VRAM first, pulls the mp4s back through ComfyUI's /view (output/wangp)."""
    base = lambda r, **kw: {'id': r['id'], 'engine': 'wangp', 'rung': r.get('rung'), 'seed': r.get('seed'), **kw}
    t0 = time.time()
    while prog is not None and not (prog.has('wangp_done') or prog.has('wangp_failed')):
        if not session.alive() or time.time() - t0 > install_wait_s:
            return [base(r, status='wangp_install_timeout') for r in batch]
        prog.tick(); sleep(15)
    if prog is not None and prog.has('wangp_failed'):
        log(f"[wangp] install failed:\n{pod.log('wangp_install')[-2500:]}")
        return [base(r, status='wangp_install_failed') for r in batch]
    try:
        for u in sorted({u for r in batch for u in r['uploads']}):
            comfy.upload(CLIPS / u)
    except Exception as e:
        return [base(r, status='upload_failed', error=str(e)[:300]) for r in batch]
    fv = comfy.free_and_wait(sleep=sleep)
    log(f'[wangp] ComfyUI VRAM freed ({fv} GB free); running {[r["id"] for r in batch]}')
    jobs = [{'id': r['id'], 'settings': r['wangp']['settings']} for r in batch]
    w = batch[0]['wangp']
    out = '"${JUTSU_PREFIX:-/opt/jutsu}/ComfyUI/output/wangp"'
    script = (f"echo {base64.b64encode(json.dumps(jobs).encode()).decode()} | base64 -d > /tmp/wangp_jobs.json && "
              f"/opt/wangp/venv/bin/python worker/wangp_driver.py --jobs /tmp/wangp_jobs.json --out {out} "
              f"--profile {w['profile']} --attention {w['attention']} >> /opt/jstatus/wangp.log 2>&1; echo rc=$?")
    t1 = time.time()
    j = _exec_wait(pod, script, session, prog, sleep)
    log(f"[wangp] driver finished ({j.get('out', '').strip()[-80:]}) after {time.time() - t1:.0f}s")
    rows = {}
    r2 = _exec_wait(pod, f'cat {out}/results.jsonl 2>/dev/null || true', session, None, sleep, 120, 3)
    for line in (r2.get('out') or '').splitlines():
        try:
            x = json.loads(line); rows[x['id']] = x
        except Exception:
            pass
    res = []
    for i, r in enumerate(batch):
        x = rows.get(r['id'])
        if not x:
            res.append(base(r, status='error', error='no result line (see pod_logs/wangp.log)', cold=i == 0)); continue
        rec = base(r, **{k: v for k, v in x.items() if k not in ('id', 'engine', 'outputs')}, cold=i == 0)
        if x.get('status') == 'success':
            try:
                dst = REVIEW / f"{r['id']}.mp4"
                comfy.download({'filename': f"{r['id']}.mp4", 'subfolder': 'wangp', 'type': 'output'}, dst)
                rec['outputs'] = [str(dst)]
            except Exception as e:
                rec['download_error'] = str(e)[:200]
        res.append(rec)
    return res


def _service_fixes(pod: PodStatus | None, log=print):
    """Box-side fix queue: drop a bash script in data/bakeoff/phase1/pod_cmds/<name>.sh -> runs on the pod -> <name>.out."""
    if not pod:
        return
    qd = OUT / 'pod_cmds'; qd.mkdir(exist_ok=True)
    for f in sorted(qd.glob('*.sh')):
        jf, of = f.with_suffix('.job'), f.with_suffix('.out')
        try:
            if not jf.exists():
                jid = pod.exec(f.read_text()); jf.write_text(jid or 'disabled'); log(f'[fix] sent {f.name} -> job {jid}')
            elif not of.exists() and jf.read_text() != 'disabled':
                j = pod.job(jf.read_text().strip())
                if j.get('done'):
                    of.write_text(f"rc={j.get('rc')}\n{j.get('out', '')}"); log(f"[fix] {f.name} done rc={j.get('rc')}")
        except Exception as e:
            log(f'[fix] {f.name}: {e}')


def finish_output(video: str, dst: pathlib.Path, audio_source: dict | None, log=print) -> dict | None:
    """the owner-facing copy: with an audio_source, the ORIGINAL source audio packets are copied back onto the render
    (h2h_identity_fix.remux_source_audio, last frame held for gaps <= 0.5 s); otherwise / on failure a plain stream copy."""
    if audio_source:
        try:
            import h2h_identity_fix as fx
            info = fx.remux_source_audio(video, audio_source['path'], dst, float(audio_source.get('start', 0.0)),
                                         float(audio_source.get('seconds', 10.0)))
            log(f"[audio] {dst.name}: original source audio copied back {info}")
            return info
        except Exception as e:
            log(f'[audio] {dst.name}: source-audio remux failed ({str(e)[:200]}); plain copy instead')
    _ff('-i', video, '-c', 'copy', dst)
    return None


def _extra_runs(seen: int) -> tuple[list[dict], int]:
    """Runs added while live (fixed graphs etc.): one JSON per line in data/bakeoff/phase1/extra_runs.jsonl,
    {"id", "graph_path", "uploads", "engine", "est_s", "params"?, "protect"?}."""
    p = OUT / 'extra_runs.jsonl'
    if not p.exists():
        return [], seen
    lines = [x for x in p.read_text().splitlines() if x.strip()]
    new = []
    for x in lines[seen:]:
        try:
            r = json.loads(x); r['graph'] = json.loads(pathlib.Path(r.pop('graph_path')).read_text())
            r.setdefault('est_s', 600); r.setdefault('uploads', []); r.setdefault('engine', 'extra'); new.append(r)
        except Exception as e:
            print(f'[extra] bad line: {e}')
    return new, len(lines)


DEADMAN_GRACE_S = 300                                     # pod self-removes this long AFTER the box hard stop (time to pull partials)
def _machine_id(prov, pod: dict) -> str | None:
    """RunPod machine id of a pod (create response, else a read-only GET). None if unknown."""
    for src in (pod, None):
        try:
            d = src if src is not None else prov.client.get_pod(pod['id'])
        except Exception:
            return None
        m = d.get('machineId') or (d.get('machine') or {}).get('id') or (d.get('machine') or {}).get('podHostId')
        if m:
            return str(m)
    return None


EST_FACTOR = float(os.environ.get('FP_EST_FACTOR', '1.1'))    # was 1.3 on top of already-padded est_s -> skipped the r4 render


def est_need_usd(est_s: float, rate: float) -> float:
    return est_s * EST_FACTOR / 3600 * rate


def deadman_seconds(rate: float, cap: float, safety: float, grace: int = DEADMAN_GRACE_S) -> int:
    """Pod lifetime (from boot) at the pod's ACTUAL $/hr: the box hard stop (cap*safety) + grace, never past the cap."""
    hard = int(cap * safety / rate * 3600)
    return min(hard + grace, int(cap / rate * 3600) - 60)


def deadman_cmd(seconds: int) -> str:
    return f'echo $(( $(cat /tmp/jutsu_t0) + {int(seconds)} )) > /tmp/jutsu_deadline && cat /tmp/jutsu_deadline'


def push_deadman(rate: float, cap: float, safety: float, log=print) -> int:
    n = deadman_seconds(rate, cap, safety)
    qd = OUT / 'pod_cmds'; qd.mkdir(parents=True, exist_ok=True)
    f = qd / '000_deadman.sh'
    for x in (f.with_suffix('.job'), f.with_suffix('.out')):
        x.unlink(missing_ok=True)
    f.write_text(deadman_cmd(n) + '\n')
    log(f'[deadman] actual rate ${rate:.3f}/hr -> pod self-removes {n}s after boot (box hard stop first)')
    return n


def checkpoint(**kw) -> dict:
    """Resumable state: OUT/checkpoint.json (merged) + one line in STATUS.md per step."""
    cp = OUT / 'checkpoint.json'
    st = json.loads(cp.read_text()) if cp.exists() else {}
    if 'prompt' in kw:
        st.setdefault('prompts', {}).update(kw.pop('prompt'))
    if 'done' in kw:
        st.setdefault('done', {}).update(kw.pop('done'))
    st.update(kw); st['t'] = time.strftime('%Y-%m-%d %H:%M:%S PT')
    cp.write_text(json.dumps(st, indent=1, default=str))
    try:
        with open(OUT / 'STATUS.md', 'a') as f:
            f.write(f"- {st['t']} [{st.get('phase')}] pod={st.get('pod')} spent=${st.get('spent')} run={st.get('run')} "
                    f"done={sorted(st.get('done', {}))}\n")
    except Exception:
        pass
    return st


def load_checkpoint() -> dict:
    cp = OUT / 'checkpoint.json'
    return json.loads(cp.read_text()) if cp.exists() else {}


def stage_prefixes(graph: dict) -> list[str]:
    """filename_prefix of every SaveVideo / VHS_VideoCombine / SaveImage node, in node-id order (swap, refine, sync)."""
    out = []
    for nid in sorted(graph, key=lambda k: int(k) if str(k).isdigit() else 0):
        n = graph[nid]
        if n.get('class_type') in ('SaveVideo', 'VHS_VideoCombine', 'SaveImage', 'SaveAnimatedWEBP'):
            pre = (n.get('inputs') or {}).get('filename_prefix')
            if isinstance(pre, str):
                out.append(pre)
    return out


def pull_partials(comfy, run: dict, log=print, dest: pathlib.Path | None = None) -> list[str]:
    """On ANY stop: fetch whatever stage outputs exist (history first, then the predictable output names)."""
    dest = dest or (REVIEW / 'partials'); dest.mkdir(parents=True, exist_ok=True)
    got, files = [], []
    pid = (load_checkpoint().get('prompts') or {}).get(run['id'])
    if pid:
        try:
            for o in ((comfy.get_json(f'/history/{pid}').get(pid) or {}).get('outputs') or {}).values():
                for k in ('videos', 'gifs', 'images'):
                    files += o.get(k) or []
        except Exception:
            pass
    for pre in stage_prefixes(run.get('graph') or {}):
        sub, _, name = pre.rpartition('/')
        for i in range(1, 4):
            for pat in (f'{name}_{i:05d}_.mp4', f'{name}_{i:05d}.mp4', f'{name}_{i:05d}-audio.mp4'):
                files.append({'filename': pat, 'subfolder': sub, 'type': 'output'})
    seen = set()
    for f in files:
        key = (f.get('subfolder', ''), f['filename'])
        if key in seen:
            continue
        seen.add(key)
        try:
            d = dest / f"{run['id']}__{f['filename']}"
            comfy.download(f, d)
            if d.exists() and d.stat().st_size > 0:
                got.append(str(d))
        except Exception:
            pass
    log(f"[partials] {run['id']}: {got or 'no stage outputs on the pod yet'}")
    return got


def _state(**kw):
    (OUT / 'live_state.json').write_text(json.dumps({'t': time.strftime('%H:%M:%S'), **kw}, indent=1, default=str))
    if kw.get('phase') in ('setup_done', 'run', 'terminated', 'resumed'):
        checkpoint(**{k: v for k, v in kw.items() if k in ('phase', 'pod', 'spent', 'run')})


def launch(plan: dict, cap: float, approve_flag: bool, option: str, allow_pro6000=False, volume_id=None, keep_volume=False,
           confirm=input, isatty=sys.stdin.isatty, client=None, comfy_http=None, log=print, sleep=time.sleep,
           setup_timeout_s=int(os.environ.get('FP_SETUP_TIMEOUT', '5400')), linger_s=600, min_download_mbps=int(os.environ.get('FP_MIN_DL', '800')), create_attempts=8, ws_factory=None,
           attach_pod_id: str | None = None, already_spent_usd: float | None = None) -> list[dict]:
    from app.providers.runpod_pod import RunPodPodProvider
    from app.providers.base import ProviderError
    if not approve_flag:
        raise ApprovalError('launch needs --i-approve-spend')
    if not isatty():
        raise ApprovalError('launch needs an interactive terminal: the owner must type the confirmation himself')
    prov_cfg = dict(config.load()['providers']['runpod_pod'])
    hard = float(prov_cfg.get('hard_cap_usd', 5.0))
    if cap > hard:
        raise ApprovalError(f'cap ${cap:.2f} > provider hard cap ${hard:.2f}')
    if plan['problems']:
        raise ApprovalError(f'plan has problems, fix first: {plan["problems"]}')
    phrase = f'SPEND {cap:.2f}'
    log(f"About to rent {plan['gpu']} ({plan['option_label']}) for up to ${cap:.2f}. Estimate ${plan['est_usd']['total']}.")
    if confirm(f'Type exactly "{phrase}" to approve: ').strip() != phrase:
        raise ApprovalError('confirmation phrase not typed; nothing launched')
    approval = Approval(job='phase1_bakeoff', max_usd=cap, confirmed=True)
    o = OPTIONS[option]
    prov_cfg['cloud_type'] = plan.get('cloud') or o['cloud']
    cloud_order = cloud_try_order(os.environ.get('JUTSU_CLOUD_ORDER') or prov_cfg['cloud_type'], bool(os.environ.get('FP_VOLUME_ID') or volume_id))
    prov_cfg['cloud_type'] = cloud_order[0]
    if option == 'B':                                          # watchdog falls back to the worst listed rate if the pod hides its price
        prov_cfg['gpus'] = [{**g, 'usd_per_hr': g.get('community_usd_per_hr', g['usd_per_hr'])} for g in prov_cfg['gpus']]
    prov = RunPodPodProvider(prov_cfg, client=client)
    prov.check_gates(approval, worst_case_usd=cap)             # key in env, cap, balance
    try:
        bal0 = prov.client.balance()
    except Exception:
        bal0 = None
    log(f'[gates] ok; RunPod balance ${bal0}')
    gpu_ids = [g['id'] for g in prov_cfg['gpus'] if g['name'] == '5090' or (allow_pro6000 and g['name'] == 'pro6000')]
    if plan.get('gpu_name') and plan['gpu_name'] != '5090':   # refine --gpu pro6000: pin to that one GPU type only
        if not allow_pro6000:
            raise ApprovalError('--gpu pro6000 needs --allow-pro6000')
        gpu_ids = [g['id'] for g in prov_cfg['gpus'] if g['name'] == plan['gpu_name']]
    worst_rate = max(float(g['usd_per_hr']) for g in prov_cfg['gpus'] if g['id'] in gpu_ids)
    dead_man = prov.max_seconds_for(worst_rate, cap)
    if option == 'A' and not volume_id:
        raise ApprovalError('option A needs --volume-id (create the 200 GB volume in the console first; it bills monthly)')
    nonce = uuid.uuid4().hex + uuid.uuid4().hex             # per-launch fix-channel nonce: memory + pod env only
    env = {'JUTSU_BOOT_B64': boot_payload(plan['model_groups']), 'JUTSU_GROUPS': ' '.join(plan['model_groups']),
           'JUTSU_PREFIX': '/opt/jutsu', 'MODEL_ROOT': '/opt/jm', 'JUTSU_FIX_NONCE': nonce,
           'HF_HUB_ENABLE_HF_TRANSFER': '1', 'PYTHONUNBUFFERED': '1'}
    if plan.get('stage1_bundles'):
        env['JUTSU_STAGE1_BUNDLES'] = ' '.join(plan['stage1_bundles'])
    if plan.get('wangp'):
        env['JUTSU_WANGP'] = '1'
    if os.environ.get('JUTSU_UNCHUNK') == '1':
        env['JUTSU_UNCHUNK'] = '1'
    for k in ('JUTSU_ENABLE_SAGE3', 'JUTSU_ENABLE_SPARK', 'JUTSU_ENABLE_H3RELAY', 'JUTSU_ENABLE_FASTH3', 'JUTSU_SAGE', 'JUTSU_SAGE_SM120_MODE'):
        if os.environ.get(k):
            env[k] = os.environ[k]   # bench v2 flags
    if option == 'A':
        env.update({'JUTSU_PREFIX': '/workspace/jutsu', 'MODEL_ROOT': '/workspace'})
    vol_env = os.environ.get('FP_VOLUME_ID') or None      # Oct 20: pre-filled network volume (models at /workspace/models)
    if vol_env and option != 'A':
        env['MODEL_ROOT'] = '/workspace'
    mk = lambda mbps: prov.pod_body('jutsu-phase1-bakeoff', gpu_ids, IMAGE, volume_id if option == 'A' else vol_env, (os.environ.get('FP_DC_IDS') or '').split(',') if os.environ.get('FP_DC_IDS') else None, env=env,
                                    start_cmd=boot_cmd(dead_man), container_disk_gb=o['container_disk_gb'],
                                    ports=['8188/http', f'{STATUS_PORT}/http'], min_download_mbps=mbps)
    graphs = {r['id']: (r.get('graph') if plan.get('refine') else json.loads((OUT / r['id'] / 'prompt_api.json').read_text()))
              for r in plan['runs'] if r.get('engine') != 'wangp'}
    cmp_ids = plan['compare_ids'] if plan.get('inpaint') else (plan.get('compare_ids') or G8_IDS)   # inpaint: composites, no grid
    cmp_src = pathlib.Path(plan['compare_source']) if plan.get('compare_source') else CLIPS / 'genjutsu8.mp4'
    cmp_out = REVIEW / ('refine_compare.mp4' if plan.get('refine') else 'h2h_compare.mp4' if plan.get('h2h') else 'g8_compare.mp4')
    results = []
    OUT.mkdir(parents=True, exist_ok=True); REVIEW.mkdir(parents=True, exist_ok=True)
    if not attach_pod_id:                                  # Oct 21: attach keeps results + extra_runs (autodrive reads them)
        if (OUT / 'results.jsonl').exists():
            (OUT / 'results.jsonl').rename(OUT / f'results.{int(time.time())}.jsonl')
        for f in (OUT / 'STOP', OUT / 'extra_runs.jsonl'):
            f.unlink(missing_ok=True)
    session = None; ws = None
    if attach_pod_id:
        # Salvage an already-running pod (box reboot / launcher death). Never create a new one.
        rate0 = float(plan.get('usd_per_hr') or (worst_rate + o['container_disk_gb'] * DISK_USD_PER_GB_HR))
        existing = prov.client.get_pod(attach_pod_id)
        spent0 = float(already_spent_usd) if already_spent_usd is not None else estimate_pod_spent(existing, rate0)
        # Floor: the owner's $5 round -> balance must not go below (pre-round balance - 5). Cap accounting uses total round spend.
        log(f'[attach] reusing pod {attach_pod_id}; estimated spent so far ${spent0:.3f} of ${cap:.2f} @ ${rate0:.3f}/hr')
        if spent0 >= cap * float(prov_cfg.get('watchdog_safety', 0.92)):
            raise BudgetExceeded(f'attach refused: already spent ${spent0:.2f} >= safety of ${cap:.2f}')
        # Prefer the live pod env nonce so /exec fixes keep working.
        nonce = ((existing.get('env') or {}).get('JUTSU_FIX_NONCE') or nonce)
        s_try = prov.attach_session(attach_pod_id, approval, spent_so_far=spent0, rate=rate0, log=log)
        session = s_try.__enter__()
        # attach_session already set rate_override to the effective (disk-incl) plan rate; do not add disk twice.
    else:
        log(f'[runpod] image {IMAGE}' + (f'; avoiding machines {sorted(AVOID_MACHINES)}' if AVOID_MACHINES else ''))
        avoid_spent = 0.0
        for i in range(create_attempts):                           # Low stock: retry, relaxing the download-speed filter after 2 tries
            prov.cfg['cloud_type'] = cloud_for_attempt(cloud_order, i, create_attempts)
            if len(cloud_order) > 1:
                log(f"[runpod] attempt {i + 1}: cloud {prov.cfg['cloud_type']}")
            s_try = prov.session(mk(min_download_mbps if i < 2 else None), approval, spent_so_far=avoid_spent, log=log)
            try:
                session = s_try.__enter__()
                mid = _machine_id(prov, session.pod)
                if mid and mid in AVOID_MACHINES:
                    log(f'[runpod] pod {session.pod["id"]} landed on avoided machine {mid}; terminating and retrying')
                    avoid_spent = session.spent() + 0.01; s_try.__exit__(None, None, None); session = None
                    if i == create_attempts - 1:
                        raise ProviderError('only avoided machines available')
                    sleep(20); continue
                break
            except ProviderError as e:
                log(f'[runpod] create attempt {i + 1}/{create_attempts} failed: {str(e)[:200]}')
                if i == create_attempts - 1:
                    raise
                sleep(45)
    s = session
    t_launch = time.time()
    try:
        if not attach_pod_id:
            s.watchdog.rate += o['container_disk_gb'] * DISK_USD_PER_GB_HR    # disk while running, so spend is never under-counted
        rate = s.watchdog.rate
        log(f"[runpod] effective rate ${rate:.3f}/hr (incl. {o['container_disk_gb']} GB container disk); pod {s.pod['id']}")
        push_deadman(rate, cap, float(prov_cfg.get('watchdog_safety', 0.92)), log=log)   # actual tier rate, not worst case
        cur = {'run': None, 'pulled': False}
        comfy = Comfy(f"https://{s.pod['id']}-8188.proxy.runpod.net", http=comfy_http)
        pod = PodStatus(f"https://{s.pod['id']}-{STATUS_PORT}.proxy.runpod.net", http=comfy_http, nonce=nonce)
        prog = Progress(pod, s, log=log)
        t0 = time.time(); last_note = 0; warned = False
        while not comfy.ready():                               # boot + install || stage-1 model pull
            st = prog.tick()
            _service_fixes(pod, log)
            if time.time() - last_note > 120:
                last_note = time.time()
                if not st and time.time() - t0 > 240:
                    log(f'[setup] WARNING pod status endpoint unreachable from the box: {pod.last_error}')
                log(f"[setup] {time.time() - t0:.0f}s  models {((st or {}).get('models_bytes') or 0) / 1e9:.1f} GB  "
                    f"disk free {(st or {}).get('disk_free_gb')} GB  ${s.spent():.2f}")
                _state(phase='setup', spent=round(s.spent(), 3), pod=s.pod['id'], status=st)
            if (prog.has('install_failed') or prog.has('dl1_failed')) and not warned:
                warned = True
                tail = pod.log('install', max(0, ((st or {}).get('log_bytes') or {}).get('install', 0) - 4000))
                log(f'[setup] pod reported a failed stage; install.log tail:\n{tail[-3000:]}')
                if prog.has('install_failed'):
                    raise BudgetExceeded('install.sh failed on the pod; terminating')
            if st: seen_status = True
            if not locals().get('seen_status') and time.time() - t0 > int(os.environ.get('FP_BOOT_TIMEOUT', '600')):
                raise BudgetExceeded('boot watchdog: image pull/boot never reached the status server; terminating')
            if time.time() - t0 > setup_timeout_s or not s.alive():
                raise BudgetExceeded('worker never became ready; terminating')
            sleep(15)
        setup_s = round(time.time() - t0, 1)
        log(f"[setup] ComfyUI ready after {setup_s:.0f}s (${s.spent():.2f} so far)")
        need = {n['class_type'] for g in graphs.values() for n in g.values() if isinstance(n, dict) and n.get('class_type')}
        try:
            objs = comfy.get_json('/object_info')
            miss = sorted(need - set(objs))
            log(f"[setup] {len(objs)} node classes; missing for this plan: {miss or 'none'}")
        except Exception as e:
            miss = None; log(f'[setup] object_info failed: {e}')
        a08_need = need & {'H3ExactLoraStack', 'H3LoadReport'}
        if a08_need and (miss is None or any(n in (miss or []) for n in a08_need)):
            raise BudgetExceeded(f'A08 nodes missing from ComfyUI (load check cannot run): {sorted(a08_need)} missing={miss}')
        if plan.get('box_uploads'):
            push_box_uploads(comfy, plan['box_uploads'], s, log=log, sleep=sleep)
        if ws_factory:
            ws = ws_factory(comfy)
        elif comfy_http is None:                               # real pod: live progress over the websocket
            ws = ComfyWS(comfy.base, {'User-Agent': UA}, log=log).start()
        if ws:
            log(f"[ws] progress websocket {'connected' if ws.connected.wait(20) else 'NOT connected (falling back to /history polling + comfy.log)'}")
        guard = DriftGuard(factor=float(plan.get('drift_factor', 1.5))) if plan.get('drift_guard') else None   # off by default (Oct 17)
        last_sig = None
        queue = [{**r, 'graph': graphs.get(r['id'])} for r in plan['runs']]
        cp = load_checkpoint() if attach_pod_id else {}
        if cp:                                                 # resume on the same pod: skip finished runs, re-attach to queued prompts
            done = {k for k, v in (cp.get('done') or {}).items() if v == 'success'}
            queue = [r for r in queue if r['id'] not in done]
            for r in queue:
                if (cp.get('prompts') or {}).get(r['id']) and cp.get('pod') == attach_pod_id:
                    r['resume_pid'] = cp['prompts'][r['id']]
            log(f"[resume] checkpoint {cp.get('t')}: skipping done {sorted(done) or 'none'}")
            _state(phase='resumed', pod=s.pod['id'], spent=round(s.spent(), 3))

        def _pull():
            if cur['run'] and not cur['pulled']:
                cur['pulled'] = True
                pull_partials(comfy, cur['run'], log=log)
        s.before_stop = _pull                                  # watchdog / signal / any terminate pulls partials first
        fails: dict[str, int] = {}
        extra_seen, last_engine, compared, linger_t0 = 0, None, False, None
        safety = float(prov_cfg.get('watchdog_safety', 0.92))
        while s.alive():
            new, extra_seen = _extra_runs(extra_seen)
            if new:
                log(f"[extra] +{[r['id'] for r in new]}"); queue = new + queue if any(r.get('protect') for r in new) else queue + new
                linger_t0 = None
            _service_fixes(pod, log)
            if (OUT / 'SKIP.txt').exists():                    # Oct 21: drop queued runs by id (the owner narrowed the focus)
                sk = set((OUT / 'SKIP.txt').read_text().split()); n0 = len(queue)
                queue = [q for q in queue if q['id'] not in sk]
                if len(queue) != n0: log(f'[skip] dropped {n0 - len(queue)} queued runs')
            if (OUT / 'STOP').exists():
                log('[stop] STOP file found'); break
            if not compared and not any(r['id'] in cmp_ids for r in queue):
                compared = True
                try:
                    c = compare_side_by_side(results, cmp_ids, cmp_src, cmp_out)
                    log(f'[compare] {c or "nothing to compare (no successful head-to-head render)"}  (${s.spent():.2f})')
                except Exception as e:
                    log(f'[compare] failed ({e})')
            if not queue:
                # A queued pod command (the A08 stitch/encode) keeps the pod up until its .out lands.
                pending_cmds = [f for f in (OUT / 'pod_cmds').glob('*.sh') if not f.with_suffix('.out').exists()]
                if pending_cmds:
                    linger_t0 = time.time()
                linger_t0 = linger_t0 or time.time()
                if time.time() - linger_t0 > linger_s:
                    break
                prog.tick(extra=int(time.time() - linger_t0) // 60); s.heartbeat()
                _state(phase='linger', spent=round(s.spent(), 3), pod=s.pod['id'], left_s=int(linger_s - (time.time() - linger_t0)))
                sleep(10); continue
            r = queue.pop(0)
            if r.get('engine') == 'wangp':                      # separate app: batch every queued WanGP run into one session
                batch = [r] + [q for q in queue if q.get('engine') == 'wangp']
                queue = [q for q in queue if q.get('engine') != 'wangp']
                need = est_need_usd(sum(q['est_s'] for q in batch), rate)
                while batch and s.spent() + need > cap * safety:
                    cut = batch.pop(); need -= cut['est_s'] * 1.3 / 3600 * rate
                    results.append({'id': cut['id'], 'status': 'skipped_budget', 'spent': round(s.spent(), 2)})
                if batch:
                    _state(phase='run', run=[q['id'] for q in batch], spent=round(s.spent(), 3), pod=s.pod['id'])
                    for res in run_wangp_batch(pod, comfy, batch, s, prog, log=log, sleep=sleep):
                        res['spent_after'] = round(s.spent(), 3); results.append(res)
                        log(f"[run] {res['id']}: {res['status']}  warm {res.get('warm_s_per_step')} s/step  wall {res.get('wall_s')} s")
                        with open(OUT / 'results.jsonl', 'a') as f:
                            f.write(json.dumps(res, default=str) + '\n')
                last_engine, last_sig = 'wangp', None
                continue
            waits = r.get('needs_stage2', r['id'] not in G8_IDS and not r.get('protect'))
            if waits and not (prog.has('dl2_done') or prog.has('dl2_failed')):
                log('[run] waiting for the stage-2 model pull before the non-head-to-head runs')
                while s.alive() and not (prog.has('dl2_done') or prog.has('dl2_failed')):
                    prog.tick(); _service_fixes(pod, log); sleep(15)
                if prog.has('dl2_failed'):
                    log(f"[pod] dl2 had failures:\n{pod.log('dl2')[-2000:]}")
            need = est_need_usd(r['est_s'], rate)
            if not s.alive() or (s.spent() + need > cap * safety and not r.get('protect')):
                res = {'id': r['id'], 'status': 'skipped_budget', 'spent': round(s.spent(), 2)}
                results.append(res); log(f"[run] {r['id']}: skipped (budget ${s.spent():.2f} + est ${need:.2f} > ${cap * safety:.2f})")
                with open(OUT / 'results.jsonl', 'a') as f:
                    f.write(json.dumps(res) + '\n')
                continue
            if 'flashvsr' in str(r.get('engine')):              # Phase 1 OOM: the previous engine still held 26.5 GB
                fv = comfy.free_and_wait(sleep=sleep); last_engine = r.get('engine')
                log(f'[run] freed VRAM before FlashVSR: {fv} GB free')
            elif r.get('engine') != last_engine:
                comfy.free(); last_engine = r.get('engine')
            sig = model_signature(r['graph']); cold = sig != last_sig
            _state(phase='run', run=r['id'], attempt=fails.get(r['id'], 0) + 1, spent=round(s.spent(), 3), pod=s.pod['id'],
                   done=[x['id'] + ':' + x['status'] for x in results])
            cur.update(run=r, pulled=False)
            res = run_one(comfy, r, s, rate, log=log, sleep=sleep, pod=pod, prog=prog, attempt=fails.get(r['id'], 0) + 1,
                          ws=ws, guard=guard, cold=cold)
            res['spent_after'] = round(s.spent(), 3)
            results.append(res)
            with open(OUT / 'results.jsonl', 'a') as f:
                f.write(json.dumps(res, default=str) + '\n')
            if res['status'] == 'success':
                cur['run'] = None; checkpoint(done={r['id']: 'success'}, phase='run_done', run=r['id'], spent=round(s.spent(), 3))
                last_sig = sig
                guard and guard.record(drift_key(r), res.get("warm_s_per_step"))
                for v in res.get('outputs', []):
                    if v.endswith('.mp4'):
                        dst = REVIEW / f"{r['id']}{'' if res.get('scale', 1.0) == 1.0 else '_x' + str(res['scale'])}.mp4"
                        if plan.get('inpaint'):          # paste the generated region back into the 1080p source + source audio
                            try:
                                import h3_inpaint as hi
                                log(f"[inpaint] composite -> {hi.composite(r['id'], v, dst)}")
                            except Exception as e:
                                log(f'[inpaint] composite failed ({e}); raw render kept'); finish_output(v, dst, None, log)
                        else:
                            finish_output(v, dst, plan.get('audio_source'), log)
                        break
                continue
            if res['status'] == 'stopped_by_watchdog':
                break
            if any(s in (res.get('error') or '') for s in ('JUTSU vae-check FAIL', 'JUTSU no-loop', 'H3 load check FAILED')):   # fail-closed guards -> stop the session, never retry
                log(f"[abort] {r['id']}: {(res.get('error') or '').strip()[:200]} -> STOP (no retries, terminating)")
                (OUT / 'STOP').write_text('fail-closed guard: ' + (res.get('error') or '')[:500]); break
            fails[r['id']] = fails.get(r['id'], 0) + 1
            if fails[r['id']] >= 2:
                log(f"[run] {r['id']}: failed twice -> skipped"); continue
            comfy.free()
            retry = dict(r)
            if res['status'] == 'drift':                       # step time drifted: restart ComfyUI, rerun right away (cold)
                restart_comfy(pod, comfy, log=log, sleep=sleep, session=s); last_sig = None
                queue.insert(0, retry)
            elif _is_oom(res.get('error', '')):
                retry['graph'] = downscale_graph(r['graph']); retry['scale'] = round(r.get('scale', 1.0) * 0.75, 3)
                log(f"[run] {r['id']}: OOM -> retry at x{retry['scale']} resolution")
                queue.insert(0, retry)
            elif r['id'] in cmp_ids:                           # head-to-head: retry after the other head-to-head runs (fix window)
                pos = max([i + 1 for i, q in enumerate(queue) if q['id'] in cmp_ids] or [0]); queue.insert(pos, retry)
            else:
                queue.append(retry)
        log(f"[done] spent ~${s.spent():.2f} of ${cap:.2f} in {time.time() - t_launch:.0f}s")
        if not compared:
            try:
                c = compare_side_by_side(results, cmp_ids, cmp_src, cmp_out)
                log(f'[compare] {c}')
            except Exception as e:
                log(f'[compare] failed ({e})')
        try:
            for name in ('install', 'dl1', 'dl2', 'comfy', 'apt', 'boot', 'wangp_install', 'wangp'):
                (OUT / 'pod_logs').mkdir(exist_ok=True); (OUT / 'pod_logs' / f'{name}.log').write_text(pod.log(name))
            (OUT / 'pod_logs' / 'status.json').write_text(json.dumps(prog.last, indent=1))
        except Exception:
            pass
        summary = {'pod': s.pod['id'], 'rate': rate, 'setup_s': setup_s, 'spent_est': round(s.spent(), 3), 'balance_before': bal0,
                   'watchdog_reason': s.watchdog.reason, 'results': results}
        if any(x.get('rung') for x in results):
            summary['ladder_rank'] = rank_runs(results)
            (OUT / 'ladder_rank.json').write_text(json.dumps(summary['ladder_rank'], indent=1))
            for x in summary['ladder_rank']:
                log(f"[rank] {x['rung']:5} warm {x['warm_s_per_step']} s/step  wall {x['wall_s']} s  seeds {x['seeds']}"
                    f"{'' if x['enough_seeds'] else '  (ONE SEED ONLY)'}")
        (OUT / 'summary.json').write_text(json.dumps(summary, indent=1, default=str))
    finally:
        if ws:
            ws.stop()
        try:
            _pull()
        except Exception:
            pass
        s.__exit__(*sys.exc_info())
        _state(phase='terminated', spent=round(s.spent(), 3), pod=s.pod['id'])
    if option == 'A' and volume_id and not keep_volume:
        log('[volume] kept as-is (pre-existing volume; delete it in the console when Phase 1 review is done)')
    return results


def verify_clean(client=None, log=print) -> dict:
    """End-of-run proof via the API: zero pods and zero network volumes (terminates any leftover jutsu pod)."""
    from app.providers.runpod_pod import RunPodClient
    c = client or RunPodClient()
    pods = c.call('GET', '/pods') or []
    for p in pods:                                         # Oct 20: report only; never kill other agents' pods by name
        if str(p.get('name', '')).startswith('jutsu-'):
            log(f"[verify] NOTE jutsu-named pod still listed: {p['id']} {p.get('name')} (not touched)")
    pods = c.call('GET', '/pods') or []
    vols = c.list_volumes() or []
    out = {'pods': [{'id': p.get('id'), 'name': p.get('name'), 'status': p.get('desiredStatus')} for p in pods],
           'network_volumes': [{'id': v.get('id'), 'name': v.get('name')} for v in vols], 'balance': c.balance()}
    log(f"[verify] pods={len(out['pods'])} volumes={len(out['network_volumes'])} balance=${out['balance']}")
    return out


def main(argv=None):
    global OUT, REVIEW
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['prep', 'plan', 'launch', 'verify'])
    ap.add_argument('--linger', type=int, default=600, help='seconds to keep the pod after the queue empties (extra_runs.jsonl)')
    ap.add_argument('--with-veda', action='store_true'); ap.add_argument('--with-x2', action='store_true')
    ap.add_argument('--option', choices=['A', 'B'], default='B')
    ap.add_argument('--cap', type=float, default=5.0)
    ap.add_argument('--i-approve-spend', action='store_true')
    ap.add_argument('--allow-pro6000', action='store_true')
    ap.add_argument('--volume-id'); ap.add_argument('--keep-volume', action='store_true')
    ap.add_argument('--json', action='store_true')
    ap.add_argument('--ladder', action='store_true', help='L0-L5 H3 fast-path ladder (2 seeds per rung) instead of the Phase 1 set')
    ap.add_argument('--h2h', action='store_true', help='10 s head-to-head + H3 ladder on --motion/--ref (or --brief)')
    ap.add_argument('--brief', default=str(ROOT / 'config' / 'h2h' / 'oct7b.json'),
                    help='h2h brief JSON (subject text, shot list, consent note); --motion/--ref/--start/--seconds override it')
    ap.add_argument('--motion'); ap.add_argument('--ref')
    ap.add_argument('--start', type=float); ap.add_argument('--seconds', type=float)
    ap.add_argument('--seeds', default=','.join(map(str, H2H_SEEDS)))
    ap.add_argument('--likeness-consent', action='store_true', help='the owner confirms rights to the likeness in --ref')
    ap.add_argument('--with-baseline', action='store_true', help='h2h: add the Phase-1 int8/PyTorch H3 rung (speed control)')
    ap.add_argument('--no-wangp', action='store_true', help='h2h: skip the WanGP/MMGP rung (no side install)')
    ap.add_argument('--rungs', help='h2h: comma list of rungs to keep (e.g. H3-P8,H3-T4,WAN-VACE)')
    ap.add_argument('--gpu', default='5090', choices=['5090', 'pro6000', 'any24', 'a100', 'h100', 'h200', 'b200'], help='refine: GPU type (pro6000 needs --allow-pro6000)')
    ap.add_argument('--cloud', default='COMMUNITY', choices=['COMMUNITY', 'SECURE', 'COMMUNITY_FIRST'],
                    help='refine: RunPod cloud type; COMMUNITY_FIRST tries Community then falls back to Secure')
    ap.add_argument('--create-attempts', type=int, default=8, help='refine: pod-create tries (45 s apart) before giving up')
    ap.add_argument('--refine', action='store_true', help='Oct 8 refine round (graphs from h2h_identity_fix.py refine-build)')
    ap.add_argument('--inpaint', action='store_true', help='with --refine: H3 masked inpaint rungs (scripts/h3_inpaint.py build)')
    ap.add_argument('--better', action='store_true', help='with --refine: Oct 13 better-build 7-way (scripts/better_build.py build)')
    ap.add_argument('--fullperson', action='store_true', help='with --refine: Oct 13 Character FULL_PERSON H3 vs Viggle (scripts/fullperson_pack.py prep)')
    ap.add_argument('--attach-pod', help='reuse an already-running pod id (no create); keeps watchdog/terminate/verify')
    a = ap.parse_args(argv)
    if a.cmd == 'prep':
        prep(); return
    if a.cmd == 'verify':
        print(json.dumps(verify_clean(), indent=1)); return
    if a.refine:
        OUT = FP_OUT if a.fullperson else BETTER_OUT if a.better else INPAINT_OUT if a.inpaint else REFINE_OUT
        REVIEW = ROOT / 'data' / 'bakeoff' / 'out' / (os.environ.get('FP_REVIEW', 'fullperson') if a.fullperson else 'better' if a.better else 'inpaint' if a.inpaint else 'refine')
        if a.cloud == 'COMMUNITY_FIRST':
            os.environ['JUTSU_CLOUD_ORDER'] = 'COMMUNITY_FIRST'
        plan = build_refine_plan(a.cap, gpu=a.gpu, cloud=('COMMUNITY' if a.cloud == 'COMMUNITY_FIRST' else a.cloud), inpaint=a.inpaint, better=a.better, fullperson=a.fullperson)
        linger = REFINE_LINGER_S if a.linger == 600 else a.linger
        print_refine_plan(plan, a.cap)
        if a.cmd == 'plan':
            if a.json:
                print(json.dumps({k: v for k, v in plan.items() if k != 'runs'}, indent=1))
            return
        try:
            res = launch(plan, a.cap, a.i_approve_spend, plan.get('option_key', 'R'), a.allow_pro6000, None, False, linger_s=linger,
                         create_attempts=a.create_attempts, attach_pod_id=a.attach_pod)
            print(json.dumps(res, indent=1, default=str))
        finally:
            if a.i_approve_spend:
                try:
                    (OUT / 'verify.json').write_text(json.dumps(verify_clean(), indent=1))
                except Exception as e:
                    print(f'[verify] FAILED to verify via API: {e}  -> check the RunPod console NOW')
        return
    h2h = None
    if a.h2h:
        OUT = H2H_OUT                                          # keep Phase 1 artifacts untouched
        b = load_brief(a.brief) if a.brief and pathlib.Path(a.brief).exists() else {}
        for k in ('motion', 'ref', 'start', 'seconds'):
            if getattr(a, k) is not None:
                b[k] = getattr(a, k)
        if a.likeness_consent:
            b['likeness_consent'] = 'confirmed on the command line (--likeness-consent)'
        if not b.get('motion') or not b.get('ref'):
            ap.error('--h2h needs --motion and --ref (or a --brief that names them)')
        info = prep_h2h(b['motion'], b['ref'], float(b.get('start') or 0.0), float(b.get('seconds') or 10.0))
        b['cuts'] = info['cuts']
        seeds = tuple(int(x) for x in a.seeds.split(',') if x.strip())
        b['runs'] = h2h_runs(b, seeds, baseline=a.with_baseline, wangp=not a.no_wangp,
                             rungs=[x.strip() for x in a.rungs.split(',')] if a.rungs else None)
        h2h = b
    plan = build_plan(a.with_veda, a.with_x2, a.option, ladder=a.ladder, h2h=h2h)
    if (a.ladder or a.h2h) and a.linger == 600:
        a.linger = H2H_LINGER_S if a.h2h else LADDER_LINGER_S
    if a.cmd == 'plan':
        print(json.dumps({k: v for k, v in plan.items() if k != 'runs'}, indent=1)) if a.json else print_plan(plan)
        return
    print_plan(plan)
    try:
        res = launch(plan, a.cap, a.i_approve_spend, plan['option'], a.allow_pro6000, a.volume_id, a.keep_volume, linger_s=a.linger,
                     attach_pod_id=a.attach_pod)
        print(json.dumps(res, indent=1, default=str))
    finally:
        if a.i_approve_spend:
            try:
                (OUT / 'verify.json').write_text(json.dumps(verify_clean(), indent=1))
            except Exception as e:
                print(f'[verify] FAILED to verify via API: {e}  -> check the RunPod console NOW')


if __name__ == '__main__':
    main()
