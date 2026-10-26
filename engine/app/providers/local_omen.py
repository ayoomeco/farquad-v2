"""the owner's HP Omen (Windows 11, RTX 2060 12GB Turing sm75, 32 GB RAM) as a FREE draft/prep node.
PLAN-ONLY stub (research/LOCAL-OMEN-PLAN.md): it produces the exact install/launch plan and the Turing-safe job
profile; the box cannot start jobs on the Omen by itself (that would go through the owner's desktop app, with his OK).

Turing rules baked in: fp16 compute (--fp16-unet; bf16 is ~6x slower), int8 / w4a8 weights (fp8/nvfp4 only as
storage), comfy-kitchen CK attention, --disable-dynamic-vram (required on Turing with ComfyUI >= 0.37),
4-step distills, ~0.4 MP (480p-class). No SageAttention 2/3, no Veda, no FlashAttention 3."""
from __future__ import annotations
from .base import Provider, ProviderError

LAUNCH_FLAGS = ['--windows-standalone-build', '--fp16-unet', '--disable-dynamic-vram', '--use-ck-attention',
                '--disable-pinned-memory']
OPTIONAL_FLAGS = {'--disable-mmap': 'add if loading crashes with an access violation; pre-warm big .safetensors first'}
CUSTOM_NODES = ['ComfyUI-KJNodes', 'ComfyUI-Viggle-Animate-H3 (Comfy Registry)', 'ComfyUI-VideoHelperSuite', 'ComfyUI-GGUF',
                'star7code/minimax-h3-chunk-star7', 'star7code/minimax-h3-fp16-exact-star7', 'comfyui_controlnet_aux',
                'ComfyUI-RMBG (SAM 3.1)', 'ComfyUI-Frame-Interpolation (RIFE)']
# What the Omen is good for, best first, with the research estimates per 5 s clip at ~0.4 MP [U = untested on a 2060]
JOBS = {
    'viggle_animate': {'est_minutes': (3, 6), 'note': 'best local engine: no text encoder (~15 GB less to load)'},
    'h3_ref2va':      {'est_minutes': (3, 8), 'note': 'w4a8 + PDD 4 steps; 2080 Ti does it in ~85 s'},
    'ltx_restyle':    {'est_minutes': (4, 10), 'note': 'LTX-2.3 int8 convrot + Gemma GGUF Q4'},
    'scail2':         {'est_minutes': (10, 25), 'note': 'GGUF Q4_K_M + block swap; overnight for 720p'},
    'prep_qc':        {'est_minutes': (0, 1), 'note': 'SAM 3.1, pose, depth, scene split, whisper, ArcFace QC: do ALL prep here'},
}
SUPPORTED = ('viggle_animate', 'h3_ref2va', 'ltx_restyle', 'scail2', 'wan_animate2', 'x2_upscale')
BLOCKED = {'veda': 'sm80+ only', 'sageattention': 'not for H3 on Turing', 'nvfp4/fp8 compute': 'Turing has no fp8/fp4 tensor cores',
           'hyperflow8': 'needs bf16-class compute for the two-time path; use pdd4', 'flashvsr_upscale': 'Triton sparse kernels untested on sm75'}


class LocalOmenProvider(Provider):
    name = 'local_omen'

    def quote(self, gpu_seconds: float, gpu_id: str | None = None, fixed_usd: float = 0.0) -> dict:
        return {'provider': self.name, 'gpu': 'rtx2060', 'usd_per_hr': 0.0, 'gpu_seconds': int(gpu_seconds), 'usd': 0.0,
                'basis': 'own hardware'}

    def job_profile(self, recipe: str) -> dict:
        if recipe not in SUPPORTED:
            raise ProviderError(f'{recipe} is not Turing-safe on the Omen ({BLOCKED.get(recipe, "not in the local plan")})')
        return {'gpu_profile': 'omen', 'tier': 'draft', 'max_megapixels': 0.4, 'accel': 'pdd4', 'veda': False,
                'dtype': 'fp16 compute, int8/w4a8 weights', 'estimate': JOBS.get(recipe)}

    def plan(self) -> dict:
        c = self.cfg
        return {'status': 'PLAN ONLY (runs on the owner\'s PC when he starts it)', 'host': c.get('host'), 'gpu': c.get('gpu'),
                'install': [f'ComfyUI portable >= v0.42.5 -> {c.get("comfy_dir", "E:/ComfyUI")}',
                            f'models -> {c.get("models_dir", "E:/models")} (pagefile on E:, 32-48 GB)',
                            'download manifest bundle "omen" (python worker/bootstrap_volume.py --group omen --root E:/)'],
                'launch': 'python main.py ' + ' '.join(LAUNCH_FLAGS), 'optional_flags': OPTIONAL_FLAGS,
                'custom_nodes': CUSTOM_NODES, 'jobs': JOBS, 'blocked': BLOCKED,
                'first_test': 'Viggle pruned int8 + DMD r64, 5 s 480x832 driving clip, 4 steps: time it, watch VRAM/RAM/pagefile'}

    def launch(self, *a, **k):
        raise ProviderError('local_omen is plan-only: the owner starts jobs on his own PC')
