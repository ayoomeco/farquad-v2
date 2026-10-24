"""ShotSpec: the one strict contract for a Genjutsu-style job (any subject, any reference).
Mirrors Higgsfield Genjutsu's three workflows (motion transfer, object swap, restyle) plus the in-place
character swap and upscale passes our open stack adds. Nothing reaches a router/engine unless it validates."""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field, model_validator

Mode = Literal['motion_transfer', 'character_swap', 'object_swap', 'restyle', 'upscale']
Subject = Literal['human', 'animal', 'object', 'creature', 'any']
Tier = Literal['draft', 'final']
Priority = Literal['likeness', 'motion', 'balanced']

MODES = ('motion_transfer', 'character_swap', 'object_swap', 'restyle', 'upscale')
# Legacy Director op words -> Genjutsu modes (so the existing planner can feed the new router)
LEGACY_OP_TO_MODE = {'recast': 'character_swap', 'transform': 'restyle', 'retexture': 'restyle',
                     'environment': 'restyle', 'insert': 'object_swap', 'remove': 'object_swap'}

GENJUTSU_LIMITS = {'min_source_seconds': 4.0, 'max_source_seconds': 30.0, 'max_refs': 30}


class Ref(BaseModel):
    path: str
    role: Literal['subject', 'object', 'style', 'background', 'audio', 'motion'] = 'subject'
    label: str = ''


class ShotSpec(BaseModel):
    mode: Mode
    source_video: str | None = None                  # driving / edited clip (path on the box or asset id)
    source_seconds: float | None = Field(default=None, gt=0)
    source_fps: float | None = Field(default=None, gt=0)
    source_size: tuple[int, int] | None = None       # (w, h)
    refs: list[Ref] = Field(default_factory=list)    # 0-30 images (subject / object / style)
    subject: Subject = 'any'
    subject_count: int = Field(default=1, ge=1, le=8)
    target: str = ''                                 # WHAT in the source to replace/track: "the man", "the phone", "dog"
    prompt: str = ''                                 # desired result in words
    style: str = ''                                  # restyle: "cel-shaded anime", "claymation" ...
    keep_audio: bool = True                          # Genjutsu Restyle keeps source audio; we default it on everywhere
    keep_background: bool = True                     # character_swap: keep the source plate
    priority: Priority = 'balanced'                  # likeness vs motion-fidelity tradeoff
    tier: Tier = 'draft'
    max_resolution: int = Field(default=1080, ge=256, le=2160)   # long-edge target after optional upscale
    seed: int = 42
    character_pack: str | None = None                # optional pack id (e.g. "pack"); never required
    engine: str | None = None                        # explicit override (router honours it if capable)
    budget_usd: float | None = Field(default=None, ge=0)
    reentry: bool = False                            # subject leaves and re-enters frame / big occlusions (Viggle weak spot)
    partial_swap: bool = False                       # replace only part of the subject (masked) -> SCAIL-2 / mask engines
    hero: bool = False                               # hero shot: quality over cost (SeedVR2 instead of FlashVSR, etc.)
    preset_id: str | None = None                     # Higgsfield-compatible restyle preset id (config/presets/restyle.json)
    # Ref2VA prepends the reference video's tokens to the sequence: resample it to this short edge (aspect kept, x32).
    # None = engine default (544 final / 384 draft); 0 = feed the clip at source size.
    motion_ref_short_edge: int | None = Field(default=None, ge=0, le=2160)
    likeness_consent: bool = False                   # "I have the rights to this person's likeness" (consent gate)

    @model_validator(mode='after')
    def _check(self):
        need_src = self.mode in ('motion_transfer', 'character_swap', 'object_swap', 'restyle', 'upscale')
        if need_src and not self.source_video:
            raise ValueError(f'{self.mode} needs a source_video')
        if len(self.refs) > GENJUTSU_LIMITS['max_refs']:
            raise ValueError('at most 30 reference images')
        subj = [r for r in self.refs if r.role == 'subject']
        if self.mode in ('motion_transfer', 'character_swap') and not subj and not self.character_pack:
            raise ValueError(f'{self.mode} needs at least one subject reference image or a character_pack')
        if self.mode == 'object_swap':
            if not self.target:
                raise ValueError('object_swap needs target (what to replace, e.g. "the phone")')
            if not any(r.role == 'object' for r in self.refs) and not self.prompt:
                raise ValueError('object_swap needs an object reference image or a prompt describing the new object')
        if self.mode == 'restyle' and not (self.style or any(r.role == 'style' for r in self.refs)):
            raise ValueError('restyle needs a style name or a style reference image')
        return self

    def refs_by(self, role: str) -> list[Ref]:
        return [r for r in self.refs if r.role == role]

    @property
    def portrait(self) -> bool:
        return bool(self.source_size and self.source_size[1] > self.source_size[0])
