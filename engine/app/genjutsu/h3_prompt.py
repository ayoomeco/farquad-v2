"""H3 Context-IR prompt builder for Ref2VA (MiniMax H3 model card: "H3-Context-IR is critical to the quality of the final
output"). Emits the Ref2VA sections in card order:
    subject_definitions / summary / retention_analysis / detailed_description / overall_soundscape / non_diegetic_music
Tags and retention labels follow the card's Ref2VA example ([video editing + ...], fully_preserved / partially_copy /
reference). Deterministic and offline; the Director (or the owner) can supply `description` (the [Shot 1] narrative) and the
character pack supplies the identity lock. Nothing here is engine-specific beyond H3's tag vocabulary."""
from __future__ import annotations
from .shotspec import ShotSpec

SUBJECT_NOUN = {'human': 'the person', 'animal': 'the animal', 'creature': 'the creature', 'object': 'the object', 'any': 'the subject'}


def _pics(n: int, start: int = 1) -> str:
    tags = [f'<Picture {i}>' for i in range(start, start + n)]
    return tags[0] if n == 1 else ', '.join(tags[:-1]) + f' and {tags[-1]}'


def context_ir(spec: ShotSpec, n_pictures: int, seconds: float, description: str | None = None,
               identity_lock: str | None = None, music: str | None = None, n_style: int = 0, n_background: int = 0,
               ref_roles: list[str] | None = None) -> str:
    """Pictures are numbered in upload order: subject pictures first, then style, then background (as h3_ref2va wires them)."""
    n = max(1, int(n_pictures))
    noun = SUBJECT_NOUN.get(spec.subject, 'the subject')
    who = (spec.prompt or '').strip()
    audio = bool(spec.keep_audio)
    secs = f'{seconds:g}'
    lines = ['subject_definitions:']
    if spec.mode == 'restyle':
        lines.append(f'<Video 1> is the source video for the editing task; it provides the whole scene, motion, timing, framing and camera.')
        if spec.refs_by('style'):
            lines.append(f'{_pics(n)} define{"s" if n == 1 else ""} the target visual style.')
    else:
        ident = f' {identity_lock.strip()}' if identity_lock else ''
        lines.append(f'<Subject 1> is {noun} shown in {_pics(n)}: face, identity, hair, body shape and outfit come from '
                     f'{"this picture" if n == 1 else "these pictures"}.{ident}')
        if n_style:
            lines.append(f'{_pics(n_style, n + 1)} define{"s" if n_style == 1 else ""} the target visual style.')
        if n_background:
            lines.append(f'{_pics(n_background, n + n_style + 1)} show{"s" if n_background == 1 else ""} the target background.')
        if spec.mode == 'character_swap':
            lines.append('<Video 1> is the source video for the editing task. The person in <Video 1> is replaced by <Subject 1>; '
                         'none of their identity is kept.')
            if ref_roles:
                named = ' '.join(f'image {i} {role}' for i, role in enumerate(ref_roles, 1))
                lines.append(f'Reference roles: {named}. The pictures are separate references, never one combined character sheet.')
            lines.append("Match the source's mouth and expression exactly.")
            lines.append('keep any objects already in the scene; add nothing.')
        else:
            lines.append('<Video 1> is the source video that provides the whole-video motion: body pose, head movement, gaze, mouth '
                         'movement, hand movement, camera framing and timing. Only its motion is used; none of its people\'s identity is kept.')
    if audio:
        lines.append('<Audio 1> is the synchronized audio track of <Video 1>.')
    # ---- summary
    if spec.mode == 'restyle':
        tag, body = 'video editing', (f'The target video is <Video 1> re-rendered in {spec.style or "the reference"} style, keeping the '
                                      'exact motion, timing, framing and camera of <Video 1>.')
    elif spec.mode == 'character_swap':
        tag, body = 'video editing', ('The target video is an edited version of <Video 1> in which <Subject 1> takes the place of the '
                                      'person, following <Video 1> moment by moment.')
    else:
        tag, body = 'reference generation', (f'One continuous {secs}-second shot of <Subject 1> following <Video 1> moment by moment.')
    if audio:
        tag += ' + audio reuse'
    lines += ['', 'summary:', f'[{tag}] {body}' + (f' {who}' if who and spec.mode != 'restyle' else '')]
    # ---- retention_analysis
    lines += ['', 'retention_analysis:']
    if spec.mode == 'restyle':
        lines.append('<Video 1> (source video editing): partially_copy - motion, timing, framing, camera and composition are kept; '
                     'the look is replaced by the target style.')
    else:
        lines.append('<Subject 1> (appears in [Shot 1]): reference - identity, face, hair, body shape and outfit come from the pictures.')
        if spec.mode == 'character_swap':
            lines.append('<Video 1> (source video editing): fully_preserved - framing, lighting, setting, other people and timing are kept '
                         'while the person is replaced.')
        else:
            lines.append('<Video 1>: reference - only the motion, timing, framing and camera movement are referenced.')
    if audio:
        lines.append('<Audio 1>: fully_preserved - the original speech and room sound are kept unchanged.')
    # ---- detailed_description
    style = 'in realistic photographic style' if spec.mode != 'restyle' else f'in {spec.style or "the reference"} style'
    shot = (description or '').strip() or (
        f'[Shot 1] <Subject 1> performs exactly the motion of <Video 1> for all {secs} seconds, with the same framing and camera '
        'movement. The face stays clearly visible and consistent.' if spec.mode != 'restyle' else
        f'[Shot 1] The scene of <Video 1>, every movement kept, rendered {style}.')
    if not shot.startswith('[Shot'):
        shot = '[Shot 1] ' + shot
    lines += ['', 'detailed_description:', f'The target video is {style}.', shot]
    lines += ['', 'overall_soundscape:', 'Keep the speech and room sound of <Video 1> (<Audio 1>) unchanged.' if audio
              else 'Natural ambient sound matching the scene.']
    lines += ['', 'non_diegetic_music:', music or 'None added.']
    return '\n'.join(lines)
