"""Likeness consent gate (stub). A shot that puts a real person's likeness from user-supplied reference images onto
footage (motion transfer / character swap) needs an explicit "I have the rights to this likeness" before any billable
dispatch: either ShotSpec.likeness_consent=True for this shot, or a character pack whose consent.likeness_rights is true.
Planning / quoting stays free and is never blocked; only dispatch is (require())."""
from __future__ import annotations
from .. import config
from .shotspec import ShotSpec

LIKENESS_MODES = ('motion_transfer', 'character_swap')
NON_HUMAN = ('animal', 'creature', 'product', 'object')


class ConsentError(Exception):
    pass


def check(spec: ShotSpec) -> dict:
    subj = spec.refs_by('subject')
    pack = config.characters().get(spec.character_pack or '') or {}
    required = spec.mode in LIKENESS_MODES and (bool(subj) or bool(pack)) and spec.subject not in NON_HUMAN
    if not required:
        return {'required': False, 'ok': True, 'source': None, 'reason': 'no human likeness transfer in this shot'}
    if spec.likeness_consent:
        return {'required': True, 'ok': True, 'source': 'shot', 'reason': 'likeness_consent=true on this shot'}
    if pack and (pack.get('consent') or {}).get('likeness_rights') in (True, 'owner-character'):
        return {'required': True, 'ok': True, 'source': f"character_pack:{pack.get('id')}", 'reason': 'character pack consent on file'}
    who = f"character pack {pack.get('id')!r} has no consent.likeness_rights=true" if pack else 'no likeness_consent on this shot'
    return {'required': True, 'ok': False, 'source': None,
            'reason': f'consent needed before dispatch: {who} (confirm you have the rights to this person\'s likeness)'}


def require(spec: ShotSpec) -> dict:
    c = check(spec)
    if not c['ok']:
        raise ConsentError(c['reason'])
    return c
