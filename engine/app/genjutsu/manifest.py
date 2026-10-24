"""Model manifest (config/models.manifest.json): which weights a graph needs, sizes, bundles."""
from __future__ import annotations
import json, functools
from .. import config

MANIFEST = config.PROJECT / 'config' / 'models.manifest.json'


@functools.lru_cache(maxsize=1)
def load() -> dict:
    return json.loads(MANIFEST.read_text())


def by_filename() -> dict[str, dict]:
    return {m['target'].split('/')[-1]: m for m in load()['models']}


# Nodes that load weights from a fixed folder instead of a filename widget -> manifest bundle they need
CLASS_BUNDLES = {'FlashVSRNode': 'flashvsr'}


def required_for(graph: dict) -> dict:
    """Map every model filename a graph references to a manifest entry. Unknown names are returned as missing."""
    from .graph import Graph
    idx = by_filename()
    have, missing = {}, []
    for nid, inp, fn in Graph(graph).model_refs():
        base = fn.split('/')[-1]
        if base in idx:
            have[base] = idx[base]
        elif not base.lower().endswith(('.mp4', '.png', '.jpg', '.jpeg', '.webp', '.mov')):
            missing.append(f'{nid}.{inp}={fn}')
    for n in graph.values():
        b = CLASS_BUNDLES.get(n.get('class_type'))
        if b:
            for m in load()['models']:
                if b in m['bundles']:
                    have[m['target'].split('/')[-1]] = m
    return {'models': sorted(have), 'bytes': sum(m['size_bytes'] for m in have.values()),
            'bundles': sorted({b for m in have.values() for b in m['bundles']}), 'missing': missing}


def group_bytes(group: str) -> int:
    return int(load()['group_bytes'][group])
