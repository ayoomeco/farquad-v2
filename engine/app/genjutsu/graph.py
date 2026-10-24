"""ComfyUI API-graph utilities: load templates, bind inputs, inject/remove nodes, and TYPE-CHECK the result
against an object_info snapshot (offline, free). This is what lets us catch broken graphs on the box instead
of on a rented GPU."""
from __future__ import annotations
import copy, json, pathlib
from .. import config

WF_DIR = config.PROJECT / 'config' / 'workflows'
API_DIR = WF_DIR / 'api'
SNAPSHOT = WF_DIR / 'object_info.snapshot.json'


class GraphError(Exception):
    pass


def load_api(name: str) -> dict:
    p = API_DIR / f'{name}.api.json'
    if not p.exists():
        raise GraphError(f'API workflow {name!r} not found ({p})')
    return json.loads(p.read_text())


_OI: dict | None = None


def object_info() -> dict:
    global _OI
    if _OI is None:
        _OI = json.loads(SNAPSHOT.read_text()) if SNAPSHOT.exists() else {}
    return _OI


def is_link(v) -> bool:
    return isinstance(v, list) and len(v) == 2 and isinstance(v[0], str) and isinstance(v[1], int)


class Graph:
    """Thin mutable wrapper over an API prompt dict."""

    def __init__(self, nodes: dict):
        self.nodes = copy.deepcopy(nodes)
        self._next = 9000

    @classmethod
    def from_template(cls, name: str) -> 'Graph':
        return cls(load_api(name))

    def node(self, nid: str) -> dict:
        if nid not in self.nodes:
            raise GraphError(f'node {nid} not in graph')
        return self.nodes[nid]

    def set(self, nid: str, inp: str, value):
        n = self.node(nid)
        n.setdefault('inputs', {})[inp] = value
        return self

    def get(self, nid: str, inp: str, default=None):
        return self.node(nid).get('inputs', {}).get(inp, default)

    def add(self, class_type: str, inputs: dict, title: str | None = None, nid: str | None = None) -> str:
        while nid is None or nid in self.nodes:
            nid = str(self._next); self._next += 1
        self.nodes[nid] = {'class_type': class_type, 'inputs': dict(inputs), '_meta': {'title': title or class_type}}
        return nid

    def find(self, class_type: str | None = None, title: str | None = None) -> list[str]:
        return [k for k, n in self.nodes.items() if (class_type is None or n['class_type'] == class_type)
                and (title is None or title.lower() in str((n.get('_meta') or {}).get('title', '')).lower())]

    def remove(self, nids) -> 'Graph':
        for nid in list(nids):
            self.nodes.pop(nid, None)
        return self

    def remove_prefix(self, prefix: str) -> 'Graph':
        return self.remove([k for k in self.nodes if k.startswith(prefix)])

    def unlink_dangling_optional(self):
        """Drop OPTIONAL inputs that point at removed nodes (required ones are reported by validate())."""
        oi = object_info()
        for nid, n in self.nodes.items():
            opt = set(((oi.get(n['class_type']) or {}).get('input') or {}).get('optional', {}) or {})
            for k, v in list(n.get('inputs', {}).items()):
                if is_link(v) and v[0] not in self.nodes and (k in opt or k.split('.')[0] in opt):
                    del n['inputs'][k]

    def prune_unreachable(self):
        """Remove nodes that no output node depends on (keeps graphs lean after surgery)."""
        oi = object_info()
        outs = [k for k, n in self.nodes.items() if (oi.get(n['class_type']) or {}).get('output_node')]
        keep, stack = set(), list(outs)
        while stack:
            k = stack.pop()
            if k in keep or k not in self.nodes:
                continue
            keep.add(k)
            stack += [v[0] for v in self.nodes[k].get('inputs', {}).values() if is_link(v)]
        if outs:
            self.remove([k for k in self.nodes if k not in keep])

    def rename_models(self, mapping: dict[str, str]):
        for n in self.nodes.values():
            for k, v in n.get('inputs', {}).items():
                if isinstance(v, str) and v in mapping:
                    n['inputs'][k] = mapping[v]

    def model_refs(self) -> list[tuple[str, str, str]]:
        """(node, input, filename) for every *.safetensors/ckpt/pt string input."""
        out = []
        for nid, n in self.nodes.items():
            for k, v in n.get('inputs', {}).items():
                if isinstance(v, str) and v.lower().endswith(('.safetensors', '.ckpt', '.pt', '.pth', '.onnx', '.gguf')):
                    out.append((nid, k, v))
        return out

    def to_json(self) -> dict:
        return copy.deepcopy(self.nodes)


def _input_spec(oi_node: dict, name: str):
    inp = oi_node.get('input') or {}
    for sect in ('required', 'optional'):
        d = inp.get(sect) or {}
        if name in d:
            return sect, d[name]
        base = name.split('.')[0]        # autogrow / dynamic-combo sub-inputs: ref_images.ref_image_0, resize_type.width
        if base in d:
            spec = d[base]
            t = spec[0] if isinstance(spec, list) else spec
            if t == 'COMFY_AUTOGROW_V3':
                tmpl = (spec[1] or {}).get('template', {}).get('input', {})
                for s2 in tmpl.values():
                    for v in (s2 or {}).values():
                        return sect, v
            return sect, None
    return None, None


def _types_ok(out_t: str, in_t) -> bool:
    if in_t is None:
        return True
    want = in_t[0] if isinstance(in_t, list) else in_t
    if isinstance(want, list):            # combo: only literal values allowed, links from primitives are fine
        return True
    if want in ('*', 'COMFY_MATCHTYPE_V3') or out_t in ('*', 'COMFY_MATCHTYPE_V3'):
        return True
    wants = set(str(want).split(','))
    return out_t in wants or str(want).startswith('COMFY_') or out_t.startswith('COMFY_')


def validate(nodes: dict, oi: dict | None = None, check_files: bool = False) -> list[str]:
    """Return problems: unknown class types, dangling links, link type mismatches, missing required inputs,
    combo values outside the allowed list (model filenames are skipped unless check_files)."""
    oi = oi if oi is not None else object_info()
    probs = []
    if not oi:
        return ['no object_info snapshot available']
    for nid, n in nodes.items():
        ct = n.get('class_type')
        spec = oi.get(ct)
        if not spec:
            probs.append(f'{nid}: unknown class_type {ct!r}'); continue
        ins = n.get('inputs', {})
        for k, v in ins.items():
            sect, ispec = _input_spec(spec, k)
            if is_link(v):
                src = nodes.get(v[0])
                if not src:
                    probs.append(f'{nid}.{k}: link to missing node {v[0]}'); continue
                souts = (oi.get(src['class_type']) or {}).get('output') or []
                if v[1] >= len(souts):
                    probs.append(f'{nid}.{k}: output index {v[1]} out of range for {src["class_type"]}'); continue
                if not _types_ok(souts[v[1]], ispec):
                    probs.append(f'{nid}.{k}: type {souts[v[1]]} -> {ispec[0] if isinstance(ispec, list) else ispec}')
            elif ispec is not None and isinstance(ispec, list) and ispec and isinstance(ispec[0], list):
                allowed = ispec[0]
                is_file = isinstance(v, str) and v.lower().endswith(('.safetensors', '.ckpt', '.pt', '.pth', '.onnx', '.mp4', '.png', '.jpg', '.jpeg', '.webp', '.avif', '.mov', '.wav'))
                if allowed and v not in allowed and (check_files or not is_file):
                    probs.append(f'{nid}.{k}: value {v!r} not in allowed options')
            elif ispec is not None and isinstance(ispec, list) and ispec and ispec[0] == 'COMBO':
                opts = (ispec[1] or {}).get('options') if len(ispec) > 1 else None
                is_file = isinstance(v, str) and '.' in v[-6:]
                if opts and v not in opts and (check_files or not is_file):
                    probs.append(f'{nid}.{k}: value {v!r} not in allowed options')
        req = (spec.get('input') or {}).get('required') or {}
        for r, rspec in req.items():
            t = rspec[0] if isinstance(rspec, list) else rspec
            if r not in ins and not any(x.startswith(r + '.') for x in ins) and t not in ('COMFY_DYNAMICCOMBO_V3', 'COMFY_AUTOGROW_V3'):
                probs.append(f'{nid}: missing required input {r!r} ({ct})')
    return probs


def validate_strict(nodes: dict, oi: dict | None = None) -> list[str]:
    """validate() + literal scalar checks (INT/FLOAT/BOOLEAN/STRING types and min/max), the class of bug UI->API
    conversion produces when a node pack's widget order changed (e.g. MochaEmbeds force_offload=832)."""
    oi = oi if oi is not None else object_info()
    probs = validate(nodes, oi)
    for nid, n in nodes.items():
        spec = oi.get(n.get('class_type')) or {}
        for k, v in (n.get('inputs') or {}).items():
            if is_link(v):
                continue
            _, ispec = _input_spec(spec, k)
            if not isinstance(ispec, list) or not ispec or not isinstance(ispec[0], str):
                continue
            t, opt = ispec[0], (ispec[1] if len(ispec) > 1 and isinstance(ispec[1], dict) else {})
            bad = ((t == 'INT' and (isinstance(v, bool) or not isinstance(v, int))) or
                   (t == 'FLOAT' and (isinstance(v, bool) or not isinstance(v, (int, float)))) or
                   (t == 'BOOLEAN' and not isinstance(v, bool)) or (t == 'STRING' and not isinstance(v, str)))
            if bad:
                probs.append(f'{nid}.{k}: literal {v!r} is not {t}'); continue
            if t in ('INT', 'FLOAT'):
                if 'min' in opt and v < opt['min']:
                    probs.append(f'{nid}.{k}: {v} < min {opt["min"]}')
                if 'max' in opt and v > opt['max']:
                    probs.append(f'{nid}.{k}: {v} > max {opt["max"]}')
    return probs
