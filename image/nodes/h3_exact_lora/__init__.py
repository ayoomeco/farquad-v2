"""H3 exact LoRA stack + load report (Oct 25 2026, for ComfyUI >= 0.40).
Why: core LoraLoaderBypass stores its hooks under ONE injection key ("bypass_lora"), so chaining two bypass loaders
silently drops the earlier LoRA's hooks; and diff/diff_b patches (ported turbo adaLN) go to regular weight merging.
This node applies every LoRA in ONE injection: per module out = f(x) + sum_i s_i*h_i(x) (LoRA pairs via the core
LoRAAdapter.h, dense diff/diff_b as F.linear(x, s*D, s*db) in x's dtype). It never merges or rounds a weight.
It sets comfy_force_forward on hooked modules so fused paths (int8 fc2 linear_input_act) still call forward.

r2 (Oct 27 2026) memory-lean forward, same math (see fastboot/README.md "h3_exact_lora memory"):
 - H3_LORA_INPLACE=1 (default): plain LoRA / dense items are added IN PLACE into the fresh base output
   (delta = up(down(x)); delta.mul_(scale); out.add_(delta)) instead of out = out + (delta*scale): drops 2-3 full-size
   activation temporaries per hooked module. Elementwise ops are the same ops in the same order -> bitwise identical.
   Falls back to the exact v1 path when the base output is not a fresh plain tensor (view / grad / subclass) or when
   a hook carries an adapter type other than plain linear LoRA / dense diff.
 - H3_LORA_CHUNK_TOKENS=N (default 0 = off): LoRA deltas are computed N tokens at a time (sequence dim), and every
   DiT MLP (fc1 -> swiglu -> fc2) runs N tokens at a time into one preallocated output (H3_MLP_CHUNK=0 keeps the
   MLP whole). Rows are independent (int8 activations are quantized per row), so the result is the same math;
   on GPU a GEMM with a different row count may pick another kernel, so bits can differ at bf16 rounding level."""
import json, logging, os, time
import torch
import torch.nn.functional as F
import folder_paths, comfy.lora, comfy.lora_convert, comfy.utils, comfy.model_management
from comfy.patcher_extension import PatcherInjection
from comfy.weight_adapter import WeightAdapterBase
try:
    from comfy.quant_ops import QuantizedTensor
except Exception:
    QuantizedTensor = ()
try:
    from comfy.weight_adapter.lora import LoRAAdapter as _LoRAAdapter
except Exception:
    _LoRAAdapter = None
INJ_KEY = "h3_exact_stack"

def _env_int(n, d):
    try: return max(0, int(os.environ.get(n, d)))
    except ValueError: return d

def mem_config():
    return {"inplace": os.environ.get("H3_LORA_INPLACE", "1") != "0", "chunk_tokens": _env_int("H3_LORA_CHUNK_TOKENS", 0),
            "mlp_chunk": os.environ.get("H3_MLP_CHUNK", "1") != "0"}

def _lora_parts(ad):
    """(up, down, scale) iff ad is a core LoRAAdapter whose h() is exactly F.linear(F.linear(x, down), up) * scale."""
    if _LoRAAdapter is None or type(ad) is not _LoRAAdapter or getattr(ad, "is_conv", False): return None
    v = ad.weights
    if len(v) < 4 or v[3] is not None or not isinstance(v[0], torch.Tensor) or v[0].dim() != 2 or v[1].dim() != 2: return None
    up, down, alpha = v[0], v[1], v[2]
    rank = down.shape[0]
    scale = alpha / rank if alpha is not None else 1.0      # same expression order as LoRAAdapter.h
    return up, down, scale * getattr(ad, "multiplier", 1.0)

def _fresh(base, x):
    """True when base is a fresh plain tensor we may mutate (not a view, no autograd, no storage shared with x)."""
    try:
        if type(base) is not torch.Tensor or base._base is not None: return False
        if base.requires_grad and torch.is_grad_enabled(): return False
        if base.is_inference() and not torch.is_inference_mode_enabled(): return False
        return base.untyped_storage().data_ptr() != x.untyped_storage().data_ptr()
    except Exception:
        return False

def _add_delta(o, ad, x):
    """o += h(x) in place, same ops as the v1 `out + ad.h(x, base).to(base.dtype)` path."""
    p = _lora_parts(ad)
    if p is not None:
        up, down, scale = p
        d = F.linear(F.linear(x, down.to(dtype=x.dtype)), up.to(dtype=x.dtype))
        d.mul_(scale)
    elif ad.D is None:
        d = (ad.db.to(x.device, x.dtype) * ad.s)          # broadcast add == v1 expand_as add
    else:
        d = ad.h(x, o)
    o.add_(d.to(o.dtype)); del d

class _Dense:
    def __init__(self, name, D, db, s): self.name, self.D, self.db, self.s = name, D, db, s
    def h(self, x, base_out):
        b = None if self.db is None else self.db.to(x.device, x.dtype) * self.s
        if self.D is None: return b.expand_as(base_out)
        return F.linear(x, self.D.to(x.device, x.dtype) * self.s, b)

class _Hook:
    def __init__(self, module, key):
        self.m, self.key, self.items, self.orig, self.calls = module, key, [], None, 0
    fast, chunk = False, 0
    def configure(self, cfg):
        self.fast = cfg["inplace"] and all(_lora_parts(ad) is not None or isinstance(ad, _Dense) for _, ad in self.items)
        self.chunk = cfg["chunk_tokens"]
    def fwd_v1(self, x, base):
        out = base
        for name, ad in self.items:
            out = out + ad.h(x, base).to(base.dtype)
        return out
    def fwd(self, x, *a, **k):
        base = self.orig(x, *a, **k); self.calls += 1
        if not self.fast or not _fresh(base, x): return self.fwd_v1(x, base)
        C, sd = self.chunk, x.dim() - 2
        if C and sd >= 0 and base.dim() == x.dim() and base.shape[:-1] == x.shape[:-1] and x.shape[sd] > C:
            n = x.shape[sd]
            for name, ad in self.items:
                for s in range(0, n, C):
                    L = min(C, n - s)
                    _add_delta(base.narrow(sd, s, L), ad, x.narrow(sd, s, L))
            return base
        for name, ad in self.items:
            _add_delta(base, ad, x)
        return base
    def inject(self):
        if self.orig is not None: return
        dev = comfy.model_management.get_torch_device()
        for name, ad in self.items:
            if isinstance(ad, WeightAdapterBase):
                ad.weights = tuple(w.to(dev) if isinstance(w, torch.Tensor) else w for w in ad.weights)
            else:
                ad.D = None if ad.D is None else ad.D.to(dev); ad.db = None if ad.db is None else ad.db.to(dev)
        self.orig = self.m.forward; self.ff = getattr(self.m, "comfy_force_forward", False)
        self.m.comfy_force_forward = True; self.m.forward = self.fwd
    def eject(self):
        if self.orig is None: return
        self.m.forward = self.orig; self.m.comfy_force_forward = self.ff; self.orig = None

class _MlpChunk:
    """Run a token-wise MLP (fc1 -> act -> fc2 on [S, C]) C tokens at a time into one preallocated output."""
    def __init__(self, m, C): self.m, self.C, self.orig, self.calls = m, C, None, 0
    def fwd(self, x, *a, **k):
        C = self.C
        if a or k or not isinstance(x, torch.Tensor) or x.dim() != 2 or x.shape[0] <= C: return self.orig(x, *a, **k)
        n, out = x.shape[0], None
        for s in range(0, n, C):
            y = self.orig(x[s:s + C]); L = min(C, n - s)
            if out is None:
                if type(y) is not torch.Tensor or y.dim() != 2 or y.shape[0] != L: return self.orig(x)
                out = y.new_empty((n, y.shape[1]))
            out[s:s + L].copy_(y); del y
        self.calls += 1
        return out
    def inject(self):
        if self.orig is not None: return
        self.orig = self.m.forward; self.m.forward = self.fwd
    def eject(self):
        if self.orig is None: return
        self.m.forward = self.orig; self.orig = None

def _get_module(root, path):
    m = root
    for p in path.split("."):
        m = m[int(p)] if p.isdigit() else getattr(m, p)
    return m

class _Catch(logging.Handler):
    def __init__(self): super().__init__(); self.msgs = []
    def emit(self, r):
        t = r.getMessage()
        if "lora key not loaded" in t or "ERROR" in t: self.msgs.append(t[:300])

def apply_stack(patcher, entries, key_map=None):
    """entries: list of (name, state_dict, strength). Returns (new_patcher, report dict)."""
    p = patcher.clone()
    if key_map is None: key_map = comfy.lora.model_lora_keys_unet(p.model, {})
    root = p.model
    hooks, rep = {}, {"loras": [], "hooks_by_type": {}, "regular_merges": 0}
    for name, sd, s in entries:
        if not s: continue
        c = _Catch(); lg = logging.getLogger(); lg.addHandler(c)
        try:
            loaded = comfy.lora.load_lora(comfy.lora_convert.convert_lora(sd), key_map)
        finally:
            lg.removeHandler(c)
        n_ad = n_dense = 0; bad = []
        dense = {}
        for key, pd in loaded.items():
            if isinstance(pd, WeightAdapterBase):
                pd.multiplier = float(s)
                mk = key[:-7] if key.endswith(".weight") else key
                hooks.setdefault(mk, None); hooks[mk] = hooks[mk] or _Hook(_get_module(root, mk), mk)
                hooks[mk].items.append((name, pd)); n_ad += 1
            elif isinstance(pd, tuple) and pd[0] == "diff":
                if key.endswith(".weight"): dense.setdefault(key[:-7], [None, None])[0] = pd[1][0]
                elif key.endswith(".bias"): dense.setdefault(key[:-5], [None, None])[1] = pd[1][0]
                else: bad.append(key)
            else: bad.append(f"{key}:{type(pd).__name__}")
        for mk, (D, db) in dense.items():
            hooks.setdefault(mk, None); hooks[mk] = hooks[mk] or _Hook(_get_module(root, mk), mk)
            hooks[mk].items.append((name, _Dense(name, None if D is None else D.float(), None if db is None else db.float(), float(s)))); n_dense += 1
        n_tensors = len(sd); 
        rep["loras"].append({"name": name, "strength": s, "tensors": n_tensors, "adapters": n_ad, "dense_modules": n_dense,
                             "unsupported": bad, "load_warnings": c.msgs[:20], "n_load_warnings": len(c.msgs)})
    for mk, hk in hooks.items():
        t = next((x for x in ("qkv_proj", "out_proj", "fc1", "fc2", "adaln_proj") if x in mk), "other")
        w = getattr(hk.m, "weight", None)
        q = "int8/quant" if isinstance(w, QuantizedTensor) and QuantizedTensor else str(getattr(w, "dtype", None))
        d = rep["hooks_by_type"].setdefault(t, {}); d[q] = d.get(q, 0) + len(hk.items)
    hl = list(hooks.values()); cfg = mem_config()
    for h in hl: h.configure(cfg)
    mw = ()   # tuple, not list: _probe counts list cells as hooks
    if cfg["chunk_tokens"] and cfg["mlp_chunk"]:
        mw = tuple(_MlpChunk(m, cfg["chunk_tokens"]) for n, m in root.named_modules()
                   if "diffusion_model" in n and type(m).__name__ == "MLP" and hasattr(m, "fc1") and hasattr(m, "fc2"))
    p.set_injections(INJ_KEY, [PatcherInjection(inject=lambda mp: [h.inject() for h in hl] + [w.inject() for w in mw],
                                                eject=lambda mp: [w.eject() for w in mw] + [h.eject() for h in hl])])
    rep["modules_hooked"] = len(hooks); rep["adapter_items"] = sum(len(h.items) for h in hl)
    rep["mem"] = dict(cfg, fast_hooks=sum(1 for h in hl if h.fast), v1_hooks=sum(1 for h in hl if not h.fast), mlp_chunked=len(mw))
    return p, rep

def _probe(model):
    """Inspect a patcher: injections + hook counts (core manager via closure), regular merges and their target dtypes."""
    r = {"injection_keys": {}, "regular_patches": {}, "lossy_merges": 0}
    for k, injs in model.injections.items():
        n = 0
        for inj in injs:
            for cell in (inj.inject.__closure__ or ()):
                obj = cell.cell_contents
                if hasattr(obj, "hooks"): n += len(obj.hooks)
                elif isinstance(obj, list): n += sum(len(getattr(h, "items", [1])) for h in obj)
        r["injection_keys"][k] = n
    sd = None
    for key in model.patches:
        try:
            w = comfy.utils.get_attr(model.model, key)
        except Exception:
            w = None
        q = "int8/quant" if (QuantizedTensor and isinstance(w, QuantizedTensor)) else str(getattr(w, "dtype", None))
        t = next((x for x in ("qkv_proj", "out_proj", "fc1", "fc2", "adaln_proj") if x in key), "other")
        r["regular_patches"].setdefault(f"{t}|{q}", 0); r["regular_patches"][f"{t}|{q}"] += 1
        if q != "torch.float32": r["lossy_merges"] += 1
    try:
        w = comfy.utils.get_attr(model.model, "diffusion_model.blocks.0.adaln_proj.linear.weight"); r["adaln_weight_dtype"] = str(w.dtype)
        w = comfy.utils.get_attr(model.model, "diffusion_model.blocks.0.mlp.fc2.weight"); r["fc2_weight_type"] = type(w).__name__ + "/" + str(getattr(w, "_layout_cls", w.dtype))
    except Exception as e:
        r["probe_err"] = str(e)[:200]
    r["wrappers"] = sorted(str(k) for k in (model.model_options.get("transformer_options", {}).get("wrappers", {}) or {}).keys())
    return r

class H3ExactLoraStack:
    @classmethod
    def INPUT_TYPES(s):
        L = ["None"] + folder_paths.get_filename_list("loras")
        req = {"model": ("MODEL",)}
        for i in range(1, 6):
            req[f"lora_{i}"] = (L,); req[f"strength_{i}"] = ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01})
        return {"required": req}
    RETURN_TYPES = ("MODEL", "STRING"); FUNCTION = "go"; CATEGORY = "loaders/h3"
    def go(self, model, **kw):
        entries = []
        for i in range(1, 6):
            n, s = kw[f"lora_{i}"], kw[f"strength_{i}"]
            if n != "None" and s:
                entries.append((n, comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("loras", n), safe_load=True), s))
        p, rep = apply_stack(model, entries)
        p.model_options["h3_exact_report"] = rep
        logging.info("[H3ExactLoraStack] " + json.dumps(rep))
        return (p, json.dumps(rep))

class H3LoadReport:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL",), "tag": ("STRING", {"default": "arm"}), "require": ("STRING", {"default": "{}", "multiline": True})}, "optional": {"note": ("STRING", {"forceInput": True})}}
    RETURN_TYPES = ("MODEL",); FUNCTION = "go"; CATEGORY = "loaders/h3"; OUTPUT_NODE = True
    def go(self, model, tag, require, note=None):
        import comfyui_version
        r = {"tag": tag, "t": time.strftime("%F %T"), "comfyui": comfyui_version.__version__, "probe": _probe(model),
             "exact_stack": model.model_options.get("h3_exact_report"), "note": note}
        req = json.loads(require or "{}"); fails = []
        if "max_lossy_merges" in req and r["probe"]["lossy_merges"] > req["max_lossy_merges"]: fails.append(f"lossy merges {r['probe']['lossy_merges']}")
        es = r["exact_stack"] or {}
        loaded = {}
        for lo in es.get("loras", []):
            loaded[lo["name"]] = lo
            if req.get("no_load_warnings") and (lo["n_load_warnings"] or lo["unsupported"]): fails.append(f"{lo['name']} warnings {lo['load_warnings'][:3]} {lo['unsupported'][:3]}")
            exp = req.get("expect", {}).get(lo["name"])
            if exp and (lo["adapters"], lo["dense_modules"]) != tuple(exp): fails.append(f"{lo['name']} adapters/dense {lo['adapters']}/{lo['dense_modules']} != {exp}")
        for name in (req.get("expect") or {}):
            if name not in loaded: fails.append(f"{name} not loaded")
        for name, exp_s in (req.get("strengths") or {}).items():
            lo = loaded.get(name)
            if lo is None: fails.append(f"{name} not loaded")
            elif abs(float(lo["strength"]) - float(exp_s)) > 1e-6: fails.append(f"{name} strength {lo['strength']} != {exp_s}")
        for t, n in (req.get("min_hooks") or {}).items():
            got = sum((es.get("hooks_by_type", {}).get(t) or {}).values())
            if got < n: fails.append(f"hooks {t} {got} < {n}")
        r["fails"] = fails; r["pass"] = not fails
        d = os.path.join(folder_paths.get_output_directory(), "h3check"); os.makedirs(d, exist_ok=True)
        open(os.path.join(d, f"{tag}.json"), "w").write(json.dumps(r, indent=1))
        logging.info(f"[H3LoadReport] {tag} pass={r['pass']} {json.dumps(r)[:3000]}")
        if fails and req.get("abort", True): raise RuntimeError(f"H3 load check FAILED for {tag}: {fails}")
        return (model,)

NODE_CLASS_MAPPINGS = {"H3ExactLoraStack": H3ExactLoraStack, "H3LoadReport": H3LoadReport}
NODE_DISPLAY_NAME_MAPPINGS = {"H3ExactLoraStack": "H3 Exact LoRA Stack (no merge)", "H3LoadReport": "H3 Load Report"}
