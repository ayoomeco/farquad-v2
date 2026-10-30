"""GJ V2 default H3 render (approved A08).

Singularity H3 v1.3 trimmed int8, Sage Attention, 8 steps, video shift 6 (audio 3),
seed 42, ref_image_size match. LoRAs through H3ExactLoraStack, then H3LoadReport
(lossy=0, hooks present, each LoRA at its set strength) or the graph is refused.
Exactly one speed LoRA: lightx2v Minimax-h3-Turbo 8-step v1.1 at 0.8.
Realism 0.7 and character-swap v1.5 at 1.0. No turbo_v4, no pro4500, no likeness LoRA
unless GJ_LIKENESS_LORA is set, and never at 0.35 or above.
"""
import json, copy, pathlib, os, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
from home import gj_home
from refs_gjv2 import ROLE_TEXT, filenames, is_sheet, roles

VERSION = "GJ V2"
STEPS = 8
SHIFT_VIDEO = 6.0
SHIFT_AUDIO = 3.0
SEED = 42
REF_IMAGE_SIZE = "match"
UNET = "Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8.safetensors"
REALISM = ("h3-realism-people-t2v-i2v-r2v.safetensors", 0.7)
SWAP = ("h3_character_swap_v1.5.safetensors", 1.0)
SPEED = ("minimax_h3_ref2v_turbo_8step_v1.1_768p_comfyui_bf16.safetensors", 0.8)
# Adapter / dense-module counts from the approved A08 load check.
LORA_EXPECT = {
    REALISM[0]: [104, 0],
    SWAP[0]: [208, 0],
    SPEED[0]: [208, 0],
}
# Stock filenames that must not appear on a default graph.
# The Meek likeness file is not banned by name: it stays off unless GJ_LIKENESS_LORA is set,
# and likeness_slot() refuses a strength of 0.35 or above.
BANNED = (
    "minimax_h3_turbo_v4_step600_ema.safetensors",
    "h3_character_swap_pro4500_1000.safetensors",
    "LoraLoaderModelOnly",
    "LoraLoaderBypass",
)
LIKENESS_FILE = "m33k_lme_s0500.safetensors"
LEN, STEP = 141, 119

def _fh():
    return gj_home()

def _load_json(name, kind):
    fh = pathlib.Path(_fh())
    p = fh / "plan" / kind / name
    if not p.exists():
        p = pathlib.Path(__file__).resolve().parent / kind / name
    return json.load(open(p))

CLIPS = {
    "lalaland": dict(full_body=True, look="Photoreal live-action, lit like the scene.",
        subj=[("the man in the white shirt and tie", "Gomez Addams", "gomez"),
              ("the woman in the yellow dress", "Morticia Addams", "morticia")],
        keep="The hilltop, dusk sky, city lights, lighting and camera stay as in <Video 1>."),
    "pulp": dict(full_body=True, look="Live-action costumed performers with stop-motion style character design, lit like the scene.",
        subj=[("the man in the black suit dancing on the right", "Jack Skellington", "jack"),
              ("the woman in the white shirt and black trousers dancing on the left", "Sally", "sally")],
        keep="The diner stage, audience, lighting and camera stay as in <Video 1>."),
    "thisisamerica": dict(full_body=True, look="Photoreal cinematic CGI character, lit like the scene.",
        subj=[("the shirtless dancing man in light trousers", "Thanos", "thanos")],
        keep="The warehouse stays exactly as in <Video 1>. Same framing, lighting and camera as <Video 1>."),
    "walken": dict(full_body=True, look="Photoreal live-action, lit like the scene.",
        subj=[("the dancing man in the suit", "Patrick Bateman", "bateman")],
        keep="The hotel lobby, lighting and camera stay as in <Video 1>."),
    "singleladies": dict(full_body=True, look="Photoreal live-action, black-and-white like the scene.",
        subj=[("the dancing woman in the center", "Michael Myers", "myers")],
        keep="The two dancers on the left and right stay exactly as they are. The white studio, lighting and camera stay as in <Video 1>."),
}
FIX = {
    "orig": " Replace each Subject completely, head, face, hair and hands.",
    "bg": " Keep the exact original scene and colours of <Video 1> in every frame.",
    "leak": " Never show the reference picture itself; only the scene of <Video 1>.",
}

def likeness_slot():
    """Optional personal likeness LoRA. Off unless GJ_LIKENESS_LORA is set. 0.35+ broke the swap."""
    name = os.environ.get("GJ_LIKENESS_LORA", "").strip()
    if not name:
        return None
    strength = float(os.environ.get("GJ_LIKENESS_STRENGTH", "0") or 0)
    if strength <= 0:
        return None
    if strength >= 0.35:
        raise SystemExit("GJ V2: a personal likeness LoRA at 0.35 or above broke the swap. Leave it off.")
    return (name, strength)

def lora_stack():
    """Default stack: realism, swap, exactly one speed LoRA. Likeness appended only when enabled."""
    items = [REALISM, SWAP, SPEED]
    extra = likeness_slot()
    if extra:
        items.append(extra)
    speed = [n for n, s in items if s and ("turbo" in n.lower() or "lightx2v" in n.lower())]
    if speed != [SPEED[0]]:
        raise SystemExit(f"GJ V2: exactly one speed LoRA ({SPEED[0]}), got {speed}")
    return items

def load_require(items, tag):
    expect = {n: LORA_EXPECT[n] for n, _s in items if n in LORA_EXPECT}
    strengths = {n: s for n, s in items}
    return {
        "max_lossy_merges": 0,
        "no_load_warnings": True,
        "expect": expect,
        "strengths": strengths,
        "min_hooks": {"fc2": 104},
        "abort": True,
        "tag": tag,
    }

def render_prompt(subj, full_body, keep, look):
    bits, role_bits, refs = [], [], []
    n = 1
    for j, (who, name, key) in enumerate(subj, 1):
        bits.append(f"Subject {j} is {who}; Subject {j} becomes {name}.")
        for role, fn in zip(roles(full_body), filenames(key, full_body)):
            if is_sheet(fn):
                raise SystemExit(f"GJ V2: ref {fn} is a combined character sheet")
            refs.append(fn)
            role_bits.append(f"<Picture {n}> is {ROLE_TEXT[role].format(n=n)}.")
            n += 1
    text = (
        "In <Video 1>: " + " ".join(bits) + " "
        "Reference roles: " + " ".join(role_bits) + " "
        "The pictures are separate references, never one combined character sheet. "
        "Each Subject keeps the motion, gestures, head turns and natural mouth movement of <Video 1>. "
        "Match the source's mouth and expression exactly. "
        + keep.rstrip() + " keep any objects already in the scene; add nothing. "
        + look
    )
    return text, refs

def prompt(slug):
    c = CLIPS[slug]
    return render_prompt(c["subj"], c.get("full_body", False), c["keep"], c["look"])

def _exact_inputs(items):
    inp = {"model": ["1", 0]}
    for i in range(1, 6):
        if i <= len(items):
            inp[f"lora_{i}"] = items[i - 1][0]
            inp[f"strength_{i}"] = items[i - 1][1]
        else:
            inp[f"lora_{i}"] = "None"
            inp[f"strength_{i}"] = 0.0
    return inp

def validate_graph(g):
    blob = json.dumps(g)
    for banned in BANNED:
        if banned in blob:
            raise SystemExit(f"GJ V2 graph still contains removed item {banned}")
    if likeness_slot() is None and LIKENESS_FILE in blob:
        raise SystemExit("GJ V2: personal likeness LoRA is off unless GJ_LIKENESS_LORA is set")
    reports = [n for n in g.values() if n.get("class_type") == "H3LoadReport"]
    stacks = [n for n in g.values() if n.get("class_type") == "H3ExactLoraStack"]
    if len(reports) != 1 or len(stacks) != 1:
        raise SystemExit("GJ V2: graph needs one H3ExactLoraStack and one H3LoadReport")
    req = json.loads(reports[0]["inputs"]["require"])
    if req.get("max_lossy_merges") != 0 or req.get("abort") is not True:
        raise SystemExit("GJ V2: load check must abort unless lossy merges are 0")
    if req.get("min_hooks", {}).get("fc2", 0) < 104:
        raise SystemExit("GJ V2: load check must require fc2 hooks")
    if g["15"]["inputs"].get("ref_image_size") != REF_IMAGE_SIZE:
        raise SystemExit("GJ V2: ref_image_size must be match")
    if g["21"]["inputs"].get("steps") != STEPS:
        raise SystemExit("GJ V2: steps must be 8")
    if g["82"]["inputs"].get("shift_video") != SHIFT_VIDEO:
        raise SystemExit("GJ V2: video shift must be 6")
    if g["54"]["inputs"].get("sage_attention") != "auto":
        raise SystemExit("GJ V2: Sage Attention must be on")
    if g["22"]["inputs"].get("noise_seed") is None:
        raise SystemExit("GJ V2: seed missing")
    pr = g["15"]["inputs"]["prompt"]
    for line in ("keep any objects already in the scene; add nothing", "Match the source's mouth and expression exactly"):
        if line not in pr:
            raise SystemExit(f"GJ V2 prompt missing {line!r}")
    return req

def graph(slug, k, guide, seed=SEED, fixes=(), t=0):
    g = copy.deepcopy(_load_json("gb_S1_t2.api.json", "base_graphs"))
    for nid in list(g):
        ct = g[nid].get("class_type")
        if ct in ("LoraLoaderModelOnly", "LoraLoader", "LoraLoaderBypassModelOnly", "LoadImage"):
            g.pop(nid, None)
    items = lora_stack()
    i = g["15"]["inputs"]
    pr, refs = prompt(slug)
    pr += "".join(FIX[f] for f in fixes)
    i.update(prompt=pr, length=LEN, width=1152, height=640, ref_image_size=REF_IMAGE_SIZE)
    for x in [x for x in list(i) if x.startswith("ref_images.")]:
        del i[x]
    g["1"]["inputs"]["unet_name"] = UNET
    for j, im in enumerate(refs):
        n = str(200 + j)
        g[n] = {"class_type": "LoadImage", "inputs": {"image": im}}
        i[f"ref_images.ref_image_{j}"] = [n, 0]
    tag = f"v5b_{slug}_p{k}_t{t}"
    g["80"] = {"class_type": "H3ExactLoraStack", "inputs": _exact_inputs(items)}
    g["81"] = {"class_type": "H3LoadReport", "inputs": {
        "model": ["80", 0], "tag": tag, "require": json.dumps(load_require(items, tag)),
    }}
    g["82"] = {"class_type": "MiniMaxH3SigmaShift", "inputs": {
        "model": ["81", 0], "shift_video": SHIFT_VIDEO, "shift_audio": SHIFT_AUDIO,
    }}
    g["54"] = {"class_type": "PathchSageAttentionKJ", "inputs": {"model": ["82", 0], "sage_attention": "auto"}}
    g["20"]["inputs"]["sampler_name"] = "euler"
    g["21"]["inputs"].update(steps=STEPS, scheduler="simple", model=["54", 0])
    g["22"]["inputs"]["noise_seed"] = int(seed)
    g["23"]["inputs"]["model"] = ["54", 0]
    g["23"]["inputs"]["conditioning"] = ["15", 0]
    up = [f"v5b_{slug}{os.environ.get('RIDTAG', '')}_d{k}.mp4"] + refs
    g["6"]["inputs"]["file"] = up[0]
    if guide:
        A = _load_json("gjB_p1_s42.api.json", "base_graphs")
        g["70"] = {"class_type": "LoadVideo", "inputs": {"file": guide}}
        g["71"] = {"class_type": "GetVideoComponents", "inputs": {"video": ["70", 0]}}
        g["72"] = copy.deepcopy(A["72"])
        g["23"]["inputs"]["conditioning"] = ["72", 0]
        up.append(guide)
    rid = f"v5b_{slug}{os.environ.get('RIDTAG', '')}_p{k}_t{t}"
    g["33"]["inputs"]["filename_prefix"] = f"v5b/{rid}"
    validate_graph(g)
    G = pathlib.Path(_fh()) / "plan" / "graphs"
    G.mkdir(parents=True, exist_ok=True)
    p = G / f"{rid}.api.json"
    p.write_text(json.dumps(g))
    return {
        "id": rid, "engine": "l21", "seed": int(seed), "graph_path": str(p), "version": VERSION,
        "uploads": up, "params": {
            "steps": STEPS, "shift_video": SHIFT_VIDEO, "shift_audio": SHIFT_AUDIO,
            "length": LEN, "W": 1152, "H": 640, "ref_image_size": REF_IMAGE_SIZE,
            "sampler": "euler", "scheduler": "simple", "seed": int(seed),
            "loras": items, "sage": True, "version": VERSION,
        },
        "est_s": 60 + 3 * LEN, "protect": True, "rung": rid, "tier": "L",
        "needs_stage2": False, "pod_post": True, "stitch": False,
        "question": f"gjv2 {slug} p{k} t{t} {list(fixes)}", "spec": {},
    }

if __name__ == "__main__":
    for s in CLIPS:
        print(s, "|", prompt(s)[0], "\n")
