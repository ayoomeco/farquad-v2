"""Offline checks for the GJ V2 default render. No GPU, no network, no spend."""
import json, os, pathlib, subprocess, sys, tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
os.environ["GJ_HOME"] = tempfile.mkdtemp(prefix="gjv2-")
os.environ.pop("FARQUAD_HOME", None)
os.environ.pop("GJ_LIKENESS_LORA", None)
os.environ.pop("GJ_LIKENESS_STRENGTH", None)
sys.path.insert(0, str(ROOT / "plan"))
sys.path.insert(0, str(ROOT / "lib"))
sys.path.insert(0, str(ROOT / "polish" / "pod"))

import graphs_v5b as G
import refs_gjv2
import post_gjv2
from home import gj_home

def fail(msg):
    raise SystemExit(msg)

run = G.graph("walken", 0, None)
g = json.loads(pathlib.Path(run["graph_path"]).read_text())
assert run["version"] == "GJ V2", run["version"]
assert run["pod_post"] is True and run["stitch"] is False
assert run["params"]["steps"] == 8
assert run["params"]["shift_video"] == 6.0 and run["params"]["shift_audio"] == 3.0
assert run["params"]["sampler"] == "euler" and run["params"]["scheduler"] == "simple"
assert run["params"]["seed"] == 42 and g["22"]["inputs"]["noise_seed"] == 42
assert g["15"]["inputs"]["ref_image_size"] == "match"
assert g["20"]["inputs"]["sampler_name"] == "euler"
assert g["21"]["inputs"]["steps"] == 8 and g["21"]["inputs"]["scheduler"] == "simple"
assert g["54"]["inputs"]["sage_attention"] == "auto"
assert g["54"]["inputs"]["model"] == ["82", 0]
assert g["82"]["inputs"]["shift_video"] == 6.0 and g["82"]["inputs"]["shift_audio"] == 3.0
stack = g["80"]["inputs"]
assert stack["lora_1"].endswith("h3-realism-people-t2v-i2v-r2v.safetensors") and stack["strength_1"] == 0.7
assert stack["lora_2"].endswith("h3_character_swap_v1.5.safetensors") and stack["strength_2"] == 1.0
assert "turbo_8step_v1.1" in stack["lora_3"] and stack["strength_3"] == 0.8
assert stack["lora_4"] == "None" and stack["strength_4"] == 0.0
req = json.loads(g["81"]["inputs"]["require"])
assert req["max_lossy_merges"] == 0 and req["abort"] is True
assert req["min_hooks"]["fc2"] >= 104
assert req["strengths"][stack["lora_3"]] == 0.8
assert G.LIKENESS_FILE not in json.dumps(g)
pr = g["15"]["inputs"]["prompt"]
for line in (
    "image 1 face", "image 2 body", "image 3 outfit",
    "Match the source's mouth and expression exactly.",
    "keep any objects already in the scene; add nothing",
    "never one combined character sheet",
):
    if line not in pr:
        fail(f"prompt missing {line!r}")
refs = [g[str(200 + i)]["inputs"]["image"] for i in range(3)]
assert refs == ["v5b_bateman_face.png", "v5b_bateman_body.png", "v5b_bateman_outfit.png"], refs
text, med = G.render_prompt([("the dancing man in the suit", "Patrick Bateman", "bateman")], False, "Keep the room.", "Photoreal.")
assert "image 1 face" in text and "image 2 outfit" in text and "image 2 body" not in text
assert med == ["v5b_bateman_face.png", "v5b_bateman_outfit.png"]
try:
    refs_gjv2.reject_sheet("v5b_hero_sheet.png")
    fail("sheet should be refused")
except ValueError:
    pass
cmd = refs_gjv2.seedvr2_image_cmd("face.png", "face_up.png")
assert "--mode" in cmd and "image" in cmd and "2048" in cmd
assert not any("codeformer" in c.lower() or "gfpgan" in c.lower() for c in cmd)

os.environ["GJ_LIKENESS_LORA"] = G.LIKENESS_FILE
os.environ["GJ_LIKENESS_STRENGTH"] = "0.35"
try:
    G.likeness_slot()
    fail("likeness 0.35 should be refused")
except SystemExit:
    pass
os.environ.pop("GJ_LIKENESS_LORA")
os.environ.pop("GJ_LIKENESS_STRENGTH")
assert G.likeness_slot() is None

toml = (ROOT / "engine" / "dsr" / "genie_dsr.toml").read_text()
assert 'version = "GJ V2"' in toml
assert 'comfyui_min_version = "0.42.5"' in toml
assert 'draft_accel = "turbo8"' in toml and 'final_accel = "turbo8"' in toml
install = (ROOT / "engine" / "worker" / "install.sh").read_text()
assert "COMFY_REF=${COMFY_REF:-v0.42.5}" in install
assert (ROOT / "engine" / "worker" / "custom_nodes" / "h3_exact_lora" / "__init__.py").is_file()
node = (ROOT / "engine" / "worker" / "custom_nodes" / "h3_exact_lora" / "__init__.py").read_text()
assert "class H3ExactLoraStack" in node and "class H3LoadReport" in node and "max_lossy_merges" in node

plan = post_gjv2.plan("walken")
assert plan["where"] == "pod" and plan["fps"] == "source"
for skipped in ("frame_interpolation", "color_grade", "grain", "face_restore", "paste_back"):
    assert skipped in plan["skipped"]
enc = post_gjv2.encode_cmd(0, 0, "24000/1001", "up.mp4", "native.mp4", "out.mp4")
blob = " ".join(enc)
for need in ("-profile:v", "high", "-level:v", "4.2", "yuv420p", "bt709", "aac", "+faststart", "passthrough"):
    if need not in blob:
        fail(f"encode missing {need}")
try:
    post_gjv2.encode_cmd(0, 0, "24", "up.mp4", "native.mp4", "out.mp4")
except SystemExit:
    fail("source 24 fps must encode")
os.environ["GJ_SOURCE_FPS"] = "24"
try:
    post_gjv2.encode_cmd(0, 0, "30", "up.mp4", "native.mp4", "out.mp4")
    fail("30 fps against a 24 fps source must be refused")
except SystemExit:
    pass
os.environ.pop("GJ_SOURCE_FPS")

for name in ("gb_S1_t2.api.json", "gjB_p1_s42.api.json"):
    raw = (ROOT / "plan" / "base_graphs" / name).read_text()
    for bad in ("turbo_v4", "pro4500", "m33k", "LoraLoaderModelOnly", "er_sde", '"steps": 6', "_sheet.png"):
        if bad in raw:
            fail(f"{name} still has {bad}")
gjB = json.loads((ROOT / "plan" / "base_graphs" / "gjB_p1_s42.api.json").read_text())
assert gjB["72"]["class_type"] == "MiniMaxH3AddGuide"
assert gjB["80"]["class_type"] == "ModelPatchLoader"

setup = (ROOT / "setup.sh").read_text()
assert 'GJ_HOME' in setup and 'FARQUAD_HOME' in setup and '_SCRIPT_DIR' in setup
assert 'GJ_V2_SETUP_OK FARQUAD_SETUP_OK' in setup
for name in ("s3.py", "s4.py"):
    body = (ROOT / "polish" / "pod" / name).read_text()
    if "Practical-RIFE" in body or "film grain" in body:
        fail(f"{name} still implements the old post")
polish = (ROOT / "polish" / "pod" / "setup.sh").read_text()
assert "SeedVR2" in polish and "Practical-RIFE" not in polish and "sczhou/CodeFormer" not in polish

env = {k: v for k, v in os.environ.items() if k not in ("GJ_HOME", "FARQUAD_HOME")}
out = subprocess.check_output([sys.executable, "-c", "import sys; sys.path.insert(0,'lib'); from home import gj_home; print(gj_home())"], cwd=ROOT, env=env, text=True).strip()
assert out == str(ROOT), out
env["FARQUAD_HOME"] = "/tmp/farquad-only"
env.pop("GJ_HOME", None)
out = subprocess.check_output([sys.executable, "-c", "import sys; sys.path.insert(0,'lib'); from home import gj_home; print(gj_home())"], cwd=ROOT, env=env, text=True).strip()
assert out == "/tmp/farquad-only", out
env["GJ_HOME"] = "/tmp/gj-wins"
out = subprocess.check_output([sys.executable, "-c", "import sys; sys.path.insert(0,'lib'); from home import gj_home; print(gj_home())"], cwd=ROOT, env=env, text=True).strip()
assert out == "/tmp/gj-wins", out

print("GJ V2 tests ok", run["id"])
