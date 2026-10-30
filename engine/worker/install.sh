#!/usr/bin/env bash
# Genie Jutsu worker install (shared by the Dockerfile AND the Phase-1 "install onto network volume" path).
#   PREFIX=/opt/jutsu  bash install.sh          # image build
#   PREFIX=/workspace/jutsu bash install.sh     # onto a RunPod network volume (persists across pods)
# Installs: Python venv, torch 2.13.0+cu130, ComfyUI v0.42.5 or newer, the vendored h3_exact_lora node,
# pinned custom nodes, SageAttention 2.2.0 (built for sm_120 Blackwell, best-effort), hf_transfer.
# NO model weights (see config/models.manifest.json). GJ V2 refuses ComfyUI older than 0.42.5:
# stock loaders keep only the last chained bypass LoRA and merge adaLN weights lossily.
#
# Boot paths (the log line starts with [install] path=...):
#   exact   — $PREFIX/.install-ok-$COMFY_REF-$TORCH_VER exists. Exit 0. Nothing is rewritten.
#   upgrade — an older .install-ok-*-$TORCH_VER marker exists AND $PREFIX/venv can import torch
#             and torchvision. The baked pod image (ComfyUI v0.39.1) hits this. Reuse that venv
#             and never run `uv venv` on it (that command exits "venv already exists" and the pod
#             never boots). Fetch the $COMFY_REF tag, check it out, pip-install ComfyUI requirements
#             into the same venv with torch/torchvision/torchaudio held at the versions already
#             installed, re-import the required custom nodes, then write the new marker.
#   fresh   — no usable marker. If $PREFIX/venv already exists, recreate it with `uv venv --clear`
#             (or remove it, then create) instead of `uv venv`, which aborts when the directory
#             is already there. A venv without a marker is not reused: it may be a half-written install.
set -euo pipefail
PREFIX=${PREFIX:-/opt/jutsu}
COMFY_REF=${COMFY_REF:-v0.42.5}            # GJ V2: 0.42.5 or newer, never below
TORCH_VER=${TORCH_VER:-2.13.0}
TORCHVISION_VER=${TORCHVISION_VER:-0.28.0}
CUDA_TAG=${CUDA_TAG:-cu130}
SAGE_REF=${SAGE_REF:-v2.2.0}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-"8.9;12.0"}   # Ada (4090/L40S) + Blackwell (5090 / PRO 6000)
# Oct 8: a pod died on a single pypi.nvidia.com timeout (uv default 30 s x 3 retries). Longer timeout, more retries, and 4 tries.
export UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-300} UV_HTTP_RETRIES=${UV_HTTP_RETRIES:-8} UV_CONCURRENT_DOWNLOADS=${UV_CONCURRENT_DOWNLOADS:-8}
HERE=$(cd "$(dirname "$0")" && pwd)
EXACT="$PREFIX/.install-ok-$COMFY_REF-$TORCH_VER"
mkdir -p "$PREFIX"

retry() { local n; for n in 1 2 3 4; do "$@" && return 0; echo "[install] attempt $n failed: $*; retrying in $((n * 15))s"; sleep $((n * 15)); done; return 1; }

# uv when it is on PATH (baked images have it). Otherwise the venv's own pip.
# Never creates the venv; callers do that only on the fresh path.
set_pip() {
  if command -v uv >/dev/null 2>&1; then
    PIP="uv pip install -p $PREFIX/venv/bin/python"
  else
    PIP="$PREFIX/venv/bin/pip install"
  fi
}

# A baked venv we are willing to upgrade in place: python starts and the torch stack imports.
venv_works() {
  [ -x "$PREFIX/venv/bin/python" ] || return 1
  "$PREFIX/venv/bin/python" -c 'import sys, torch, torchvision; raise SystemExit(0 if sys.prefix else 1)'
}

# Drop direct torch/torchvision/torchaudio lines. `torchsde` stays. ComfyUI requirements.txt
# lists unpinned `torch` and `torchvision`; installing that file as-is would move the baked wheels.
filter_comfy_reqs() {
  python3 - "$1" "$2" <<'PY'
import pathlib, re, sys
src, dest = sys.argv[1], sys.argv[2]
drop = re.compile(r"^\s*(torch|torchvision|torchaudio)(\s|$|[<>=!~\[;])", re.I)
lines = pathlib.Path(src).read_text(encoding="utf-8", errors="replace").splitlines()
kept = [ln for ln in lines if not drop.match(ln)]
pathlib.Path(dest).write_text("\n".join(kept) + "\n")
PY
}

fetch_comfy_tag() {
  if [ ! -d "$PREFIX/ComfyUI/.git" ]; then
    echo "[install] $PREFIX/ComfyUI has no .git checkout; version check will overlay $COMFY_REF if needed"
    return 0
  fi
  echo "[install] fetching tag $COMFY_REF"
  if ! git -C "$PREFIX/ComfyUI" fetch --depth 1 origin "refs/tags/${COMFY_REF}:refs/tags/${COMFY_REF}"; then
    echo "[install] WARN git fetch of tag $COMFY_REF failed"
    return 0
  fi
  echo "[install] checking out $COMFY_REF"
  git -C "$PREFIX/ComfyUI" checkout -f "$COMFY_REF" || echo "[install] WARN git checkout $COMFY_REF failed"
}

# Clone if needed, fetch the tag, check out, overlay the tag tarball when the tree is still < 0.42.5.
ensure_comfy_ref() {
  [ -d "$PREFIX/ComfyUI" ] || git clone --depth 1 --branch "$COMFY_REF" https://github.com/comfyanonymous/ComfyUI.git "$PREFIX/ComfyUI"
  fetch_comfy_tag
  export COMFY_REF PREFIX
  set +e
  COMFY_VER=$(PREFIX="$PREFIX" COMFY_REF="$COMFY_REF" "$PREFIX/venv/bin/python" - <<'PY'
import os, pathlib, re, sys
def parse(s):
    out = []
    for part in s.replace("v", "").split("."):
        num = ""
        for ch in part:
            if ch.isdigit():
                num += ch
            else:
                break
        if num:
            out.append(int(num))
    return tuple(out)
want = parse(os.environ.get("COMFY_REF", ""))
if want < (0, 42, 5):
    sys.exit(f"[install] FATAL COMFY_REF {os.environ.get('COMFY_REF')} is older than 0.42.5")
p = pathlib.Path(os.environ["PREFIX"]) / "ComfyUI" / "comfyui_version.py"
text = p.read_text() if p.exists() else ""
m = re.search(r"__version__\s*=\s*['\"]([^'\"]+)", text)
got = m.group(1) if m else ""
if parse(got) < (0, 42, 5):
    sys.exit(2)
print(got)
PY
)
  rc=$?
  set -e
  if [ "$rc" = 2 ]; then
    echo "[install] overlaying ComfyUI ${COMFY_REF}"
    tmp=$(mktemp -d)
    curl -fsSL "https://github.com/Comfy-Org/ComfyUI/archive/refs/tags/${COMFY_REF}.tar.gz" -o "$tmp/c.tgz"
    tar -xzf "$tmp/c.tgz" -C "$tmp"
    mkdir -p "$PREFIX/ComfyUI"
    cp -a "$tmp"/ComfyUI-*/. "$PREFIX/ComfyUI/"
    rm -rf "$tmp"
    COMFY_VER=$(PREFIX="$PREFIX" COMFY_REF="$COMFY_REF" "$PREFIX/venv/bin/python" - <<'PY'
import os, pathlib, re, sys
p = pathlib.Path(os.environ["PREFIX"]) / "ComfyUI" / "comfyui_version.py"
m = re.search(r"__version__\s*=\s*['\"]([^'\"]+)", p.read_text())
ver = m.group(1) if m else ""
parts = []
for part in ver.split("."):
    num = "".join(ch for ch in part if ch.isdigit())
    if not num:
        break
    parts.append(int(num))
if tuple(parts) < (0, 42, 5):
    sys.exit(f"[install] FATAL ComfyUI {ver or 'missing'} is older than 0.42.5")
print(ver)
PY
)
    rc=0
  fi
  if [ "$rc" != 0 ] || [ -z "${COMFY_VER:-}" ]; then echo "[install] FATAL ComfyUI ${COMFY_VER:-missing} is older than 0.42.5"; exit 1; fi
  echo "[install] ComfyUI ${COMFY_VER}"
}

install_gj_files() {
  cp "$HERE/extra_model_paths.yaml" "$PREFIX/ComfyUI/extra_model_paths.yaml"
  # GJ V2: exact LoRA stack. Stock LoraLoader is not used for the default H3 render.
  mkdir -p "$PREFIX/ComfyUI/custom_nodes"
  rm -rf "$PREFIX/ComfyUI/custom_nodes/h3_exact_lora"
  cp -a "$HERE/custom_nodes/h3_exact_lora" "$PREFIX/ComfyUI/custom_nodes/h3_exact_lora"
  # Repo root follows GJ_HOME / FARQUAD_HOME when that clone holds the post script, else this file's checkout.
  if [ -n "${GJ_HOME:-}" ] && [ -f "${GJ_HOME}/polish/pod/post_gjv2.py" ]; then
    repo="${GJ_HOME}"
  elif [ -n "${FARQUAD_HOME:-}" ] && [ -f "${FARQUAD_HOME}/polish/pod/post_gjv2.py" ]; then
    repo="${FARQUAD_HOME}"
  else
    repo=$(cd "$HERE/../.." && pwd)
  fi
  cp -a "$repo/polish/pod/post_gjv2.py" "$PREFIX/post_gjv2.py"
}

# Default GJ V2 graph nodes that are not ComfyUI core: exact LoRA stack, KJ Sage patch, VHS LoadVideo.
recheck_required_nodes() {
  "$PREFIX/venv/bin/python" - "$PREFIX" <<'PY'
import importlib.util, pathlib, sys
prefix = pathlib.Path(sys.argv[1])
root = prefix / "ComfyUI"
nodes = root / "custom_nodes"
required = ["h3_exact_lora", "ComfyUI-KJNodes", "ComfyUI-VideoHelperSuite"]
sys.path[:0] = [str(root), str(nodes)]
failed = []
for name in required:
    init = nodes / name / "__init__.py"
    if not init.is_file():
        failed.append(f"{name}: missing {init}")
        continue
    try:
        spec = importlib.util.spec_from_file_location(name, init)
        if spec is None or spec.loader is None:
            failed.append(f"{name}: no loader")
            continue
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    except Exception as e:
        failed.append(f"{name}: {type(e).__name__}: {e}")
if failed:
    sys.exit("[install] FATAL required custom nodes failed to import: " + " | ".join(failed))
print("[install] required custom nodes import OK: " + ", ".join(required))
PY
}

# In-place upgrade of a baked image. Does not create or delete $PREFIX/venv.
upgrade_inplace() {
  local old="$1"
  local torch_before tv_before ta_before torch_after tv_after con filtered
  echo "[install] path=upgrade from $(basename "$old"); reusing venv $PREFIX/venv (not recreating)"
  set_pip
  ensure_comfy_ref
  if [ ! -f "$PREFIX/ComfyUI/requirements.txt" ]; then
    echo "[install] FATAL upgrade: missing $PREFIX/ComfyUI/requirements.txt"
    exit 1
  fi
  torch_before=$("$PREFIX/venv/bin/python" -c 'import torch; print(torch.__version__)')
  tv_before=$("$PREFIX/venv/bin/python" -c 'import torchvision; print(torchvision.__version__)')
  ta_before=$("$PREFIX/venv/bin/python" -c 'import torchaudio; print(torchaudio.__version__)' 2>/dev/null || true)
  con=$(mktemp)
  {
    printf 'torch==%s\n' "$torch_before"
    printf 'torchvision==%s\n' "$tv_before"
    if [ -n "$ta_before" ]; then printf 'torchaudio==%s\n' "$ta_before"; fi
  } > "$con"
  filtered=$(mktemp)
  filter_comfy_reqs "$PREFIX/ComfyUI/requirements.txt" "$filtered"
  echo "[install] upgrade: installing ComfyUI requirements into the existing venv; keeping $(tr '\n' ' ' < "$con")"
  # Direct torch lines are already gone. The constraint file stops a transitive dep from moving the
  # baked wheels. A resolve that cannot see the local +cu build is retried with the pytorch wheel
  # index, still pinned. Never install this requirements set without that pin: an unpinned retry
  # would swap the CUDA torch and a later boot would treat the new version as the one to keep.
  if ! $PIP -c "$con" -r "$filtered"; then
    echo "[install] upgrade: constrained resolve failed; retrying against the pytorch wheel index with the baked torch pin"
    retry $PIP -c "$con" -r "$filtered" --extra-index-url "https://download.pytorch.org/whl/$CUDA_TAG"
  fi
  torch_after=$("$PREFIX/venv/bin/python" -c 'import torch; print(torch.__version__)')
  tv_after=$("$PREFIX/venv/bin/python" -c 'import torchvision; print(torchvision.__version__)')
  if [ "$torch_after" != "$torch_before" ] || [ "$tv_after" != "$tv_before" ]; then
    echo "[install] FATAL upgrade changed torch ($torch_before -> $torch_after) or torchvision ($tv_before -> $tv_after)"
    exit 1
  fi
  echo "[install] upgrade: torch unchanged $torch_after torchvision unchanged $tv_after"
  install_gj_files
  recheck_required_nodes
  touch "$EXACT"
  echo "[install] path=upgrade done -> $PREFIX"
}

if [ -f "$EXACT" ]; then echo "[install] path=exact already installed in $PREFIX"; exit 0; fi

# Same torch version, older ComfyUI ref. The chywgja image ships .install-ok-v0.39.1-$TORCH_VER.
old_marker=$(find "$PREFIX" -maxdepth 1 -type f -name ".install-ok-*-${TORCH_VER}" ! -name ".install-ok-${COMFY_REF}-${TORCH_VER}" | sort | tail -n 1 || true)
if [ -n "$old_marker" ] && venv_works; then
  upgrade_inplace "$old_marker"
  exit 0
fi
if [ -n "$old_marker" ]; then
  echo "[install] path=fresh: marker $(basename "$old_marker") is present but $PREFIX/venv is not a working venv; installing fresh"
else
  echo "[install] path=fresh"
fi

PY=${PY:-python3.12}
command -v uv >/dev/null 2>&1 || pip install -q uv || true
if [ -e "$PREFIX/venv" ]; then
  # Documented fresh-path choice: recreate. `uv venv` without --clear exits "venv already exists".
  echo "[install] path=fresh: $PREFIX/venv exists without a usable install marker; recreating it with uv venv --clear (or removing it first) so install does not abort with 'venv already exists'"
  if command -v uv >/dev/null 2>&1; then
    uv venv --clear -p "$PY" "$PREFIX/venv" || { rm -rf "$PREFIX/venv"; uv venv -p "$PY" "$PREFIX/venv"; }
  else
    rm -rf "$PREFIX/venv"
    "$PY" -m venv "$PREFIX/venv"
  fi
elif command -v uv >/dev/null 2>&1; then
  echo "[install] path=fresh: creating venv $PREFIX/venv"
  uv venv -p "$PY" "$PREFIX/venv"
else
  echo "[install] path=fresh: creating venv $PREFIX/venv"
  "$PY" -m venv "$PREFIX/venv"
fi
set_pip
retry $PIP "torch==$TORCH_VER" "torchvision==$TORCHVISION_VER" torchaudio --index-url "https://download.pytorch.org/whl/$CUDA_TAG"
ensure_comfy_ref
retry $PIP -r "$PREFIX/ComfyUI/requirements.txt"
retry $PIP hf_transfer huggingface_hub websocket-client opencv-python-headless triton
grep -vE '^\s*(#|$)' "$HERE/custom_nodes.txt" | while read -r url pin dir; do
  if [ -n "${dir:-}" ]; then                       # zip entry: <url> <sha256> <dir>  (Comfy Registry / HF hosted packs)
    d="$PREFIX/ComfyUI/custom_nodes/$dir"
    if [ ! -d "$d" ]; then
      tmpz=$(mktemp --suffix=.zip)
      curl -fsSL "$url" -o "$tmpz"
      echo "$pin  $tmpz" | sha256sum -c --quiet - || { echo "[install] FATAL sha256 mismatch for $dir"; exit 1; }
      mkdir -p "$d" && python3 -m zipfile -e "$tmpz" "$d" && rm -f "$tmpz"
      rm -rf "$d/__pycache__"
    fi
  else                                             # git entry: <url> <commit>
    d="$PREFIX/ComfyUI/custom_nodes/$(basename "$url")"
    if [ ! -d "$d" ]; then git clone "$url" "$d"; fi
    git -C "$d" fetch --depth 1 origin "$pin" 2>/dev/null || true
    git -C "$d" checkout -q "$pin" || echo "[install] WARN could not pin $(basename "$url") to $pin"
  fi
  # uv resolves a requirements file all-or-nothing: one unbuildable line (sam2 git, groundingdino) dropped the whole pack's
  # deps before. Fall back to line-by-line so everything installable is baked in.
  if [ -f "$d/requirements.txt" ]; then
    $PIP -r "$d/requirements.txt" || grep -vE '^\s*(#|$)' "$d/requirements.txt" | while read -r req; do $PIP "$req" || echo "[install] WARN $(basename "$d"): $req"; done
  fi
done
# Pre-baked runtime deps the packs otherwise pip-install on first use (Oct 16: LatentSyncNode died with "Failed to install
# required package: pytorch_lightning" -- uv venvs have no pip module, and it probes 'ffmpeg-python' by dist name, so it
# ALWAYS tried). pip itself is added so any remaining `python -m pip` fallback works.
PACK_DEPS="pip pytorch_lightning accelerate omegaconf einops diffusers transformers huggingface_hub ffmpeg-python
  face-alignment decord DeepCache soundfile mediapipe librosa ultralytics insightface scenedetect scipy onnxruntime-gpu
  segment-anything scikit-image piexif dill matplotlib transparent-background hydra-core iopath ftfy"
retry $PIP $PACK_DEPS || for req in $PACK_DEPS; do $PIP "$req" || echo "[install] WARN pack dep $req"; done
$PIP "git+https://github.com/facebookresearch/sam2" || echo "[install] WARN sam2 (Impact SAM2 loader) not built; SAM v1 path still works"
touch /root/.latentsync16_dependencies_installed || echo "[install] WARN could not write /root/.latentsync16_dependencies_installed"
# Oct 16: torchaudio>=2.9 save() needs TorchCodec ("TorchCodec is required for save_with_torchcodec") and torchvision 0.28 has no
# io.write_video/read_video. Patch LatentSyncWrapper to soundfile (PCM_16) + imageio-ffmpeg; pinned IO deps. Build fails if it can't patch.
retry $PIP "soundfile==0.13.1" "imageio==2.37.0" "imageio-ffmpeg==0.6.0" "decord==0.6.0"
LSW="$PREFIX/ComfyUI/custom_nodes/ComfyUI-LatentSyncWrapper"
LSP="$HERE/latentsync_audio_patch.py"; [ -f "$LSP" ] || LSP="$HERE/patches/latentsync_audio_patch.py"
if [ -d "$LSW" ]; then "$PREFIX/venv/bin/python" "$LSP" "$LSW" || { echo "[install] FATAL LatentSync audio patch failed"; exit 1; }; fi
LVP="$HERE/ls_vae_patch.py"; [ -f "$LVP" ] || LVP="$HERE/patches/ls_vae_patch.py"
if [ -d "$LSW" ]; then "$PREFIX/venv/bin/python" "$LVP" "$LSW" || { echo "[install] FATAL LatentSync VAE patch failed"; exit 1; }; fi
"$PREFIX/venv/bin/python" -c "import pytorch_lightning, accelerate, ffmpeg, face_alignment, decord, ultralytics, insightface, scenedetect, segment_anything, transparent_background; print('[install] pack deps import OK')" \
  || echo "[install] WARN a pack dependency failed to import (see above)"
# SageAttention 2.2.0 (2x attention on Blackwell; KJ PathchSageAttentionKJ / --use-sage-attention). Needs nvcc, not a GPU.
# Uses $PIP (uv venvs have no pip module: `python -m pip` failed on the first Phase-1 pod).
# Kept OUT of the CI image (Oct 15: the SageAttention CUDA build crashed the GitHub runner). Opt in with JUTSU_BUILD_SAGE=1.
if [ "${JUTSU_BUILD_SAGE:-0}" != 1 ]; then
  echo "[install] SageAttention skipped (JUTSU_BUILD_SAGE!=1): SDPA"
elif command -v nvcc >/dev/null 2>&1; then
  ( tmp=$(mktemp -d) && git clone --depth 1 --branch "$SAGE_REF" https://github.com/thu-ml/SageAttention.git "$tmp/sage" \
    && cd "$tmp/sage" && $PIP setuptools wheel ninja packaging \
    && EXT_PARALLEL=4 NVCC_APPEND_FLAGS="--threads 8" MAX_JOBS=8 $PIP --no-build-isolation . ) \
    && echo "[install] SageAttention $SAGE_REF built" || echo "[install] WARN SageAttention build failed: ComfyUI falls back to PyTorch SDPA"
else
  echo "[install] WARN nvcc not found: skipping SageAttention (SDPA fallback)"
fi
install_gj_files
"$PREFIX/venv/bin/python" -c "import torch; print('[install] torch', torch.__version__, 'cuda', torch.version.cuda)"
touch "$EXACT"
echo "[install] path=fresh done -> $PREFIX"
