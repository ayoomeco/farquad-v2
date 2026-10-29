#!/usr/bin/env bash
# Image-build install for the GJ fastboot worker. Runs ONCE at docker build time, never at pod boot.
# Uses the base image's torch as-is (runpod/pytorch: torch 2.13.0 + cu130): a constraints file pins
# torch/torchvision/torchaudio to the installed versions so no requirement can reinstall or upgrade them.
set -euo pipefail
GJ=${GJ:-/opt/gj}; COMFY_REF=${COMFY_REF:-v0.42.5}; COMFY_SHA=${COMFY_SHA:-483dae4326356b81de05d27aca3f433dd323cc25}
HERE=$(cd "$(dirname "$0")" && pwd)
export UV_HTTP_TIMEOUT=300 UV_HTTP_RETRIES=8 UV_CONCURRENT_DOWNLOADS=8 PIP_BREAK_SYSTEM_PACKAGES=1
PY=python3.12
retry() { local n; for n in 1 2 3 4; do "$@" && return 0; echo "[install] attempt $n failed: $*"; sleep $((n*15)); done; return 1; }
torch_id() { $PY -c "import torch,torchvision,torchaudio;print(torch.__version__,torchvision.__version__,torchaudio.__version__,torch.version.cuda)"; }
BEFORE=$(torch_id); echo "[install] base torch: $BEFORE"
$PY - > "$GJ/constraints.txt" <<'PY'
import importlib.metadata as m
for d in ("torch", "torchvision", "torchaudio", "torchcodec", "triton", "nvidia-cudnn-cu13", "nvidia-cublas"):
    try: print(f"{d}=={m.version(d)}")
    except m.PackageNotFoundError: pass
PY
cat "$GJ/constraints.txt"
if command -v uv >/dev/null 2>&1; then PIP="uv pip install --system --break-system-packages -c $GJ/constraints.txt"; else PIP="$PY -m pip install -c $GJ/constraints.txt"; fi
git clone -q --depth 1 --branch "$COMFY_REF" https://github.com/comfyanonymous/ComfyUI.git "$GJ/ComfyUI"
[ "$(git -C "$GJ/ComfyUI" rev-parse HEAD)" = "$COMFY_SHA" ] || { echo "[install] FATAL ComfyUI $COMFY_REF is not $COMFY_SHA"; exit 1; }
retry $PIP -r "$GJ/ComfyUI/requirements.txt"
# Same pins as today's A08 pods (pod_setup_a08.sh) + boot-time fetch tools. sageattention 1.0.6 is the pure-Triton build (no nvcc compile).
retry $PIP "sageattention==1.0.6" "comfyui-frontend-package==1.55.18" "comfyui-workflow-templates==0.11.96" "comfyui-embedded-docs==0.5.14" \
  "comfy-kitchen==0.2.39" "comfy-aimdo==0.5.6" "huggingface_hub[hf_xet]" hf_transfer av requests websocket-client opencv-python-headless
grep -vE '^\s*(#|$)' "$HERE/custom_nodes.txt" | while read -r url pin; do
  d="$GJ/ComfyUI/custom_nodes/$(basename "$url")"
  git clone -q "$url" "$d" && git -C "$d" checkout -q "$pin"
  [ "$(git -C "$d" rev-parse HEAD)" = "$pin" ] || { echo "[install] FATAL $(basename "$url") not at $pin"; exit 1; }
  if [ -f "$d/requirements.txt" ]; then retry $PIP -r "$d/requirements.txt"; fi
  rm -rf "$d/.git"
done
cp -r "$HERE/nodes/h3_exact_lora" "$GJ/ComfyUI/custom_nodes/h3_exact_lora"
cp "$HERE/extra_model_paths.yaml" "$GJ/ComfyUI/extra_model_paths.yaml"
AFTER=$(torch_id); echo "[install] torch after: $AFTER"
[ "$BEFORE" = "$AFTER" ] || { echo "[install] FATAL torch changed: '$BEFORE' -> '$AFTER'"; exit 1; }
# Build-time import check on CPU: every custom node must import under 0.42.5 or the build fails.
( cd "$GJ/ComfyUI" && timeout 600 $PY main.py --cpu --quick-test-for-ci --disable-auto-launch 2>&1 | tee /tmp/quicktest.log ) || true
if grep -q "IMPORT FAILED" /tmp/quicktest.log; then echo "[install] FATAL custom node import failed"; grep -B2 -A2 "IMPORT FAILED" /tmp/quicktest.log; exit 1; fi
for n in ComfyUI-KJNodes ComfyUI-VideoHelperSuite comfyui-krea2edit h3_exact_lora; do grep -q "custom_nodes/$n" /tmp/quicktest.log || echo "[install] WARN $n not listed in import times"; done
$PY - <<PY > "$GJ/BUILD_INFO.json"
import json, torch, subprocess
print(json.dumps({"comfyui": "$COMFY_REF", "comfyui_sha": "$COMFY_SHA", "torch": torch.__version__, "cuda": torch.version.cuda,
  "nodes": open("$HERE/custom_nodes.txt").read().split("\n"), "built": subprocess.check_output(["date", "-u", "+%FT%TZ"]).decode().strip()}, indent=1))
PY
find "$GJ" -name '__pycache__' -prune -exec rm -rf {} + ; rm -rf /root/.cache /workspace/.cache /tmp/*
echo "[install] done"
