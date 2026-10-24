#!/usr/bin/env bash
# Phase-1 pod boot (no volume: everything on the container disk, nothing persists). Order is tuned for
# time-to-first-render:  status server :8001 -> apt -> [ install.sh  ||  stage-1 pull = head-to-head bundles ]
#                        -> ComfyUI :8188 -> stage-2 pull (rest of the groups) in the background -> dl2.done
# The box runner polls /status (progress = real stall detection) and waits for dl2.done before non-head-to-head runs.
set -u
SRC=$(cd "$(dirname "$0")/.." && pwd); S=/opt/jstatus; mkdir -p "$S"
export JUTSU_PREFIX=${JUTSU_PREFIX:-/opt/jutsu} MODEL_ROOT=${MODEL_ROOT:-/opt/jm}
mkdir -p "$MODEL_ROOT/models"
stage(){ echo "$(date +%s) $*" >> "$S/stages.txt"; echo "[boot] $*"; }
python3 "$SRC/worker/status_server.py" "$S" "$MODEL_ROOT/models" 8001 >> "$S/boot.log" 2>&1 &
stage boot gpu=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | tr ' ' '_')
# Oct 21: u1 pod hung in apt (no apt_done in 7 min) -> skip when the image has the tools, else bounded by timeout
if command -v ffmpeg >/dev/null && command -v git >/dev/null && command -v aria2c >/dev/null && python3 -c "import ensurepip" 2>/dev/null; then stage apt_skipped; else
timeout 180 apt-get update -qq > "$S/apt.log" 2>&1 && timeout 300 apt-get install -y -qq --no-install-recommends git git-lfs ffmpeg libgl1 libglib2.0-0 \
  python3.12-venv python3.12-dev build-essential ninja-build curl ca-certificates aria2 >> "$S/apt.log" 2>&1 && stage apt_done || stage apt_failed; fi
# optional WanGP/MMGP side install (JUTSU_WANGP=1): own venv, runs in parallel; the runner waits for wangp_done before its rung
if [ "${JUTSU_WANGP:-0}" = 1 ]; then
  ( stage wangp_install_start; TORCH_CUDA_ARCH_LIST="12.0" bash worker/wangp_install.sh > "$S/wangp_install.log" 2>&1 \
      && stage wangp_done || stage wangp_failed ) &
fi
# Oct 20: baked image? reassemble model chunks first so the downloader finds them present
[ "${JUTSU_UNCHUNK:-0}" = 1 ] && { stage unchunk_start; MODEL_ROOT=$MODEL_ROOT bash worker/unchunk_models.sh >> "$S/boot.log" 2>&1 && stage unchunk_done || stage unchunk_failed; }
# downloader venv (tiny) so the 90 GB stage-1 pull runs in parallel with the torch/ComfyUI install
( python3 -m venv /opt/dl && /opt/dl/bin/pip install -q huggingface_hub hf_transfer ) > "$S/dl1.log" 2>&1 || stage dl_venv_failed
DLPY=/opt/dl/bin/python; [ -x "$DLPY" ] || DLPY=python3
BARGS=""; for b in ${JUTSU_STAGE1_BUNDLES:-h3 h3_hyperflow wan_vace}; do BARGS="$BARGS --bundle $b"; done
( stage dl1_start; $DLPY worker/bootstrap_volume.py --manifest config/models.manifest.json --root "$MODEL_ROOT" $BARGS --workers 8 \
    >> "$S/dl1.log" 2>&1 && stage dl1_done || stage dl1_failed ) &
DL1=$!
( stage install_start; TORCH_CUDA_ARCH_LIST="12.0" PREFIX=$JUTSU_PREFIX bash worker/install.sh > "$S/install.log" 2>&1 \
    && stage install_done || stage install_failed ) &
INST=$!
wait $INST; wait $DL1
cp worker/extra_model_paths.yaml "$JUTSU_PREFIX/ComfyUI/" 2>/dev/null || true
( for g in ${JUTSU_GROUPS:-phase1}; do $DLPY worker/bootstrap_volume.py --manifest config/models.manifest.json --root "$MODEL_ROOT" \
    --group "$g" --workers 8; done > "$S/dl2.log" 2>&1 && stage dl2_done || stage dl2_failed ) &
for i in 1 2 3 4 5; do                       # restarted if it dies (or a fix kills it: pkill -f "main.py --listen")
  stage comfy_start_$i; bash worker/start.sh >> "$S/comfy.log" 2>&1; stage comfy_exited_$i; sleep 3
done
sleep infinity
