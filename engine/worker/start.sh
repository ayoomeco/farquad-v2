#!/usr/bin/env bash
# Worker entrypoint. Finds the model root on the attached network volume, links it to /models, starts ComfyUI.
# Optional dead-man switch: JUTSU_MAX_SECONDS=N terminates THIS pod after N seconds using the pod-scoped
# RUNPOD_API_KEY that RunPod injects into every pod (no user key ever leaves the box).
set -euo pipefail
# Oct 21: CUDA 804 guard - never load forward-compat libcuda on GeForce (driver is new enough for cu130)
for f in /etc/ld.so.conf.d/*; do grep -q compat "$f" 2>/dev/null && rm -f "$f"; done; ldconfig 2>/dev/null || true
export LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v compat | paste -sd: -)
# LatentSyncWrapper skips its runtime pip step when this flag exists (deps are baked by install.sh)
touch "$HOME/.latentsync16_dependencies_installed" 2>/dev/null || true
for root in "${MODEL_ROOT:-}" /runpod-volume /workspace; do
  if [ -n "$root" ] && [ -d "$root/models" ]; then MODEL_ROOT=$root; break; fi
done
MODEL_ROOT=${MODEL_ROOT:-/workspace}
ln -sfn "$MODEL_ROOT/models" /models
JUTSU=${JUTSU_PREFIX:-}
for p in "$JUTSU" /opt/jutsu "$MODEL_ROOT/jutsu"; do
  if [ -n "$p" ] && [ -x "$p/venv/bin/python" ]; then JUTSU=$p; break; fi
done
echo "[start] models=$MODEL_ROOT/models jutsu=$JUTSU"
if [ -n "${JUTSU_MAX_SECONDS:-}" ] && [ -n "${RUNPOD_POD_ID:-}" ]; then
  ( sleep "$JUTSU_MAX_SECONDS"; echo "[start] dead-man switch: terminating pod $RUNPOD_POD_ID";
    (command -v runpodctl >/dev/null && runpodctl remove pod "$RUNPOD_POD_ID") || \
    curl -s -X DELETE -H "Authorization: Bearer ${RUNPOD_API_KEY:-}" "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID" ) &
fi
# Box-uploaded files (character LoRAs) arrive in <=90 MB parts via ComfyUI /upload/image; reassemble in the background
HERE=$(cd "$(dirname "$0")" && pwd)
if [ -f "$HERE/assemble_uploads.sh" ] && ! pgrep -f assemble_uploads.sh >/dev/null; then
  JUTSU_PREFIX=$JUTSU bash "$HERE/assemble_uploads.sh" "$JUTSU/ComfyUI/input/jutsu_parts" &
fi
# GPU-profile flags (see research: KJ chunking + cu130 nvfp4 matter far more than Sage on a 5090)
FLAGS=(--listen 0.0.0.0 --port "${COMFY_PORT:-8188}" --disable-auto-launch --preview-method none)
# Global Sage is OPT-IN (JUTSU_GLOBAL_SAGE=1): recipes pick attention per model (ModelAttentionBackend / KJ patches), and a
# global flag would silently change what the speed ladder measures.
if [ "${JUTSU_GLOBAL_SAGE:-0}" = 1 ]; then "$JUTSU/venv/bin/python" -c "import sageattention" 2>/dev/null && FLAGS+=(--use-sage-attention) || true; fi
[ -n "${COMFY_EXTRA_FLAGS:-}" ] && FLAGS+=(${COMFY_EXTRA_FLAGS})
# FlashVSR node reads ComfyUI/models/FlashVSR-v1.1 directly (ignores extra_model_paths) and calls snapshot_download when that
# dir is missing. Phase 1: ComfyUI started before the background download created it -> no link -> HF 429. Link ALWAYS.
mkdir -p /models/FlashVSR-v1.1 && ln -sfn /models/FlashVSR-v1.1 "$JUTSU/ComfyUI/models/FlashVSR-v1.1" || true
# Round 3 packs read fixed dirs under ComfyUI/models or their own checkpoints dir (not extra_model_paths): link them to the volume.
for d in ultralytics sams; do mkdir -p "/models/$d" && { [ -L "$JUTSU/ComfyUI/models/$d" ] || rm -rf "$JUTSU/ComfyUI/models/$d"; } && ln -sfn "/models/$d" "$JUTSU/ComfyUI/models/$d" || true; done
LS="$JUTSU/ComfyUI/custom_nodes/ComfyUI-LatentSyncWrapper"
# volume installs made before Oct 16 lack the soundfile patch: apply it at boot (idempotent)
for P in /opt/jutsu-src/latentsync_audio_patch.py "$(dirname "$0")/patches/latentsync_audio_patch.py"; do [ -d "$LS" ] && [ -f "$P" ] && { "$JUTSU/venv/bin/python" "$P" "$LS" || true; break; }; done
# Oct 17: VAE fix + fail-closed VAE check (payload copy first, so no image rebuild is needed). A failed patch = no LatentSync.
if [ -d "$LS" ]; then
  VP="$(dirname "$0")/patches/ls_vae_patch.py"; [ -f "$VP" ] || VP=/opt/jutsu-src/ls_vae_patch.py
  if [ -f "$VP" ]; then "$JUTSU/venv/bin/python" "$VP" "$LS" || { echo "[start] FATAL ls-vae patch failed: disabling LatentSync"; rm -rf "$LS"; }
  else echo "[start] WARNING ls_vae_patch.py missing"; fi
fi
[ -d "$LS" ] && mkdir -p /models/latentsync && rm -rf "$LS/checkpoints" && ln -sfn /models/latentsync "$LS/checkpoints" || true
# bench: runtime flag overrides (box drops /tmp/jutsu_flags.env via pod_cmds, then restarts ComfyUI)
[ -f /tmp/jutsu_flags.env ] && . /tmp/jutsu_flags.env
# ---- v2 feature flags (all OFF by default; see worker/v2/flags.md) ----
DIS="$JUTSU/custom_nodes.disabled"; CN="$JUTSU/ComfyUI/custom_nodes"
flag(){ local on=$1 node=$2; if [ "$on" = 1 ] && [ -d "$DIS/$node" ]; then ln -sfn "$DIS/$node" "$CN/$node"; elif [ -L "$CN/$node" ]; then rm -f "$CN/$node"; fi; }
PP=""
[ "${JUTSU_SAGE:-2.2}" = 1.0.6 ] && [ -d /opt/sage106 ] && { PP="/opt/sage106"; export JUTSU_SAGE_SM120_MODE=auto; echo "[start] Sage 1.0.6 fallback"; }
[ "${JUTSU_ENABLE_SAGE3:-0}" = 1 ] && [ -d /opt/sage3 ] && PP="$PP:/opt/sage3"
flag "${JUTSU_ENABLE_SAGE3:-0}" ComfyUI-SageAttention3
[ "${JUTSU_ENABLE_SPARK:-0}" = 1 ] && [ -d /opt/spark-ck ] && PP="/opt/spark-ck:$PP"
flag "${JUTSU_ENABLE_SPARK:-0}" ComfyUI-Spark-H3
flag "${JUTSU_ENABLE_H3RELAY:-0}" h3-relay
if [ "${JUTSU_ENABLE_FASTH3:-0}" = 1 ]; then MODEL_ROOT=$MODEL_ROOT JUTSU_UNCHUNK_OPT=fasth3 bash "$HERE/unchunk_models.sh" || echo "[start] WARN fasth3 unchunk failed"; fi
PP=${PP#:}; PP=${PP%:}; [ -n "$PP" ] && export PYTHONPATH="$PP${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p /models/vae_approx /models/latent_upscale_models /models/mediapipe && ln -sfn /models/vae_approx "$JUTSU/ComfyUI/models/vae_approx" 2>/dev/null || true
echo "[start] flags sage=${JUTSU_SAGE:-2.2} sm120mode=${JUTSU_SAGE_SM120_MODE:-qk_int8_pv_fp16_cuda} sage3=${JUTSU_ENABLE_SAGE3:-0} spark=${JUTSU_ENABLE_SPARK:-0} relay=${JUTSU_ENABLE_H3RELAY:-0} fasth3=${JUTSU_ENABLE_FASTH3:-0}"
# A08: every boot refreshes h3_exact_lora and refuses to start on ComfyUI older than 0.42.5.
JUTSU_PREFIX=$JUTSU bash "$HERE/ensure_a08.sh"
cd "$JUTSU/ComfyUI"
exec "$JUTSU/venv/bin/python" main.py "${FLAGS[@]}"
