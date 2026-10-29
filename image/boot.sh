#!/usr/bin/env bash
# GJ fastboot pod entrypoint (image CMD). No installs at boot. Order:
#   1. start ComfyUI immediately (models dir may still be empty)   2. fetch ONLY $GJ_MODEL_SETS in parallel + sha256
#   3. READY.txt once ComfyUI answers AND weights verified        4. auto-exit: idle timeout / hard max seconds -> pod removes itself
# Env: GJ_MODEL_SETS (default h3_a08)  GJ_MANIFEST_OVERLAY_B64 (extra sets, base64 JSON {set: {...}})  GJ_VERIFY full|size  GJ_COMFY_ARGS  GJ_IDLE_EXIT_MIN (default 15, 0=off)  GJ_MAX_SECONDS (0=off)
#      GJ_START_RUNPOD_SERVICES=1 to also run the base image's /start.sh (SSH/Jupyter). Timings: output/boot_timings.json (served via /view).
set -uo pipefail
GJ=/opt/gj; C=$GJ/ComfyUI; OUT=$C/output; L=/workspace/gj_logs; mkdir -p "$OUT" "$L" /workspace/models
exec > >(tee -a "$L/boot.log") 2>&1
T0=$(date +%s.%N); SETS=${GJ_MODEL_SETS:-h3_a08}; export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
declare -A TS; mark() { TS[$1]=$(python3 -c "print(round($(date +%s.%N)-$T0,1))"); echo "[boot +${TS[$1]}s] $1"; write_t; }
write_t() { { echo "{"; for k in "${!TS[@]}"; do echo "  \"$k\": ${TS[$k]},"; done; echo "  \"sets\": \"$SETS\", \"pod\": \"${RUNPOD_POD_ID:-?}\", \"gpu\": \"$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)\""; echo "}"; } > "$OUT/boot_timings.json.tmp" && mv "$OUT/boot_timings.json.tmp" "$OUT/boot_timings.json"; }
fail() { echo "$1" > "$OUT/BOOT_FAILED.txt"; mark boot_failed; echo "[boot] FAILED: $1"; }
self_remove() {   # terminate (not stop) this pod so nothing keeps billing
  echo "[boot] self-remove: $1"; mark "self_remove"
  if command -v runpodctl >/dev/null && [ -n "${RUNPOD_POD_ID:-}" ]; then runpodctl remove pod "$RUNPOD_POD_ID" && return; fi
  if [ -n "${RUNPOD_API_KEY:-}" ] && [ -n "${RUNPOD_POD_ID:-}" ]; then
    curl -s -X DELETE -H "Authorization: Bearer $RUNPOD_API_KEY" "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID" && return; fi
  echo "[boot] WARN could not self-remove; launcher must terminate this pod"
}
mark container_start
[ "${GJ_START_RUNPOD_SERVICES:-0}" = 1 ] && [ -x /start.sh ] && (/start.sh >/workspace/gj_logs/runpod_start.log 2>&1 &)
# Host check: the image is CUDA 13.0 (torch cu130). A host driver below 580 cannot run it -> fail fast so the launcher replaces the pod.
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1); echo "[boot] driver ${DRV:-none}"
if [ -z "$DRV" ] || [ "${DRV%%.*}" -lt 580 ]; then fail "driver ${DRV:-missing} < 580 (CUDA 13 needs >= 580)"; fi
python3 -c "import torch; assert torch.cuda.is_available(); print('[boot] torch', torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))" || fail "torch cannot see the GPU"
mark gpu_checked
cd "$C" && nohup python3 main.py --listen 0.0.0.0 --port 8188 --disable-auto-launch ${GJ_COMFY_ARGS:---disable-dynamic-vram} > "$L/comfy.log" 2>&1 &
CPID=$!
# r3 (Oct 28): optional manifest overlay (sets added after the image build), same format the r2 launcher start-cmd uses. Size + sha256 checks unchanged.
if [ -n "${GJ_MANIFEST_OVERLAY_B64:-}" ]; then python3 -c "import os,json,base64;p='$GJ/models.manifest.json';m=json.load(open(p));m['sets'].update(json.loads(base64.b64decode(os.environ['GJ_MANIFEST_OVERLAY_B64'])));json.dump(m,open(p,'w'),indent=1);print('[boot] overlay sets',sorted(m['sets']))" || echo "[boot] WARN manifest overlay failed"; fi
python3 "$GJ/fetch_models.py" --sets "$SETS" --root /workspace/models > "$L/fetch.log" 2>&1 &
FPID=$!
for i in $(seq 1 300); do curl -fs http://127.0.0.1:8188/system_stats >/dev/null 2>&1 && { mark comfy_up; break; }; kill -0 $CPID 2>/dev/null || { fail "ComfyUI exited (see comfy.log)"; break; }; sleep 1; done
if wait $FPID; then mark weights_ok; else fail "model fetch failed (see fetch.log)"; fi
cp /workspace/models/FETCH_REPORT.json "$OUT/fetch_report.json" 2>/dev/null
if [ ! -f "$OUT/BOOT_FAILED.txt" ] && curl -fs http://127.0.0.1:8188/system_stats >/dev/null; then echo ready > "$OUT/READY.txt"; mark ready; fi
# Auto-exit watchdog: idle = queue empty and no new history for GJ_IDLE_EXIT_MIN minutes; hard cap = GJ_MAX_SECONDS since boot.
IDLE=${GJ_IDLE_EXIT_MIN:-15}; MAXS=${GJ_MAX_SECONDS:-0}; last=$(date +%s); lastn=-1
while sleep 30; do
  now=$(date +%s); up=$(( now - ${T0%.*} ))
  if [ "$MAXS" -gt 0 ] && [ "$up" -ge "$MAXS" ]; then self_remove "max seconds $MAXS reached"; sleep 120; fi
  q=$(curl -fs http://127.0.0.1:8188/queue 2>/dev/null | python3 -c "import sys,json;d=json.load(sys.stdin);print(len(d['queue_running'])+len(d['queue_pending']))" 2>/dev/null || echo 0)
  n=$(curl -fs "http://127.0.0.1:8188/history?max_items=1000" 2>/dev/null | python3 -c "import sys,json;print(len(json.load(sys.stdin)))" 2>/dev/null || echo 0)
  if [ "$q" != 0 ] || [ "$n" != "$lastn" ]; then last=$now; lastn=$n; fi
  if [ "$IDLE" -gt 0 ] && [ $(( now - last )) -ge $(( IDLE * 60 )) ]; then self_remove "idle ${IDLE} min"; sleep 120; fi
done
