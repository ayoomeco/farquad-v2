#!/usr/bin/env bash
# A08 polish entry: SeedVR2 upscale, then the H.264 High 4.2 encode at the source frame rate.
# No frame interpolation, no colour grade, no grain, no face restore.
set -euo pipefail
IN=${IN:-/w/in.mp4}
SRC=${SRC:-/w/src.mp4}
FINAL=${FINAL:-/w/a08_final.mp4}
export IN
export OUT=${S1_OUT:-/w/s1_raw.mp4}
bash "$(dirname "$0")/s1.sh"
POST=""
for c in "${POST_A08:-}" "${FARQUAD_HOME:-}/engine/worker/post_a08.py" /opt/jutsu-src/worker/post_a08.py "$(dirname "$0")/../../engine/worker/post_a08.py"; do
  if [ -n "$c" ] && [ -f "$c" ]; then POST=$c; break; fi
done
[ -n "$POST" ] || { echo "POST_FAIL post_a08.py not found"; exit 1; }
A08_UPSCALE=skip python3 "$POST" --encode-only --video "$OUT" --audio "$SRC" --out "$FINAL"
echo POLISH_POST_DONE
