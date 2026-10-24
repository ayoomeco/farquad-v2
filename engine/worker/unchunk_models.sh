#!/usr/bin/env bash
# Reassemble baked model chunks (build_baked_crane.sh) into $MODEL_ROOT. Idempotent; size-checked against the manifest by
# the normal bootstrap pass afterwards (which then downloads nothing). v2: optional groups live in /opt/jm-chunks-opt/<group>
# and are only unpacked when JUTSU_UNCHUNK_OPT=<group> (start.sh sets it from JUTSU_ENABLE_FASTH3=1).
set -euo pipefail; R=${MODEL_ROOT:-/opt/jm}; S=/opt/jm-chunks
[ -n "${JUTSU_UNCHUNK_OPT:-}" ] && S=/opt/jm-chunks-opt/$JUTSU_UNCHUNK_OPT
[ -d "$S" ] || exit 0
find "$S" -name '*.chunk.000' | while IFS= read -r c0; do
  base=${c0%.chunk.000}; rel=${base#"$S/"}; out="$R/$rel"; mkdir -p "$(dirname "$out")"
  [ -f "$out" ] || { cat "$base".chunk.* > "$out.part" && mv "$out.part" "$out" && rm -f "$base".chunk.*; }
done; echo "[unchunk] done $S"
