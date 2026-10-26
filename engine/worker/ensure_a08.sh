#!/usr/bin/env bash
# A08 pod gate. ComfyUI must be 0.42.5 or newer. The vendored h3_exact_lora node is copied in on every boot.
# An older checkout is upgraded to the pinned 0.42.5 tag (tarball sha checked). Torch is not changed.
# Already-newer checkouts are left on that version. Exits 3 when the version is still too old.
set -euo pipefail
MIN=${A08_COMFY_MIN:-0.42.5}
PIN_SHA=d779e73af13e620c80af0ae00c60629e73578988f4ab80522aa0c0a082267d42
PIN_URL=https://github.com/Comfy-Org/ComfyUI/archive/refs/tags/v0.42.5.tar.gz
HERE=$(cd "$(dirname "$0")" && pwd)
JUTSU=${JUTSU_PREFIX:-${PREFIX:-/opt/jutsu}}
C=""
PY=""
for p in "$JUTSU" /opt/jutsu /workspace/jutsu; do
  if [ -n "$p" ] && [ -d "$p/ComfyUI" ]; then C="$p/ComfyUI"; PY="$p/venv/bin/python"; break; fi
done
[ -n "$C" ] || { echo "A08_FAIL ComfyUI directory not found"; exit 3; }
[ -x "$PY" ] || PY=python3
ver() { (cd "$C" && "$PY" -c "import comfyui_version; print(comfyui_version.__version__)"); }
at_least() {
  CUR="$1" MINV="$2" "$PY" - <<'PY'
import os, sys
def vt(v):
    out = []
    for piece in str(v).split('.'):
        digits = ''
        for ch in piece:
            if ch.isdigit():
                digits += ch
            else:
                break
        if not digits:
            break
        out.append(int(digits))
    return tuple(out)
a, b = vt(os.environ['CUR']), vt(os.environ['MINV'])
n = max(len(a), len(b))
a, b = a + (0,) * (n - len(a)), b + (0,) * (n - len(b))
sys.exit(0 if a >= b else 1)
PY
}
copy_node() {
  src="$HERE/custom_nodes/h3_exact_lora"
  [ -f "$src/__init__.py" ] && [ -f "$src/check.py" ] || { echo "A08_FAIL h3_exact_lora missing from $src"; exit 3; }
  rm -rf "$C/custom_nodes/h3_exact_lora"
  mkdir -p "$C/custom_nodes"
  cp -a "$src" "$C/custom_nodes/h3_exact_lora"
  rm -rf "$C/custom_nodes/h3_exact_lora/__pycache__"
}
CUR=$(ver || true)
echo "[a08] ComfyUI ${CUR:-unknown} at $C (need >= $MIN)"
if [ -z "$CUR" ] || ! at_least "$CUR" "$MIN"; then
  echo "[a08] upgrading ComfyUI to v0.42.5"
  UP=0
  if [ -d "$C/.git" ]; then
    git -C "$C" fetch --depth 1 origin refs/tags/v0.42.5:refs/tags/v0.42.5 && git -C "$C" checkout -f v0.42.5 && UP=1 || true
  fi
  if [ "$UP" != 1 ]; then
    tmp=$(mktemp -d)
    curl -fsSL "$PIN_URL" -o "$tmp/c.tgz"
    echo "$PIN_SHA  $tmp/c.tgz" | sha256sum -c --quiet -
    tar -xzf "$tmp/c.tgz" -C "$tmp"
    cp -a "$tmp/ComfyUI-0.42.5/." "$C/"
    rm -rf "$tmp"
    UP=1
  fi
  TV=$("$PY" -c "import torch; print(torch.__version__)")
  PKGS="comfyui-frontend-package==1.55.18 comfyui-workflow-templates==0.11.96 comfyui-embedded-docs==0.5.14 comfy-kitchen==0.2.39 comfy-aimdo==0.5.6"
  if command -v uv >/dev/null 2>&1; then uv pip install -p "$PY" $PKGS; else "$PY" -m pip install $PKGS; fi
  TV2=$("$PY" -c "import torch; print(torch.__version__)")
  [ "$TV" = "$TV2" ] || { echo "A08_FAIL torch changed $TV -> $TV2"; exit 4; }
  CUR=$(ver || true)
fi
copy_node
if [ -z "$CUR" ] || ! at_least "$CUR" "$MIN"; then
  echo "A08_FAIL ComfyUI ${CUR:-unknown} is older than $MIN"
  exit 3
fi
echo "A08_COMFY_OK $CUR"
