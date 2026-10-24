#!/usr/bin/env bash
# WanGP (deepbeepmeep/Wan2GP v17.17, MMGP v4) in its OWN venv next to ComfyUI (Astra brief A1: H3 on MMGP vs ComfyUI).
# Pinned commit; torch 2.10 cu130 + SageAttention 2.2.0 per docs/INSTALLATION.md (Python 3.11 via uv). No GPU needed to install.
# Models are NOT pulled here: WanGP downloads the H3 Ref2VA pruned PDD files itself on the first (cold, discarded) run.
set -eu
WANGP_SHA=${WANGP_SHA:-6479db36bdc2619a904a852bba9c2d78e1a83f82}
D=${WANGP_DIR:-/opt/wangp}
mkdir -p "$D"
command -v uv >/dev/null 2>&1 || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
if [ ! -d "$D/src/.git" ]; then
  git clone --filter=blob:none https://github.com/deepbeepmeep/Wan2GP.git "$D/src"
fi
git -C "$D/src" checkout -q "$WANGP_SHA"
uv venv -p 3.11 "$D/venv"
PIP="uv pip install -p $D/venv/bin/python"
$PIP torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu130
$PIP -r "$D/src/requirements.txt"
if command -v nvcc >/dev/null 2>&1; then
  tmp=$(mktemp -d) && git clone --depth 1 --branch v2.2.0 https://github.com/thu-ml/SageAttention.git "$tmp/sage" \
    && $PIP "setuptools<=75.8.2" wheel ninja packaging \
    && (cd "$tmp/sage" && TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}" MAX_JOBS=${MAX_JOBS:-8} $PIP --no-build-isolation .) \
    && echo "[wangp] SageAttention 2.2.0 built" || echo "[wangp] WARN Sage2 build failed: driver falls back to sdpa"
fi
"$D/venv/bin/python" -c "import torch; print('[wangp] torch', torch.__version__, 'cuda', torch.version.cuda)"
"$D/venv/bin/python" -c "import sageattention" 2>/dev/null && touch "$D/sage.ok" || true
touch "$D/install.ok"
