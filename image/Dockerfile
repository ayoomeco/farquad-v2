# GJ fastboot worker, ComfyUI v0.42.5. Tag: ghcr.io/ayoomeco/genie-jutsu-worker:fastboot-comfy-0.42.5
# (rollback: ghcr.io/ayoomeco/genie-jutsu-worker:fastboot-v0.39.1 is left untouched).
# Base = RunPod's official PyTorch image (fleet-cached on RunPod hosts), torch 2.13.0 + CUDA 13.0 + Python 3.12.
# torch is NOT reinstalled (install_fastboot.sh pins it with a constraints file and fails the build if it changes).
# ComfyUI + pinned nodes + h3_exact_lora + Python deps are baked. NO models, NO secrets: boot.sh fetches only the job's model sets.
ARG BASE=runpod/pytorch:1.5.0-cu1300-torch2130-ubuntu2404
FROM ${BASE}
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 HF_XET_HIGH_PERFORMANCE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PIP_BREAK_SYSTEM_PACKAGES=1 \
    HF_HOME=/root/.cache/huggingface GJ_MODEL_SETS=h3_a08
RUN apt-get update && apt-get install -y --no-install-recommends aria2 ffmpeg libgl1 libglib2.0-0 libsndfile1 git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY fastboot/ /opt/gj/
RUN chmod +x /opt/gj/*.sh /opt/gj/*.py && GJ=/opt/gj bash /opt/gj/install_fastboot.sh
LABEL org.opencontainers.image.source="https://github.com/ayoomeco/genie-jutsu-worker" \
      org.opencontainers.image.description="GJ fastboot: ComfyUI v0.42.5 + KJNodes/VHS/krea2edit + h3_exact_lora, base torch 2.13.0 cu130, no models"
EXPOSE 8188
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s CMD curl -fs http://127.0.0.1:8188/system_stats || exit 1
ENTRYPOINT []
CMD ["bash", "/opt/gj/boot.sh"]
