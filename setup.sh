#!/usr/bin/env bash
# Fire Marshal Bill setup. FREE: rents no GPU, spends nothing, prints no keys. Self-contained: needs nothing outside this folder.
set -euo pipefail
FMB_HOME="${FMB_HOME:-$(cd "$(dirname "$0")" && pwd)}"; export FMB_HOME
FMB_DATA="${FMB_DATA:-$HOME/fmb-data}"; export FMB_DATA
export FMB_JOB="${FMB_JOB:-$FMB_HOME/jobs/example_job.json}"
for t in python3 ffmpeg ffprobe curl; do command -v $t >/dev/null || { echo "missing $t (apt-get install -y ffmpeg curl python3-venv)"; exit 1; }; done
python3 -c 'import sys; assert sys.version_info >= (3,10), sys.version' || { echo "need Python 3.10+"; exit 1; }
mkdir -p "$FMB_DATA"/{run,cv}
[ -d "$FMB_HOME/venv" ] || python3 -m venv "$FMB_HOME/venv"
. "$FMB_HOME/venv/bin/activate"
pip install -q --upgrade pip && pip install -q requests pillow numpy
if [ "${FMB_MASKS:-1}" = "1" ]; then
  pip install -q ultralytics opencv-python-headless
  [ -s "$FMB_DATA/cv/yolo11m-seg.pt" ] || curl -fsSL -o "$FMB_DATA/cv/yolo11m-seg.pt" https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m-seg.pt
  [ -s "$FMB_DATA/cv/sam2.1_t.pt" ]    || curl -fsSL -o "$FMB_DATA/cv/sam2.1_t.pt"    https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2.1_t.pt
fi
python3 -m py_compile "$FMB_HOME"/lib/*.py "$FMB_HOME"/video/*.py "$FMB_HOME"/refs/*.py "$FMB_HOME"/qc/*.py "$FMB_HOME"/demo/*.py "$FMB_HOME"/dbx/*.py
python3 -c "import json; json.load(open('$FMB_JOB')); m=json.load(open('$FMB_HOME/image/models.manifest.json')); assert {'h3_a08','seedvr2','qwen21','qwen2511_lite'} <= set(m['sets'])"
for k in RUNPOD_API_KEY DROPBOX_TOKEN HF_TOKEN; do if [ -n "${!k:-}" ]; then echo "$k: set"; else echo "$k: missing"; fi; done
echo FMB_SETUP_OK
