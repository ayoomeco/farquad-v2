#!/usr/bin/env bash
# Farquad box setup (run once after every File block is written under $FARQUAD_HOME).
# Makes a venv with the box-side Python deps, downloads the CPU models (sha256-pinned), makes the work folders,
# then runs a free import + dry test. No GPU, no spend, no keys needed.
set -euo pipefail
FH="${FARQUAD_HOME:-$HOME/farquad}"; export FARQUAD_HOME="$FH"
python3 -c 'import sys; assert sys.version_info>=(3,11), "Farquad needs Python 3.11+ (tomllib)"'
for b in ffmpeg ffprobe curl; do command -v $b >/dev/null || { echo "missing $b (e.g. apt-get install -y ffmpeg curl)"; exit 1; }; done
mkdir -p "$FH"/{models,plan/graphs,engine/data/bakeoff/clips}
[ -d "$FH/venv" ] || python3 -m venv "$FH/venv"
. "$FH/venv/bin/activate"
python -m pip install -q --upgrade pip
python -m pip install -q numpy 'opencv-python>=4.8' pillow pydantic httpx websocket-client ultralytics
dl(){ local url=$1 out=$2 sha=$3
  if [ -f "$out" ] && echo "$sha  $out" | sha256sum -c --status; then return 0; fi
  curl -fsSL "$url" -o "$out.part"
  echo "$sha  $out.part" | sha256sum -c --status || { echo "sha256 mismatch: $out"; rm -f "$out.part"; exit 1; }
  mv "$out.part" "$out"; }
dl https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx \
   "$FH/models/face_detection_yunet_2023mar.onnx" 8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4
dl https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx \
   "$FH/models/face_recognition_sface_2021dec.onnx" 0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79
dl https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l-seg.pt \
   "$FH/models/yolo11l-seg.pt" cabe90049795dfc9a370b7934d6dec7f6b9e44a20e573b0ff81b7e205512c872
chmod +x "$FH"/*.sh "$FH"/polish/pod/*.sh "$FH"/engine/worker/*.sh 2>/dev/null || true
cd "$FH/plan"
python -c "import sys; sys.path.insert(0,'$FH/lib'); import fx, qc_v5b3, graphs_v5b; print('imports ok')"
python test_a08_defaults.py
DRYCHECK=1 DSR_RUN=/tmp python driver_v5b3.py
DRY=1 DRY_STATE='{"pods":[],"bal":10}' python idle_watchdog.py
(cd "$FH/engine" && python -c "import sys; sys.path.insert(0,'scripts'); sys.argv=['x']; import phase1_bakeoff; print('engine import ok')")
echo "FARQUAD_SETUP_OK  (activate with: . $FH/venv/bin/activate)"
