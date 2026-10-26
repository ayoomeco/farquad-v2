# GJ V2 polish pod: SeedVR2 upscale only. No frame interpolation, no face restore, no grain.
set -euo pipefail
cd /w
apt-get update -qq && apt-get install -y -qq ffmpeg >/dev/null
git clone -q https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git sv
pip install -q -r sv/requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name())"
ffmpeg -version | head -1
echo "GJ V2 polish: SeedVR2 only. No RIFE, no CodeFormer, no grain."
echo SETUP_DONE
