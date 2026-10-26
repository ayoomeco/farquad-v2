set -eu
cd /w
(apt-get update -qq && apt-get install -y -qq ffmpeg >/dev/null) &
git clone -q https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git sv &
wait
pip install -q -r sv/requirements.txt 2>&1 | tail -3
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name())"
ffmpeg -version | head -1
echo SETUP_DONE
