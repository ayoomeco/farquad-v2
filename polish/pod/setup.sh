set -x; cd /w
(apt-get update -qq && apt-get install -y -qq ffmpeg >/dev/null) &
git clone -q https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git sv &
git clone -q https://github.com/sczhou/CodeFormer.git cf &
git clone -q https://github.com/hzwer/Practical-RIFE.git rife &
wait
pip install -q -r sv/requirements.txt scikit-image lpips addict yapf lmdb 2>&1 | tail -3
cd cf && python basicsr/setup.py develop >/dev/null 2>&1; python scripts/download_pretrained_models.py facelib >/dev/null 2>&1; python scripts/download_pretrained_models.py CodeFormer >/dev/null 2>&1; cd ..
mkdir -p rife/train_log; for f in IFNet_HDv3.py RIFE_HDv3.py refine.py flownet.pkl; do curl -sL -o rife/train_log/$f https://huggingface.co/MonsterMMORPG/RIFE_4_26/resolve/main/train_log/$f; done
ls -la cf/weights/*/ rife/train_log; python -c "import torch;print(torch.__version__,torch.cuda.get_device_name())"; ffmpeg -version|head -1
echo SETUP_DONE
