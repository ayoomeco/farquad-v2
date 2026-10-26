# Stage 1: SeedVR2 7B fp16 -> 1080p. No colour correction (A08 keeps the render's colour).
set -euo pipefail
IN=${IN:-/w/in.mp4}
OUT=${OUT:-/w/s1_raw.mp4}
LR=${LR:-/w/lr.mp4}
cd /w
ffmpeg -y -v error -i "$IN" -vf scale=1152:648:flags=area -c:v libx264 -crf 10 -pix_fmt yuv420p "$LR"
cd sv
M=${M:-seedvr2_ema_7b_fp16.safetensors}
python inference_cli.py "$LR" --output "$OUT" --dit_model $M --resolution ${RES:-1080} --batch_size ${BS:-33} --uniform_batch_size --temporal_overlap 4 --prepend_frames 4 --video_backend ffmpeg --vae_decode_tiled --vae_encode_tiled ${EXTRA:-} 2>&1 | grep -v "it/s\]" | tail -40 || true
# SeedVR's progress noise is filtered above; a missing output is still a failure.
[ -s "$OUT" ]
cd /w
ffprobe -v error -count_frames -show_entries stream=width,height,nb_read_frames -of csv=p=0 "$OUT"
