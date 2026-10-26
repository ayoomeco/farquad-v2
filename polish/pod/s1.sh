# GJ V2 stage 1: SeedVR2 7B fp16 -> 1080p. Upscale only (no colour correction, no face restore).
cd /w; ffmpeg -y -v error -i in.mp4 -vf scale=1152:648:flags=area -c:v libx264 -crf 10 -pix_fmt yuv420p lr.mp4
cd sv; M=${M:-seedvr2_ema_7b_fp16.safetensors}
python inference_cli.py /w/lr.mp4 --output /w/s1_raw.mp4 --dit_model $M --resolution ${RES:-1080} --batch_size ${BS:-33} --uniform_batch_size --temporal_overlap 4 --prepend_frames 4 --video_backend ffmpeg --vae_decode_tiled --vae_encode_tiled ${EXTRA} 2>&1 | grep -v "it/s\]" | tail -40
cd /w; ffprobe -v error -count_frames -show_entries stream=width,height,nb_read_frames -of csv=p=0 ${OUT:-s1_raw.mp4}
