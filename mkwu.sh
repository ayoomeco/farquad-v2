# mkwu <src> <start> <dur> <outdir> <slug> <tag> [pingpong]
src=$1; s=$2; d=$3; od=$4; slug=$5; tag=$6; pp=$7
enc="-c:v libx264 -preset slow -crf 12 -pix_fmt yuv420p"
ffmpeg -y -v error -ss $s -i "$src" -t $d -map 0:v:0 -map 0:a:0 -vf scale=1920:1080:flags=lanczos $enc -c:a aac -b:a 256k -movflags +faststart $od/${slug}_${tag}_native.mp4
if [ -z "$pp" ]; then
  ws=$(python3 -c "print(max(0,$s-1.0))")
  ffmpeg -y -v error -ss $ws -i "$src" -t $(python3 -c "print($d+1.0)") -map 0:v:0 -an -vf scale=1920:1080:flags=lanczos $enc $od/${slug}_wu.mp4
else
  ffmpeg -y -v error -i $od/${slug}_${tag}_native.mp4 -filter_complex "[0:v]trim=start_frame=1:end_frame=25,setpts=PTS-STARTPTS,reverse[r];[0:v]setpts=PTS-STARTPTS[f];[r][f]concat=n=2:v=1[o]" -map "[o]" -an $enc $od/${slug}_wu.mp4
fi
echo $slug $(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 $od/${slug}_${tag}_native.mp4) $(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 $od/${slug}_wu.mp4)
