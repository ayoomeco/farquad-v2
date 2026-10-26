# A08 keeps the source frame rate. Frame interpolation stays off unless ALLOW_RIFE=1.
import os, sys
if os.environ.get('ALLOW_RIFE') != '1':
    sys.exit('A08 keeps source fps: s3.py frame interpolation is off (set ALLOW_RIFE=1 to override)')
# Stage 3: RIFE 4.26 exact-timestamp interpolation 24 -> FPS (60 or 30). Output frame k at t=k*24/FPS.
import subprocess, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, '/w/rife'); os.chdir('/w/rife')
from train_log.RIFE_HDv3 import Model
IN, OUT, FPS = sys.argv[1], sys.argv[2], int(sys.argv[3]); W, H = 1920, 1080
m = Model(); m.load_model('train_log', -1); m.eval(); m.device()
pr = subprocess.Popen(['ffmpeg', '-v', 'error', '-i', IN, '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], stdout=subprocess.PIPE)
fr = []
while len(b := pr.stdout.read(W * H * 3)) == W * H * 3:
    fr.append(np.frombuffer(b, np.uint8).reshape(H, W, 3))
n = len(fr); N = round(n * FPS / 24); print('in', n, 'out', N, flush=True)
ph, pw = ((H - 1) // 64 + 1) * 64, ((W - 1) // 64 + 1) * 64
T = lambda a: F.pad(torch.from_numpy(a.copy()).cuda().permute(2, 0, 1)[None].float() / 255., (0, pw - W, 0, ph - H))
wp = subprocess.Popen(['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-crf', '12', '-preset', 'slow', '-pix_fmt', 'yuv420p', OUT], stdin=subprocess.PIPE)
for k in range(N):
    t = k * 24 / FPS; i = int(np.floor(t + 1e-6)); s = t - i
    if s < 1e-4 or i >= n - 1:
        o = fr[min(i, n - 1)]
    else:
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            r = m.inference(T(fr[i]), T(fr[i + 1]), s, 1.0)
        o = (r[0, :, :H, :W].float().clamp(0, 1) * 255).round().byte().permute(1, 2, 0).cpu().numpy()
    wp.stdin.write(o.tobytes())
wp.stdin.close(); wp.wait(); print('S3_DONE', N)
