# A08 does not colour-grade or add grain. s4.py stays off unless ALLOW_GRADE=1.
import os, sys
if os.environ.get('ALLOW_GRADE') != '1':
    sys.exit('A08 has no colour grade and no grain: s4.py is off (set ALLOW_GRADE=1 to override)')
# Stage 4: LAB colour match to the ORIGINAL source (time-aligned, temporally smoothed stats) + subtle film grain.
import subprocess, numpy as np, cv2
IN, SRC, OUT, FPS = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]); W, H = 1920, 1080
def rd(p):
    pr = subprocess.Popen(['ffmpeg', '-v', 'error', '-i', p, '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'], stdout=subprocess.PIPE)
    while len(b := pr.stdout.read(W * H * 3)) == W * H * 3:
        yield np.frombuffer(b, np.uint8).reshape(H, W, 3)
def st(f):
    l = cv2.cvtColor(cv2.resize(f, (480, 270)), cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32); return l.mean(0), l.std(0) + 1e-3
S = [st(f) for f in rd(SRC)]; R = [st(f) for f in rd(IN)]
def sm(a, k=9):
    a = np.array(a); return np.stack([np.convolve(np.pad(a[:, j], (k // 2, k // 2), 'edge'), np.ones(k) / k, 'valid') for j in range(a.shape[1])], 1)
sm_s, ss_s = sm([s[0] for s in S]), sm([s[1] for s in S]); rm, rs = sm([r[0] for r in R], int(9 * FPS / 24) | 1), sm([r[1] for r in R], int(9 * FPS / 24) | 1)
rng = np.random.default_rng(7)
wp = subprocess.Popen(['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-crf', '14', '-preset', 'slow', '-tune', 'grain', '-pix_fmt', 'yuv420p', OUT], stdin=subprocess.PIPE)
for k, f in enumerate(rd(IN)):
    j = min(int(round(k * 24 / FPS)), len(S) - 1); STR = 0.8
    l = cv2.cvtColor(f, cv2.COLOR_BGR2LAB).astype(np.float32)
    tm, ts = sm_s[j], ss_s[j]; m, s = rm[k], rs[k]
    g = (l - m) * (ts / s) + tm; l = l * (1 - STR) + g * STR
    o = cv2.cvtColor(np.clip(l, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)
    n = cv2.GaussianBlur(rng.normal(0, 1, (H // 2, W // 2)).astype(np.float32), (0, 0), 0.6); n = cv2.resize(n, (W, H), interpolation=cv2.INTER_LINEAR)
    lum = o.mean(2, keepdims=True) / 255.; amp = 3.2 * (0.6 + 0.8 * lum * (1 - lum) * 4) / 1.4
    wp.stdin.write(np.clip(o + n[..., None] * amp, 0, 255).astype(np.uint8).tobytes())
wp.stdin.close(); wp.wait(); print('S4_DONE')
