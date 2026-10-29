"""Local, free checks used by the runners and qc/selfcheck.py. Each returns (ok: bool, detail: str)."""
import json, os, re, subprocess
from PIL import Image, ImageStat
def image_check(path, size=None, aspect=None, long_side=None, min_std=10.0, need_rgb=True, tol=0.02):
    """size: exact (w, h); aspect: w/h expected (tol); long_side: exact long side; min_std: blank/washed-out guard; need_rgb: flat 8-bit RGB, no alpha."""
    if not os.path.exists(path): return False, 'missing'
    im = Image.open(path); w, h = im.size; bad = []
    if need_rgb and im.mode != 'RGB': bad.append(f'mode {im.mode} (want flat 8-bit RGB, no alpha)')
    if size and (w, h) != tuple(size): bad.append(f'size {w}x{h} != {size[0]}x{size[1]}')
    if aspect and abs(w / h - aspect) / aspect > tol: bad.append(f'aspect {w / h:.3f} != {aspect:.3f}')
    if long_side and max(w, h) != long_side: bad.append(f'long side {max(w, h)} != {long_side}')
    std = max(ImageStat.Stat(im.convert('RGB')).stddev)
    if std <= min_std: bad.append(f'blank/washed out (pixel std {std:.1f} <= {min_std})')
    return not bad, (f'{w}x{h} {im.mode} std {std:.0f}' if not bad else '; '.join(bad))
def probe(path):
    o = subprocess.run(['ffprobe', '-v', 'error', '-count_frames', '-show_entries', 'stream=codec_type,width,height,r_frame_rate,nb_read_frames,duration',
                        '-show_entries', 'format=duration', '-of', 'json', path], capture_output=True, text=True)
    d = json.loads(o.stdout or '{}'); v = next((s for s in d.get('streams', []) if s['codec_type'] == 'video'), None); a = next((s for s in d.get('streams', []) if s['codec_type'] == 'audio'), None)
    fr = (lambda x: float(x[0]) / float(x[1]))(v['r_frame_rate'].split('/')) if v else 0
    return dict(frames=int(v.get('nb_read_frames') or 0) if v else 0, fps=fr, w=v and v['width'], h=v and v['height'],
                vdur=float(v.get('duration') or 0) if v else 0, adur=float(a.get('duration') or 0) if a else 0, audio=bool(a))
def luma(path, dark=16, washed=235):
    """FAIL if most frames are black (YMAX<=16) or flat white (YMIN>=235)."""
    out = subprocess.run(['ffmpeg', '-v', 'info', '-i', path, '-vf', 'signalstats,metadata=print', '-an', '-f', 'null', '-'], capture_output=True, text=True).stderr
    ymax = [float(x) for x in re.findall(r'YMAX=([0-9.]+)', out)]; ymin = [float(x) for x in re.findall(r'YMIN=([0-9.]+)', out)]
    if not ymax: return False, 'no frames decoded'
    nb = sum(y <= dark for y in ymax); nw = sum(y >= washed for y in ymin)
    if nb > len(ymax) / 2: return False, f'{nb}/{len(ymax)} frames black'
    if nw > len(ymin) / 2: return False, f'{nw}/{len(ymin)} frames washed out'
    return True, f'{len(ymax)} frames, {nb} black, {nw} washed'
def frame_stills(path, n=5, outdir=None):
    """pull n evenly spaced stills from a video (for the size/aspect/blank check on video frames)."""
    p = probe(path); outdir = outdir or os.path.splitext(path)[0] + '_stills'; os.makedirs(outdir, exist_ok=True); out = []
    for k in range(n):
        t = (p['vdur'] or p['frames'] / max(p['fps'], 1)) * (k + 0.5) / n; f = f'{outdir}/f{k}.png'
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', f'{t:.3f}', '-i', path, '-frames:v', '1', f], check=False); out.append(f)
    return out
def cut_audio(src, start, dur, out):
    """source audio window -> wav (48 kHz stereo)."""
    r = subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', f'{start:.4f}', '-t', f'{dur:.4f}', '-i', src, '-vn', '-ac', '2', '-ar', '48000', out], capture_output=True, text=True)
    return r.returncode == 0 and os.path.exists(out), r.stderr[-300:]
def trim_remux(video, audio, frames, fps, out):
    """keep exactly the planned frames (no frozen padded tail) and put the source audio back, cut to the same length. Phone-safe H.264."""
    dur = frames / fps
    r = subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', video, '-i', audio, '-map', '0:v:0', '-map', '1:a:0', '-frames:v', str(frames), '-t', f'{dur:.4f}',
                        '-c:v', 'libx264', '-profile:v', 'high', '-level', '4.2', '-pix_fmt', 'yuv420p', '-colorspace', 'bt709', '-crf', '17',
                        '-c:a', 'aac', '-b:a', '192k', '-movflags', '+faststart', '-shortest', out], capture_output=True, text=True)
    return r.returncode == 0, r.stderr[-300:]
