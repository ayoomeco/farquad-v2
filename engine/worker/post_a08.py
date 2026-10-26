"""On-pod A08 stitch, upscale, and encode.

Stitch blends the 22-frame overlap (10-frame hold, then a 12-frame crossfade). That blend is not frame
interpolation. Letterbox bars come from the source canvas; the picture area is the full render, not a
person-mask paste-back. No colour grade, no grain, no face restore.

Upscale is SeedVR2 only. A missing CLI fails the job. Set A08_UPSCALE=skip to encode the stitch as-is
(tests). The final file is H.264 High 4.2, yuv420p, bt709, AAC, +faststart, at the source frame rate.
"""
import argparse, json, os, subprocess, sys
import numpy as np

W, H = 1152, 640
WU = 24
OVERLAP = 22
HOLD = 10


def source_fps(path):
    r = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=r_frame_rate',
         '-of', 'csv=p=0', path],
        capture_output=True, text=True, check=True)
    fps = r.stdout.strip()
    if not fps or fps in ('0/0', 'N/A'):
        raise RuntimeError(f'POST_FAIL no source fps in {path}')
    return fps


def _ratio(value):
    text = str(value)
    if '/' in text:
        a, b = text.split('/', 1)
        return float(a) / float(b)
    return float(text)


def fps_close(a, b, tol=0.02):
    return abs(_ratio(a) - _ratio(b)) < tol


def encode_cmd(width, height, fps, dst, audio_from=None, preset=None):
    """A08 final-encode argv. `fps` is the source rate (passed to -r); it is never rewritten to 30."""
    preset = preset or os.environ.get('A08_X264_PRESET', 'slow')
    cmd = ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{width}x{height}',
           '-r', str(fps), '-i', '-']
    if audio_from:
        cmd += ['-i', audio_from, '-map', '0:v', '-map', '1:a?']
    cmd += [
        '-c:v', 'libx264', '-profile:v', 'high', '-level:v', '4.2', '-pix_fmt', 'yuv420p',
        '-crf', '16', '-preset', preset,
        '-vf', 'scale=out_color_matrix=bt709:out_range=tv',
        '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709',
        '-x264-params', 'colorprim=bt709:transfer=bt709:colormatrix=bt709',
        '-c:a', 'aac', '-b:a', '160k', '-movflags', '+faststart', '-shortest', dst,
    ]
    return cmd


def faststart_ok(path):
    with open(path, 'rb') as f:
        data = f.read(65536)
    moov = data.find(b'moov')
    mdat = data.find(b'mdat')
    if moov < 0:
        return False
    if mdat < 0:
        return True
    return moov < mdat


def probe_streams(path):
    r = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_streams', '-of', 'json', path],
        capture_output=True, text=True, check=True)
    return json.loads(r.stdout).get('streams') or []


def probe_ok(path, src_fps):
    """Return (ok, reason). Level 42 is ffprobe's spelling of High 4.2."""
    try:
        streams = probe_streams(path)
    except Exception as e:
        return False, f'ffprobe failed: {e}'
    video = next((s for s in streams if s.get('codec_type') == 'video'), None)
    audio = next((s for s in streams if s.get('codec_type') == 'audio'), None)
    if not video:
        return False, 'no video stream'
    profile = str(video.get('profile') or '')
    level = video.get('level')
    level_ok = level in (42, '42', 4.2, '4.2')
    if profile.lower() != 'high' or not level_ok:
        return False, f'profile/level {profile}/{level}'
    if video.get('pix_fmt') != 'yuv420p':
        return False, f"pix_fmt {video.get('pix_fmt')}"
    for field in ('color_space', 'color_primaries', 'color_transfer'):
        if video.get(field) != 'bt709':
            return False, f'{field} {video.get(field)}'
    if not fps_close(video.get('r_frame_rate'), src_fps):
        return False, f"fps {video.get('r_frame_rate')} != {src_fps}"
    if not audio or audio.get('codec_name') != 'aac':
        return False, f"audio {None if not audio else audio.get('codec_name')}"
    if not faststart_ok(path):
        return False, 'moov is not before mdat (faststart)'
    return True, 'ok'


def read_bgr(path):
    cap_cmd = ['ffmpeg', '-v', 'error', '-i', path, '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-']
    # Size from ffprobe so a 1080p native and a 1152x640 render both load.
    meta = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height',
         '-of', 'csv=p=0', path],
        capture_output=True, text=True, check=True).stdout.strip().split(',')
    w, h = int(meta[0]), int(meta[1])
    proc = subprocess.Popen(cap_cmd, stdout=subprocess.PIPE)
    frames = []
    nbytes = w * h * 3
    while True:
        buf = proc.stdout.read(nbytes)
        if len(buf) < nbytes:
            break
        frames.append(np.frombuffer(buf, np.uint8).reshape(h, w, 3).copy())
    proc.wait()
    return frames


def blend_pieces(pieces):
    """Crossfade the overlap. Hold the first HOLD frames, then fade across the rest. Not interpolation."""
    out = []
    for frames in pieces:
        frames = list(frames)
        if out:
            n = min(OVERLAP, len(frames), len(out))
            for j in range(n):
                wt = 0.0 if j < HOLD else (j - (HOLD - 1)) / float(OVERLAP - HOLD)
                a = out[-n + j].astype(np.float32)
                b = frames[j].astype(np.float32)
                out[-n + j] = np.clip(a * (1 - wt) + b * wt, 0, 255).astype(np.uint8)
            frames = frames[n:]
        out.extend(frames)
    return out[WU:]


def _active_box(src_frames):
    step = max(1, len(src_frames) // 40)
    gray = [np.max(f, axis=2) for f in src_frames[::step]]
    g = np.max(np.stack(gray, 0), 0)
    rows = np.where(g.max(1) > 28)[0]
    cols = np.where(g.max(0) > 28)[0]
    if len(rows) == 0 or len(cols) == 0:
        h, w = src_frames[0].shape[:2]
        return 0, h, 0, w
    return int(rows[0]), int(rows[-1] + 1), int(cols[0]), int(cols[-1] + 1)


def _keepbars(frame, src, box, fit_window):
    """Bars stay the source's bars. The active picture is the render, not a masked paste of a person."""
    ay0, ay1, ax0, ax1 = box
    y0, x0, h2, w2 = fit_window
    crop = frame[y0:y0 + h2, x0:x0 + w2]
    placed = cv2_resize(crop, (src.shape[1], src.shape[0]))
    out = src.copy()
    out[ay0:ay1, ax0:ax1] = placed[ay0:ay1, ax0:ax1]
    return out


def cv2_resize(frame, wh):
    import cv2
    return cv2.resize(frame, wh, interpolation=cv2.INTER_AREA)


def letterbox(render_frames, src_frames):
    import cv2
    sh, sw = src_frames[0].shape[:2]
    box = _active_box(src_frames)
    s = min(W / sw, H / sh)
    w2, h2 = int(sw * s) // 2 * 2, int(sh * s) // 2 * 2
    x0, y0 = (W - w2) // 2, (H - h2) // 2
    n = min(len(render_frames), len(src_frames))
    out = []
    for i in range(n):
        frame = render_frames[i]
        if frame.shape[1] != W or frame.shape[0] != H:
            frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
        out.append(_keepbars(frame, src_frames[i], box, (y0, x0, h2, w2)))
    return out


def write_raw(frames, dst, fps, audio_from, preset=None):
    if not frames:
        raise RuntimeError('POST_FAIL no frames to encode')
    h, w = frames[0].shape[:2]
    if audio_from and not os.path.exists(audio_from):
        raise RuntimeError(f'POST_FAIL audio missing: {audio_from}')
    cmd = encode_cmd(w, h, fps, dst, audio_from=audio_from, preset=preset)
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for frame in frames:
        proc.stdin.write(np.ascontiguousarray(frame).tobytes())
    proc.stdin.close()
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f'POST_FAIL ffmpeg encode rc={rc}')
    ok, why = probe_ok(dst, fps)
    if not ok:
        raise RuntimeError(f'POST_FAIL encode: {why}')
    return dst


def find_seedvr():
    env = os.environ.get('SEEDVR_CLI')
    candidates = [env, '/w/sv/inference_cli.py', '/opt/SeedVR2/inference_cli.py',
                  os.path.expanduser('~/SeedVR2/inference_cli.py')]
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return None


def upscale(src, dst):
    mode = os.environ.get('A08_UPSCALE', 'seedvr')
    if mode == 'skip':
        subprocess.check_call(['ffmpeg', '-y', '-v', 'error', '-i', src, '-c', 'copy', dst])
        return dst
    cli = find_seedvr()
    if not cli:
        raise RuntimeError('POST_FAIL upscale: SeedVR2 is not installed')
    cmd = [sys.executable, cli, src, '--output', dst, '--dit_model', os.environ.get('SEEDVR_MODEL', 'seedvr2_ema_7b_fp16.safetensors'),
           '--resolution', os.environ.get('SEEDVR_RES', '1080'), '--batch_size', os.environ.get('SEEDVR_BS', '33'),
           '--uniform_batch_size', '--temporal_overlap', '4', '--prepend_frames', '4',
           '--video_backend', 'ffmpeg', '--vae_decode_tiled', '--vae_encode_tiled']
    subprocess.check_call(cmd, cwd=os.path.dirname(cli) or None)
    return dst


def stitch_and_encode(pieces, native, out, side=None, preset=None):
    fps = source_fps(native)
    piece_frames = [read_bgr(p) for p in pieces]
    blended = blend_pieces(piece_frames)
    src = read_bgr(native)
    delivered = letterbox(blended, src)
    staged = out + '.stitched.mp4'
    write_raw(delivered, staged, fps, native, preset=preset)
    up = out + '.upscaled.mp4'
    upscale(staged, up)
    # Re-encode the upscaled picture with the A08 flags and the source audio, at the source fps.
    up_frames = read_bgr(up)
    write_raw(up_frames, out, fps, native, preset=preset)
    if side:
        import cv2
        paired = []
        n = min(len(delivered), len(src))
        for i in range(n):
            a = src[i]
            b = delivered[i]
            if a.shape != b.shape:
                b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
            paired.append(np.hstack([a, b]))
        write_raw(paired, side, fps, native, preset=preset)
    for tmp in (staged, up):
        if os.path.exists(tmp) and os.environ.get('A08_KEEP_STAGED') != '1':
            os.remove(tmp)
    scores = {
        'pass': True, 'fps': fps, 'out': out, 'upscale': os.environ.get('A08_UPSCALE', 'seedvr'),
        'grade': False, 'grain': False, 'interpolate': False, 'face_restore': False, 'paste_back': False,
    }
    json.dump(scores, open(os.path.splitext(out)[0] + '.scores.json', 'w'), indent=1)
    print('POST_DONE', os.path.basename(out), flush=True)
    return scores


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--pieces', nargs='*', default=[])
    ap.add_argument('--native')
    ap.add_argument('--out')
    ap.add_argument('--side')
    ap.add_argument('--encode-only', action='store_true')
    ap.add_argument('--video')
    ap.add_argument('--audio')
    ap.add_argument('--preset', default=None)
    args = ap.parse_args(argv)
    if args.encode_only:
        fps = source_fps(args.audio or args.video)
        frames = read_bgr(args.video)
        write_raw(frames, args.out, fps, args.audio, preset=args.preset)
        print('POST_DONE', os.path.basename(args.out), flush=True)
        return 0
    if not args.pieces or not args.native or not args.out:
        print('POST_FAIL need --pieces, --native, and --out', flush=True)
        return 1
    try:
        stitch_and_encode(args.pieces, args.native, args.out, side=args.side, preset=args.preset)
    except Exception as e:
        print(str(e), flush=True)
        if 'POST_FAIL' not in str(e):
            print('POST_FAIL', e, flush=True)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
