"""Oct 7b identity fix for the H3 Ref2VA head-to-head (the main man kept the source actor's face in every H3 rung).

Root causes (see docs/H2H-IDENTITY-FIX.md):
  1. The Context-IR prompt broke the H3 full-reference label rules (MiniMax VIDEO_PROMPT_WRITING_GUIDE_ref_en.md):
     <Video 1> was `fully_preserved` (= keep the whole video, actor included), <Subject 1> used `reference` (an AUDIO-only
     marker, invalid for visible content) and the source actor was never defined as a subject, so nothing told H3 to
     transfer only his outfit/motion (`attribute_transfer`). Shot times were "0.7-1.2 s" instead of "At 00:00.700".
  2. Identity leakage: the raw RGB motion ref (960x544) shows the actor's face sharply in 5 shots; in editing mode H3 copies it.
     Fix: 384 short edge (community MiniMaxH3TaggedMotionReference default) and blank / pre-swap the actor's face in the ref.
  3. Weak identity signal: one lying-down selfie scaled to the generation area ('match'), face ~285 px. Fix: a tight
     head-and-shoulders crop as <Picture 1> (face ~480 px) plus the full photo as <Picture 2>.

Box-side and free: face detection (OpenCV YuNet + SFace, optional dependency) picks the main man's faces only (cosine to two
anchor frames, propagated inside a shot), then the motion ref is rendered with that face greyed out ('mask') or roughly
replaced by the ref face ('paste'). The graphs are the h2h_H3-P4 graph with prompt/refs/motion swapped; they are queued on
the live launch through its extra_runs.jsonl channel (the launcher's own budget gate still applies)."""
from __future__ import annotations
import argparse, copy, json, pathlib, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = pathlib.Path(__file__).resolve().parent.parent
CLIPS = ROOT / 'data' / 'bakeoff' / 'clips'
H2H = ROOT / 'data' / 'bakeoff' / 'h2h'
FPS = 24
VISIBLE_MARKERS = ('fully_preserved', 'partially_preserved', 'attribute_transfer', 'weak_reference')
AUDIO_MARKERS = ('fully_copy', 'partially_copy', 'reference', 'weak_reference')


# ---------------------------------------------------------------- prompt (H3 full-reference rules)
def ts(t: float) -> str:
    ms = int(round(t * 1000))
    return f'{ms // 60000:02d}:{(ms // 1000) % 60:02d}.{ms % 1000:03d}'


def h3_prompt_v2(b: dict, seconds: float, mode: str = 'edit', n_pictures: int = 2, face_note: str = '') -> str:
    """mode 'edit'  : [video editing + audio reuse], <Video 1> partially_preserved, actor = <Subject 2> attribute_transfer.
       mode 'motion': [reference generation + audio reuse], <Video 1> only gives cuts/camera/motion (Higgsfield motion-transfer)."""
    if mode not in ('edit', 'motion'):
        raise ValueError(mode)
    shots = b.get('shots') or [[0.0, seconds, True, '<Subject 1> follows <Video 1> moment by moment']]
    main = [f'[Shot {i + 1}]' for i, s in enumerate(shots) if s[2]] or ['[Shot 1]']
    pics = ' and '.join(f'<Picture {i + 1}>' for i in range(n_pictures))
    pic_src = ('<Picture 1> (tight head-and-shoulders crop) and <Picture 2> (the full selfie)' if n_pictures >= 2 else '<Picture 1>')
    look = b.get('look', 'face, deep brown skin, short dark twisted hair, goatee and silver nose stud')
    inline = b.get('inline', "<Subject 1> (deep brown skin, short dark twisted hair under <Subject 2>'s grey cap, goatee, silver nose stud)")
    role = b['role']                                     # "the main man in <Video 1> (grey baseball cap, ...)"
    s2 = f'<Subject 2> is {role}{(" " + face_note) if face_note else ""}.'
    if mode == 'edit':
        vdef = '<Video 1> is the source video for the target video edit.'
        summ = (f'[video editing + audio reuse] The target video is an edited version of <Video 1>. In every shot where <Subject 2> '
                f'appears he is recast as <Subject 1>: the man now has <Subject 1>\'s {look} from {pics}, while he wears <Subject 2>\'s outfit, '
                f'holds the same props and performs <Subject 2>\'s exact motion. Every cut, camera move, set, villain and the timing '
                'follow <Video 1>, and <Audio 1> is kept.')
        vret = ('<Video 1> (source video editing): partially_preserved - every cut, framing, camera move, set, lighting, villain and '
                'the timing are kept; only the main man\'s face and identity are replaced by <Subject 1>.')
    else:
        vdef = '<Video 1> is the motion, cut and camera reference: its shot order, cut timing, framing, camera moves and body motion are followed.'
        summ = (f'[reference generation + audio reuse] The target video shows <Subject 1>, with the {look} from {pics}, performing '
                f'<Subject 2>\'s motion shot by shot, with the cut timing, framing and camera moves of <Video 1>. <Audio 1> is kept.')
        vret = ('<Video 1> (motion, cut and camera structure): partially_preserved - shot order, cut timing, framing, camera moves and '
                'body motion are followed; no person\'s face or identity is taken from it.')
    desc = [b.get('style', '')]
    for i, (t0, t1, has, txt) in enumerate(shots):
        lead = f'[Shot {i + 1}]' if i == 0 else f'[Shot {i + 1}] At {ts(t0)},'
        if has:
            txt = txt.replace('<Subject 1>', inline, 1)
        desc.append(f'{lead} {(txt[0].upper() if i == 0 else txt[0].lower()) + txt[1:]}.')
    return '\n'.join([
        'subject_definitions:',
        f'<Subject 1> is {b["subject"]}; his identity comes from {pic_src}. Only his face, head, hair, skin tone and facial hair come '
        'from the photos; his body is the same build as <Subject 2>.',
        s2 + ' Only his outfit, props, poses, body motion and expressions are used; his face, skin tone and identity are not used.',
        vdef,
        '<Audio 1> is the synchronized audio track of <Video 1> and is reused in the target video.', '',
        'summary:', summ, '',
        'retention_analysis:',
        f'<Subject 1> (appears in {", ".join(main)}): fully_preserved - his identity, face shape, deep brown skin tone, short dark twisted '
        'hair, goatee, thin mustache and silver nose stud are kept in every appearance.',
        f'<Subject 2> (appears in {", ".join(main)}): attribute_transfer - his {b.get("outfit", "grey baseball cap, black t-shirt, blue jeans")}, '
        'weapons, poses and motion are transferred to <Subject 1>; his own face, skin tone and identity do not appear anywhere.',
        vret,
        '<Audio 1>: fully_copy - <Audio 1> is reused 1:1 as the complete audio track, in sync with the cuts.', '',
        'detailed_description:', *[d for d in desc if d], '',
        'overall_soundscape:', b.get('soundscape', 'The audio of <Audio 1> is kept.'), '',
        'non_diegetic_music:', b.get('music', 'The music in <Audio 1> is reused as is.')])


def prompt_problems(p: str) -> list[str]:
    """Lint a full-reference prompt against the H3 guide (the failures that sank the first H3 rungs)."""
    out = []
    sec = p.split('retention_analysis:')[-1].split('detailed_description:')[0]
    for line in [x for x in sec.splitlines() if x.strip()]:
        marker = line.split(':', 1)[1].strip().split(' ')[0] if ':' in line else ''
        ok = AUDIO_MARKERS if line.startswith('<Audio') else VISIBLE_MARKERS
        if marker not in ok:
            out.append(f'bad marker {marker!r} in: {line[:60]}')
        if line.startswith('<Video') and marker == 'fully_preserved' and 'replace' in p:
            out.append('<Video N> fully_preserved in a replacement task keeps the source actor')
    import re
    if re.search(r'\[Shot \d+\] \d+(\.\d+)?-\d', p):
        out.append('shot times must be "[Shot N] At MM:SS.mmm,"')
    return out


# ---------------------------------------------------------------- picking the main man's faces
def shot_of(frame: int, cut_frames: list[int]) -> int:
    return sum(1 for c in cut_frames if c <= frame)


def select_main_faces(dets: dict, cut_frames: list[int], sim_thr: float = 0.3, prop_sim: float = 0.08, reach: float = 0.8) -> dict:
    """dets {frame: [[x, y, w, h, sim], ...]} (sim = cosine to the main man's anchor faces) -> {frame: [x, y, w, h]}.
    Seeds = sim >= sim_thr; then grow frame by frame inside the same shot to the nearest detection whose centre is within
    reach * box size (motion blur / small faces score low), never across a cut and never onto a sim < prop_sim face."""
    dets = {int(k): v for k, v in dets.items()}
    pick = {f: max(v, key=lambda r: r[4])[:4] for f, v in dets.items() if max(r[4] for r in v) >= sim_thr}
    for step in (1, -1):
        for f in sorted(pick, reverse=step < 0):
            g, cur = f + step, pick[f]
            while g in dets and g not in pick and shot_of(g, cut_frames) == shot_of(f, cut_frames):
                cx, cy, sz = cur[0] + cur[2] / 2, cur[1] + cur[3] / 2, max(cur[2], cur[3])
                near = [r for r in dets[g] if r[4] >= prop_sim and abs(r[0] + r[2] / 2 - cx) + abs(r[1] + r[3] / 2 - cy) < reach * sz]
                if not near:
                    break
                cur = min(near, key=lambda r: abs(r[0] + r[2] / 2 - cx) + abs(r[1] + r[3] / 2 - cy))[:4]
                pick[g] = cur; g += step
    return dict(sorted(pick.items()))


def face_region(box, W: int, H: int, sx: float = 1.3, sy: float = 1.35, up: float = 0.05):
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2 - up * h
    w2, h2 = w * sx, h * sy
    return (max(0, int(cx - w2 / 2)), max(0, int(cy - h2 / 2)), min(W, int(cx + w2 / 2)), min(H, int(cy + h2 / 2)))


def _ellipse(w: int, h: int, feather: float = 0.12) -> Image.Image:
    m = Image.new('L', (w, h), 0)
    ImageDraw.Draw(m).ellipse((w * feather / 2, h * feather / 2, w * (1 - feather / 2), h * (1 - feather / 2)), fill=255)
    return m.filter(ImageFilter.GaussianBlur(max(1.0, min(w, h) * feather / 2)))


def apply_face(frame: np.ndarray, box, mode: str, face: Image.Image | None = None) -> np.ndarray:
    """'mask': flat mid-grey feathered ellipse over the face (no identity left to copy); 'paste': ref face resized in."""
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = face_region(box, W, H)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return frame
    im = Image.fromarray(frame)
    alpha = _ellipse(x1 - x0, y1 - y0)
    if mode == 'mask':
        patch = Image.new('RGB', (x1 - x0, y1 - y0), (128, 128, 128))
    elif mode == 'paste':
        if face is None:
            raise ValueError('paste needs the ref face')
        patch = face.convert('RGB').resize((x1 - x0, y1 - y0), Image.LANCZOS)
    else:
        raise ValueError(mode)
    im.paste(patch, (x0, y0), alpha)
    return np.asarray(im)


def short_edge_size(w: int, h: int, short: int = 384, mult: int = 32) -> tuple[int, int]:
    """Target (w, h) with the short edge = short and the long edge rounded DOWN to a multiple of mult (centre crop)."""
    if w >= h:
        return (int(w * short / h) // mult * mult, short)
    return (short, int(h * short / w) // mult * mult)


# ---------------------------------------------------------------- box-side detection + render (OpenCV optional)
def _probe(path) -> tuple[int, int]:
    r = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height', '-of', 'csv=p=0',
                        str(path)], capture_output=True, text=True, check=True)
    w, h = r.stdout.strip().split(',')[:2]
    return int(w), int(h)


def read_frames(path):
    w, h = _probe(path)
    p = subprocess.Popen(['ffmpeg', '-v', 'error', '-i', str(path), '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], stdout=subprocess.PIPE)
    n = w * h * 3
    while True:
        buf = p.stdout.read(n)
        if len(buf) < n:
            break
        yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    p.wait()


def detect_faces(video, anchors: list[int], yunet: str, sface: str, thr: float = 0.5) -> dict:
    import cv2                                           # optional: pip install --target <tmp> opencv-python-headless
    w, h = _probe(video)
    det = cv2.FaceDetectorYN.create(yunet, '', (w, h), thr)
    rec = cv2.FaceRecognizerSF.create(sface, '')
    feats = {}
    for i, rgb in enumerate(read_frames(video)):
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        _, fs = det.detect(bgr)
        for r in (fs if fs is not None else []):
            feats.setdefault(i, []).append((rec.feature(rec.alignCrop(bgr, r)), [float(v) for v in r[:4]]))
    anc = [feats[a][0][0] for a in anchors]
    return {i: [b + [float(max(rec.match(f, a, cv2.FaceRecognizerSF_FR_COSINE) for a in anc))] for f, b in v]
            for i, v in feats.items()}


def ref_face_crop(ref, box, out, scale: float = 1.9, aspect: float = 1.25) -> pathlib.Path:
    """Head-and-shoulders crop around the detected ref face (aspect = h / w)."""
    im = Image.open(ref).convert('RGB')
    x, y, w, h = box
    cw = w * scale; ch = cw * aspect
    cx, cy = x + w / 2, y + h / 2 + 0.12 * ch
    x0, y0 = max(0, int(cx - cw / 2)), max(0, int(cy - ch / 2))
    im.crop((x0, y0, min(im.width, int(x0 + cw)), min(im.height, int(y0 + ch)))).save(out)
    return pathlib.Path(out)


def render_motion(src, out, faces: dict, mode: str | None, face_img: Image.Image | None = None, short: int = 384,
                  frame_fn=None) -> tuple[int, int]:
    """Motion ref at `short` short edge, 24 fps, source audio kept; faces greyed ('mask') / replaced ('paste') / untouched (None).
    frame_fn(i, rgb) -> rgb overrides the per-frame edit (depthhead / depthbody)."""
    w, h = _probe(src)
    tw, th = short_edge_size(w, h, short)
    sw = int(round(w * th / h)) if w >= h else tw
    sh = th if w >= h else int(round(h * tw / w))
    enc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}', '-r', str(FPS),
                            '-i', '-', '-i', str(src), '-map', '0:v', '-map', '1:a?', '-vf',
                            f'scale={sw}:{sh}:flags=area,crop={tw}:{th}', '-c:v', 'libx264', '-crf', '16', '-preset', 'fast',
                            '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '192k', '-shortest', str(out)], stdin=subprocess.PIPE)
    for i, fr in enumerate(read_frames(src)):
        if frame_fn is not None:
            fr = frame_fn(i, fr)
        elif mode and i in faces:
            fr = apply_face(fr, faces[i], mode, face_img)
        enc.stdin.write(np.ascontiguousarray(fr).tobytes())
    enc.stdin.close(); enc.wait()
    if enc.returncode:
        raise RuntimeError(f'ffmpeg encode failed for {out}')
    return tw, th


# ---------------------------------------------------------------- Oct 8 refine: depth head / depth body (sirioberati prep idea)
# The open Genjutsu workflow (github.com/sirioberati/Genjustsu-Open-Source-Workflow, MIT) hides WHO the actor is but keeps HOW he
# moves: Video-Depth-Anything inferno depth over the subject (SAM 3 mask) + a MediaPipe 478-point face mesh drawn from the
# original frames. Our port is box-side and free (CPU): Depth Anything V2 small (ONNX), MediaPipe FaceLandmarker (IMAGE mode
# per frame, cuts are hard), torchvision Mask R-CNN v2 person instances; the main man's instance = the one holding our tracked
# face box (select_main_faces), carried by mask IoU through face-less frames of the same shot, never across a cut.
INFERNO = np.array([(0, 0, 4), (31, 12, 72), (85, 15, 109), (136, 34, 106), (186, 54, 85), (227, 89, 51), (249, 140, 10),
                    (249, 201, 50), (252, 255, 164)], np.float32)
MESH_COLORS = {'tess': (50, 170, 205), 'lips': (255, 135, 100), 'eyes': (130, 255, 150)}   # repo mesh_worker.py colours


def inferno(d01: np.ndarray) -> np.ndarray:
    """HxW in [0, 1] (1 = near) -> HxWx3 uint8 inferno (matplotlib anchor colours, linear in between)."""
    x = np.clip(np.nan_to_num(np.asarray(d01, np.float32)), 0, 1) * (len(INFERNO) - 1)
    i = np.minimum(x.astype(np.int32), len(INFERNO) - 2)
    f = (x - i)[..., None]
    return np.clip(INFERNO[i] * (1 - f) + INFERNO[i + 1] * f + 0.5, 0, 255).astype(np.uint8)


def depth_range(values: list, lo: float = 3.0, hi: float = 99.5) -> tuple[float, float]:
    """Robust (lo, hi) over a shot's subject depth samples, so the colours don't flicker frame to frame."""
    v = np.concatenate([np.asarray(x, np.float32).ravel() for x in values if np.size(x)]) if values else np.zeros(1, np.float32)
    a, b = (float(np.percentile(v, lo)), float(np.percentile(v, hi))) if v.size else (0.0, 1.0)
    return (a, b) if b - a > 1e-6 else (a, a + 1.0)


def depth_rgb(depth: np.ndarray, rng: tuple[float, float]) -> np.ndarray:
    return inferno((np.asarray(depth, np.float32) - rng[0]) / (rng[1] - rng[0]))


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


def pick_instance(masks: list, face_box) -> int | None:
    """Index of the person instance holding the face box: most face-box pixels inside the mask, ties -> smaller mask
    (the nearer of two overlapping people). None if no instance covers >= 25 % of the face box."""
    x, y, w, h = [int(round(v)) for v in face_box]
    best, key = None, None
    for i, m in enumerate(masks):
        H, W = m.shape
        sub = m[max(0, y):min(H, y + h), max(0, x):min(W, x + w)]
        cov = float(sub.mean()) if sub.size else 0.0
        if cov < 0.25:
            continue
        k = (round(cov, 2), -int(m.sum()))
        if key is None or k > key:
            best, key = i, k
    return best


def track_main_masks(insts: dict, faces: dict, cut_frames: list[int], frames: list[int], min_iou: float = 0.3) -> dict:
    """insts {frame: [bool HxW, ...]} (person instances), faces {frame: [x, y, w, h]} (main man only) -> {frame: mask}.
    Seeds = frames whose face box sits in an instance; then grow to neighbours in the same shot by mask IoU >= min_iou."""
    out = {}
    for f in frames:
        if f in faces and insts.get(f):
            i = pick_instance(insts[f], faces[f])
            if i is not None:
                out[f] = insts[f][i]
    for step in (1, -1):
        for f in sorted(list(out), reverse=step < 0):
            g, cur = f + step, out[f]
            while g in insts and g not in out and shot_of(g, cut_frames) == shot_of(f, cut_frames):
                ious = [mask_iou(cur, m) for m in insts[g]]
                if not ious or max(ious) < min_iou:
                    break
                cur = insts[g][int(np.argmax(ious))]; out[g] = cur; g += step
    return dict(sorted(out.items()))


def main_shot_frames(faces: dict, cut_frames: list[int], n_frames: int) -> list[int]:
    """Every frame of every shot in which the main man's face was found (his face-less frames in those shots included)."""
    shots = {shot_of(int(f), cut_frames) for f in faces}
    return [i for i in range(n_frames) if shot_of(i, cut_frames) in shots]


def draw_mesh(im: Image.Image, pts: list, conns: dict, scale: float = 1.0) -> Image.Image:
    """pts = [(x, y), ...] (478 MediaPipe landmarks, image px); conns {'tess'|'lips'|'eyes': [(a, b), ...]}.
    Thin tessellation, lips and eyes thicker in their own colours (they carry the expression)."""
    d = ImageDraw.Draw(im)
    for name, width in (('tess', 1), ('lips', 2), ('eyes', 2)):
        for a, b in conns.get(name, []):
            if a < len(pts) and b < len(pts):
                d.line([tuple(pts[a]), tuple(pts[b])], fill=MESH_COLORS[name], width=max(1, int(round(width * scale))))
    return im


def depthhead_frame(frame: np.ndarray, box, drgb: np.ndarray, pts: list | None, conns: dict) -> np.ndarray:
    """Main man's face region (same feathered ellipse as 'mask') -> inferno depth, then the face mesh on top."""
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = face_region(box, W, H, sx=1.4, sy=1.5, up=0.08)
    im = Image.fromarray(frame)
    if x1 - x0 >= 4 and y1 - y0 >= 4:
        im.paste(Image.fromarray(drgb[y0:y1, x0:x1]), (x0, y0), _ellipse(x1 - x0, y1 - y0))
    if pts:
        draw_mesh(im, pts, conns)
    return np.asarray(im)


def depthbody_frame(frame: np.ndarray, mask: np.ndarray | None, drgb: np.ndarray, pts: list | None, conns: dict,
                    feather: float = 1.5) -> np.ndarray:
    """Main man's whole silhouette -> inferno depth over the untouched original (villains stay RGB), face mesh on top."""
    im = Image.fromarray(frame)
    if mask is not None and mask.any():
        a = Image.fromarray(mask.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.GaussianBlur(feather))
        im.paste(Image.fromarray(drgb), (0, 0), a)
    if pts:
        draw_mesh(im, pts, conns)
    return np.asarray(im)


def remux_source_audio(video, source, out, start: float = 0.0, seconds: float = 10.0, max_hold: float = 0.5) -> dict:
    """Final output = generated video + the ORIGINAL source audio packets (stream copy, no re-encode, never stretched).
    Port of the repo's finish_preview: a video up to max_hold s shorter than the audio window holds its last frame."""
    def dur(p, sel):
        r = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', sel, '-show_entries', 'stream=duration:format=duration',
                            '-of', 'json', str(p)], capture_output=True, text=True, check=True)
        j = json.loads(r.stdout or '{}')
        v = [float(s['duration']) for s in j.get('streams', []) if s.get('duration') not in (None, 'N/A')]
        return v[0] if v else float((j.get('format') or {}).get('duration') or 0)
    vd = dur(video, 'v:0'); sd = min(seconds, max(0.0, dur(source, 'a:0') - start))
    gap = round(sd - vd, 3); info = {'video_s': round(vd, 3), 'audio_s': round(sd, 3), 'gap_s': gap, 'held_s': 0.0}
    vf_args = ['-c:v', 'copy']
    if 0 < gap <= max_hold:
        vf_args = ['-vf', f'tpad=stop_mode=clone:stop_duration={gap:.3f}', '-c:v', 'libx264', '-crf', '16', '-preset', 'fast',
                   '-pix_fmt', 'yuv420p']
        info['held_s'] = gap
    elif gap > max_hold:
        info['warning'] = f'video {gap:.2f}s shorter than the audio window; audio ends with the video (not stretched)'
    cmd = ['ffmpeg', '-nostdin', '-v', 'error', '-y', '-i', str(video), '-ss', f'{start:.3f}', '-t', f'{sd:.3f}', '-i', str(source),
           '-map', '0:v:0', '-map', '1:a:0', *vf_args, '-c:a', 'copy', *(['-shortest'] if gap > max_hold else []),
           '-movflags', '+faststart', str(out)]
    subprocess.run(cmd, check=True, capture_output=True)
    return info


# ---- heavy, box-side prep (CPU; run with a python that has onnxruntime + mediapipe + torchvision, e.g. comfy-cpu/.venv)
def mesh_conns() -> dict:
    import mediapipe as mp
    c = mp.tasks.vision.FaceLandmarksConnections
    pairs = lambda xs: [(int(x.start), int(x.end)) for x in xs]
    return {'tess': pairs(c.FACE_LANDMARKS_TESSELATION), 'lips': pairs(c.FACE_LANDMARKS_LIPS),
            'eyes': pairs(c.FACE_LANDMARKS_LEFT_EYE) + pairs(c.FACE_LANDMARKS_RIGHT_EYE) + pairs(c.FACE_LANDMARKS_LEFT_IRIS)
            + pairs(c.FACE_LANDMARKS_RIGHT_IRIS)}


def face_meshes(src_hi, faces: dict, model: str, scale: float, pad: float = 1.1) -> dict:
    """MediaPipe 478-point mesh on the full-res (1080p) source, cropped around each tracked main-man face box (box coords are
    in the motion clip's px; scale = hi / lo). Returns {frame: [[x, y], ...]} in the motion clip's px."""
    import mediapipe as mp
    opts = mp.tasks.vision.FaceLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
                                                 running_mode=mp.tasks.vision.RunningMode.IMAGE, num_faces=3,
                                                 min_face_detection_confidence=0.2, min_face_presence_confidence=0.2)
    out = {}
    with mp.tasks.vision.FaceLandmarker.create_from_options(opts) as det:
        for i, rgb in enumerate(read_frames(src_hi)):
            if i not in faces:
                continue
            H, W = rgb.shape[:2]
            x, y, w, h = [v * scale for v in faces[i]]
            cx, cy, r = x + w / 2, y + h / 2, max(w, h) * (0.5 + pad)
            x0, y0, x1, y1 = int(max(0, cx - r)), int(max(0, cy - r)), int(min(W, cx + r)), int(min(H, cy + r))
            crop = np.ascontiguousarray(rgb[y0:y1, x0:x1])
            up = max(1.0, 256 / max(1, min(crop.shape[:2])))
            if up > 1:
                crop = np.asarray(Image.fromarray(crop).resize((int(crop.shape[1] * up), int(crop.shape[0] * up)), Image.LANCZOS))
            res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=crop))
            best = None
            for lm in res.face_landmarks or []:
                pts = [((x0 + l.x * crop.shape[1] / up) / scale, (y0 + l.y * crop.shape[0] / up) / scale) for l in lm]
                mx, my = np.mean([p[0] for p in pts]), np.mean([p[1] for p in pts])
                bx, by = faces[i][0] + faces[i][2] / 2, faces[i][1] + faces[i][3] / 2
                dist = abs(mx - bx) + abs(my - by)
                if dist < max(faces[i][2], faces[i][3]) and (best is None or dist < best[0]):
                    best = (dist, [[round(p[0], 1), round(p[1], 1)] for p in pts])
            if best:
                out[i] = best[1]
    return out


def depth_frames(src, frames: set, onnx: str, size: tuple[int, int] = (924, 518)) -> dict:
    """Depth Anything V2 small (ONNX, CPU) on the chosen frames -> {frame: float32 HxW relative inverse depth (near = high)}."""
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx), providers=['CPUExecutionProvider'])
    name = sess.get_inputs()[0].name
    mean, std = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
    out = {}
    for i, rgb in enumerate(read_frames(src)):
        if i not in frames:
            continue
        H, W = rgb.shape[:2]
        x = (np.asarray(Image.fromarray(rgb).resize(size, Image.BICUBIC), np.float32) / 255 - mean) / std
        d = sess.run(None, {name: x.transpose(2, 0, 1)[None]})[0][0]
        out[i] = np.asarray(Image.fromarray(d.astype(np.float32)).resize((W, H), Image.BILINEAR), np.float32)
    return out


def person_instances(src, frames: set, weights: str, thr: float = 0.5) -> dict:
    """torchvision Mask R-CNN v2 (COCO) person instances on the chosen frames -> {frame: [bool HxW, ...]}."""
    import torch, torchvision
    torch.set_num_threads(max(1, (__import__('os').cpu_count() or 4)))
    m = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None)
    m.load_state_dict(torch.load(weights, map_location='cpu')); m.eval()
    out = {}
    with torch.no_grad():
        for i, rgb in enumerate(read_frames(src)):
            if i not in frames:
                continue
            r = m([torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255])[0]
            keep = (r['labels'] == 1) & (r['scores'] >= thr)
            out[i] = [mk[0].numpy() > 0.5 for mk in r['masks'][keep]]
    return out


# ---------------------------------------------------------------- graphs + live queue
def fix_graph(base: dict, prompt: str, motion: str, refs: list[str], size: tuple[int, int], seed: int, prefix: str,
              ref_image_size: str = 'match') -> dict:
    """h2h H3 graph (LoadImage 137 -> ref_image_0, LoadVideo 9000 -> ImageScale 9002 -> ref_video_0, prompt 138) with the fix applied."""
    g = copy.deepcopy(base)
    r2v = next(k for k, n in g.items() if n['class_type'] == 'MiniMaxH3ReferenceToVideo')
    pk = g[r2v]['inputs']['prompt'][0]
    g[pk]['inputs']['value'] = prompt
    vid = g[r2v]['inputs']['ref_videos.ref_video_0'][0]          # ImageScale
    g[vid]['inputs'].update({'width': size[0], 'height': size[1]})
    src = g[vid]['inputs']['image'][0]                            # GetVideoComponents
    load = g[src]['inputs']['video'][0]
    g[load]['inputs']['file'] = motion
    for k in [k for k in g[r2v]['inputs'] if k.startswith('ref_images.')]:
        del g[r2v]['inputs'][k]
    for i, r in enumerate(refs):
        nid = '137' if i == 0 else str(9100 + i)
        g[nid] = {'class_type': 'LoadImage', 'inputs': {'image': r}}
        g[r2v]['inputs'][f'ref_images.ref_image_{i}'] = [nid, 0]
    g[r2v]['inputs']['ref_image_size'] = ref_image_size
    for n in g.values():
        if n['class_type'] == 'RandomNoise':
            n['inputs']['noise_seed'] = seed
        if n['class_type'] == 'SaveVideo':
            n['inputs']['filename_prefix'] = prefix
    return g


def base_steps(g: dict, steps: int = 20, sampler: str = 'res_multistep') -> dict:
    """PDD-4 graph -> base schedule (Comfy H3 docs: at 4 steps a reference can end up barely applied; identity-critical shots
    want the base 20-step schedule, 25 if the face still drifts). Drops MiniMaxH3PDDAccApply, adds BasicScheduler(simple)."""
    g = copy.deepcopy(g)
    pdd = next(k for k, n in g.items() if n['class_type'] == 'MiniMaxH3PDDAccApply')
    up = g[pdd]['inputs']['model']
    for n in g.values():
        for k, v in n['inputs'].items():
            if isinstance(v, list) and len(v) == 2 and v[0] == pdd and v[1] == 0:
                n['inputs'][k] = up
    del g[pdd]
    g['9200'] = {'class_type': 'BasicScheduler', 'inputs': {'model': up, 'scheduler': 'simple', 'steps': int(steps), 'denoise': 1.0},
                 '_meta': {'title': f'base {steps}-step schedule'}}
    for n in g.values():
        if n['class_type'] == 'SamplerCustomAdvanced':
            n['inputs']['sigmas'] = ['9200', 0]
        if n['class_type'] == 'KSamplerSelect':
            n['inputs']['sampler_name'] = sampler
    return g


def extra_line(rid: str, graph_path, uploads: list[str], est_s: int, rung: str, seed: int, steps: int) -> dict:
    return {'id': rid, 'graph_path': str(graph_path), 'uploads': uploads, 'engine': 'genjutsu_h3_ref2va', 'est_s': int(est_s),
            'rung': rung, 'seed': seed, 'params': {'steps': steps}, 'protect': False, 'needs_stage2': False}


VARIANTS = {   # id: (prompt mode, motion file key, label)
    'fixA_prompt':    ('edit',   'raw544', 'guide-correct Context-IR + face crop as Picture 1 (motion ref unchanged, 544)'),
    'fixB_mask384':   ('edit',   'mask',   'A + motion ref 384 short edge with the actor\'s face greyed out'),
    'fixC_paste384':  ('edit',   'paste',  'A + motion ref 384 with the owner\'s face roughly pasted over the actor (pre-swap)'),
    'fixD_motion384': ('motion', 'mask',   'reference-generation (motion-transfer) framing + masked 384 ref'),
    'fixE_base20':    ('edit',   'mask',   'fixB at the base 20-step schedule (res_multistep), no distill: full reference adherence'),
}
STEPS = {'fixE_base20': 20}
FACE_NOTE = {'mask': "(his face is blanked out with a grey oval in <Video 1>; fill it with <Subject 1>'s face)",
             'depthhead': ("(his head in <Video 1> is shown as a colored depth silhouette in yellow, orange and purple heat colors with "
                           "thin face-mesh lines; the silhouette and lines only guide his head pose, gaze and expression and must not "
                           "appear in the target video; his face is <Subject 1>'s real face with natural skin)"),
             'depthbody': ("(his whole body in <Video 1> is shown as a colored depth silhouette in yellow, orange and purple heat colors "
                           "with thin face-mesh lines on his face; the silhouette and lines only guide his pose, body motion, gaze and "
                           "expression and must not appear in the target video. In the target video he is a real person wearing a grey "
                           "baseball cap, a plain black t-shirt and blue jeans, with <Subject 1>'s real face and deep brown skin)"),
             'paste': "(his face in <Video 1> is already roughly replaced by <Subject 1>'s face; refine it into <Subject 1>)",
             'raw544': ''}


def fix_shots(shots: list) -> list:
    """Brief correction found by the face pass: the main man is back on screen at 9.83-10.04 s (cosine 0.5-0.67 to the anchors)."""
    out = [list(s) for s in shots]
    if out and abs(out[-1][0] - 9.57) < 0.05 and not out[-1][2]:
        out[-1] = [out[-1][0], out[-1][1], True, 'close-up of the ghost-mask figure spreading its gloved hands, then a close-up of '
                   '<Subject 1> recoiling as a machete swings past']
    return out


# ---------------------------------------------------------------- Oct 8 refine round (the owner approved, $3 cap)
REFINE_DIR = ROOT / 'data' / 'bakeoff' / 'refine'
REFINE_REVIEW = ROOT / 'data' / 'bakeoff' / 'out' / 'refine'
REFINE_MOTION = {'mask': 'fix_motion_mask.mp4', 'depthhead': 'refine_motion_depthhead.mp4', 'depthbody': 'refine_motion_depthbody.mp4'}
# (run id, motion key, steps, seed, base graph): base = PDD-4 graph (-> base_steps for 20) or the PDD-8 graph. Priority order:
# a quote over the cap drops runs from the END of this list. est_s = warm wall at the 384 motion ref (Oct 7 measured: base20
# 708 s, PDD-4 176 s, PDD-8 ~8 x 33.4 s + overhead), the first run carries the cold model load.
REFINE = [
    ('fixF_depthhead_base20', 'depthhead', 20, 42, 'P4'),
    ('fixF_depthhead_pdd4', 'depthhead', 4, 42, 'P4'),
    ('fixF_depthhead_pdd8', 'depthhead', 8, 42, 'P8'),
    ('fixG_depthbody_base20', 'depthbody', 20, 42, 'P4'),
    ('fixG_depthbody_pdd4', 'depthbody', 4, 42, 'P4'),
    ('fixE_base20_s7', 'mask', 20, 7, 'P4'),
    ('fixE_base20_s2026', 'mask', 20, 2026, 'P4'),
]
REFINE_EST = {20: 720, 8: 330, 4: 190}
REFINE_COLD_S = 120
REFINE_LABEL = {'mask': 'fixE exactly (grey-oval face mask 384), new seed',
                'depthhead': "main man's face -> inferno depth + MediaPipe face mesh (pose/expression kept, identity removed), 384",
                'depthbody': "main man's whole body -> inferno depth + face mesh over the untouched frame (villains RGB), 384"}


def refine_runs(cap: float | None = None, rate: float | None = None, setup_s: int = 0) -> tuple[list[dict], list[str]]:
    """The refine run list (graphs must already be built); with cap/rate, drops from the end until the quote fits."""
    runs = []
    for i, (rid, key, steps, seed, base) in enumerate(REFINE):
        runs.append({'id': rid, 'rung': rid.rsplit('_s', 1)[0] if rid.startswith('fixE_base20_s') else rid, 'seed': seed,
                     'engine': 'genjutsu_h3_ref2va', 'tier': 'final', 'protect': False, 'needs_stage2': False,
                     'question': f"{rid}: {REFINE_LABEL[key]}, {steps} steps{'' if steps != 20 else ' (base, res_multistep)'}",
                     'uploads': [REFINE_MOTION[key], 'fix_ref_face.png', 'h2h_ref.png'], 'params': {'steps': steps, 'motion': key},
                     'est_s': REFINE_EST[steps] + (REFINE_COLD_S if i == 0 else 0),
                     'graph_path': str(REFINE_DIR / rid / 'prompt_api.json')})
    dropped = []
    if cap is not None and rate:
        while runs and (setup_s + sum(r['est_s'] * 1.3 for r in runs)) / 3600 * rate > cap:
            dropped.append(runs.pop()['id'])
    return runs, dropped


def build_refine_graphs(brief: dict, meta: dict) -> list[str]:
    bases = {'P4': json.loads((H2H / 'h2h_H3-P4_s42' / 'prompt_api.json').read_text()),
             'P8': json.loads((H2H / 'h2h_H3-P8_s42' / 'prompt_api.json').read_text())}
    size = tuple(meta['files']['mask']['size'])
    built = []
    for rid, key, steps, seed, base in REFINE:
        prompt = h3_prompt_v2(brief, float(brief.get('seconds', 10)), 'edit', 2, FACE_NOTE[key])
        probs = prompt_problems(prompt)
        if probs:
            raise SystemExit(f'{rid}: prompt problems {probs}')
        if not (CLIPS / REFINE_MOTION[key]).exists():
            raise SystemExit(f'{rid}: motion ref {REFINE_MOTION[key]} missing (run refine-prep first)')
        g = fix_graph(bases[base], prompt, REFINE_MOTION[key], ['fix_ref_face.png', 'h2h_ref.png'], size, seed, f'genjutsu/{rid}')
        if steps == 20:
            g = base_steps(g, 20)
        d = REFINE_DIR / rid; d.mkdir(parents=True, exist_ok=True)
        (d / 'prompt_api.json').write_text(json.dumps(g, indent=1))
        (d / 'label.txt').write_text(REFINE_LABEL[key] + f', {steps} steps, seed {seed}\n')
        built.append(rid)
    return built


def refine_sheet(out, frames: list[int], files: list[tuple[str, str]], cols_w: int = 336) -> pathlib.Path:
    """Contact sheet: one row per frame, one column per motion clip (label, CLIPS file name)."""
    from PIL import ImageFont
    rows = []
    for f in frames:
        row = []
        for _, name in files:
            r = subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(CLIPS / name), '-vf', f'select=eq(n\\,{f}),scale={cols_w}:-2',
                                '-vframes', '1', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], capture_output=True, check=True)
            h = len(r.stdout) // (cols_w * 3)
            row.append(Image.frombytes('RGB', (cols_w, h), r.stdout[:cols_w * h * 3]))
        rows.append((f, row))
    h = rows[0][1][0].height
    sheet = Image.new('RGB', (60 + cols_w * len(files), 24 + h * len(rows)), (20, 20, 20))
    d = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 13)
    except Exception:
        font = ImageFont.load_default()
    for j, (lab, _) in enumerate(files):
        d.text((60 + j * cols_w + 4, 4), lab, fill=(255, 255, 0), font=font)
    for i, (f, row) in enumerate(rows):
        d.text((4, 24 + i * h + 4), f'f{f}\n{f / FPS:.2f}s', fill=(255, 255, 255), font=font)
        for j, im in enumerate(row):
            sheet.paste(im, (60 + j * cols_w, 24 + i * h))
    out = pathlib.Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=88)
    return out


def parse_frames(spec: str) -> set[int]:
    """'101-113,202,204-205' -> {101..113, 202, 204, 205}."""
    out = set()
    for part in [x.strip() for x in spec.split(',') if x.strip()]:
        a, _, b = part.partition('-')
        out.update(range(int(a), int(b or a) + 1))
    return out


# Oct 8 review of the first depthbody pass (contact sheet, data/bakeoff/out/refine/): frames 101-113 are the "dark figure tumbles
# through a doorway" villain shot behind a cut the scene detector missed (the brief has it at 4.2 s; the detector cut on the
# muzzle flash at 98), and in 202-205 Mask R-CNN merged the ghost-robe villain climbing on him into his instance.
REFINE_DROP = '101-113,202-205'


def refine_prep(a, meta: dict, cut_frames: list[int]) -> dict:
    """Box-side, free: depth + mesh + main-man masks -> refine_motion_depthhead.mp4 / refine_motion_depthbody.mp4 (384, 24 fps)."""
    src = CLIPS / 'h2h_motion.mp4'
    faces = {int(k): v for k, v in meta['faces'].items()}
    n = sum(1 for _ in read_frames(src)) if not a.frames else a.frames
    cache = pathlib.Path(a.cache); cache.mkdir(parents=True, exist_ok=True)
    frames = main_shot_frames(faces, cut_frames, n)
    hi = cache / 'motion_hi.mp4'                 # same window/fps as h2h_motion.mp4, native 1080p (mesh is found here)
    if not hi.exists():
        w = json.loads((H2H / 'window.json').read_text())
        subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-ss', str(w['start']), '-i', str(ROOT / 'data' / 'bakeoff' / 'inputs-oct7b' / 'motion_src.mp4'),
                        '-t', str(float(w['seconds']) + 0.08), '-vf', 'fps=24', '-an', '-c:v', 'libx264', '-crf', '12', '-preset', 'fast',
                        str(hi)], check=True)
    lw, _ = _probe(src); hw, _ = _probe(hi)
    mf = cache / 'meshes.json'
    if mf.exists():
        meshes = {int(k): v for k, v in json.loads(mf.read_text()).items()}
    else:
        meshes = face_meshes(hi, faces, a.landmarker, hw / lw); mf.write_text(json.dumps(meshes))
    df = cache / 'depth.npz'
    if df.exists():
        z = np.load(df); depth = {int(k[1:]): z[k].astype(np.float32) for k in z.files}
    else:
        depth = depth_frames(src, set(frames), a.onnx)
        np.savez_compressed(df, **{f'f{k}': v.astype(np.float16) for k, v in depth.items()})
    kf = cache / 'insts.npz'
    if kf.exists():
        z = np.load(kf); shape = tuple(z['shape'])
        insts = {}
        for k in z.files:
            if k.startswith('f'):
                f, j = (int(x) for x in k[1:].split('_'))
                insts.setdefault(f, []).append(np.unpackbits(z[k])[:shape[0] * shape[1]].reshape(shape).astype(bool))
        for f in frames:
            insts.setdefault(f, [])
    else:
        insts = person_instances(src, set(frames), a.maskrcnn)
        any_m = next((m for v in insts.values() for m in v), None)
        shape = any_m.shape if any_m is not None else (1, 1)
        np.savez_compressed(kf, shape=np.array(shape), **{f'f{f}_{j}': np.packbits(m) for f, v in insts.items() for j, m in enumerate(v)})
    drop = parse_frames(a.drop_frames or '')
    masks = {f: m for f, m in track_main_masks(insts, faces, cut_frames, frames).items() if f not in drop}
    conns = mesh_conns()
    # per-shot depth ranges: head = values inside the face boxes, body = values inside his mask (colours stable within a shot)
    head_rng, body_rng = {}, {}
    for sh in sorted({shot_of(f, cut_frames) for f in frames}):
        fs = [f for f in frames if shot_of(f, cut_frames) == sh and f in depth]
        hv = []
        for f in fs:
            if f in faces:
                x0, y0, x1, y1 = face_region(faces[f], depth[f].shape[1], depth[f].shape[0])
                hv.append(depth[f][y0:y1, x0:x1])
        head_rng[sh] = depth_range(hv, 5.0, 99.5)
        body_rng[sh] = depth_range([depth[f][masks[f]] for f in fs if f in masks], 2.0, 99.5)

    def head_fn(i, fr):
        if i not in faces or i not in depth:
            return fr
        return depthhead_frame(fr, faces[i], depth_rgb(depth[i], head_rng[shot_of(i, cut_frames)]), meshes.get(i), conns)

    def body_fn(i, fr):
        if i not in depth or (i not in masks and i not in faces):
            return fr
        drgb = depth_rgb(depth[i], body_rng[shot_of(i, cut_frames)])
        if i not in masks:                    # face found but no body instance: fall back to the depth head for that frame
            return depthhead_frame(fr, faces[i], depth_rgb(depth[i], head_rng[shot_of(i, cut_frames)]), meshes.get(i), conns)
        return depthbody_frame(fr, masks[i], drgb, meshes.get(i), conns)

    files = {}
    for key, fn in (('depthhead', head_fn), ('depthbody', body_fn)):
        out = CLIPS / REFINE_MOTION[key]
        files[key] = {'file': out.name, 'size': render_motion(src, out, faces, None, short=384, frame_fn=fn)}
    stats = {'frames_total': n, 'main_shot_frames': len(frames), 'face_frames': len(faces), 'mesh_frames': len(meshes),
             'mesh_missing': sorted(set(faces) - set(meshes)), 'body_mask_frames': len(masks),
             'body_missing_in_main_shots': [f for f in frames if f not in masks], 'dropped_frames': sorted(drop),
             'mask_area_frac': {f: round(float(m.mean()), 3) for f, m in masks.items()}, 'files': files}
    REFINE_DIR.mkdir(parents=True, exist_ok=True)
    (REFINE_DIR / 'refine_meta.json').write_text(json.dumps(stats, indent=1))
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('prep'); p.add_argument('--yunet', required=True); p.add_argument('--sface', required=True)
    p.add_argument('--anchors', default='22,92')
    q = sub.add_parser('build'); q.add_argument('--base', default=str(H2H / 'h2h_H3-P4_s42' / 'prompt_api.json'))
    q.add_argument('--seed', type=int, default=42); q.add_argument('--est', type=int, default=330)
    q.add_argument('--only', default=''); q.add_argument('--queue', action='store_true', help='append to h2h/extra_runs.jsonl')
    r = sub.add_parser('refine-prep', help='box-side depth head / depth body motion refs (free, CPU)')
    r.add_argument('--onnx', required=True, help='Depth Anything V2 small ONNX'); r.add_argument('--landmarker', required=True)
    r.add_argument('--maskrcnn', required=True, help='torchvision maskrcnn_resnet50_fpn_v2 COCO weights (.pth)')
    r.add_argument('--cache', default='/tmp/refine-tmp/cache'); r.add_argument('--frames', type=int, default=0)
    r.add_argument('--drop-frames', default=REFINE_DROP,
                   help='frames/ranges whose body mask is wrong (depthbody falls back to the depth head if a face is there)')
    sub.add_parser('refine-sheet', help='contact sheet of the 384 motion refs -> data/bakeoff/out/refine/')
    sub.add_parser('refine-build', help='refine graphs + refine_runs.json (data/bakeoff/refine/)')
    a = ap.parse_args(argv)
    brief = json.loads((ROOT / 'config' / 'h2h' / 'oct7b.json').read_text())
    brief['shots'] = fix_shots(brief.get('shots') or [])
    win = json.loads((H2H / 'window.json').read_text())
    cut_frames = [int(round(c['t'] * FPS)) for c in win['cuts']]
    src = CLIPS / 'h2h_motion.mp4'
    if a.cmd == 'refine-prep':
        st = refine_prep(a, json.loads((H2H / 'fix_meta.json').read_text()), cut_frames)
        print(f"[refine-prep] {json.dumps({k: v for k, v in st.items() if k != 'mask_area_frac'})}")
        return 0
    if a.cmd == 'refine-sheet':
        meta = json.loads((H2H / 'fix_meta.json').read_text())
        faces = sorted(int(k) for k in meta['faces'])
        frames = main_shot_frames({f: 1 for f in faces}, cut_frames, 242)
        by_shot = {}
        for f in frames:
            by_shot.setdefault(shot_of(f, cut_frames), []).append(f)
        pick = sorted({fs[k] for fs in by_shot.values() for k in (0, len(fs) // 2, len(fs) - 1)} | {5, 60, 130, 185})
        out = refine_sheet(REFINE_REVIEW / 'review_depth_sheet.jpg', pick,
                           [('original 384', 'fix_motion_raw384.mp4'), ('fixE mask (baseline)', 'fix_motion_mask.mp4'),
                            ('depthhead', REFINE_MOTION['depthhead']), ('depthbody', REFINE_MOTION['depthbody'])])
        print(f'[refine-sheet] {out} ({len(pick)} frames: {pick})')
        return 0
    if a.cmd == 'refine-build':
        built = build_refine_graphs(brief, json.loads((H2H / 'fix_meta.json').read_text()))
        runs, _ = refine_runs()
        (REFINE_DIR / 'refine_runs.json').write_text(json.dumps(runs, indent=1))
        print(f'[refine-build] {built}')
        return 0
    if a.cmd == 'prep':
        import cv2
        anchors = [int(x) for x in a.anchors.split(',')]
        dets = detect_faces(src, anchors, a.yunet, a.sface)
        faces = select_main_faces(dets, cut_frames)
        ref = CLIPS / 'h2h_ref.png'
        rw, rh = _probe(ref)
        det = cv2.FaceDetectorYN.create(a.yunet, '', (rw, rh), 0.6)
        _, rf = det.detect(cv2.imread(str(ref)))
        rbox = [float(v) for v in rf[0][:4]]
        ref_face_crop(ref, rbox, CLIPS / 'fix_ref_face.png')
        x, y, w, h = rbox
        face_img = Image.open(ref).convert('RGB').crop((int(x - .08 * w), int(y - .12 * h), int(x + 1.08 * w), int(y + 1.05 * h)))
        meta = {'faces': faces, 'ref_box': rbox, 'cut_frames': cut_frames, 'files': {}}
        for key, mode in (('raw384', None), ('mask', 'mask'), ('paste', 'paste')):
            out = CLIPS / f'fix_motion_{key}.mp4'
            meta['files'][key] = {'file': out.name, 'size': render_motion(src, out, faces, mode, face_img)}
        meta['files']['raw544'] = {'file': 'h2h_motion.mp4', 'size': [960, 544]}
        (H2H / 'fix_meta.json').write_text(json.dumps(meta, indent=1))
        print(f'[prep] main-man faces in {len(faces)} frames; files {[v["file"] for v in meta["files"].values()]}')
        return 0
    meta = json.loads((H2H / 'fix_meta.json').read_text())
    base = json.loads(pathlib.Path(a.base).read_text())
    lines = []
    for vid, (mode, key, label) in VARIANTS.items():
        if a.only and vid not in a.only.split(','):
            continue
        prompt = h3_prompt_v2(brief, float(brief.get('seconds', 10)), mode, 2, FACE_NOTE[key])
        probs = prompt_problems(prompt)
        if probs:
            raise SystemExit(f'{vid}: prompt problems {probs}')
        f = meta['files'][key]
        g = fix_graph(base, prompt, f['file'], ['fix_ref_face.png', 'h2h_ref.png'], tuple(f['size']), a.seed, f'genjutsu/{vid}')
        steps = STEPS.get(vid, 4)
        if steps != 4:
            g = base_steps(g, steps)
        d = H2H / vid; d.mkdir(exist_ok=True)
        (d / 'prompt_api.json').write_text(json.dumps(g, indent=1)); (d / 'label.txt').write_text(label + '\n')
        lines.append(extra_line(vid, d / 'prompt_api.json', sorted({f['file'], 'fix_ref_face.png', 'h2h_ref.png'}),
                                a.est if steps == 4 else int(a.est * steps / 4 * 0.75), 'H3-' + vid, a.seed, steps))
        print(f'[build] {vid}: {label}')
    if a.queue:
        with open(H2H / 'extra_runs.jsonl', 'a') as fh:
            for ln in lines:
                fh.write(json.dumps(ln) + '\n')
        print(f'[queue] appended {len(lines)} runs to {H2H / "extra_runs.jsonl"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
