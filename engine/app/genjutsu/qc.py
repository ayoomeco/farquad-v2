"""Genjutsu QC: cheap automatic checks on a finished shot vs its source + reference.

Free checks (ffmpeg/ffprobe only, always run):
  duration match (after trim_to_source), fps, resolution, audio kept, black frames, frozen stretches.
Learned metrics (optional deps; reported as {'status': 'unavailable', 'why': ...} when missing, never faked):
  identity  ArcFace cosine (insightface buffalo_l) between the subject ref and sampled output frames.
  pose      2D keypoint agreement (rtmlib RTMPose / mediapipe) between source and output, normalised by bbox.
Thresholds are first guesses to be calibrated on the Phase 1 bake-off; `verdict` is advisory, not a gate.
An unavailable learned check (missing dependency, no person detected) never passes: verdict 'review' + the reason.
checks['scope'] says whether only the technical checks ran (a technical pass is not a likeness/motion pass)."""
from __future__ import annotations
import json, re, subprocess
from dataclasses import dataclass, field, asdict

THRESHOLDS = {'duration_tol_s': 0.15, 'black_max_s': 0.5, 'freeze_max_s': 1.5, 'identity_min': 0.35, 'pose_min': 0.6}


@dataclass
class QCReport:
    output: str
    checks: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    verdict: str = 'unknown'
    reasons: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def probe(path: str) -> dict:
    r = subprocess.run(['ffprobe', '-v', 'error', '-print_format', 'json', '-show_format', '-show_streams', path],
                       capture_output=True, text=True, timeout=30)
    d = json.loads(r.stdout or '{}')
    v = next((s for s in d.get('streams', []) if s.get('codec_type') == 'video'), {})
    num, _, den = (v.get('r_frame_rate') or '0/1').partition('/')
    return {'seconds': float(d.get('format', {}).get('duration') or 0), 'w': v.get('width'), 'h': v.get('height'),
            'fps': round(float(num) / float(den or 1), 3) if float(den or 1) else 0.0,
            'audio': any(s.get('codec_type') == 'audio' for s in d.get('streams', []))}


def _filter_log(path: str, vf: str) -> str:
    r = subprocess.run(['ffmpeg', '-nostdin', '-hide_banner', '-i', path, '-vf', vf, '-an', '-f', 'null', '-'],
                       capture_output=True, text=True, timeout=600)
    return r.stderr


def black_segments(path: str) -> list[tuple[float, float]]:
    log = _filter_log(path, 'blackdetect=d=0.1:pix_th=0.10')
    return [(float(a), float(b)) for a, b in re.findall(r'black_start:([\d.]+) black_end:([\d.]+)', log)]


def frozen_segments(path: str) -> list[tuple[float, float]]:
    log = _filter_log(path, 'freezedetect=n=0.001:d=0.5')
    starts = [float(x) for x in re.findall(r'freeze_start: ([\d.]+)', log)]
    ends = [float(x) for x in re.findall(r'freeze_end: ([\d.]+)', log)]
    total = probe(path)['seconds']
    return [(s, ends[i] if i < len(ends) else total) for i, s in enumerate(starts)]


def identity_score(ref_image: str, video: str, samples: int = 8) -> dict:
    try:
        import numpy as np, cv2                                   # noqa: F401
        from insightface.app import FaceAnalysis
    except Exception as e:
        return {'status': 'unavailable', 'why': f'insightface/opencv not installed ({type(e).__name__}); '
                'pip install insightface onnxruntime opencv-python-headless'}
    fa = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider']); fa.prepare(ctx_id=-1, det_size=(640, 640))
    ref = fa.get(cv2.imread(ref_image))
    if not ref:
        return {'status': 'no_face_in_ref'}
    r = max(ref, key=lambda f: f.bbox[2] - f.bbox[0]).normed_embedding
    cap = cv2.VideoCapture(video); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    sims, missing = [], 0
    for i in range(samples):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i * (n - 1) / max(1, samples - 1)))
        ok, fr = cap.read()
        faces = fa.get(fr) if ok else []
        if not faces:
            missing += 1; continue
        sims.append(float(np.dot(r, max(faces, key=lambda f: f.bbox[2] - f.bbox[0]).normed_embedding)))
    if not sims:
        return {'status': 'no_face_in_output', 'frames_without_face': missing}
    return {'status': 'ok', 'mean': round(sum(sims) / len(sims), 4), 'min': round(min(sims), 4), 'frames': len(sims),
            'frames_without_face': missing}


def pose_score(source: str, video: str, samples: int = 8) -> dict:
    try:
        import numpy as np, cv2                                   # noqa: F401
        from rtmlib import Body
    except Exception as e:
        return {'status': 'unavailable', 'why': f'rtmlib/opencv not installed ({type(e).__name__}); '
                'pip install rtmlib onnxruntime opencv-python-headless'}
    body = Body(mode='lightweight', backend='onnxruntime', device='cpu')

    def kps(path):
        cap = cv2.VideoCapture(path); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1; out = []
        for i in range(samples):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i * (n - 1) / max(1, samples - 1)))
            ok, fr = cap.read()
            k, s = body(fr) if ok else ([], [])
            out.append(k[0] if len(k) else None)
        return out
    a, b = kps(source), kps(video)
    scores = []
    for x, y in zip(a, b):
        if x is None or y is None:
            continue
        nx = (x - x.min(0)) / np.maximum(np.ptp(x, 0), 1e-6); ny = (y - y.min(0)) / np.maximum(np.ptp(y, 0), 1e-6)
        scores.append(float(1 - np.clip(np.linalg.norm(nx - ny, axis=1).mean() / 0.5, 0, 1)))   # 1 = same pose
    if not scores:
        return {'status': 'no_person_detected'}
    return {'status': 'ok', 'mean': round(sum(scores) / len(scores), 4), 'frames': len(scores)}


def run(output: str, source: str | None = None, ref_image: str | None = None, mode: str = 'motion_transfer',
        keep_audio: bool = True, learned: bool = True) -> QCReport:
    rep = QCReport(output=output)
    o = probe(output); rep.checks['output'] = o
    if not o['seconds']:
        rep.verdict = 'fail'; rep.reasons.append('output unreadable or empty'); return rep
    if source:
        s = probe(source); rep.checks['source'] = s
        rep.checks['duration_ok'] = abs(o['seconds'] - s['seconds']) <= THRESHOLDS['duration_tol_s']
        if not rep.checks['duration_ok']:
            rep.reasons.append(f"duration {o['seconds']:.2f}s vs source {s['seconds']:.2f}s")
        if keep_audio and s['audio']:
            rep.checks['audio_ok'] = o['audio']
            if not o['audio']:
                rep.reasons.append('source audio was dropped')
    blk = black_segments(output); rep.checks['black'] = blk
    if sum(b - a for a, b in blk) > THRESHOLDS['black_max_s']:
        rep.reasons.append(f'{sum(b - a for a, b in blk):.1f}s of black frames')
    frz = frozen_segments(output); rep.checks['frozen'] = frz
    if any(b - a > THRESHOLDS['freeze_max_s'] for a, b in frz):
        rep.reasons.append('frozen stretch > %.1fs' % THRESHOLDS['freeze_max_s'])
    unavailable = []
    if learned:
        if ref_image and mode in ('motion_transfer', 'character_swap'):
            rep.metrics['identity'] = identity_score(ref_image, output)
            m = rep.metrics['identity']
            if m.get('status') == 'ok' and m['mean'] < THRESHOLDS['identity_min']:
                rep.reasons.append(f"identity {m['mean']:.2f} < {THRESHOLDS['identity_min']}")
            elif m.get('status') != 'ok':
                unavailable.append(f"identity check {m.get('status')}: {m.get('why', 'no score')}"[:200])
        if source and mode in ('motion_transfer', 'character_swap', 'restyle'):
            rep.metrics['pose'] = pose_score(source, output)
            m = rep.metrics['pose']
            if m.get('status') == 'ok' and m['mean'] < THRESHOLDS['pose_min']:
                rep.reasons.append(f"pose agreement {m['mean']:.2f} < {THRESHOLDS['pose_min']}")
            elif m.get('status') != 'ok':
                unavailable.append(f"pose check {m.get('status')}: {m.get('why', 'no score')}"[:200])
    # A missing detector / undetected subject must never read as a pass (Oct 7 review): it goes to review with the reason.
    rep.reasons += unavailable
    rep.checks['scope'] = 'technical+learned' if learned else 'technical_only'
    rep.checks['unavailable'] = unavailable
    rep.verdict = 'review' if rep.reasons else 'pass'
    return rep


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='QC a Genjutsu output')
    ap.add_argument('output'); ap.add_argument('--source'); ap.add_argument('--ref'); ap.add_argument('--mode', default='motion_transfer')
    ap.add_argument('--no-learned', action='store_true')
    a = ap.parse_args()
    print(json.dumps(run(a.output, a.source, a.ref, a.mode, learned=not a.no_learned).to_dict(), indent=1))
