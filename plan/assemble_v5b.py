"""Box entry for the A08 stitch. The pod runs the same engine/worker/post_a08.py during linger.
Overlap crossfade stays (stitch blending, not frame interpolation). No LAB match, no B&W grade,
no grain, no frame interpolation, no face restore, no person-mask paste-back.
"""
import glob, json, os, pathlib, subprocess, sys
slug = sys.argv[1]
V = os.environ.get('FARQUAD_HOME', os.path.expanduser('~/farquad'))
acc = json.load(open(f'{V}/plan/run_{slug}/accepted.json'))
keys = sorted([k for k in acc if k.startswith(slug + ':')], key=lambda k: int(k.split(':')[1]))
pieces = [acc[k]['out'] for k in keys]
native = glob.glob(f'{V}/{slug}/{slug}_*_native.mp4')[0]
fd = pathlib.Path(f'{V}/{slug}/final')
fd.mkdir(exist_ok=True)
post = os.path.join(V, 'engine', 'worker', 'post_a08.py')
cmd = [sys.executable, post, '--native', native, '--out', str(fd / f'{slug}_v1_final.mp4'),
       '--side', str(fd / f'{slug}_v1_side_by_side.mp4'), '--pieces', *pieces]
raise SystemExit(subprocess.call(cmd))
