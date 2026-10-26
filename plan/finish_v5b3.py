"""Run finisher: on DRIVER_DONE, stop keepalive, run the A08 stitch/upscale/encode on the pod,
then terminate this run's pod and verify 0 pods remain. Does not stitch on the box when the pod post fails.
Usage: python finish_v5b3.py <run dir> <slug...>   DRY=1 = no API calls (test).
POST_LOCAL=1 runs engine/worker/post_a08.py on this box (tests). POST_WAIT seconds to wait for the pod log.
"""
import sys, os, json, time, subprocess, pathlib, glob, uuid, urllib.request
FH = os.environ.get('FARQUAD_HOME', os.path.expanduser('~/farquad'))
R = pathlib.Path(sys.argv[1]); SL = sys.argv[2:]; DRY = os.environ.get('DRY') == '1'

def log(*a):
    s = time.strftime('%T') + ' ' + ' '.join(map(str, a)); print(s, flush=True); open(R / 'finish.log', 'a').write(s + '\n')

(R / 'KEEPALIVE_OFF').write_text('finish')

def pods():
    if DRY:
        return json.loads(os.environ.get('DRY_PODS', '[]'))
    sys.path.insert(0, FH + '/lib'); import pods as P; return P.ls(), P

def pod_id():
    if (R / 'POD_ID').exists():
        return (R / 'POD_ID').read_text().strip()
    if (R / 'checkpoint.json').exists():
        try:
            return json.load(open(R / 'checkpoint.json')).get('pod')
        except Exception:
            return None
    return None

def kill_all():
    for a in range(5):
        if DRY:
            log('DRY pods', 0); return 0
        L, P = pods()
        others = [r for r in R.parent.glob('run_*') if r != R and not (r / 'DRIVER_DONE').exists() and not (r / 'STOP').exists()]
        pid = pod_id()
        if pid:
            mine = [p for p in L if p['id'] == pid]
        elif others:
            log('other runs active, no POD_ID: leaving pods to idle_watchdog', [o.name for o in others]); return len(L)
        else:
            log('no POD_ID for this run: not touching other pods'); return len(L)
        if not mine:
            log('verified 0 pods', 'try', a + 1); return 0
        for p in mine:
            try:
                P.c.terminate_pod(p['id']); log('terminated', p['id'])
            except Exception as e:
                log('terminate err', type(e).__name__)
        time.sleep(20)
    L, _ = pods(); log('WARNING pods still listed', len(L)); return len(L)

def accepted(slug):
    for path in (R / 'accepted.json', pathlib.Path(FH) / 'plan' / f'run_{slug}' / 'accepted.json'):
        if path.exists():
            acc = json.load(open(path))
            keys = sorted([k for k in acc if k.startswith(slug + ':')], key=lambda k: int(k.split(':')[1]))
            if keys:
                return [acc[k]['out'] for k in keys]
    return []

def native_for(slug):
    hits = sorted(glob.glob(f'{FH}/{slug}/{slug}_*_native.mp4'))
    return hits[0] if hits else None

def upload(pod, path, name):
    data = pathlib.Path(path).read_bytes()
    bnd = uuid.uuid4().hex
    body = (f'--{bnd}\r\nContent-Disposition: form-data; name="image"; filename="{name}"\r\n'
            f'Content-Type: application/octet-stream\r\n\r\n').encode() + data + (
            f'\r\n--{bnd}\r\nContent-Disposition: form-data; name="overwrite"\r\n\r\ntrue\r\n--{bnd}--\r\n').encode()
    req = urllib.request.Request(
        f'https://{pod}-8188.proxy.runpod.net/upload/image', data=body,
        headers={'Content-Type': f'multipart/form-data; boundary={bnd}'})
    with urllib.request.urlopen(req, timeout=600) as resp:
        payload = json.loads(resp.read().decode())
    got = payload.get('name')
    if not got:
        raise RuntimeError(f'upload of {name} returned no name')
    return got

def download(pod, name, dest):
    url = f'https://{pod}-8188.proxy.runpod.net/view?filename={name}&type=output'
    urllib.request.urlretrieve(url, dest)

def run_local(slug, pieces, native, dest, side):
    post = os.path.join(FH, 'engine', 'worker', 'post_a08.py')
    cmd = [sys.executable, post, '--native', native, '--out', dest, '--side', side, '--pieces', *pieces]
    p = subprocess.run(cmd)
    return p.returncode == 0

def queue_pod(slug_jobs):
    pid = pod_id()
    if not pid:
        log('POST_FAIL no POD_ID'); return False
    lines = ['#!/bin/bash', 'set -uo pipefail', 'ok=1']
    for slug, pieces, native, final_name, side_name in slug_jobs:
        try:
            nat_name = upload(pid, native, f'a08_{slug}_native.mp4')
            piece_names = []
            for i, piece in enumerate(pieces):
                piece_names.append(upload(pid, piece, f'a08_{slug}_piece_{i}.mp4'))
        except Exception as e:
            log('POST_FAIL upload', slug, type(e).__name__, str(e)[:200]); return False
        inp = '/opt/jutsu/ComfyUI/input'
        outp = '/opt/jutsu/ComfyUI/output'
        args = ' '.join(f'{inp}/{n}' for n in piece_names)
        lines.append(
            f'python3 /opt/jutsu-src/worker/post_a08.py --native {inp}/{nat_name} '
            f'--out {outp}/{final_name} --side {outp}/{side_name} --pieces {args} || ok=0')
    lines.append('if [ "$ok" = 1 ]; then echo POST_DONE; else echo POST_FAIL; exit 1; fi')
    qd = R / 'pod_cmds'; qd.mkdir(parents=True, exist_ok=True)
    script = qd / '90_a08_post.sh'
    for extra in (script.with_suffix('.job'), script.with_suffix('.out')):
        extra.unlink(missing_ok=True)
    script.write_text('\n'.join(lines) + '\n')
    log('queued', script.name, 'pod', pid)
    deadline = time.time() + int(os.environ.get('POST_WAIT', '1500'))
    outp = script.with_suffix('.out')
    while time.time() < deadline:
        if outp.exists():
            text = outp.read_text()
            if 'POST_DONE' in text and 'rc=0' in text:
                fd = pathlib.Path(FH)
                for slug, _pieces, _native, final_name, side_name in slug_jobs:
                    dest_dir = fd / slug / 'final'
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    try:
                        download(pid, final_name, dest_dir / f'{slug}_v1_final.mp4')
                        download(pid, side_name, dest_dir / f'{slug}_v1_side_by_side.mp4')
                        log('downloaded', slug)
                    except Exception as e:
                        log('POST_FAIL download', slug, type(e).__name__); return False
                return True
            if 'POST_FAIL' in text or 'rc=' in text:
                log('POST_FAIL pod post', text[-500:]); return False
        time.sleep(10)
    log('POST_FAIL timed out waiting for pod post'); return False

res = {'pods_left': None, 'stitched': {}}
post_ok = True
if DRY:
    (R / 'STOP').write_text('driver done')
    for s in SL:
        res['stitched'][s] = 'dry'
else:
    jobs = []
    for s in SL:
        pieces, native = accepted(s), native_for(s)
        if not pieces or not native:
            log('POST_FAIL missing pieces or native audio', s, 'pieces', len(pieces), 'native', bool(native))
            post_ok = False
            res['stitched'][s] = 'fail: missing inputs'
            continue
        if os.environ.get('POST_LOCAL') == '1':
            dest_dir = pathlib.Path(FH) / s / 'final'
            dest_dir.mkdir(parents=True, exist_ok=True)
            ok = run_local(s, pieces, native, str(dest_dir / f'{s}_v1_final.mp4'), str(dest_dir / f'{s}_v1_side_by_side.mp4'))
            res['stitched'][s] = 'ok' if ok else 'fail: local post'
            post_ok = post_ok and ok
        else:
            jobs.append((s, pieces, native, f'a08_{s}_final.mp4', f'a08_{s}_side.mp4'))
    if jobs:
        post_ok = queue_pod(jobs) and post_ok
        for s, *_rest in jobs:
            res['stitched'].setdefault(s, 'ok' if post_ok else 'fail: pod post')
    (R / 'STOP').write_text('post done' if post_ok else 'post failed')
res['pods_left'] = kill_all()
json.dump(res, open(R / 'FINISHED.json', 'w'), indent=1)
log('finished', res)
if not DRY and not post_ok:
    sys.exit(1)
