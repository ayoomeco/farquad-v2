#!/usr/bin/env python3
"""Download the Genie Jutsu model manifest onto a RunPod network volume ONCE (idempotent, resumable).

    python bootstrap_volume.py --manifest models.manifest.json --root /workspace --group phase1
    python bootstrap_volume.py ... --dry-run          # print plan + total size, download nothing

- Files land at <root>/<target> (e.g. /workspace/models/diffusion_models/x.safetensors).
- Already-complete files (size match) are skipped, so re-running costs nothing. Partial files resume (HTTP Range).
- Uses huggingface_hub + hf_transfer when installed (fastest), else plain stdlib HTTP with resume.
- HF_TOKEN is read from the environment only if set (all phase-1 repos are public). Never written anywhere.
- Writes <root>/models/.jutsu-manifest-state.json with what is present."""
from __future__ import annotations
import argparse, concurrent.futures as cf, hashlib, json, os, pathlib, shutil, subprocess, sys, threading, time, urllib.request


def select(manifest: dict, group: str | None, bundles: list[str] | None, ids: list[str] | None) -> list[dict]:
    want = set(bundles or [])
    if group:
        want |= set(manifest['groups'][group])
    out = []
    for m in manifest['models']:
        if (ids and m['id'] in ids) or (want and set(m['bundles']) & want) or (not want and not ids):
            out.append(m)
    return out


def _http_get(url: str, dest: pathlib.Path, size: int | None, token: str | None):
    part = dest.with_suffix(dest.suffix + '.part')
    have = part.stat().st_size if part.exists() else 0
    headers = {'User-Agent': 'genie-jutsu-bootstrap/1'}
    if token:
        headers['Authorization'] = f'Bearer {token}'
    if have:
        headers['Range'] = f'bytes={have}-'
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=120) as r, open(part, 'ab' if have else 'wb') as f:
        if have and r.status != 206:      # server ignored Range: restart
            f.seek(0); f.truncate()
        shutil.copyfileobj(r, f, length=16 * 1024 * 1024)
    if size and part.stat().st_size != size:
        raise IOError(f'size mismatch for {dest.name}: {part.stat().st_size} != {size}')
    part.rename(dest)


# --- Oct 20: fast + stall-proof path. aria2c (16 conns, resume) or curl -C -, each run under a byte-growth watchdog:
# a file whose bytes don't grow for STALL_S is killed and restarted (resumes); if NO file anywhere grows for ABORT_S the
# whole run aborts (pod then terminates instead of burning money, cf. run_v3 stuck at 71/90 GB for 20 min).
STALL_S = int(os.environ.get('JUTSU_DL_STALL_S', 180))
ABORT_S = int(os.environ.get('JUTSU_DL_ABORT_S', 300))
_last_global = [time.time()]
_abort = threading.Event()


class Aborted(RuntimeError):
    pass


def _bytes(paths) -> int:
    n = 0
    for q in paths:
        try:
            n += q.stat().st_size
        except OSError:
            pass
    return n


def watched_run(cmd: list[str], watch: list[pathlib.Path], stall_s: float | None = None, poll: float = 5.0,
                popen=subprocess.Popen, clock=time.time, sleep=time.sleep) -> int:
    """Run cmd; kill it if the watched files stop growing for stall_s. Returns rc, or -9 on stall kill.
    Raises Aborted if the global no-progress timer (ABORT_S, across all workers) expires."""
    stall_s = STALL_S if stall_s is None else stall_s
    pr = popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    last_n, last_t = _bytes(watch), clock()
    while pr.poll() is None:
        sleep(poll); now = clock(); n = _bytes(watch)
        if n > last_n:
            last_n, last_t = n, now; _last_global[0] = now
        if _abort.is_set() or now - _last_global[0] > ABORT_S:
            _abort.set(); pr.kill(); pr.wait(); raise Aborted(f'no download progress anywhere for {ABORT_S}s')
        if now - last_t > stall_s:
            pr.kill(); pr.wait(); return -9
    return pr.returncode


def _fast_get(url: str, dest: pathlib.Path, size: int | None, token: str | None, conns: int = 16):
    """aria2c multi-connection (falls back to curl). Restarts stalled transfers; both tools resume."""
    part = dest.with_suffix(dest.suffix + '.part')
    hdr = [f'Authorization: Bearer {token}'] if token else []
    if shutil.which('aria2c'):
        cmd = ['aria2c', '-q', '-c', '-x', str(conns), '-s', str(conns), '-k', '64M', '--file-allocation=none',
               '--max-tries=0', '--retry-wait=3', '--timeout=60', '--connect-timeout=30', '--lowest-speed-limit=200K',
               '--auto-file-renaming=false', '--allow-overwrite=true', '-d', str(part.parent), '-o', part.name]
        cmd += [x for h in hdr for x in ('--header', h)] + [url]
        watch = [part, part.with_name(part.name + '.aria2')]
    else:
        cmd = ['curl', '-fsSL', '-C', '-', '--retry', '5', '--speed-limit', '200000', '--speed-time', '60', '-o', str(part)]
        cmd += [x for h in hdr for x in ('-H', h)] + [url]
        watch = [part]
    for restart in range(30):
        rc = watched_run(cmd, watch)
        if rc == 0 and (not size or part.stat().st_size == size):
            part.rename(dest); return
        if rc == 33 or (rc == 0 and size and part.stat().st_size > size):   # curl 33 = server ignores Range
            part.unlink()                                   # corrupt/oversized partial: start over
        print(f'[bootstrap] restart {restart + 1} {dest.name} rc={rc} have={_bytes([part]) / 1e9:.2f} GB', flush=True)
    raise IOError(f'{dest.name}: too many restarts')


def _hf_get(m: dict, dest: pathlib.Path, token: str | None):
    from huggingface_hub import hf_hub_download
    tmpdir = dest.parent / '.hf-staging'
    p = hf_hub_download(m['repo'], m['file'], local_dir=str(tmpdir), token=token)
    shutil.move(p, dest)


def sha256(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(64 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def fetch(m: dict, root: pathlib.Path, verify: bool, token: str | None, prefer_hf: bool, fast: bool = False) -> dict:
    dest = root / m['target']
    dest.parent.mkdir(parents=True, exist_ok=True)
    size = m.get('size_bytes')
    if dest.exists() and (not size or dest.stat().st_size == size):
        return {'id': m['id'], 'status': 'present', 'bytes': dest.stat().st_size, 'seconds': 0}
    t0 = time.time()
    for attempt in range(1, 5):
        try:
            if fast:
                _fast_get(m['url'], dest, size, token)
            elif prefer_hf:
                try:
                    _hf_get(m, dest, token)
                except ImportError:
                    _http_get(m['url'], dest, size, token)
            else:
                _http_get(m['url'], dest, size, token)
            break
        except Aborted as e:
            return {'id': m['id'], 'status': 'failed', 'error': f'aborted: {e}'}
        except Exception as e:
            if attempt == 4:
                return {'id': m['id'], 'status': 'failed', 'error': f'{type(e).__name__}: {e}'}
            time.sleep(5 * attempt)
    st = {'id': m['id'], 'status': 'downloaded', 'bytes': dest.stat().st_size, 'seconds': round(time.time() - t0, 1)}
    if size and dest.stat().st_size != size:
        st.update(status='failed', error='size mismatch')
    elif verify and m.get('sha256'):
        ok = sha256(dest) == m['sha256']
        st['sha256_ok'] = ok
        if not ok:
            st['status'] = 'failed'; dest.unlink()
    return st


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', default=str(pathlib.Path(__file__).with_name('models.manifest.json')))
    ap.add_argument('--root', default=os.environ.get('MODEL_ROOT', '/workspace'))
    ap.add_argument('--group'); ap.add_argument('--bundle', action='append'); ap.add_argument('--id', action='append')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--verify', action='store_true', help='sha256-verify downloads (slow on 20 GB files)')
    ap.add_argument('--no-hf', action='store_true', help='force stdlib HTTP instead of huggingface_hub/hf_transfer')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--engine', default=os.environ.get('JUTSU_DL_ENGINE', 'fast'), choices=['fast', 'hf', 'http'],
                    help='fast = aria2c/curl with stall watchdog (default); hf = huggingface_hub; http = stdlib')
    a = ap.parse_args(argv)
    manifest = json.loads(pathlib.Path(a.manifest).read_text())
    sel = select(manifest, a.group, a.bundle, a.id)
    for m in [m for m in sel if not m.get('url')]:
        print(f"[bootstrap] skip {m['id']}: {m.get('source', 'no url')} (delivered by the runner, not downloaded)")
    sel = [m for m in sel if m.get('url')]
    root = pathlib.Path(a.root)
    total = sum(m.get('size_bytes') or 0 for m in sel)
    todo = [m for m in sel if not ((root / m['target']).exists() and (root / m['target']).stat().st_size == m.get('size_bytes'))]
    need = sum(m.get('size_bytes') or 0 for m in todo)
    print(f'[bootstrap] {len(sel)} files, {total/1e9:.2f} GB selected; {len(todo)} missing ({need/1e9:.2f} GB) -> {root}')
    if a.dry_run:
        for m in sel:
            print(f"  {'have' if m not in todo else 'GET '} {m['size_bytes']/1e9:7.2f} GB  {m['target']}")
        return 0
    free = shutil.disk_usage(root if root.exists() else '/').free
    if need > free:
        print(f'[bootstrap] ERROR not enough space: need {need/1e9:.1f} GB, free {free/1e9:.1f} GB'); return 2
    token = os.environ.get('HF_TOKEN') or None
    os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '1')
    t0 = time.time(); results = []; _last_global[0] = t0
    with cf.ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
        for st in ex.map(lambda m: fetch(m, root, a.verify, token, a.engine == 'hf' and not a.no_hf, a.engine == 'fast'), sel):
            results.append(st); print(f"[bootstrap] {st['status']:10} {st['id']} {st.get('seconds', '')}s {st.get('error', '')}", flush=True)
    secs = time.time() - t0
    state = {'manifest_generated': manifest.get('generated'), 'finished': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
             'seconds': round(secs, 1), 'downloaded_gb': round(need / 1e9, 2),
             'throughput_MBps': round(need / 1e6 / secs, 1) if secs and need else None, 'files': results}
    sp = root / 'models' / '.jutsu-manifest-state.json'; sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(json.dumps(state, indent=1))
    failed = [r for r in results if r['status'] == 'failed']
    print(f'[bootstrap] done in {secs:.0f}s, failed={len(failed)}')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
