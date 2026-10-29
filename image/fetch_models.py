#!/usr/bin/env python3
"""Fetch only the model sets a job needs, in parallel, with sha256 checks and per-file timings.
  python3 fetch_models.py --sets h3_a08[,qwen21,...] [--manifest models.manifest.json] [--root /workspace/models] [--verify full|size]
Primary: huggingface_hub (xet high-performance, progress bars off). Fallback per file: aria2c -x16 on the pinned resolve URL.
Exit 0 only when every file is present with the right size (and sha256 when --verify full). Writes <root>/FETCH_REPORT.json."""
import argparse, concurrent.futures as cf, hashlib, json, os, shutil, subprocess, sys, time
os.environ.setdefault('HF_XET_HIGH_PERFORMANCE', '1'); os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '1')
T0 = time.time()
def log(m): print(f'[fetch +{time.time() - T0:6.1f}s] {m}', flush=True)
def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for ch in iter(lambda: f.read(1 << 24), b''): h.update(ch)
    return h.hexdigest()
def ok(p, e, verify):
    if not os.path.exists(p) or os.path.getsize(p) != e['size']: return False
    return verify != 'full' or not e.get('sha256') or sha256(p) == e['sha256']
def via_hf(e, dst, tmp):
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(e['repo'], e['path'], revision=e['revision'], local_dir=tmp); shutil.move(p, dst)
def via_aria(e, dst, tmp):
    url = f"https://huggingface.co/{e['repo']}/resolve/{e['revision']}/{e['path']}"
    subprocess.run(['aria2c', '-q', '-x16', '-s16', '-k8M', '--file-allocation=none', '--retry-wait=3', '--max-tries=5',
                    '-d', tmp, '-o', os.path.basename(dst), url], check=True); shutil.move(os.path.join(tmp, os.path.basename(dst)), dst)
def get(e, root, verify):
    dst = os.path.join(root, e['dest']); os.makedirs(os.path.dirname(dst), exist_ok=True); t = time.time(); how = 'cached'
    if not (os.path.exists(dst) and os.path.getsize(dst) == e['size']):
        tmp = os.path.join(root, '.dl', e['dest'].replace('/', '_')); os.makedirs(tmp, exist_ok=True); how = 'hf'
        try: via_hf(e, dst, tmp)
        except Exception as x:
            log(f"hf failed {e['dest']}: {str(x)[:200]}; trying aria2c"); how = 'aria2c'
            if os.path.exists(dst): os.remove(dst)
            via_aria(e, dst, tmp)
        shutil.rmtree(tmp, ignore_errors=True)
    dl = time.time() - t; good = ok(dst, e, verify); tot = time.time() - t
    r = {'dest': e['dest'], 'gb': round(e['size'] / 1e9, 2), 'how': how, 'download_s': round(dl, 1), 'verify_s': round(tot - dl, 1),
         'mb_per_s': round(e['size'] / 1e6 / dl, 0) if how != 'cached' and dl > 0 else None, 'ok': good}
    log(json.dumps(r)); return r
def main():
    a = argparse.ArgumentParser(); a.add_argument('--sets', required=True); a.add_argument('--root', default='/workspace/models')
    a.add_argument('--manifest', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models.manifest.json'))
    a.add_argument('--verify', default=os.environ.get('GJ_VERIFY', 'full'), choices=['full', 'size']); a.add_argument('--workers', type=int, default=8)
    o = a.parse_args(); M = json.load(open(o.manifest))['sets']; files, seen = [], set()
    for s in [x for x in o.sets.split(',') if x]:
        if s not in M: sys.exit(f'unknown set {s}; known: {sorted(M)}')
        for e in M[s]['files']:
            if e['dest'] not in seen: seen.add(e['dest']); files.append(e)
    files.sort(key=lambda e: -e['size'])   # biggest first so the long pole starts at t=0
    log(f"{len(files)} files, {sum(e['size'] for e in files) / 1e9:.1f} GB, sets={o.sets}, verify={o.verify}")
    with cf.ThreadPoolExecutor(min(o.workers, len(files))) as ex: res = list(ex.map(lambda e: get(e, o.root, o.verify), files))
    tot = time.time() - T0; gb = sum(r['gb'] for r in res); bad = [r['dest'] for r in res if not r['ok']]
    rep = {'sets': o.sets, 'files': res, 'total_s': round(tot, 1), 'gb': round(gb, 1), 'avg_mb_per_s': round(gb * 1e3 / tot, 0), 'failed': bad}
    json.dump(rep, open(os.path.join(o.root, 'FETCH_REPORT.json'), 'w'), indent=1)
    log(f"done {gb:.1f} GB in {tot:.0f}s ({rep['avg_mb_per_s']} MB/s); failed={bad}"); sys.exit(1 if bad else 0)
if __name__ == '__main__': main()
