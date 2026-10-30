"""Pod jobs: per-job ledger + cap, SECURE launch with a cap guard, run_graph (FULL error text), terminate/sweep, spend watchdog. Never prints keys."""
import fmbcfg as CFG
import os, sys, json, time, copy, subprocess, threading, requests, pathlib, glob
import rpod as R
GPU = R.GPU
BASE = CFG.RUN_DIR
KEY = os.environ.get('RUNPOD_API_KEY', '')
PODFILES = {}
COMFY = '/opt/gj/ComfyUI'
class Job:
    def __init__(s, name, pfx, cap):
        s.name, s.pfx, s.cap = name, pfx, cap; s.dir = BASE / name; s.dir.mkdir(parents=True, exist_ok=True); s.L = s.dir / 'ledger.json'; s.lock = threading.Lock()
        (s.dir / 'CAP.txt').write_text(str(cap))
    def log(s, m):
        line = f"{time.strftime('%T')} {m}"; open(s.dir / 'run.log', 'a').write(line + '\n'); print(f'[{s.name}] {line}', flush=True)
    def led(s): return json.load(open(s.L)) if s.L.exists() else {}
    def save(s, d): json.dump(d, open(s.L, 'w'), indent=1)
    def spend(s):
        now = time.time(); return sum(p['rate'] * ((p.get('t_end') or now) - p['t0']) / 3600 for p in s.led().values())
    def burn(s): return sum(p['rate'] for p in s.led().values() if not p.get('t_end'))
    def stopped(s): return (s.dir / 'STOP').exists()
    def launch(s, role, sets, env=None, max_wait_s=1200, est_min=30):
        """SECURE only, the job's GPU type (R.GPU / GJ_GPU). Refuses if spend + (burn + rate) * est would pass 95% of the cap. Retries for stock."""
        t_start = time.time(); n = 0; rate_guess = float(os.environ.get('FMB_RATE', '2.49'))
        while time.time() - t_start < max_wait_s:
            if s.stopped(): return None
            if s.spend() + (s.burn() + rate_guess) * est_min / 60 > s.cap * 0.95: s.log(f'launch {role}: projected spend would pass cap ${s.cap}; not launching'); return None
            os.environ['CLOUD'] = 'SECURE'; name = f'{s.pfx}{role}'
            b = R.pod_body(name, sets); b['env'].update(env or {}); assert b['cloudType'] == 'SECURE'
            try:
                r = requests.post(R.REST + '/pods', json=b, headers={'Authorization': f'Bearer {KEY}'}, timeout=90)
                if r.status_code >= 300: raise RuntimeError(f'{r.status_code} {r.text[:200]}')
                p = r.json(); pid = p['id']; rate = float(p.get('costPerHr') or rate_guess)
                with s.lock: d = s.led(); d[pid] = dict(name=name, role=role, sets=sets, gpu=R.GPU, cloud='SECURE', rate=rate, t0=time.time(), t_end=None); s.save(d)
                s.log(f'pod {pid} {name} SECURE ${rate}/h sets {sets}'); return pid
            except Exception as e:
                n += 1
                if n % 5 == 1: s.log(f'launch {role} failed ({n}): {str(e)[:160]}')
            time.sleep(20)
        s.log(f'launch {role}: no {R.GPU} SECURE stock after {max_wait_s}s'); return None
    def terminate(s, pid, why):
        try: R.v2_terminate(pid)   # v2 DELETE /v2/pods/{id} (was GraphQL podTerminate)
        except Exception as e: s.log(f'terminate {pid} err {str(e)[:120]}')
        with s.lock:
            d = s.led()
            if pid in d and not d[pid].get('t_end'): d[pid]['t_end'] = time.time(); s.save(d)
        s.log(f'terminated {pid} ({why}) spend now ${s.spend():.2f}')
    def live(s): return [p for p in R.mypods() if p['name'].startswith(s.pfx)]
    def wait_ready(s, pid, deadline_s=780):
        t0 = s.led().get(pid, {}).get('t0', time.time())   # deadline counts from launch
        while time.time() - t0 < deadline_s:
            if s.stopped(): return False
            if R.ready(pid): s.log(f'{pid} READY after {round(time.time() - s.led()[pid]["t0"])}s'); R.save_timings(pid); return True
            if R.boot_failed(pid): s.log(f'{pid} BOOT_FAILED'); R.save_timings(pid); return False
            time.sleep(15)
        s.log(f'{pid} not ready after {deadline_s}s'); return False
    def get_pod(s, role, sets, env=None, est_min=30, tries=3):
        for pid, m in s.led().items():   # adopt a pod this job already launched (script restart)
            if m['name'] == s.pfx + role and not m.get('t_end') and any(p['id'] == pid for p in s.live()):
                s.log(f'adopting {pid} for {role}')
                if s.wait_ready(pid): return pid
                s.terminate(pid, 'adopted pod not ready')
        for t in range(tries):
            pid = s.launch(role + ('' if t == 0 else f'r{t}'), sets, env, est_min=est_min)
            if not pid: return None
            if s.wait_ready(pid): return pid
            s.terminate(pid, 'not ready / boot failed')
        return None

def upload(pid, path, name=None):
    with open(path, 'rb') as f:
        r = requests.post(R.url(pid) + '/upload/image', files={'image': (name or os.path.basename(path), f, 'application/octet-stream')}, data={'overwrite': 'true'}, timeout=600); r.raise_for_status()
def run_graph(job, pid, graph, uploads, outdir, name, timeout=3600):
    """uploads: {input_name: local_path}. Returns (ok, [local files], err, secs)."""
    U = R.url(pid); t0 = time.time(); os.makedirs(outdir, exist_ok=True)
    for nm, lp in uploads.items(): upload(pid, lp, nm)
    r = requests.post(U + '/prompt', json={'prompt': graph}, timeout=60)
    if r.status_code != 200: return False, [], 'POST ' + r.text[:600], 0
    p = r.json()['prompt_id']; files = []; err = None
    while True:
        if time.time() - t0 > timeout: err = 'timeout'; break
        if job.stopped(): err = 'job STOP'; break
        try: h = requests.get(U + f'/history/{p}', timeout=30).json()
        except Exception: time.sleep(10); continue
        if p in h:
            st = h[p].get('status', {})
            if st.get('status_str') == 'error': err = json.dumps(st.get('messages', [])); break   # full (was [-1500:], which cut the exception message; Oct 29)
            for o in h[p].get('outputs', {}).values():
                for v in o.values():
                    if isinstance(v, list):
                        for it in v:
                            if isinstance(it, dict) and it.get('filename') and it.get('type', 'output') == 'output': files.append(it)
            break
        time.sleep(5)
    out = []; PODFILES[name] = [f"/opt/gj/ComfyUI/output/{it.get('subfolder') + '/' if it.get('subfolder') else ''}{it['filename']}" for it in files]
    for i, it in enumerate(files):
        ext = os.path.splitext(it['filename'])[1]; lp = f'{outdir}/{name}{"" if len(files) == 1 else "_" + str(i)}{ext}'
        rr = requests.get(U + '/view', params={'filename': it['filename'], 'subfolder': it.get('subfolder', ''), 'type': 'output'}, timeout=600); rr.raise_for_status()
        open(lp, 'wb').write(rr.content); out.append(lp)
    secs = round(time.time() - t0); job.log(f'{name} on {pid}: ok={bool(out) and not err} {secs}s {err[:300] if err else ""}')
    return bool(out) and not err, out, err, secs
def comfy_log(pid):
    try: return requests.get(R.url(pid) + '/internal/logs/raw', timeout=30).text
    except Exception as e: return f'log fetch failed {e!r}'

def sheet(files, out, labels=None, h=360):
    """simple contact sheet with ffmpeg-free PIL."""
    from PIL import Image, ImageDraw
    ims = []
    for f in files:
        im = Image.open(f).convert('RGB'); im = im.resize((max(1, int(im.width * h / im.height)), h)); ims.append(im)
    W = sum(i.width for i in ims) + 10 * (len(ims) + 1); S = Image.new('RGB', (W, h + 40), 'white'); x = 10; d = ImageDraw.Draw(S)
    for k, im in enumerate(ims):
        S.paste(im, (x, 30)); d.text((x, 8), (labels[k] if labels else os.path.basename(files[k]))[:60], fill='black'); x += im.width + 10
    S.save(out, quality=88); return out
def wait_pick(job, key, cands, default, timeout=420):
    """write a pick sheet and wait for <dir>/PICK_<key>.txt (a candidate basename or index); else default."""
    f = job.dir / f'PICK_{key}.txt'; sh = job.dir / f'pick_{key}.jpg'; sheet(cands, sh, [f'{i}: {os.path.basename(c)}' for i, c in enumerate(cands)])
    job.log(f'PICK {key}: sheet {sh}; default {os.path.basename(default)}; waiting up to {timeout}s'); t0 = time.time()
    while time.time() - t0 < timeout:
        if f.exists():
            v = f.read_text().strip()
            sel = cands[int(v)] if v.isdigit() and int(v) < len(cands) else next((c for c in cands if os.path.basename(c).startswith(v)), default)
            job.log(f'PICK {key} -> {os.path.basename(sel)}'); return sel
        time.sleep(5)
    job.log(f'PICK {key} timeout -> default {os.path.basename(default)}'); return default

def run_phase(job, pid, jobs, uploads_common, outdir, parallel=False):
    res = {}
    for j in jobs:
        if job.stopped(): break
        ex = sorted(glob.glob(f'{outdir}/{j["name"]}.*'))
        if ex: res[j['name']] = ex[0]; continue   # resume: already made
        ok, files, err, secs = run_graph(job, pid, j['graph'], uploads_common, outdir, j['name'])
        if not ok:   # one retry
            ok, files, err, secs = run_graph(job, pid, j['graph'], uploads_common, outdir, j['name'])
        res[j['name']] = files[0] if ok and files else None
    return res

def watchdog(job, poll=20):
    """runs in its own process: terminates every pod with the job prefix once spend >= 97% of cap; writes STOP."""
    seen = {}
    while not (job.dir / 'DONE').exists():
        try:
            live = job.live(); now = time.time(); d = job.led()
            for p in live:   # pods the ledger missed (should not happen): add with first-seen time
                if p['id'] not in d:
                    d[p['id']] = dict(name=p['name'], role='unknown', rate=float(p.get('costPerHr') or 2.49), t0=now - (p.get('runtime') or {}).get('uptimeInSeconds', 0) if p.get('runtime') else now, t_end=None); job.save(d)
            live_ids = {p['id'] for p in live}
            for pid, m in d.items():   # pods gone from RunPod (self-removed): close them
                if not m.get('t_end') and pid not in live_ids and now - m['t0'] > 120:
                    seen[pid] = seen.get(pid, 0) + 1
                    if seen[pid] >= 3: m['t_end'] = now; job.save(d); job.log(f'watchdog: {pid} gone, closed in ledger')
            sp = job.spend()
            if sp >= 0.97 * job.cap:
                (job.dir / 'STOP').write_text(f'cap {sp:.2f}'); job.log(f'WATCHDOG CAP: spend ${sp:.2f} >= 97% of ${job.cap}; terminating all {job.pfx} pods')
                for p in live: job.terminate(p['id'], 'watchdog cap')
        except Exception as e: job.log(f'watchdog err {e!r}'[:200])
        time.sleep(poll)
if __name__ == '__main__' and sys.argv[1] == 'watchdog':
    watchdog(Job(sys.argv[2], sys.argv[3], float(sys.argv[4])))
