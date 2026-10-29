"""RunPod helpers for Fire Marshal Bill (from the H3-check controller (fastboot image by default; LEGACY=1 for the old install-at-boot path) (my pods only, name prefix 'k2h3-'). launch N | run | stop. Never prints keys. Combined hard cap with Modal ledger."""
import os, sys, json, time, requests, pathlib
import fmbcfg as CFG
D = CFG.DATA; R = D / 'runpod'; R.mkdir(parents=True, exist_ok=True); KEY = os.environ.get('RUNPOD_API_KEY', ''); GQL = 'https://api.runpod.io/graphql'   # key read by name; only paid calls need it
def q(query, var=None):
    r = requests.post(GQL, json={'query': query, 'variables': var or {}}, headers={'Authorization': f'Bearer {KEY}'}, timeout=60); r.raise_for_status(); d = r.json()
    if d.get('errors'): raise RuntimeError(json.dumps(d['errors'])[:500])
    return d['data']
def mypods(): return [p for p in q('query{myself{pods{id name desiredStatus costPerHr runtime{uptimeInSeconds}}}}')['myself']['pods'] if p['name'].startswith('k2h3-')]
def log(m): open(R / 'rp.log', 'a').write(f"{time.strftime('%T')} {m}\n"); print(m, flush=True)
# ---- fastboot (Oct 27 2026): prebuilt image, no installs at boot. LEGACY=1 -> old runpod/pytorch + SETUP script path.
IMAGE = os.environ.get('GJ_IMAGE', 'ghcr.io/ayoomeco/genie-jutsu-worker:fastboot-comfy-0.42.5-r2')   # r2 = memory-lean h3_exact_lora; rollback GJ_IMAGE=...:fastboot-comfy-0.42.5 (or LEGACY=1)
REG_AUTH = os.environ.get('GJ_REGISTRY_AUTH_ID', '')   # only if the image is private: your RunPod registry-auth id
GPU = os.environ.get('GJ_GPU', 'NVIDIA RTX PRO 6000 Blackwell Server Edition'); REST = 'https://rest.runpod.io/v1'
REPLACE_AFTER = int(os.environ.get('REPLACE_AFTER_S', 360)); MAX_REPLACE = int(os.environ.get('MAX_REPLACE', 3))
# ---- manifest overlay + disk sizing (Oct 28 2026). The image bakes /opt/gj/models.manifest.json at build time, so a set added to
# fast_gpu_start/repo/fastboot/models.manifest.json after the build is NOT on the pod. For such sets the launcher sends just those set
# entries in env GJ_MANIFEST_OVERLAY_B64 and a start cmd that merges them into the baked manifest, then runs the normal boot.sh.
# Sets already baked into the image -> pod body is unchanged (no overlay, no start-cmd override). Same sha256/size checks apply.
import base64, math
MANIFEST = os.environ.get('GJ_MANIFEST', str(CFG.HOME / 'image' / 'models.manifest.json'))
BAKED_SETS = {   # sets inside each image's /opt/gj/models.manifest.json (r2 = commit be2b8c2)
    'ghcr.io/ayoomeco/genie-jutsu-worker:fastboot-comfy-0.42.5-r2': {'h3_a08', 'qwen21', 'krea2', 'seedvr2', 'qwen21_mva'},
    'ghcr.io/ayoomeco/genie-jutsu-worker:fastboot-comfy-0.42.5': {'h3_a08', 'qwen21', 'krea2', 'seedvr2'}}
OVERLAY_CMD = ("python3 -c \"import os,json,base64;p='/opt/gj/models.manifest.json';m=json.load(open(p));"
               "m['sets'].update(json.loads(base64.b64decode(os.environ['GJ_MANIFEST_OVERLAY_B64'])));json.dump(m,open(p,'w'),indent=1);"
               "print('[overlay] manifest sets now',sorted(m['sets']),flush=True)\" || echo '[overlay] FAILED (boot.sh will report unknown set)'; exec bash /opt/gj/boot.sh")
DISK_MIN_GB = 150; DISK_HEADROOM_GB = 40   # headroom: image layers (~11.5 GB compressed) + outputs/logs + .dl temp
def sets_gb(sets):
    M = json.load(open(MANIFEST))['sets']; seen = {}
    for s in [x for x in sets.split(',') if x]:
        for e in M[s]['files']: seen[e['dest']] = e['size']
    return sum(seen.values()) / 1e9
def disk_gb(sets):
    if os.environ.get('GJ_DISK_GB'): return int(os.environ['GJ_DISK_GB'])
    try: need = sets_gb(sets)
    except Exception: return DISK_MIN_GB
    return max(DISK_MIN_GB, int(math.ceil((need + DISK_HEADROOM_GB) / 10.0) * 10))
def apply_overlay(b, sets):
    baked = BAKED_SETS.get(b['imageName'])
    want = [x for x in sets.split(',') if x]
    if baked is None or os.environ.get('GJ_NO_OVERLAY') == '1': return b      # unknown image (e.g. a newer tag with the sets baked) -> leave as is
    extra = [s for s in want if s not in baked]
    if not extra: return b
    M = json.load(open(MANIFEST))['sets']; missing = [s for s in extra if s not in M]
    if missing: raise RuntimeError(f'sets not in {MANIFEST}: {missing}')
    b['env']['GJ_MANIFEST_OVERLAY_B64'] = base64.b64encode(json.dumps({s: M[s] for s in extra}, separators=(',', ':')).encode()).decode()
    b['dockerStartCmd'] = ['bash', '-c', OVERLAY_CMD]
    return b
def pod_body(name, sets):
    env = {'GJ_MODEL_SETS': sets, 'GJ_IDLE_EXIT_MIN': os.environ.get('GJ_IDLE_EXIT_MIN', '15'), 'GJ_MAX_SECONDS': os.environ.get('GJ_MAX_SECONDS', '5400'),
           'GJ_VERIFY': os.environ.get('GJ_VERIFY', 'full'),
           # h3_exact_lora r2 memory (same math): in-place LoRA add + 16384-token chunks for LoRA deltas and DiT MLPs. Off: H3_LORA_CHUNK_TOKENS=0 / H3_LORA_INPLACE=0
           'H3_LORA_INPLACE': os.environ.get('H3_LORA_INPLACE', '1'), 'H3_LORA_CHUNK_TOKENS': os.environ.get('H3_LORA_CHUNK_TOKENS', '16384'), 'H3_MLP_CHUNK': os.environ.get('H3_MLP_CHUNK', '1')}
    b = {'name': name, 'imageName': IMAGE, 'cloudType': os.environ.get('CLOUD', 'SECURE'), 'computeType': 'GPU', 'gpuTypeIds': [GPU], 'gpuCount': 1,
         'containerDiskInGb': disk_gb(sets), 'volumeInGb': 0, 'minVCPUPerGPU': int(os.environ.get('GJ_MIN_VCPU', '8')), 'minRAMPerGPU': int(os.environ.get('GJ_MIN_RAM', '40')), 'ports': ['8188/http'], 'env': env,   # Oct 28: RAM min was 64 GB, which hid A6000/A40 hosts (they list ~50 GB); GJ_MIN_RAM overrides
         **({'containerRegistryAuthId': REG_AUTH} if REG_AUTH else {})}
    # CUDA: no exact pin (Oct 27 fix). Hosts with free cards report 13.1/13.2, and the REST enum stops at 13.0, so an exact ['13.0'] pin hid them.
    # boot.sh refuses drivers < 580 -> BOOT_FAILED -> wait_ready replaces the pod. GJ_CUDA_PIN='13.0' (comma list) restores an explicit pin.
    pin = os.environ.get('GJ_CUDA_PIN', '').strip()
    if pin and os.environ.get('CUDA_ANY') != '1': b['allowedCudaVersions'] = [v.strip() for v in pin.split(',') if v.strip()]
    return apply_overlay(b, sets)   # Oct 28: overlay only for sets not baked into the image; disk sized from the sets (min 150)
def launch_one(name, sets):
    r = requests.post(REST + '/pods', json=pod_body(name, sets), headers={'Authorization': f'Bearer {KEY}'}, timeout=90)
    if r.status_code >= 300: raise RuntimeError(f'{r.status_code} {r.text[:300]}')
    p = r.json(); return p['id'], p.get('costPerHr')
def _pods_meta():
    f = R / 'pods_meta.json'; return json.load(open(f)) if f.exists() else {}
def _save_meta(m): json.dump(m, open(R / 'pods_meta.json', 'w'), indent=1)
def launch(n, sets=None):
    if os.environ.get('LEGACY') == '1': return launch_legacy(n)
    sets = sets or os.environ.get('GJ_MODEL_SETS', 'h3_a08'); ids = []; meta = _pods_meta()
    for i in range(n):
        name = os.environ.get('PODPFX', 'k2h3-') + f'{i:02d}'
        try:
            pid, c = launch_one(name, sets); ids.append(pid); meta[pid] = {'name': name, 'sets': sets, 't': time.time(), 'replaced': 0}; log(f"pod {pid} up {name} ${c}/h image {IMAGE} sets {sets}")
        except Exception as e: log(f'launch {i} failed {str(e)[:300]}')
    _save_meta(meta); json.dump(ids, open(R / 'pods.json', 'w')); return ids
def launch_legacy(n):
    sh = (R / os.environ.get('SETUP', 'pod_setup.sh')).read_text(); ids = []
    for i in range(n):
        inp = {'cloudType': 'ALL', 'gpuCount': 1, 'volumeInGb': 0, 'containerDiskInGb': 150, 'minVcpuCount': 8, 'minMemoryInGb': int(os.environ.get('GJ_MIN_RAM', '40')),
               'gpuTypeId': 'NVIDIA RTX PRO 6000 Blackwell Server Edition', 'name': os.environ.get('PODPFX', 'k2h3-') + f'{i:02d}', 'imageName': 'runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04',
               'dockerArgs': "bash -c 'printf \"%s\" \"$SETUP_SH\" > /s.sh && bash /s.sh'", 'ports': '8188/http', 'env': [{'key': 'SETUP_SH', 'value': sh}]}
        try:
            p = q('mutation($i:PodFindAndDeployOnDemandInput){podFindAndDeployOnDemand(input:$i){id costPerHr}}', {'i': inp})['podFindAndDeployOnDemand']; ids.append(p['id']); log(f"pod {p['id']} up {i:02d} ${p['costPerHr']}/h")
        except Exception as e: log(f'launch {i} failed {str(e)[:300]}')
    json.dump(ids, open(R / 'pods.json', 'w'))
def spend():
    L = R / 'ledger.json'; led = json.load(open(L)) if L.exists() else {'integ': 0.0, 'last': time.time()}
    now = time.time(); ps = mypods(); rate = sum(float(p.get('costPerHr') or 0) for p in ps if p['desiredStatus'] == 'RUNNING')
    led['integ'] += rate * (now - led['last']) / 3600; led['last'] = now; led['rate'] = rate; led['n'] = len(ps); json.dump(led, open(L, 'w')); return led['integ'], ps
def stop():
    for p in mypods(): q('mutation($i:PodTerminateInput!){podTerminate(input:$i)}', {'i': {'podId': p['id']}}); log(f"terminated {p['id']}")
if __name__ == '__main__':
    a = sys.argv[1]
    if a == 'launch': launch(int(sys.argv[2]))
    elif a == 'stop': stop()
    elif a == 'spend': print(spend()[0], len(mypods()))

# ---------------- job runner (per pod thread) ----------------
def url(pid): return f'https://{pid}-8188.proxy.runpod.net'
def ready(pid):
    try: return requests.get(url(pid) + '/view', params={'filename': 'READY.txt', 'type': 'output'}, timeout=20).status_code == 200
    except Exception: return False
def boot_failed(pid):
    try: return requests.get(url(pid) + '/view', params={'filename': 'BOOT_FAILED.txt', 'type': 'output'}, timeout=20).status_code == 200
    except Exception: return False
def terminate(pid, why):
    try: q('mutation($i:PodTerminateInput!){podTerminate(input:$i)}', {'i': {'podId': pid}}); log(f"terminated {pid} ({why})")
    except Exception as e: log(f'terminate {pid} failed {str(e)[:200]}')
def save_timings(pid):
    (D / 'logs').mkdir(exist_ok=True)
    for fn in ('boot_timings.json', 'fetch_report.json'):
        try:
            r = requests.get(url(pid) + '/view', params={'filename': fn, 'type': 'output'}, timeout=20)
            if r.status_code == 200: open(D / 'logs' / f'{pid}_{fn}', 'wb').write(r.content)
        except Exception: pass
def wait_ready(pods, need, deadline_s=1500):
    """Poll until `need` pods are READY. Pods with BOOT_FAILED or not ready REPLACE_AFTER s after launch are terminated and replaced (max MAX_REPLACE total)."""
    meta = _pods_meta(); t0 = time.time(); pods = list(pods); rd = []; replaced = 0
    while True:
        for p in list(pods):
            if p in rd: continue
            if ready(p): rd.append(p); age = round(time.time() - meta.get(p, {}).get('t', t0)); log(f'{p} READY {age}s after launch'); save_timings(p); continue
            age = time.time() - meta.get(p, {}).get('t', t0); bf = boot_failed(p)
            if bf or age > REPLACE_AFTER:
                save_timings(p); terminate(p, 'BOOT_FAILED' if bf else f'not ready after {round(age)}s'); pods.remove(p)
                if replaced < MAX_REPLACE and len(rd) < need and os.environ.get('LEGACY') != '1':
                    m = meta.get(p, {}); name = m.get('name', 'k2h3-r') + 'r'
                    try: nid, c = launch_one(name, m.get('sets', os.environ.get('GJ_MODEL_SETS', 'h3_a08'))); pods.append(nid); replaced += 1; meta[nid] = {'name': name, 'sets': m.get('sets'), 't': time.time(), 'replaced': m.get('replaced', 0) + 1}; _save_meta(meta); log(f'replacement {nid} for {p} ${c}/h')
                    except Exception as e: log(f'replacement launch failed {str(e)[:200]}')
        if len(rd) >= need or not [p for p in pods if p not in rd] or time.time() - t0 > deadline_s: return rd, [p for p in pods if p not in rd]
        time.sleep(15)
def run_jobs(pid, jobs, refdir, outdir):
    import concurrent.futures
    U = url(pid); res = []
    for j in jobs:
        t0 = time.time()
        try:
            for src, name in j['needs']:
                with open(f'{refdir}/{os.path.basename(src)}', 'rb') as f:
                    r = requests.post(U + '/upload/image', files={'image': (name, f, 'image/png')}, data={'overwrite': 'true'}, timeout=300); r.raise_for_status()
            r = requests.post(U + '/prompt', json={'prompt': j['graph']}, timeout=60)
            if r.status_code != 200: res.append({'name': j['name'], 'ok': False, 'err': 'POST ' + r.text[:800], 'pod': pid}); continue
            p = r.json()['prompt_id']; path = None; err = None
            while time.time() - t0 < 3600:
                h = requests.get(U + f'/history/{p}', timeout=30).json()
                if p in h:
                    st = h[p].get('status', {})
                    if st.get('status_str') == 'error': err = json.dumps(st.get('messages', []))[-1500:]; break
                    for o in h[p].get('outputs', {}).values():
                        for v in o.values():
                            if isinstance(v, list):
                                for it in v:
                                    if isinstance(it, dict) and it.get('filename', '').endswith('.mp4'): path = it
                    break
                time.sleep(10)
            if path:
                r = requests.get(U + '/view', params={'filename': path['filename'], 'subfolder': path.get('subfolder', ''), 'type': 'output'}, timeout=600); r.raise_for_status()
                open(f'{outdir}/{j["name"]}.mp4', 'wb').write(r.content)
            res.append({'name': j['name'], 'ok': bool(path), 'err': err or (None if path else 'timeout/no output'), 'secs': round(time.time() - t0), 'pod': pid})
        except Exception as e: res.append({'name': j['name'], 'ok': False, 'err': repr(e)[:500], 'pod': pid})
        log(json.dumps(res[-1])[:400])
    return res
def run(jobfile, refdir, outdir, per=2):
    import concurrent.futures
    J = json.load(open(jobfile)); os.makedirs(outdir, exist_ok=True)
    BF = R / 'busy.json'; busy = set(json.load(open(BF))) if BF.exists() else set()
    pods = [p['id'] for p in mypods() if p['desiredStatus'] == 'RUNNING' and p['id'] not in busy]; t0 = time.time()
    need = max(1, -(-len(J) // per)); rd, notready = wait_ready(pods, need)
    log(f'{len(rd)}/{len(pods)} pods ready after {round(time.time()-t0)}s')
    for p in rd[need:] + notready: terminate(p, 'not needed')          # never leave spare pods billing
    if not rd: log('no pods ready; nothing run'); return
    rd = rd[:max(1, -(-len(J) // per))]; busy = set(json.load(open(BF))) if BF.exists() else set(); json.dump(sorted(busy | set(rd)), open(BF, 'w'))
    chunks = [J[i::len(rd)] for i in range(len(rd))]
    def work(a):
        r = run_jobs(*a)
        if os.environ.get('KEEP_WARM') != '1': save_timings(a[0]); terminate(a[0], 'jobs done')   # KEEP_WARM=1: pod self-removes after GJ_IDLE_EXIT_MIN idle
        return r
    with concurrent.futures.ThreadPoolExecutor(len(rd)) as ex: out = sum(ex.map(work, [(p, c, refdir, outdir) for p, c in zip(rd, chunks)]), [])
    busy = set(json.load(open(BF))) - set(rd); json.dump(sorted(busy), open(BF, 'w'))
    json.dump(out, open(D / f'results_h3check_{pathlib.Path(jobfile).stem}.json', 'w'), indent=1); log(f"h3check done {sum(r['ok'] for r in out)}/{len(out)}")
if __name__ == '__main__' and sys.argv[1] == 'run': run(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]) if len(sys.argv) > 5 else 2)
if __name__ == '__main__' and sys.argv[1] == 'plan': print(json.dumps(pod_body('k2h3-00', os.environ.get('GJ_MODEL_SETS', 'h3_a08')), indent=1))   # $0: show the pod body, launches nothing
if __name__ == '__main__' and sys.argv[1] == 'ready': print({p['id']: ready(p['id']) for p in mypods()})
if __name__ == '__main__' and sys.argv[1] == 'stopone':
    for i in sys.argv[2:]:
        if any(p['id'] == i for p in mypods()): q('mutation($i:PodTerminateInput!){podTerminate(input:$i)}', {'i': {'podId': i}}); log(f"terminated {i}")
