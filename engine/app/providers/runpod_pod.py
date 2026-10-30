"""RunPod on-demand Pod adapter: the ONLY live-capable provider in this build.

Safety model (all four must hold before a single cent is spent):
  1. Approval(confirmed=True) for this job, made by the owner (interactive confirm in the bake-off runner).
  2. approval.max_usd <= provider hard_cap_usd (config) -> the cap can only shrink, never grow, per run.
  3. RUNPOD_API_KEY present in the ENVIRONMENT (read at call time, never stored, logged or written to files).
  4. Account balance check: refuses if the projected worst case exceeds the cap.
Once running, a Watchdog terminates the pod when ANY of: spend >= cap * safety, wall clock >= max_seconds,
idle (no heartbeat) >= idle_seconds, Ctrl-C / SIGTERM / process exit (atexit), or an exception in the session.
The pod also gets a dead-man switch (JUTSU_MAX_SECONDS) so it removes itself even if the box dies."""
from __future__ import annotations
import os
import atexit, json, os, signal, threading, time
from typing import Callable
from .base import Provider, ProviderError, ApprovalError, BudgetExceeded, Approval

# ---- RunPod REST v2 shim (migrated Oct 29 2026: REST v1 retires Nov 15 2026; docs.runpod.io/api-reference-v2/migrate-from-v1) ----
# Pure translation only: every recipe value (GPU, image, disk, env, ports, cmd, CUDA, RAM/vCPU mins, DCs) is passed through unchanged.
V2 = 'https://api.runpod.io/v2'
def v2_bodies(b):
    """legacy v1 flat pod body -> list of v2 CreatePodRequest bodies (v2 takes ONE gpu id and ONE cloud; tried in v1 priority order)."""
    o = {'name': b['name']}
    for k1, k2 in (('imageName', 'image'), ('templateId', 'templateId'), ('containerDiskInGb', 'disk'), ('containerRegistryAuthId', 'registry'),
                   ('dockerStartCmd', 'cmd'), ('dockerEntrypoint', 'entrypoint'), ('dockerArgs', 'args'), ('dataCenterIds', 'dataCenterIds')):
        if b.get(k1) not in (None, '', []): o[k2] = b[k1]
    if b.get('dataCenterId') and not o.get('dataCenterIds'): o['dataCenterIds'] = [b['dataCenterId']]
    env = b.get('env') or {}
    if isinstance(env, list): env = {e['key']: e['value'] for e in env}
    if env: o['env'] = {k: str(v) for k, v in env.items()}
    ports = b.get('ports')
    if isinstance(ports, str): ports = [p.strip() for p in ports.split(',') if p.strip()]
    if ports: o['ports'] = ports
    if b.get('networkVolumeId'): o['mounts'] = {'network': [{'volumeId': b['networkVolumeId'], 'path': b.get('volumeMountPath') or '/workspace'}]}
    elif int(b.get('volumeInGb') or 0) > 0: o['mounts'] = {'persistent': {'size': int(b['volumeInGb']), 'path': b.get('volumeMountPath') or '/workspace'}}
    if str(b.get('computeType', 'GPU')).upper() == 'CPU':
        o['cpu'] = {'id': (b.get('cpuFlavorIds') or ['cpu5c'])[0], 'vcpuCount': int(b.get('vcpuCount') or 8)}; gpus = [None]
    else:
        gpus = b.get('gpuTypeIds') or ([b['gpuTypeId']] if b.get('gpuTypeId') else [])
    clouds = ['SECURE', 'COMMUNITY'] if str(b.get('cloudType', 'SECURE')).upper() == 'ALL' else [str(b.get('cloudType') or 'SECURE').upper()]
    out = []
    for c in clouds:
        for g in gpus:
            x = dict(o, cloud=c)
            if g:
                gc = {'id': g, 'count': int(b.get('gpuCount') or 1)}
                if b.get('minRAMPerGPU') or b.get('minMemoryInGb'): gc['minRamPerGpu'] = int(b.get('minRAMPerGPU') or b.get('minMemoryInGb'))
                if b.get('minVCPUPerGPU') or b.get('minVcpuCount'): gc['minVcpuCountPerGpu'] = int(b.get('minVCPUPerGPU') or b.get('minVcpuCount'))
                if b.get('allowedCudaVersions'): gc['allowedCudaVersions'] = list(b['allowedCudaVersions'])
                x['gpu'] = gc
            out.append(x)
    return out   # dropped (no v2 field): computeType, gpuTypePriority, dataCenterPriority, minDownloadMbps, supportPublicIp, volumeInGb=0
def legacy_pod(p):
    """v2 Pod -> the v1-style keys the rest of this code reads (id, name, desiredStatus, costPerHr, runtime.uptimeInSeconds)."""
    if not isinstance(p, dict) or 'id' not in p: return p
    st = str(p.get('status') or '').upper(); rt = p.get('runtime') or None
    d = dict(p); d.update({'desiredStatus': 'RUNNING' if st in ('PROVISIONING', 'STARTING', 'RUNNING') else st, 'costPerHr': p.get('cost'),
                           'adjustedCostPerHr': p.get('cost'), 'runtime': ({'uptimeInSeconds': rt.get('uptime'), 'ports': rt.get('ports')} if rt else None)})
    return d
def legacy_volume(v): d = dict(v); d.setdefault('dataCenterId', v.get('dataCenter')); return d
def legacy_stock(cat):
    """v2 /catalog/gpus?include=AVAILABILITY&product=POD -> {dataCenterId: {gpuTypeId: 'High'|'Medium'|'Low'}} (old GraphQL stockStatus shape)."""
    out = {}
    for g in (cat or {}).get('gpus') or []:
        for dc in g.get('dataCenters') or []:
            if dc.get('availability') and dc['availability'] != 'NONE': out.setdefault(dc['id'], {})[g['id']] = dc['availability'].title()
    return out
# ---- end v2 shim ----
RUN_ID = os.environ.get('RP_RUN_ID') or __import__('uuid').uuid4().hex[:10]   # step 2b: per-run tag in pod env; dedup only touches pods with this tag
def _same_run(pods, name, err=RuntimeError):
    """step 2c: same-name pods of THIS run. A same-name pod with no env in the list can't be told apart -> stop loud, never touch it."""
    same = [p for p in pods if p.get('name') == name]
    for p in same:
        if p.get('env') is None:
            raise err(f"DEDUP_NO_ENV: pod {p.get('id')} named {name} has no env in list; can't tell if it's ours, not creating a possible duplicate")
    return [p for p in same if (p.get('env') or {}).get('RP_RUN_ID') == RUN_ID]
REST = V2   # was rest v1
GQL = 'https://api.runpod.io/graphql'   # kept ONLY for balance(): v2 has no balance endpoint


def _key() -> str:
    k = os.environ.get('RUNPOD_API_KEY', '').strip()
    if not k:
        raise ApprovalError('RUNPOD_API_KEY is not set in the environment (it is never read from files)')
    return k


class RunPodClient:
    """Minimal REST v2 client (+ GraphQL for balance only). `transport(method, url, headers, body) -> (status, json)` is injectable
    for tests; the default uses httpx. The key lives only in this object's request headers."""

    def __init__(self, transport: Callable | None = None, key: str | None = None):
        self._key = key
        self._t = transport or self._httpx

    @staticmethod
    def _httpx(method, url, headers, body):
        import httpx
        r = httpx.request(method, url, headers=headers, json=body, timeout=60)
        try:
            data = r.json()
        except Exception:
            data = {'text': r.text[:500]}
        return r.status_code, data

    def _h(self):
        return {'Authorization': f'Bearer {self._key or _key()}', 'Content-Type': 'application/json'}

    def call(self, method: str, path: str, body: dict | None = None):
        st, data = self._t(method, REST + path, self._h(), body)
        if st >= 400:
            msg = (data.get('detail') or data.get('title') or data.get('error') or data.get('message') or data) if isinstance(data, dict) else data
            raise ProviderError(f'RunPod {method} {path} -> {st}: {str(msg)[:300]}')
        return data

    def gql(self, query: str, variables: dict | None = None):
        st, data = self._t('POST', GQL, self._h(), {'query': query, 'variables': variables or {}})
        if st >= 400 or (isinstance(data, dict) and data.get('errors')):
            raise ProviderError(f'RunPod GraphQL -> {st}: {str(data)[:300]}')
        return data.get('data', {})

    # --- reads (free) ---
    def balance(self) -> float | None:
        d = self.gql('query { myself { clientBalance currentSpendPerHr } }')
        return float(((d or {}).get('myself') or {}).get('clientBalance') or 0)

    def gpu_stock(self) -> dict[str, dict[str, str]]:
        """{dataCenterId: {gpuTypeId: stockStatus}} (best effort; same data runpodctl datacenter list shows)."""
        return legacy_stock(self.call('GET', '/catalog/gpus?include=AVAILABILITY&product=POD'))   # v2 catalog

    def get_pod(self, pod_id: str) -> dict:
        return legacy_pod(self.call('GET', f'/pods/{pod_id}'))

    def list_volumes(self) -> list:
        return [legacy_volume(v) for v in (self.call('GET', '/network-volumes').get('networkVolumes') or [])]

    # --- writes (billable / destructive): only reachable through RunPodPodProvider gates ---
    def create_pod(self, body: dict) -> dict:
        return self._create_pod_v2(body)   # v2 takes any CUDA major.minor

    def _create_pod_gql(self, body: dict) -> dict:
        import base64 as _b
        sc = body.get('dockerStartCmd') or []
        script = sc[-1] if sc else ''
        args = f"bash -c 'echo {_b.b64encode(script.encode()).decode()} | base64 -d > /tmp/jboot.sh && exec bash /tmp/jboot.sh'" if script else ''
        last = None
        for gid in body['gpuTypeIds']:
            inp = {'cloudType': body.get('cloudType', 'SECURE'), 'gpuCount': 1, 'gpuTypeId': gid, 'name': body['name'],
                   'imageName': body['imageName'], 'containerDiskInGb': body['containerDiskInGb'], 'volumeInGb': 0,
                   'minVcpuCount': body.get('minVCPUPerGPU', 8), 'minMemoryInGb': body.get('minRAMPerGPU', 48),
                   'ports': ','.join(body.get('ports') or []), 'dockerArgs': args, 'allowedCudaVersions': body['allowedCudaVersions'],
                   'env': [{'key': k, 'value': str(v)} for k, v in (body.get('env') or {}).items()]}
            if body.get('containerRegistryAuthId'):
                inp['containerRegistryAuthId'] = body['containerRegistryAuthId']
            if body.get('networkVolumeId'):
                inp['networkVolumeId'] = body['networkVolumeId']; inp['volumeMountPath'] = body.get('volumeMountPath', '/workspace')
            if body.get('dataCenterIds'):
                inp['dataCenterId'] = body['dataCenterIds'][0]
            if body.get('minDownloadMbps'):
                inp['minDownload'] = int(body['minDownloadMbps'])
            try:
                d = self.gql('mutation($i: PodFindAndDeployOnDemandInput) { podFindAndDeployOnDemand(input: $i) { id costPerHr machineId } }', {'i': inp})
                pod = (d or {}).get('podFindAndDeployOnDemand')
                if pod and pod.get('id'):
                    return pod
                last = d
            except Exception as e:
                last = e
        raise ProviderError(f'RunPod GraphQL deploy: There are no instances currently available ({str(last)[:200]})')

    def _create_pod_v2(self, body: dict) -> dict:
        """v1 body -> v2 CreatePodRequest; one single-GPU create per gpuTypeId in v1 order. Returns v1-shaped pod."""
        last = None
        for i, vb in enumerate([dict(vb, env={**(vb.get('env') or {}), 'RP_RUN_ID': RUN_ID}) for vb in v2_bodies(body)]):
            if i:   # dedup (Oct 29): a timed-out create may have landed; reuse it instead of making a 2nd pod
                hit = self._dedup(body['name'], last)
                if hit: return hit
            try:
                return legacy_pod(self.call('POST', '/pods', vb))
            except Exception as e:   # ProviderError or a transport timeout
                last = e
        hit = self._dedup(body['name'], last)
        if hit: return hit
        raise ProviderError(f'RunPod v2 create failed for every GPU type ({str(last)[:200]})')

    def _dedup(self, name: str, last) -> dict | None:
        """List pods with this exact name. Reuse the first live one, terminate extras and stopped leftovers. Can't list -> stop (never risk a duplicate)."""
        try:
            allp = self.call('GET', '/pods?limit=1000').get('pods') or []
        except Exception as e:
            raise ProviderError(f'RunPod v2 dedup: could not list pods before retry, not creating a possible duplicate ({str(e)[:150]}; last create error: {str(last)[:150]})')
        ps = _same_run(allp, name, ProviderError)   # DEDUP_NO_ENV raises here, outside the list-failure handler
        live = [p for p in ps if str(p.get('status', '')).upper() in ('PROVISIONING', 'STARTING', 'RUNNING')]
        for p in ps:
            if not live or p is not live[0]:
                self.call('DELETE', f"/pods/{p['id']}")
        return legacy_pod(live[0]) if live else None

    def terminate_pod(self, pod_id: str) -> dict:
        return self.call('DELETE', f'/pods/{pod_id}')

    def create_volume(self, name: str, size_gb: int, dc: str) -> dict:
        return legacy_volume(self.call('POST', '/network-volumes', {'name': name, 'size': int(size_gb), 'dataCenter': dc}))

    def delete_volume(self, vol_id: str) -> dict:
        return self.call('DELETE', f'/network-volumes/{vol_id}')


class Watchdog:
    """Polls spend/wall/idle and terminates the pod at the first breach. Clock is injectable for tests."""

    def __init__(self, terminate: Callable[[], None], usd_per_hr: float, max_usd: float, max_seconds: float,
                 idle_seconds: float = 900, safety: float = 0.92, clock: Callable[[], float] = time.time,
                 log: Callable[[str], None] = print, already_spent_usd: float = 0.0):
        self.terminate, self.rate, self.max_usd, self.max_seconds = terminate, float(usd_per_hr), float(max_usd), float(max_seconds)
        self.idle_seconds, self.safety, self.clock, self.log = float(idle_seconds), float(safety), clock, log
        self.base_spent = float(already_spent_usd)
        self.t0 = clock(); self.last_beat = self.t0
        self.reason: str | None = None
        self._stop = threading.Event(); self._th: threading.Thread | None = None
        self._lock = threading.Lock()

    def heartbeat(self):
        self.last_beat = self.clock()

    def elapsed(self) -> float:
        return self.clock() - self.t0

    def spent(self) -> float:
        return self.base_spent + self.elapsed() / 3600 * self.rate

    def remaining_usd(self) -> float:
        return self.max_usd * self.safety - self.spent()

    def check(self) -> str | None:
        if self.spent() >= self.max_usd * self.safety:
            return f'budget: ${self.spent():.3f} >= {self.safety:.0%} of ${self.max_usd:.2f} cap'
        if self.elapsed() >= self.max_seconds:
            return f'wall clock: {self.elapsed():.0f}s >= {self.max_seconds:.0f}s'
        if self.clock() - self.last_beat >= self.idle_seconds:
            return f'idle: no heartbeat for {self.clock() - self.last_beat:.0f}s'
        return None

    def fire(self, reason: str):
        with self._lock:
            if self.reason:
                return
            self.reason = reason
        self.log(f'[watchdog] TERMINATING: {reason}')
        try:
            self.terminate()
        except Exception as e:                # retry once; a failed terminate is the worst case, shout about it
            self.log(f'[watchdog] terminate failed ({e}); retrying')
            time.sleep(2)
            self.terminate()

    def tick(self) -> str | None:
        r = self.check()
        if r:
            self.fire(r)
        return r

    def start(self, interval: float = 10.0):
        def loop():
            while not self._stop.wait(interval):
                if self.tick():
                    return
        self._th = threading.Thread(target=loop, daemon=True, name='runpod-watchdog'); self._th.start()

    def stop(self):
        self._stop.set()


class RunPodPodProvider(Provider):
    name = 'runpod_pod'
    live_capable = True

    def __init__(self, cfg: dict, client: RunPodClient | None = None, clock: Callable[[], float] = time.time):
        super().__init__(cfg)
        self.client = client or RunPodClient()
        self.clock = clock

    # ---------------------------------------------------------------- planning (free)
    def pod_body(self, name: str, gpu_ids: list[str], image: str, volume_id: str | None, dc_ids: list[str] | None,
                 env: dict | None = None, start_cmd: list[str] | None = None, container_disk_gb: int = 30,
                 ports: list[str] | None = None, max_seconds: int | None = None, compute: str = 'GPU',
                 min_download_mbps: int | None = None, cuda_versions: list[str] | None = None) -> dict:
        env = dict(env or {})
        if any(k.upper() in ('RUNPOD_API_KEY', 'HF_TOKEN') or 'KEY' in k.upper() or 'TOKEN' in k.upper() for k in env):
            raise ProviderError('refusing to put credentials into pod env (pods get their own scoped key)')
        if max_seconds:
            env['JUTSU_MAX_SECONDS'] = str(int(max_seconds))
        body = {'name': name, 'imageName': image, 'cloudType': self.cfg.get('cloud_type', 'SECURE'), 'computeType': compute,
                'containerDiskInGb': int(container_disk_gb), 'env': env, 'ports': ports or ['8188/http', '22/tcp'],
                'volumeInGb': 0}                     # no pod volume (its 20 GB default would persist + bill); container disk only
        if min_download_mbps:
            body['minDownloadMbps'] = int(min_download_mbps)
        if compute == 'GPU':
            body.update({'gpuTypeIds': gpu_ids, 'gpuCount': 1, 'gpuTypePriority': 'custom',
                         'minRAMPerGPU': int(self.cfg.get('min_ram_gb', 48)), 'minVCPUPerGPU': int(self.cfg.get('min_vcpu', 8))})
            if os.environ.get('JUTSU_CUDA_ANY') != '1':   # REST enum tops out at "13.0": hosts advertising 13.1/13.2 are excluded by it.
                body['allowedCudaVersions'] = cuda_versions or (os.environ.get('JUTSU_CUDA_LIST') or '13.0').split(',')   # JUTSU_CUDA_ANY=1 drops the filter; boot_cmd then refuses <13 hosts
        else:
            body.update({'cpuFlavorIds': list(self.cfg.get('cpu_flavors', ['cpu5c', 'cpu3c'])), 'vcpuCount': int(self.cfg.get('cpu_vcpus', 8))})
        if volume_id:
            body['networkVolumeId'] = volume_id; body['volumeMountPath'] = '/workspace'
        if dc_ids:
            body['dataCenterIds'] = dc_ids; body['dataCenterPriority'] = 'custom'
        if start_cmd:
            body['dockerStartCmd'] = start_cmd
        if os.environ.get('FP_REGISTRY_AUTH_ID'):   # bench v2: private GHCR image
            body['containerRegistryAuthId'] = os.environ['FP_REGISTRY_AUTH_ID']
        return body

    def max_seconds_for(self, usd_per_hr: float, max_usd: float, safety: float | None = None) -> int:
        s = float(safety if safety is not None else self.cfg.get('watchdog_safety', 0.92))
        return int(max_usd * s / usd_per_hr * 3600)

    # ---------------------------------------------------------------- gates
    def check_gates(self, approval: Approval | None, worst_case_usd: float) -> None:
        cap = float(self.cfg.get('hard_cap_usd', 5.0))
        if not approval or not approval.confirmed:
            raise ApprovalError('no confirmed approval from the owner for this spend')
        if approval.max_usd > cap:
            raise ApprovalError(f'approval ${approval.max_usd:.2f} exceeds provider hard cap ${cap:.2f}')
        if worst_case_usd > approval.max_usd + 1e-9:
            raise BudgetExceeded(f'worst case ${worst_case_usd:.2f} exceeds approved ${approval.max_usd:.2f}')
        _key()
        bal = self.client.balance()
        if bal is not None and bal < worst_case_usd + float(self.cfg.get('min_balance_buffer_usd', 1.0)):
            raise BudgetExceeded(f'RunPod balance ${bal:.2f} too low for worst case ${worst_case_usd:.2f} + buffer')

    # ---------------------------------------------------------------- live session
    def session(self, body: dict, approval: Approval, spent_so_far: float = 0.0, log=print, interval: float = 10.0,
                idle_seconds: float | None = None) -> 'PodSession':
        return PodSession(self, body, approval, spent_so_far, log, interval,
                          idle_seconds if idle_seconds is not None else float(self.cfg.get('idle_seconds', 900)))

    def attach_session(self, pod_id: str, approval: Approval, spent_so_far: float = 0.0, rate: float | None = None,
                       log=print, interval: float = 10.0, idle_seconds: float | None = None) -> 'PodSession':
        """Reuse an already-running pod (no create). Watchdog + terminate-on-exit still apply."""
        pod = self.client.get_pod(pod_id)
        if str(pod.get('desiredStatus', '')).upper() != 'RUNNING':
            raise ProviderError(f'pod {pod_id} is not RUNNING ({pod.get("desiredStatus")})')
        return PodSession(self, body={}, approval=approval, spent_so_far=spent_so_far, log=log, interval=interval,
                          idle_seconds=idle_seconds if idle_seconds is not None else float(self.cfg.get('idle_seconds', 900)),
                          existing_pod=pod, rate_override=rate)


class PodSession:
    """`with provider.session(body, approval) as s:` -> pod is ALWAYS terminated on exit (normal, error,
    Ctrl-C, SIGTERM, interpreter exit). s.pod, s.watchdog, s.heartbeat(), s.spent()."""

    def __init__(self, prov: RunPodPodProvider, body: dict, approval: Approval, spent_so_far: float, log, interval, idle_seconds,
                 existing_pod: dict | None = None, rate_override: float | None = None):
        self.p, self.body, self.approval, self.log = prov, body, approval, log
        self.spent_so_far, self.interval, self.idle_seconds = spent_so_far, interval, idle_seconds
        self.existing_pod, self.rate_override = existing_pod, rate_override
        self.pod: dict | None = None
        self.watchdog: Watchdog | None = None
        self._terminated = False
        self._old = {}

    def _terminate(self):
        if self._terminated or not self.pod:
            return
        hook, self.before_stop = getattr(self, 'before_stop', None), None
        if hook:                                  # pull partial outputs before the pod disappears (best effort)
            try:
                hook()
            except Exception as e:
                self.log(f'[runpod] before_stop failed: {e}')
        self.p.client.terminate_pod(self.pod['id'])
        self._terminated = True
        self.log(f"[runpod] pod {self.pod['id']} terminated")

    def __enter__(self):
        remaining = self.approval.max_usd - self.spent_so_far
        if remaining <= 0:
            raise BudgetExceeded('approved budget already used up')
        if self.existing_pod is not None:
            self.pod = self.existing_pod
            rate = float(self.rate_override or self.pod.get('costPerHr') or self.pod.get('adjustedCostPerHr') or 0)
            tag = 'attached'
        else:
            self.pod = self.p.client.create_pod(self.body)
            rate = float(self.pod.get('costPerHr') or self.pod.get('adjustedCostPerHr') or 0)
            tag = 'up'
        if rate <= 0:
            rate = max(float(g['usd_per_hr']) for g in self.p.gpus())       # unknown -> assume the most expensive
        safety = float(self.p.cfg.get('watchdog_safety', 0.92))
        max_s = self.p.max_seconds_for(rate, remaining, safety)
        self.watchdog = Watchdog(self._terminate, rate, self.approval.max_usd, max_s, self.idle_seconds, safety,
                                 clock=self.p.clock, log=self.log, already_spent_usd=self.spent_so_far)
        self.watchdog.start(self.interval)
        atexit.register(self._terminate)
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                self._old[sig] = signal.signal(sig, self._on_signal)
            except ValueError:            # not main thread (tests / server): atexit + context exit still apply
                pass
        self.log(f"[runpod] pod {self.pod['id']} {tag} at ${rate:.3f}/hr; hard stop after {max_s}s or ${self.approval.max_usd:.2f} (already ${self.spent_so_far:.2f})")
        return self

    def _on_signal(self, signum, frame):
        if self.watchdog:
            self.watchdog.fire(f'signal {signum}')
        raise KeyboardInterrupt

    def heartbeat(self):
        if self.watchdog:
            self.watchdog.heartbeat()

    def spent(self) -> float:
        return self.watchdog.spent() if self.watchdog else self.spent_so_far

    def alive(self) -> bool:
        return not self._terminated and not (self.watchdog and self.watchdog.reason)

    def __exit__(self, et, ev, tb):
        keep = os.environ.get('FP_KEEP_POD') == '1' and et is None and not (self.watchdog and self.watchdog.reason)
        if keep:                                   # Oct 21 15:40: the owner wants the pod left up after a clean finish (pod-side deadman still enforces the cap)
            if self.watchdog: self.watchdog.stop()
            print(f"[keep] session ended cleanly; pod {self.pod and self.pod.get('id')} LEFT RUNNING (FP_KEEP_POD=1)", flush=True)
        elif self.watchdog:
            self.watchdog.stop()
            if not self.watchdog.reason:
                self.watchdog.fire('session ended' if et is None else f'exception: {et.__name__}')
        else:
            self._terminate()
        for sig, h in self._old.items():
            try:
                signal.signal(sig, h)
            except Exception:
                pass
        try:
            atexit.unregister(self._terminate)
        except Exception:
            pass
        return False
