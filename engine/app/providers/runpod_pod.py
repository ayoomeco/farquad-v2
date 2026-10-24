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

REST = 'https://rest.runpod.io/v1'
GQL = 'https://api.runpod.io/graphql'


def _key() -> str:
    k = os.environ.get('RUNPOD_API_KEY', '').strip()
    if not k:
        raise ApprovalError('RUNPOD_API_KEY is not set in the environment (it is never read from files)')
    return k


class RunPodClient:
    """Minimal REST v1 + GraphQL client. `transport(method, url, headers, body) -> (status, json)` is injectable
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
            msg = data.get('error') or data.get('message') or data if isinstance(data, dict) else data
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
        d = self.gql('query { dataCenters { id name location gpuAvailability { gpuTypeId stockStatus } } }')
        out = {}
        for dc in (d or {}).get('dataCenters') or []:
            out[dc['id']] = {g['gpuTypeId']: g.get('stockStatus') for g in dc.get('gpuAvailability') or []}
        return out

    def get_pod(self, pod_id: str) -> dict:
        return self.call('GET', f'/pods/{pod_id}')

    def list_volumes(self) -> list:
        return self.call('GET', '/networkvolumes')

    # --- writes (billable / destructive): only reachable through RunPodPodProvider gates ---
    def create_pod(self, body: dict) -> dict:
        cu = body.get('allowedCudaVersions') or []
        if any(v not in ('12.8', '12.9', '13.0') for v in cu):   # Oct 20: REST enum stops at 13.0 -> GraphQL deploy for 13.x hosts
            return self._create_pod_gql(body)
        return self.call('POST', '/pods', body)

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

    def terminate_pod(self, pod_id: str) -> dict:
        return self.call('DELETE', f'/pods/{pod_id}')

    def create_volume(self, name: str, size_gb: int, dc: str) -> dict:
        return self.call('POST', '/networkvolumes', {'name': name, 'size': int(size_gb), 'dataCenterId': dc})

    def delete_volume(self, vol_id: str) -> dict:
        return self.call('DELETE', f'/networkvolumes/{vol_id}')


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
