"""Live ComfyUI progress for the bake-off runner (websocket instead of mid-sample /history polling).

RunTrace     event sink for one prompt: per-node wall time, per-sampler-step timestamps, cached nodes, done/error.
ComfyWS      background websocket reader (websocket-client) that routes events to the active RunTrace; reconnects.
DriftGuard   warm s/step baseline per config key; flags a run whose live steps drift > factor x baseline (ComfyUI is then
             restarted through the pod fix channel and the run requeued once: Phase 1 saw DynamicVRAM staging slow down).
rank()       ranks configs by WARM sampler s/step (first step of each bar dropped, cold runs only used when no warm one).
dedup_bars() ComfyUI prints the final tqdm line twice; drop the repeat (same total + elapsed) so sampling_s isn't doubled.
Everything here is offline-testable: feed() takes plain dicts + timestamps."""
from __future__ import annotations
import json, re, statistics, threading, time, uuid

SAMPLER_HINT = ('Sampler', 'KSampler')


def sampler_nodes(graph: dict) -> set[str]:
    return {k for k, n in graph.items() if any(h in n.get('class_type', '') for h in SAMPLER_HINT)}


class RunTrace:
    def __init__(self, prompt_id: str | None, graph: dict | None = None, clock=time.time):
        self.pid, self.clock = prompt_id, clock
        self.classes = {k: n.get('class_type') for k, n in (graph or {}).items()}
        self.samplers = sampler_nodes(graph or {})
        self.t_start = self.t_done = None
        self.node_s: dict[str, float] = {}
        self.cached: list[str] = []
        self.cur: tuple[str, float] | None = None
        self.bars: list[dict] = []                     # [{'node', 'max', 'start', 'ticks': [(value, t)]}]
        self.error: dict | None = None
        self.status: str | None = None
        self.done = threading.Event()
        self.last_event: float | None = None            # wall time of the last ws event for this prompt (stall detector)

    # ---- event intake
    def feed(self, msg: dict, t: float | None = None):
        t = self.clock() if t is None else t
        typ, d = msg.get('type'), msg.get('data') or {}
        if self.pid and d.get('prompt_id') not in (None, self.pid):
            return
        self.last_event = t
        if typ == 'execution_start':
            self.t_start = t
        elif typ == 'execution_cached':
            self.cached += [str(x) for x in d.get('nodes') or []]
        elif typ == 'executing':
            self._close(t)
            node = d.get('node')
            if node is None:                            # legacy end-of-prompt marker
                self._finish(t, self.status or 'success')
            else:
                self.cur = (str(node), t)
        elif typ == 'progress':
            node, v, mx = str(d.get('node')), int(d.get('value', 0)), int(d.get('max', 0))
            b = self.bars[-1] if self.bars else None
            if not b or b['node'] != node or b['max'] != mx or v <= (b['ticks'][-1][0] if b['ticks'] else 0):
                start = self.cur[1] if (self.cur and self.cur[0] == node and not self._bars_for(node)) else \
                    (b['ticks'][-1][1] if b and b['ticks'] else t)
                b = {'node': node, 'max': mx, 'start': start, 'ticks': []}; self.bars.append(b)
            b['ticks'].append((v, t))
        elif typ == 'execution_success':
            self._close(t); self._finish(t, 'success')
        elif typ == 'execution_error':
            self._close(t); self.error = d; self._finish(t, 'error')
        elif typ == 'execution_interrupted':
            self._close(t); self._finish(t, 'interrupted')

    def stalled(self, stall_s: float = 600.0, now: float | None = None) -> bool:
        """True only when NO progress/executing event arrived for stall_s (default 10 min). Slow-but-moving is never a stall."""
        ref = self.last_event or self.t_start
        return ref is not None and ((self.clock() if now is None else now) - ref) > stall_s

    def _bars_for(self, node):
        return [b for b in self.bars if b['node'] == node]

    def _close(self, t):
        if self.cur:
            n, t0 = self.cur
            self.node_s[n] = round(self.node_s.get(n, 0.0) + (t - t0), 3); self.cur = None

    def _finish(self, t, status):
        self.t_done, self.status = t, status; self.done.set()

    # ---- derived numbers
    def sampler_bars(self) -> list[dict]:
        out = []
        for b in self.bars:
            if self.samplers and b['node'] not in self.samplers:
                continue
            ts = [b['start']] + [t for _, t in b['ticks']]
            steps = [round(ts[i + 1] - ts[i], 3) for i in range(len(ts) - 1)]
            out.append({'node': b['node'], 'max': b['max'], 'done': b['ticks'][-1][0] if b['ticks'] else 0, 'step_s': steps})
        return out

    def step_times(self, warm_only: bool = True) -> list[float]:
        return [s for b in self.sampler_bars() for s in (b['step_s'][1:] if warm_only else b['step_s'])]

    def warm_s_per_step(self) -> float | None:
        w = self.step_times(True)
        return round(statistics.median(w), 2) if w else None

    def live_s_per_step(self, last: int = 2) -> float | None:
        w = self.step_times(True)
        return round(statistics.median(w[-last:]), 2) if len(w) >= last else None

    def summary(self) -> dict:
        bars = self.sampler_bars()
        top = sorted(self.node_s.items(), key=lambda kv: -kv[1])[:8]
        return {'status': self.status, 'exec_s': round(self.t_done - self.t_start, 1) if self.t_done and self.t_start else None,
                'warm_s_per_step': self.warm_s_per_step(),
                'first_step_s': [b['step_s'][0] for b in bars if b['step_s']],
                'sampling_s': round(sum(sum(b['step_s']) for b in bars), 1), 'steps': sum(len(b['step_s']) for b in bars),
                'bars': [{'node': b['node'], 'max': b['max'], 'step_s': b['step_s']} for b in bars],
                'node_s': {f'{k}:{self.classes.get(k, "?")}': v for k, v in top}, 'cached': self.cached}


class ComfyWS:
    """ws(s)://<comfy>/ws?clientId=<id>. Text frames are JSON events; binary frames (previews) are ignored."""

    def __init__(self, base: str, headers: dict | None = None, connect=None, log=print):
        self.client_id = 'jutsu-' + uuid.uuid4().hex[:12]
        self.url = re.sub(r'^http', 'ws', base.rstrip('/')) + f'/ws?clientId={self.client_id}'
        self.headers = headers or {}
        self.log, self._connect = log, connect
        self.trace: RunTrace | None = None
        self.connected = threading.Event()
        self._stop = threading.Event(); self._th = None; self.last_msg = 0.0

    def _conn(self):
        if self._connect:
            return self._connect(self.url, self.headers)
        import websocket                                   # websocket-client
        return websocket.create_connection(self.url, header=[f'{k}: {v}' for k, v in self.headers.items()], timeout=30)

    def start(self):
        self._th = threading.Thread(target=self._loop, daemon=True); self._th.start(); return self

    def stop(self):
        self._stop.set()

    def _loop(self):
        fails = 0
        while not self._stop.is_set():
            try:
                ws = self._conn(); self.connected.set(); fails = 0
                while not self._stop.is_set():
                    try:
                        m = ws.recv()
                    except Exception as e:
                        if 'timed out' in str(e).lower():
                            continue
                        raise
                    if isinstance(m, (bytes, bytearray)) or not m:
                        continue
                    self.last_msg = time.time()
                    try:
                        msg = json.loads(m)
                    except ValueError:
                        continue
                    tr = self.trace
                    if tr is not None:
                        tr.feed(msg, self.last_msg)
            except Exception as e:
                self.connected.clear(); fails += 1
                if fails in (1, 5, 20):
                    self.log(f'[ws] disconnected ({type(e).__name__}: {str(e)[:120]}); retrying')
                self._stop.wait(min(30, 2 * fails))


class DriftGuard:
    """Baseline = best warm s/step seen for a config key. A run drifts when its live median of the last `last`
    warm steps exceeds factor x baseline (min_abs keeps tiny absolute jitter from tripping it)."""

    def __init__(self, factor: float = 1.5, last: int = 2, min_abs: float = 3.0):
        self.factor, self.last, self.min_abs = factor, last, min_abs
        self.base: dict[str, float] = {}

    def record(self, key: str, warm: float | None):
        if warm:
            self.base[key] = min(self.base.get(key, warm), warm)

    def drifting(self, key: str, trace: RunTrace) -> float | None:
        b, live = self.base.get(key), trace.live_s_per_step(self.last)
        if b and live and live > self.factor * b and live - b > self.min_abs:
            return live
        return None


def drift_key(run: dict) -> str:
    p = run.get('params') or {}
    return '|'.join(str(p.get(k)) for k in ('unet', 'accel', 'attention', 'veda', 'vsa', 'megapixels', 'seconds')) + '|' + str(run.get('engine'))


def model_signature(graph: dict) -> tuple:
    """Weights a run needs resident (loader file names). A change vs the previous run => the run is COLD."""
    keys = ('unet_name', 'lora_name', 'pdd_file', 'gate_file', 'hyperflow_file', 'ckpt_name', 'vae_name', 'clip_name', 'predictor')
    return tuple(sorted({(k, str(n['inputs'][k])) for n in graph.values() for k in keys if isinstance(n.get('inputs', {}).get(k), str)}))


def rank(results: list[dict], key: str = 'rung') -> list[dict]:
    """Per config: median warm s/step over WARM runs (cold ones only if nothing warm), seeds, mean wall."""
    groups: dict[str, list[dict]] = {}
    for r in results:
        if r.get('status') == 'success' and r.get(key):
            groups.setdefault(r[key], []).append(r)
    out = []
    for k, rs in groups.items():
        warm = [r for r in rs if not r.get('cold')] or rs
        vals = [r.get('warm_s_per_step') or r.get('s_per_step') for r in warm]
        vals = [v for v in vals if v]
        out.append({key: k, 'runs': len(rs), 'warm_runs': len([r for r in rs if not r.get('cold')]),
                    'seeds': sorted({r.get('seed') for r in rs if r.get('seed') is not None}),
                    'warm_s_per_step': round(statistics.median(vals), 2) if vals else None,
                    'wall_s': round(statistics.mean(r['wall_s'] for r in warm if r.get('wall_s')), 1) if any(r.get('wall_s') for r in warm) else None,
                    'enough_seeds': len({r.get('seed') for r in rs}) >= 2})
    return sorted(out, key=lambda x: (x['wall_s'] is None, x['wall_s'] or 0))


TQDM = re.compile(r'(\d+)/(\d+) \[([\d:]+)<(?:[\d:?]+),\s*([\d.]+)(s/it|it/s)\]')


def dedup_bars(text: str) -> list[dict]:
    """Final reading of every sampler bar in a log chunk; ComfyUI prints the closing line twice (same total + elapsed)."""
    out, last = [], None
    for m in TQDM.finditer(text.replace('\r', '\n')):
        n, tot, el, v, unit = int(m[1]), int(m[2]), m[3], float(m[4]), m[5]
        if n != tot or tot <= 0:
            continue
        rate = round(v if unit == 's/it' else 1 / v, 2)
        if last == (tot, el):
            out[-1]['s_per_step'] = rate                    # keep the later reading of the same bar
            continue
        out.append({'steps': tot, 's_per_step': rate, 'elapsed': el}); last = (tot, el)
    return out
