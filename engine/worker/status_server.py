#!/usr/bin/env python3
"""Tiny read-only status endpoint for a Phase-1 pod (stdlib only; port 8001 via the RunPod proxy).

    GET /status              -> JSON: boot stages, model bytes on disk, comfy log size, latest GPU sample
    GET /gpu?since=<epoch>   -> JSON: peak / mean GPU memory + util over samples taken since <epoch>
    GET /log/<name>?from=N   -> text of <status_dir>/<name>.log from byte N (install, apt, dl1, dl2, comfy, boot)
    POST /exec  (header X-Jutsu-Nonce)  body = bash script -> {"job": id}; GET /job/<id> -> {"done", "rc", "out"}
        on-the-fly fixes (missing node, pip dep, restart ComfyUI). Disabled unless JUTSU_FIX_NONCE is set: a random
        one-time value the box runner generates per launch and keeps in memory only (not a RunPod/HF credential).
Never serves the environment, so no secret can leak through it; logs are the boot/ComfyUI stdout only."""
import hmac, json, os, re, subprocess, sys, threading, time, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

S, MODELS = sys.argv[1], sys.argv[2]
PORT = int(sys.argv[3]) if len(sys.argv) > 3 else 8001
NONCE = os.environ.get('JUTSU_FIX_NONCE', '')
JOBS = {}
SAMPLES = []                        # (t, mem_used_mib, mem_total_mib, util_pct)
DU = {'bytes': 0, 't': 0}


def sampler():
    while True:
        try:
            out = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'],
                                 capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
            u, t, g = (float(x) for x in out.split(','))
            SAMPLES.append((time.time(), u, t, g)); del SAMPLES[:-20000]
        except Exception:
            pass
        time.sleep(2)


def du():
    while True:
        n = 0
        for d, _, fs in os.walk(MODELS):
            for f in fs:
                try:
                    n += os.lstat(os.path.join(d, f)).st_blocks * 512
                except OSError:
                    pass
        DU.update(bytes=n, t=time.time()); time.sleep(10)


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype='application/json'):
        b = body if isinstance(body, bytes) else body.encode()
        self.send_response(code); self.send_header('Content-Type', ctype); self.send_header('Content-Length', str(len(b)))
        self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        u = urlparse(self.path); q = parse_qs(u.query)
        if u.path == '/status':
            st = open(os.path.join(S, 'stages.txt')).read().splitlines() if os.path.exists(os.path.join(S, 'stages.txt')) else []
            logs = {f[:-4]: os.path.getsize(os.path.join(S, f)) for f in os.listdir(S) if f.endswith('.log')}
            free = os.statvfs(MODELS); g = SAMPLES[-1] if SAMPLES else None
            self._send(200, json.dumps({'t': time.time(), 'stages': st, 'models_bytes': DU['bytes'], 'log_bytes': logs,
                                        'disk_free_gb': round(free.f_bavail * free.f_frsize / 1e9, 1),
                                        'gpu': {'mem_used_mib': g[1], 'mem_total_mib': g[2], 'util': g[3]} if g else None}))
        elif u.path == '/gpu':
            since = float((q.get('since') or ['0'])[0]); xs = [s for s in SAMPLES if s[0] >= since]
            self._send(200, json.dumps({'n': len(xs), 'peak_mib': max((s[1] for s in xs), default=None),
                                        'total_mib': xs[-1][2] if xs else None,
                                        'mean_util': round(sum(s[3] for s in xs) / len(xs), 1) if xs else None}))
        elif u.path.startswith('/log/') and re.fullmatch(r'[a-z0-9_]+', u.path[5:]):
            p = os.path.join(S, u.path[5:] + '.log')
            if not os.path.exists(p):
                return self._send(404, 'no such log', 'text/plain')
            with open(p, 'rb') as f:
                f.seek(int((q.get('from') or ['0'])[0])); self._send(200, f.read(8 * 1024 * 1024), 'text/plain; charset=utf-8')
        elif u.path.startswith('/job/') and u.path[5:] in JOBS:
            self._send(200, json.dumps(JOBS[u.path[5:]]))
        else:
            self._send(404, 'not found', 'text/plain')

    def do_POST(self):
        if urlparse(self.path).path != '/exec' or not NONCE or not hmac.compare_digest(self.headers.get('X-Jutsu-Nonce', ''), NONCE):
            return self._send(403, 'forbidden', 'text/plain')
        script = self.rfile.read(int(self.headers.get('Content-Length') or 0)).decode()
        jid = uuid.uuid4().hex[:12]; JOBS[jid] = {'done': False, 'rc': None, 'out': ''}
        env = {k: v for k, v in os.environ.items() if k != 'JUTSU_FIX_NONCE'}

        def go():
            try:
                r = subprocess.run(['bash', '-c', script], capture_output=True, text=True, timeout=3600, env=env, cwd='/opt/jutsu-src' if os.path.isdir('/opt/jutsu-src') else None)
                JOBS[jid].update(rc=r.returncode, out=(r.stdout + r.stderr)[-60000:])
            except Exception as e:
                JOBS[jid].update(rc=-1, out=f'{type(e).__name__}: {e}')
            JOBS[jid]['done'] = True
        threading.Thread(target=go, daemon=True).start()
        self._send(200, json.dumps({'job': jid}))


if __name__ == '__main__':
    threading.Thread(target=sampler, daemon=True).start(); threading.Thread(target=du, daemon=True).start()
    ThreadingHTTPServer(('0.0.0.0', PORT), H).serve_forever()
