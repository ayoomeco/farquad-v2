#!/usr/bin/env python3
"""Run WanGP (MMGP v4) jobs headless on the pod and time them like the ComfyUI runner does.

    /opt/wangp/venv/bin/python worker/wangp_driver.py --jobs jobs.json --out /opt/jutsu/ComfyUI/output/wangp \
        --profile 4 --attention sage2

jobs.json = [{"id", "settings": {WanGP settings: model_type, prompt, image_refs, video_guide, ...}}, ...]
One WanGP session for all jobs (models stay loaded, like ComfyUI), so job 1 is the cold run (model download + load) and
later jobs are warm. Per job it writes one JSON line to <out>/results.jsonl: wall_s, per-step times from progress events
(warm s/step excludes the first step), the output file copied to <out>/<id>.mp4, errors. Falls back to
`wgp.py --process <settings.json>` per job if the Python API cannot be imported (wall time only).
Uses the WanGP API (docs/API.md: shared.api.init / get_default_settings / submit_task). Stdlib only besides WanGP."""
import argparse, json, os, pathlib, shutil, subprocess, sys, time


def log(*a):
    print('[wangp]', *a, flush=True)


def run_api(root, jobs, out, cli_args, results):
    sys.path.insert(0, str(root)); os.chdir(root)
    from shared.api import init                                  # noqa: E402  (WanGP)
    t0 = time.time()
    session = init(root=pathlib.Path(root), cli_args=cli_args)
    log(f'session ready in {time.time() - t0:.0f}s with {cli_args}')
    for j in jobs:
        rec = {'id': j['id'], 'engine': 'wangp', 'cli': cli_args, 'status': 'error'}
        try:
            st = dict(session.get_default_settings(j['settings']['model_type']) or {})
            st.update(j['settings'])
            (out / f"{j['id']}.settings.json").write_text(json.dumps(st, indent=1, default=str))
            steps, t_start = [], time.time()
            job = session.submit_task(st)
            last = None
            for ev in job.events.iter(timeout=0.5):
                if ev.kind == 'progress':
                    p = ev.data
                    cur = getattr(p, 'current_step', None)
                    if cur is not None and cur != last:
                        steps.append((time.time(), cur, getattr(p, 'total_steps', None), str(getattr(p, 'phase', ''))))
                        last = cur
                elif ev.kind == 'stream':
                    print(f'  {ev.data.text}'.rstrip(), flush=True)
            res = job.result()
            rec['wall_s'] = round(time.time() - t_start, 1)
            den = [s for s in steps if 'denois' in s[3].lower()] or steps
            dts = [b[0] - a[0] for a, b in zip(den, den[1:])]
            rec['steps_seen'] = len(den)
            rec['s_per_step'] = round(sum(dts) / len(dts), 2) if dts else None
            rec['warm_s_per_step'] = round(sum(dts[1:]) / len(dts[1:]), 2) if len(dts) > 1 else rec['s_per_step']
            if res.success:
                files = [f for f in (res.generated_files or []) if str(f).endswith(('.mp4', '.mkv', '.mov'))]
                if files:
                    dst = out / f"{j['id']}.mp4"; shutil.copy(files[-1], dst); rec['outputs'] = [str(dst)]
                rec['status'] = 'success'
            else:
                rec['error'] = '; '.join(str(getattr(e, 'message', e)) for e in (res.errors or []))[:2000]
        except Exception as e:                                      # one bad job must not kill the rest
            rec['error'] = f'{type(e).__name__}: {e}'[:2000]
        results(rec)


def run_cli(root, jobs, out, cli_args, results):
    for j in jobs:
        sp = out / f"{j['id']}.settings.json"; sp.write_text(json.dumps(j['settings'], indent=1))
        od = out / j['id']; od.mkdir(exist_ok=True)
        t = time.time()
        r = subprocess.run([sys.executable, 'wgp.py', '--process', str(sp), '--output-dir', str(od), *cli_args], cwd=root)
        rec = {'id': j['id'], 'engine': 'wangp', 'cli': cli_args, 'mode': 'process_cli', 'wall_s': round(time.time() - t, 1),
               'status': 'success' if r.returncode == 0 else 'error', 'rc': r.returncode}
        vids = sorted(od.glob('*.mp4'))
        if vids:
            dst = out / f"{j['id']}.mp4"; shutil.copy(vids[-1], dst); rec['outputs'] = [str(dst)]
        results(rec)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/opt/wangp/src'); ap.add_argument('--jobs', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--profile', default='4'); ap.add_argument('--attention', default='sage2')
    ap.add_argument('--mode', choices=['api', 'cli'], default='api')
    a = ap.parse_args(argv)
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    jobs = json.loads(pathlib.Path(a.jobs).read_text())
    att = a.attention if a.attention != 'sage2' or pathlib.Path(a.root).parent.joinpath('sage.ok').exists() else 'sdpa'
    cli_args = ['--attention', att, '--profile', str(a.profile)]

    def results(rec):
        log(json.dumps(rec))
        with open(out / 'results.jsonl', 'a') as f:
            f.write(json.dumps(rec) + '\n')
    if a.mode == 'api':
        try:
            return run_api(a.root, jobs, out, cli_args, results)
        except ImportError as e:
            log(f'API import failed ({e}); falling back to wgp.py --process')
    run_cli(a.root, jobs, out, cli_args, results)


if __name__ == '__main__':
    main()
