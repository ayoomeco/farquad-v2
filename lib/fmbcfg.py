"""Fire Marshal Bill config. Every path, cap, rate and budget comes from here + the job JSON. Nothing is read from outside FMB_HOME / FMB_DATA
unless you point an env var at it.
  FMB_HOME  code root (default: the folder above lib/)
  FMB_DATA  work root: runs, ledgers, prepared refs/audio, outputs (default ~/fmb-data)
  FMB_JOB   job JSON (default jobs/example_job.json)
Keys by NAME only: RUNPOD_API_KEY (paid runs), DROPBOX_TOKEN (uploads), HF_TOKEN (optional). Never printed."""
import os, json, glob, time, datetime, pathlib
HOME = pathlib.Path(os.environ.get('FMB_HOME', pathlib.Path(__file__).resolve().parent.parent))
DATA = pathlib.Path(os.environ.get('FMB_DATA', str(pathlib.Path.home() / 'fmb-data')))
os.environ.setdefault('FMB_DATA', str(DATA)); os.environ.setdefault('FMB_HOME', str(HOME))
RUN_DIR = DATA / 'run'
CV = pathlib.Path(os.environ.get('FMB_CV', str(DATA / 'cv')))
def job_path(): return pathlib.Path(os.environ.get('FMB_JOB', str(HOME / 'jobs' / 'example_job.json')))
def job(): return json.load(open(job_path()))
def P(p):
    """resolve a job path: absolute stays, '~' expands, relative is relative to the job JSON's folder."""
    p = os.path.expanduser(os.path.expandvars(str(p))); return p if os.path.isabs(p) else str((job_path().parent / p).resolve())
def ledger_roots():
    J = job(); roots = J.get('ledger_roots') or []
    return [P(r) for r in roots] + ([str(RUN_DIR)] if str(RUN_DIR) not in [P(r) for r in roots] else [])
def _ledger_cost(f, d0, now):
    """per-pod ledgers {pid: {rate, t0, t_end}} are counted exactly for pods started on/after budget.start. Older pool ledgers
    {'pods': {pid: {integ, rate}}, 'total'} hold lifetime totals with no start times: listed with a WARN, never counted (they can't be dated)."""
    d = json.load(open(f))
    if isinstance(d, dict) and d and all(isinstance(v, dict) and 't0' in v for v in d.values()):
        return sum(p['rate'] * ((p.get('t_end') or now) - p['t0']) / 3600 for p in d.values() if p['t0'] >= d0), 'per-pod'
    if isinstance(d, dict) and ('pods' in d or 'integ' in d):
        # pool ledgers keep a running lifetime total with no per-pod start time, so the spend can't be dated: never counted, always listed.
        return 0.0, 'pool format, undatable: skipped (WARN)'
    return 0.0, 'unknown format: skipped'
def budget_spend(verbose=False):
    """sum EVERY ledger under every ledger root since budget.start. verbose -> print each ledger counted."""
    J = job(); d0 = datetime.datetime.fromisoformat(J['budget']['start']).timestamp(); now = time.time(); tot = 0.0; rows = []
    seen = set()
    for r in ledger_roots():
        for f in sorted(glob.glob(f'{r}/**/ledger*.json', recursive=True)):
            rf = os.path.realpath(f)
            if rf in seen: continue
            seen.add(rf)
            try: c, how = _ledger_cost(f, d0, now)
            except Exception as e: c, how = 0.0, f'unreadable: {e.__class__.__name__}'
            tot += c; rows.append((c, how, f))
    if verbose:
        print(f"ledger roots: {ledger_roots()}  (budget start {J['budget']['start']})")
        for c, how, f in rows:
            if c or 'WARN' in how or verbose == 'all': print(f'  ${c:7.3f}  {how:28s} {f}')
        print(f'  total ${tot:.3f} from {sum(1 for c, _, _ in rows if c)} ledgers with spend ({len(rows)} scanned)')
    return tot
def budget_gate(new_caps, verbose=False):
    J = job(); B = J['budget']['total']; sp = budget_spend(verbose); ok = sp + new_caps <= B
    return ok, f"budget gate {'ok' if ok else 'REFUSED'}: ledgers since {J['budget']['start']} ${sp:.3f} + caps ${new_caps:.2f} = ${sp + new_caps:.2f} vs ${B:.0f}"
if __name__ == '__main__':
    import sys; budget_spend(verbose='all' if '--all' in sys.argv else True)
