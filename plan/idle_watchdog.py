"""Idle-pod watchdog: every 60 s lists RunPod pods. A pod counts as idle when no run dir under plan/run_* has an
active driver (no DRIVER_DONE/STOP) OR its run's results.jsonl hasn't changed for IDLE_MIN (default 15) min.
Idle >= IDLE_MIN -> terminate + verify. Also enforces the PER-JOB cap from <run>/CAP.txt (set by the owner per job):
spend since <run>/B0.txt >= 95% of CAP -> STOP + terminate. Never prints keys. Stop with plan/WATCHDOG_OFF.
DRY=1 + DRY_STATE=<json> for tests."""
import os, sys, json, time, pathlib
V=pathlib.Path(os.environ.get('FARQUAD_HOME',os.path.expanduser('~/farquad')))/'plan'; IDLE=float(os.environ.get('IDLE_MIN','15'))*60; DRY=os.environ.get('DRY')=='1'
L=V/'watchdog.log'
def log(s): open(L,'a').write(time.strftime('%T ')+s+'\n'); print(s,flush=True)
def active_runs(now):
    act=[]
    for r in V.glob('run_*'):
        if (r/'DRIVER_DONE').exists() or (r/'STOP').exists(): continue
        f=r/'results.jsonl'; t=max([f.stat().st_mtime] if f.exists() else [0]+[r.stat().st_mtime])
        if now-t<IDLE: act.append(r)
    return act
def cap_hit(bal):
    hit=[]
    for r in V.glob('run_*'):
        if (r/'CAP.txt').exists() and (r/'B0.txt').exists() and not (r/'DRIVER_DONE').exists() and not (r/'STOP').exists():
            cap=float((r/'CAP.txt').read_text()); b0=float((r/'B0.txt').read_text())
            if b0-bal>=0.95*cap: hit.append((r,b0-bal,cap))
    return hit
def tick(pods,bal,term,now=None):
    now=now or time.time(); acts=active_runs(now); out=[]
    for r,sp,cap in cap_hit(bal):
        (r/'STOP').write_text(f'per-job cap {cap}'); log(f'CAP {r.name} spent {sp:.2f}/{cap}'); out.append('cap')
    if pods and (not acts or 'cap' in out):
        for p in pods: term(p['id']); log('idle/cap terminate '+p['id']); out.append(p['id'])
    return out
if __name__=='__main__':
    if DRY:
        st=json.loads(os.environ['DRY_STATE']); print(tick(st['pods'],st['bal'],lambda i:None)); sys.exit()
    sys.path.insert(0,str(V.parent/'lib')); import pods as P
    idle_since=None
    while not (V/'WATCHDOG_OFF').exists():
        try:
            pl=P.ls(); r=tick(pl,P.c.balance(),P.c.terminate_pod)
            if r: time.sleep(20); log(f'verify pods now {len(P.ls())}')
        except Exception as e: log('err '+type(e).__name__)
        time.sleep(60)
