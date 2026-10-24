"""Run finisher: on DRIVER_DONE -> stop keepalive, terminate every pod of this run, verify 0 pods remain
(retry up to 5x), then auto-stitch each slug with assemble_v5b.py (CPU, box). Writes FINISHED.json.
Usage: python finish_v5b3.py <run dir> <slug...>   DRY=1 = no API calls (test)."""
import sys, os, json, time, subprocess, pathlib
FH=os.environ.get('FARQUAD_HOME',os.path.expanduser('~/farquad'))
R=pathlib.Path(sys.argv[1]); SL=sys.argv[2:]; DRY=os.environ.get('DRY')=='1'
def log(*a):
    s=time.strftime('%T')+' '+' '.join(map(str,a)); print(s,flush=True); open(R/'finish.log','a').write(s+'\n')
(R/'KEEPALIVE_OFF').write_text('finish'); (R/'STOP').write_text('driver done')
def pods():
    if DRY: return json.loads(os.environ.get('DRY_PODS','[]'))
    sys.path.insert(0,FH+'/lib'); import pods as P; return P.ls(),P
def kill_all():
    for a in range(5):
        if DRY: P=[];  log('DRY pods',0); return 0
        L,P=pods()
        others=[r for r in R.parent.glob('run_*') if r!=R and not (r/'DRIVER_DONE').exists() and not (r/'STOP').exists()]
        pid=(R/'POD_ID').read_text().strip() if (R/'POD_ID').exists() else None
        if not pid and (R/'checkpoint.json').exists():
            try: pid=json.load(open(R/'checkpoint.json')).get('pod')
            except Exception: pid=None
        if pid: mine=[p for p in L if p['id']==pid]
        elif others: log('other runs active, no POD_ID: leaving pods to idle_watchdog',[o.name for o in others]); return len(L)
        else: log('no POD_ID for this run: not touching other pods'); return len(L)
        if not mine: log('verified 0 pods', 'try',a+1); return 0
        for p in mine:
            try: P.c.terminate_pod(p['id']); log('terminated',p['id'])
            except Exception as e: log('terminate err',type(e).__name__)
        time.sleep(20)
    L,_=pods(); log('WARNING pods still listed',len(L)); return len(L)
left=kill_all()   # terminate FIRST so no idle card bills during the CPU stitch
res={'pods_left':left,'stitched':{}}
for s in SL:
    if DRY: res['stitched'][s]='dry'; continue
    env={**os.environ}; p=subprocess.run([sys.executable,os.environ.get('ASSEMBLE',FH+'/plan/assemble_v5b.py'),s],capture_output=True,text=True,env=env)
    res['stitched'][s]='ok' if p.returncode==0 else 'fail: '+p.stderr[-300:]; log('stitch',s,res['stitched'][s][:60])
json.dump(res,open(R/'FINISHED.json','w'),indent=1); log('finished',res)
