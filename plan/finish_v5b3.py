"""GJ V2 finisher: on DRIVER_DONE, run post_gjv2 on the pod (upscale, stitch, score, encode),
then terminate this run's pod and verify 0 remain. A missing pod channel falls back to
assemble_v5b.py, which keeps the source fps and does not grade. Writes FINISHED.json.
Usage: python finish_v5b3.py <run dir> <slug...>   DRY=1 = no API calls (test)."""
import sys, os, json, time, subprocess, pathlib
sys.path.insert(0, str(next(p/'lib' for p in pathlib.Path(__file__).resolve().parents if (p/'lib'/'home.py').is_file())))
from home import gj_home
FH=gj_home()
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
def pod_post(slugs):
    """Upscale, stitch, score, and encode on the pod while it is still up."""
    sys.path.insert(0, str(pathlib.Path(FH)/'polish'/'pod'))
    import post_gjv2
    url=os.environ.get('POD_STATUS_URL','').strip(); nonce=os.environ.get('JUTSU_FIX_NONCE','').strip()
    meta=R/'pod.json'
    if meta.exists():
        try:
            d=json.load(open(meta)); url=url or str(d.get('status_url') or ''); nonce=nonce or str(d.get('nonce') or '')
        except Exception as e:
            log('pod.json', type(e).__name__)
    out={}
    for s in slugs:
        (R/f'post_{s}.json').write_text(json.dumps(post_gjv2.plan(s), indent=1))
        if DRY:
            out[s]='dry'; log('pod post dry', s); continue
        if not url or not nonce:
            out[s]='no-pod-channel'; log('pod post', s, 'no POD_STATUS_URL/JUTSU_FIX_NONCE'); continue
        sys.path.insert(0, FH+'/engine/scripts')
        from phase1_bakeoff import PodStatus
        pod=PodStatus(url, nonce=nonce)
        video=os.environ.get('GJ_POD_VIDEO', f'/opt/jutsu/work/{s}.mp4')
        native=os.environ.get('GJ_POD_NATIVE', f'/opt/jutsu/work/{s}_native.mp4')
        dest=os.environ.get('GJ_POD_DEST', f'/opt/jutsu/work/{s}_final.mp4')
        jid=pod.exec(f"set -e\npython3 /opt/jutsu/post_gjv2.py {s} --video {video} --native {native} --dest {dest}\n")
        if not jid:
            out[s]='exec-failed'; log('pod post exec failed', s); continue
        state='posted'
        for _ in range(90):
            st=pod.job(jid)
            if st.get('done'):
                state='ok' if st.get('rc')==0 else 'fail rc='+str(st.get('rc'))
                if st.get('out'): log('pod post out', s, str(st.get('out'))[-180:])
                break
            time.sleep(2)
        out[s]=state; log('pod post', s, state)
    return out
posted=pod_post(SL)
left=kill_all()   # terminate after the on-pod post
res={'pods_left':left,'where':'pod','pod_post':posted,'stitched':{}}
for s in SL:
    if DRY: res['stitched'][s]='dry'; continue
    if posted.get(s)=='ok': res['stitched'][s]='pod'; continue
    env={**os.environ}; p=subprocess.run([sys.executable,os.environ.get('ASSEMBLE',FH+'/plan/assemble_v5b.py'),s],capture_output=True,text=True,env=env)
    res['stitched'][s]='box-nograde' if p.returncode==0 else 'fail: '+p.stderr[-300:]; log('stitch',s,res['stitched'][s][:60])
json.dump(res,open(R/'FINISHED.json','w'),indent=1); log('finished',res)
