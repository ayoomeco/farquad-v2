# Viral 5b launcher. Cap = argv[1] = the cap the owner approved for THIS job (per job, no fixed default).
import pty, os, re, sys, pathlib
sys.path.insert(0, str(next(p/'lib' for p in pathlib.Path(__file__).resolve().parents if (p/'lib'/'home.py').is_file())))
from home import gj_home
FH=gj_home()
B=FH+'/engine'; R=sys.argv[3]; cap=sys.argv[1]; tag=sys.argv[2]
# per-job cap files for idle_watchdog.py: CAP.txt = approved cap, B0.txt = RunPod balance before this job spends anything
os.makedirs(R,exist_ok=True); open(f'{R}/CAP.txt','w').write(str(float(cap)))
if not os.path.exists(f'{R}/B0.txt'):
    try:
        sys.path.insert(0,FH+'/lib'); import pods as _P; open(f'{R}/B0.txt','w').write(f'{_P.c.balance():.4f}')
    except Exception as e: print('B0 not written:',type(e).__name__)
args=['python3',B+'/scripts/phase1_bakeoff.py','launch']+(['--attach-pod',os.environ['ATTACH']] if os.environ.get('ATTACH') else [])+['--refine','--fullperson','--gpu',os.environ.get('BENCH_GPU','any24'),'--cloud',os.environ.get('BENCH_CLOUD','COMMUNITY'),'--cap',cap,'--linger',os.environ.get('LINGER','600'),
      '--i-approve-spend','--create-attempts','8','--allow-pro6000']
os.environ.update(JUTSU_CUDA_LIST=os.environ.get('BENCH_CUDA','13.1,13.2'), JUTSU_UNCHUNK='0', FP_IMAGE='ghcr.io/ayoomeco/chywgja@sha256:5727e5567c6a6b86ed8a0162b9834e6a93030f6b4a0daae4c9988a7a27d7d1f9', FP_BOOT_TIMEOUT='720',
    JUTSU_CONFIG=B+'/dsr/genie_dsr.toml', JUTSU_STALL_S='900', FP_RUN_DIR=R, FP_GROUP='gjfast', JUTSU_ENABLE_H3RELAY='1', FP_REVIEW='dsr', FP_DISK_GB='240',
    FP_SETUP_TIMEOUT=os.environ.get('BENCH_SETUP','1200'), FP_MIN_DL='2000', JUTSU_CLOUD_ORDER=os.environ.get('BENCH_ORDER','COMMUNITY_FIRST'))
os.environ.pop('FP_KEEP_POD',None)
log=open(f'{R}/launch_{tag}.log','w',buffering=1)
pid,fd=pty.fork()
if pid==0: os.execvp(args[0],args)
buf='';sent=False
while True:
    try: d=os.read(fd,4096).decode('utf-8','replace')
    except OSError: break
    if not d: break
    log.write(d)
    if not sent:
        buf+=d; m=re.search(r'Type exactly "(SPEND [0-9.]+)"',buf)
        if m: os.write(fd,(m.group(1)+'\n').encode()); sent=True
_,st=os.waitpid(pid,0); log.write(f'\n[wrapper exit {os.waitstatus_to_exitcode(st)}]\n')
