"""Viral5b hard stop: if RunPod spend since B0 >= $9.50 -> STOP all drivers + terminate ALL pods."""
import sys,time,pathlib,os; sys.path.insert(0,os.environ.get('FARQUAD_HOME',os.path.expanduser('~/farquad'))+'/lib'); import pods
V=pathlib.Path(os.environ.get('FARQUAD_HOME',os.path.expanduser('~/farquad')))/'plan'; B0=float((V/'B0').read_text()); L=V/'guard.log'
while not (V/'GUARD_OFF').exists():
    try:
        b=pods.c.balance(); n=len(pods.ls()); open(L,'a').write(f"{time.strftime('%T')} bal {b:.3f} spent {B0-b:.3f} pods {n}\n")
        if B0-b>=9.5:
            for r in V.glob('run_*'): (r/'STOP').write_text('hard stop $9.50')
            for p in pods.ls(): pods.c.terminate_pod(p['id']); open(L,'a').write(f"HARD STOP terminated {p['id']}\n")
    except Exception as e: open(L,'a').write(f'err {type(e).__name__}\n')
    time.sleep(30)
