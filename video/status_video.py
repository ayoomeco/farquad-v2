"""Live status relay: python3 status_video.py <clip>. Adds DOWNLOADING once the pod's ComfyUI proxy answers; prints each new status line. No secrets."""
import re, sys, time, pathlib, requests
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'lib'))
import fmbcfg as CFG
S = CFG.DATA / CFG.job()['job'] / f'status_{sys.argv[1]}.txt'; pos = len(S.read_text()) if S.exists() else 0; dl = set()
while True:
    txt = S.read_text() if S.exists() else ''
    for pid in re.findall(r'LAUNCHED pod (\w+)', txt):
        if pid in dl: continue
        try:
            if requests.get(f'https://{pid}-8188.proxy.runpod.net/view', params={'filename': 'boot_timings.json', 'type': 'output'}, timeout=10).status_code == 200:
                dl.add(pid); open(S, 'a').write(f"{time.strftime('%Y-%m-%d %H:%M:%S %Z')} DOWNLOADING models on {pid} (sha-checked)\n")
        except Exception: pass
    txt = S.read_text() if S.exists() else ''
    if len(txt) > pos: print(txt[pos:], end='', flush=True); pos = len(txt)
    if 'TERMINATED' in txt: break
    time.sleep(15)
