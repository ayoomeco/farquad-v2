"""Live status relay for the ref job: prints each new line of $FMB_DATA/<job>/refjob/status.txt until TERMINATED."""
import sys, time, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'lib'))
import fmbcfg as CFG
S = CFG.DATA / CFG.job()['job'] / 'refjob' / 'status.txt'; pos = len(S.read_text()) if S.exists() else 0
while True:
    t = S.read_text() if S.exists() else ''
    if len(t) > pos: print(t[pos:], end='', flush=True); pos = len(t)
    if 'TERMINATED' in t: break
    time.sleep(15)
