"""Character-ref runner. Dry run (default, $0) prints the stage counts, cost estimate, budget gate, missing inputs. --go spends (owner yes to quote + cap first).
One SECURE pod (ref_job.gpu, default RTX A6000 $0.59/hr), sets qwen21,qwen2511_lite,seedvr2. Every output is checked as it lands:
T whites: size == canvas, pixel std > 10 (blank-canvas wash-out guard), RGB  -> FIRST MISS STOPS THE STAGE.  R/U: RGB, std > 10, aspect kept.
Budget gate = every ledger since budget.start + this cap. 2x time stop per graph and on the wall. Graph errors: no reroll. Pods terminated + swept."""
import os, sys, json, time, subprocess, pathlib, traceback
F = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(F.parent / 'lib')); sys.path.insert(0, str(F))
import fmbcfg as CFG, checks as CK, graph_refs as GR
J = CFG.job(); RJ = J['ref_job']; CAP = float(RJ['cap']); RATE = float(RJ.get('rate', 0.59)); EST = RJ.get('est_s', {'T': 150, 'R': 120, 'U': 60})
os.environ['FMB_RATE'] = str(RATE); os.environ['GJ_GPU'] = RJ.get('gpu', 'NVIDIA RTX A6000')
W = CFG.DATA / J['job'] / 'refjob'; W.mkdir(parents=True, exist_ok=True); ST = W / 'status.txt'; GEN = W / 'gen'
def status(m): open(ST, 'a').write(f"{time.strftime('%Y-%m-%d %H:%M:%S %Z')} [refs] {m}\n")
def main():
    P = GR.plan(); n = {k: len(P[k]) for k in 'TRU'}; rs = sum(n[k] * EST[k] for k in 'TRU'); est_min = 12 + rs / 60; TMUL = float(J.get('time_stop_multiplier', 2.0))
    ok, msg = CFG.budget_gate(CAP)
    miss = sorted({x for t in P['T'] for x in t['uploads'].values() if not os.path.exists(x)} | {r['src'] for r in P['R'] if os.path.isabs(r['src']) and not os.path.exists(r['src'])})
    if '--go' not in sys.argv:
        print(f"BUILT, NOT RUN. refs: T={n['T']} R={n['R']} U={n['U']} on {os.environ['GJ_GPU']} SECURE ${RATE}/hr, est {est_min:.0f} min ~${est_min / 60 * RATE:.2f}, cap ${CAP:.2f}, "
              f"{TMUL:g}x stop {TMUL * est_min:.0f} min. {msg}. missing inputs: {miss or 'none'}. Pre-launch self-check + owner yes, then --go"); return
    if not ok or miss: status(f'FAILED before launch: {msg}; missing {miss}; $0'); return
    import gj
    gj.BASE = CFG.RUN_DIR; Jb = gj.Job(f"{J['job']}_refs", f"fmb-{J['job'][:10]}-refs-".lower(), CAP); t0 = time.time(); pid = None; made = {}; res = {}
    wd = subprocess.Popen([sys.executable, str(F.parent / 'lib' / 'gj.py'), 'watchdog', Jb.name, Jb.pfx, str(CAP)], stdout=open(Jb.dir / 'watchdog.out', 'a'), stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONPATH=str(F.parent / 'lib')))
    try:
        status(f"STARTING refs on {os.environ['GJ_GPU']} SECURE, cap ${CAP:.2f}; {msg}")
        pid = Jb.get_pod('r', 'qwen21,qwen2511_lite,seedvr2', {'GJ_MAX_SECONDS': str(int(TMUL * est_min * 60))}, est_min=est_min)
        if not pid: status('FAILED: no pod'); return
        status(f'LAUNCHED pod {pid} ${RATE}/hr confirmed; READY'); first = True
        for stg in 'TRU':
            for j in P[stg]:
                if Jb.stopped() or time.time() - t0 > TMUL * est_min * 60: status(f'FAILED: {TMUL:g}x time stop or STOP'); raise SystemExit
                if Jb.spend() + (Jb.burn() or RATE) * EST[stg] / 3600 > CAP * 0.95: status(f"CAP GUARD at ${Jb.spend():.2f}: stopped before {j['name']}"); raise SystemExit
                ups = dict(j.get('uploads', {}))
                if 'src' in j: s = made.get(j['src'], j['src']); ups['SRC.png'] = s
                okk, files, err, secs = gj.run_graph(Jb, pid, j['graph'], ups, str(GEN), j['name'], timeout=TMUL * EST[stg] + (600 if first else 0)); first = False
                if not okk:
                    (W / f"fail_{j['name']}.log").write_text(f'ERR (full):\n{err}\n\n==== comfy log tail ====\n{gj.comfy_log(pid)[-20000:]}')
                    status(f"FAILED {j['name']}: {(err or '')[:120]} (no reroll; full error in fail_{j['name']}.log); stage {stg} stops"); break
                f = files[0]; chk = CK.image_check(f, size=j.get('expect_size')) if stg == 'T' else CK.image_check(f)
                res[j['name']] = dict(file=f, secs=secs, check=chk)
                if not chk[0]: status(f"FAILED check {j['name']}: {chk[1]}; stage {stg} stops on the first miss"); break
                made[j['name']] = f; status(f"{'FIRST OUTPUT' if len(made) == 1 else 'done'} {j['name']} in {secs}s ({chk[1]}), spend ${Jb.spend():.2f}")
            else: status(f'STAGE {stg} done ({len(P[stg])})'); continue
            break
    except SystemExit: pass
    except Exception: (W / 'crash.log').write_text(traceback.format_exc()); status('FAILED: crash, see crash.log')
    finally:
        if pid: Jb.terminate(pid, 'done')
        for p in Jb.live(): Jb.terminate(p['id'], 'sweep')
        time.sleep(20); left = Jb.live()
        json.dump(dict(end=time.strftime('%F %T %Z'), spend=round(Jb.spend(), 3), cap=CAP, results=res, pods_left=len(left), pod_prefix=Jb.pfx), open(W / 'end.json', 'w'), indent=1)
        status(f'TERMINATED: spend ${Jb.spend():.2f} of ${CAP:.2f}; pods left {len(left)}'); (Jb.dir / 'DONE').write_text('1'); wd.terminate()
if __name__ == '__main__': main()
