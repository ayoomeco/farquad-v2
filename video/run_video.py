"""H3 clip runner. Dry run (default, $0): builds graphs, preps nothing paid, runs the budget gate, lists missing inputs, writes plan_<clip>.json.
Paid: --go (only after the owner said yes to the quote + cap).   python3 run_video.py --clip <id> [--go]
Guards: SECURE only, cap guard before every graph (95%) + watchdog (97%), 2x time stop (per piece and wall), OOM stops the clip (no setting changes),
graph/validation errors stop the clip (NO rerolls), rerolls (job 'rerolls') only for other errors, full error + ComfyUI log saved to fail_*.log,
budget gate = every ledger since budget.start + all clip caps. After the pod: trim to planned frames + source audio re-muxed, then checks
(frames, fps, audio, luma, frame-still size/aspect/blank) -> post_<clip>.json. Native res is the deliverable (no SeedVR2 on video). Never prints keys."""
import os, sys, json, time, subprocess, pathlib, traceback
F = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(F.parent / 'lib')); sys.path.insert(0, str(F))
import fmbcfg as CFG, checks as CK, graph_video as GV, prep_audio as PA
A = sys.argv; arg = lambda k, d=None: A[A.index(k) + 1] if k in A else d
J = CFG.job(); CLIP = arg('--clip')
if CLIP not in J['clips']: sys.exit(f"REFUSED: --clip one of {list(J['clips'])} required")
C = GV.build(CLIP); W = CFG.DATA / J['job']; W.mkdir(parents=True, exist_ok=True)
RATE = float(J.get('rate', 2.49)); os.environ['FMB_RATE'] = str(RATE); os.environ.setdefault('GJ_GPU', J.get('gpu', 'NVIDIA RTX PRO 6000 Blackwell Server Edition'))
CAP = float(C['cap']); STOP_AT = 0.95; BOOT_MIN = J.get('boot_min', 12); ASM_MIN = 3
REND_S = sum(p['est'] for p in C['pieces']); EST_MIN = BOOT_MIN + REND_S / 60 + ASM_MIN + 1; TMUL = float(J.get('time_stop_multiplier', 2.0))
SEED = J.get('seed', 77); REROLLS = tuple(J.get('rerolls', [78, 79]))
ST = W / f'status_{CLIP}.txt'
def status(m): open(ST, 'a').write(f"{time.strftime('%Y-%m-%d %H:%M:%S %Z')} [{CLIP}] {m}\n")
def plan():
    return dict(clip=CLIP, name=C['name'], fps=C['fps'], W=C['W'], H=C['H'], frames_out=C['frames_out'], pieces=[(p['piece'], p['frames'], p['est'], p['guide']) for p in C['pieces']],
                render_s=REND_S, est_min=round(EST_MIN, 1), est_cost=round(EST_MIN / 60 * RATE, 2), cap=CAP, time_stop_min=round(TMUL * EST_MIN), rate=RATE, gpu=os.environ['GJ_GPU'],
                refs=sorted({r for p in C['pieces'] for r in p['refs']}), seed=SEED, audio=str(W / 'audio' / f'{CLIP}.wav') if C['audio'] else None)
def inputs():
    rd = GV.refs_dir(); need = [str(rd / r) for p in C['pieces'] for r in p['refs']] + [p['driver'] for p in C['pieces']]
    return sorted({x for x in need if not os.path.exists(x)})
def segs(done, key):
    import assemble as CR
    P = C['pieces']
    if C['assembly'] == 'chain': return CR.chain_segs([(done[p['piece']][key], p['guide'][1] if p['guide'] else None) for p in P])
    return [('seg', done[p['piece']][key], 0, p['trim_to']) for p in P]
GRAPH_ERR = ('H3LoadReport', 'prompt_outputs_failed_validation', 'Prompt outputs failed validation', 'invalid_prompt', 'KeyError', 'TypeError')
def post(final_raw):
    """local: trim to planned frames + re-mux the source audio, then the checks the rules promise."""
    out = str(W / 'final' / f"{C['name']}_final.mp4"); pl = plan(); res = {}
    if pl['audio']: ok, err = CK.trim_remux(final_raw, pl['audio'], C['frames_out'], C['fps'], out); res['trim_remux'] = (ok, err or 'ok')
    else:
        r = subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', final_raw, '-frames:v', str(C['frames_out']), '-an', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '17', '-movflags', '+faststart', out]); res['trim_remux'] = (r.returncode == 0, 'SILENT (no audio block)')
    json.dump(dict(final=out, plan=pl), open(W / f'post_{CLIP}.json', 'w'), indent=1); return out, res
def main():
    ok, msg = CFG.budget_gate(sum(J['caps'].values())); pl = plan(); miss = inputs()
    aud = PA.prep(CLIP) if C['audio'] and os.path.exists(CFG.P(C['audio']['src'])) else (None, 'SILENT (no audio block)' if not C['audio'] else 'audio source missing')
    if not C['audio'] or aud[0] is None: miss += [] if not C['audio'] else [CFG.P(C['audio']['src'])]
    json.dump(pl, open(W / f'plan_{CLIP}.json', 'w'), indent=1)
    if '--go' not in A:
        print(f"BUILT, NOT RUN. clip {CLIP} ({C['name']}): {len(C['pieces'])} pieces, render {REND_S}s, est {pl['est_min']} min ~${pl['est_cost']}, cap ${CAP}, "
              f"{TMUL:g}x stop {pl['time_stop_min']} min. {msg}. audio: {aud[1]}. missing inputs: {miss or 'none'}. Pre-launch self-check + owner yes, then --go"); return
    if not ok or miss: status(f'FAILED before launch: {msg}; missing {miss}. Nothing launched, $0.'); print(msg, miss); return
    import gj, assemble as CR
    gj.BASE = CFG.RUN_DIR; gj.BASE.mkdir(parents=True, exist_ok=True)
    Jb = gj.Job(f"{J['job']}_{CLIP}", f"fmb-{J['job'][:10]}-{CLIP[:6]}-".lower(), CAP); t0 = time.time()
    OUT = W / 'render' / CLIP; FIN = W / 'final'; OUT.mkdir(parents=True, exist_ok=True); FIN.mkdir(parents=True, exist_ok=True)
    wd = subprocess.Popen([sys.executable, str(F.parent / 'lib' / 'gj.py'), 'watchdog', Jb.name, Jb.pfx, str(CAP)], stdout=open(Jb.dir / 'watchdog.out', 'a'), stderr=subprocess.STDOUT,
                          env=dict(os.environ, PYTHONPATH=str(F.parent / 'lib')))
    pid = None; done = {}; res = {}
    try:
        status(f"STARTING {C['name']} on {os.environ['GJ_GPU']} SECURE, cap ${CAP:.2f}, est {pl['est_min']} min ~${pl['est_cost']}; {msg}")
        pid = Jb.get_pod('r', 'h3_a08', {'GJ_IDLE_EXIT_MIN': '20', 'GJ_MAX_SECONDS': str(int(TMUL * EST_MIN * 60))}, est_min=EST_MIN)
        if not pid: status('FAILED: no pod (stock or boot); nothing rendered'); return
        status(f"LAUNCHED pod {pid} ${RATE}/hr confirmed on RunPod"); status(f"READY: models verified + ComfyUI up on {pid}; rendering {len(C['pieces'])} pieces")
        rd = GV.refs_dir()
        for k, p in enumerate(C['pieces']):
            if Jb.stopped() or time.time() - t0 > TMUL * EST_MIN * 60: status(f'FAILED: {TMUL:g}x time stop (wall) or STOP; clip stops'); break
            g = json.loads(json.dumps(p['graph'])); up = {r: str(rd / r) for r in p['refs']}; up[os.path.basename(p['driver'])] = p['driver']
            if p['guide']:
                prev, gs = p['guide']
                if prev not in done: status(f"FAILED {p['piece']}: previous piece {prev} missing; chain stops"); break
                g['400'] = CR.lvp(done[prev][1], gs, 22); g.pop('401', None); g['16']['inputs']['image'] = ['400', 0]
            ok = False
            for seed in (SEED,) + REROLLS:
                sp = Jb.spend()
                if sp + (Jb.burn() or RATE) * p['est'] / 3600 > CAP * STOP_AT: status(f"CAP GUARD stop at ${sp:.2f}: skipped {p['piece']} and the rest"); break
                g['22']['inputs']['noise_seed'] = seed
                ok, files, err, secs = gj.run_graph(Jb, pid, g, up, str(OUT), f"{p['piece']}_s{seed}", timeout=TMUL * p['est'] + (600 if k == 0 else 0))
                res[f"{p['piece']}_s{seed}"] = dict(ok=ok, secs=secs, err=(err or '')[:300])
                if ok:
                    i = [n for n, f in enumerate(files) if f.endswith('.mp4')][0]; done[p['piece']] = (files[i], gj.PODFILES[f"{p['piece']}_s{seed}"][i])
                    status(f"{'FIRST OUTPUT' if len(done) == 1 else 'PIECE done'} {p['piece']} seed {seed} in {secs}s ({len(done)}/{len(C['pieces'])}), spend ${Jb.spend():.2f}"); break
                fl = W / f"fail_{CLIP}_{p['piece']}_s{seed}.log"; fl.write_text(f'ERR (full):\n{err}\n\n==== comfy log tail ====\n{gj.comfy_log(pid)[-20000:]}'); status(f'error saved: {fl.name}')
                if err == 'timeout': status(f"FAILED {p['piece']}: {TMUL:g}x time stop ({TMUL * p['est']:.0f}s); clip stops"); break
                if err and 'out of memory' in err.lower(): status(f"FAILED {p['piece']}: OOM; clip stops (no setting changes)"); break
                if err and any(x in err for x in GRAPH_ERR): status(f"FAILED {p['piece']}: graph/validation error (not seed-related); no rerolls. Fix the graph, re-quote if a setting changes"); break
                status(f"FAILED {p['piece']} seed {seed}: {(err or '')[:140]}; reroll" + (' (last)' if seed == REROLLS[-1] else ''))
            if not ok: break
            json.dump(res, open(W / f'times_{CLIP}.json', 'w'), indent=1)
        if len(done) == len(C['pieces']):
            podaud = CR.push(pid, pl['audio'], f'audio_{CLIP}.wav') if pl['audio'] else None
            src = {p['piece']: (None, CR.push(pid, p['driver'], 'src_' + os.path.basename(p['driver']))) for p in C['pieces']}
            ga = CR.assembly_graph(C['name'], segs(done, 1), segs(src, 1), podaud, C['fps'], qc_idx=[0, 60, 120, 180, 240, 300, C['frames_out'] - 1])
            okk, files = CR.run_assembly(Jb, pid, C['name'], ga, str(FIN)); status(f'ASSEMBLY ok={okk}: {[os.path.basename(f) for f in files]}')
            if okk:
                raw = [f for f in files if f.endswith('.mp4') and 'SBS' not in f][0]; fin, r = post(raw); status(f"TRIM + AUDIO {'ok' if r['trim_remux'][0] else 'FAILED'}: {C['frames_out']} frames -> {os.path.basename(fin)}")
                status('DONE: run  python3 ../qc/selfcheck.py post --clip ' + CLIP + '  before showing anyone')
    except Exception:
        tb = traceback.format_exc(); (W / f'crash_{CLIP}.log').write_text(tb); status('FAILED: crash ' + tb[-200:].replace('\n', ' '))
    finally:
        if pid: Jb.terminate(pid, 'clip done')
        for p in Jb.live(): Jb.terminate(p['id'], 'sweep')
        time.sleep(20); left = Jb.live()
        json.dump(dict(end=time.strftime('%F %T %Z'), spend=round(Jb.spend(), 3), cap=CAP, pieces=res, pods_left=len(left), pod_prefix=Jb.pfx), open(W / f'end_{CLIP}.json', 'w'), indent=1)
        status(f'TERMINATED: spend ${Jb.spend():.2f} of ${CAP:.2f}; pods left {len(left)}' + (' (0 confirmed on RunPod)' if not left else ' -> CHECK NOW')); (Jb.dir / 'DONE').write_text('1'); wd.terminate()
if __name__ == '__main__': main()
