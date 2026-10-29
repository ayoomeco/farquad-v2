"""Fire Marshal Bill self-check. Prints a PASS/FAIL table to paste to the owner (there is no audit bot: this IS the check). $0.
  python3 qc/selfcheck.py pre  [--clip ID ...] [--quote PATH]   before asking for a yes
  python3 qc/selfcheck.py post --clip ID [--final PATH] [--live]   before showing anyone (--live asks RunPod that no job pod is alive)
Exit code 0 only if every row passes."""
import sys, os, json, glob, pathlib, re
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'lib')); sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'video'))
import fmbcfg as CFG, checks as CK
A = sys.argv; MODE = A[1] if len(A) > 1 else ''; arg = lambda k, d=None: A[A.index(k) + 1] if k in A else d
J = CFG.job(); W = CFG.DATA / J['job']; rows = []
def row(check, ok, ev): rows.append((check, 'PASS' if ok else 'FAIL', ev))
def clips(): return [A[i + 1] for i, x in enumerate(A) if x == '--clip'] or list(J['clips'])
def pre():
    import graph_video as GV
    q = pathlib.Path(arg('--quote', str(W / 'QUOTE.md'))); qt = q.read_text() if q.exists() else ''
    m = re.search(r'About \$([0-9.]+) \(cap \$([0-9.]+)\)', qt)
    row('Quote line "About $X (cap $Y), about N min"', bool(m and 'min' in qt), m.group(0) if m else f'not found in {q}')
    tbl = re.search(r'\|\s*Setting\s*\|\s*Old\s*\|\s*New\s*\|', qt, re.I); n = len(re.findall(r'^\|(?!\s*Setting)(?!-).*\|.*\|.*\|$', qt, re.M)) if tbl else 0
    row("What's-different table (Setting, Old, New columns)", bool(tbl and n), f'{n} setting rows' if tbl else 'missing')
    caps = {c: J['caps'].get(c) for c in clips()}; row('Cap set for every clip', all(v and v > 0 for v in caps.values()), ', '.join(f'{k} ${v}' for k, v in caps.items()))
    if m: row('Quote cap = sum of clip caps', abs(float(m.group(2)) - sum(caps.values())) < 0.01, f'quote ${m.group(2)} vs caps ${sum(caps.values()):.2f}')
    ok, msg = CFG.budget_gate(sum(J['caps'].values())); row('Budget gate (all ledgers since start + caps)', ok, msg)
    rd = W / 'refs'; L = J.get('ref_long_side', 1536); bad = []; seen = 0
    for cid in clips():
        try: C = GV.build(cid)
        except Exception as e: row(f'Graphs build + asserts ({cid})', False, f'{e.__class__.__name__}: {e}'); continue
        row(f'Graphs build + asserts ({cid})', True, f"{len(C['pieces'])} pieces, 8 steps euler/simple shift 6, seed {J.get('seed', 77)}, expect=layer counts, no ResCascade/Fizgig")
        for p in C['pieces']:
            for r in p['refs']:
                seen += 1; okr, ev = CK.image_check(str(rd / r), long_side=L)
                if not okr: bad.append(f'{r}: {ev}')
            if not os.path.exists(p['driver']): bad.append(f"driver missing {p['driver']}")
        if C['audio'] is None: row(f'Audio ({cid})', False, 'SILENT deliverable: tell the owner up front')
    row(f'Refs flat 8-bit RGB, {L} long side, not blank; drivers present', not bad and seen > 0, f'{seen} ref uses OK' if not bad else '; '.join(sorted(set(bad)))[:300])
    row('RUNPOD_API_KEY set (by name only)', bool(os.environ.get('RUNPOD_API_KEY')), 'set' if os.environ.get('RUNPOD_API_KEY') else 'missing: ask via the secure secret card')
def post():
    cid = clips()[0]; c = J['clips'][cid]; pj = W / f'post_{cid}.json'
    fin = arg('--final') or (json.load(open(pj))['final'] if pj.exists() else None)
    if not fin or not os.path.exists(fin): row('Final file exists', False, f'{fin}'); return
    p = CK.probe(fin); fo, fps = c['frames_out'], c['fps']
    row('Frame count = plan (trimmed, no frozen tail)', p['frames'] == fo, f"{p['frames']} vs planned {fo}")
    row('fps = source', abs(p['fps'] - fps) < 0.01, f"{p['fps']:.3f} vs {fps}")
    row('Size = render size', (p['w'], p['h']) == (c['W'], c['H']), f"{p['w']}x{p['h']} vs {c['W']}x{c['H']}")
    if c.get('audio'):
        d = abs(p['adur'] - fo / fps); row('Audio present + synced (within 1 frame + 50 ms)', p['audio'] and d <= 1 / fps + 0.05, f"audio {p['adur']:.3f}s vs video {fo / fps:.3f}s")
    else: row('Audio', False, 'SILENT (flagged)')
    ok, ev = CK.luma(fin); row('Luma: not black / not washed out', ok, ev)
    st = CK.frame_stills(fin, 5, str(W / f'post_{cid}_stills')); res = [CK.image_check(s, size=(c['W'], c['H']), aspect=c['W'] / c['H']) for s in st]
    row('Frame stills: size, aspect, not blank (5 samples)', all(r[0] for r in res), '; '.join(r[1] for r in res)[:200])
    ej = W / f'end_{cid}.json'
    if ej.exists():
        e = json.load(open(ej)); row('Spend within cap', e['spend'] <= e['cap'], f"${e['spend']} of ${e['cap']}")
        row('Pods left = 0 (runner end file)', e['pods_left'] == 0, str(e['pods_left']))
        if '--live' in A:
            import rpod as R
            try: live = [x for x in R.mypods() if x['name'].startswith(e.get('pod_prefix', 'fmb-'))]; row('Pods left = 0 (RunPod now)', not live, f'{len(live)} live')
            except Exception as ex: row('Pods left = 0 (RunPod now)', False, f'could not ask RunPod: {ex.__class__.__name__}')
    else: row('Runner end file (spend, pods left)', False, f'{ej.name} missing')
    row('Per-piece eyeball vs masks (right person, no leftover, no ID swap, limbs, outfit back)', False, 'MANUAL: watch the side-by-side + overlay, then mark it') if '--eyeballed' not in A else row('Per-piece eyeball vs masks', True, 'owner-facing notes written')
if MODE == 'pre': pre()
elif MODE == 'post': post()
else: sys.exit(__doc__)
w = max(len(r[0]) for r in rows)
print(f"\nSELF-CHECK {MODE.upper()} — job {J['job']}\n| Check | Result | Evidence |\n|---|---|---|")
for c, r, e in rows: print(f'| {c} | {r} | {e} |')
nf = sum(r == 'FAIL' for _, r, _ in rows); print(f"\n{len(rows) - nf}/{len(rows)} PASS" + ('' if not nf else f' — {nf} FAIL: fix before going on'))
sys.exit(1 if nf else 0)
