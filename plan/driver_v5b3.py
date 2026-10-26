"""Farquad V1 card driver v3 = driver_v5b (V1 recipe unchanged) + pop-in check per piece,
auto re-roll (max 5, keep best) with targeted prompt fixes, then AUTO-STITCH + AUTO-TERMINATE when done.
Usage: DSR_RUN=<run dir> python driver_v5b3.py <slug...>   (GRAPHS=graphs_v5b2 to use the V2 graph module)"""
import os, sys, json, subprocess, importlib, pathlib, time
FH=os.environ.get('FARQUAD_HOME',os.path.expanduser('~/farquad')); sys.path.insert(0,FH+'/plan')
G=importlib.import_module(os.environ.get('GRAPHS','graphs_v5b'))
G.FIX.setdefault('pop'," Keep every person present for the whole shot exactly as in <Video 1>; no one appears, vanishes or changes identity.")
import qc_v5b3 as Q
src=open(FH+'/plan/driver_v5b.py').read()
src=src.replace("from graphs_v5b import graph, LEN, STEP, CLIPS","from %s import graph, LEN, STEP, CLIPS"%os.environ.get('GRAPHS','graphs_v5b'))
# extend check(): add pop-in, mask-aware
src=src.replace("    return dict(orig=orig,leak=leak,bg_dE=round(float(np.median(bgd)),1) if bgd else 0.0)",
"""    mf=lambda j:mask_at(slug,k*STEP+j)
    return dict(orig=orig,leak=leak,bg_dE=round(float(np.median(bgd)),1) if bgd else 0.0,**Q.popin(O[:n],S[:n],mf))""")
src=src.replace("m['ok']=m['orig']<ORIG_MAX and m['bg_dE']<20 and m['leak']<2","m['ok']=m['orig']<ORIG_MAX and m['bg_dE']<20 and m['leak']<2 and m['pop_ok']")
src=src.replace("(('orig',m['orig']>=ORIG_MAX),('bg',m['bg_dE']>=20),('leak',m['leak']>=2))","(('orig',m['orig']>=ORIG_MAX),('bg',m['bg_dE']>=20),('leak',m['leak']>=2),('pop',not m['pop_ok']))")
src=src.replace("key=lambda m:(not m['ok'],m['orig'],m['leak'],m['bg_dE'])","key=lambda m:(not m['ok'],m['orig'],m['leak'],m['pop'],m['bg_dE'])")
src=src.replace("int(os.environ.get(\"TRIES\",\"5\"))","min(5,int(os.environ.get(\"TRIES\",\"5\")))")
src=src.replace("(R/'DRIVER_DONE').write_text('1'); (R/'STOP').write_text('driver done')","(R/'DRIVER_DONE').write_text('1')")
assert 'Q.popin' in src and 'expr' not in src and "'pop'" in src
assert "write_text('driver done')" not in src
if os.environ.get('DRYCHECK'): print('patched ok'); sys.exit(0)
exec(compile(src,'driver_v5b3','exec'))
# ---- finish: auto-stitch, then hand to finisher (terminate + verify 0 pods)
R=pathlib.Path(os.environ['DSR_RUN'])
subprocess.run([sys.executable,FH+'/plan/finish_v5b3.py',str(R)]+SLUGS)
