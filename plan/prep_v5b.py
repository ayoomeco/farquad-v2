import sys,json,pathlib,os
sys.path.insert(0, str(next(p/'lib' for p in pathlib.Path(__file__).resolve().parents if (p/'lib'/'home.py').is_file())))
from home import gj_home
FH=gj_home()
slug=sys.argv[1]; R=pathlib.Path(FH+f'/plan/run_{slug}'); R.mkdir(exist_ok=True); os.environ['DSR_RUN']=str(R)
sys.path.insert(0,FH+'/plan'); sys.argv=[sys.argv[0]]
from graphs_v5b import graph, LEN
import cv2,numpy as np,subprocess,glob
exec(open(FH+'/plan/driver_v5b.py').read().split("accf=R/")[0])
F=frames(src(slug),fit=True)[:LEN]; F=F+[F[-1]]*(LEN-len(F)); write(F,f'{CL}/v5b_{slug}_d0.mp4')
json.dump([graph(slug,0,None,42,(),0)],open(R/'fp_runs.json','w'),indent=1); print(slug,'ok')
