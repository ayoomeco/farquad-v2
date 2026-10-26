"""Farquad V1 per-piece pop-in check (free, CPU, box-side).
popin(): flags faces/people appearing or vanishing vs source, identity flips between frames, and
  temporal spikes (render frame-to-frame change >> source's) inside the lead area.
Returns a dict with pop_ok. Thresholds via env (POP_MAX, POP_SPIKE, POP_HOLD)."""
import os, numpy as np, cv2, sys, pathlib
sys.path.insert(0, str(next(p/'lib' for p in pathlib.Path(__file__).resolve().parents if (p/'lib'/'home.py').is_file())))
from home import gj_home
sys.path.insert(0, gj_home()+'/lib'); from fx import faces
POP_MAX=int(os.environ.get('POP_MAX','2')); SPIKE=float(os.environ.get('POP_SPIKE','3.0'))
def popin(O,S,mask_fn=lambda j:None,step=2):
    n=min(len(O),len(S)); pops=0; flips=0; spikes=0; prev=None; ev=[]
    pres_o=[];pres_s=[]
    def inm(r,m): 
        cx,cy=int(r[0]+r[2]/2),int(r[1]+r[3]/2); return m is None or (0<=cy<m.shape[0] and 0<=cx<m.shape[1] and m[cy,cx])
    def near(r,L):
        c=(r[0]+r[2]/2,r[1]+r[3]/2); b=min(L,key=lambda q:np.hypot(q[0][0]+q[0][2]/2-c[0],q[0][1]+q[0][3]/2-c[1]),default=None)
        return b if b is not None and np.hypot(b[0][0]+b[0][2]/2-c[0],b[0][1]+b[0][3]/2-c[1])<0.5*r[2] else None
    pO=pS=None
    for j in range(0,n,step):
        m=mask_fn(j); big=lambda L:[x for x in L if x[0][2]>=0.06*O[j].shape[1] and inm(x[0],m)]
        fo=faces(O[j]); fs=faces(S[j]); pres_o.append(len(big(fo))); pres_s.append(len(big(fs)))
        if pO is not None:
            for r,e in fo:
                if not inm(r,m) or r[2]<40: continue
                a=near(r,pO); sa=near(r,fs); sb=near(r,pS) if pS is not None else None
                # flip = render face at same spot changed identity while the SOURCE face there stayed the same person
                if a is not None and sa is not None and sb is not None and float(a[1]@e)<0.30 and float(sa[1]@sb[1])>0.55: flips+=1; ev.append(('flip',j))
        pO,pS=fo,fs
    po=np.array(pres_o); ps=np.array(pres_s)
    # extra/missing face held >=2 samples but not in source = pop-in/out
    diff=po-ps; run=0
    for i,v in enumerate(diff):
        run=(run+1 if i and diff[i-1]==v else 1) if v!=0 else 0
        if run==int(os.environ.get('POP_HOLD','4')): pops+=1; ev.append(('count',i*step,int(v)))
    for j in range(1,n):
        dO=float(np.abs(O[j].astype(np.int16)-O[j-1]).mean()); dS=float(np.abs(S[j].astype(np.int16)-S[j-1]).mean())
        if dO>SPIKE*(dS+2.0) and dO>12: spikes+=1; ev.append(('spike',j))
    tot=pops+flips+spikes
    return dict(pop=tot,pop_counts=pops,pop_flips=flips,pop_spikes=spikes,pop_events=ev[:10],pop_ok=tot<POP_MAX)

