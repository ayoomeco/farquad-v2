import json,sys,os,numpy as np,cv2
sys.path.insert(0,'.'); import qc_v5b3 as Q
sys.argv=['x']; os.environ['DSR_RUN']='/tmp'
W,H=1152,640
def frames(p,fit=False):
    c=cv2.VideoCapture(p);o=[]
    while True:
        ok,f=c.read()
        if not ok: return o
        if fit:
            h,w=f.shape[:2]; s=min(W/w,H/h); f2=cv2.resize(f,(int(w*s)//2*2,int(h*s)//2*2),interpolation=cv2.INTER_AREA)
            b=np.zeros((H,W,3),np.uint8); y=(H-f2.shape[0])//2; x=(W-f2.shape[1])//2; b[y:y+f2.shape[0],x:x+f2.shape[1]]=f2; f=b
        else: f=cv2.resize(f,(W,H))
        o.append(f)
slug=os.environ.get("SLUG","walken")
acc=json.load(open(f"run_{slug}/accepted.json")); k=int(os.environ.get("K","0")); a=acc[f"{slug}:{k}"]
import glob; src=(glob.glob(os.environ.get('FARQUAD_HOME',os.path.expanduser('~/farquad'))+f"/{slug}/{slug}_wu_lb.mp4") or [os.environ.get('FARQUAD_HOME',os.path.expanduser('~/farquad'))+f"/{slug}/{slug}_wu.mp4"])[0]; S=frames(src,fit=True)[k*119:k*119+141]; O=frames(a["out"])[:141]
print('stored V1 checks',{x:a[x] for x in ('orig','leak','bg_dE')})
r=Q.popin(O,S); print('ACCEPTED piece  ', r)
# self-test: source vs itself must pass with ~0
print('SOURCE vs SOURCE', Q.popin(S,S)['pop'])
# bad 1: expression mismatch = render reversed in time against source
# bad 2: pop-in = person vanishes for 12 frames (replace with source-background flat blur) + identity flip
P=[f.copy() for f in O]
for j in range(40,64): P[j]=cv2.GaussianBlur(P[j],(151,151),0)
print('POP-IN INJECTED ', Q.popin(P,S))
