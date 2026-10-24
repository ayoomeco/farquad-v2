"""Viral 5b V1 card driver (natural mouth, full frame, 24-f warm-up): for each slug in argv, linked 141-f pieces, <=5 targeted tries each, keep best. Checks inside the lead masks."""
import json, time, pathlib, sys, os, subprocess, glob, numpy as np, cv2
FH=os.environ.get('FARQUAD_HOME',os.path.expanduser('~/farquad')); sys.path.insert(0,FH+'/lib'); sys.path.insert(0,FH+'/plan')
from graphs_v5b import graph, LEN, STEP, CLIPS; from fx import faces
R=pathlib.Path(os.environ['DSR_RUN']); CL=FH+'/engine/data/bakeoff/clips'; V=FH; WU=24; W,H=1152,640
LOG=open(R/'checks.jsonl','a',buffering=1); SLUGS=sys.argv[1:]
def frames(p,size=(W,H),fit=False):
    c=cv2.VideoCapture(p); o=[]
    while True:
        ok,f=c.read()
        if not ok: return o
        if fit:
            h,w=f.shape[:2]; s=min(W/w,H/h); f2=cv2.resize(f,(int(w*s)//2*2,int(h*s)//2*2),interpolation=cv2.INTER_AREA)
            b=np.zeros((H,W,3),np.uint8); y=(H-f2.shape[0])//2; x=(W-f2.shape[1])//2; b[y:y+f2.shape[0],x:x+f2.shape[1]]=f2; f=b
        elif size: f=cv2.resize(f,size,interpolation=cv2.INTER_AREA)
        o.append(f)
def write(fr,path):
    p=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s',f'{W}x{H}','-r','24','-i','-','-an','-c:v','libx264','-qp','0','-pix_fmt','yuv420p',path],stdin=subprocess.PIPE)
    for f in fr: p.stdin.write(f.tobytes())
    p.stdin.close(); p.wait()
def src(slug): return f'{V}/{slug}/{slug}_wu_lb.mp4' if os.path.exists(f'{V}/{slug}/{slug}_wu_lb.mp4') else f'{V}/{slug}/{slug}_wu.mp4'
NOMASK={'lalaland'}; ORIG_MAX=int(os.environ.get('ORIG_MAX','3'))
def results():
    out={}
    if (R/'results.jsonl').exists():
        for l in open(R/'results.jsonl'):
            try: r=json.loads(l); out[r['id']]=r
            except Exception: pass
    return out
def mp4(r): return next((o for o in r.get('outputs',[]) if o.endswith('.mp4')),None)
def queue(run):
    pre={x['id'] for x in json.load(open(R/'fp_runs.json'))} if (R/'fp_runs.json').exists() else set()
    q=[json.loads(l)['id'] for l in open(R/'extra_runs.jsonl')] if (R/'extra_runs.jsonl').exists() else []
    if run['id'] in pre or run['id'] in q: return
    with open(R/'extra_runs.jsonl','a') as f: f.write(json.dumps(run)+'\n')
    print(time.strftime('%T'),'queued',run['id'],flush=True)
def wait(rid):
    while True:
        r=results().get(rid)
        if r and r.get('status') in ('success','failed','error','no_output','skipped') and (r['status']!='success' or mp4(r)): return r
        if (R/'STOP').exists(): sys.exit('STOP')
        time.sleep(10)
_MC={}
def masks(slug):
    if slug not in _MC:
        m=np.unpackbits(np.load(f'{V}/{slug}/premask_{slug}.npz')['m'],axis=-1)  # (N,roles,320,576), work-res/2 of the fitted 1152x640 frame
        _MC[slug]=m
    return _MC[slug]
def mask_at(slug,gi):
    if slug in NOMASK: return None
    M=masks(slug); i=gi-WU
    if i<0 or i>=len(M): return None
    return cv2.resize(M[i].max(0).astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST)>0
def flat(f):
    g=cv2.cvtColor(f,cv2.COLOR_BGR2GRAY).astype(np.float32); hsv=cv2.cvtColor(f,cv2.COLOR_BGR2HSV)
    v=cv2.blur(g*g,(15,15))-cv2.blur(g,(15,15))**2
    return float(((v<4)&(hsv[...,1]<25)&(g>90)&(g<235)).mean())
def check(out,slug,k,S):
    O=frames(out); n=min(len(O),len(S)); orig=0; bgd=[]; leak=0
    for j in range(0,n,3):
        gi=k*STEP+j; m=mask_at(slug,gi); M=m
        if m is None: m=np.ones((H,W),bool)
        if flat(O[j])-flat(S[j])>0.08: leak+=1
        if j%6==0:
            SF=faces(S[j])
            for r,e in faces(O[j]):
                cx,cy=int(r[0]+r[2]/2),int(r[1]+r[3]/2)
                if 0<=cy<H and 0<=cx<W and m[cy,cx] and any(float(e@e2)>0.40 for _,e2 in SF): orig+=1
        if M is None: continue
        bg=~cv2.dilate(m.astype(np.uint8),np.ones((25,25),np.uint8)).astype(bool)
        if bg.sum()>5000: bgd.append(float(np.median(np.linalg.norm(cv2.cvtColor(O[j],cv2.COLOR_BGR2LAB).astype(np.float32)-cv2.cvtColor(S[j],cv2.COLOR_BGR2LAB).astype(np.float32),axis=2)[bg])))
    return dict(orig=orig,leak=leak,bg_dE=round(float(np.median(bgd)),1) if bgd else 0.0)
accf=R/'accepted.json'; acc=json.load(open(accf)) if accf.exists() else {}
for slug in SLUGS:
    F=frames(src(slug),fit=True); N=len(F); P=(N-LEN+STEP-1)//STEP+1 if N>LEN else 1
    PCS=[]
    for k in range(P):
        pc=F[k*STEP:k*STEP+LEN]; pc=pc+[pc[-1]]*(LEN-len(pc)); PCS.append(pc); write(pc,f'{CL}/v5b_{slug}'+os.environ.get('RIDTAG','')+f'_d{k}.mp4')
    for k in range(P):
        key=f'{slug}:{k}'
        if key in acc: continue
        guide=None
        if k:
            guide=f'v5b_{slug}'+os.environ.get('RIDTAG','')+f'_g{k}.mp4'
            subprocess.run(['ffmpeg','-v','error','-y','-i',acc[f'{slug}:{k-1}']['out'],'-vf',f'select=gte(n\\,{STEP})','-fps_mode','passthrough','-an','-c:v','libx264','-qp','0',f'{CL}/{guide}'],check=True)
        tries=[]; fixes=[]
        for t in range(int(os.environ.get("TRIES","5"))):
            run=graph(slug,k,guide,42+t,tuple(fixes),t); queue(run); r=wait(run['id'])
            if r['status']!='success': LOG.write(json.dumps({'id':run['id'],'status':r['status']})+'\n'); continue
            o=mp4(r); m={'id':run['id'],'slug':slug,'piece':k,'t':t,'fixes':list(fixes),**check(o,slug,k,PCS[k]),'render_s':r.get('render_s'),'out':o}
            m['ok']=m['orig']<ORIG_MAX and m['bg_dE']<20 and m['leak']<2
            LOG.write(json.dumps(m)+'\n'); print(time.strftime('%T'),m,flush=True); tries.append(m)
            if m['ok']: break
            for f,c in (('orig',m['orig']>=ORIG_MAX),('bg',m['bg_dE']>=20),('leak',m['leak']>=2)):
                if c and f not in fixes: fixes.append(f)
        if not tries: print(slug,'piece',k,'no output; stop',flush=True); break
        b=min(tries,key=lambda m:(not m['ok'],m['orig'],m['leak'],m['bg_dE'])); acc[key]=b; json.dump(acc,open(accf,'w'),indent=1)
    (R/f'DONE_{slug}').write_text('1')
print('card driver done',flush=True); (R/'DRIVER_DONE').write_text('1'); (R/'STOP').write_text('driver done')
