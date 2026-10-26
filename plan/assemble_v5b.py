"""GJ V2 assemble: link accepted pieces (10-hold/12-xfade on the 22-f overlap, no Lab match), drop the 24 warm-up frames,
map the FULL rendered frame back onto the source canvas (letterbox/pillarbox bars of the source kept as-is; no person masks, no paste-back).
No colour grade, no grain, no B&W grade. Encode at the source fps: H.264 High 4.2, yuv420p, bt709, AAC, +faststart.
The pod post (polish/pod/post_gjv2.py) is the default deliverable; this is the no-grade fallback for pieces already on disk."""
import json,sys,glob,subprocess,numpy as np,cv2,pathlib,os
sys.path.insert(0, str(next(p/'lib' for p in pathlib.Path(__file__).resolve().parents if (p/'lib'/'home.py').is_file())))
from home import gj_home
slug=sys.argv[1]; V=gj_home(); W,H=1152,640; WU=24
acc=json.load(open(f'{V}/plan/run_{slug}/accepted.json')); tag=os.environ.get('RIDTAG','')
P_=sorted([k for k in acc if k.startswith(slug+':')],key=lambda k:int(k.split(':')[1])); print(slug,[acc[k]['id'] for k in P_])
NAT=glob.glob(f'{V}/{slug}/{slug}_*_native.mp4')[0]
def rd(p):
    c=cv2.VideoCapture(p);o=[]
    while True:
        ok,f=c.read()
        if not ok: return o
        o.append(f)
enc=lambda f: cv2.imencode('.jpg',f,[cv2.IMWRITE_JPEG_QUALITY,98])[1]
dec=lambda b: cv2.imdecode(b,1)
out=[]
for k in P_:
    F=rd(acc[k]['out'])
    if out:
        # No Lab match. Crossfade the overlap only.
        for j in range(22):
            wt=0 if j<10 else (j-9)/12.0; out[-22+j]=enc(np.clip(dec(out[-22+j]).astype(np.float32)*(1-wt)+F[j].astype(np.float32)*wt,0,255).astype(np.uint8))
        F=F[22:]
    out+=[enc(f) for f in F]; del F
out=out[WU:]
class LazyS:
    def __init__(s,p): s.c=cv2.VideoCapture(p); s.n=int(s.c.get(7)); s.i=-1; s.f=None
    def __len__(s): return s.n
    def __getitem__(s,i):
        if i<s.i: s.c.set(1,i); s.i=i-1
        while s.i<i:
            ok,f=s.c.read(); s.i+=1
            if ok: s.f=f
        return s.f
S=LazyS(NAT); _c=cv2.VideoCapture(NAT); N=0
while _c.grab(): N+=1
S.n=N; sh,sw=S[0].shape[:2]; out=out[:N]; assert len(out)>=N-2,(len(out),N)
while len(out)<N: out.append(out[-1])
# active (non-bar) area of the source
g=np.max(np.array([cv2.cvtColor(S[i],cv2.COLOR_BGR2GRAY) for i in range(0,N,max(1,N//40))]),0); S=LazyS(NAT); S.n=N
rows=np.where(g.max(1)>28)[0]; cols=np.where(g.max(0)>28)[0]; ay0,ay1,ax0,ax1=rows[0],rows[-1]+1,cols[0],cols[-1]+1
if os.environ.get('ACTIVE'): ay0,ay1,ax0,ax1=map(int,os.environ['ACTIVE'].split(','))
s=min(W/sw,H/sh); w2,h2=int(sw*s)//2*2,int(sh*s)//2*2; x0,y0=(W-w2)//2,(H-h2)//2
def cand_fit(o):   # model kept the source framing (bars included)
    return cv2.resize(o[y0:y0+h2,x0:x0+w2],(sw,sh),interpolation=cv2.INTER_LANCZOS4)
def cand_fill(o):  # model filled the bars: whole render == active area
    f=S_cur.copy(); f[ay0:ay1,ax0:ax1]=cv2.resize(o,(ax1-ax0,ay1-ay0),interpolation=cv2.INTER_LANCZOS4); return f
def keepbars(f,src):  # bars come from the source canvas; the picture area is 100% the render
    r=src.copy(); r[ay0:ay1,ax0:ax1]=f[ay0:ay1,ax0:ax1]; return r
def score(f,src):
    e1=cv2.Canny(cv2.cvtColor(f,cv2.COLOR_BGR2GRAY),60,150)[ay0:ay1,ax0:ax1].astype(np.float32); e2=cv2.Canny(cv2.cvtColor(src,cv2.COLOR_BGR2GRAY),60,150)[ay0:ay1,ax0:ax1].astype(np.float32)
    e1=cv2.GaussianBlur(e1,(0,0),4); e2=cv2.GaussianBlur(e2,(0,0),4); return float((e1*e2).sum()/(np.linalg.norm(e1)*np.linalg.norm(e2)+1e-6))
sc={'fit':[], 'fill':[]}
for i in range(0,N,max(1,N//12)):
    S_cur=S[i]; o=dec(out[i]); sc['fit'].append(score(keepbars(cand_fit(o),S[i]),S[i])); sc['fill'].append(score(cand_fill(o),S[i]))
S=LazyS(NAT); S.n=N
mode=os.environ.get('MODE') or max(sc,key=lambda k:np.mean(sc[k])); print('active',ay0,ay1,ax0,ax1,'scores',{k:round(float(np.mean(v)),3) for k,v in sc.items()},'->',mode)
TW,TH=1920,1080; FD=f'{V}/{slug}/final'; pathlib.Path(FD).mkdir(exist_ok=True)
fps=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=r_frame_rate','-of','csv=p=0',NAT],capture_output=True,text=True).stdout.strip()
def pp(path,w,h,vf):
    # Source fps via -r on the raw frames (passthrough does not apply to rawvideo). No interpolation filter.
    return subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s',f'{w}x{h}','-r',fps,'-i','-','-i',NAT,'-map','0:v','-map','1:a?',
        '-vf',vf,'-c:v','libx264','-profile:v','high','-level:v','4.2','-crf','16','-preset','slow','-pix_fmt','yuv420p',
        '-colorspace','bt709','-color_primaries','bt709','-color_trc','bt709','-x264-params','colorprim=bt709:transfer=bt709:colormatrix=bt709',
        '-c:a','aac','-b:a','160k','-movflags','+faststart','-shortest',path],stdin=subprocess.PIPE)
pad=f'scale={TW}:{TH}:force_original_aspect_ratio=decrease:flags=lanczos:out_color_matrix=bt709:out_range=tv,pad={TW}:{TH}:(ow-iw)/2:(oh-ih)/2'
p1=pp(f'{FD}/{slug}_v1_final.mp4',sw,sh,pad); p2=pp(f'{FD}/{slug}_v1_side_by_side.mp4',2*sw,sh,f'scale=-2:{TH}:flags=lanczos:out_color_matrix=bt709:out_range=tv')
for i in range(N):
    S_cur=S[i]; o=dec(out[i]); f=cand_fill(o) if mode=='fill' else keepbars(cand_fit(o),S[i])
    p1.stdin.write(f.tobytes()); p2.stdin.write(np.hstack([S[i],f]).tobytes())
for p in (p1,p2): p.stdin.close(); p.wait()
json.dump({'slug':slug,'pieces':[acc[k]['id'] for k in P_],'mode':mode,'active':[int(ay0),int(ay1),int(ax0),int(ax1)],'bw_grade':False,'color_grade':False,'grain':False,'fps':fps,'frames':N,'version':'GJ V2'},open(f'{FD}/assemble.json','w'))
print('ok',slug,N)
