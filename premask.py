import sys,glob,cv2,numpy as np,json,os,pathlib
sys.path.insert(0, str(next(p/'lib' for p in pathlib.Path(__file__).resolve().parents if (p/'lib'/'home.py').is_file())))
from home import gj_home
from ultralytics import YOLO
slug=sys.argv[1]; W,H=1152,640
CAST={'lalaland':['Sebastian','Mia'],'pulp':['Vincent','Mia'],'thisisamerica':['Lead'],'walken':['Lead'],'singleladies':['Lead (center)']}[slug]
v=glob.glob(f'{slug}/*_native.mp4')[0]; det=YOLO(gj_home()+'/models/yolo11l-seg.pt')
c=cv2.VideoCapture(v); M=[]; stat=[]; thumbs=[]; i=0
def score(f,b,mk):  # higher = 'female/second' role
    x0,y0,x1,y1=map(int,b); lab=cv2.cvtColor(f,cv2.COLOR_BGR2LAB); sel=mk>0
    if slug=='lalaland': return lab[...,2][sel].mean()      # yellow dress
    return lab[...,0][sel].mean()                          # pulp: Mia white shirt vs black suit
while True:
    ok,f=c.read()
    if not ok: break
    f=cv2.resize(f,(W,H),interpolation=cv2.INTER_AREA)
    r=det(f,classes=[0],conf=0.2,imgsz=768,verbose=False)[0]
    P=[]
    if r.masks is not None:
        for b,poly in zip(r.boxes.xyxy.tolist(),r.masks.xy):
            if (b[2]-b[0])*(b[3]-b[1])<W*H*0.003 or len(poly)<3: continue
            mk=np.zeros((H,W),np.uint8); cv2.fillPoly(mk,[poly.astype(np.int32)],255); P.append((b,mk))
    P.sort(key=lambda p:-(p[0][2]-p[0][0])*(p[0][3]-p[0][1]))
    n=len(CAST)
    if slug=='singleladies':
        P3=sorted(P[:3],key=lambda p:(p[0][0]+p[0][2])/2); pick=[P3[1]] if len(P3)==3 else ([min(P3,key=lambda p:abs((p[0][0]+p[0][2])/2-W/2))] if P3 else [])
    elif n==1: pick=P[:1]
    else:
        two=P[:2]; 
        if len(two)==2: pick=sorted(two,key=lambda p:score(f,*p))
        else: pick=two+[None]
    fm=[(p[1] if p is not None else np.zeros((H,W),np.uint8)) for p in pick][:n]
    while len(fm)<n: fm.append(np.zeros((H,W),np.uint8))
    M.append(np.packbits(np.stack([cv2.resize(m,(W//2,H//2),interpolation=cv2.INTER_NEAREST)>0 for m in fm]),axis=-1)); stat.append([int(m.any()) for m in fm])
    if i%max(1,1)==0: pass
    if i%50==0: print(slug,i,flush=True)
    if i%24==0: thumbs.append((i,f,[p for p in pick if p is not None][:n] if n>1 else pick))
    i+=1
M=np.stack(M); np.savez_compressed(f'{slug}/premask_{slug}.npz',m=M,note='packbits along last axis, half-res 576x320, bool per role')
cov=np.array(stat).sum(0).tolist(); json.dump({'frames':len(M),'coverage_frames':cov,'cast_roles':CAST},open(f'{slug}/premask_stats.json','w'))
T=[t for t in thumbs if t[0]%24==0]; idx=np.linspace(0,len(T)-1,6).astype(int); tiles=[]
cols=[(0,0,255),(0,220,255)]
for k in idx:
    fi,f,pick=T[k]; o=f.copy()
    for nn,(b,mk) in enumerate(pick):
        ov=o.copy(); ov[mk>0]=cols[nn]; o=cv2.addWeighted(ov,0.35,o,0.65,0)
        x0,y0,x1,y1=map(int,b); cv2.rectangle(o,(x0,y0),(x1,y1),cols[nn],3)
        cv2.putText(o,f'{nn+1} {CAST[nn]}',(x0,max(30,y0-8)),0,0.9,(0,0,0),5); cv2.putText(o,f'{nn+1} {CAST[nn]}',(x0,max(30,y0-8)),0,0.9,cols[nn],2)
    cv2.putText(o,f'f{fi}',(15,625),0,1,(255,255,255),2); tiles.append(o)
cv2.imwrite(f'{slug}/{slug}_numbered_board.jpg',np.vstack([np.hstack(tiles[0:3]),np.hstack(tiles[3:6])]),[1,88])
print(slug,len(M),cov,flush=True)
