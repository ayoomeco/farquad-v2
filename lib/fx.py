import numpy as np, cv2, os
FH=os.environ.get('FARQUAD_HOME',os.path.expanduser('~/farquad'))
YU=FH+'/models/face_detection_yunet_2023mar.onnx'; SF=FH+'/models/face_recognition_sface_2021dec.onnx'
det=cv2.FaceDetectorYN.create(YU,'',(320,320),0.6); rec=cv2.FaceRecognizerSF.create(SF,'')
def unit(v): return v/(np.linalg.norm(v)+1e-9)
def faces(bgr,up=1.0):
    im=cv2.resize(bgr,None,fx=up,fy=up) if up!=1 else bgr
    det.setInputSize((im.shape[1],im.shape[0])); _,f=det.detect(im); out=[]
    for r in (f if f is not None else []):
        r=r.copy(); r[:14]/=up; out.append((r,unit(rec.feature(rec.alignCrop(bgr,r)).flatten())))
    return out
def frames(p,sz=None,n=None):
    c=cv2.VideoCapture(p); o=[]
    while True:
        ok,f=c.read()
        if not ok or (n and len(o)>=n): return o
        o.append(cv2.resize(f,sz) if sz else f)
