import cv2,sys,numpy as np
p=sys.argv[1]; c=cv2.VideoCapture(p); n=int(c.get(7)); idx=np.linspace(0,n-1,10).astype(int); T=[]
for i in idx:
    c.set(1,int(i)); ok,f=c.read(); T.append(cv2.resize(f,(768,216)))
cv2.imwrite(sys.argv[2],np.vstack(T),[cv2.IMWRITE_JPEG_QUALITY,75])
