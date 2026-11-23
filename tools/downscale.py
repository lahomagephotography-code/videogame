# Downscale full-res RGBA frames to native game pixels with a shared palette + crisp outline.
import sys, cv2, numpy as np, pickle
S=sys.argv[1]; F=float(sys.argv[2]); mode=sys.argv[3] if len(sys.argv)>3 else 'area'
frames=pickle.load(open(S+'/frames_hi.pkl','rb'))
al=np.zeros(next(iter(frames.values())).shape[:2],np.uint8)
for f in frames.values(): al=np.maximum(al,f[...,3])
ys,xs=np.where(al>128); x0,x1,y0,y1=xs.min()-8,xs.max()+9,ys.min()-8,792
w=int(round((x1-x0)/F)); h=int(round((y1-y0)/F))
small={}
for k,f in frames.items():
    c=f[y0:y1,x0:x1].astype(np.float32); a=c[...,3:4]/255
    pm=np.dstack([c[...,:3]*a,a[...,0]*255])
    r=cv2.resize(pm,(w,h),interpolation=cv2.INTER_AREA)
    A=r[...,3]/255; rgb=r[...,:3]/np.maximum(A[...,None],1e-3)
    bl=cv2.GaussianBlur(rgb,(0,0),0.8); rgb=rgb+0.7*(rgb-bl)
    small[k]=(np.clip(rgb,0,255),A>0.5)
# shared palette via k-means
px=np.concatenate([v[0][v[1]] for v in small.values()]).astype(np.float32)
K=int(sys.argv[4]) if len(sys.argv)>4 else 40
_,lab,cent=cv2.kmeans(px,K,None,(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,30,0.5),3,cv2.KMEANS_PP_CENTERS)
cent=cent.astype(np.float32)
out={}
for k,(rgb,m) in small.items():
    d=((rgb[...,None,:]-cent[None,None])**2).sum(-1); q=cent[d.argmin(-1)].astype(np.uint8)
    img=np.zeros((h,w,4),np.uint8); img[...,:3]=q; img[...,3]=m*255
    # crisp 1px outline outside the silhouette
    mm=m.astype(np.uint8); dil=cv2.dilate(mm,np.array([[0,1,0],[1,1,1],[0,1,0]],np.uint8))
    ring=(dil>0)&(mm==0); img[ring]=(28,10,26,255)
    out[k]=img
pickle.dump(out,open(S+'/frames_small.pkl','wb'))
sheet=np.concatenate([np.concatenate([out[k] for k in ks],1) for ks in (['raptor_run_0','raptor_run_1','raptor_run_2'],['raptor_run_3','raptor_jump','raptor_run_open_0'])],0)
bg=np.full_like(sheet,(255,200,120,255)); a=sheet[...,3:4]/255; bg[...,:3]=(sheet[...,:3]*a+bg[...,:3]*(1-a))
cv2.imwrite(S+'/small_sheet.png',cv2.resize(bg[...,:3],None,fx=3,fy=3,interpolation=cv2.INTER_NEAREST))
print('native size',w,h)
