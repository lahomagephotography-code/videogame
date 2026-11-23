# Generic cut-out: GrabCut inside a rect, with optional colour rules, then downscale to game pixels.
# usage: cutout.py SRC NAME x0 y0 x1 y1 FACTOR [rules] [extra json]
import sys, json, cv2, numpy as np
src,name=sys.argv[1],sys.argv[2]; x0,y0,x1,y1=map(int,sys.argv[3:7]); F=float(sys.argv[7])
rules=sys.argv[8] if len(sys.argv)>8 else ''
extra=json.loads(sys.argv[9]) if len(sys.argv)>9 else {}
OUT=extra.get('out','/tmp/claude-0/-home-user-videogame/184d5145-a87d-56d3-adf5-417bc138e0c9/scratchpad/cut')
img=cv2.imread(src); hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV).astype(int); H,S,V=hsv[...,0],hsv[...,1],hsv[...,2]
m=np.full(img.shape[:2],cv2.GC_BGD,np.uint8); m[y0:y1,x0:x1]=cv2.GC_PR_FGD
ins=np.zeros(m.shape,bool); ins[y0:y1,x0:x1]=True
sky=(H>=95)&(H<=118)&(S>100)&(V>150)
green=(H>=30)&(H<=85)&(S>90)&(V>60)
if 'sky' in rules: m[ins&sky]=cv2.GC_BGD
if 'green' in rules: m[ins&green]=cv2.GC_BGD
if 'notsky' in rules: m[ins&~sky]=cv2.GC_FGD
for r in extra.get('bg',[]): m[r[1]:r[3],r[0]:r[2]]=cv2.GC_BGD
for r in extra.get('fg',[]): m[r[1]:r[3],r[0]:r[2]]=cv2.GC_FGD
if 'nogc' in rules: a=np.where((m==1)|(m==3),255,0).astype(np.uint8)
else:
    bg=np.zeros((1,65)); fg=np.zeros((1,65))
    cv2.grabCut(img,m,None,bg,fg,6,cv2.GC_INIT_WITH_MASK)
    a=np.where((m==1)|(m==3),255,0).astype(np.uint8)
if 'sky' in rules: a[sky]=0
a=cv2.morphologyEx(a,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
if 'largest' in rules:
    n,lab,st,_=cv2.connectedComponentsWithStats(a); a=np.where(lab==1+np.argmax(st[1:,4]),255,0).astype(np.uint8)
c=img[y0:y1,x0:x1].astype(np.float32); al=a[y0:y1,x0:x1].astype(np.float32)/255
w=max(1,int(round((x1-x0)/F))); h=max(1,int(round((y1-y0)/F)))
pm=np.dstack([c*al[...,None],al*255]); r=cv2.resize(pm,(w,h),interpolation=cv2.INTER_AREA)
A=r[...,3]/255; rgb=r[...,:3]/np.maximum(A[...,None],1e-3)
if 'sharp' in rules: rgb=rgb+0.6*(rgb-cv2.GaussianBlur(rgb,(0,0),0.8))
out=np.zeros((h,w,4),np.uint8); out[...,:3]=np.clip(rgb,0,255); out[...,3]=(A>0.5)*255
if 'outline' in rules:
    mm=(A>0.5).astype(np.uint8); ring=(cv2.dilate(mm,np.array([[0,1,0],[1,1,1],[0,1,0]],np.uint8))>0)&(mm==0)
    out[ring]=(28,10,26,255)
# trim transparent border
ys,xs=np.where(out[...,3]>0); out=out[ys.min():ys.max()+1, xs.min():xs.max()+1]
cv2.imwrite(OUT+'/'+name+'.png',out)
prev=img[y0:y1,x0:x1].copy(); prev[a[y0:y1,x0:x1]==0]=(255,0,255)
cv2.imwrite(OUT+'/'+name+'_prev.png',prev)
print(name,out.shape[1],'x',out.shape[0])
