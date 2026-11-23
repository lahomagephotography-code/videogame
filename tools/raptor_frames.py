# Builds raptor+rider animation frames (full reference resolution) from the cutout mask.
import sys, cv2, numpy as np, pickle
S=sys.argv[1]
img=cv2.imread('reference/ref1_raptor.png')
a=np.load(S+'/raptor_mask.npy')
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV).astype(int)
sky=(hsv[...,0]>=95)&(hsv[...,0]<=118)&(hsv[...,1]>100)&(hsv[...,2]>150)
a[sky]=0
a[495:540,120:210]=0
cloud=(hsv[...,1]<45)&(hsv[...,2]>200); cloud[260:]=False; a[cloud]=0                     # palm trunk remnant under tail
a=cv2.morphologyEx(a,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
n,lab,st,_=cv2.connectedComponentsWithStats(a); a=np.where(lab==1+np.argmax(st[1:,4]),255,0).astype(np.uint8)
rgba=np.dstack([img,a]); Hh,Ww=a.shape
GROUND=791
def over(dst,src):
    al=src[...,3:4]/255.0
    dst[...,:3]=(src[...,:3]*al+dst[...,:3]*(1-al)).astype(np.uint8); dst[...,3]=np.maximum(dst[...,3],src[...,3]); return dst
yy,xx=np.mgrid[0:Hh,0:Ww].astype(np.float32)
def legs_warp(src, back_dx, front_dx, back_lift, front_lift, bob):
    # back leg: shear below hip line y=630, front leg below y=680; x split ~ 510
    tb=np.clip((yy-630)/(GROUND-630),0,1)**1.3; tf=np.clip((yy-680)/(GROUND-680),0,1)**1.3
    wb=np.clip((515-xx)/15,0,1); wf=1-wb                        # which leg owns this column
    dx=wb*back_dx*tb + wf*front_dx*tf
    dy=wb*back_lift*tb + wf*front_lift*tf
    mx=xx-dx; my=yy+dy+bob*(1-np.maximum(wb*tb,wf*tf))         # bob the body, feet stay planted
    return cv2.remap(src,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0,0))
# lower jaw piece rotated about its hinge
jaw=np.zeros(a.shape,np.uint8); cv2.fillPoly(jaw,[np.array([(915,452),(1000,448),(1062,472),(1062,525),(990,525),(905,500)],np.int32)],255)
P_jaw=(938,452)
MOUTH=(30,20,90,255)
def with_jaw(src, ang):
    body=src.copy(); body[...,3][jaw>0]=0
    piece=src.copy(); piece[...,3]=np.minimum(piece[...,3],jaw)
    M=cv2.getRotationMatrix2D(P_jaw,ang,1.0)
    rp=cv2.warpAffine(piece,M,(Ww,Hh),flags=cv2.INTER_LINEAR,borderValue=(0,0,0,0))
    out=np.zeros_like(src)
    if ang<0:
        tip=M@np.array([1052,490,1.0])
        mid=M@np.array([1000,478,1.0])
        cv2.fillPoly(out,[np.array([(936,450),(1005,462),(int(mid[0]),int(mid[1])),(955,497)],np.int32)],MOUTH)
    out=over(out,body); out=over(out,rp); return out
specs={  # back_dx, front_dx, back_lift, front_lift, bob   (full-res pixels)
 'run_0':(0,0,0,0,0), 'run_1':(28,-22,30,0,-8), 'run_2':(56,-44,0,0,0), 'run_3':(28,-22,0,30,-8),
 'jump':(30,-10,75,85,0),
}
frames={}
for k,(bd,fd,bl,fl,bob) in specs.items():
    w=legs_warp(rgba,bd,fd,bl,fl,bob)
    frames['raptor_'+k]=w
    frames['raptor_'+(k.replace('run_','run_open_') if k!='jump' else 'jump_open')]=with_jaw(w,-20)
pickle.dump(frames,open(S+'/frames_hi.pkl','wb'))
x0,y0,x1,y1=150,560,800,800
sel=np.concatenate([np.concatenate([over(np.full((y1-y0,x1-x0,4),(255,200,120,255),np.uint8),frames['raptor_'+k][y0:y1,x0:x1]) for k in ks],1) for ks in (['run_0','run_1','run_2'],['run_3','jump','run_open_0'])],0)
cv2.imwrite(S+'/legs_frames.png',cv2.resize(sel[...,:3],None,fx=0.5,fy=0.5))
h=over(np.full((200,260,4),(255,200,120,255),np.uint8),frames['raptor_run_open_0'][380:580,850:1110]); h2=over(np.full((200,260,4),(255,200,120,255),np.uint8),frames['raptor_run_0'][380:580,850:1110])
cv2.imwrite(S+'/jaw.png',np.concatenate([h2,h],1)[...,:3])
