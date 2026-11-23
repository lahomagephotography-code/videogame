import sys, cv2, numpy as np
S=sys.argv[1]
img=cv2.imread('reference/ref1_raptor.png')
OX,OY=0,190                      # grid crop offset used when reading coords
X0,Y0,X1,Y1=5,200,1100,800
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV).astype(int)   # H 0-179, S,V 0-255
H,Sa,V=hsv[...,0],hsv[...,1],hsv[...,2]
m=np.full(img.shape[:2], cv2.GC_BGD, np.uint8)
m[Y0:Y1,X0:X1]=cv2.GC_PR_FGD
def rect(x0,y0,x1,y1,v): m[y0+OY:y1+OY, x0+OX:x1+OX]=v
sky=(H>=95)&(H<=115)&(Sa>120)&(V>170)
green=(H>=30)&(H<=85)&(Sa>90)&(V>70)
pink=((H>=150)|(H<=5))&(Sa>90)&(V>120)
inrect=np.zeros_like(sky); inrect[Y0:Y1,X0:X1]=True
m[inrect&(sky|green)]=cv2.GC_BGD
# hand-marked background: dust, palm trunks, gap between legs
rect(0,470,228,600,cv2.GC_BGD)      # dust left of back foot
rect(405,520,560,600,cv2.GC_BGD)    # dust right of back foot
rect(125,0,205,262,cv2.GC_BGD); rect(125,348,205,600,cv2.GC_BGD)   # palm trunk 1
rect(215,180,270,262,cv2.GC_BGD); rect(215,350,272,520,cv2.GC_BGD)  # palm trunk 2
rect(0,0,420,250,cv2.GC_BGD)        # sky/palm tops
rect(430,470,545,575,cv2.GC_BGD)    # between legs
rect(660,0,1110,120,cv2.GC_BGD)     # sky right of rider head
rect(890,430,1110,600,cv2.GC_BGD)   # bush right of arms
m[inrect&pink]=cv2.GC_FGD
# definite foreground cores: rider + saddle
rect(500,40,640,140,cv2.GC_FGD); rect(480,140,620,260,cv2.GC_FGD); rect(500,260,640,370,cv2.GC_FGD)
rect(545,380,640,440,cv2.GC_FGD)
bg=np.zeros((1,65)); fg=np.zeros((1,65))
cv2.grabCut(img,m,None,bg,fg,8,cv2.GC_INIT_WITH_MASK)
a=np.where((m==1)|(m==3),255,0).astype(np.uint8)
a=cv2.morphologyEx(a,cv2.MORPH_OPEN,np.ones((5,5),np.uint8))
n,lab,st,_=cv2.connectedComponentsWithStats(a)
big=1+np.argmax(st[1:,cv2.CC_STAT_AREA]); a=np.where(lab==big,255,0).astype(np.uint8)
# fill holes
inv=255-a; n,lab,st,_=cv2.connectedComponentsWithStats(inv)
for i in range(1,n):
    if st[i,cv2.CC_STAT_AREA]<1500 and st[i,0]>0: a[lab==i]=255
np.save(S+'/raptor_mask.npy',a)
vis=img.copy(); vis[a==0]=(0,0,0)
cv2.imwrite(S+'/raptor_cut.png', vis[Y0-10:Y1, X0:X1])
