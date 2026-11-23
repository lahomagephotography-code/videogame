# Cut the pink T-rex + rider (reference/ref5_trex.png) onto transparency -> WORKDIR/trex_rgba.npy
import sys, cv2, numpy as np
S = sys.argv[1]
img = cv2.imread('reference/ref5_trex.png')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(int); H, Sa, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
X0, Y0, X1, Y1 = 205, 215, 1335, 800
m = np.full(img.shape[:2], cv2.GC_BGD, np.uint8); m[Y0:Y1, X0:X1] = cv2.GC_PR_FGD
ins = np.zeros(m.shape, bool); ins[Y0:Y1, X0:X1] = True
blue_green = (H >= 35) & (H <= 127) & (Sa > 45)            # sky, skyline, forest, grass
cloud = (Sa < 40) & (V > 215); cloud[360:600, 1120:1335] = False   # keep the teeth
m[ins & (blue_green | cloud)] = cv2.GC_BGD
for x0, y0, x1, y1 in [(420, 728, 528, 800), (660, 728, 770, 800), (1000, 745, 1095, 800),   # dust puffs
                       (205, 790, 1335, 800)]:                                                # grass line
    m[y0:y1, x0:x1] = cv2.GC_BGD
pink = ((H >= 145) | (H <= 6)) & (Sa > 80) & (V > 110)
m[ins & pink] = cv2.GC_FGD
m[300:560, 800:900] = cv2.GC_FGD                            # rider's body core
claw = (V < 140) & (Sa < 60); m[ins & claw & (np.mgrid[0:img.shape[0], 0:img.shape[1]][0] > 700)] = cv2.GC_PR_FGD
bg = np.zeros((1, 65)); fg = np.zeros((1, 65))
cv2.grabCut(img, m, None, bg, fg, 6, cv2.GC_INIT_WITH_MASK)
a = np.where((m == 1) | (m == 3), 255, 0).astype(np.uint8)
a[ins & blue_green & ~pink] = 0
a = cv2.morphologyEx(a, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
n, lab, st, _ = cv2.connectedComponentsWithStats(a); a = np.where(lab == 1 + np.argmax(st[1:, 4]), 255, 0).astype(np.uint8)
n, lab, st, _ = cv2.connectedComponentsWithStats((a == 0).astype(np.uint8), connectivity=4)
for i in range(1, n):                                       # fill small enclosed holes
    x, y, w, h, area = st[i]
    if area < 2500 and x > 0 and y > 0: a[lab == i] = 255
# put back the teeth/mouth (nothing behind the open mouth but the T-rex itself)
mouth = np.zeros(a.shape, bool); mouth[440:565, 1150:1300] = True
a[mouth & ~blue_green] = 255
# put back the dark claws that touch the grass, grown out from the toes
claw = (V < 215) & (Sa < 70) & ~blue_green
for _ in range(22):
    grow = cv2.dilate(a, np.ones((3, 3), np.uint8)) > 0
    add = grow & claw & (np.mgrid[0:a.shape[0], 0:a.shape[1]][0] > 690) & (a == 0)
    if not add.any(): break
    a[add] = 255
a = cv2.morphologyEx(a, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
np.save(S + '/trex_rgba.npy', np.dstack([img, a]))
vis = img.copy(); vis[a == 0] = (255, 0, 255)
cv2.imwrite(S + '/trex_cut.png', cv2.resize(vis[200:810, 190:1350], None, fx=0.6, fy=0.6))
