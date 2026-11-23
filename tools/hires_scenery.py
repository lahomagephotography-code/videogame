# Scenery at half the reference resolution: fern, skyline, ground tile and a stitched
# (image-quilted, wrap-around) mountains + forest backdrop built only from clean areas.
import sys, os, cv2, numpy as np
OUT = sys.argv[1]
# size of each asset relative to the original reference images (matches the zoomed-out mock-up)
SCALES = {'fern': 0.30, 'skyline': 0.32, 'ground_h': 79, 'ground': 0.45, 'backdrop': 0.345}
r1 = cv2.imread('reference/ref1_raptor.png'); r3 = cv2.imread('reference/ref3_sign.png')
def shrink_rgba(c, a, SC):
    c = c.astype(np.float32); a = a.astype(np.float32)
    w, h = round(c.shape[1] * SC), round(c.shape[0] * SC)
    r = cv2.resize(np.dstack([c * a[..., None], a * 255]), (w, h), interpolation=cv2.INTER_AREA)
    A = r[..., 3] / 255; rgb = r[..., :3] / np.maximum(A[..., None], 1e-3)
    o = np.zeros((h, w, 4), np.uint8); o[..., :3] = np.clip(rgb, 0, 255); o[..., 3] = (A > 0.5) * 255
    return o
def hsv(im): return cv2.cvtColor(im, cv2.COLOR_BGR2HSV).astype(int)

# ---- fern (ref1), blue forest behind it removed ------------------------------
x0, y0, x1, y1 = 868, 590, 1108, 780
h_ = hsv(r1)
m = np.full(r1.shape[:2], cv2.GC_BGD, np.uint8); m[y0:y1, x0:x1] = cv2.GC_PR_FGD
blue = (h_[..., 0] >= 88) & (h_[..., 0] <= 135) & (h_[..., 1] > 50)
ins = np.zeros(m.shape, bool); ins[y0:y1, x0:x1] = True
m[ins & blue] = cv2.GC_BGD; m[ins & (h_[..., 0] >= 28) & (h_[..., 0] <= 50) & (h_[..., 1] > 120) & (h_[..., 2] > 120)] = cv2.GC_FGD
bg = np.zeros((1, 65)); fg = np.zeros((1, 65)); cv2.grabCut(r1, m, None, bg, fg, 6, cv2.GC_INIT_WITH_MASK)
a = ((m == 1) | (m == 3)).astype(np.uint8); a[blue] = 0
a = cv2.morphologyEx(a, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
fern = shrink_rgba(r1[y0:y1, x0:x1], a[y0:y1, x0:x1], SCALES['fern'])
ys, xs = np.where(fern[..., 3] > 0); fern = fern[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
cv2.imwrite(OUT + '/fern.png', fern)

# ---- skyline (ref3): buildings are darker than the sky -------------------------
x0, y0, x1, y1 = 560, 300, 945, 425
a = (hsv(r3)[y0:y1, x0:x1, 2] < 238).astype(np.uint8)
a = cv2.morphologyEx(a, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
cv2.imwrite(OUT + '/skyline.png', shrink_rgba(r3[y0:y1, x0:x1], a, SCALES['skyline']))

# ---- ground tile: seamless period found in the clean stretch of ref1 -----------
g = r1[765:941].astype(np.float32)
best = None
for s in range(760, 860, 2):
    for P in range(560, 700, 2):
        if s + P > 1490: continue
        d = np.abs(g[:, s] - g[:, s + P]).mean() + np.abs(g[:, s + 1] - g[:, s + P + 1]).mean()
        if best is None or d < best[0]: best = (d, s, P)
_, s, P = best
tile = cv2.resize(g[:, s:s + P], (round(P * SCALES['ground']), SCALES['ground_h']), interpolation=cv2.INTER_AREA)
cv2.imwrite(OUT + '/ground.png', np.clip(tile, 0, 255).astype(np.uint8))

# ---- backdrop: quilt patches from clean regions into a wrap-around panorama ----
Y0, Y1 = 380, 768
src = [r1[Y0:Y1, 1105:1498].copy(), r3[Y0:Y1, 372:482].copy()]
for im in src:                      # paint out the little cloud with sky colour
    hs = hsv(im); cl = (hs[..., 1] < 60) & (hs[..., 2] > 200)
    im[cl] = (253, 165, 60)
PW, OV, TARGET = 100, 36, 2400
rng = np.random.default_rng(3)
cands = [(i, x) for i, im in enumerate(src) for x in range(0, im.shape[1] - PW + 1, 3)]
def patch(c): i, x = c; return src[i][:, x:x + PW].astype(np.float32)
def mincut(err):  # err: H x OV -> boolean mask "take new patch" (True right of seam)
    Hh, Ww = err.shape; E = err.copy(); back = np.zeros((Hh, Ww), int)
    for y in range(1, Hh):
        for x in range(Ww):
            lo, hi = max(0, x - 1), min(Ww, x + 2); j = lo + np.argmin(E[y - 1, lo:hi])
            E[y, x] += E[y - 1, j]; back[y, x] = j
    mask = np.zeros((Hh, Ww), bool); x = int(np.argmin(E[-1]))
    for y in range(Hh - 1, -1, -1): mask[y, x:] = True; x = back[y, x]
    return mask
pano = patch(cands[rng.integers(len(cands))])
first = pano.copy()
while pano.shape[1] < TARGET + OV:
    left = pano[:, -OV:]
    sample = [cands[k] for k in rng.choice(len(cands), min(60, len(cands)), replace=False)]
    errs = [((patch(c)[:, :OV] - left) ** 2).sum() for c in sample]
    order = np.argsort(errs)[:5]; c = sample[order[rng.integers(len(order))]]
    p = patch(c); m = mincut(((p[:, :OV] - left) ** 2).sum(2))
    blend = np.where(m[..., None], p[:, :OV], left)
    pano = np.concatenate([pano[:, :-OV], blend, p[:, OV:]], 1)
# wrap: make the end flow into the start
L = pano.shape[1] - OV
tail = pano[:, L:]; head = pano[:, :OV]
m = mincut(((head - tail) ** 2).sum(2))
pano[:, :OV] = np.where(m[..., None], head, tail)
pano = pano[:, :L]
pano = np.clip(pano, 0, 255).astype(np.uint8)
hs = hsv(pano); sky = (hs[..., 2] > 238) & (hs[..., 0] >= 95) & (hs[..., 0] <= 118)
a = (~sky).astype(np.uint8)
for x in range(a.shape[1]):
    col = np.where(a[:, x] > 0)[0]
    if len(col): a[col.min():, x] = 1
cv2.imwrite(OUT + '/backdrop.png', shrink_rgba(pano, a, SCALES['backdrop']))
print('fern', fern.shape[1], fern.shape[0], 'ground', tile.shape[1], tile.shape[0], 'backdrop', round(L * SCALES['backdrop']))
