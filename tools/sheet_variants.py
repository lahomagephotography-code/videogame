# Build every raptor sprite from the 12 aligned walk frames (tools/sheet_frames.py):
#   raptor_run_0..11            walk cycle (your sprite sheet)
#   raptor_rise/peak/fall       jump poses, made from the walk frames with the tail and hair swung
#   + '_open'                   lower jaw dropped (swallowing a house)
#   + '_espro'                  pink suit + unicorn horn (ESPRO power-up)
# Writes WORKDIR/frames_hi.pkl (sheet resolution) for tools/hires_raptor.py-style export.
#   python3 tools/sheet_variants.py WORKDIR
import sys, pickle, cv2, numpy as np

S = sys.argv[1]
stack = np.load(S + '/sheet_frames.npy')            # 12 x 270 x 400 x 4 (BGRA)
N, Hh, Ww = stack.shape[:3]
yy, xx = np.mgrid[0:Hh, 0:Ww].astype(np.float32)

def over(dst, src):
    al = src[..., 3:4] / 255.0
    dst[..., :3] = (src[..., :3] * al + dst[..., :3] * (1 - al)).astype(np.uint8)
    dst[..., 3] = np.maximum(dst[..., 3], src[..., 3]); return dst
def remap(src, mx, my):
    return cv2.remap(src, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

def snout(f):
    """Tip of the raptor's snout: right-most solid pixel in the head band. Everything on the head follows it."""
    a = f[..., 3] > 128
    band = a[90:190]
    xs = np.where(band.any(0))[0]; x = xs.max()
    ys = np.where(band[:, x - 2:x + 1].any(1))[0]
    return np.array([float(x), 90.0 + ys.mean()])
REF_TIP = snout(stack[0])

def head_top(f):
    ys = np.where((f[..., 3] > 128)[:, 150:240].any(1))[0]; return ys.min()
REF_TOP = head_top(stack[0])

# ---- jaw: rotate the lower jaw down around its hinge -----------------------------
JAW = np.array([(-58, 12), (-15, 14), (4, 17), (4, 46), (-45, 48), (-62, 30)], np.float32)
HINGE = np.array([-56.0, 14.0])
def open_jaw(f, ang=-22):
    tip = snout(f)
    jm = np.zeros((Hh, Ww), np.uint8); cv2.fillPoly(jm, [(JAW + tip).astype(np.int32)], 255)
    body = f.copy(); body[..., 3][jm > 0] = 0
    piece = f.copy(); piece[..., 3] = np.minimum(piece[..., 3], jm)
    piv = tuple((HINGE + tip).tolist())
    M = cv2.getRotationMatrix2D(piv, ang, 1.0)
    rp = cv2.warpAffine(piece, M, (Ww, Hh), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
    out = np.zeros_like(f)
    mid = M @ np.array([tip[0] - 20, tip[1] + 20, 1.0])
    mouth = np.array([HINGE + tip + (-2, -3), tip + (-18, 12), mid, HINGE + tip + (4, 18)], np.int32)
    cv2.fillPoly(out, [mouth], (30, 20, 90, 255))
    out = over(out, body); return over(out, rp)

# ---- ESPRO look -----------------------------------------------------------------------
def pink_suit(f):
    out = f.copy(); dy = head_top(f) - REF_TOP
    h = cv2.cvtColor(np.ascontiguousarray(f[..., :3]), cv2.COLOR_BGR2HSV).astype(np.float32)
    H_, S_, V_ = h[..., 0], h[..., 1], h[..., 2]
    solid = f[..., 3] > 128
    region = solid & (xx > 150) & (xx < 250)
    shirt = region & (yy > 78 + dy) & (yy < 140 + dy) & (V_ > 18) & (V_ < 115) & (S_ < 95) & ~((H_ > 5) & (H_ < 25) & (S_ > 60))
    jeans = region & (yy > 120 + dy) & (yy < 185 + dy) & (H_ >= 90) & (H_ <= 125) & (V_ > 105)
    def ramp(mask, lo, hi, t):
        lo = np.array(lo, np.float32); hi = np.array(hi, np.float32)
        out[..., :3][mask] = (lo[None] + (hi - lo)[None] * np.clip(t[mask], 0, 1)[:, None]).astype(np.uint8)
    ramp(shirt, (80, 20, 120), (215, 110, 255), (V_ - 15) / 95.0)        # BGR: hot-pink jacket
    ramp(jeans, (130, 60, 200), (235, 190, 255), (V_ - 105) / 140.0)      # light pink trousers
    return out

HORN = np.array([(-62, -13), (-44, -16), (-36, -60)], np.float32)
def draw_horn(f):
    tip = snout(f); bl, br, ht = HORN + tip
    hb = np.zeros_like(f)
    cv2.fillPoly(hb, [np.array([bl, br, ht], np.int32)], (235, 245, 255, 255))
    for k in range(1, 6):
        t = k / 6.0
        l = bl + (ht - bl) * t; r = br + (ht - br) * (t - 0.08)
        cv2.line(hb, tuple(np.round(l).astype(int)), tuple(np.round(r).astype(int)), (60, 190, 245, 255), 2)
    cv2.polylines(hb, [np.array([bl, br, ht], np.int32)], True, (40, 10, 30, 255), 2)
    out = f.copy(); return over(out, hb)

# ---- tail / hair swing for the jump poses ---------------------------------------------
def bend_tail(f, dy):
    w = np.clip((150 - xx) / 140, 0, 1) ** 1.5
    return remap(f, xx, yy - dy * w)
def bounce_hair(f, dy):
    w = np.clip((200 - xx) / 60, 0, 1); w[yy > 90] = 0
    return remap(f, xx, yy - dy * w)

frames = {}
def add(name, f):
    for espro in (False, True):
        g = draw_horn(pink_suit(f)) if espro else f
        sfx = '_espro' if espro else ''
        frames['raptor_' + name + sfx] = g
        frames['raptor_' + name + '_open' + sfx] = open_jaw(g)
for i in range(N): add('run_%d' % i, stack[i])
add('rise', bounce_hair(bend_tail(stack[3], 10), 4))      # legs passing, tail trailing low
add('peak', bend_tail(stack[9], -2))
add('fall', bounce_hair(bend_tail(stack[6], -10), -4))    # legs reaching, tail up
pickle.dump(frames, open(S + '/frames_hi.pkl', 'wb'))

def comp(f):
    bg = np.full(f.shape[:2] + (3,), (253, 165, 60), np.uint8); al = f[..., 3:4] / 255
    return (f[..., :3] * al + bg * (1 - al)).astype(np.uint8)
names = ['raptor_run_0', 'raptor_run_0_open', 'raptor_run_5_espro', 'raptor_run_8_open_espro', 'raptor_rise', 'raptor_fall']
cv2.imwrite(S + '/variants.png', np.concatenate([np.concatenate([comp(frames[k]) for k in names[i:i + 3]], 1) for i in (0, 3)], 0))
print('frames', len(frames))
