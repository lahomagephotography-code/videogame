# Builds every raptor + rider animation frame at full reference resolution.
#   python3 tools/raptor_frames.py WORKDIR      (needs WORKDIR/raptor_mask.npy from cut_raptor.py)
# Output: WORKDIR/frames_hi.pkl, a dict name -> RGBA image (same size as reference 1).
#
# How the motion is made from the single reference picture:
#   * Both lower legs are cut into separate layers drawn behind the body. Each leg slides between a
#     "back" and a "forward" position, leaning with it, and lifts while it swings forward, so the
#     legs cross each other like a real stride (8-frame walk cycle).
#   * The tail is bent up/down and the dreadlocks bounce with a smooth warp (no seams).
#   * Jaw-open copies (for swallowing houses) rotate the lower jaw down.
#   * ESPRO copies recolour the rider's clothes into a pink suit and add a unicorn horn.
import sys, math, pickle, cv2, numpy as np

S = sys.argv[1]
img = cv2.imread('reference/ref1_raptor.png')
a = np.load(S + '/raptor_mask.npy')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(int)
sky = (hsv[..., 0] >= 95) & (hsv[..., 0] <= 118) & (hsv[..., 1] > 100) & (hsv[..., 2] > 150)
a[sky] = 0
a[495:540, 120:210] = 0                      # palm trunk remnant under the tail
cloud = (hsv[..., 1] < 45) & (hsv[..., 2] > 200); cloud[260:] = False; a[cloud] = 0
a = cv2.morphologyEx(a, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
n, lab, st, _ = cv2.connectedComponentsWithStats(a); a = np.where(lab == 1 + np.argmax(st[1:, 4]), 255, 0).astype(np.uint8)
n, lab, st, _ = cv2.connectedComponentsWithStats((a == 0).astype(np.uint8), connectivity=4)
for i in range(1, n):                         # fill small enclosed holes (light jeans / teeth)
    x, y, w, h, area = st[i]
    if area < 6000 and x > 0 and y > 0 and x + w < a.shape[1] and y + h < a.shape[0]: a[lab == i] = 255
rgba = np.dstack([img, a]); Hh, Ww = a.shape
GROUND = 791
yy, xx = np.mgrid[0:Hh, 0:Ww].astype(np.float32)

def poly(pts):
    m = np.zeros(a.shape, np.uint8); cv2.fillPoly(m, [np.array(pts, np.int32)], 255); return m
def layer(mask):
    p = rgba.copy(); p[..., 3] = np.minimum(p[..., 3], mask); return p
def over(dst, src):
    al = src[..., 3:4] / 255.0
    dst[..., :3] = (src[..., :3] * al + dst[..., :3] * (1 - al)).astype(np.uint8)
    dst[..., 3] = np.maximum(dst[..., 3], src[..., 3]); return dst
def remap(src, mx, my):
    return cv2.remap(src, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

# ---- legs: the lower legs are redrawn so they can stride and cross ---------------------
# The original lower legs are removed; the big rear thigh stays as the shared hip.
cut_legs = poly([(215, 600), (320, 615), (350, 648), (395, 688), (470, 690), (530, 668), (600, 668),
                 (700, 680), (765, 700), (765, 800), (215, 800)])
body0 = rgba.copy(); body0[..., 3][cut_legs > 0] = 0

HIP = np.array([450.0, 612.0])
L_THIGH, L_SHIN = 125.0, 100.0
PINK, PINK_LT, PINK_DK = (114, 35, 223), (150, 95, 252), (78, 18, 160)        # BGR, sampled from the art
STRIPE, OUTLINE, CLAW, CLAW_HI = (92, 36, 70), (22, 9, 18), (58, 52, 60), (205, 200, 215)

def capsule(c, p, q, w0, w1, col):
    p, q = np.array(p, float), np.array(q, float)
    d = q - p; n = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-6)
    pts = np.array([p + n * w0 / 2, q + n * w1 / 2, q - n * w1 / 2, p - n * w0 / 2], np.int32)
    cv2.fillPoly(c, [pts], col, cv2.LINE_AA)
    cv2.circle(c, tuple(p.astype(int)), int(w0 / 2), col, -1, cv2.LINE_AA)
    cv2.circle(c, tuple(q.astype(int)), int(w1 / 2), col, -1, cv2.LINE_AA)

def leg_points(toe_x, lift):
    T = np.array([toe_x, GROUND - 6 - lift])
    A = T + np.array([-48.0, -66.0])            # ankle: the long raptor foot slants up and back
    B = T + np.array([-6.0, -12.0])             # ball of the foot
    d = A - HIP; dist = float(np.clip(np.linalg.norm(d), abs(L_THIGH - L_SHIN) + 1, L_THIGH + L_SHIN - 1))
    a = math.acos(np.clip((L_THIGH ** 2 + dist ** 2 - L_SHIN ** 2) / (2 * L_THIGH * dist), -1, 1))
    base = math.atan2(d[1], d[0])
    K = HIP + L_THIGH * np.array([math.cos(base - a), math.sin(base - a)])  # knee bends forward
    return K, A, B, T

def draw_leg(toe_x, lift, darken):
    c = np.zeros((Hh, Ww, 3), np.uint8); m = np.zeros((Hh, Ww), np.uint8)
    K, A, B, T = leg_points(toe_x, lift)
    segs = [(HIP, K, 96, 66), (K, A, 60, 44), (A, B, 44, 36)]
    toes = [B + np.array(v) for v in ((58, 12), (50, 0), (38, -12))]
    for layer_col, grow in ((OUTLINE, 10), (PINK, 0)):
        for p, q, w0, w1 in segs: capsule(c, p, q, w0 + grow, w1 + grow, layer_col)
        for t in toes: capsule(c, B, t, 26 + grow, 18 + grow, layer_col)
        for p, q, w0, w1 in segs: capsule(m, p, q, w0 + 10, w1 + 10, 255)
        for t in toes: capsule(m, B, t, 36, 28, 255)
    # shading: darker back edge, lighter front edge, stripes across the shin
    for p, q, w0, w1 in segs[1:]:
        p2, q2 = np.array(p) + (-8, 4), np.array(q) + (-8, 4)
        capsule(c, p2, q2, w0 * 0.45, w1 * 0.45, PINK_DK)
        capsule(c, np.array(p) + (5, -2), np.array(q) + (5, -2), w0 * 0.25, w1 * 0.25, PINK_LT)
    for f in (0.3, 0.55, 0.8):
        s0 = K + (A - K) * f; dv = (A - K) / np.linalg.norm(A - K); nv = np.array([-dv[1], dv[0]])
        cv2.line(c, tuple((s0 - nv * 20).astype(int)), tuple((s0 + nv * 14 + dv * 10).astype(int)), STRIPE, 9, cv2.LINE_AA)
    for t in toes:                              # claws
        tip = t + np.array([22, 20])
        cv2.fillPoly(c, [np.array([t + (0, -11), tip, t + (-6, 11)], np.int32)], CLAW, cv2.LINE_AA)
        cv2.polylines(c, [np.array([t + (0, -11), tip, t + (-6, 11)], np.int32)], True, OUTLINE, 4, cv2.LINE_AA)
        cv2.line(c, tuple((t + (2, -4)).astype(int)), tuple((tip + (-4, -6)).astype(int)), CLAW_HI, 2, cv2.LINE_AA)
        cv2.fillPoly(m, [np.array([t + (0, -14), tip + (3, 3), t + (-8, 14)], np.int32)], 255)
    sickle = A + np.array([4, 6])               # the raised sickle claw
    cv2.fillPoly(c, [np.array([sickle, sickle + (26, -6), sickle + (8, 10)], np.int32)], CLAW, cv2.LINE_AA)
    cv2.fillPoly(m, [np.array([sickle + (-3, -3), sickle + (30, -8), sickle + (8, 14)], np.int32)], 255)
    if darken: c = (c * 0.8).astype(np.uint8)
    out = np.dstack([c, m]); out[..., 3][(c.sum(2) == 0)] = 0
    return out

def bend_tail(src, dy):
    """Swing the tail tip up (dy<0) or down (dy>0); the base stays attached."""
    w = np.clip((380 - xx) / 365, 0, 1) ** 1.6
    w[yy > 640] = 0
    return remap(src, xx, yy - dy * w)

def bounce_hair(src, dy):
    """Bounce the dreadlocks / ponytail; the face doesn't move."""
    w = np.clip((570 - xx) / 140, 0, 1) ** 1.2
    w[yy > 335] = 0
    return remap(src, xx, yy - dy * w)

def shift(src, dy):
    M = np.float32([[1, 0, 0], [0, 1, dy]])
    return cv2.warpAffine(src, M, (Ww, Hh), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))

# ---- ESPRO look: pink suit + unicorn horn ---------------------------------------
def pink_suit(src):
    out = src.copy()
    h = cv2.cvtColor(np.ascontiguousarray(src[..., :3]), cv2.COLOR_BGR2HSV).astype(np.float32)
    H_, S_, V_ = h[..., 0], h[..., 1], h[..., 2]
    region = (xx > 460) & (xx < 720) & (yy > 300) & (yy < 640) & (src[..., 3] > 0)
    jacket = region & (yy < 480) & (V_ < 95) & (S_ < 90)
    pants = region & (yy > 440) & (H_ >= 92) & (H_ <= 122) & (V_ > 135) & (S_ < 200)
    def ramp(mask, lo, hi, t):
        lo = np.array(lo, np.float32); hi = np.array(hi, np.float32)
        col = lo[None] + (hi - lo)[None] * np.clip(t[mask], 0, 1)[:, None]
        out[..., :3][mask] = col.astype(np.uint8)
    ramp(jacket, (80, 20, 120), (215, 110, 255), V_ / 95.0)          # BGR: deep magenta -> hot pink
    ramp(pants, (130, 60, 200), (235, 190, 255), (V_ - 135) / 120.0)  # lighter pink trousers
    return out

def draw_horn(src):
    out = src.copy()
    base_l, base_r, tip = np.array([958, 344]), np.array([1000, 340]), np.array([1022, 236])
    hb = np.zeros((Hh, Ww, 4), np.uint8)
    pts = np.array([base_l, base_r, tip], np.int32)
    cv2.fillPoly(hb, [pts], (235, 245, 255, 255))                       # pearly white
    for k in range(1, 7):                                               # gold spiral bands
        f = k / 7.0
        l = base_l + (tip - base_l) * f; r = base_r + (tip - base_r) * (f - 0.06)
        cv2.line(hb, tuple(l.astype(int)), tuple(r.astype(int)), (60, 190, 245, 255), 5)
    cv2.polylines(hb, [pts], True, (60, 16, 48, 255), 5)                 # dark outline
    cv2.line(hb, tuple((base_l + (tip - base_l) * 0.15 + [6, 0]).astype(int)), tuple((base_l + (tip - base_l) * 0.75 + [6, 0]).astype(int)), (255, 255, 255, 255), 3)
    return over(out, hb)

# ---- jaw (open for swallowing) ---------------------------------------------------
jaw = poly([(915, 452), (1000, 448), (1062, 472), (1062, 525), (990, 525), (905, 500)])
P_jaw = (938, 452)
def open_jaw(src, ang=-20):
    body = src.copy(); body[..., 3][jaw > 0] = 0
    piece = src.copy(); piece[..., 3] = np.minimum(piece[..., 3], jaw)
    M = cv2.getRotationMatrix2D(P_jaw, ang, 1.0)
    rp = cv2.warpAffine(piece, M, (Ww, Hh), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
    out = np.zeros_like(src)
    mid = M @ np.array([1000, 478, 1.0])
    cv2.fillPoly(out, [np.array([(936, 450), (1005, 462), (int(mid[0]), int(mid[1])), (955, 497)], np.int32)], (30, 20, 90, 255))
    out = over(out, body); return over(out, rp)

# ---- frame assembly ---------------------------------------------------------------
FRONT_X, BACK_X = 640.0, 300.0            # toe positions at the ends of a stride
def gait(ph):
    """Toe x and lift for a leg at phase ph: stance pushes the foot back, swing brings it forward lifted."""
    ph %= 2 * math.pi
    if ph < math.pi: return FRONT_X + (BACK_X - FRONT_X) * ph / math.pi, 0.0
    t = (ph - math.pi) / math.pi
    return BACK_X + (FRONT_X - BACK_X) * (0.5 - 0.5 * math.cos(t * math.pi)), 46 * math.sin(t * math.pi)

def frame(near, far, bob, tail_dy, hair_dy, espro):
    body = body0
    if espro: body = draw_horn(pink_suit(body))
    body = bounce_hair(body, hair_dy)
    body = bend_tail(body, tail_dy)
    body = shift(body, -bob)
    out = np.zeros_like(rgba)
    out = over(out, draw_leg(*far, darken=True))
    out = over(out, draw_leg(*near, darken=False))
    return over(out, body)

poses = {}
for i in range(8):                       # walk: the legs pass each other twice per cycle
    ph = i / 8 * 2 * math.pi
    poses['run_%d' % i] = dict(near=gait(ph), far=gait(ph + math.pi),
        bob=12 * abs(math.sin(ph)), tail_dy=10 * math.sin(2 * ph), hair_dy=12 * math.cos(2 * ph + 0.8))
poses['rise'] = dict(near=(580, 45), far=(360, 60), bob=0, tail_dy=55, hair_dy=18)
poses['peak'] = dict(near=(600, 60), far=(420, 70), bob=0, tail_dy=-10, hair_dy=0)
poses['fall'] = dict(near=(620, 40), far=(480, 55), bob=0, tail_dy=-55, hair_dy=-18)

frames = {}
for name, p in poses.items():
    for espro in (False, True):
        f = frame(espro=espro, **p)
        sfx = '_espro' if espro else ''
        frames['raptor_' + name + sfx] = f
        frames['raptor_' + name + '_open' + sfx] = open_jaw(f)
pickle.dump(frames, open(S + '/frames_hi.pkl', 'wb'))

# preview sheet
def comp(f):
    bg = np.full(f.shape[:2] + (3,), (255, 200, 120), np.uint8); al = f[..., 3:4] / 255
    return (f[..., :3] * al + bg * (1 - al)).astype(np.uint8)[190:800, 0:1110]
names = ['raptor_run_%d' % i for i in range(8)] + ['raptor_rise', 'raptor_peak', 'raptor_fall', 'raptor_run_0_open_espro']
tiles = [cv2.resize(comp(frames[k]), None, fx=0.28, fy=0.28) for k in names]
rows = [np.concatenate(tiles[i:i + 4], 1) for i in range(0, 12, 4)]
cv2.imwrite(S + '/frames_preview.png', np.concatenate(rows, 0))
print('frames', len(frames))
