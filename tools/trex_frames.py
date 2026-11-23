# T-rex power-up frames from the cut-out (tools/cut_trex.py): 8-frame stride + airborne pose.
#   python3 tools/trex_frames.py WORKDIR OUTDIR [scale]
import sys, os, math, cv2, numpy as np
S, OUT = sys.argv[1], sys.argv[2]; SC = float(sys.argv[3]) if len(sys.argv) > 3 else 0.27
src = np.load(S + '/trex_rgba.npy')
Hh, Ww = src.shape[:2]; yy, xx = np.mgrid[0:Hh, 0:Ww].astype(np.float32)
GROUND = 795
def remap(f, mx, my): return cv2.remap(f, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
def frame(ph, air=False):
    # which leg owns a column: back leg left of 770, front leg right of it
    wb = np.clip((785 - xx) / 30, 0, 1); wf = 1 - wb
    tb = np.clip((yy - 600) / (GROUND - 600), 0, 1) ** 1.3      # back leg bends below its hip
    tf = np.clip((yy - 640) / (GROUND - 640), 0, 1) ** 1.3      # front leg below its knee
    if air:
        dxb, dxf, lb, lf, bob = 30, -20, 45, 55, 0
    else:
        u = math.cos(ph)                                     # stride: back leg forward when u>0
        dxb, dxf = 55 * u, -55 * u
        lb, lf = 38 * max(0, math.sin(ph)), 38 * max(0, -math.sin(ph))
        bob = 12 * abs(math.sin(ph))
    dx = wb * dxb * tb + wf * dxf * tf
    dy = wb * lb * tb + wf * lf * tf
    tail = np.clip((470 - xx) / 260, 0, 1) ** 1.5 * (14 * math.sin(2 * ph) if not air else 30)
    hair = np.where(yy < 345, np.clip((830 - xx) / 90, 0, 1), 0) * (10 * math.cos(2 * ph + 0.8) if not air else 14)
    body = 1 - np.maximum(wb * tb, wf * tf)                  # feet stay planted, body bobs
    return remap(src, xx - dx, yy + dy + bob * body - tail - hair)
frames = {'trex_run_%d' % i: frame(i / 8 * 2 * math.pi) for i in range(8)}
frames['trex_air'] = frame(0, air=True)
al = np.zeros((Hh, Ww), np.uint8)
for f in frames.values(): al = np.maximum(al, f[..., 3])
ys, xs = np.where(al > 128); x0, x1, y0 = xs.min() - 3, xs.max() + 4, ys.min() - 3; y1 = GROUND + 2
w, h = round((x1 - x0) * SC), round((y1 - y0) * SC)
for k, f in frames.items():
    c = f[y0:y1, x0:x1].astype(np.float32); a = c[..., 3:4] / 255
    r = cv2.resize(np.dstack([c[..., :3] * a, a[..., 0] * 255]), (w, h), interpolation=cv2.INTER_AREA)
    A = r[..., 3] / 255; rgb = r[..., :3] / np.maximum(A[..., None], 1e-3)
    rgb = rgb + 0.35 * (rgb - cv2.GaussianBlur(rgb, (0, 0), 0.7))
    o = np.zeros((h, w, 4), np.uint8); o[..., :3] = np.clip(rgb, 0, 255); o[..., 3] = (A > 0.5) * 255
    cv2.imwrite(os.path.join(OUT, k + '.webp'), o, [cv2.IMWRITE_WEBP_QUALITY, 92])
u = lambda px: round(px * SC / 2, 1)          # reference pixels -> game units
print('size', w, h, 'units', u(x1 - x0), u(y1 - y0))
print('geometry (units from sprite left / from feet): body centre x', u(770 - x0), 'mouth', u(1280 - x0), u(500 - GROUND),
      'rear', u(470 - x0), u(560 - GROUND), 'head top', u(y0 - GROUND), 'feet', u(560 - x0), u(960 - x0))
