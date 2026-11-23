# Export the T-rex walk sheet frames (tools/sheet_frames.py ... trex_sheet) as game sprites:
#   trex_run_0..11 (walk) and trex_air (a tucked-leg frame with the tail swung up).
#   python3 tools/trex_sheet_export.py WORKDIR OUTDIR [scale]
import sys, os, cv2, numpy as np
S, OUT = sys.argv[1], sys.argv[2]; SC = float(sys.argv[3]) if len(sys.argv) > 3 else 0.68
BASE, MID = 258, 260
st = np.load(S + '/trex_sheet.npy'); N, Hh, Ww = st.shape[:3]
yy, xx = np.mgrid[0:Hh, 0:Ww].astype(np.float32)
air = st[5].copy()                                     # legs pulled up the most
w_tail = np.clip((170 - xx) / 150, 0, 1) ** 1.5
air = cv2.remap(air, xx, yy + 14 * w_tail, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
frames = {'trex_run_%d' % i: st[i] for i in range(N)}; frames['trex_air'] = air
al = np.max(np.stack([f[..., 3] for f in frames.values()]), 0)
ys, xs = np.where(al > 128); x0, x1, y0, y1 = xs.min() - 2, xs.max() + 3, ys.min() - 2, BASE + 2
w, h = round((x1 - x0) * SC), round((y1 - y0) * SC)
for k, f in frames.items():
    c = f[y0:y1, x0:x1].astype(np.float32); a = c[..., 3:4] / 255
    r = cv2.resize(np.dstack([c[..., :3] * a, a[..., 0] * 255]), (w, h), interpolation=cv2.INTER_AREA)
    A = r[..., 3] / 255; rgb = r[..., :3] / np.maximum(A[..., None], 1e-3)
    rgb = rgb + 0.3 * (rgb - cv2.GaussianBlur(rgb, (0, 0), 0.7))
    o = np.zeros((h, w, 4), np.uint8); o[..., :3] = np.clip(rgb, 0, 255); o[..., 3] = (A > 0.5) * 255
    cv2.imwrite(os.path.join(OUT, k + '.webp'), o, [cv2.IMWRITE_WEBP_QUALITY, 92])
# geometry in game units relative to (playerX = sheet column MID, feet line)
f0 = st[0][..., 3] > 128; ys0, xs0 = np.where(f0)
u = lambda px: round(px * SC / 2, 1)
head = ys0.min(); hx = xs0[ys0 < head + 15].mean()
snout = xs0[(ys0 > head + 30) & (ys0 < head + 110)].max()
print('size', w, h, '| anchorX', u(MID - x0), '| hitH', u(BASE - head), '| headX', u(hx - MID),
      '| snout x', u(snout - MID), '| tail end', u(xs0.min() - MID))
