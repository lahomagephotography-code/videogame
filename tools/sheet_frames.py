# Cut the 12 walk frames out of reference/walk_sheet.webp (green screen), remove the green,
# and line them up: same ground line for the feet, body centred on the same column.
# Writes WORKDIR/sheet_frames.npy  (12 x H x W x 4, BGRA, sheet resolution)
import sys, cv2, numpy as np
S = sys.argv[1]
im = cv2.imread('reference/walk_sheet.webp')
b, g, r = [im[..., i].astype(np.float32) for i in range(3)]
# soft key: how much greener than the other channels
spill = g - np.maximum(r, b)
alpha = np.clip(1 - (spill - 40) / 50, 0, 1)                 # >90 greener = background
alpha[(g < 120)] = np.maximum(alpha[(g < 120)], (spill[(g < 120)] < 60).astype(np.float32))
fg = (alpha > 0.5).astype(np.uint8)
n, lab, st, cen = cv2.connectedComponentsWithStats(fg)
cells = sorted([i for i in range(1, n) if st[i, 4] > 3000], key=lambda i: (round(st[i, 1] / 300), st[i, 0]))
assert len(cells) == 12, len(cells)
# despill: pull green down to the brighter of red/blue
out = im.astype(np.float32).copy()
out[..., 1] = np.minimum(out[..., 1], np.maximum(out[..., 0], out[..., 2]) + 10)
frames, info = [], []
for i in cells:
    x, y, w, h = st[i, :4]
    x0, y0, x1, y1 = x - 6, y - 6, x + w + 6, y + h + 6
    m = np.zeros(fg.shape, np.uint8)
    # this frame = main body + any little bits (claw tips) inside its box
    sub = lab[y0:y1, x0:x1]; keep = np.isin(sub, [j for j in range(1, n) if j == i or (st[j, 4] < 3000 and x0 <= st[j, 0] and st[j, 0] + st[j, 2] <= x1 and y0 <= st[j, 1] and st[j, 1] + st[j, 3] <= y1)])
    a = (alpha[y0:y1, x0:x1] * keep * 255).astype(np.uint8)
    f = np.dstack([out[y0:y1, x0:x1].clip(0, 255).astype(np.uint8), a])
    ys, xs = np.where(a > 128)
    feet = ys.max()
    upper = ys < ys.min() + 0.55 * (feet - ys.min())          # rider + body, not the legs
    cx = xs[upper].mean()
    frames.append(f); info.append((feet, cx))
CW, CH = 400, 270; BASE = 258; MID = 215                     # common canvas, ground line, centre column
stack = np.zeros((12, CH, CW, 4), np.uint8)
for k, (f, (feet, cx)) in enumerate(zip(frames, info)):
    ox, oy = int(round(MID - cx)), int(BASE - feet)
    h, w = f.shape[:2]
    stack[k, oy:oy + h, ox:ox + w] = f
np.save(S + '/sheet_frames.npy', stack)
print('ok', [int(i[1]) for i in info])
