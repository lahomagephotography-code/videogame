# Export frames_hi.pkl (from sheet_variants.py) as game sprites: common crop, feet on the bottom row.
#   python3 tools/export_raptor.py WORKDIR OUTDIR [scale]
import sys, os, pickle, cv2, numpy as np
S, OUT = sys.argv[1], sys.argv[2]; SC = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
BASE, MID = 258, 215                                  # ground line and centre column used by sheet_frames.py
frames = pickle.load(open(S + '/frames_hi.pkl', 'rb'))
al = np.zeros(next(iter(frames.values())).shape[:2], np.uint8)
for f in frames.values(): al = np.maximum(al, f[..., 3])
ys, xs = np.where(al > 128)
x0, x1, y0, y1 = xs.min() - 2, xs.max() + 3, ys.min() - 2, BASE + 2
w, h = round((x1 - x0) * SC), round((y1 - y0) * SC)
for k, f in frames.items():
    c = f[y0:y1, x0:x1].astype(np.float32); a = c[..., 3:4] / 255
    r = cv2.resize(np.dstack([c[..., :3] * a, a[..., 0] * 255]), (w, h), interpolation=cv2.INTER_AREA)
    A = r[..., 3] / 255; rgb = r[..., :3] / np.maximum(A[..., None], 1e-3)
    rgb = rgb + 0.35 * (rgb - cv2.GaussianBlur(rgb, (0, 0), 0.7))
    o = np.zeros((h, w, 4), np.uint8); o[..., :3] = np.clip(rgb, 0, 255); o[..., 3] = (A > 0.5) * 255
    cv2.imwrite(os.path.join(OUT, k + '.webp'), o, [cv2.IMWRITE_WEBP_QUALITY, 92])
# geometry for the game (game units = sprite pixels / 2), relative to (playerX, feet line)
u = lambda px: px * SC / 2
print('size', w, h, '| anchorX (game units from sprite left) =', round(u(MID - x0), 1), '| bottom pad', round(u(y1 - BASE), 1))
