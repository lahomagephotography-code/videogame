# Export the raptor+rider frames at half the reference resolution (keeps the reference detail).
# Needs the full-res frames produced by raptor_frames.py (frames_hi.pkl).
import sys, os, cv2, numpy as np, pickle
S, OUT = sys.argv[1], sys.argv[2]
SCALE = float(sys.argv[3]) if len(sys.argv) > 3 else 0.459
frames = pickle.load(open(S + '/frames_hi.pkl', 'rb'))
al = np.zeros(next(iter(frames.values())).shape[:2], np.uint8)
for f in frames.values(): al = np.maximum(al, f[..., 3])
ys, xs = np.where(al > 128)
x0, x1, y0, y1 = xs.min() - 2, xs.max() + 3, ys.min() - 2, 792
w, h = round((x1 - x0) * SCALE), round((y1 - y0) * SCALE)
for k, f in frames.items():
    c = f[y0:y1, x0:x1].astype(np.float32); a = c[..., 3:4] / 255
    r = cv2.resize(np.dstack([c[..., :3] * a, a[..., 0] * 255]), (w, h), interpolation=cv2.INTER_AREA)
    A = r[..., 3] / 255; rgb = r[..., :3] / np.maximum(A[..., None], 1e-3)
    if SCALE < 0.35: rgb = rgb + 0.4 * (rgb - cv2.GaussianBlur(rgb, (0, 0), 0.7))   # keep it crisp when shrunk a lot
    out = np.zeros((h, w, 4), np.uint8); out[..., :3] = np.clip(rgb, 0, 255); out[..., 3] = (A > 0.5) * 255
    cv2.imwrite(os.path.join(OUT, k + '.webp'), out, [cv2.IMWRITE_WEBP_QUALITY, 92])
print('frame size', w, h, 'origin', x0, y0)
