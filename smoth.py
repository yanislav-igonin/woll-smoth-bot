"""Wol Smoth filters: classic landmark warps, no generative models."""
import os

import cv2
import mediapipe as mp
import numpy as np

fm = mp.solutions.face_mesh
_detector = cv2.FaceDetectorYN.create(os.path.join(os.path.dirname(__file__), "models", "yunet.onnx"), "", (320, 320), 0.7)
_mesh = fm.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)


def _idx(conns):
    return sorted({i for c in conns for i in c})


OVAL = _idx(fm.FACEMESH_FACE_OVAL)
FEATURES = _idx(fm.FACEMESH_LEFT_EYE | fm.FACEMESH_RIGHT_EYE | fm.FACEMESH_LEFT_EYEBROW
                | fm.FACEMESH_RIGHT_EYEBROW | fm.FACEMESH_LIPS | fm.FACEMESH_NOSE)
EYES = [_idx(fm.FACEMESH_LEFT_EYE), _idx(fm.FACEMESH_RIGHT_EYE)]

_adj = np.zeros((468, 468), np.float32)
for _a, _b in fm.FACEMESH_TESSELATION:
    _adj[_a, _b] = _adj[_b, _a] = 1
TRIANGLES = np.array(sorted({tuple(sorted((a, b, c))) for a in range(468) for b in np.nonzero(_adj[a])[0]
                             for c in np.nonzero(_adj[a] * _adj[b])[0]}))
_avg = _adj / np.maximum(_adj.sum(1, keepdims=True), 1)  # neighbour-mean operator


def faces(img):
    """Landmarks (468x2 px) for every face. YuNet finds faces of any size (group shots too),
    then Face Mesh runs on an upscaled crop of each one; the mesh model alone only sees close-ups."""
    h, w = img.shape[:2]
    k = min(1.0, 1600 / max(h, w))
    small = cv2.resize(img, None, fx=k, fy=k) if k < 1 else img
    _detector.setInputSize((small.shape[1], small.shape[0]))
    dets = _detector.detect(small)[1]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = []
    for d in [] if dets is None else dets:
        bx, by, bw, bh = d[:4] / k
        cx, cy, side = bx + bw / 2, by + bh / 2, max(bw, bh) * 1.8
        x0, y0 = int(max(cx - side / 2, 0)), int(max(cy - side / 2, 0))
        x1, y1 = int(min(cx + side / 2, w)), int(min(cy + side / 2, h))
        crop = rgb[y0:y1, x0:x1]
        if min(crop.shape[:2]) < 20:
            continue
        if crop.shape[0] < 256:  # mesh is unreliable on tiny crops
            crop = cv2.resize(crop, None, fx=256 / crop.shape[0], fy=256 / crop.shape[0], interpolation=cv2.INTER_CUBIC)
        res = _mesh.process(np.ascontiguousarray(crop))
        if not res.multi_face_landmarks:
            continue
        lm = res.multi_face_landmarks[0].landmark
        out.append(np.array([(x0 + p.x * (x1 - x0), y0 + p.y * (y1 - y0)) for p in lm[:468]], np.float32))
    return out


def pinch(img, center, axes, angle, sx, sy, inner):
    """Shrink content inside an ellipse toward `center`: by (sx, sy) at the core (radius <= inner),
    easing back to identity at the ellipse edge. Skin in between gets stretched, so no holes."""
    h, w = img.shape[:2]
    ax, ay = max(axes[0], 1), max(axes[1], 1)
    pad = int(max(ax, ay)) + 2
    x0, y0 = max(int(center[0]) - pad, 0), max(int(center[1]) - pad, 0)
    x1, y1 = min(int(center[0]) + pad, w), min(int(center[1]) + pad, h)
    ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    c, s = np.cos(np.radians(angle)), np.sin(np.radians(angle))
    dx, dy = xs - center[0], ys - center[1]
    u, v = (dx * c + dy * s) / ax, (-dx * s + dy * c) / ay  # ellipse-local, edge at r=1
    r = np.sqrt(u * u + v * v) + 1e-6
    # source radius per dest radius, per axis: core (src r<=inner) scaled by s, ring linearly back to r=1.
    # Monotonic by construction, so no folds/duplicated features.
    def src(k):
        a = inner * k
        return np.where(r < a, r / k, inner + (1 - inner) * (r - a) / (1 - a)) / r
    u, v = u * src(sx), v * src(sy)
    mx = center[0] + (u * ax * c - v * ay * s)
    my = center[1] + (u * ax * s + v * ay * c)
    roi = img[y0:y1, x0:x1]
    warped = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    m = (r < 1)[..., None]
    img[y0:y1, x0:x1] = np.where(m, warped, roi)
    return img


def _ellipse(pts, grow):
    (cx, cy), (a, b), ang = cv2.fitEllipse(pts)
    return (cx, cy), (a / 2 * grow, b / 2 * grow), ang


def _mesh_warp(img, src, dst):
    """Piecewise-affine warp of the face mesh (each triangle dst -> src) as a dense backward map.
    Pixels no triangle covers (rounding cracks, eye/mouth holes) get the map by normalized convolution,
    so there are no seams; outside the face the map is identity."""
    h, w = img.shape[:2]
    x, y, bw, bh = cv2.boundingRect(dst)
    pad = 4
    x0, y0, x1, y1 = max(x - pad, 0), max(y - pad, 0), min(x + bw + pad, w), min(y + bh + pad, h)
    gx, gy = np.meshgrid(np.arange(x0, x1, dtype=np.float32), np.arange(y0, y1, dtype=np.float32))
    disp = np.zeros((y1 - y0, x1 - x0, 2), np.float32)
    cov = np.zeros((y1 - y0, x1 - x0), np.float32)
    for t in TRIANGLES:
        d = dst[t] - [x0, y0]
        tx, ty, tw, th = cv2.boundingRect(d.astype(np.float32))
        tx, ty = max(tx, 0), max(ty, 0)
        sl = (slice(ty, ty + th + 1), slice(tx, tx + tw + 1))
        m = np.zeros(cov[sl].shape, np.uint8)
        if not m.size:
            continue
        cv2.fillConvexPoly(m, np.round(d - [tx, ty]).astype(np.int32), 1)
        mm = m.astype(bool)
        A = cv2.getAffineTransform(dst[t], src[t])
        px, py = gx[sl][mm], gy[sl][mm]
        disp[sl][mm] = np.stack([A[0, 0] * px + A[0, 1] * py + A[0, 2] - px,
                                 A[1, 0] * px + A[1, 1] * py + A[1, 2] - py], 1)
        cov[sl][mm] = 1
    face = np.zeros_like(cov, np.uint8)
    cv2.fillConvexPoly(face, cv2.convexHull(np.round(dst - [x0, y0]).astype(np.int32)), 1)
    holes = (face > 0) & (cov == 0)
    if holes.any():
        sig = max(bw, bh) * 0.03
        k = cv2.GaussianBlur(cov, (0, 0), sig) + 1e-6
        fill = cv2.GaussianBlur(disp * cov[..., None], (0, 0), sig) / k[..., None]
        disp[holes] = fill[holes]
    mx, my = gx + disp[..., 0], gy + disp[..., 1]
    img[y0:y1, x0:x1] = cv2.remap(img, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    return img


def tiny_face(img, lm, scale=0.5):
    """Liquify-style, like the original meme: features shrink as one block, the real skin of the face
    stretches in to fill the gap (mesh warp, so it keeps its texture), the face outline stays put."""
    h, w = img.shape[:2]
    c = lm[FEATURES].mean(0)
    hull = cv2.convexHull(lm[FEATURES])
    size = np.ptp(lm[OVAL], 0).max()
    inner = np.array([cv2.pointPolygonTest(hull, (float(p[0]), float(p[1])), True) > -size * 0.01 for p in lm])
    fixed = np.zeros(468, bool)
    fixed[OVAL] = True
    D = np.zeros((468, 2), np.float32)
    D[inner] = (lm[inner] - c) * (scale - 1)
    free = ~(inner | fixed)
    for _ in range(300):  # harmonic fill: displacement fades smoothly from the features to the outline
        D[free] = (_avg @ D)[free]
    src = img.copy()
    out = _mesh_warp(img, lm, (lm + D).astype(np.float32))
    # eyes/mouth interiors are holes in the mesh: paste the whole feature block, scaled the same way
    M = np.array([[scale, 0, c[0] * (1 - scale)], [0, scale, c[1] * (1 - scale)]], np.float32)
    feat = cv2.warpAffine(src, M, (w, h), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REFLECT)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(mask, cv2.transform(hull, M).astype(np.int32), 255)
    f = max(int(size * scale * 0.02), 1)
    a = cv2.GaussianBlur(cv2.erode(mask, np.ones((f * 2 + 1,) * 2, np.uint8)), (0, 0), f)[..., None] / 255.0
    return (feat * a + out * (1 - a)).astype(np.uint8)


def tiny_eyes(img, lm, scale=0.55):
    for eye in EYES:
        c, (a, b), ang = _ellipse(lm[eye], 1.0)
        # eye contour is a thin ellipse; give the warp room to ease out over the lids
        img = pinch(img, c, (max(a, b) * 1.4, max(a, b) * 1.4), ang, scale, scale, 0.5)
    return img


def squint(img, lm, scale=0.45):
    for eye in EYES:
        p = lm[eye]
        c = p.mean(0)
        v = p[p[:, 0].argmax()] - p[p[:, 0].argmin()]  # corner to corner
        ang = np.degrees(np.arctan2(v[1], v[0]))
        half = np.linalg.norm(v) / 2
        img = pinch(img, c, (half * 1.5, half * 1.1), ang, 1.0, scale, 0.35)
    return img


EFFECTS = {"tiny": tiny_face, "eyes": tiny_eyes, "squint": squint}


def apply(img, effect):
    found = faces(img)
    for lm in found:
        img = EFFECTS[effect](img, lm)
    return img, len(found)


if __name__ == "__main__":
    import sys
    src = cv2.imread(sys.argv[1])
    assert src is not None, sys.argv[1]
    for name in EFFECTS:
        out, n = apply(src.copy(), name)
        cv2.imwrite(f"{sys.argv[1].rsplit('.', 1)[0]}_{name}.jpg", out)
        print(name, n, "faces")
