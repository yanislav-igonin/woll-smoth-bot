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


FEATURES = _idx(fm.FACEMESH_LEFT_EYE | fm.FACEMESH_RIGHT_EYE | fm.FACEMESH_LEFT_EYEBROW
                | fm.FACEMESH_RIGHT_EYEBROW | fm.FACEMESH_LIPS | fm.FACEMESH_NOSE)
EYES = [_idx(fm.FACEMESH_LEFT_EYE), _idx(fm.FACEMESH_RIGHT_EYE)]
LIPS = _idx(fm.FACEMESH_LIPS)
OVAL = _idx(fm.FACEMESH_FACE_OVAL)

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


def tiny_face(img, lm, scale=0.55, grow=2.0):
    """Features shrink as one block toward their centre; skin around them stretches in, fading out
    over an ellipse `grow` times the feature size. A radial pinch can't fold, so turned heads don't
    tear the way a face-mesh warp did."""
    c, axes, ang = _ellipse(lm[FEATURES], grow)
    return pinch(img, c, axes, ang, scale, scale, 1 / grow)


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


def _paste(img, src, M, center, axes, ang, feather):
    """Blend `src` warped by affine M into img inside a soft ellipse; only the ellipse's box is touched."""
    h, w = img.shape[:2]
    pad = int(max(axes) + feather * 3) + 2
    x0, y0 = max(int(center[0]) - pad, 0), max(int(center[1]) - pad, 0)
    x1, y1 = min(int(center[0]) + pad, w), min(int(center[1]) + pad, h)
    if x1 <= x0 or y1 <= y0:
        return img
    M = M.copy()
    M[:, 2] -= (x0, y0)
    patch = cv2.warpAffine(src, M, (x1 - x0, y1 - y0), borderMode=cv2.BORDER_REFLECT)
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.ellipse(m, (int(center[0]) - x0, int(center[1]) - y0), (int(axes[0]), int(axes[1])), ang, 0, 360, 255, -1)
    a = cv2.GaussianBlur(m, (0, 0), max(feather, 1))[..., None] / 255.0
    img[y0:y1, x0:x1] = (patch * a + img[y0:y1, x0:x1] * (1 - a)).astype(np.uint8)
    return img


def _eye_r(lm, eye):
    return np.ptp(lm[eye][:, 0]) / 2


def thatcher(img, lm):
    """Eyes and mouth flipped upside down in place."""
    src = img.copy()
    for idx, grow in [(EYES[0], 1.9), (EYES[1], 1.9), (LIPS, 1.35)]:
        c, (a, b), ang = _ellipse(lm[idx], grow)
        img = _paste(img, src, cv2.getRotationMatrix2D(c, 180, 1.0), c, (a, b), ang, min(a, b) * 0.25)
    return img


def mouth_eyes(img, lm, size=2.0):
    """The mouth, scaled to `size` x eye width, pasted over both eyes."""
    src = img.copy()
    mc, (ma, mb), ang = _ellipse(lm[LIPS], 1.25)
    for eye in EYES:
        ec = lm[eye].mean(0)
        k = _eye_r(lm, eye) * size / max(ma, mb)
        M = cv2.getRotationMatrix2D(mc, 0, k)
        M[:, 2] += ec - mc
        img = _paste(img, src, M, ec, (ma * k, mb * k), ang, mb * k * 0.3)
    return img


def symmetry(img, lm):
    """Face rebuilt from its more visible half, mirrored across the midline (forehead to chin)."""
    top, bot = lm[10], lm[152]
    d = (bot - top) / np.linalg.norm(bot - top)
    n = np.array([-d[1], d[0]], np.float32)
    if ((lm[OVAL] - top) @ n).sum() < 0:  # n points to the wider (more visible) half; the other half gets replaced
        n = -n
    R = np.eye(2, dtype=np.float32) - 2 * np.outer(n, n)
    M = np.hstack([R, (top - R @ top)[:, None]]).astype(np.float32)
    size = np.ptp(lm[OVAL], 0).max()
    h, w = img.shape[:2]
    x, y, bw, bh = cv2.boundingRect(lm[OVAL].astype(np.float32))
    pad = int(size * 0.1)
    x0, y0, x1, y1 = max(x - pad, 0), max(y - pad, 0), min(x + bw + pad, w), min(y + bh + pad, h)
    M[:, 2] -= (x0, y0)
    mir = cv2.warpAffine(img, M, (x1 - x0, y1 - y0), borderMode=cv2.BORDER_REFLECT)
    ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    side = np.clip(-((xs - top[0]) * n[0] + (ys - top[1]) * n[1]) / (size * 0.04) + 0.5, 0, 1)
    face = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(face, cv2.convexHull(lm[OVAL].astype(np.int32)), 255)
    face = face[y0:y1, x0:x1]
    e = int(size * 0.04) * 2 + 1
    face = cv2.GaussianBlur(cv2.erode(face, np.ones((e, e), np.uint8)), (0, 0), size * 0.03) / 255.0
    a = (side * face)[..., None]
    img[y0:y1, x0:x1] = (mir * a + img[y0:y1, x0:x1] * (1 - a)).astype(np.uint8)
    return img


def big_eyes(img, lm, scale=2.0):
    for eye in EYES:
        c, (a, b), ang = _ellipse(lm[eye], 1.0)
        r = max(a, b) * 2.0
        img = pinch(img, c, (r, r), ang, scale, scale, 0.42)
    return img


def big_mouth(img, lm, scale=1.8):
    c, axes, ang = _ellipse(lm[LIPS], 1.5)
    return pinch(img, c, axes, ang, scale, scale, 0.5)


EFFECTS = {"tiny": tiny_face, "eyes": tiny_eyes, "squint": squint, "thatcher": thatcher,
           "moutheyes": mouth_eyes, "mirror": symmetry, "bigeyes": big_eyes, "bigmouth": big_mouth}


def apply(img, effect, found=None):
    found = faces(img) if found is None else found
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
