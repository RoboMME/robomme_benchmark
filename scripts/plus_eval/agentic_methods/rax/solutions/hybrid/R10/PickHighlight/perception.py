"""Cube / highlight / button detection in RGB images (numpy + cv2)."""
import numpy as np
import cv2

def hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

_GEOM = {}

def set_geometry(meta, depth, cam, deproject_fn):
    """Register current camera geometry so cube_mask can use height above table."""
    _GEOM["P"] = deproject_fn(meta, depth, cam)
    _GEOM["valid"] = depth > 0

def cube_mask(img):
    if "P" in _GEOM:
        z = _GEOM["P"][..., 2]
        h = hsv(img)
        S = h[..., 1].astype(int)
        m = (z > 0.012) & (z < 0.075) & _GEOM["valid"] & (S >= 45)
        return m.astype(np.uint8)
    return cube_mask_color(img)

def cube_mask_color(img):
    h = hsv(img)
    H, S, V = h[..., 0].astype(int), h[..., 1].astype(int), h[..., 2].astype(int)
    # green..teal hues (opencv H 0..180): ~40-95; saturated
    return ((H >= 35) & (H <= 100) & (S >= 70) & (V >= 60)).astype(np.uint8)

def white_mask(img):
    h = hsv(img)
    S, V = h[..., 1].astype(int), h[..., 2].astype(int)
    return ((S <= 40) & (V >= 200)).astype(np.uint8)

def blobs(mask, min_area=8, max_area=5000):
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        if min_area <= a <= max_area:
            out.append(dict(u=float(cent[i][0]), v=float(cent[i][1]), area=int(a),
                            bbox=tuple(int(x) for x in stats[i, :4]), label=i))
    return out, lab

def detect_cubes(img, min_area=8):
    m = cube_mask(img)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    b, _ = blobs(m, min_area=min_area)
    return b

def highlight_scores(frames, cubes, radius=None):
    """frames: list of RGB images over the highlight window (same camera pose).
    cubes: blobs from a frame. Returns per-cube max ring-white fraction."""
    scores = np.zeros(len(cubes))
    H, W = frames[0].shape[:2]
    vv, uu = np.mgrid[0:H, 0:W]
    base = None
    for f in frames:
        wm = white_mask(f)
        for i, c in enumerate(cubes):
            r = radius or max(6.0, 1.6 * np.sqrt(c["area"]))
            d = np.hypot(uu - c["u"], vv - c["v"])
            ring = (d <= r) & (d >= 0.55 * r)
            frac = wm[ring].mean() if ring.any() else 0
            scores[i] = max(scores[i], frac)
    return scores

def cube_world(meta, img, depth, cam, deproject_fn, min_area=8, zmin=0.025):
    """Detect cubes and return list with world centre (top face) and yaw (mod 90deg)."""
    m = cube_mask(img)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    bl, lab = blobs(m, min_area=min_area)
    P = deproject_fn(meta, depth, cam)
    out = []
    for b in bl:
        sel = (lab == b["label"]) & (depth > 0)
        pts = P[sel]
        if len(pts) < 5:
            continue
        top = pts[pts[:, 2] > pts[:, 2].max() - 0.008]
        if len(top) < 4:
            top = pts
        xy = top[:, :2].astype(np.float32)
        yaw = None
        if len(xy) >= 5:
            (cx, cy), (w, h), ang = cv2.minAreaRect(xy * 1000.0)
            yaw = float(np.deg2rad(ang))
            yaw = (yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4
            c = np.array([cx / 1000.0, cy / 1000.0])
        else:
            c = np.median(xy, 0)
        out.append(dict(u=b["u"], v=b["v"], area=b["area"], xy=np.array(c, float),
                        ztop=float(np.median(top[:, 2])), yaw=yaw, wh=None))
    return out
