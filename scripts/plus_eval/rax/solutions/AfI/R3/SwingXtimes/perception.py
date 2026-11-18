"""Perception helpers: decode MCP replies, back-project depth, find colored blobs."""
import math
import base64, io, json
import numpy as np
from PIL import Image


def parse_reply(reply):
    """Return (meta dict, {(camera, kind): ndarray})."""
    content = reply.get("content", [])
    meta = {}
    for b in content:
        if b.get("type") == "text":
            try:
                meta = json.loads(b["text"]); break
            except Exception:
                pass
    imgs = {}
    for att in meta.get("attachments", []) or []:
        idx = att.get("content_index")
        if idx is None or idx >= len(content):
            continue
        b = content[idx]
        if b.get("type") != "image":
            continue
        try:
            im = Image.open(io.BytesIO(base64.b64decode(b["data"])))
            if att["kind"] == "rgb":
                arr = np.asarray(im.convert("RGB"))
            else:
                arr = np.asarray(im).astype(np.float32)
                mpu = meta.get("depth", {}).get(att["camera"], {}).get("meters_per_unit", 0.001)
                arr = arr * mpu
            imgs[(att["camera"], att["kind"])] = arr
        except Exception:
            pass
    return meta, imgs


def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def backproject(meta, cam, us, vs, depth):
    """us, vs, depth arrays -> Nx3 world points (OpenGL camera convention)."""
    K = np.array(meta["camera_intrinsics"][cam]["matrix"], dtype=float)
    pose = meta["camera_poses"][cam]
    R = quat_to_R(pose["quaternion_wxyz"])
    t = np.array(pose["position_m"], dtype=float)
    us = np.asarray(us, float); vs = np.asarray(vs, float); d = np.asarray(depth, float)
    xc = (us - K[0, 2]) / K[0, 0] * d
    yc = -(vs - K[1, 2]) / K[1, 1] * d
    zc = -d
    P = np.stack([xc, yc, zc], axis=-1)
    return P @ R.T + t


def world_map(meta, imgs, cam):
    """Per-pixel world xyz (HxWx3) and validity mask."""
    D = imgs[(cam, "depth")]
    H, W = D.shape
    vs, us = np.mgrid[0:H, 0:W]
    P = backproject(meta, cam, us.ravel() + 0.5 * 0, vs.ravel(), D.ravel()).reshape(H, W, 3)
    return P, D > 0


def project(meta, cam, P):
    """World points Nx3 -> (u, v, depth)."""
    K = np.array(meta["camera_intrinsics"][cam]["matrix"], dtype=float)
    pose = meta["camera_poses"][cam]
    R = quat_to_R(pose["quaternion_wxyz"])
    t = np.array(pose["position_m"], dtype=float)
    Pc = (np.atleast_2d(P) - t) @ R
    d = -Pc[:, 2]
    u = Pc[:, 0] / d * K[0, 0] + K[0, 2]
    v = -Pc[:, 1] / d * K[1, 1] + K[1, 2]
    return u, v, d


COLOR_NAMES = ["red", "green", "blue", "yellow", "purple", "orange", "cyan", "pink", "white", "black"]


def color_mask(rgb, color):
    r = rgb[..., 0].astype(int); g = rgb[..., 1].astype(int); b = rgb[..., 2].astype(int)
    mx = np.maximum(np.maximum(r, g), b); mn = np.minimum(np.minimum(r, g), b)
    sat = mx - mn
    if color == "red":
        return (r > 120) & (g < 0.35 * r) & (b < 0.35 * r)
    if color == "green":
        return (g > 100) & (r < 0.6 * g) & (b < 0.6 * g)
    if color == "blue":
        return (b > 100) & (r < 0.5 * b) & (g < 0.6 * b)
    if color == "yellow":
        return (r > 150) & (g > 150) & (b < 0.45 * g)
    if color == "purple":
        return (r > 80) & (b > 80) & (g < 0.55 * np.minimum(r, b))
    if color == "orange":
        return (r > 170) & (g > 70) & (g < 0.75 * r) & (b < 0.35 * r)
    if color == "cyan":
        return (g > 120) & (b > 120) & (r < 0.5 * np.minimum(g, b))
    if color == "white":
        return (mn > 190) & (sat < 30)
    return np.zeros(rgb.shape[:2], bool)


def locate_color_object(meta, imgs, cam, color, min_px=6, zmax=None, top_band=0.012,
                        ztop_range=None, size_range=(0.025, 0.075)):
    """Find the best blob of `color` (largest that satisfies geometric checks); return dict(center xyz
    of top face, yaw mod 90deg, top z, n)."""
    import cv2
    rgb = imgs[(cam, "rgb")]; D = imgs[(cam, "depth")]
    m = color_mask(rgb, color) & (D > 0)
    if m.sum() < min_px:
        return None
    n, lab, stats, cent = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    order = sorted(range(1, n), key=lambda i: -stats[i, cv2.CC_STAT_AREA])
    for i in order[:6]:
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_px:
            continue
        vs, us = np.nonzero(lab == i)
        P = backproject(meta, cam, us, vs, D[vs, us])
        if zmax is not None:
            keep = P[:, 2] < zmax
            if keep.sum() < min_px:
                continue
            P = P[keep]; us = us[keep]; vs = vs[keep]
        ztop = np.percentile(P[:, 2], 90)
        if ztop_range is not None and not (ztop_range[0] <= ztop <= ztop_range[1]):
            continue
        top = P[P[:, 2] > ztop - top_band]
        if len(top) < 4:
            top = P
        xy = top[:, :2].astype(np.float32)
        rect = cv2.minAreaRect(xy)
        (cx, cy), (w, h), ang = rect
        if size_range is not None and not (size_range[0] <= max(w, h) <= size_range[1]):
            continue
        yaw = math.radians(ang) % (math.pi / 2)
        if yaw > math.pi / 4:
            yaw -= math.pi / 2
        return {"center": np.array([cx, cy, ztop]), "yaw": yaw, "size": (w, h), "n": len(P),
                "uv": (float(us.mean()), float(vs.mean()))}
    return None


def locate_flat_whites(meta, imgs, cam="front"):
    """Cluster bright low-saturation pixels near the table. Returns list of clusters
    dict(center xy, zmax, n) sorted by n desc."""
    import cv2
    rgb = imgs[(cam, "rgb")].astype(int); D = imgs[(cam, "depth")]
    mx = rgb.max(-1); mn = rgb.min(-1)
    m = (mn > 120) & ((mx - mn) < 40) & (D > 0)
    P, _ = world_map(meta, imgs, cam)
    m &= (P[..., 2] < 0.08) & (P[..., 2] > -0.01)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    out = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 8:
            continue
        sel = lab == i
        Q = P[sel]
        out.append({"center": (Q[:, :2].max(0) + Q[:, :2].min(0)) / 2, "mean": Q[:, :2].mean(0), "zmax": float(np.percentile(Q[:, 2], 95)),
                    "zmed": float(np.median(Q[:, 2])), "n": int(sel.sum()),
                    "extent": (Q[:, :2].max(0) - Q[:, :2].min(0)).tolist()})
    out.sort(key=lambda c: -c["n"])
    return out


def detect_discs(meta, imgs, cam="front", zlo=0.0045, zhi=0.0130, P=None, valid=None):
    """Geometric target-disc detection: connected regions whose world z lies in the disc-top band.
    Robust to coloured lights. Returns list of dict(center, n, extent, fit)."""
    import cv2
    if P is None:
        P, valid = world_map(meta, imgs, cam)
    z = P[..., 2]
    band = valid & (z > zlo) & (z < zhi) & (P[..., 0] > -0.45)
    band_u8 = band.astype(np.uint8)
    band_u8 = cv2.morphologyEx(band_u8, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(band_u8, 8)
    table = valid & (z < 0.003) & (z > -0.004)
    out = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 12:
            continue
        sel = lab == i
        Q = P[sel][:, :2]
        ext = Q.max(0) - Q.min(0)
        if ext.max() > 0.10 or ext.max() < 0.02:
            continue
        # edge pixels of the region that border the table (true disc rim)
        er = cv2.erode(sel.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        edge = sel & ~er
        tb = cv2.dilate(table.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        rim = edge & tb
        E = P[rim][:, :2]
        c0 = (Q.max(0) + Q.min(0)) / 2
        fit = None
        if len(E) >= 8:
            c = Q.mean(0).copy()
            r = 0.039
            for _ in range(15):
                d = E - c
                dist = np.linalg.norm(d, axis=1) + 1e-9
                res = dist - r
                J = -d / dist[:, None]
                step, *_ = np.linalg.lstsq(J, -res, rcond=None)
                c = c + step
            res = np.abs(np.linalg.norm(E - c, axis=1) - r)
            fit = {"center": c, "rms": float(np.sqrt(np.mean(res ** 2))), "m": int(len(E))}
        full = ext.min() > 0.066
        center = c0 if (full or fit is None or fit["rms"] > 0.006) else fit["center"]
        out.append({"center": np.asarray(center), "bbox_center": c0, "n": int(sel.sum()),
                    "extent": ext.tolist(), "fit": fit, "full": bool(full)})
    out.sort(key=lambda c: -c["n"])
    return out


def detect_buttons(meta, imgs, cam="front", P=None, valid=None):
    """Geometric button detection: square base top (z~0.016) region with a raised cap (z~0.034)."""
    import cv2
    if P is None:
        P, valid = world_map(meta, imgs, cam)
    z = P[..., 2]
    band = valid & (z > 0.0133) & (z < 0.0215) & (P[..., 0] > -0.45)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(band.astype(np.uint8), 8)
    capm = valid & (z > 0.025) & (z < 0.045)
    out = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 15:
            continue
        sel = lab == i
        Q = P[sel][:, :2]
        ext = Q.max(0) - Q.min(0)
        if ext.min() < 0.035 or ext.max() > 0.11:
            continue
        c0 = (Q.max(0) + Q.min(0)) / 2
        C = P[capm & (np.hypot(P[..., 0] - c0[0], P[..., 1] - c0[1]) < 0.035)]
        if len(C) < 5:
            continue
        ztop = float(np.percentile(C[:, 2], 95))
        hi = C[C[:, 2] > ztop - 0.004]
        out.append({"center": hi[:, :2].mean(0), "base_center": c0, "ztop": ztop, "n": int(sel.sum()),
                    "ncap": int(len(C)), "extent": ext.tolist()})
    out.sort(key=lambda c: -c["n"])
    return out
