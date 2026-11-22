import numpy as np

def hsv(img):
    x = img.astype(np.float32) / 255.0
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    mx = x.max(-1); mn = x.min(-1); c = mx - mn
    h = np.zeros_like(mx)
    m = c > 1e-6
    rm = m & (mx == r); gm = m & (mx == g) & ~rm; bm = m & ~rm & ~gm
    h[rm] = ((g - b)[rm] / c[rm]) % 6
    h[gm] = (b - r)[gm] / c[gm] + 2
    h[bm] = (r - g)[bm] / c[bm] + 4
    h = h * 60.0
    s = np.where(mx > 1e-6, c / np.maximum(mx, 1e-6), 0)
    return h, s, mx

from scipy import ndimage
import lib

COLORS = {
    "red": [(0, 14), (345, 361)],
    "orange": [(22, 48)],
    "yellow": [(48, 75)],
    "green": [(85, 165)],
    "cyan": [(165, 200)],
    "blue": [(200, 258)],
    "purple": [(258, 345)],
}

def world_points(meta, cam, depth):
    K, R, t = lib.cam_params(meta, cam)
    h, w = depth.shape
    vv, uu = np.mgrid[0:h, 0:w].astype(np.float32)
    d = depth
    pc = np.stack([(uu - K[0, 2]) * d / K[0, 0], -(vv - K[1, 2]) * d / K[1, 1], -d], -1)
    P = pc.reshape(-1, 3) @ R.T + t
    return P.reshape(h, w, 3)

def detect_cubes(rgb, depth, meta, cam="front", min_pix=25, zmin=0.008):
    h, s, v = hsv(rgb)
    sat = (s > 0.6) & (v > 0.2)
    P = world_points(meta, cam, depth) if depth is not None else None
    out = []
    for name, ranges in COLORS.items():
        m = np.zeros_like(sat)
        for a, b in ranges:
            m |= (h >= a) & (h < b)
        m &= sat
        lab, n = ndimage.label(m)
        for i in range(1, n + 1):
            ys, xs = np.nonzero(lab == i)
            if len(xs) < min_pix:
                continue
            item = {"color": name, "u": float(xs.mean()), "v": float(ys.mean()), "n": int(len(xs))}
            if P is not None:
                pts = P[ys, xs]
                ok = depth[ys, xs] > 0
                if ok.sum() >= 3:
                    item["xyz"] = np.median(pts[ok], 0).tolist()
                    if item["xyz"][2] < zmin:
                        continue
                else:
                    continue
            out.append(item)
    return out

def detect_white(rgb, depth, meta, cam="front", zmin=0.012, zmax=0.09, xy_box=(-0.45, 0.25, -0.35, 0.35), min_pix=15):
    h, s, v = hsv(rgb)
    P = world_points(meta, cam, depth)
    m = (s < 0.18) & (v > 0.55) & (depth > 0)
    z = P[..., 2]
    m &= (z > zmin) & (z < zmax)
    m &= (P[..., 0] > xy_box[0]) & (P[..., 0] < xy_box[1]) & (P[..., 1] > xy_box[2]) & (P[..., 1] < xy_box[3])
    return m, P

def rect_yaw(xy):
    """Min-area rectangle yaw (mod 90 deg) of 2D points, returns (yaw, center, size)."""
    best = None
    c0 = xy.mean(0)
    for ang in np.radians(np.arange(0, 90, 2.0)):
        ca, sa = np.cos(ang), np.sin(ang)
        R = np.array([[ca, sa], [-sa, ca]])
        q = (xy - c0) @ R.T
        lo, hi = np.percentile(q, 3, 0), np.percentile(q, 97, 0)
        area = np.prod(hi - lo)
        if best is None or area < best[0]:
            ctr = c0 + ((lo + hi) / 2) @ R
            best = (area, ang, ctr, hi - lo)
    return best[1], best[2], best[3]

def detect_containers(rgb, depth, meta, button_xy=None, cam="front"):
    m, P = detect_white(rgb, depth, meta, cam)
    lab, n = ndimage.label(m)
    out = []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(lab == i)
        if len(xs) < 25:
            continue
        pts = P[ys, xs]
        ztop = np.percentile(pts[:, 2], 95)
        if button_xy is not None and np.linalg.norm(np.median(pts[:, :2], 0) - button_xy) < 0.03 and ztop < 0.04:
            continue
        top = pts[pts[:, 2] > ztop - 0.008]
        if len(top) < 8:
            top = pts
        yaw, ctr, size = rect_yaw(top[:, :2])
        if not (0.025 < min(size) and max(size) < 0.065):
            continue
        out.append({"u": float(xs.mean()), "v": float(ys.mean()), "n": int(len(xs)), "ztop": float(ztop),
                    "xy": ctr.tolist(), "yaw": float(yaw), "size": size.tolist(), "xy_med": np.median(top[:, :2], 0).tolist()})
    return out

class CubeTracker:
    """Accumulate colored-cube detections across frames into distinct 3-D clusters."""
    def __init__(self, merge=0.03):
        self.clusters = []  # dict color, pts(list), steps(list)
        self.merge = merge

    def add_frame(self, item):
        step = item.get("step")
        fmeta = item.get("meta") or {}
        if "camera_poses" not in fmeta:
            return
        for cam in ("front", "wrist"):
            rgb = item["rgb"].get(cam); dep = item["depth"].get(cam)
            if rgb is None or dep is None:
                continue
            try:
                dets = detect_cubes(rgb, dep, fmeta, cam=cam, min_pix=25 if cam == "front" else 20)
            except Exception:
                continue
            for d in dets:
                xyz = np.array(d["xyz"])
                if not (-0.5 < xyz[0] < 0.3 and -0.35 < xyz[1] < 0.35 and xyz[2] < 0.06):
                    continue
                self._add(d["color"], xyz, step, cam)

    def _add(self, color, xyz, step, cam):
        best = None
        for c in self.clusters:
            if c["color"] != color:
                continue
            dd = np.linalg.norm(np.median(c["pts"], 0)[:2] - xyz[:2])
            if dd < self.merge and (best is None or dd < best[0]):
                best = (dd, c)
        if best is None:
            self.clusters.append({"color": color, "pts": [xyz], "steps": [step], "cams": [cam]})
        else:
            c = best[1]; c["pts"].append(xyz); c["steps"].append(step); c["cams"].append(cam)

    def cubes(self, min_obs=2, exclude_xy=None, exclude_r=0.035):
        out = []
        for c in self.clusters:
            if len(c["pts"]) < min_obs:
                continue
            if exclude_xy is not None and np.linalg.norm(np.median(np.array(c["pts"]), 0)[:2] - np.asarray(exclude_xy)) < exclude_r:
                continue
            P = np.array(c["pts"])
            wr = [p for p, cam in zip(c["pts"], c["cams"]) if cam == "wrist"]
            fr = [p for p, cam in zip(c["pts"], c["cams"]) if cam == "front"]
            ctr = np.median(np.array(wr), 0) if len(wr) >= 2 else (np.median(np.array(fr), 0) if fr else np.median(P, 0))
            out.append({"color": c["color"], "xyz": ctr.tolist(), "n": len(c["pts"]),
                        "first": min(s for s in c["steps"] if s is not None) if any(s is not None for s in c["steps"]) else None,
                        "last": max(s for s in c["steps"] if s is not None) if any(s is not None for s in c["steps"]) else None})
        return out

def detect_button(rgb, depth, meta):
    """Find the button at step 0 (no containers yet): small object, top 0.02-0.06 m."""
    P = world_points(meta, "front", depth)
    m = (depth > 0) & (P[..., 2] > 0.006) & (P[..., 2] < 0.08)
    m &= (P[..., 0] > -0.45) & (P[..., 0] < 0.3) & (np.abs(P[..., 1]) < 0.35)
    lab, n = ndimage.label(m)
    best = None
    for i in range(1, n + 1):
        ys, xs = np.nonzero(lab == i)
        if len(xs) < 15:
            continue
        pts = P[ys, xs]
        ztop = np.percentile(pts[:, 2], 95)
        ext = np.percentile(pts[:, :2], 97, 0) - np.percentile(pts[:, :2], 3, 0)
        if not (0.018 < ztop < 0.06) or ext.max() > 0.12:
            continue
        if best is None or len(xs) > best[0]:
            best = (len(xs), ys, xs)
    if best is None:
        return None
    pts = P[best[1], best[2]]
    ztop = np.percentile(pts[:, 2], 95)
    cap = pts[pts[:, 2] > ztop - 0.006]
    base_xy = np.median(pts[:, :2], 0)
    return {"xy": np.median(cap[:, :2], 0).tolist(), "base_xy": base_xy.tolist(), "ztop": float(ztop), "n": int(best[0])}

def refine_container_wrist(rgb, depth, meta, exp_xy, zlo=0.055, zhi=0.095, maxd=0.03):
    P = world_points(meta, "wrist", depth)
    z = P[..., 2]
    m = (depth > 0) & (z > zlo) & (z < zhi)
    lab, n = ndimage.label(m)
    best = None
    for i in range(1, n + 1):
        ys, xs = np.nonzero(lab == i)
        if len(xs) < 40:
            continue
        pts = P[ys, xs]
        ctr = np.median(pts[:, :2], 0)
        d = np.linalg.norm(ctr - np.asarray(exp_xy))
        if d < maxd and (best is None or d < best[0]):
            best = (d, pts)
    if best is None:
        return None
    pts = best[1]
    ztop = float(np.percentile(pts[:, 2], 90))
    top = pts[pts[:, 2] > ztop - 0.01]
    yaw, ctr, size = rect_yaw(top[:, :2])
    return {"xy": ctr.tolist(), "yaw": float(yaw), "size": size.tolist(), "ztop": ztop, "n": int(len(pts))}


def detect_containers_wrist(rgb, depth, meta, zlo=0.055, zhi=0.095, button_xy=None):
    """Top faces of containers from a (roughly) top-down wrist view."""
    P = world_points(meta, "wrist", depth)
    z = P[..., 2]
    m = (depth > 0) & (z > zlo) & (z < zhi)
    m &= (P[..., 0] > -0.45) & (P[..., 0] < 0.3) & (np.abs(P[..., 1]) < 0.38)
    lab, n = ndimage.label(m)
    out = []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(lab == i)
        if len(xs) < 12:
            continue
        pts = P[ys, xs]
        ztop = float(np.percentile(pts[:, 2], 90))
        top = pts[pts[:, 2] > ztop - 0.012]
        if len(top) < 8:
            continue
        yaw, ctr, size = rect_yaw(top[:, :2])
        if not (0.025 < min(size) and max(size) < 0.065):
            continue
        out.append({"xy": ctr.tolist(), "yaw": float(yaw), "size": size.tolist(), "ztop": ztop, "n": int(len(xs)), "src": "wrist"})
    return out

def visible_cubes(rgb, depth, meta, cam, exclude_xy=None):
    try:
        return [d for d in detect_cubes(rgb, depth, meta, cam=cam, min_pix=15) if -0.5 < d["xyz"][0] < 0.3 and abs(d["xyz"][1]) < 0.38 and d["xyz"][2] < 0.06
                and (exclude_xy is None or np.linalg.norm(np.array(d["xyz"][:2]) - np.asarray(exclude_xy)) > 0.035)]
    except Exception:
        return []
