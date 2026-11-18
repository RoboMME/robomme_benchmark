"""Perception helpers for VideoUnmask: colour segmentation + depth back-projection."""
import json
from pathlib import Path
import numpy as np
from PIL import Image

WS = Path("/workspace")

# PIL HSV hue (0-255) centres of saturated cube colours.
HUES = {"red": 0, "orange": 29, "yellow": 42, "green": 85, "cyan": 128,
        "blue": 170, "purple": 200, "magenta": 220}


def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def deproject(u, v, d, K, pose):
    """Pixel (u,v) with linear depth d (m) -> world xyz. OpenGL camera (-Z forward, +Y up)."""
    K = K["matrix"] if isinstance(K, dict) else K
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    pc = np.array([(u - cx) / fx * d, -(v - cy) / fy * d, -d])
    R = quat_to_R(pose["quaternion_wxyz"])
    return R @ pc + np.array(pose["position_m"])


def deproject_mask(mask, depth, K, pose):
    """All pixels of mask -> Nx3 world points (valid depth only)."""
    vs, us = np.nonzero(mask & (depth > 0))
    if len(vs) == 0:
        return np.zeros((0, 3))
    d = depth[vs, us]
    K = K["matrix"] if isinstance(K, dict) else K
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    pc = np.stack([(us - cx) / fx * d, -(vs - cy) / fy * d, -d], 1)
    R = quat_to_R(pose["quaternion_wxyz"])
    return pc @ R.T + np.array(pose["position_m"])


def load_rgb_hsv(path):
    im = Image.open(path).convert("RGB")
    return np.array(im).astype(int), np.array(im.convert("HSV")).astype(int)


def load_depth(path, mpu=0.001):
    return np.array(Image.open(path)).astype(np.float64) * mpu


def hue_dist(h, c):
    d = np.abs(h - c)
    return np.minimum(d, 256 - d)


def color_mask(hsv, name, tol=8, smin=215, vmin=60):
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return (s >= smin) & (v >= vmin) & (hue_dist(h, HUES[name]) <= tol)


def white_mask(rgb, hsv, vmin=150, smax=50):
    return (hsv[..., 1] <= smax) & (hsv[..., 2] >= vmin)


def components(mask, min_px=15):
    """Simple 4-connected components; returns list of (N, (vs, us))."""
    H, W = mask.shape
    lab = np.zeros((H, W), int)
    out = []
    cur = 0
    for v0 in range(H):
        for u0 in range(W):
            if mask[v0, u0] and lab[v0, u0] == 0:
                cur += 1
                stack = [(v0, u0)]
                lab[v0, u0] = cur
                pts = []
                while stack:
                    v, u = stack.pop()
                    pts.append((v, u))
                    for dv, du in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        a, b = v + dv, u + du
                        if 0 <= a < H and 0 <= b < W and mask[a, b] and lab[a, b] == 0:
                            lab[a, b] = cur
                            stack.append((a, b))
                if len(pts) >= min_px:
                    arr = np.array(pts)
                    out.append((len(pts), arr[:, 0], arr[:, 1]))
    out.sort(key=lambda t: -t[0])
    return out


def frame_files(frame):
    files = {}
    for f in frame["files"]:
        files[(f["camera"], f["kind"])] = (WS / f["path"], f.get("meters_per_unit", 0.001))
    return files


def load_manifest(path):
    return json.loads((WS / path).read_text())


def decode_reply(reply):
    """MCP reply -> dict(meta, rgb{cam}, hsv{cam}, depth{cam} in metres)."""
    import base64, io
    blocks = reply.get("content", [])
    m = {}
    for b in blocks:
        if b.get("type") == "text":
            try:
                m = json.loads(b["text"])
                break
            except Exception:
                pass
    out = {"meta": m, "rgb": {}, "hsv": {}, "depth": {}}
    for att in m.get("attachments", []):
        idx = att.get("content_index")
        if idx is None or idx >= len(blocks):
            continue
        b = blocks[idx]
        data = b.get("data")
        if not data:
            continue
        im = Image.open(io.BytesIO(base64.b64decode(data)))
        cam = att.get("camera")
        if att.get("kind") == "rgb":
            im = im.convert("RGB")
            out["rgb"][cam] = np.array(im).astype(int)
            out["hsv"][cam] = np.array(im.convert("HSV")).astype(int)
        elif att.get("kind") == "depth":
            mpu = (m.get("depth", {}).get(cam, {}) or {}).get("meters_per_unit", 0.001)
            out["depth"][cam] = np.array(im).astype(np.float64) * mpu
    return out


def cloud(obs, cam, mask=None):
    m = obs["meta"]
    if mask is None:
        mask = white_mask(obs["rgb"][cam], obs["hsv"][cam])
    return deproject_mask(mask, obs["depth"][cam], m["camera_intrinsics"][cam], m["camera_poses"][cam])


def cluster_xy(pts, radius=0.03, min_pts=20):
    """Greedy xy clustering on points; returns list of point arrays, largest first."""
    pts = np.asarray(pts)
    left = np.ones(len(pts), bool)
    clusters = []
    while left.any():
        idx = np.nonzero(left)[0]
        seed = pts[idx[0], :2]
        members = np.zeros(len(pts), bool)
        c = seed
        for _ in range(6):
            d = np.linalg.norm(pts[:, :2] - c, axis=1)
            members = left & (d < radius)
            if not members.any():
                break
            c = pts[members, :2].mean(0)
        if members.sum() == 0:
            left[idx[0]] = False
            continue
        if members.sum() >= min_pts:
            clusters.append(pts[members])
        left &= ~members
        left[idx[0]] = False
    clusters.sort(key=lambda a: -len(a))
    return clusters


def merge_close(centres, radius=0.03):
    out = []
    for c in centres:
        for o in out:
            if np.linalg.norm(o["xy"] - c["xy"]) < radius:
                n = o["n"] + c["n"]
                o["xy"] = (o["xy"] * o["n"] + c["xy"] * c["n"]) / n
                o["n"] = n
                o["ztop"] = max(o["ztop"], c["ztop"])
                break
        else:
            out.append(dict(c))
    return out


def grid_components(p, cell=0.005, dilate=1, min_pts=30):
    """Connected components of xy occupancy grid; returns list of point arrays."""
    if len(p) == 0:
        return []
    ij = np.floor(p[:, :2] / cell).astype(int)
    occ = {}
    for k, key in enumerate(map(tuple, ij)):
        occ.setdefault(key, []).append(k)
    keys = set(occ)
    grown = set()
    for (i, j) in keys:
        for di in range(-dilate, dilate + 1):
            for dj in range(-dilate, dilate + 1):
                grown.add((i + di, j + dj))
    seen = set()
    comps = []
    for start in keys:
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        members = []
        while stack:
            c = stack.pop()
            if c in occ:
                members.extend(occ[c])
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    nb = (c[0] + di, c[1] + dj)
                    if nb in grown and nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
        if len(members) >= min_pts:
            comps.append(p[members])
    comps.sort(key=lambda a: -len(a))
    return comps


def find_containers(obs, cams=("front",), zmin=0.015, zmax=0.12, exclude_xy=None, exclude_r=0.06,
                    min_pts=30):
    """White box-like objects on the table -> list of dict(xy, ztop, n, pts, top)."""
    pts = []
    for cam in cams:
        if cam not in obs["rgb"]:
            continue
        pts.append(cloud(obs, cam))
    if not pts:
        return []
    p = np.concatenate(pts)
    keep = (p[:, 2] > zmin) & (p[:, 2] < zmax) & (np.abs(p[:, 1]) < 0.45) & (p[:, 0] > -0.4) & (p[:, 0] < 0.35)
    if exclude_xy is not None:
        keep &= np.linalg.norm(p[:, :2] - np.asarray(exclude_xy)[:2], axis=1) > exclude_r
    p = p[keep]
    cs = []
    for c in grid_components(p, min_pts=min_pts):
        ztop = float(np.percentile(c[:, 2], 97))
        top = c[c[:, 2] > ztop - 0.01]
        ext = c[:, :2].max(0) - c[:, :2].min(0)
        cs.append({"xy": top[:, :2].mean(0) if len(top) >= 8 else c[:, :2].mean(0),
                   "ztop": ztop, "n": len(c), "pts": c, "top": top, "extent": ext})
    return cs


def estimate_yaw(pts_xy):
    """Yaw (rad, in [0, pi/2)) of a square footprint minimizing bounding-box area."""
    best = (1e9, 0.0)
    for a in np.linspace(0, np.pi / 2, 91)[:-1]:
        c, s = np.cos(a), np.sin(a)
        R = np.array([[c, s], [-s, c]])
        q = pts_xy @ R.T
        lo = np.percentile(q, 3, axis=0)
        hi = np.percentile(q, 97, axis=0)
        area = np.prod(hi - lo)
        if area < best[0]:
            best = (area, a)
    return best[1], best[0]


def table_containers(obs, cams=("front", "wrist")):
    """Container candidates from several cameras, merged and shape-filtered."""
    allc = []
    for cam in cams:
        if cam not in obs["rgb"]:
            continue
        for c in find_containers(obs, cams=(cam,)):
            if not (0.05 <= c["ztop"] <= 0.095):
                continue
            if c["extent"].max() > 0.10 or len(c["top"]) < 8:
                continue
            allc.append({"xy": c["xy"], "ztop": c["ztop"], "n": c["n"], "cam": cam})
    merged = []
    for c in sorted(allc, key=lambda t: -t["n"]):
        for m in merged:
            if np.linalg.norm(m["xy"] - c["xy"]) < 0.035:
                m["cams"].add(c["cam"])
                break
        else:
            merged.append({"xy": c["xy"], "ztop": c["ztop"], "n": c["n"], "cams": {c["cam"]}})
    return merged
