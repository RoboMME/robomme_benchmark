"""Perception helpers: back-projection, colour segmentation, video event parsing."""
import json, math
from pathlib import Path
import numpy as np
from PIL import Image

WS = Path("/workspace")


def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def backproject(us, vs, depth_m, K, pose):
    """us, vs pixel arrays; depth_m same shape (linear depth along optical axis). OpenGL camera."""
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    us = np.asarray(us, float); vs = np.asarray(vs, float); d = np.asarray(depth_m, float)
    xc = (us - cx) / fx * d
    yc = -(vs - cy) / fy * d
    zc = -d
    P = np.stack([xc, yc, zc], -1)
    R = quat_to_R(pose["quaternion_wxyz"]); t = np.array(pose["position_m"])
    return P @ R.T + t


def load_rgb(path):
    return np.asarray(Image.open(WS / path).convert("RGB")).astype(np.int16)


def load_depth(path, mpu=0.001):
    return np.asarray(Image.open(WS / path)).astype(np.float64) * mpu


COLORS = {
    # simple rgb rules for saturated cube colours
    "red": lambda r, g, b: (r > 150) & (g < 80) & (b < 80),
    "green": lambda r, g, b: (g > 150) & (r < 100) & (b < 100),
    "blue": lambda r, g, b: (b > 120) & (r < 60) & (g < 60),
    "yellow": lambda r, g, b: (r > 170) & (g > 170) & (b < 80),
}


def target_mask(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    purple = (r > 140) & (b > 170) & (g < 140) & (b - g > 60)
    return purple


def color_mask(rgb, color):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return COLORS[color](r, g, b)


def components(mask, min_px=3):
    """Simple 4-connected components; returns list of (ys, xs)."""
    H, W = mask.shape
    seen = np.zeros_like(mask, bool)
    out = []
    for y0, x0 in zip(*np.nonzero(mask)):
        if seen[y0, x0]:
            continue
        stack = [(y0, x0)]; seen[y0, x0] = True; ys = []; xs = []
        while stack:
            y, x = stack.pop(); ys.append(y); xs.append(x)
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                yy, xx = y + dy, x + dx
                if 0 <= yy < H and 0 <= xx < W and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True; stack.append((yy, xx))
        if len(ys) >= min_px:
            out.append((np.array(ys), np.array(xs)))
    return out


def world_points(mask, depth, K, pose):
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return np.zeros((0, 3))
    d = depth[ys, xs]
    ok = d > 0
    return backproject(xs[ok], ys[ok], d[ok], K, pose)


def cluster_xy(P, radius=0.05):
    """Greedy clustering of world points in xy; returns list of (centroid, n)."""
    rest = P.copy(); out = []
    while len(rest):
        c = rest[0]
        for _ in range(5):
            m = np.linalg.norm(rest[:, :2] - c[:2], axis=1) < radius
            c = rest[m].mean(0)
        m = np.linalg.norm(rest[:, :2] - c[:2], axis=1) < radius
        out.append((rest[m].mean(0), int(m.sum())))
        rest = rest[~m]
    return out


ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
            "seventh": 7, "eighth": 8, "1st": 1, "2nd": 2, "3rd": 3, "4th": 4, "5th": 5, "6th": 6,
            "once": 1, "twice": 2}


def parse_instruction(text):
    t = text.lower()
    color = None
    for c in COLORS:
        if c + " cube" in t or c + " block" in t:
            color = c; break
    if color is None:
        for c in COLORS:
            if c in t:
                color = c; break
    ordinal = None
    words = t.replace(",", " ").replace(".", " ").split()
    for w in words:
        if w in ORDINALS:
            ordinal = ORDINALS[w]; break
    last = ("last" in words) or ("final" in words)
    return {"color": color, "ordinal": ordinal, "last": last}


def video_events(frames, grasp_lo=0.010, grasp_hi=0.075, open_thr=0.074):
    """Parse gripper events from video_demo frame states.
    Returns list of dicts with kind 'grasp'/'release', frame index, xyz."""
    vid = [f for f in frames if f.get("phase") == "video_demo"]
    vid.sort(key=lambda f: f["frame_index"])
    events = []
    holding = False
    for i, f in enumerate(vid):
        s = f["state"]; g = s[6] + s[7]
        if not holding:
            # grasp = fingers settled at partial closure (object between them)
            if grasp_lo < g < grasp_hi - 0.01 and i + 2 < len(vid):
                g2 = vid[i + 2]["state"][6] + vid[i + 2]["state"][7]
                if grasp_lo < g2 < grasp_hi - 0.01 and abs(g2 - g) < 0.006:
                    holding = True
                    events.append({"kind": "grasp", "i": i, "frame_index": f["frame_index"], "xyz": s[:3], "width": g2})
        else:
            if g > open_thr:
                holding = False
                events.append({"kind": "release", "i": i, "frame_index": f["frame_index"], "xyz": s[:3]})
            elif g < grasp_lo:
                holding = False  # closed fully: lost the object
                events.append({"kind": "lost", "i": i, "frame_index": f["frame_index"], "xyz": s[:3]})
    return vid, events


def decode_obs(reply):
    """Decode an MCP observation/step reply into dict with meta, rgb/depth per camera."""
    import base64, io
    m = None
    blocks = reply.get("content", [])
    for b in blocks:
        if b.get("type") == "text":
            m = json.loads(b["text"]); break
    out = {"meta": m, "rgb": {}, "depth": {}}
    if not m:
        return out
    for a in m.get("attachments", []):
        b = blocks[a["content_index"]]
        if b.get("type") != "image":
            continue
        im = Image.open(io.BytesIO(base64.b64decode(b["data"])))
        if a["kind"] == "rgb":
            out["rgb"][a["camera"]] = np.asarray(im.convert("RGB")).astype(np.int16)
        elif a["kind"] == "depth":
            mpu = (m.get("depth", {}).get(a["camera"], {}) or {}).get("meters_per_unit", 0.001)
            out["depth"][a["camera"]] = np.asarray(im).astype(np.float64) * mpu
    out["K"] = {c: v["matrix"] for c, v in (m.get("camera_intrinsics") or {}).items()}
    out["pose"] = m.get("camera_poses") or {}
    return out


def locate_color(obs, camera, color, min_px=8):
    """World points of colour pixels in a decoded obs; returns (centroid_top, points) or (None, None)."""
    if camera not in obs["rgb"] or camera not in obs["depth"]:
        return None, None
    mask = color_mask(obs["rgb"][camera], color)
    if mask.sum() < min_px:
        return None, None
    P = world_points(mask, obs["depth"][camera], obs["K"][camera], obs["pose"][camera])
    if len(P) < min_px:
        return None, None
    # keep largest xy cluster
    cl = cluster_xy(P, 0.04)
    cl.sort(key=lambda c: -c[1])
    c = cl[0][0]
    sel = P[np.linalg.norm(P[:, :2] - c[:2], axis=1) < 0.04]
    return sel.mean(0), sel


def locate_targets(obs, camera="front"):
    if camera not in obs["rgb"]:
        return []
    P = world_points(target_mask(obs["rgb"][camera]), obs["depth"][camera], obs["K"][camera], obs["pose"][camera])
    P = P[P[:, 2] < 0.03]  # on the table
    return [c for c, n in cluster_xy(P, 0.05) if n >= 15]


def top_yaw(P):
    """Yaw (rad, in [-pi/4, pi/4)) of a square top face from its world points (min-area box)."""
    if P is None or len(P) < 10:
        return 0.0
    zt = P[:, 2].max(); top = P[P[:, 2] > zt - 0.006][:, :2]
    if len(top) < 8:
        top = P[:, :2]
    best = (1e9, 0.0)
    for a in np.radians(np.arange(-45, 45, 2.0)):
        c, s = math.cos(a), math.sin(a)
        q = top @ np.array([[c, -s], [s, c]])
        area = np.ptp(q[:, 0]) * np.ptp(q[:, 1])
        if area < best[0]:
            best = (area, a)
    return best[1]


def held_color(frames, i0, n=12):
    """Classify the colour of the object held, from wrist images after grasp index i0 (video list)."""
    votes = {}
    for f in frames[i0 + 4:i0 + 4 + n]:
        files = {(x["camera"], x["kind"]): x["path"] for x in f.get("files", [])}
        p = files.get(("wrist", "rgb"))
        if not p:
            continue
        try:
            rgb = load_rgb(p)
        except Exception:
            continue
        reg = rgb[150:, 60:200]
        for c in COLORS:
            k = int(color_mask(reg, c).sum())
            votes[c] = votes.get(c, 0) + k
    if not votes:
        return None, votes
    best = max(votes, key=votes.get)
    return (best if votes[best] > 200 else None), votes
