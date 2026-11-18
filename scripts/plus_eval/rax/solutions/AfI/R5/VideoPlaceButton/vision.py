import base64, io, json
from pathlib import Path
import numpy as np
from PIL import Image

WS = Path("/workspace")


def parse_reply(reply):
    """Return (meta, images) where images[(camera, kind)] = np.ndarray."""
    content = reply["content"]
    meta = json.loads(next(b["text"] for b in content if b.get("type") == "text"))
    images = {}
    for att in meta.get("attachments", []) or []:
        idx = att["content_index"]
        if idx >= len(content):
            continue
        blk = content[idx]
        if blk.get("type") != "image":
            continue
        im = Image.open(io.BytesIO(base64.b64decode(blk["data"])))
        arr = np.array(im)
        images[(att["camera"], att["kind"])] = arr
    return meta, images


def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def backproject(meta, depth_png, camera, us, vs):
    """Pixels -> world points (N,3); invalid depth -> nan."""
    K = np.array(meta["camera_intrinsics"][camera]["matrix"], float)
    pose = meta["camera_poses"][camera]
    R = quat_to_R(pose["quaternion_wxyz"])
    t = np.array(pose["position_m"], float)
    mpu = meta.get("depth", {}).get(camera, {}).get("meters_per_unit", 0.001)
    us = np.asarray(us, float); vs = np.asarray(vs, float)
    d = depth_png[vs.astype(int), us.astype(int)].astype(float) * mpu
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    pc = np.stack([(us - cx) * d / fx, -(vs - cy) * d / fy, -d], 1)
    pw = pc @ R.T + t
    pw[d <= 0] = np.nan
    return pw


def red_mask(rgb):
    return color_mask(rgb, "red")


def color_mask(rgb, color):
    """Saturated primary-colour cube pixels (cubes are pure red/green/blue; targets are pastel)."""
    r = rgb[..., 0].astype(int); g = rgb[..., 1].astype(int); b = rgb[..., 2].astype(int)
    if color == "green":
        return (g > 60) & (r < 0.35 * g) & (b < 0.35 * g)
    if color == "blue":
        return (b > 60) & (r < 0.35 * b) & (g < 0.35 * b)
    return (r > 60) & (g < 0.35 * r) & (b < 0.35 * r)


def color_from_instruction(text):
    t = (text or "").lower()
    for c in ("red", "green", "blue"):
        if c + " cube" in t:
            return c
    for c in ("red", "green", "blue"):
        if c in t:
            return c
    return "red"


def largest_component(mask):
    import cv2
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 1:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return lab == k, int(stats[k, cv2.CC_STAT_AREA])


def components(mask, min_area=4):
    import cv2
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    out = []
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] >= min_area:
            out.append((lab == k, int(stats[k, cv2.CC_STAT_AREA])))
    out.sort(key=lambda c: -c[1])
    return out


def cube_from_points(pts):
    """Given world points on the red cube, return center xy, top z, yaw (rad mod pi/2), n_top."""
    import cv2
    pts = pts[np.all(np.isfinite(pts), 1)]
    if len(pts) < 5:
        return None
    ztop = np.percentile(pts[:, 2], 95)
    top = pts[pts[:, 2] > ztop - 0.006]
    if len(top) < 5:
        top = pts
    rect = cv2.minAreaRect(top[:, :2].astype(np.float32))
    (cx, cy), (w, h), ang = rect
    yaw = np.deg2rad(ang) % (np.pi / 2)
    return dict(xy=np.array([cx, cy]), ztop=float(ztop), yaw=float(yaw), size=(float(w), float(h)), n_top=len(top))


def target_mask(rgb):
    r = rgb[..., 0].astype(int); g = rgb[..., 1].astype(int); b = rgb[..., 2].astype(int)
    return (r > 140) & (r - g > 50) & (b > g + 15) & (b < r)


def detect_targets(meta, rgb, depth, camera="front"):
    """Return list of world xy centres of pink/white bullseye targets (table-level blobs)."""
    import cv2
    m = target_mask(rgb).astype(np.uint8)
    m = cv2.dilate(m, np.ones((3, 3), np.uint8))
    out = []
    for c, a in components(m, min_area=6):
        vs, us = np.nonzero(c)
        p = backproject(meta, depth, camera, us, vs)
        p = p[np.all(np.isfinite(p), 1)]
        if len(p) < 4:
            continue
        if np.median(p[:, 2]) > 0.03:
            continue
        out.append(dict(xy=np.median(p[:, :2], 0), area=a, uv=(float(us.mean()), float(vs.mean()))))
    return out
