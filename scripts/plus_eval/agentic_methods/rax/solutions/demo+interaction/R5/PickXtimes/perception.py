import base64, io, json, math
import numpy as np
from PIL import Image


def decode(result):
    """Decode an MCP observation result into meta dict + numpy images."""
    content = result["content"]
    meta = json.loads(content[0]["text"])
    imgs = {}
    for att in meta.get("attachments", []):
        blk = content[att["content_index"]]
        raw = base64.b64decode(blk["data"])
        im = Image.open(io.BytesIO(raw))
        arr = np.array(im)
        if att["kind"] == "rgb":
            arr = np.array(im.convert("RGB"))
        else:
            arr = arr.astype(np.float32) * meta["depth"][att["camera"]]["meters_per_unit"]
        imgs[(att["camera"], att["kind"])] = arr
    return meta, imgs


def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def backproject(meta, depth, cam, us, vs):
    """us, vs arrays of pixel coords -> world points (N,3); invalid depth -> nan."""
    K = np.array(meta["camera_intrinsics"][cam]["matrix"])
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    pose = meta["camera_poses"][cam]
    R = quat_to_R(pose["quaternion_wxyz"])
    t = np.array(pose["position_m"])
    us = np.asarray(us, float); vs = np.asarray(vs, float)
    d = depth[vs.astype(int), us.astype(int)]
    pc = np.stack([(us - cx) * d / fx, -(vs - cy) * d / fy, -d], -1)
    pw = pc @ R.T + t
    pw[d <= 0] = np.nan
    return pw


def world_cloud(meta, imgs, cam):
    depth = imgs[(cam, "depth")]
    H, W = depth.shape
    vs, us = np.mgrid[0:H, 0:W]
    pw = backproject(meta, depth, cam, us.ravel(), vs.ravel()).reshape(H, W, 3)
    return pw


def color_masks(rgb):
    r = rgb[..., 0].astype(int); g = rgb[..., 1].astype(int); b = rgb[..., 2].astype(int)
    mx = np.maximum(np.maximum(r, g), b); mn = np.minimum(np.minimum(r, g), b)
    sat = mx - mn
    m = {}
    m["green"] = (g > 90) & (g - r > 50) & (g - b > 50)
    m["red"] = (r > 90) & (r - g > 60) & (r - b > 60)
    m["blue"] = (b > 90) & (b - r > 50) & (b - g > 50)
    m["yellow"] = (r > 120) & (g > 120) & (b < 80) & (abs(r - g) < 60)
    # target rings: purple/magenta (r and b high, g low)
    m["purple"] = (r > 100) & (b > 100) & (g < r - 40) & (g < b - 40)
    return m


def blobs(mask, min_px=4):
    """Simple connected components (4-neighbour) -> list of (count, cu, cv, pixels)."""
    from collections import deque
    H, W = mask.shape
    seen = np.zeros_like(mask, bool)
    out = []
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys, xs):
        if seen[y0, x0]:
            continue
        q = deque([(y0, x0)]); seen[y0, x0] = True; pix = []
        while q:
            y, x = q.popleft(); pix.append((y, x))
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                yy, xx = y + dy, x + dx
                if 0 <= yy < H and 0 <= xx < W and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True; q.append((yy, xx))
        if len(pix) >= min_px:
            p = np.array(pix)
            out.append((len(pix), p[:, 1].mean(), p[:, 0].mean(), p))
    out.sort(key=lambda b: -b[0])
    return out


def axang_to_R(v):
    v = np.asarray(v, float); th = np.linalg.norm(v)
    if th < 1e-9:
        return np.eye(3)
    k = v / th; Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * Kx + (1 - math.cos(th)) * Kx @ Kx


def finger_yaw(state):
    R = axang_to_R(state[3:6])
    a = R[:, 1]
    return math.atan2(a[1], a[0])


def wrap90(a):
    """wrap angle to (-pi/4, pi/4]"""
    while a > math.pi / 4: a -= math.pi / 2
    while a <= -math.pi / 4: a += math.pi / 2
    return a


def rect_yaw(xy):
    """Yaw (mod 90deg) of the min-area bounding rectangle of 2D points."""
    best = None
    c = xy.mean(0)
    p = xy - c
    for deg in np.arange(0, 90, 1.0):
        t = math.radians(deg)
        u = p @ np.array([math.cos(t), math.sin(t)]); w = p @ np.array([-math.sin(t), math.cos(t)])
        area = (np.percentile(u, 97) - np.percentile(u, 3)) * (np.percentile(w, 97) - np.percentile(w, 3))
        if best is None or area < best[0]:
            best = (area, t)
    return best[1]


def find_objects(meta, imgs, cam, color="green"):
    """Return dict with detections in world frame from one camera."""
    cloud = world_cloud(meta, imgs, cam)
    rgb = imgs[(cam, "rgb")]
    ms = color_masks(rgb)
    res = {}
    # cube of color
    bl = blobs(ms[color], 6) if color in ms else []
    cands = []
    for n, cu, cv, pix in bl:
        pts = cloud[pix[:, 0], pix[:, 1]]
        pts = pts[np.isfinite(pts).all(1)]
        if len(pts) < 5:
            continue
        top = np.nanmax(pts[:, 2])
        if top < 0.01:
            continue
        tp = pts[pts[:, 2] > top - 0.006]
        cands.append(dict(n=n, uv=(cu, cv), top=float(top), xy=tp[:, :2].mean(0), top_pts=tp, pts=pts))
    res["cube"] = cands
    # target: purple rings
    bl = blobs(ms["purple"], 4)
    tg = []
    for n, cu, cv, pix in bl:
        pts = cloud[pix[:, 0], pix[:, 1]]
        pts = pts[np.isfinite(pts).all(1)]
        if len(pts) < 3:
            continue
        tg.append(dict(n=n, uv=(cu, cv), pts=pts))
    if tg:
        # merge all purple pixels near the largest blob
        main = tg[0]["pts"]; c = np.median(main[:, :2], 0)
        allp = np.concatenate([t["pts"] for t in tg])
        near = allp[np.linalg.norm(allp[:, :2] - c, axis=1) < 0.05]
        res["target"] = dict(xy=near[:, :2].mean(0), z=float(np.median(near[:, 2])), n=len(near),
                             xmax=float(np.percentile(near[:, 0], 97)), xmin=float(np.percentile(near[:, 0], 3)),
                             ymin=float(np.percentile(near[:, 1], 3)), ymax=float(np.percentile(near[:, 1], 97)))
    else:
        res["target"] = None
    # button: non-colored raised object
    z = cloud[..., 2]
    col = np.zeros(z.shape, bool)
    for m in ms.values():
        col |= m
    rr = rgb.astype(int)
    grayish = (np.max(rr, -1) - np.min(rr, -1)) < 40
    hm = (z > 0.012) & (z < 0.08) & ~col & grayish & np.isfinite(z)
    bb = []
    for n, cu, cv, pix in blobs(hm, 8):
        pts = cloud[pix[:, 0], pix[:, 1]]
        ext = np.ptp(pts[:, :2], 0)
        bb.append(dict(n=n, uv=(cu, cv), xy=np.median(pts[:, :2], 0), top=float(np.max(pts[:, 2])), ext=ext))
    res["button_cands"] = bb
    return res, cloud
