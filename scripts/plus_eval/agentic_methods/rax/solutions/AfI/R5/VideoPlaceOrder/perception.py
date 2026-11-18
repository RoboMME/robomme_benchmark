import base64, io, json
import numpy as np
from PIL import Image


def parse_reply(reply):
    """Return (meta, images) where images[(camera, kind)] = np.array."""
    content = reply.get("content", [])
    meta = None
    for b in content:
        if b.get("type") == "text":
            try:
                meta = json.loads(b["text"])
                break
            except Exception:
                continue
    images = {}
    if meta is None:
        return None, images
    for att in meta.get("attachments", []) or []:
        idx = att.get("content_index")
        if idx is None or idx >= len(content):
            continue
        blk = content[idx]
        data = blk.get("data")
        if not data:
            continue
        im = Image.open(io.BytesIO(base64.b64decode(data)))
        images[(att.get("camera"), att.get("kind"))] = np.array(im)
    return meta, images


def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def backproject(meta, cam, us, vs, depth_m):
    K = np.array(meta["camera_intrinsics"][cam]["matrix"], dtype=float)
    pose = meta["camera_poses"][cam]
    R = quat_to_R(pose["quaternion_wxyz"])
    t = np.array(pose["position_m"], dtype=float)
    us = np.asarray(us, float); vs = np.asarray(vs, float); d = np.asarray(depth_m, float)
    pc = np.stack([(us - K[0, 2]) * d / K[0, 0], -(vs - K[1, 2]) * d / K[1, 1], -d], axis=-1)
    return pc @ R.T + t


def world_points(meta, images, cam):
    """Full HxWx3 world point map and validity mask for one camera."""
    dep = images.get((cam, "depth"))
    if dep is None:
        return None, None
    scale = meta.get("depth", {}).get(cam, {}).get("meters_per_unit", 0.001)
    d = dep.astype(float) * scale
    H, W = d.shape[:2]
    vs, us = np.mgrid[0:H, 0:W]
    P = backproject(meta, cam, us, vs, d)
    return P, d > 0


def components(mask, min_px=3):
    """Simple connected components (4-neighbour) without scipy."""
    H, W = mask.shape
    lab = np.zeros((H, W), int)
    comps = []
    n = 0
    for y0 in range(H):
        for x0 in range(W):
            if mask[y0, x0] and lab[y0, x0] == 0:
                n += 1
                stack = [(y0, x0)]
                lab[y0, x0] = n
                pix = []
                while stack:
                    y, x = stack.pop()
                    pix.append((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        yy, xx = y + dy, x + dx
                        if 0 <= yy < H and 0 <= xx < W and mask[yy, xx] and lab[yy, xx] == 0:
                            lab[yy, xx] = n
                            stack.append((yy, xx))
                if len(pix) >= min_px:
                    comps.append(np.array(pix))
    return comps


def purple_mask(rgb):
    r, g, b = [rgb[..., i].astype(int) for i in range(3)]
    return (b > 120) & (r > 90) & (g < 0.75 * b) & (g < 0.8 * r) & (b - g > 50)


def red_mask(rgb):
    r, g, b = [rgb[..., i].astype(int) for i in range(3)]
    return (r > 120) & (g < 50) & (b < 50)


def detect_blobs(meta, images, cam, maskfn, min_px=4, zmax=None):
    rgb = images[(cam, "rgb")][..., :3]
    P, valid = world_points(meta, images, cam)
    m = maskfn(rgb) & valid
    out = []
    for pix in components(m, min_px):
        pts = P[pix[:, 0], pix[:, 1]]
        if zmax is not None:
            pts = pts[pts[:, 2] < zmax]
            if len(pts) == 0:
                continue
        c = np.median(pts, axis=0)
        out.append({"xyz": c.tolist(), "n": int(len(pix)), "uv": [float(pix[:, 1].mean()), float(pix[:, 0].mean())],
                    "zmax": float(pts[:, 2].max())})
    return out


def cluster_points(points, radius=0.03):
    clusters = []
    for p in points:
        p = np.asarray(p, float)
        for c in clusters:
            if np.linalg.norm(c["sum"][:2] / c["n"] - p[:2]) < radius:
                c["sum"] += p; c["n"] += 1
                break
        else:
            clusters.append({"sum": p.copy(), "n": 1})
    return [(c["sum"] / c["n"]).tolist() for c in clusters]
