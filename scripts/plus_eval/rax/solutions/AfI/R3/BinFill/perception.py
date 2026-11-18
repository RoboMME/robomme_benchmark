"""Perception helpers: decode observation replies, segment cubes/bin/button, back-project."""
import base64, io, json
import numpy as np
from PIL import Image


def parse_reply(reply):
    """Return (meta dict, list of image arrays in content order)."""
    content = reply.get("content", [])
    meta = None
    imgs = {}
    for i, b in enumerate(content):
        if b.get("type") == "text" and meta is None:
            try:
                meta = json.loads(b["text"])
            except Exception:
                meta = {"raw": b.get("text")}
        elif b.get("type") == "image":
            imgs[i] = b
    return meta, imgs


def decode_obs(reply):
    """Decode an observe/step reply into dict with rgb/depth per camera + calib."""
    meta, blocks = parse_reply(reply)
    out = {"meta": meta}
    if meta is None:
        return out
    atts = meta.get("attachments") or []
    for a in atts:
        b = blocks.get(a["content_index"])
        if b is None:
            continue
        raw = base64.b64decode(b["data"])
        im = Image.open(io.BytesIO(raw))
        if a["kind"] == "rgb":
            out[(a["camera"], "rgb")] = np.array(im.convert("RGB"))
        else:
            arr = np.array(im).astype(np.float64)
            mpu = (meta.get("depth", {}).get(a["camera"], {}) or {}).get("meters_per_unit", 0.001)
            out[(a["camera"], "depth")] = arr * mpu
    return out


def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def backproject(obs, cam, us, vs):
    """Pixels (arrays) -> world points Nx3 (nan where invalid)."""
    meta = obs["meta"]
    K = np.array(meta["camera_intrinsics"][cam]["matrix"])
    pose = meta["camera_poses"][cam]
    R = quat_to_R(pose["quaternion_wxyz"])
    t = np.array(pose["position_m"])
    D = obs[(cam, "depth")]
    us = np.asarray(us); vs = np.asarray(vs)
    d = D[np.clip(vs.astype(int), 0, 255), np.clip(us.astype(int), 0, 255)]
    x = (us - K[0, 2]) / K[0, 0] * d
    y = -(vs - K[1, 2]) / K[1, 1] * d
    z = -d
    pc = np.stack([x, y, z], -1)
    pw = pc @ R.T + t
    pw[d <= 0] = np.nan
    return pw


def world_points(obs, cam):
    """Full HxWx3 world point map."""
    vs, us = np.mgrid[0:256, 0:256]
    return backproject(obs, cam, us.ravel(), vs.ravel()).reshape(256, 256, 3)


def color_masks(rgb):
    r = rgb[..., 0].astype(int); g = rgb[..., 1].astype(int); b = rgb[..., 2].astype(int)
    red = (r > 45) & (g * 3 < r) & (b * 3 < r)
    blue = (b > 40) & (r * 3 < b) & (g * 3 < b)
    green = (g > 40) & (r * 2.5 < g) & (b * 2.5 < g)
    return {"red": red, "blue": blue, "green": green}


try:
    from scipy import ndimage as _ndi
except Exception:  # pragma: no cover - fallback when scipy missing
    _ndi = None


def components(mask, min_px=4):
    """Connected components (4-neigh) -> list of (N,2) [row, col] pixel arrays."""
    if _ndi is not None:
        lab, n = _ndi.label(mask)
        if n == 0:
            return []
        idx = np.argsort(lab.ravel(), kind="stable")
        sl = lab.ravel()[idx]
        bounds = np.searchsorted(sl, np.arange(1, n + 2))
        out = []
        for k in range(n):
            ii = idx[bounds[k]:bounds[k + 1]]
            if len(ii) >= min_px:
                out.append(np.stack([ii // mask.shape[1], ii % mask.shape[1]], 1))
        return out
    return _components_bfs(mask, min_px)


def _components_bfs(mask, min_px=4):
    from collections import deque
    H, W = mask.shape
    seen = np.zeros_like(mask, bool)
    comps = []
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
            comps.append(np.array(pix))
    return comps
