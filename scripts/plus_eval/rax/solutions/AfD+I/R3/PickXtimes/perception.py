"""Observation decoding, back-projection and colour segmentation."""
import base64, io, json, math
import numpy as np
from PIL import Image


def decode(reply):
    blocks = reply["content"]
    meta = json.loads(blocks[0]["text"])
    imgs = {}
    for a in meta.get("attachments", []):
        b = blocks[a["content_index"]]
        data = base64.b64decode(b["data"])
        im = Image.open(io.BytesIO(data))
        arr = np.array(im)
        if a["kind"] == "rgb":
            arr = np.array(im.convert("RGB"))
        else:
            arr = arr.astype(np.float32) * 0.001
        imgs[(a["camera"], a["kind"])] = arr
    return meta, imgs


def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def world_points(meta, imgs, cam):
    """HxWx3 world points for every pixel of camera `cam` (NaN where depth invalid)."""
    d = imgs[(cam, "depth")]
    K = np.array(meta["camera_intrinsics"][cam]["matrix"])
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    H, W = d.shape
    u, v = np.meshgrid(np.arange(W) + 0.0, np.arange(H) + 0.0)
    # pixel centre convention: use integer pixel index (matches typical renderers within <1px)
    xc = (u - cx) * d / fx
    yc = -(v - cy) * d / fy
    zc = -d
    P = np.stack([xc, yc, zc], -1)
    pose = meta["camera_poses"][cam]
    R = quat_to_R(pose["quaternion_wxyz"])
    t = np.array(pose["position_m"])
    Pw = P @ R.T + t
    Pw[d <= 0] = np.nan
    return Pw


def hsv(rgb):
    a = rgb.astype(np.float32) / 255.0
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx = a.max(-1); mn = a.min(-1); df = mx - mn + 1e-6
    h = np.where(mx == r, ((g - b) / df) % 6, np.where(mx == g, (b - r) / df + 2, (r - g) / df + 4)) * 60
    s = np.where(mx > 0, df / (mx + 1e-6), 0)
    return h, s, mx


COLOR_HUES = {"red": [(0, 15), (345, 360)], "yellow": [(45, 70)], "green": [(90, 150)],
              "blue": [(210, 250)], "purple": [(265, 320)], "magenta": [(280, 330)],
              "orange": [(16, 40)], "cyan": [(170, 200)]}


def color_mask(rgb, color, smin=0.45, vmin=0.25):
    h, s, v = hsv(rgb)
    m = np.zeros(h.shape, bool)
    for lo, hi in COLOR_HUES[color]:
        m |= (h >= lo) & (h < hi)
    return m & (s > smin) & (v > vmin)


def components(mask, min_px=4):
    """Simple 4-connected component labelling; returns list of (count, ys, xs)."""
    H, W = mask.shape
    lab = np.zeros((H, W), np.int32)
    out = []
    cur = 0
    ys0, xs0 = np.nonzero(mask)
    for y0, x0 in zip(ys0, xs0):
        if lab[y0, x0]:
            continue
        cur += 1
        stack = [(y0, x0)]; lab[y0, x0] = cur; py = []; px = []
        while stack:
            y, x = stack.pop(); py.append(y); px.append(x)
            for yy, xx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                if 0 <= yy < H and 0 <= xx < W and mask[yy, xx] and not lab[yy, xx]:
                    lab[yy, xx] = cur; stack.append((yy, xx))
        if len(py) >= min_px:
            out.append((len(py), np.array(py), np.array(px)))
    out.sort(key=lambda c: -c[0])
    return out


def yaw_of_square(xy):
    """Yaw (rad, in [0, pi/2)) of a square footprint from its world xy points via min-area box."""
    best = (1e9, 0.0)
    c = xy - xy.mean(0)
    for deg in np.arange(0, 90, 2.0):
        a = math.radians(deg)
        R = np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]])
        p = c @ R.T
        area = np.ptp(p[:, 0]) * np.ptp(p[:, 1])
        if area < best[0]:
            best = (area, a)
    return best[1]
