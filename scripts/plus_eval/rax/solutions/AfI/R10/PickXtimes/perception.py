import base64, io, json
import numpy as np
from PIL import Image


def parse(reply):
    """Decode an MCP reply into (meta, images) where images[(camera, kind)] = np array."""
    content = reply["content"]
    meta = json.loads(content[0]["text"])
    imgs = {}
    for att in meta.get("attachments", []):
        blk = content[att["content_index"]]
        data = base64.b64decode(blk["data"])
        im = Image.open(io.BytesIO(data))
        arr = np.array(im)
        if att["kind"] == "depth":
            arr = arr.astype(np.float32) * 0.001
        else:
            arr = np.array(im.convert("RGB"))
        imgs[(att["camera"], att["kind"])] = arr
    return meta, imgs


def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def backproject(meta, depth, cam, us, vs):
    K = np.array(meta["camera_intrinsics"][cam]["matrix"])
    pose = meta["camera_poses"][cam]
    R = quat_to_R(pose["quaternion_wxyz"])
    t = np.array(pose["position_m"])
    us = np.asarray(us, float); vs = np.asarray(vs, float)
    d = depth[vs.astype(int), us.astype(int)]
    x = (us - K[0, 2]) * d / K[0, 0]
    y = -(vs - K[1, 2]) * d / K[1, 1]
    pc = np.stack([x, y, -d], -1)
    pw = pc @ R.T + t
    return pw, d


def world_map(meta, imgs, cam):
    depth = imgs[(cam, "depth")]
    H, W = depth.shape
    vs, us = np.mgrid[0:H, 0:W]
    pw, d = backproject(meta, depth, cam, us.ravel(), vs.ravel())
    pw = pw.reshape(H, W, 3)
    valid = depth > 0
    return pw, valid


def hsv(rgb):
    im = Image.fromarray(rgb).convert("HSV")
    return np.array(im).astype(np.float32)  # H,S,V in 0..255


# hue ranges in PIL 0..255 scale
COLOR_HUES = {
    "red": [(0, 10), (245, 256)],
    "orange": [(10, 22)],
    "yellow": [(30, 50)],
    "green": [(60, 115)],
    "cyan": [(115, 135)],
    "blue": [(140, 180)],
    "purple": [(185, 215)],
    "magenta": [(185, 235)],
    "pink": [(215, 245)],
}


def color_mask(rgb, color, smin=120, vmin=60):
    h = hsv(rgb)
    m = np.zeros(rgb.shape[:2], bool)
    for lo, hi in COLOR_HUES[color]:
        m |= (h[..., 0] >= lo) & (h[..., 0] < hi)
    return m & (h[..., 1] >= smin) & (h[..., 2] >= vmin)


def components(mask, min_px=4):
    from scipy import ndimage
    lab, n = ndimage.label(mask)
    out = []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(lab == i)
        if len(xs) >= min_px:
            out.append({"n": len(xs), "u": xs.mean(), "v": ys.mean(), "xs": xs, "ys": ys})
    out.sort(key=lambda c: -c["n"])
    return out
