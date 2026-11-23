"""Observation decoding, geometry and colour segmentation helpers."""
import base64, io, json
import numpy as np
from PIL import Image


def decode(reply):
    content = reply["content"]
    meta = json.loads(content[0]["text"])
    rgb, depth = {}, {}
    for att in meta.get("attachments", []):
        blk = content[att["content_index"]]
        im = Image.open(io.BytesIO(base64.b64decode(blk["data"])))
        if att["kind"] == "rgb":
            rgb[att["camera"]] = np.asarray(im.convert("RGB"))
        else:
            depth[att["camera"]] = np.asarray(im).astype(np.float32) * 0.001
    return meta, rgb, depth


def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def backproject(meta, cam, us, vs, ds):
    K = np.array(meta["camera_intrinsics"][cam]["matrix"])
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    us, vs, ds = np.asarray(us, float), np.asarray(vs, float), np.asarray(ds, float)
    pc = np.stack([(us - cx) * ds / fx, -(vs - cy) * ds / fy, -ds], -1)
    pose = meta["camera_poses"][cam]
    R = quat_to_R(pose["quaternion_wxyz"])
    return pc @ R.T + np.array(pose["position_m"])


def world_points(meta, depth, cam):
    d = depth[cam]
    H, W = d.shape
    vv, uu = np.mgrid[0:H, 0:W]
    P = backproject(meta, cam, uu.ravel(), vv.ravel(), d.ravel()).reshape(H, W, 3)
    P[d <= 0] = np.nan
    return P


def hsv(rgb):
    return np.asarray(Image.fromarray(rgb).convert("HSV")).astype(np.int32)


def color_mask(rgb, name):
    h = hsv(rgb)
    H, S, V = h[..., 0], h[..., 1], h[..., 2]
    r, g, b = [rgb[..., i].astype(int) for i in range(3)]
    if name == "red":
        return (S > 150) & (V > 80) & ((H < 8) | (H > 245)) & (r > 120)
    if name == "green":
        return (S > 120) & (V > 60) & (H > 60) & (H < 110)
    if name == "blue":
        return (S > 120) & (V > 50) & (H > 150) & (H < 185)
    if name == "white":
        return (S < 45) & (V > 70)
    raise ValueError(name)


def components(mask, min_px=3):
    from scipy import ndimage
    lab, n = ndimage.label(mask)
    out = []
    for i in range(1, n + 1):
        vs, us = np.nonzero(lab == i)
        if len(us) >= min_px:
            out.append({"n": len(us), "u": us.mean(), "v": vs.mean(), "us": us, "vs": vs,
                        "bbox": (us.min(), vs.min(), us.max(), vs.max())})
    out.sort(key=lambda c: -c["n"])
    return out
