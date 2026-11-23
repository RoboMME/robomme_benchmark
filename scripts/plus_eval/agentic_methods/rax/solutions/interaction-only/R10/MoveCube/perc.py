"""Perception helpers: decode MCP observation replies, back-project depth, color masks."""
import base64, io, json
import numpy as np
from PIL import Image


def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def decode(reply):
    """Return (meta, imgs) where imgs[(camera, kind)] = np.array."""
    content = reply["content"]
    meta = json.loads(content[0]["text"])
    imgs = {}
    for att in meta.get("attachments", []):
        blk = content[att["content_index"]]
        data = base64.b64decode(blk["data"])
        im = Image.open(io.BytesIO(data))
        arr = np.array(im)
        if att["kind"] == "rgb":
            arr = np.array(im.convert("RGB"))
        else:
            arr = arr.astype(np.float32) * 0.001
        imgs[(att["camera"], att["kind"])] = arr
    return meta, imgs


def load_frame(files, root="/workspace"):
    imgs = {}
    for f in files:
        im = Image.open(f"{root}/{f['path']}")
        if f["kind"] == "rgb":
            imgs[(f["camera"], "rgb")] = np.array(im.convert("RGB"))
        else:
            imgs[(f["camera"], "depth")] = np.array(im).astype(np.float32) * 0.001
    return imgs


def backproject(meta, cam, depth, us, vs):
    """us, vs pixel arrays -> world points (N,3). depth: HxW metres."""
    K = np.array(meta["camera_intrinsics"][cam]["matrix"])
    pose = meta["camera_poses"][cam]
    R = quat_to_R(pose["quaternion_wxyz"]); t = np.array(pose["position_m"])
    us = np.asarray(us); vs = np.asarray(vs)
    d = depth[vs.astype(int), us.astype(int)]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    pc = np.stack([(us - cx) * d / fx, -(vs - cy) * d / fy, -d], -1)
    return pc @ R.T + t, d


def world_points(meta, cam, depth):
    H, W = depth.shape
    vs, us = np.mgrid[0:H, 0:W]
    P, d = backproject(meta, cam, depth, us.ravel(), vs.ravel())
    return P.reshape(H, W, 3), (d.reshape(H, W) > 0)


def hsv(rgb):
    return np.array(Image.fromarray(rgb).convert("HSV")).astype(np.float32)


def mask_red(rgb):
    r, g, b = [rgb[..., i].astype(int) for i in range(3)]
    return (r > 120) & (g < 60) & (b < 60) & (r > 2.5 * g)


def mask_purple(rgb):
    r, g, b = [rgb[..., i].astype(int) for i in range(3)]
    return (b > 120) & (r > 100) & (g < 0.75 * b) & (b > g + 50) & (r < b + 40)


def mask_teal(rgb):
    r, g, b = [rgb[..., i].astype(int) for i in range(3)]
    return (g > 1.5 * r) & (abs(g - b) < 20) & (g > 70) & (r < 90)


def mask_shaft(rgb):
    r, g, b = [rgb[..., i].astype(int) for i in range(3)]
    grey = (abs(r - g) < 25) & (b >= g) & (b - r < 35) & (r > 100)
    return grey


def components(mask):
    from scipy import ndimage
    lab, n = ndimage.label(mask)
    return [(lab == i) for i in range(1, n + 1)]
