"""Observation decoding and RGB-D back-projection (public calibration fields only)."""
import base64, io, json
import numpy as np
from PIL import Image


def decode(result):
    """MCP result -> (meta dict, {(camera,kind): ndarray})."""
    blocks = result.get("content", [])
    meta = json.loads(blocks[0]["text"])
    imgs = {}
    for att in meta.get("attachments", []):
        b = blocks[att["content_index"]]
        if b.get("type") != "image":
            continue
        im = Image.open(io.BytesIO(base64.b64decode(b["data"])))
        arr = np.array(im)
        if att["kind"] == "rgb":
            arr = np.array(im.convert("RGB"))
        imgs[(att["camera"], att["kind"])] = arr
    return meta, imgs


def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def cam_params(meta, cam):
    K = np.array(meta["camera_intrinsics"][cam]["matrix"], float)
    pose = meta["camera_poses"][cam]
    R = quat_to_R(pose["quaternion_wxyz"])
    t = np.array(pose["position_m"], float)
    return K, R, t


def depth_m(meta, imgs, cam):
    d = imgs[(cam, "depth")].astype(np.float64) * meta["depth"][cam]["meters_per_unit"]
    return d


def backproject(meta, imgs, cam, us, vs):
    """Pixel arrays -> world points (N,3); NaN where depth invalid."""
    K, R, t = cam_params(meta, cam)
    d = depth_m(meta, imgs, cam)
    us = np.asarray(us); vs = np.asarray(vs)
    dd = d[vs.astype(int), us.astype(int)]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    pc = np.stack([(us - cx) * dd / fx, -(vs - cy) * dd / fy, -dd], -1)
    pw = pc @ R.T + t
    pw[dd <= 0] = np.nan
    return pw


def world_map(meta, imgs, cam):
    """(H,W,3) world coordinates of every pixel."""
    H, W = imgs[(cam, "depth")].shape[:2]
    vs, us = np.mgrid[0:H, 0:W]
    return backproject(meta, imgs, cam, us.ravel(), vs.ravel()).reshape(H, W, 3)


def project(meta, cam, pw):
    """World points (N,3) -> pixel (u,v)."""
    K, R, t = cam_params(meta, cam)
    pc = (np.asarray(pw) - t) @ R
    d = -pc[..., 2]
    u = pc[..., 0] * K[0, 0] / d + K[0, 2]
    v = -pc[..., 1] * K[1, 1] / d + K[1, 2]
    return u, v
