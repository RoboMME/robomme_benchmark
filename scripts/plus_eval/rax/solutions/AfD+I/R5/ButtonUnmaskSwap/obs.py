"""Decode MCP observation replies and back-project depth to world points."""
import base64, io, json
import numpy as np
from PIL import Image


def decode(reply):
    blocks = reply["content"]
    meta = json.loads(blocks[0]["text"])
    imgs = {}
    for a in meta.get("attachments", []):
        b = blocks[a["content_index"]]
        im = Image.open(io.BytesIO(base64.b64decode(b["data"])))
        arr = np.array(im)
        if a["kind"] == "rgb":
            arr = np.array(im.convert("RGB"))
        imgs[(a["camera"], a["kind"])] = arr
    return meta, imgs


def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def world_points(meta, imgs, cam, mask=None):
    """Return (N,3) world points and (N,2) pixel coords (u,v) for valid depth pixels in mask."""
    d = imgs[(cam, "depth")].astype(float) * meta["depth"][cam].get("meters_per_unit", 0.001)
    K = np.array(meta["camera_intrinsics"][cam]["matrix"], float)
    pose = meta["camera_poses"][cam]
    R = quat_to_R(pose["quaternion_wxyz"]); t = np.array(pose["position_m"], float)
    valid = d > 0
    if mask is not None:
        valid &= mask
    v, u = np.nonzero(valid)
    z = d[v, u]
    x = (u - K[0, 2]) * z / K[0, 0]
    y = -(v - K[1, 2]) * z / K[1, 1]
    pc = np.stack([x, y, -z], 1)
    pw = pc @ R.T + t
    return pw, np.stack([u, v], 1)


def project(meta, cam, pw):
    """World points (N,3) -> pixel (N,2)."""
    K = np.array(meta["camera_intrinsics"][cam]["matrix"], float)
    pose = meta["camera_poses"][cam]
    R = quat_to_R(pose["quaternion_wxyz"]); t = np.array(pose["position_m"], float)
    pc = (np.asarray(pw) - t) @ R
    z = -pc[:, 2]
    u = pc[:, 0] / z * K[0, 0] + K[0, 2]
    v = -pc[:, 1] / z * K[1, 1] + K[1, 2]
    return np.stack([u, v], 1)
