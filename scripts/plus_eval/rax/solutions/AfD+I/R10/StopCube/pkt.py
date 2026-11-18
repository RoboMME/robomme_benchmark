"""Packet decoding helpers (MCP content blocks -> dict, numpy images, geometry)."""
import base64, io, json
import numpy as np
from PIL import Image


def parse(blocks):
    """Return (meta, images) where images[(camera, kind)] = np.ndarray."""
    if isinstance(blocks, dict) and 'content' in blocks:
        blocks = blocks['content']
    meta = None
    for b in blocks:
        if b.get('type') == 'text':
            try:
                meta = json.loads(b['text']); break
            except Exception:
                continue
    imgs = {}
    if meta is None:
        return {}, imgs
    for att in meta.get('attachments', []):
        i = att['content_index']
        if i >= len(blocks):
            continue
        b = blocks[i]
        data = b.get('data')
        if data is None:
            continue
        im = Image.open(io.BytesIO(base64.b64decode(data)))
        if att['kind'] == 'rgb':
            arr = np.array(im.convert('RGB'))
        else:
            arr = np.array(im).astype(np.float32) * meta.get('depth', {}).get(att['camera'], {}).get('meters_per_unit', 0.001)
        imgs[(att['camera'], att['kind'])] = arr
    return meta, imgs


def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def backproject(meta, depth, cam='front'):
    """World xyz (H,W,3) for every pixel; NaN where depth invalid."""
    K = np.array(meta['camera_intrinsics'][cam]['matrix'], float)
    pose = meta['camera_poses'][cam]
    R = quat_to_R(pose['quaternion_wxyz']); t = np.array(pose['position_m'], float)
    H, W = depth.shape
    v, u = np.mgrid[:H, :W].astype(float)
    d = depth.astype(float)
    xc = (u - K[0, 2]) * d / K[0, 0]
    yc = -(v - K[1, 2]) * d / K[1, 1]
    zc = -d
    P = np.stack([xc, yc, zc], -1) @ R.T + t
    P[d <= 0] = np.nan
    return P


def project(meta, pts, cam='front'):
    """World points (N,3) -> pixel (u,v)."""
    K = np.array(meta['camera_intrinsics'][cam]['matrix'], float)
    pose = meta['camera_poses'][cam]
    R = quat_to_R(pose['quaternion_wxyz']); t = np.array(pose['position_m'], float)
    pc = (np.asarray(pts, float) - t) @ R  # world->cam
    d = -pc[:, 2]
    u = pc[:, 0] / d * K[0, 0] + K[0, 2]
    v = -pc[:, 1] / d * K[1, 1] + K[1, 2]
    return np.stack([u, v], -1)
