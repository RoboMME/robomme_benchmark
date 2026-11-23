"""Decode live observations and back-project pixels to world points."""
import base64, io, json
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as Rr

def decode(reply):
    blocks = reply.get('content', [])
    meta = None
    for b in blocks:
        if b.get('type') == 'text':
            try:
                meta = json.loads(b['text']); break
            except Exception:
                pass
    imgs = {}
    for att in (meta or {}).get('attachments', []):
        b = blocks[att['content_index']]
        data = base64.b64decode(b['data'])
        im = Image.open(io.BytesIO(data))
        if att['kind'] == 'rgb':
            imgs[(att['camera'], 'rgb')] = np.array(im.convert('RGB'))
        else:
            imgs[(att['camera'], 'depth')] = np.array(im).astype(np.float32) * 0.001
    return meta, imgs

def backproject(meta, cam, us, vs, depth):
    K = np.array(meta['camera_intrinsics'][cam]['matrix'], float)
    P = meta['camera_poses'][cam]
    q = P['quaternion_wxyz']
    R = Rr.from_quat([q[1], q[2], q[3], q[0]])
    us = np.asarray(us, float); vs = np.asarray(vs, float)
    d = depth[np.clip(vs.round().astype(int), 0, depth.shape[0] - 1), np.clip(us.round().astype(int), 0, depth.shape[1] - 1)]
    x = (us - K[0, 2]) * d / K[0, 0]
    y = -(vs - K[1, 2]) * d / K[1, 1]
    z = -d
    pts = R.apply(np.stack([x, y, z], 1)) + np.array(P['position_m'])
    return pts, d > 0

def project(meta, cam, pts):
    K = np.array(meta['camera_intrinsics'][cam]['matrix'], float)
    P = meta['camera_poses'][cam]
    q = P['quaternion_wxyz']
    R = Rr.from_quat([q[1], q[2], q[3], q[0]])
    c = R.inv().apply(np.asarray(pts, float) - np.array(P['position_m']))
    d = -c[:, 2]
    u = c[:, 0] / d * K[0, 0] + K[0, 2]
    v = -c[:, 1] / d * K[1, 1] + K[1, 2]
    return np.stack([u, v], 1), d

def mask_world(meta, imgs, cam, mask):
    vs, us = np.nonzero(mask)
    if len(us) == 0:
        return np.zeros((0, 3))
    pts, ok = backproject(meta, cam, us, vs, imgs[(cam, 'depth')])
    return pts[ok]

def cube_pose_from_points(pts):
    """Top-face points -> (center xyz, yaw mod 90deg in radians in [-pi/4, pi/4])."""
    if len(pts) < 10:
        return None
    ztop = np.percentile(pts[:, 2], 90)
    top = pts[pts[:, 2] > ztop - 0.006]
    if len(top) < 8:
        top = pts
    xy = top[:, :2]
    c = xy.mean(0)
    best = None
    for ang in np.radians(np.arange(-45, 45, 1.0)):
        ca, sa = np.cos(ang), np.sin(ang)
        rx = (xy - c) @ np.array([ca, sa]); ry = (xy - c) @ np.array([-sa, ca])
        area = (np.percentile(rx, 98) - np.percentile(rx, 2)) * (np.percentile(ry, 98) - np.percentile(ry, 2))
        if best is None or area < best[0]:
            best = (area, ang, rx, ry)
    _, ang, rx, ry = best
    ca, sa = np.cos(ang), np.sin(ang)
    # centre of the bounding box in rotated frame
    cx = (np.percentile(rx, 98) + np.percentile(rx, 2)) / 2; cy = (np.percentile(ry, 98) + np.percentile(ry, 2)) / 2
    center = c + cx * np.array([ca, sa]) + cy * np.array([-sa, ca])
    size = (np.percentile(rx, 98) - np.percentile(rx, 2), np.percentile(ry, 98) - np.percentile(ry, 2))
    return dict(center=np.array([center[0], center[1], ztop]), yaw=float(ang), size=size, n=len(top))
