"""Decode RoboMME MCP replies into numpy observations; RGB-D back-projection."""
import base64, io, json, os
import numpy as np
from PIL import Image

WS_ROOT = os.environ.get('BUS_WS_ROOT', '/workspace')


def _decode_png(data_b64):
    return Image.open(io.BytesIO(base64.b64decode(data_b64)))


def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class View:
    """One camera: rgb uint8 HxWx3, depth metres HxW (0 invalid), K, R, t (OpenGL cam->world)."""

    def __init__(self, name, rgb, depth_m, K, pos, quat):
        self.name, self.rgb, self.depth = name, rgb, depth_m
        self.K = np.asarray(K, dtype=float)
        self.R = quat_to_R(quat)
        self.t = np.asarray(pos, dtype=float)

    def world_points(self):
        """Return (H,W,3) world points and valid mask."""
        H, W = self.depth.shape
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        u, v = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
        d = self.depth
        cam = np.stack([(u - cx) * d / fx, -(v - cy) * d / fy, -d], axis=-1)
        world = cam @ self.R.T + self.t
        return world, d > 0

    def project(self, P):
        """World points (N,3) -> pixel (N,2) u,v and depth (N,)."""
        P = np.atleast_2d(np.asarray(P, dtype=float))
        cam = (P - self.t) @ self.R  # R^T (P - t)
        d = -cam[:, 2]
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        u = cam[:, 0] * fx / np.maximum(d, 1e-6) + cx
        v = -cam[:, 1] * fy / np.maximum(d, 1e-6) + cy
        return np.stack([u, v], 1), d


class Obs:
    pass


def parse_meta(result):
    for b in result.get('content', []):
        if b.get('type') == 'text':
            try:
                return json.loads(b['text'])
            except Exception:
                continue
    return {}


def decode_reply(result, want_images=True):
    """MCP result (observe/step) -> Obs with meta, state, views dict."""
    meta = parse_meta(result)
    o = Obs()
    o.meta = meta
    o.step_id = meta.get('step_id')
    o.state = np.asarray(meta.get('state', [0] * 8), dtype=float)
    o.ended = bool(meta.get('episode_ended', False))
    o.task_complete = None
    tr = meta.get('transition') or {}
    for s in tr.get('steps', []) or []:
        if 'task_complete' in s:
            o.task_complete = s.get('task_complete')
        if s.get('episode_ended'):
            o.ended = True
    o.limits = meta.get('episode_limits', {})
    o.views = {}
    if not want_images:
        return o
    content = result.get('content', [])
    rgb, dep = {}, {}
    for att in meta.get('attachments', []):
        idx = att.get('content_index')
        if idx is None or idx >= len(content):
            continue
        blk = content[idx]
        if blk.get('type') != 'image':
            continue
        im = _decode_png(blk['data'])
        if att.get('kind') == 'rgb':
            rgb[att['camera']] = np.asarray(im.convert('RGB'))
        elif att.get('kind') == 'depth':
            mpu = (meta.get('depth', {}).get(att['camera'], {}) or {}).get('meters_per_unit', 0.001)
            dep[att['camera']] = np.asarray(im).astype(np.float32) * float(mpu)
    o.views = _make_views(meta, rgb, dep)
    return o


def _make_views(meta, rgb, dep):
    views = {}
    intr = meta.get('camera_intrinsics', {}) or {}
    poses = meta.get('camera_poses', {}) or {}
    for cam in rgb:
        if cam in dep and cam in intr and cam in poses:
            views[cam] = View(cam, rgb[cam], dep[cam], intr[cam]['matrix'],
                              poses[cam]['position_m'], poses[cam]['quaternion_wxyz'])
    return views


def load_manifest(path):
    p = path if os.path.isabs(path) else os.path.join(WS_ROOT, path)
    with open(p) as f:
        return json.load(f)


def frame_to_obs(frame):
    """A manifest frame record -> Obs (reads files from disk)."""
    o = Obs()
    o.meta = frame
    o.step_id = frame.get('step_id')
    o.phase = frame.get('phase')
    o.state = np.asarray(frame.get('state', [0] * 8), dtype=float)
    o.ended = False
    o.task_complete = None
    rgb, dep = {}, {}
    for f in frame.get('files', []):
        p = f['path'] if os.path.isabs(f['path']) else os.path.join(WS_ROOT, f['path'])
        if not os.path.exists(p):
            continue
        im = Image.open(p)
        if f['kind'] == 'rgb':
            rgb[f['camera']] = np.asarray(im.convert('RGB'))
        elif f['kind'] == 'depth':
            mpu = (frame.get('depth', {}).get(f['camera'], {}) or {}).get('meters_per_unit', 0.001)
            dep[f['camera']] = np.asarray(im).astype(np.float32) * float(mpu)
    o.views = _make_views(frame, rgb, dep)
    return o
