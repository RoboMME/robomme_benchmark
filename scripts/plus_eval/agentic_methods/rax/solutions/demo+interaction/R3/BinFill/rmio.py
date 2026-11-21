"""Packet decoding, geometry and robot I/O helpers for the BinFill controller."""
import base64
import io
import json
import math
import time

import numpy as np
from PIL import Image


def decode(result):
    """MCP result -> (meta dict, {(camera, kind): ndarray})."""
    content = result.get("content", [])
    meta = {}
    for block in content:
        if block.get("type") == "text":
            try:
                meta = json.loads(block["text"])
                break
            except Exception:
                continue
    images = {}
    for att in meta.get("attachments", []) or []:
        idx = att.get("content_index")
        if idx is None or idx >= len(content):
            continue
        block = content[idx]
        data = block.get("data")
        if not data:
            continue
        im = Image.open(io.BytesIO(base64.b64decode(data)))
        if att.get("kind") == "depth":
            arr = np.array(im).astype(np.float32)
            scale = 0.001
            try:
                scale = float(meta["depth"][att["camera"]]["meters_per_unit"])
            except Exception:
                pass
            arr = arr * scale  # metres, 0 invalid
        else:
            arr = np.array(im.convert("RGB"))
        images[(att.get("camera"), att.get("kind"))] = arr
    return meta, images


def quat_to_R(q):
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def aa_to_R(aa):
    aa = np.asarray(aa, float)
    th = np.linalg.norm(aa)
    if th < 1e-9:
        return np.eye(3)
    k = aa / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def R_to_aa(R):
    c = (np.trace(R) - 1) / 2
    c = max(-1.0, min(1.0, c))
    th = math.acos(c)
    if th < 1e-9:
        return np.zeros(3)
    if abs(th - math.pi) < 1e-4:
        # near pi: use the symmetric part
        M = (R + np.eye(3)) / 2
        i = int(np.argmax(np.diag(M)))
        v = M[:, i] / math.sqrt(max(M[i, i], 1e-12))
        return v * th
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return v * th


def Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


class Camera:
    def __init__(self, meta, cam):
        K = np.array(meta["camera_intrinsics"][cam]["matrix"], float)
        self.fx, self.fy, self.cx, self.cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        pose = meta["camera_poses"][cam]
        self.pos = np.array(pose["position_m"], float)
        self.R = quat_to_R(pose["quaternion_wxyz"])

    def backproject(self, depth, us=None, vs=None):
        """depth HxW metres -> HxWx3 world points (nan where invalid) or points at (us, vs)."""
        if us is None:
            H, W = depth.shape
            vs, us = np.mgrid[0:H, 0:W]
            d = depth
        else:
            d = depth[vs, us]
        x = (us - self.cx) * d / self.fx
        y = -(vs - self.cy) * d / self.fy
        z = -d
        cam = np.stack([x, y, z], axis=-1)
        world = cam @ self.R.T + self.pos
        world = np.where((d > 0)[..., None], world, np.nan)
        return world

    def project(self, P):
        """world points Nx3 -> (u, v) arrays."""
        P = np.atleast_2d(P)
        c = (P - self.pos) @ self.R  # camera coords
        d = -c[:, 2]
        u = self.fx * c[:, 0] / d + self.cx
        v = -self.fy * c[:, 1] / d + self.cy
        return u, v, d

    def axis_world(self, i):
        return self.R[:, i]


class Obs:
    """One decoded observation."""

    def __init__(self, meta, images):
        self.meta = meta
        self.step_id = meta.get("step_id")
        self.state = np.array(meta.get("state", [0] * 8), float)
        self.rgb = {c: images.get((c, "rgb")) for c in ("front", "wrist")}
        self.depth = {c: images.get((c, "depth")) for c in ("front", "wrist")}
        self.cams = {}
        for c in ("front", "wrist"):
            try:
                self.cams[c] = Camera(meta, c)
            except Exception:
                pass
        self._pts = {}

    def points(self, cam):
        if cam not in self._pts:
            d = self.depth.get(cam)
            if d is None or cam not in self.cams:
                return None
            self._pts[cam] = self.cams[cam].backproject(d)
        return self._pts[cam]

    @property
    def pos(self):
        return self.state[0:3]

    @property
    def R(self):
        return aa_to_R(self.state[3:6])

    @property
    def grip(self):
        return float(self.state[6] + self.state[7])


def tool_yaw(R):
    """Yaw of the tool x-axis projected into the world XY plane."""
    return math.atan2(R[1, 0], R[0, 0])


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def wrap90(a):
    """Wrap an angle to [-pi/4, pi/4) (cube 4-fold symmetry)."""
    return (a + math.pi / 4) % (math.pi / 2) - math.pi / 4
