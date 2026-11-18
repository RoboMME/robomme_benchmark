"""Packet parsing, RGB-D back-projection and colour-agnostic segmentation."""
import base64
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

WORKSPACE = Path("/workspace")


def quat_to_R(q):
    w, x, y, z = q
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class View:
    """One camera view: rgb (H,W,3 uint8), depth (H,W metres, 0 invalid), calibration."""

    def __init__(self, rgb, depth_m, K, pos, quat):
        self.rgb = rgb
        self.depth = depth_m
        self.K = np.asarray(K, dtype=float)
        self.pos = np.asarray(pos, dtype=float)
        self.R = quat_to_R(quat)
        self._world = None

    @property
    def valid(self):
        return self.depth > 0

    def world(self):
        """(H,W,3) world points; invalid pixels are NaN."""
        if self._world is None:
            H, W = self.depth.shape
            fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
            u, v = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
            d = self.depth
            cam = np.stack([(u - cx) * d / fx, -(v - cy) * d / fy, -d], axis=-1)
            w = cam @ self.R.T + self.pos
            w[~self.valid] = np.nan
            self._world = w
        return self._world

    def ray(self, u, v):
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        d = np.array([(u - cx) / fx, -(v - cy) / fy, -1.0])
        d = self.R @ d
        return d / np.linalg.norm(d)

    def cam_x_world(self):
        return self.R @ np.array([1.0, 0.0, 0.0])


def _decode_png(data_b64):
    return Image.open(io.BytesIO(base64.b64decode(data_b64)))


def parse_reply(reply):
    """Return (meta dict, {camera: View}) from an MCP tool result."""
    content = reply.get("content", [])
    meta = None
    for b in content:
        if b.get("type") == "text":
            try:
                meta = json.loads(b["text"])
                break
            except Exception:
                continue
    if meta is None:
        raise RuntimeError("no JSON text block in reply")
    views = {}
    att = meta.get("attachments") or []
    imgs = {}
    for a in att:
        idx = a.get("content_index")
        if idx is None or idx >= len(content):
            continue
        blk = content[idx]
        if blk.get("type") != "image":
            continue
        imgs[(a.get("camera"), a.get("kind"))] = blk.get("data")
    intr = meta.get("camera_intrinsics") or {}
    poses = meta.get("camera_poses") or {}
    dmeta = meta.get("depth") or {}
    for cam in ("front", "wrist"):
        if (cam, "rgb") not in imgs or (cam, "depth") not in imgs:
            continue
        if cam not in intr or cam not in poses:
            continue
        rgb = np.array(_decode_png(imgs[(cam, "rgb")]).convert("RGB"))
        dimg = np.array(_decode_png(imgs[(cam, "depth")])).astype(np.float64)
        mpu = float((dmeta.get(cam) or {}).get("meters_per_unit", 0.001))
        views[cam] = View(rgb, dimg * mpu, intr[cam]["matrix"], poses[cam]["position_m"],
                          poses[cam]["quaternion_wxyz"])
    return meta, views


def load_frame_view(frame, cam):
    """Build a View for camera `cam` from a published frame-manifest entry."""
    files = {(f["camera"], f["kind"]): f["path"] for f in frame.get("files", [])}
    if (cam, "rgb") not in files:
        return None
    rgb = np.array(Image.open(WORKSPACE / files[(cam, "rgb")]).convert("RGB"))
    depth = None
    if (cam, "depth") in files and (WORKSPACE / files[(cam, "depth")]).exists():
        mpu = float(((frame.get("depth") or {}).get(cam) or {}).get("meters_per_unit", 0.001))
        depth = np.array(Image.open(WORKSPACE / files[(cam, "depth")])).astype(np.float64) * mpu
    intr = (frame.get("camera_intrinsics") or {}).get(cam)
    pose = (frame.get("camera_poses") or {}).get(cam)
    if depth is None or intr is None or pose is None:
        return None
    return View(rgb, depth, intr["matrix"], pose["position_m"], pose["quaternion_wxyz"])


# ---------------------------------------------------------------- masks

def white_mask(rgb, lo=185, spread=40):
    r = rgb.astype(int)
    mn = r.min(axis=2)
    mx = r.max(axis=2)
    return (mn >= lo) & ((mx - mn) <= spread)


def dilate(mask, r=1):
    out = mask.copy()
    H, W = mask.shape
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx == 0 and dy == 0:
                continue
            sh = np.zeros_like(mask)
            ys = slice(max(0, dy), H + min(0, dy))
            yd = slice(max(0, -dy), H + min(0, -dy))
            xs = slice(max(0, dx), W + min(0, dx))
            xd = slice(max(0, -dx), W + min(0, -dx))
            sh[ys, xs] = mask[yd, xd]
            out |= sh
    return out


def components(mask, min_area=1):
    """Connected components (8-connectivity). Returns list of (ys, xs) arrays."""
    try:
        import cv2
        n, lab = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
        out = []
        for i in range(1, n):
            ys, xs = np.nonzero(lab == i)
            if len(ys) >= min_area:
                out.append((ys, xs))
        return out
    except Exception:
        from scipy import ndimage
        lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
        out = []
        for i in range(1, n + 1):
            ys, xs = np.nonzero(lab == i)
            if len(ys) >= min_area:
                out.append((ys, xs))
        return out


def table_height(view):
    """Mode of world z over valid pixels (the table dominates the front view)."""
    z = view.world()[..., 2]
    z = z[np.isfinite(z)]
    if len(z) == 0:
        return 0.0
    lo, hi = np.percentile(z, 2), np.percentile(z, 98)
    hist, edges = np.histogram(z, bins=200, range=(lo - 0.01, hi + 0.01))
    i = int(np.argmax(hist))
    sel = z[(z >= edges[i] - 0.004) & (z <= edges[i + 1] + 0.004)]
    return float(np.median(sel))
