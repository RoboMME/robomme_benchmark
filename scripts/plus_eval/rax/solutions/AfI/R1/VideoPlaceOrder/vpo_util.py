"""Generic helpers: MCP reply parsing, rotations, camera models, depth loading.

Everything here is defensive: the exact packet layout was never observed live
(AfI, R=1), so every accessor searches the JSON for the documented key names and
falls back gracefully.
"""
import base64
import io
import json
import math
import os
from pathlib import Path

import numpy as np

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None

WORKSPACE = Path("/workspace")


# ----------------------------------------------------------------------------
# JSON search helpers
# ----------------------------------------------------------------------------
def bfs_find(obj, key, pred=None, max_nodes=200000):
    """Breadth-first search for the shallowest value stored under `key`."""
    queue = [obj]
    seen = 0
    while queue:
        nxt = []
        for node in queue:
            seen += 1
            if seen > max_nodes:
                return None
            if isinstance(node, dict):
                if key in node:
                    v = node[key]
                    if pred is None or pred(v):
                        return v
                nxt.extend(node.values())
            elif isinstance(node, list):
                nxt.extend(node)
        queue = nxt
    return None


def bfs_find_all(obj, key, max_nodes=200000):
    out = []
    queue = [obj]
    seen = 0
    while queue:
        nxt = []
        for node in queue:
            seen += 1
            if seen > max_nodes:
                return out
            if isinstance(node, dict):
                if key in node:
                    out.append(node[key])
                nxt.extend(node.values())
            elif isinstance(node, list):
                nxt.extend(node)
        queue = nxt
    return out


def is_num_list(v, n=None):
    if not isinstance(v, (list, tuple)):
        return False
    if n is not None and len(v) != n:
        return False
    try:
        return all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in v)
    except Exception:
        return False


def reply_meta(result):
    """First JSON text block of an MCP tool result (or structuredContent)."""
    if not isinstance(result, dict):
        return {}
    for block in result.get("content", []) or []:
        if isinstance(block, dict) and block.get("type") == "text":
            try:
                v = json.loads(block.get("text", ""))
                if isinstance(v, dict):
                    return v
            except Exception:
                continue
    sc = result.get("structuredContent")
    if isinstance(sc, dict):
        return sc
    return {}


def reply_texts(result):
    out = []
    for block in (result or {}).get("content", []) or []:
        if isinstance(block, dict) and block.get("type") == "text":
            out.append(block.get("text", ""))
    return out


def decode_image_bytes(data):
    if Image is None:
        return None
    try:
        raw = base64.b64decode(data)
        im = Image.open(io.BytesIO(raw))
        im.load()
        return im
    except Exception:
        return None


def reply_images(result, meta=None):
    """Return list of dicts {camera, kind, image(PIL)} for image blocks.

    Camera/kind come from block annotations, preceding short text labels, a
    metadata list describing attachments, or (last resort) order + pixel mode.
    """
    blocks = (result or {}).get("content", []) or []
    imgs = []
    last_label = ""
    for block in blocks:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            t = block.get("text", "")
            if len(t) < 200:
                last_label = t.lower()
            continue
        if block.get("type") != "image":
            continue
        im = decode_image_bytes(block.get("data", ""))
        if im is None:
            continue
        label = last_label
        for k in ("annotations", "_meta", "meta", "metadata", "name", "camera", "title"):
            if k in block:
                try:
                    label += " " + json.dumps(block[k]).lower()
                except Exception:
                    pass
        cam = "front" if "front" in label else ("wrist" if "wrist" in label else None)
        kind = "depth" if "depth" in label else ("rgb" if ("rgb" in label or "color" in label) else None)
        if kind is None:
            kind = "rgb" if im.mode in ("RGB", "RGBA", "P", "L") and im.mode != "I;16" else "depth"
            if im.mode in ("I;16", "I;16B", "I", "F"):
                kind = "depth"
        imgs.append({"camera": cam, "kind": kind, "image": im})
        last_label = ""
    # Metadata attachment list in order?
    if meta and any(d["camera"] is None for d in imgs):
        desc = None
        for key in ("attachments", "images", "image_blocks", "cameras_order"):
            v = bfs_find(meta, key)
            if isinstance(v, list) and len(v) == len(imgs) and all(isinstance(x, dict) for x in v):
                desc = v
                break
        if desc:
            for d, m in zip(imgs, desc):
                c = str(m.get("camera", m.get("name", ""))).lower()
                if d["camera"] is None and c:
                    d["camera"] = "front" if "front" in c else ("wrist" if "wrist" in c else None)
                kd = str(m.get("kind", m.get("type", ""))).lower()
                if kd:
                    d["kind"] = "depth" if "depth" in kd else "rgb"
    # Order fallback per kind: front first, wrist second.
    for kind in ("rgb", "depth"):
        group = [d for d in imgs if d["kind"] == kind]
        used = {d["camera"] for d in group if d["camera"]}
        order = [c for c in ("front", "wrist") if c not in used]
        for d in group:
            if d["camera"] is None and order:
                d["camera"] = order.pop(0)
    return imgs


def pil_to_rgb(im):
    return np.asarray(im.convert("RGB"), dtype=np.uint8)


def pil_to_depth_m(im, meters_per_unit=0.001):
    a = np.asarray(im)
    if a.ndim == 3:
        a = a[..., 0]
    a = a.astype(np.float64)
    if a.max() < 50 and a.max() > 0 and meters_per_unit == 0.001:
        # Already metres (float image)
        d = a
    else:
        d = a * float(meters_per_unit)
    d[~np.isfinite(d)] = 0.0
    d[d <= 0] = 0.0
    return d


def resolve_path(p, base_dirs=()):
    p = str(p)
    cands = []
    if os.path.isabs(p):
        cands.append(Path(p))
    cands.append(WORKSPACE / p)
    for b in base_dirs:
        cands.append(Path(b) / p)
    for c in cands:
        try:
            if c.is_file():
                return c
        except Exception:
            pass
    return None


def load_png_depth(path, meters_per_unit=0.001):
    with Image.open(path) as im:
        im.load()
        return pil_to_depth_m(im, meters_per_unit if meters_per_unit else 0.001)


def load_png_rgb(path):
    with Image.open(path) as im:
        return pil_to_rgb(im)


# ----------------------------------------------------------------------------
# Rotations
# ----------------------------------------------------------------------------
def rotvec_to_mat(r):
    r = np.asarray(r, dtype=float)
    th = float(np.linalg.norm(r))
    if th < 1e-12:
        return np.eye(3)
    k = r / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def mat_to_rotvec(R):
    R = np.asarray(R, dtype=float)
    c = (np.trace(R) - 1.0) / 2.0
    c = min(1.0, max(-1.0, c))
    th = math.acos(c)
    if th < 1e-9:
        return np.zeros(3)
    if th > math.pi - 1e-4:
        # near pi: axis from the diagonal
        M = (R + np.eye(3)) / 2.0
        i = int(np.argmax(np.diag(M)))
        axis = M[:, i] / math.sqrt(max(M[i, i], 1e-12))
        axis /= np.linalg.norm(axis)
        return axis * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return w * th


def quat_wxyz_to_mat(q):
    w, x, y, z = [float(v) for v in q]
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def rotz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def wrap_pi(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def wrap_quarter(a):
    """Wrap an angle into [-pi/4, pi/4) (square symmetry)."""
    q = math.pi / 2
    return (a + q / 2) % q - q / 2


# ----------------------------------------------------------------------------
# Camera models
# ----------------------------------------------------------------------------
def parse_intrinsics(v, width=256, height=256):
    """Return 3x3 K or None."""
    try:
        if v is None:
            return None
        if isinstance(v, dict):
            low = {str(k).lower(): val for k, val in v.items()}
            if "fx" in low and "fy" in low:
                cx = low.get("cx", low.get("ppx", width / 2.0))
                cy = low.get("cy", low.get("ppy", height / 2.0))
                return np.array([[float(low["fx"]), 0, float(cx)], [0, float(low["fy"]), float(cy)], [0, 0, 1.0]])
            for k in ("k", "matrix", "intrinsic", "intrinsics", "intrinsic_matrix", "camera_matrix", "intrinsic_cv", "k_matrix"):
                if k in low:
                    K = parse_intrinsics(low[k], width, height)
                    if K is not None:
                        return K
            w = float(low.get("width", width))
            h = float(low.get("height", height))
            for k in ("fovy", "fov_y", "fov"):
                if k in low:
                    f = float(low[k])
                    if f > 3.2:
                        f = math.radians(f)
                    fy = (h / 2.0) / math.tan(f / 2.0)
                    return np.array([[fy, 0, w / 2.0], [0, fy, h / 2.0], [0, 0, 1.0]])
            return None
        a = np.asarray(v, dtype=float)
        if a.shape == (3, 3):
            return a
        if a.size == 9:
            return a.reshape(3, 3)
        if a.shape == (3, 4) or a.size == 12:
            return a.reshape(3, 4)[:, :3]
        if a.size == 4:
            fx, fy, cx, cy = a.ravel()
            return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    except Exception:
        return None
    return None


def parse_pose(v):
    """Return (4x4 matrix as given, convention string) or (None, None)."""
    try:
        if v is None:
            return None, None
        conv = ""
        if isinstance(v, dict):
            low = {str(k).lower(): val for k, val in v.items()}
            for ck in ("convention", "frame_convention", "camera_convention", "axes"):
                if ck in low:
                    conv = str(low[ck]).lower()
            pos = None
            for pk in ('position_m', "position", "p", "translation", "t", "xyz", "pos"):
                if pk in low and is_num_list(low[pk], 3):
                    pos = np.asarray(low[pk], dtype=float)
                    break
            R = None
            for qk in ("quaternion_wxyz", "quat_wxyz", "wxyz", "quaternion", "q", "orientation", "rotation_quaternion"):
                if qk in low and is_num_list(low[qk], 4):
                    R = quat_wxyz_to_mat(low[qk])
                    break
            if R is None:
                for rk in ("rotation", "rotation_matrix", "r"):
                    if rk in low:
                        a = np.asarray(low[rk], dtype=float)
                        if a.size == 9:
                            R = a.reshape(3, 3)
                            break
            if pos is not None and R is not None:
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = pos
                return T, conv
            for mk in ("matrix", "pose", "cam2world", "camera_to_world", "extrinsic", "extrinsics", "transform", "world_from_camera", "cam2world_gl"):
                if mk in low:
                    T, c2 = parse_pose(low[mk])
                    if T is not None:
                        return T, (conv or c2 or mk)
            return None, None
        a = np.asarray(v, dtype=float)
        if a.size == 16:
            return a.reshape(4, 4), ""
        if a.size == 12:
            T = np.eye(4)
            T[:3, :] = a.reshape(3, 4)
            return T, ""
        if a.size == 7:
            T = np.eye(4)
            T[:3, :3] = quat_wxyz_to_mat(a[3:7])
            T[:3, 3] = a[:3]
            return T, ""
    except Exception:
        return None, None
    return None, None


def per_camera(v, cam):
    """Pick the entry for `cam` from a per-camera container."""
    if isinstance(v, dict):
        for k, val in v.items():
            if cam in str(k).lower():
                return val
        return None
    if isinstance(v, list):
        for item in v:
            if isinstance(item, dict):
                name = str(item.get("camera", item.get("name", ""))).lower()
                if cam in name:
                    return item
    return None


class CameraModel:
    """Pinhole back-projection with a selectable convention variant."""

    def __init__(self, K, T, variant=None):
        self.K = np.asarray(K, dtype=float)
        self.T_given = np.asarray(T, dtype=float)
        self.variant = variant or {"conv": "gl", "off": 0.0, "transpose": False, "invert": False, "ray": False}

    @property
    def T(self):
        if self.variant.get("invert"):
            return np.linalg.inv(self.T_given)
        return self.T_given

    def backproject(self, u, v, d):
        """u (column), v (row), d (depth, m) arrays -> Nx3 world."""
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        d = np.asarray(d, dtype=float)
        if self.variant.get("transpose"):
            u, v = v, u
        off = self.variant.get("off", 0.0)
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        xn = (u + off - cx) / fx
        yn = (v + off - cy) / fy
        if self.variant.get("ray"):
            d = d / np.sqrt(1 + xn * xn + yn * yn)
        if self.variant.get("conv") == "gl":
            pc = np.stack([xn * d, -yn * d, -d], axis=-1)
        else:
            pc = np.stack([xn * d, yn * d, d], axis=-1)
        T = self.T
        return pc @ T[:3, :3].T + T[:3, 3]

    def backproject_image(self, depth):
        H, W = depth.shape
        vv, uu = np.mgrid[0:H, 0:W]
        P = self.backproject(uu.ravel(), vv.ravel(), depth.ravel()).reshape(H, W, 3)
        valid = depth > 0
        return P, valid

    def project(self, X):
        """World Nx3 -> (u, v, depth) (inverse of backproject)."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        T = self.T
        Rt = T[:3, :3].T
        pc = (X - T[:3, 3]) @ Rt.T
        if self.variant.get("conv") == "gl":
            d = -pc[:, 2]
            xn = pc[:, 0] / np.where(np.abs(d) < 1e-9, 1e-9, d)
            yn = -pc[:, 1] / np.where(np.abs(d) < 1e-9, 1e-9, d)
        else:
            d = pc[:, 2]
            xn = pc[:, 0] / np.where(np.abs(d) < 1e-9, 1e-9, d)
            yn = pc[:, 1] / np.where(np.abs(d) < 1e-9, 1e-9, d)
        off = self.variant.get("off", 0.0)
        u = xn * self.K[0, 0] + self.K[0, 2] - off
        v = yn * self.K[1, 1] + self.K[1, 2] - off
        if self.variant.get("transpose"):
            u, v = v, u
        return u, v, d


def all_variants(default_conv="gl"):
    out = []
    for conv in ("gl", "cv"):
        for off in (0.0, 0.5):
            for transpose in (False, True):
                for invert in (False, True):
                    for ray in (False, True):
                        out.append({"conv": conv, "off": off, "transpose": transpose, "invert": invert, "ray": ray})
    # Put the documented convention first so ties favour it.
    out.sort(key=lambda v: (v["conv"] != default_conv, v["off"], v["transpose"], v["invert"], v["ray"]))
    return out


def parse_pixel_points(result):
    """Parse robomme_pixel_to_position result -> list of (xyz or None, valid, spread)."""
    meta = reply_meta(result)
    lst = None
    for key in ("points", "results", "positions", "selections", "items", "world_points"):
        v = bfs_find(meta, key)
        if isinstance(v, list) and v:
            lst = v
            break
    if lst is None:
        # any list of dicts
        def walk(o):
            if isinstance(o, list) and o and all(isinstance(x, dict) for x in o):
                return o
            if isinstance(o, dict):
                for val in o.values():
                    r = walk(val)
                    if r is not None:
                        return r
            return None
        lst = walk(meta) or []
    out = []
    for item in lst:
        xyz, valid, spread = None, True, None
        if isinstance(item, dict):
            for k in ("position", "world_position", "position_world", "world", "point", "xyz", "world_point", "position_m", "point_world"):
                if k in item and is_num_list(item[k], 3):
                    xyz = np.asarray(item[k], dtype=float)
                    break
            if xyz is None:
                for k, val in item.items():
                    if is_num_list(val, 3) and "pixel" not in str(k).lower():
                        xyz = np.asarray(val, dtype=float)
                        break
            if "valid" in item:
                valid = bool(item["valid"])
            s = item.get("depth_spread_m")
            spread = float(s) if isinstance(s, (int, float)) else None
        elif is_num_list(item, 3):
            xyz = np.asarray(item, dtype=float)
        if xyz is None:
            valid = False
        out.append((xyz, valid, spread))
    return out
