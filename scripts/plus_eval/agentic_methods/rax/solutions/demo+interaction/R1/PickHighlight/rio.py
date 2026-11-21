"""Defensive parsing of RAX/RoboMME MCP replies.

Nothing here assumes exact key names beyond what the docs state; every lookup
searches the JSON tree and falls back gracefully.  All functions are pure.
"""
import base64
import io
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

WORKSPACE = Path("/workspace")


# ----------------------------------------------------------------- JSON tree
def first_text_json(result):
    """JSON metadata of an MCP result (first text block that parses)."""
    for block in (result or {}).get("content", []) or []:
        if block.get("type") == "text":
            try:
                return json.loads(block.get("text", ""))
            except Exception:
                continue
    return {}


def bfs_find(obj, key, pred=None, max_nodes=200000):
    """Shallowest value stored under `key` (optionally satisfying pred)."""
    queue = [obj]
    n = 0
    while queue and n < max_nodes:
        nxt = []
        for o in queue:
            n += 1
            if isinstance(o, dict):
                if key in o and (pred is None or pred(o[key])):
                    return o[key]
                nxt.extend(v for v in o.values() if isinstance(v, (dict, list)))
            elif isinstance(o, list):
                nxt.extend(v for v in o if isinstance(v, (dict, list)))
        queue = nxt
    return None


def find_all(obj, key, out=None, depth=0):
    if out is None:
        out = []
    if depth > 30:
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                out.append(v)
            if isinstance(v, (dict, list)):
                find_all(v, key, out, depth + 1)
    elif isinstance(obj, list):
        for v in obj:
            if isinstance(v, (dict, list)):
                find_all(v, key, out, depth + 1)
    return out


def is_numlist(v, n=None):
    if not isinstance(v, (list, tuple)):
        return False
    if n is not None and len(v) != n:
        return False
    return len(v) > 0 and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v)


def get_state(meta):
    s = bfs_find(meta, "state", lambda v: is_numlist(v, 8))
    if s is None:
        s = bfs_find(meta, "state", lambda v: isinstance(v, dict))
        if isinstance(s, dict):
            for k in ("values", "vector", "data"):
                if is_numlist(s.get(k), 8):
                    return [float(x) for x in s[k]]
        return None
    return [float(x) for x in s]


def get_step_id(meta):
    v = bfs_find(meta, "step_id", lambda v: isinstance(v, int) and not isinstance(v, bool))
    return v


def get_bool(meta, key):
    v = bfs_find(meta, key, lambda v: isinstance(v, bool))
    return bool(v) if v is not None else False


def get_instruction(meta):
    v = bfs_find(meta, "instruction", lambda v: isinstance(v, str))
    return v or ""


def remaining_steps(meta):
    """Best effort: remaining control steps from episode_limits."""
    lim = bfs_find(meta, "episode_limits", lambda v: isinstance(v, dict))
    if not isinstance(lim, dict):
        return None, None
    rem = None
    mx = None
    for k, v in lim.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        kl = k.lower()
        if "remaining" in kl and "step" in kl:
            rem = int(v)
        elif ("max" in kl or "limit" in kl) and "step" in kl and "second" not in kl:
            mx = int(v)
    if rem is None:
        for k, v in lim.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and "remaining" in k.lower():
                rem = int(v)
    return rem, mx


# ----------------------------------------------------------------- images
def _decode_image_bytes(data):
    im = Image.open(io.BytesIO(data))
    im.load()
    return im


def _pil_to_array(im):
    """Return (kind, array): kind 'rgb' (HxWx3 uint8) or 'depth' (HxW float metres or raw)."""
    mode = im.mode
    if mode in ("I;16", "I;16B", "I;16L", "I", "F", "I;16N"):
        arr = np.asarray(im).astype(np.float64)
        return "depth", arr
    if mode == "L":
        return "gray", np.asarray(im)
    arr = np.asarray(im.convert("RGB"))
    return "rgb", arr


def _label_from_obj(o):
    """Look for camera / kind labels in an attachment description."""
    cam = kind = None
    if not isinstance(o, dict):
        return cam, kind
    texts = []
    for k, v in o.items():
        if k in ("data", "blob"):
            continue
        if isinstance(v, str) and len(v) < 300:
            texts.append(v.lower())
            if k in ("camera", "camera_name", "view"):
                cam = v.lower()
            if k in ("kind", "modality", "type_name", "channel"):
                kind = v.lower()
        elif isinstance(v, dict):
            c2, k2 = _label_from_obj(v)
            cam = cam or c2
            kind = kind or k2
    blob = " ".join(texts)
    if cam is None:
        if "front" in blob or "third" in blob:
            cam = "front"
        elif "wrist" in blob or "hand" in blob:
            cam = "wrist"
    if kind is None:
        if "depth" in blob:
            kind = "depth"
        elif "rgb" in blob or "color" in blob:
            kind = "rgb"
    if cam is not None:
        cam = "front" if ("front" in cam or "third" in cam) else ("wrist" if ("wrist" in cam or "hand" in cam) else cam)
    return cam, kind


def collect_images(result, meta):
    """List of dicts {cam, kind, arr, src} from inline image blocks of an MCP result."""
    out = []
    last_text = ""
    blocks = (result or {}).get("content", []) or []
    for idx, block in enumerate(blocks):
        t = block.get("type")
        if t == "text":
            last_text = block.get("text", "")[:400]
            continue
        data = None
        if t == "image":
            data = block.get("data")
        elif t == "resource":
            res = block.get("resource", {}) or {}
            data = res.get("blob")
        if not data:
            continue
        try:
            im = _decode_image_bytes(base64.b64decode(data))
        except Exception:
            continue
        kind, arr = _pil_to_array(im)
        cam, lkind = _label_from_obj(block)
        if cam is None and len(last_text) < 300:
            lt = last_text.lower()
            if "front" in lt:
                cam = "front"
            elif "wrist" in lt:
                cam = "wrist"
            if "depth" in lt and kind != "rgb":
                lkind = "depth"
        if lkind == "depth" and kind == "gray":
            kind = "depth"
            arr = arr.astype(np.float64)
        out.append({"cam": cam, "kind": kind, "arr": arr, "src": "inline", "idx": idx})
        last_text = ""
    return out


def classify_rgb_camera(arr):
    """Heuristic: wrist view shows the black gripper fingers near the bottom."""
    a = arr.astype(int)
    h = a.shape[0]
    bottom = a[int(h * 0.78):, :, :]
    dark = (bottom.max(axis=2) < 50).mean()
    return "wrist" if dark > 0.04 else "front"


def load_file_image(path):
    p = Path(path)
    if not p.is_absolute():
        p = WORKSPACE / p
    im = Image.open(p)
    im.load()
    return _pil_to_array(im)


def files_from_meta(meta):
    """Collect explicit file descriptors {camera, kind, path, meters_per_unit} for the *latest* frame."""
    found = []
    # direct 'files' lists
    for files in find_all(meta, "files"):
        if isinstance(files, list):
            for f in files:
                if isinstance(f, dict) and isinstance(f.get("path"), str):
                    found.append(f)
    return found


def manifest_paths(meta):
    out = []
    for v in find_all(meta, "manifest_path"):
        if isinstance(v, str):
            out.append(v)
    return out


def read_manifest(path):
    p = Path(path)
    if not p.is_absolute():
        p = WORKSPACE / p
    return json.loads(p.read_text())


def depth_scale_from_meta(meta):
    v = bfs_find(meta, "meters_per_unit", lambda v: isinstance(v, (int, float)) and not isinstance(v, bool))
    return float(v) if v else 0.001


# ----------------------------------------------------------------- calibration
def _find_cam_entry(obj, cam):
    """obj may be {cam: {...}} or list of {camera: cam, ...}."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and cam in k.lower():
                return v
        if "camera" in obj and isinstance(obj["camera"], str) and cam in obj["camera"].lower():
            return obj
    if isinstance(obj, list):
        for v in obj:
            if isinstance(v, dict):
                name = str(v.get("camera", v.get("name", ""))).lower()
                if cam in name:
                    return v
    return None


def _as_matrix(v):
    try:
        a = np.asarray(v, dtype=float)
    except Exception:
        return None
    if a.shape == (3, 3) or a.shape == (4, 4) or a.shape == (3, 4):
        return a
    if a.size == 9:
        return a.reshape(3, 3)
    if a.size == 16:
        return a.reshape(4, 4)
    return None


def parse_intrinsics(meta, cam):
    root = bfs_find(meta, "camera_intrinsics")
    if root is None:
        return None
    e = _find_cam_entry(root, cam)
    if e is None:
        return None
    if isinstance(e, dict):
        lk = {k.lower(): v for k, v in e.items()}
        if all(k in lk for k in ("fx", "fy", "cx", "cy")):
            return dict(fx=float(lk["fx"]), fy=float(lk["fy"]), cx=float(lk["cx"]), cy=float(lk["cy"]))
        for k in ("k", "matrix", "intrinsic", "intrinsics", "intrinsic_matrix", "camera_matrix"):
            if k in lk:
                m = _as_matrix(lk[k])
                if m is not None:
                    return dict(fx=m[0, 0], fy=m[1, 1], cx=m[0, 2], cy=m[1, 2])
        for v in e.values():
            m = _as_matrix(v)
            if m is not None and m.shape == (3, 3):
                return dict(fx=m[0, 0], fy=m[1, 1], cx=m[0, 2], cy=m[1, 2])
    m = _as_matrix(e)
    if m is not None and m.shape == (3, 3):
        return dict(fx=m[0, 0], fy=m[1, 1], cx=m[0, 2], cy=m[1, 2])
    return None


def parse_pose(meta, cam):
    root = bfs_find(meta, "camera_poses")
    if root is None:
        return None
    e = _find_cam_entry(root, cam)
    if e is None:
        return None
    if isinstance(e, dict):
        lk = {k.lower(): v for k, v in e.items()}
        pos = None
        for k in ('position_m', "position", "p", "pos", "translation", "t"):
            if is_numlist(lk.get(k), 3):
                pos = np.asarray(lk[k], float)
                break
        q = None
        for k in ("quaternion_wxyz", "wxyz", "q", "quaternion", "orientation", "rotation_wxyz"):
            if is_numlist(lk.get(k), 4):
                q = np.asarray(lk[k], float)
                break
        conv = str(lk.get("convention", lk.get("frame_convention", "opengl"))).lower()
        if pos is not None and q is not None:
            return dict(pos=pos, quat=q, convention=conv)
        for k, v in lk.items():
            m = _as_matrix(v)
            if m is not None and m.shape == (4, 4):
                return dict(mat=m, convention=conv)
    m = _as_matrix(e)
    if m is not None and m.shape == (4, 4):
        return dict(mat=m, convention="opengl")
    return None


# ----------------------------------------------------------------- pixel tool
def parse_pixel_points(meta, n):
    """Return list of (xyz or None) for n selections from robomme_pixel_to_position."""
    cands = []

    def walk(o, depth=0):
        if depth > 12:
            return
        if isinstance(o, list) and len(o) == n and all(isinstance(x, dict) for x in o):
            cands.append(o)
        if isinstance(o, dict):
            for v in o.values():
                walk(v, depth + 1)
        elif isinstance(o, list):
            for v in o:
                walk(v, depth + 1)

    walk(meta)
    for lst in cands:
        pts = []
        ok_any = False
        for d in lst:
            p = _point_in_dict(d)
            valid = d.get("valid", True)
            if p is not None and valid is not False:
                pts.append(np.asarray(p, float))
                ok_any = True
            else:
                pts.append(None)
        if ok_any:
            return pts
    return [None] * n


def _point_in_dict(d, depth=0):
    if depth > 4 or not isinstance(d, dict):
        return None
    pref = ("world_position", "position_world", "world", "position", "point", "xyz", "world_point", "point_world")
    for k in pref:
        v = d.get(k)
        if is_numlist(v, 3):
            return v
        if isinstance(v, dict):
            for kk in ("xyz", "position", "value"):
                if is_numlist(v.get(kk), 3):
                    return v[kk]
            if all(c in v for c in ("x", "y", "z")):
                return [v["x"], v["y"], v["z"]]
    if all(c in d for c in ("x", "y", "z")) and all(isinstance(d[c], (int, float)) for c in ("x", "y", "z")):
        return [d["x"], d["y"], d["z"]]
    for k, v in d.items():
        if k in ("pixel", "uv", "selection", "camera_position", "camera_pos", "ray", "ray_direction"):
            continue
        if isinstance(v, dict):
            p = _point_in_dict(v, depth + 1)
            if p is not None:
                return p
    for k, v in d.items():
        if k in ("pixel", "uv", "selection", "camera_position", "camera_pos", "ray", "ray_direction"):
            continue
        if is_numlist(v, 3) and "cam" not in k.lower():
            return v
    return None
