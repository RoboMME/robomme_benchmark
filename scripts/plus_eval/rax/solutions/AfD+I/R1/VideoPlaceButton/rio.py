"""Defensive parsing of RAX/RoboMME MCP replies (format details are not documented by example).

Every function tolerates missing keys and returns None instead of raising.
"""
import base64
import io
import json
import os
import time

import numpy as np
from PIL import Image

WORKSPACE = "/workspace"


def log(*a):
    msg = " ".join(str(x) for x in a)
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        os.makedirs(os.path.join(WORKSPACE, "output"), exist_ok=True)
        with open(os.path.join(WORKSPACE, "output", "controller_log.txt"), "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def text_blocks(result):
    out = []
    for b in (result or {}).get("content", []) or []:
        if isinstance(b, dict) and b.get("type") == "text":
            out.append(b.get("text", ""))
    return out


def meta_of(result):
    """First JSON-decodable text block (the documented metadata block)."""
    for t in text_blocks(result):
        try:
            v = json.loads(t)
            if isinstance(v, (dict, list)):
                return v
        except Exception:
            continue
    sc = (result or {}).get("structuredContent")
    if isinstance(sc, (dict, list)):
        return sc
    return {}


def find_key(obj, key, maxdepth=8):
    """Breadth-first search for a key in nested dicts/lists; returns the first value."""
    frontier = [obj]
    for _ in range(maxdepth):
        nxt = []
        for o in frontier:
            if isinstance(o, dict):
                if key in o:
                    return o[key]
                nxt.extend(o.values())
            elif isinstance(o, list):
                nxt.extend(o)
        if not nxt:
            break
        frontier = nxt
    return None


def find_all_keys(obj, key, maxdepth=8):
    res = []
    frontier = [obj]
    for _ in range(maxdepth):
        nxt = []
        for o in frontier:
            if isinstance(o, dict):
                if key in o:
                    res.append(o[key])
                nxt.extend(o.values())
            elif isinstance(o, list):
                nxt.extend(o)
        frontier = nxt
        if not frontier:
            break
    return res


def as_vec(v, n=None):
    try:
        a = np.array(v, dtype=float).reshape(-1)
        if n is not None and a.size != n:
            return None
        if not np.all(np.isfinite(a)):
            return None
        return a
    except Exception:
        return None


def get_state(meta):
    """8-D state (ee pos 3, axis-angle 3, fingers 2)."""
    for cand in find_all_keys(meta, "state"):
        if isinstance(cand, dict):
            for k in ("values", "vector", "data", "state"):
                if k in cand:
                    v = as_vec(cand[k], 8)
                    if v is not None:
                        return v
            continue
        v = as_vec(cand, 8)
        if v is not None:
            return v
    return None


def get_step_id(meta):
    v = find_key(meta, "step_id")
    try:
        return int(v)
    except Exception:
        return None


def flag(meta, key):
    v = find_key(meta, key)
    return bool(v) if v is not None else False


def decode_images(result):
    """Return list of (label_text, PIL image) for image blocks, label = preceding text/annotations."""
    out = []
    last_text = ""
    for b in (result or {}).get("content", []) or []:
        if not isinstance(b, dict):
            continue
        if b.get("type") == "text":
            last_text = b.get("text", "")[:300]
        elif b.get("type") == "image":
            try:
                raw = base64.b64decode(b.get("data", ""))
                im = Image.open(io.BytesIO(raw))
                im.load()
                label = last_text
                ann = b.get("annotations") or b.get("_meta") or b.get("meta")
                if ann:
                    label = json.dumps(ann)[:300] + " " + label
                out.append((label, im))
            except Exception as e:
                log("image decode failed", repr(e))
            last_text = ""
    return out


def manifest_frames(meta):
    """Load the frame_sequence manifest referenced by a reply (paths relative to /workspace)."""
    fs = find_key(meta, "frame_sequence")
    path = None
    if isinstance(fs, dict):
        path = fs.get("manifest_path") or find_key(fs, "manifest_path")
    elif isinstance(fs, str):
        path = fs
    if path is None:
        path = find_key(meta, "manifest_path")
    if not path:
        return None, None
    full = path if os.path.isabs(path) else os.path.join(WORKSPACE, path)
    try:
        with open(full) as f:
            man = json.load(f)
        return man, full
    except Exception as e:
        log("manifest read failed", full, repr(e))
        return None, full


def frame_file(frame, camera, kind, manifest_full=None):
    for f in frame.get("files", []) or []:
        if f.get("camera") == camera and f.get("kind") == kind:
            p = f.get("path")
            if p is None:
                continue
            cands = [p if os.path.isabs(p) else os.path.join(WORKSPACE, p)]
            if manifest_full:
                cands.append(os.path.join(os.path.dirname(manifest_full), p))
            for c in cands:
                if os.path.exists(c):
                    return c
            return cands[0]
    return None


def load_rgb(path):
    with Image.open(path) as im:
        return np.array(im.convert("RGB"))


def parse_points(meta, n_expected):
    """Extract per-selection world points from a pixel_to_position reply.
    Returns list of (xyz np.array or None, info dict)."""
    lists = []

    def walk(o, depth=0):
        if depth > 6:
            return
        if isinstance(o, list) and o and all(isinstance(x, dict) for x in o):
            lists.append(o)
        if isinstance(o, dict):
            for v in o.values():
                walk(v, depth + 1)
        elif isinstance(o, list):
            for v in o:
                walk(v, depth + 1)
    walk(meta)
    best = None
    for L in lists:
        if len(L) == n_expected:
            best = L
            break
    if best is None and lists:
        best = max(lists, key=len)
    out = []
    if best is None:
        return [(None, {}) for _ in range(n_expected)]
    keys = ("position", "world_position", "position_m", "world", "point", "xyz", "world_point",
            "world_xyz", "point_world", "position_world")
    for d in best:
        xyz = None
        for k in keys:
            if k in d:
                xyz = as_vec(d[k], 3)
                if xyz is None and isinstance(d[k], dict):
                    xyz = as_vec([d[k].get("x"), d[k].get("y"), d[k].get("z")], 3)
                if xyz is not None:
                    break
        if xyz is None:
            for k, v in d.items():
                if k in ("pixel", "uv", "pixel_uv"):
                    continue
                vv = as_vec(v, 3) if isinstance(v, list) else None
                if vv is not None:
                    xyz = vv
                    break
        if xyz is None and all(k in d for k in ("x", "y", "z")):
            xyz = as_vec([d["x"], d["y"], d["z"]], 3)
        valid = d.get("valid", True)
        if valid is False:
            xyz = None
        out.append((xyz, d))
    while len(out) < n_expected:
        out.append((None, {}))
    return out[:n_expected]
