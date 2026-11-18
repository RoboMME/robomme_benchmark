"""Defensive parsing of MCP replies (format partly unknown before the single live run)."""
import base64
import io
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

WORKSPACE = Path("/workspace")


def blocks(result):
    if isinstance(result, dict):
        return result.get("content") or []
    return []


def meta(result):
    """Merge structuredContent and every JSON text block into one dict."""
    out = {}
    if isinstance(result, dict):
        sc = result.get("structuredContent")
        if isinstance(sc, dict):
            out.update(sc)
    for b in blocks(result):
        if isinstance(b, dict) and b.get("type") == "text":
            t = b.get("text", "")
            try:
                v = json.loads(t)
            except Exception:
                continue
            if isinstance(v, dict):
                for k, val in v.items():
                    out.setdefault(k, val)
    return out


def walk(obj, path=()):
    """Yield (path, key, value) for every dict entry, depth-first."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield path, k, v
            yield from walk(v, path + (k,))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk(v, path + (i,))


def find_all(obj, key):
    return [(p, v) for p, k, v in walk(obj) if k == key]


def find_shallow(obj, key, pred=None, skip=("frame_sequence", "frames", "steps", "history")):
    best = None
    for p, v in find_all(obj, key):
        if any(s in [str(x) for x in p] for s in skip):
            continue
        if pred is not None and not pred(v):
            continue
        if best is None or len(p) < len(best[0]):
            best = (p, v)
    if best is None:
        # retry without skipping
        for p, v in find_all(obj, key):
            if pred is not None and not pred(v):
                continue
            if best is None or len(p) < len(best[0]):
                best = (p, v)
    return None if best is None else best[1]


def _is_state(v):
    return isinstance(v, list) and len(v) == 8 and all(isinstance(x, (int, float)) for x in v)


def get_state(m):
    s = find_shallow(m, "state", _is_state)
    return None if s is None else np.array(s, dtype=float)


def get_step_id(m):
    v = find_shallow(m, "step_id", lambda x: isinstance(x, int))
    return v


def get_bool(m, key):
    v = find_shallow(m, key, lambda x: isinstance(x, bool))
    return bool(v) if v is not None else False


def get_instruction(m):
    v = find_shallow(m, "instruction", lambda x: isinstance(x, str))
    return v or ""


def decode_image_block(b):
    try:
        data = base64.b64decode(b.get("data", ""))
        im = Image.open(io.BytesIO(data))
        im.load()
        return im
    except Exception:
        return None


def pil_to_rgb(im):
    return np.asarray(im.convert("RGB"), dtype=np.uint8)


def pil_to_depth(im, mpu=0.001):
    a = np.asarray(im)
    if a.ndim == 3:
        a = a[..., 0]
    return a.astype(np.float64) * mpu


def load_file(path):
    p = Path(path)
    if not p.is_absolute():
        p = WORKSPACE / p
    im = Image.open(p)
    im.load()
    return im


def read_manifest(m):
    fs = find_shallow(m, "frame_sequence", lambda x: isinstance(x, dict))
    if not fs:
        return None
    mp = fs.get("manifest_path")
    if not mp:
        return None
    p = Path(mp)
    if not p.is_absolute():
        p = WORKSPACE / p
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def frame_images(frame, want_depth=True):
    """Load {camera: {'rgb':arr,'depth':arr}} from a manifest frame record."""
    out = {}
    for f in frame.get("files", []) or []:
        cam, kind, path = f.get("camera"), f.get("kind"), f.get("path")
        if not cam or not kind or not path:
            continue
        if kind == "depth" and not want_depth:
            continue
        try:
            im = load_file(path)
        except Exception:
            continue
        if kind == "rgb":
            out.setdefault(cam, {})["rgb"] = pil_to_rgb(im)
        elif kind == "depth":
            out.setdefault(cam, {})["depth"] = pil_to_depth(im, float(f.get("meters_per_unit", 0.001) or 0.001))
    return out


def inline_images(result):
    """Images attached inline. Returns {camera:{'rgb':..,'depth':..}} by labels or order."""
    rgbs, depths = [], []
    label = ""
    for b in blocks(result):
        if not isinstance(b, dict):
            continue
        if b.get("type") == "text":
            label = (b.get("text") or "")[-200:].lower()
            continue
        if b.get("type") != "image":
            continue
        im = decode_image_block(b)
        if im is None:
            continue
        ann = json.dumps({k: v for k, v in b.items() if k != "data"}).lower() + " " + label
        cam = "wrist" if "wrist" in ann else ("front" if "front" in ann else None)
        isdepth = im.mode in ("I;16", "I;16B", "I;16L", "I", "F") or "depth" in ann
        (depths if isdepth else rgbs).append((cam, im))
        label = ""
    out = {}
    for lst, kind in ((rgbs, "rgb"), (depths, "depth")):
        order = ["front", "wrist"]
        for i, (cam, im) in enumerate(lst):
            c = cam or (order[i] if i < 2 else None)
            if c is None:
                continue
            arr = pil_to_rgb(im) if kind == "rgb" else pil_to_depth(im)
            out.setdefault(c, {}).setdefault(kind, arr)
    return out


def nodes(obj, path=()):
    yield path, obj
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from nodes(v, path + (k,))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from nodes(v, path + (i,))


def file_records(m):
    """File dicts (camera/kind/path) in metadata outside frame_sequence."""
    recs = []
    for p, v in nodes(m):
        if isinstance(v, dict) and {"camera", "kind", "path"} <= set(v.keys()):
            if "frame_sequence" in [str(x) for x in p]:
                continue
            recs.append(v)
    return recs
