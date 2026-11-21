"""Defensive parsing of RoboMME-Plus MCP replies (format not previewed live).

Every accessor tolerates missing fields; nothing here raises on odd formats
except where a caller must know (returns None instead).
"""
import base64
import io
import json
import math
import os
import time

import numpy as np
from PIL import Image

WORKSPACE = "/workspace"


def _json_texts(result):
    out = []
    for b in (result or {}).get("content", []) or []:
        if b.get("type") == "text":
            t = b.get("text", "")
            try:
                out.append(json.loads(t))
            except Exception:
                out.append(t)
    return out


def meta_of(result):
    """Merge all JSON-object text blocks (first occurrence wins)."""
    meta = {}
    for j in _json_texts(result):
        if isinstance(j, dict):
            for k, v in j.items():
                meta.setdefault(k, v)
    sc = (result or {}).get("structuredContent")
    if isinstance(sc, dict):
        for k, v in sc.items():
            meta.setdefault(k, v)
    return meta


def find_key(obj, key, pred=None, depth=0, maxdepth=6):
    """Breadth-first search for key in nested dict/list; returns first value."""
    if depth > maxdepth:
        return None
    if isinstance(obj, dict):
        if key in obj and (pred is None or pred(obj[key])):
            return obj[key]
        for v in obj.values():
            if isinstance(v, (dict, list)):
                r = find_key(v, key, pred, depth + 1, maxdepth)
                if r is not None:
                    return r
    elif isinstance(obj, list):
        for v in obj[:64]:
            if isinstance(v, (dict, list)):
                r = find_key(v, key, pred, depth + 1, maxdepth)
                if r is not None:
                    return r
    return None


def _is_num_list(v, n=None):
    if not isinstance(v, (list, tuple)):
        return False
    if n is not None and len(v) != n:
        return False
    return all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v)


def decode_image_block(b):
    data = b.get("data")
    if not data:
        return None
    try:
        im = Image.open(io.BytesIO(base64.b64decode(data)))
        im.load()
        return im
    except Exception:
        return None


def _to_rgb_array(im):
    if im.mode in ("RGB", "RGBA", "P", "L"):
        return np.array(im.convert("RGB"))
    return None


def _to_depth_array(im, mpu=0.001):
    if im.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        a = np.array(im).astype(np.float64)
        return a * mpu
    return None


def _cam_of_label(text):
    t = (text or "").lower()
    if "wrist" in t or "hand" in t:
        return "wrist"
    if "front" in t or "third" in t or "base" in t:
        return "front"
    return None


def _kind_of_label(text):
    t = (text or "").lower()
    if "depth" in t:
        return "depth"
    if "rgb" in t or "color" in t or "colour" in t:
        return "rgb"
    return None


def collect_file_entries(obj, depth=0, out=None):
    """All dicts with 'path' and 'camera' anywhere in obj."""
    if out is None:
        out = []
    if depth > 7:
        return out
    if isinstance(obj, dict):
        if isinstance(obj.get("path"), str) and ("camera" in obj or "kind" in obj):
            out.append(obj)
        for v in obj.values():
            if isinstance(v, (dict, list)):
                collect_file_entries(v, depth + 1, out)
    elif isinstance(obj, list):
        for v in obj[:200]:
            if isinstance(v, (dict, list)):
                collect_file_entries(v, depth + 1, out)
    return out


def load_file(path, kind, mpu=0.001):
    p = path if os.path.isabs(path) else os.path.join(WORKSPACE, path)
    if not os.path.exists(p):
        return None
    try:
        im = Image.open(p)
        im.load()
    except Exception:
        return None
    if kind == "depth":
        d = _to_depth_array(im, mpu)
        if d is None and im.mode == "L":
            return None
        return d
    return _to_rgb_array(im)


class Obs:
    """Parsed observation."""

    def __init__(self):
        self.meta = {}
        self.step_id = None
        self.state = None
        self.instruction = None
        self.ended = False
        self.complete = False
        self.rgb = {}
        self.depth = {}
        self.limits = None
        self.manifest_path = None
        self.notes = []


def parse_obs(result, prev=None):
    o = Obs()
    meta = meta_of(result)
    o.meta = meta
    sid = meta.get("step_id")
    if not isinstance(sid, int):
        sid = find_key(meta, "step_id", lambda v: isinstance(v, int) and not isinstance(v, bool))
    o.step_id = sid
    st = meta.get("state")
    if not _is_num_list(st, 8):
        st = find_key(meta, "state", lambda v: _is_num_list(v, 8))
    o.state = [float(x) for x in st] if st is not None else (prev.state if prev else None)
    ins = meta.get("instruction")
    if not isinstance(ins, str):
        ins = find_key(meta, "instruction", lambda v: isinstance(v, str))
    o.instruction = ins if ins else (prev.instruction if prev else None)
    ended = find_key(meta, "episode_ended", lambda v: isinstance(v, bool))
    o.ended = bool(ended) if ended is not None else False
    for k in ("task_complete", "success"):
        v = find_key(meta, k, lambda v: isinstance(v, bool))
        if v:
            o.complete = True
    if find_key(meta, "terminated", lambda v: isinstance(v, bool)) or \
            find_key(meta, "truncated", lambda v: isinstance(v, bool)):
        o.ended = True
    o.limits = find_key(meta, "episode_limits")
    fs = find_key(meta, "frame_sequence")
    if isinstance(fs, dict) and isinstance(fs.get("manifest_path"), str):
        o.manifest_path = fs["manifest_path"]

    # ---- inline images
    blocks = (result or {}).get("content", []) or []
    imgs = []
    label = ""
    for b in blocks:
        if b.get("type") == "text":
            t = b.get("text", "")
            if not t.lstrip().startswith("{"):
                label = t[:300]
        elif b.get("type") == "image":
            im = decode_image_block(b)
            if im is not None:
                imgs.append((label, im, b))
            label = ""
    # descriptor list in meta matching image count?
    desc = None
    for key in ("images", "attachments", "image_blocks", "cameras_order"):
        d = find_key(meta, key, lambda v: isinstance(v, list) and v and all(isinstance(x, dict) for x in v))
        if d is not None and len(d) == len(imgs):
            desc = d
            break
    rgb_order = []
    depth_order = []
    for idx, (lab, im, b) in enumerate(imgs):
        cam, kind = _cam_of_label(lab), _kind_of_label(lab)
        if desc is not None:
            dd = desc[idx]
            cam = cam or _cam_of_label(str(dd.get("camera", "")) + " " + str(dd.get("name", "")))
            kind = kind or _kind_of_label(str(dd.get("kind", "")) + " " + str(dd.get("name", "")))
        for k in ("annotations", "_meta", "meta"):
            if isinstance(b.get(k), dict):
                cam = cam or _cam_of_label(json.dumps(b[k]))
                kind = kind or _kind_of_label(json.dumps(b[k]))
        if im.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
            kind = "depth"
        if kind == "depth":
            arr = _to_depth_array(im)
            if arr is not None:
                depth_order.append((cam, arr))
        else:
            arr = _to_rgb_array(im)
            if arr is not None:
                rgb_order.append((cam, arr))
    default = ["front", "wrist"]
    unlabeled = [i for i, (cam, _) in enumerate(rgb_order) if cam is None]
    if unlabeled:
        # content check: real scene images are mostly wood; the wrist view always shows the
        # gripper in its bottom band (demo calibration: front 0.00, wrist >= 0.32 non-wood)
        guess = {}
        for i in unlabeled:
            wf, bn = scene_stats(rgb_order[i][1])
            if wf is None or wf < 0.2:
                continue
            guess[i] = "wrist" if bn > 0.15 else "front"
        o.notes.append("content guess %s" % guess)
        if len(rgb_order) == 2 and len(guess) == 2 and sorted(guess.values()) == ["front", "wrist"]:
            for i, c in guess.items():
                rgb_order[i] = (c, rgb_order[i][1])
        elif len(rgb_order) > 2 or len(guess) < 2:
            for i, c in guess.items():
                if c not in [cc for cc, _ in rgb_order if cc]:
                    rgb_order[i] = (c, rgb_order[i][1])
            rgb_order = [(c, a) for j, (c, a) in enumerate(rgb_order) if c is not None or j in guess or len(rgb_order) <= 2]
    for i, (cam, arr) in enumerate(rgb_order):
        c = cam or (default[i] if i < 2 else None)
        if c and c not in o.rgb:
            o.rgb[c] = arr
    for i, (cam, arr) in enumerate(depth_order):
        c = cam or (default[i] if i < 2 else None)
        if c and c not in o.depth:
            o.depth[c] = arr
    o.notes.append("inline rgb=%d depth=%d labels=%s" % (
        len(rgb_order), len(depth_order), [c for c, _ in rgb_order]))

    # ---- file entries in metadata (prefer for missing cameras)
    entries = collect_file_entries(meta)
    cur = [e for e in entries if e.get("step_id") in (None, sid)]
    for e in cur:
        cam = _cam_of_label(str(e.get("camera", "")))
        kind = (str(e.get("kind", "")) or "rgb").lower()
        if cam is None:
            continue
        if kind == "rgb" and cam not in o.rgb:
            a = load_file(e["path"], "rgb")
            if a is not None:
                o.rgb[cam] = a
                o.notes.append("rgb %s from file" % cam)
        if kind == "depth" and cam not in o.depth:
            a = load_file(e["path"], "depth", float(e.get("meters_per_unit", 0.001) or 0.001))
            if a is not None:
                o.depth[cam] = a
    # ---- manifest frame with matching step (fallback)
    if (("front" not in o.rgb) or ("wrist" not in o.rgb)) and o.manifest_path:
        try:
            frames = read_manifest(o.manifest_path)
            exe = [f for f in frames if f.get("phase") != "video_demo"]
            cand = [f for f in exe if f.get("step_id") == sid] or exe[-1:]
            if cand:
                files = frame_files(cand[-1])
                for cam in ("front", "wrist"):
                    if cam not in o.rgb and (cam, "rgb") in files:
                        a = load_file(files[(cam, "rgb")][0], "rgb")
                        if a is not None:
                            o.rgb[cam] = a
                            o.notes.append("rgb %s from manifest" % cam)
        except Exception as ex:  # noqa
            o.notes.append("manifest fallback failed %r" % ex)
    return o


def scene_stats(rgb):
    """(wood fraction, bottom-band non-wood fraction) of an RGB array."""
    try:
        import cv2
        hsv = cv2.cvtColor(np.ascontiguousarray(rgb[..., :3]), cv2.COLOR_RGB2HSV)
        H, S, V = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
        wood = (H >= 5) & (H <= 17) & (S >= 60) & (V >= 50)
        h = wood.shape[0]
        return float(wood.mean()), float(1.0 - wood[int(0.8 * h):, :].mean())
    except Exception:
        return None, None


def read_manifest(path):
    p = path if os.path.isabs(path) else os.path.join(WORKSPACE, path)
    with open(p) as f:
        m = json.load(f)
    frames = m.get("frames", []) if isinstance(m, dict) else []
    frames = [f for f in frames if isinstance(f, dict)]
    frames.sort(key=lambda f: (f.get("frame_index", 0) if isinstance(f.get("frame_index"), int) else 0))
    return frames


def frame_files(frame):
    files = {}
    for e in frame.get("files", []) or []:
        if not isinstance(e, dict) or not isinstance(e.get("path"), str):
            continue
        cam = _cam_of_label(str(e.get("camera", "")))
        kind = str(e.get("kind", "rgb")).lower()
        if cam:
            files[(cam, kind)] = (e["path"], float(e.get("meters_per_unit", 0.001) or 0.001))
    return files


def parse_points(result, n):
    """Parse robomme_pixel_to_position reply into list of (xyz|None, spread|None)."""
    meta = meta_of(result)
    cands = []

    def walk(o, d=0):
        if d > 6:
            return
        if isinstance(o, list) and o and all(isinstance(x, dict) for x in o):
            cands.append(o)
        if isinstance(o, dict):
            for v in o.values():
                walk(v, d + 1)
        elif isinstance(o, list):
            for v in o[:64]:
                if isinstance(v, (dict, list)):
                    walk(v, d + 1)
    walk(meta)
    if not cands:
        for j in _json_texts(result):
            if isinstance(j, list):
                walk(j)
    lst = None
    for c in cands:
        if len(c) == n:
            lst = c
            break
    out = []
    if lst is None:
        return [(None, None)] * n
    for d in lst:
        xyz = None
        for k in ("position", "world_position", "position_m", "world", "point", "xyz",
                  "world_point", "position_world"):
            v = d.get(k)
            if _is_num_list(v, 3):
                xyz = [float(x) for x in v]
                break
            if isinstance(v, dict) and all(a in v for a in ("x", "y", "z")):
                xyz = [float(v["x"]), float(v["y"]), float(v["z"])]
                break
        if xyz is None:
            for k, v in d.items():
                if _is_num_list(v, 3) and "pixel" not in k and "uv" not in k:
                    xyz = [float(x) for x in v]
                    break
        valid = d.get("valid", True)
        if valid is False:
            xyz = None
        if xyz is not None and not all(math.isfinite(x) for x in xyz):
            xyz = None
        sp = d.get("depth_spread_m")
        out.append((xyz, float(sp) if isinstance(sp, (int, float)) else None))
    return out
