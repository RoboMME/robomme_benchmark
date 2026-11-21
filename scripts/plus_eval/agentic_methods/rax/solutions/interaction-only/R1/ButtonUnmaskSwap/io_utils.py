"""Defensive parsing of MCP replies and frame manifests (formats only partly documented)."""
import base64
import io
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image

WS = Path("/workspace")


def reply_json(reply):
    """First text block parsed as JSON (or {})."""
    try:
        for b in reply.get("content", []):
            if b.get("type") == "text":
                try:
                    return json.loads(b.get("text", ""))
                except Exception:
                    continue
    except Exception:
        pass
    if isinstance(reply, dict) and "structuredContent" in reply:
        return reply["structuredContent"]
    return {}


def reply_texts(reply):
    out = []
    try:
        for b in reply.get("content", []):
            if b.get("type") == "text":
                out.append(b.get("text", ""))
    except Exception:
        pass
    return out


def reply_images(reply):
    """List of (preceding_text, PIL.Image) for inline image blocks."""
    out = []
    last_text = ""
    try:
        for b in reply.get("content", []):
            if b.get("type") == "text":
                last_text = b.get("text", "")[:200]
            elif b.get("type") == "image":
                try:
                    im = Image.open(io.BytesIO(base64.b64decode(b.get("data", ""))))
                    im.load()
                    out.append((last_text, im))
                except Exception:
                    pass
    except Exception:
        pass
    return out


def walk(obj, path=()):
    """Yield (path, key, value) for every dict entry recursively."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield path, k, v
            yield from walk(v, path + (k,))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk(v, path + (i,))


def find_key(obj, key, pred=None):
    """Shallowest value for `key` satisfying pred."""
    best = None
    for path, k, v in walk(obj):
        if k == key and (pred is None or pred(v)):
            if best is None or len(path) < best[0]:
                best = (len(path), v)
    return None if best is None else best[1]


def is_num_list(v, n=None):
    if not isinstance(v, (list, tuple)):
        return False
    if n is not None and len(v) != n:
        return False
    try:
        return all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in v)
    except Exception:
        return False


def _flatten_nums(v, out):
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        out.append(float(v))
    elif isinstance(v, (list, tuple)):
        for x in v:
            _flatten_nums(x, out)
    elif isinstance(v, dict):
        for x in v.values():
            _flatten_nums(x, out)
    return out


def get_state(js):
    st = find_key(js, "state", lambda v: is_num_list(v, 8))
    if st is None:
        for key in ("proprio", "proprioception", "robot_state", "state_vector"):
            st = find_key(js, key, lambda v: is_num_list(v, 8))
            if st is not None:
                break
    if st is None:
        d = find_key(js, "state", lambda v: isinstance(v, dict))
        if isinstance(d, dict):
            flat = _flatten_nums(d, [])
            if len(flat) == 8:
                return flat
    if st is None:
        st = find_key(js, "state", lambda v: isinstance(v, dict))
        if isinstance(st, dict):
            for k in ("values", "vector", "data"):
                if is_num_list(st.get(k), 8):
                    return [float(x) for x in st[k]]
        return None
    return [float(x) for x in st]


def get_bool(js, key):
    v = find_key(js, key, lambda v: isinstance(v, bool))
    return bool(v) if v is not None else None


def get_step_id(js):
    for key in ("step_id", "step", "steps_taken", "elapsed_steps"):
        v = find_key(js, key, lambda v: isinstance(v, int) and not isinstance(v, bool))
        if v is not None:
            return int(v)
    return None


def get_instruction(js):
    for key in ("instruction", "language_instruction", "goal", "task_instruction"):
        v = find_key(js, key, lambda v: isinstance(v, str) and len(v) > 3)
        if v:
            return v
    return ""


def resolve_path(p, manifest_dir=None):
    p = str(p)
    cands = []
    if os.path.isabs(p):
        cands.append(Path(p))
    cands.append(WS / p)
    if manifest_dir is not None:
        cands.append(Path(manifest_dir) / p)
    for c in cands:
        if c.exists():
            return c
    return cands[0]


def manifest_paths(js):
    out = []
    for path, k, v in walk(js):
        if k == "manifest_path" and isinstance(v, str):
            out.append(v)
    return out


def load_manifest(mp):
    p = resolve_path(mp)
    with open(p) as f:
        m = json.load(f)
    return m, p.parent


def frame_files(frame, manifest_dir=None):
    """dict (camera, kind) -> (Path, meters_per_unit)."""
    out = {}
    for f in frame.get("files", []) or []:
        try:
            cam = str(f.get("camera", "")).lower()
            kind = str(f.get("kind", "")).lower()
            out[(cam, kind)] = (resolve_path(f["path"], manifest_dir), float(f.get("meters_per_unit", 0.001) or 0.001))
        except Exception:
            continue
    return out


def load_rgb(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def load_depth(path, mpu=0.001):
    with Image.open(path) as im:
        a = np.asarray(im)
    if a.ndim == 3:
        a = a[..., 0]
    a = a.astype(np.float32)
    if a.max() > 0 and a.max() < 20 and mpu == 0.001 and a.dtype != np.uint16:
        pass
    return a * float(mpu)


def file_entries(js):
    """Collect any dict with camera/kind/path in a reply JSON."""
    out = {}
    for path, k, v in walk(js):
        if isinstance(v, dict) and "path" in v and "camera" in v:
            cam = str(v.get("camera", "")).lower()
            kind = str(v.get("kind", "rgb")).lower()
            out[(cam, kind)] = (resolve_path(v["path"]), float(v.get("meters_per_unit", 0.001) or 0.001))
    return out


def summarize(obj, depth=0, maxlen=120):
    """Compact structural summary for logs."""
    if depth > 4:
        return "..."
    if isinstance(obj, dict):
        return {k: summarize(v, depth + 1, maxlen) for k, v in list(obj.items())[:40]}
    if isinstance(obj, list):
        if len(obj) > 8:
            return [summarize(obj[0], depth + 1, maxlen), f"...({len(obj)} items)"]
        return [summarize(v, depth + 1, maxlen) for v in obj]
    if isinstance(obj, str) and len(obj) > maxlen:
        return obj[:maxlen] + "..."
    return obj
