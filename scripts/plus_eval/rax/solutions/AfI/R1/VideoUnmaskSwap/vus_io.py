"""Reply parsing, logging and image loading for the VideoUnmaskSwap controller.

Everything here is defensive: unknown reply layouts are searched recursively and
failures return None instead of raising.
"""
import base64
import io
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None

WORKSPACE = Path("/workspace")
OUT_DIR = WORKSPACE / "output"
VERSION = "vus-v1"

_T0 = time.time()
_LOG_FH = None


def log(*parts, **kv):
    """One line per decision: to stdout (captured as stdout.log) and output/vus_log.txt."""
    global _LOG_FH
    msg = " ".join(str(p) for p in parts)
    if kv:
        try:
            msg += " " + json.dumps(kv, default=_json_default, allow_nan=True)
        except Exception:
            msg += " " + str(kv)
    line = f"[{VERSION} t={time.time() - _T0:7.1f}s] {msg}"
    try:
        print(line, flush=True)
    except Exception:
        pass
    try:
        if _LOG_FH is None:
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            _LOG_FH = open(OUT_DIR / "vus_log.txt", "a", buffering=1)
        _LOG_FH.write(line + "\n")
    except Exception:
        pass


def log_exc(where):
    log(f"EXCEPTION in {where}: " + traceback.format_exc().replace("\n", " | "))


def elapsed():
    return time.time() - _T0


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.round(4).tolist()
    if isinstance(o, (np.floating,)):
        return round(float(o), 5)
    if isinstance(o, (np.integer,)):
        return int(o)
    return str(o)


def save_json(name, obj):
    try:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(OUT_DIR / name, "w") as fh:
            json.dump(obj, fh, default=_json_default, indent=1)
    except Exception:
        log_exc("save_json")


# ----------------------------------------------------------------------------- replies

def parse_reply(reply):
    """MCP result -> (meta dict, list of image blocks)."""
    meta, images = {}, []
    if not isinstance(reply, dict):
        return meta, images
    content = reply.get("content") or []
    for b in content:
        if isinstance(b, dict) and b.get("type") == "text" and not meta:
            try:
                m = json.loads(b.get("text", ""))
                if isinstance(m, dict):
                    meta = m
            except Exception:
                pass
    for b in content:
        if isinstance(b, dict) and b.get("type") == "image":
            images.append(b)
    if not meta and isinstance(reply.get("structuredContent"), dict):
        meta = reply["structuredContent"]
    return meta, images


def find_key(obj, key, maxdepth=8):
    """Breadth-first search for the first value stored under `key`."""
    frontier = [obj]
    for _ in range(maxdepth):
        nxt = []
        for o in frontier:
            if isinstance(o, dict):
                if key in o:
                    return o[key]
                nxt.extend(v for v in o.values() if isinstance(v, (dict, list)))
            elif isinstance(o, list):
                nxt.extend(v for v in o if isinstance(v, (dict, list)))
        frontier = nxt
        if not frontier:
            break
    return None


def find_all_keys(obj, key, maxdepth=8, limit=50):
    out, frontier = [], [obj]
    for _ in range(maxdepth):
        nxt = []
        for o in frontier:
            if isinstance(o, dict):
                if key in o:
                    out.append(o[key])
                    if len(out) >= limit:
                        return out
                nxt.extend(v for v in o.values() if isinstance(v, (dict, list)))
            elif isinstance(o, list):
                nxt.extend(v for v in o if isinstance(v, (dict, list)))
        frontier = nxt
        if not frontier:
            break
    return out


def get_state(meta):
    for cand in find_all_keys(meta, "state"):
        if isinstance(cand, list) and len(cand) >= 8 and all(isinstance(x, (int, float)) for x in cand[:8]):
            return np.array(cand[:8], dtype=float)
        if isinstance(cand, dict):
            for k in ("values", "vector", "data"):
                v = cand.get(k)
                if isinstance(v, list) and len(v) >= 8:
                    try:
                        return np.array(v[:8], dtype=float)
                    except Exception:
                        pass
    return None


def get_bool(meta, key):
    v = find_key(meta, key)
    if isinstance(v, bool):
        return v
    return None


def get_step_id(meta):
    v = meta.get("step_id") if isinstance(meta, dict) else None
    if isinstance(v, int):
        return v
    for v in find_all_keys(meta, "step_id"):
        if isinstance(v, int) and not isinstance(v, bool):
            return v
    return None


def get_instruction(meta):
    for v in find_all_keys(meta, "instruction"):
        if isinstance(v, str) and v.strip():
            return v
    for k in ("goal", "language_instruction", "task_description"):
        v = find_key(meta, k)
        if isinstance(v, str) and v.strip():
            return v
    return ""


# ----------------------------------------------------------------------------- images

def load_png(path):
    if Image is None:
        return None
    try:
        p = Path(path)
        if not p.is_absolute():
            p = WORKSPACE / p
        with Image.open(p) as im:
            im.load()
            return _pil_to_array(im)
    except Exception:
        return None


def _pil_to_array(im):
    mode = im.mode
    if mode in ("RGB", "RGBA", "P"):
        return ("rgb", np.array(im.convert("RGB")))
    if mode == "L":
        return ("gray", np.array(im))
    arr = np.array(im)
    if arr.dtype == np.uint8 and arr.ndim == 3:
        return ("rgb", arr[..., :3])
    return ("depth_raw", arr.astype(np.float64))


def decode_image_block(block):
    if Image is None:
        return None
    try:
        data = base64.b64decode(block.get("data", ""))
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            return _pil_to_array(im)
    except Exception:
        return None


def frame_files(frame):
    """Manifest frame -> {(camera, kind): (path, meters_per_unit)}."""
    out = {}
    for f in frame.get("files", []) or []:
        try:
            out[(f.get("camera"), f.get("kind"))] = (f.get("path"), float(f.get("meters_per_unit", 0.001) or 0.001))
        except Exception:
            pass
    return out


def load_frame_images(frame, cameras=("front",)):
    """Return {camera: {'rgb': HxWx3 uint8, 'depth': HxW float metres}} for a manifest frame."""
    files = frame_files(frame)
    res = {}
    for cam in cameras:
        d = {}
        if (cam, "rgb") in files:
            r = load_png(files[(cam, "rgb")][0])
            if r is not None and r[0] == "rgb":
                d["rgb"] = r[1]
        if (cam, "depth") in files:
            path, mpu = files[(cam, "depth")]
            r = load_png(path)
            if r is not None and r[0] in ("depth_raw", "gray"):
                d["depth"] = r[1].astype(np.float64) * mpu
        res[cam] = d
    return res


def load_manifest(meta):
    seq = find_key(meta, "frame_sequence")
    path = None
    if isinstance(seq, dict):
        path = seq.get("manifest_path") or find_key(seq, "manifest_path")
    if path is None:
        path = find_key(meta, "manifest_path")
    if not isinstance(path, str):
        return None, None
    p = Path(path)
    if not p.is_absolute():
        p = WORKSPACE / p
    try:
        return json.loads(p.read_text()), str(p)
    except Exception:
        log_exc("load_manifest")
        return None, str(p)
