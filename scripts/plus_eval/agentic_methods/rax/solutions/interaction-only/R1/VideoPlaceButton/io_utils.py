"""Defensive parsing of MCP replies, manifests and image files."""
import base64
import io
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

WS = Path("/workspace")
OUT = WS / "output"
_LOGF = None
T0 = time.time()


def log(msg, **kv):
    global _LOGF
    line = "[vpb %7.1fs] %s" % (time.time() - T0, msg)
    if kv:
        try:
            line += " " + json.dumps(kv, default=_jdefault)
        except Exception:
            line += " " + str(kv)
    try:
        print(line, flush=True)
    except Exception:
        pass
    try:
        if _LOGF is None:
            OUT.mkdir(parents=True, exist_ok=True)
            _LOGF = open(OUT / "vpb_log.txt", "a")
        _LOGF.write(line + "\n")
        _LOGF.flush()
    except Exception:
        pass


def _jdefault(o):
    if isinstance(o, np.ndarray):
        return np.round(o.astype(float), 4).tolist()
    if isinstance(o, (np.floating,)):
        return round(float(o), 5)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def save_json(name, obj):
    try:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / name).write_text(json.dumps(obj, default=_jdefault, indent=1))
    except Exception as e:
        log("save_json failed", name=name, err=str(e))


def reply_json(result):
    if not isinstance(result, dict):
        return {}
    for b in result.get("content", []) or []:
        if isinstance(b, dict) and b.get("type") == "text":
            try:
                j = json.loads(b.get("text", ""))
                if isinstance(j, dict):
                    return j
            except Exception:
                continue
    sc = result.get("structuredContent")
    if isinstance(sc, dict):
        return sc
    return {}


def reply_texts(result):
    out = []
    for b in (result or {}).get("content", []) or []:
        if isinstance(b, dict) and b.get("type") == "text":
            out.append(b.get("text", ""))
    return out


def reply_images(result):
    """List of dicts {index, mime, img(PIL), annotations} for image blocks."""
    out = []
    blocks = (result or {}).get("content", []) or []
    for i, b in enumerate(blocks):
        if not isinstance(b, dict) or b.get("type") != "image":
            continue
        try:
            data = base64.b64decode(b.get("data", ""))
            img = Image.open(io.BytesIO(data))
            img.load()
            out.append({"index": i, "mime": b.get("mimeType"), "img": img,
                        "meta": {k: v for k, v in b.items() if k not in ("data",)}})
        except Exception:
            continue
    return out


def find_key(obj, key, maxdepth=6):
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


def find_state(j):
    frontier = [j]
    for _ in range(5):
        nxt = []
        for o in frontier:
            if isinstance(o, dict):
                s = o.get("state")
                if isinstance(s, list) and len(s) >= 8 and all(isinstance(x, (int, float)) for x in s[:8]):
                    return [float(x) for x in s[:8]]
                if isinstance(s, dict):
                    for k in ("values", "vector", "data"):
                        v = s.get(k)
                        if isinstance(v, list) and len(v) >= 8:
                            return [float(x) for x in v[:8]]
                nxt.extend(v for v in o.values() if isinstance(v, (dict, list)))
            elif isinstance(o, list):
                nxt.extend(v for v in o if isinstance(v, (dict, list)))
        frontier = nxt
    return None


def truthy(v):
    return bool(v) and v not in ("false", "False", 0)


def load_manifest(j):
    fs = find_key(j, "frame_sequence")
    mp = None
    if isinstance(fs, dict):
        mp = fs.get("manifest_path") or fs.get("path")
    if mp is None:
        mp = find_key(j, "manifest_path")
    if not mp:
        return None, None
    p = Path(mp)
    if not p.is_absolute():
        p = WS / p
    try:
        return json.loads(p.read_text()), p
    except Exception as e:
        log("manifest read failed", path=str(p), err=str(e))
        return None, p


def resolve_path(path, manifest_path=None):
    p = Path(path)
    if p.is_absolute() and p.exists():
        return p
    q = WS / p
    if q.exists():
        return q
    if manifest_path is not None:
        q = Path(manifest_path).parent / p
        if q.exists():
            return q
    return WS / p


def frame_files(frame, manifest_path=None):
    """{(camera, kind): (Path, meters_per_unit)} for a manifest frame."""
    out = {}
    for f in frame.get("files", []) or []:
        try:
            cam = str(f.get("camera", "")).lower()
            kind = str(f.get("kind", "")).lower()
            if "depth" in kind:
                kind = "depth"
            elif kind in ("rgb", "color", "image"):
                kind = "rgb"
            out[(cam, kind)] = (resolve_path(f["path"], manifest_path), float(f.get("meters_per_unit", 0.001) or 0.001))
        except Exception:
            continue
    return out


def load_rgb(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def depth_from_image(im, mpu=0.001):
    a = np.asarray(im)
    if a.ndim == 3:
        return None  # colour image, not depth
    a = a.astype(np.float64)
    d = a * mpu
    d[a <= 0] = np.nan
    return d


def load_depth(path, mpu=0.001):
    with Image.open(path) as im:
        im.load()
        return depth_from_image(im, mpu)


def files_from_json(j):
    """Find {camera, kind, path} dicts anywhere in the JSON (current observation files)."""
    res = {}
    frontier = [j]
    for _ in range(6):
        nxt = []
        for o in frontier:
            if isinstance(o, dict):
                if "path" in o and "camera" in o and "kind" in o:
                    cam = str(o["camera"]).lower()
                    kind = "depth" if "depth" in str(o["kind"]).lower() else "rgb"
                    res.setdefault((cam, kind), (resolve_path(o["path"]), float(o.get("meters_per_unit", 0.001) or 0.001)))
                nxt.extend(v for k, v in o.items() if isinstance(v, (dict, list)) and k != "frame_sequence")
            elif isinstance(o, list):
                nxt.extend(v for v in o if isinstance(v, (dict, list)))
        frontier = nxt
    return res
