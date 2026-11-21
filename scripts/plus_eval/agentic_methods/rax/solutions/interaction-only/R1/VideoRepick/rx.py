"""Robust parsing of RAX/RoboMME MCP replies (format only partly documented).

Everything here is defensive: the exact JSON layout was never observed before the
single rehearsal, so every accessor searches recursively and returns None on failure.
"""
import base64
import io
import json
import os
import re
from pathlib import Path

import numpy as np

WORKSPACE = Path("/workspace")

LOG_LINES = []
_LOG_FILE = None


def log(*parts):
    global _LOG_FILE
    text = " ".join(str(p) for p in parts)
    LOG_LINES.append(text)
    try:
        print(text, flush=True)
    except Exception:
        pass
    try:
        if _LOG_FILE is None:
            os.makedirs("/workspace/output", exist_ok=True)
            _LOG_FILE = open("/workspace/output/controller_log.txt", "a")
        _LOG_FILE.write(text + "\n")
        _LOG_FILE.flush()
    except Exception:
        pass


# ----------------------------------------------------------------------------- JSON
def reply_texts(result):
    out = []
    if isinstance(result, dict):
        for b in result.get("content", []) or []:
            if isinstance(b, dict) and b.get("type") == "text":
                out.append(b.get("text", ""))
    return out


def reply_meta(result):
    """Merge all JSON dicts found in text blocks (first wins on key conflicts) plus
    structuredContent if present."""
    meta = {}
    if not isinstance(result, dict):
        return meta
    sc = result.get("structuredContent")
    candidates = []
    for t in reply_texts(result):
        try:
            obj = json.loads(t)
        except Exception:
            continue
        if isinstance(obj, dict):
            candidates.append(obj)
    if isinstance(sc, dict):
        candidates.append(sc)
    for obj in candidates:
        for k, v in obj.items():
            if k not in meta:
                meta[k] = v
    return meta


def find_all(obj, key, max_depth=12):
    """All values stored under `key` anywhere inside obj (BFS order)."""
    found = []
    queue = [(obj, 0)]
    while queue:
        cur, d = queue.pop(0)
        if d > max_depth:
            continue
        if isinstance(cur, dict):
            for k, v in cur.items():
                if k == key:
                    found.append(v)
                if isinstance(v, (dict, list)):
                    queue.append((v, d + 1))
        elif isinstance(cur, list):
            for v in cur[:2000]:
                if isinstance(v, (dict, list)):
                    queue.append((v, d + 1))
    return found


def find_first(obj, key, default=None, pred=None):
    for v in find_all(obj, key):
        if pred is None or pred(v):
            return v
    return default


def _is_num_list(v, n=None):
    if not isinstance(v, (list, tuple)):
        return False
    if n is not None and len(v) != n:
        return False
    try:
        [float(x) for x in v]
        return True
    except Exception:
        return False


def find_all_depth(obj, key, max_depth=12):
    """[(depth, order, value)] for every occurrence of key (BFS)."""
    found = []
    queue = [(obj, 0)]
    order = 0
    while queue:
        cur, d = queue.pop(0)
        if d > max_depth:
            continue
        if isinstance(cur, dict):
            for k, v in cur.items():
                if k == key:
                    found.append((d, order, v))
                    order += 1
                if isinstance(v, (dict, list)):
                    queue.append((v, d + 1))
        elif isinstance(cur, list):
            for v in cur[:2000]:
                if isinstance(v, (dict, list)):
                    queue.append((v, d + 1))
    return found


def get_state(meta):
    """Shallowest 8-vector 'state'; among equally shallow ones the last (final step)."""
    c = [(d, o, v) for d, o, v in find_all_depth(meta, "state") if _is_num_list(v, 8)]
    if c:
        dmin = min(d for d, _, _ in c)
        v = [v for d, _, v in c if d == dmin][-1]
        return np.array([float(x) for x in v])
    v = find_first(meta, "state", pred=lambda x: _is_num_list(x, 8))
    if v is None:
        # maybe nested {"state": {"values": [...]}}
        for cand in find_all(meta, "state"):
            if isinstance(cand, dict):
                for vv in cand.values():
                    if _is_num_list(vv, 8):
                        v = vv
                        break
            if v is not None:
                break
    return None if v is None else np.array([float(x) for x in v])


def get_bool(meta, key):
    for v in find_all(meta, key):
        if isinstance(v, bool):
            return v
    return None


def any_true(meta, key):
    return any(v is True for v in find_all(meta, key))


def is_ended(meta):
    return any_true(meta, "episode_ended")


def is_complete(meta):
    return any_true(meta, "task_complete")


def get_step_id(meta):
    v = meta.get("step_id") if isinstance(meta, dict) else None
    if isinstance(v, (int, float)):
        return int(v)
    for v in find_all(meta, "step_id"):
        if isinstance(v, (int, float)):
            return int(v)
    return None


def get_instruction(meta):
    for v in find_all(meta, "instruction"):
        if isinstance(v, str) and v.strip():
            return v
    for v in find_all(meta, "goal"):
        if isinstance(v, str) and v.strip():
            return v
    return ""


def remaining_steps(meta):
    """Best-effort remaining control steps; None if unknown."""
    lim = find_first(meta, "episode_limits")
    if isinstance(lim, dict):
        for k, v in lim.items():
            if "remaining" in k and "step" in k and isinstance(v, (int, float)) and not isinstance(v, bool):
                return int(v)
        mx = None
        used = None
        for k, v in lim.items():
            if not isinstance(v, (int, float)):
                continue
            if k in ("max_steps", "step_limit", "max_episode_steps", "limit", "horizon"):
                mx = v
            if k in ("steps", "steps_used", "step", "elapsed_steps", "used", "step_count"):
                used = v
        if mx is not None and used is not None:
            return int(mx - used)
    return None


# ----------------------------------------------------------------------------- images
def _decode_image_bytes(data):
    from PIL import Image
    im = Image.open(io.BytesIO(data))
    im.load()
    return _pil_to_np(im)


def _pil_to_np(im):
    mode = im.mode
    if mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        arr = np.array(im)
        return arr.astype(np.float64), "depth"
    if mode in ("L", "P", "LA", "RGBA", "RGB", "CMYK", "YCbCr"):
        arr = np.array(im.convert("RGB"))
        return arr, "rgb"
    arr = np.array(im)
    if arr.ndim == 2:
        return arr.astype(np.float64), "depth"
    return arr[..., :3].astype(np.uint8), "rgb"


def load_image_file(path_str, base_dirs=()):
    from PIL import Image
    p = Path(path_str)
    cands = []
    if p.is_absolute():
        cands.append(p)
    cands.append(WORKSPACE / p)
    for b in base_dirs:
        cands.append(Path(b) / p)
    for c in cands:
        try:
            if c.is_file():
                with Image.open(c) as im:
                    im.load()
                    return _pil_to_np(im)
        except Exception:
            continue
    return None, None


def _label_from_text(text):
    t = (text or "").lower()
    cam = None
    kind = None
    if "wrist" in t or "hand" in t:
        cam = "wrist"
    elif "front" in t or "third" in t or "base" in t:
        cam = "front"
    if "depth" in t:
        kind = "depth"
    elif "rgb" in t or "color" in t or "colour" in t:
        kind = "rgb"
    return cam, kind


def extract_images(result, meta=None):
    """Return dict {(camera, kind): array} for the reply's inline images; depth arrays
    are converted to metres where a depth scale is known (default 0.001)."""
    out = {}
    if not isinstance(result, dict):
        return out
    if meta is None:
        meta = reply_meta(result)
    blocks = result.get("content", []) or []
    prev_text = ""
    entries = []
    for b in blocks:
        if not isinstance(b, dict):
            continue
        typ = b.get("type")
        if typ == "text":
            prev_text = b.get("text", "")
            continue
        data = None
        if typ == "image" and b.get("data"):
            data = b.get("data")
        elif typ == "resource" and isinstance(b.get("resource"), dict) and b["resource"].get("blob"):
            data = b["resource"]["blob"]
        if data is None:
            continue
        try:
            arr, kind = _decode_image_bytes(base64.b64decode(data))
        except Exception:
            continue
        label = ""
        for k in ("name", "title", "annotations", "description", "camera", "_meta", "uri"):
            if k in b:
                label += " " + json.dumps(b[k]) if not isinstance(b[k], str) else " " + b[k]
        if isinstance(b.get("resource"), dict):
            label += " " + str(b["resource"].get("uri", ""))
        cam, kind2 = _label_from_text(label)
        if cam is None:
            # only use preceding text if it is short (a caption, not the meta JSON)
            if len(prev_text) < 300:
                cam, k3 = _label_from_text(prev_text)
                kind2 = kind2 or k3
        entries.append([cam, kind2 or kind, arr])
        prev_text = ""
    # attachments listing in meta (ordered) as another label source
    att = None
    for key in ("attachments", "images", "image_attachments"):
        v = find_first(meta, key, pred=lambda x: isinstance(x, list) and x and isinstance(x[0], dict))
        if v is not None:
            att = v
            break
    if att is not None and len(att) == len(entries):
        for e, a in zip(entries, att):
            c, k = _label_from_text(json.dumps(a))
            e[0] = e[0] or c
            e[1] = e[1] or k
    # fill unknown cameras by order: first of each kind -> front, second -> wrist
    counts = {}
    for e in entries:
        if e[0] is None:
            n = counts.get(e[1], 0)
            e[0] = "front" if n == 0 else "wrist"
        counts[e[1]] = counts.get(e[1], 0) + 1
    scale = depth_scale(meta)
    for cam, kind, arr in entries:
        if kind == "depth":
            arr = _depth_to_m(arr, scale)
        if (cam, kind) not in out:
            out[(cam, kind)] = arr
    return out


def _depth_to_m(arr, scale):
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 3:
        a = a[..., 0]
    if np.nanmax(a) > 50:  # integer millimetres
        a = a * (scale if scale else 0.001)
    a[~np.isfinite(a)] = 0.0
    return a


def depth_scale(meta):
    v = find_first(meta, "meters_per_unit", pred=lambda x: isinstance(x, (int, float)))
    if v is None:
        v = find_first(meta, "depth_scale", pred=lambda x: isinstance(x, (int, float)))
    return float(v) if v else 0.001


def file_entries(obj):
    """All dicts with camera/kind/path anywhere inside obj."""
    out = []
    queue = [obj]
    n = 0
    while queue and n < 200000:
        cur = queue.pop(0)
        n += 1
        if isinstance(cur, dict):
            if "path" in cur and "camera" in cur and isinstance(cur.get("path"), str):
                out.append(cur)
            for v in cur.values():
                if isinstance(v, (dict, list)):
                    queue.append(v)
        elif isinstance(cur, list):
            queue.extend([v for v in cur if isinstance(v, (dict, list))])
    return out


def manifest_paths(meta):
    out = []
    for v in find_all(meta, "manifest_path"):
        if isinstance(v, str):
            out.append(v)
    return out


def load_manifest(path_str):
    p = Path(path_str)
    for c in ([p] if p.is_absolute() else []) + [WORKSPACE / p, p]:
        try:
            if c.is_file():
                return json.loads(c.read_text()), c.parent
        except Exception:
            continue
    return None, None


def frame_files(frame):
    files = {}
    for f in frame.get("files", []) or []:
        if isinstance(f, dict) and isinstance(f.get("path"), str):
            cam = str(f.get("camera", "")).lower()
            kind = str(f.get("kind", "")).lower()
            files[(cam, kind)] = f
    return files


def load_frame_images(frame, manifest_dir=None, cams=("front", "wrist")):
    """Load rgb (uint8) and depth (metres) arrays for one manifest frame."""
    out = {}
    files = frame_files(frame)
    for (cam, kind), f in files.items():
        if cam not in cams:
            continue
        arr, k = load_image_file(f["path"], base_dirs=[manifest_dir] if manifest_dir else [])
        if arr is None:
            continue
        if kind == "depth" or k == "depth":
            scale = f.get("meters_per_unit") or 0.001
            arr = _depth_to_m(arr, float(scale))
            out[(cam, "depth")] = arr
        else:
            out[(cam, "rgb")] = arr
    return out


# ----------------------------------------------------------------------------- calibration
def get_calibration(meta, cam):
    """Return (K 3x3 or None, cam_pose dict or None) from meta."""
    K = None
    pose = None
    intr = find_first(meta, "camera_intrinsics")
    if isinstance(intr, dict):
        c = intr.get(cam)
        if c is None:
            for k, v in intr.items():
                if cam in str(k).lower():
                    c = v
        K = _parse_K(c)
    poses = find_first(meta, "camera_poses")
    if isinstance(poses, dict):
        c = poses.get(cam)
        if c is None:
            for k, v in poses.items():
                if cam in str(k).lower():
                    c = v
        pose = _parse_pose(c)
    return K, pose


def _parse_K(c):
    try:
        if c is None:
            return None
        if isinstance(c, dict):
            for key in ("K", "matrix", "intrinsic_matrix", "intrinsics", "camera_matrix"):
                if key in c:
                    return _parse_K(c[key])
            if "fx" in c:
                return np.array([[c["fx"], 0, c.get("cx", 128)], [0, c.get("fy", c["fx"]), c.get("cy", 128)], [0, 0, 1]], float)
            return None
        a = np.array(c, dtype=float)
        if a.shape == (3, 3):
            return a
        if a.size == 9:
            return a.reshape(3, 3)
        if a.size == 4:
            return np.array([[a[0], 0, a[2]], [0, a[1], a[3]], [0, 0, 1]])
    except Exception:
        return None
    return None


def _parse_pose(c):
    try:
        if c is None:
            return None
        if isinstance(c, dict):
            pos = None
            q = None
            for key in ('position_m', "position", "p", "translation", "xyz"):
                if key in c and _is_num_list(c[key], 3):
                    pos = np.array(c[key], float)
            for key in ("quaternion_wxyz", "wxyz", "quaternion", "q", "orientation"):
                if key in c and _is_num_list(c[key], 4):
                    q = np.array(c[key], float)
                    qkey = key
                    break
            if pos is not None and q is not None:
                order = "wxyz"
                if "xyzw" in json.dumps(c).lower():
                    order = "xyzw"
                return {"position": pos, "quat": q, "order": order}
            for key in ("matrix", "pose", "extrinsic", "cam2world", "T"):
                if key in c:
                    m = np.array(c[key], float)
                    if m.size == 16:
                        return {"matrix": m.reshape(4, 4)}
            return None
        m = np.array(c, float)
        if m.size == 16:
            return {"matrix": m.reshape(4, 4)}
        if m.size == 7:
            return {"position": m[:3], "quat": m[3:], "order": "wxyz"}
    except Exception:
        return None
    return None


# ----------------------------------------------------------------------------- pixel tool
def parse_pixel_points(result, n):
    """Parse robomme_pixel_to_position reply into list of (xyz or None)."""
    meta = reply_meta(result)
    lst = None
    for key in ("points", "positions", "results", "selections", "world_points"):
        v = find_first(meta, key, pred=lambda x: isinstance(x, list))
        if v is not None and len(v) == n:
            lst = v
            break
    if lst is None:
        # any list of length n of dicts
        queue = [meta]
        while queue:
            cur = queue.pop(0)
            if isinstance(cur, dict):
                for v in cur.values():
                    if isinstance(v, list) and len(v) == n and all(isinstance(x, dict) or _is_num_list(x, 3) for x in v):
                        lst = v
                        break
                    if isinstance(v, (dict, list)):
                        queue.append(v)
            elif isinstance(cur, list):
                queue.extend([v for v in cur if isinstance(v, (dict, list))])
            if lst is not None:
                break
    out = []
    if lst is None:
        return [None] * n
    for item in lst:
        out.append(_point_from_item(item))
    return out


def _point_from_item(item):
    if _is_num_list(item, 3):
        return np.array(item, float)
    if not isinstance(item, dict):
        return None
    if item.get("valid") is False:
        return None
    for key in ("position", "world_position", "world", "point", "xyz", "position_m", "world_point", "position_world"):
        v = item.get(key)
        if _is_num_list(v, 3):
            return np.array(v, float)
        if isinstance(v, dict) and all(k in v for k in ("x", "y", "z")):
            return np.array([v["x"], v["y"], v["z"]], float)
    if all(k in item for k in ("x", "y", "z")):
        try:
            return np.array([item["x"], item["y"], item["z"]], float)
        except Exception:
            return None
    for v in item.values():
        if _is_num_list(v, 3):
            return np.array(v, float)
    return None


# ----------------------------------------------------------------------------- instruction
_NUM_WORDS = {"once": 1, "twice": 2, "thrice": 3, "one": 1, "two": 2, "three": 3, "four": 4,
              "five": 5, "six": 6, "single": 1}


def parse_instruction(text):
    t = (text or "").lower()
    count = None
    m = re.search(r"(\d+)\s*(?:times|x\b)", t)
    if m:
        count = int(m.group(1))
    if count is None:
        for w, n in _NUM_WORDS.items():
            if re.search(r"\b" + w + r"\b", t) and (w in ("once", "twice", "thrice") or re.search(r"\b" + w + r"\s+times?\b", t)):
                count = n
                break
    place = bool(re.search(r"\b(put|place|set|lay|drop|release|down|back)\b", t))
    ordinal = None
    for w, n in (("first", 0), ("second", 1), ("third", 2), ("last", -1), ("final", -1)):
        if re.search(r"\b" + w + r"\b", t):
            ordinal = n
            break
    return {"count": count, "place": place, "ordinal": ordinal, "text": text}
