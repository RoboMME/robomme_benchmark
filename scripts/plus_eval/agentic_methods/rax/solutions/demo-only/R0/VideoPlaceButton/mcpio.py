"""Defensive parsing of MCP replies (JSON metadata + image blocks) for RoboMME-Plus.

The exact live packet layout was never observed (demo-only: zero simulator starts), so every
accessor searches the decoded JSON recursively and validates types.
"""
import base64
import io
import json
import numpy as np
from PIL import Image


def blocks(result):
    if isinstance(result, dict):
        return result.get("content") or []
    if isinstance(result, list):
        return result
    return []


def meta_of(result):
    """Merge JSON from text blocks (first block has priority) and structuredContent."""
    meta = {}
    texts = []
    for b in blocks(result):
        if isinstance(b, dict) and b.get("type") == "text":
            try:
                v = json.loads(b.get("text", ""))
                texts.append(v)
            except Exception:
                pass
    for v in reversed(texts):
        if isinstance(v, dict):
            meta.update(v)
    if isinstance(result, dict) and isinstance(result.get("structuredContent"), dict):
        for k, v in result["structuredContent"].items():
            meta.setdefault(k, v)
    if not meta and texts:
        meta = {"_list": texts}
    return meta


def images_of(result):
    """List of (PIL image, mimeType, block index) for image blocks."""
    out = []
    for i, b in enumerate(blocks(result)):
        if isinstance(b, dict) and b.get("type") == "image" and b.get("data"):
            try:
                im = Image.open(io.BytesIO(base64.b64decode(b["data"])))
                im.load()
                out.append((im, b.get("mimeType", ""), i))
            except Exception:
                pass
    return out


def walk(obj, depth=0):
    """Breadth-first walk yielding (key, value) for every dict entry."""
    queue = [obj]
    while queue:
        nxt = []
        for o in queue:
            if isinstance(o, dict):
                for k, v in o.items():
                    yield k, v
                    nxt.append(v)
            elif isinstance(o, list):
                nxt.extend(x for x in o if isinstance(x, (dict, list)))
        queue = nxt


def find(obj, key, pred=None):
    for k, v in walk(obj):
        if k == key and (pred is None or pred(v)):
            return v
    return None


def is_numlist(v, n=None):
    if not isinstance(v, (list, tuple)):
        return False
    if n is not None and len(v) != n:
        return False
    return all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v)


def state_of(meta):
    s = find(meta, "state", lambda v: is_numlist(v) and len(v) >= 8)
    return None if s is None else np.asarray(s[:8], float)


def step_id_of(meta):
    for key in ("step_id", "observation_step_id"):
        v = find(meta, key, lambda v: isinstance(v, int) and not isinstance(v, bool))
        if v is not None:
            return v
    return None


def bool_of(meta, key):
    v = find(meta, key, lambda v: isinstance(v, bool))
    return bool(v) if v is not None else False


def instruction_of(meta):
    return find(meta, "instruction", lambda v: isinstance(v, str))


def manifest_path_of(meta):
    fs = find(meta, "frame_sequence", lambda v: isinstance(v, dict))
    if fs and isinstance(fs.get("manifest_path"), str):
        return fs["manifest_path"]
    return find(meta, "manifest_path", lambda v: isinstance(v, str))


def ended_of(meta):
    if bool_of(meta, "episode_ended") or bool_of(meta, "task_complete"):
        return True
    for k in ("terminated", "truncated", "done"):
        if bool_of(meta, k):
            return True
    return False


def remaining_steps_of(meta):
    lim = find(meta, "episode_limits", lambda v: isinstance(v, dict))
    if not lim:
        return None
    for k, v in lim.items():
        if "remaining" in k and "step" in k and isinstance(v, (int, float)):
            return int(v)
    mx = None
    used = None
    for k, v in lim.items():
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            continue
        if k in ("max_steps", "step_limit", "max_episode_steps", "steps_limit"):
            mx = v
        if k in ("steps_used", "step", "steps", "elapsed_steps", "step_id", "used_steps"):
            used = v
    if mx is not None and used is not None:
        return int(mx - used)
    return None


def to_rgb_array(im):
    return np.asarray(im.convert("RGB"), np.uint8)


def is_depth_image(im):
    return im.mode in ("I;16", "I;16B", "I;16L", "I", "F")


def camera_labels(meta, n_images):
    """Try to find metadata describing image attachments in order (camera, kind)."""
    best = None
    for k, v in walk(meta):
        if isinstance(v, list) and len(v) == n_images and n_images > 0 and \
                all(isinstance(x, dict) and "camera" in x for x in v):
            best = [(str(x.get("camera")), str(x.get("kind", x.get("type", "")))) for x in v]
            break
    return best


def split_images(result, meta=None):
    """Return {'front': rgb, 'wrist': rgb, 'front_depth': m, 'wrist_depth': m} (best effort)."""
    meta = meta if meta is not None else meta_of(result)
    ims = images_of(result)
    out = {}
    labels = camera_labels(meta, len(ims))
    if labels:
        for (im, mime, _), (cam, kind) in zip(ims, labels):
            cam = "front" if "front" in cam else ("wrist" if "wrist" in cam or "hand" in cam else cam)
            if "depth" in kind or is_depth_image(im):
                out.setdefault(cam + "_depth", np.asarray(im, np.float64) * 0.001)
            else:
                out.setdefault(cam, to_rgb_array(im))
        if "front" in out and "wrist" in out:
            return out
        out = {}
    rgbs = [im for im, _, _ in ims if not is_depth_image(im)]
    deps = [im for im, _, _ in ims if is_depth_image(im)]
    if len(rgbs) >= 1:
        out["front"] = to_rgb_array(rgbs[0])
    if len(rgbs) >= 2:
        out["wrist"] = to_rgb_array(rgbs[1])
    if len(deps) >= 1:
        out["front_depth"] = np.asarray(deps[0], np.float64) * 0.001
    if len(deps) >= 2:
        out["wrist_depth"] = np.asarray(deps[1], np.float64) * 0.001
    return out


def ptp_points(result, n_expected):
    """Parse robomme_pixel_to_position reply -> list of (xyz or None, spread or None)."""
    meta = meta_of(result)
    cands = []
    for k, v in walk(meta):
        if isinstance(v, list) and len(v) == n_expected and all(isinstance(x, dict) for x in v):
            cands.append(v)
    out = None
    for lst in cands:
        pts = []
        okcount = 0
        for d in lst:
            p = None
            for key in ("world_position", "position", "world_point", "point", "xyz",
                        "position_m", "world", "world_xyz", "point_world"):
                v = d.get(key)
                if is_numlist(v, 3):
                    p = np.asarray(v, float)
                    break
                if isinstance(v, dict) and all(c in v for c in "xyz"):
                    p = np.asarray([v["x"], v["y"], v["z"]], float)
                    break
            if p is None:
                for kk, v in d.items():
                    if is_numlist(v, 3) and kk not in ("pixel", "uv", "color", "rgb"):
                        p = np.asarray(v, float)
                        break
            valid = d.get("valid", True)
            if valid is False or p is None or not np.all(np.isfinite(p)):
                p = None
            else:
                okcount += 1
            spread = d.get("depth_spread_m")
            pts.append((p, spread if isinstance(spread, (int, float)) else None))
        if okcount > 0:
            out = pts
            break
    return out
