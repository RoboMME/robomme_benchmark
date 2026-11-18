"""Defensive parsing of RAX/RoboMME MCP replies (format known only from docs)."""
import base64
import io
import json
import os
import time

import numpy as np
from PIL import Image

WORKSPACE = "/workspace"


def log(*args):
    msg = " ".join(str(a) for a in args)
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def reply_meta(result):
    """First JSON object found in a tool result (structuredContent or a text block)."""
    if not isinstance(result, dict):
        return {}
    sc = result.get("structuredContent")
    if isinstance(sc, dict) and sc:
        return sc
    for block in result.get("content", []) or []:
        if isinstance(block, dict) and block.get("type") == "text":
            txt = block.get("text", "")
            try:
                obj = json.loads(txt)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue
    return {}


def find_key(obj, key, depth=0):
    """Depth-first search for the first value stored under `key`."""
    if depth > 8:
        return None
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            r = find_key(v, key, depth + 1)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj[:200]:
            r = find_key(v, key, depth + 1)
            if r is not None:
                return r
    return None


def find_all(obj, key, out=None, depth=0):
    if out is None:
        out = []
    if depth > 8:
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                out.append(v)
            find_all(v, key, out, depth + 1)
    elif isinstance(obj, list):
        for v in obj[:2000]:
            find_all(v, key, out, depth + 1)
    return out


def get_state(meta):
    """8-D state vector from a reply (last 'state' that is an 8-number list)."""
    cands = find_all(meta, "state")
    best = None
    for c in cands:
        if isinstance(c, dict):
            for k in ("values", "vector", "data"):
                if isinstance(c.get(k), list):
                    c = c[k]
                    break
        if isinstance(c, list) and len(c) >= 8 and all(isinstance(x, (int, float)) for x in c[:8]):
            best = [float(x) for x in c[:8]]
            # prefer a top-level/observation state: first found by DFS is usually top-level
            break
    if best is None:
        obs = find_key(meta, "observation")
        if isinstance(obs, dict):
            return get_state(obs)
    return best


def get_step_id(meta):
    v = meta.get("step_id") if isinstance(meta, dict) else None
    if isinstance(v, int):
        return v
    v = find_key(meta, "step_id")
    return v if isinstance(v, int) else None


def episode_ended(meta):
    for k in ("episode_ended", "terminated", "truncated"):
        v = meta.get(k) if isinstance(meta, dict) else None
        if v is True:
            return True
    v = find_key(meta, "episode_ended")
    if v is True:
        return True
    v = find_key(meta, "task_complete")
    if v is True:
        return True
    return False


def task_complete(meta):
    return find_key(meta, "task_complete") is True


def manifest_path(meta):
    fs = find_key(meta, "frame_sequence")
    if isinstance(fs, dict):
        p = fs.get("manifest_path") or find_key(fs, "manifest_path")
        if isinstance(p, str):
            return p
    p = find_key(meta, "manifest_path")
    return p if isinstance(p, str) else None


def abspath(p):
    if p is None:
        return None
    if os.path.isabs(p):
        return p
    return os.path.join(WORKSPACE, p)


def load_rgb(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def load_depth(path, meters_per_unit=0.001):
    with Image.open(path) as im:
        a = np.asarray(im).astype(np.float64)
    if a.ndim == 3:
        a = a[..., 0]
    return a * float(meters_per_unit or 0.001)


def read_manifest(path):
    p = abspath(path)
    with open(p) as f:
        man = json.load(f)
    frames = man.get("frames", []) if isinstance(man, dict) else []
    out = []
    for k, fr in enumerate(frames):
        files = {}
        for f in fr.get("files", []) or []:
            cam, kind, fp = f.get("camera"), f.get("kind"), f.get("path")
            if cam and kind and fp:
                files[(cam, kind)] = dict(path=fp, mpu=f.get("meters_per_unit", 0.001))
        out.append(dict(frame_index=fr.get("frame_index", k), phase=fr.get("phase"),
                        timestamp_s=fr.get("timestamp_s"), state=fr.get("state"),
                        step_id=fr.get("step_id"), files=files, raw=fr))
    out.sort(key=lambda d: (d["frame_index"] if isinstance(d["frame_index"], (int, float)) else 0))
    return man, out


def resolve_file(fp, manifest_file):
    """Paths are documented as workspace-relative; fall back to manifest-relative."""
    cands = [abspath(fp)]
    if manifest_file:
        cands.append(os.path.join(os.path.dirname(abspath(manifest_file)), fp))
    for c in cands:
        if c and os.path.exists(c):
            return c
    return cands[0]


def inline_images(result):
    """List of (label, rgb array) for image blocks in a tool result."""
    out = []
    label = None
    for block in (result.get("content", []) if isinstance(result, dict) else []) or []:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            t = block.get("text", "")
            if len(t) < 200:
                tl = t.lower()
                label = "front" if "front" in tl else ("wrist" if "wrist" in tl else label)
        elif block.get("type") == "image":
            try:
                data = base64.b64decode(block.get("data", ""))
                with Image.open(io.BytesIO(data)) as im:
                    arr = np.asarray(im.convert("RGB"))
                out.append((label, arr))
            except Exception:
                pass
            label = None
    return out


def pick_front(images, reference):
    """Choose the inline image that is the front camera, using the static front
    reference image (the front camera does not move)."""
    if not images:
        return None
    labelled = [im for lab, im in images if lab == "front"]
    if len(labelled) == 1:
        return labelled[0]
    if reference is None:
        return images[0][1]
    best, bestd = None, 1e18
    for lab, im in images:
        if im.shape != reference.shape:
            continue
        d = float(np.mean(np.abs(im.astype(np.int16) - reference.astype(np.int16))))
        if d < bestd:
            best, bestd = im, d
    return best if best is not None else images[0][1]


def _is_vec3(v):
    return (isinstance(v, (list, tuple)) and len(v) == 3
            and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v))


_POINT_KEYS = ("position", "world_position", "position_m", "world", "point", "xyz", "world_point",
               "point_world", "world_xyz", "position_world", "world_position_m", "point_m", "p")


def _vec_from(o):
    """3-vector stored in dict o under a point-like key (or nested x/y/z)."""
    for k in _POINT_KEYS:
        if k in o:
            v = o[k]
            if _is_vec3(v):
                return v
            if isinstance(v, dict):
                for kk in ("xyz", "position", "value", "world", "m"):
                    if _is_vec3(v.get(kk)):
                        return v[kk]
                if all(isinstance(v.get(c), (int, float)) for c in ("x", "y", "z")):
                    return [v["x"], v["y"], v["z"]]
    for k, v in o.items():
        kl = str(k).lower()
        if _is_vec3(v) and ("world" in kl or "point" in kl or "xyz" in kl or kl.startswith("pos")) \
                and "camera" not in kl and "cam" not in kl:
            return v
    if all(isinstance(o.get(c), (int, float)) for c in ("x", "y", "z")):
        return [o["x"], o["y"], o["z"]]
    return None


def parse_points(meta, n):
    """Extract n world points from a robomme_pixel_to_position reply.
    Returns list of (xyz or None, spread or None)."""
    recs = []

    def walk(o, depth=0):
        if depth > 8:
            return
        if isinstance(o, dict):
            selection_like = any(k in o for k in ("u", "v", "pixel", "camera", "valid", "depth_m",
                                                  "depth_spread_m", "selection"))
            vec = _vec_from(o)
            if selection_like and (vec is not None or "valid" in o):
                valid = o.get("valid", True)
                spread = o.get("depth_spread_m")
                recs.append((None if (vec is None or valid is False) else [float(x) for x in vec],
                             float(spread) if isinstance(spread, (int, float)) else None))
                return
            for v in o.values():
                walk(v, depth + 1)
        elif isinstance(o, list):
            for v in o:
                walk(v, depth + 1)

    walk(meta)
    if len(recs) < n:
        # fallback: a list of n 3-vectors (or nulls)
        lists = []

        def walk2(o, depth=0):
            if depth > 8:
                return
            if isinstance(o, dict):
                for v in o.values():
                    walk2(v, depth + 1)
            elif isinstance(o, list):
                if len(o) == n and all((_is_vec3(x) or x is None) for x in o) and any(_is_vec3(x) for x in o):
                    lists.append(o)
                    return
                for v in o:
                    walk2(v, depth + 1)

        walk2(meta)
        if lists:
            recs = [(None if x is None else [float(c) for c in x], None) for x in lists[0]]
        else:
            # last resort: any dicts with point-like vectors
            recs2 = []

            def walk3(o, depth=0):
                if depth > 8:
                    return
                if isinstance(o, dict):
                    vec = _vec_from(o)
                    if vec is not None:
                        recs2.append(([float(x) for x in vec], None))
                        return
                    for v in o.values():
                        walk3(v, depth + 1)
                elif isinstance(o, list):
                    for v in o:
                        walk3(v, depth + 1)

            walk3(meta)
            if len(recs2) >= n:
                recs = recs2[-n:] if len(recs2) > n else recs2
    return recs[:n] if len(recs) >= n else recs + [(None, None)] * (n - len(recs))
