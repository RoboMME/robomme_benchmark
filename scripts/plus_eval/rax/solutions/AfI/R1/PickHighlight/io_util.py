"""Generic, defensive parsing of MCP replies (formats never seen live)."""
import base64
import io
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

WS = Path("/workspace")
OUT = WS / "output"


def log(*a):
    msg = " ".join(str(x) for x in a)
    line = "[%7.1f] %s" % (time.time() - T0, msg)
    print(line, flush=True)
    try:
        with open(OUT / "controller_log.txt", "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


T0 = time.time()


def save_json(name, obj, limit=200000):
    try:
        OUT.mkdir(parents=True, exist_ok=True)
        s = json.dumps(obj, default=str, indent=1)
        if len(s) > limit:
            s = s[:limit] + "\n...TRUNCATED"
        (OUT / name).write_text(s)
    except Exception as e:  # never fail on logging
        print("save_json failed", name, e)


def strip_blobs(obj, depth=0):
    """Copy of a reply with base64 blobs removed (for logging)."""
    if depth > 30:
        return "..."
    if isinstance(obj, dict):
        r = {}
        for k, v in obj.items():
            if k in ("data", "blob") and isinstance(v, str) and len(v) > 200:
                r[k] = "<%d chars>" % len(v)
            else:
                r[k] = strip_blobs(v, depth + 1)
        return r
    if isinstance(obj, list):
        if len(obj) > 60 and all(isinstance(x, (int, float)) for x in obj[:60]):
            return obj[:20] + ["...%d" % len(obj)]
        return [strip_blobs(x, depth + 1) for x in obj]
    return obj


def reply_meta(reply):
    """Merge the JSON of text blocks and structuredContent into one dict."""
    meta = {}
    texts = []
    if isinstance(reply, dict):
        sc = reply.get("structuredContent")
        if isinstance(sc, dict):
            meta.update(sc)
        for b in reply.get("content", []) or []:
            if isinstance(b, dict) and b.get("type") == "text":
                t = b.get("text", "")
                try:
                    j = json.loads(t)
                    if isinstance(j, dict):
                        for k, v in j.items():
                            meta.setdefault(k, v)
                        continue
                except Exception:
                    pass
                texts.append(t)
    if texts:
        meta.setdefault("_texts", texts)
    return meta


def walk(obj, path=()):
    """Yield (path, key, value) for every dict entry, depth-first in order."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield path, k, v
            yield from walk(v, path + (k,))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk(v, path + (i,))


def find_key(obj, names, pred=None):
    """All values whose key is in names (in traversal order)."""
    out = []
    for p, k, v in walk(obj):
        if k in names and (pred is None or pred(v)):
            out.append((p, v))
    return out


def is_numlist(v, n=None):
    return isinstance(v, list) and (n is None or len(v) == n) and len(v) > 0 and all(
        isinstance(x, (int, float)) and not isinstance(x, bool) for x in v)


def get_state(meta):
    for keypath in (("observation", "state"), ("state",), ("obs", "state")):
        cur = meta
        ok = True
        for k in keypath:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                ok = False
                break
        if ok and is_numlist(cur, 8):
            return [float(x) for x in cur]
    c = find_key(meta, ("state", "robot_state", "proprio"), lambda v: is_numlist(v, 8))
    if c:
        # prefer shortest path (top level), then last
        c.sort(key=lambda pv: len(pv[0]))
        return [float(x) for x in c[0][1]]
    return None


def get_step_id(meta):
    for k in ("step_id", "step", "observation_step_id"):
        if isinstance(meta.get(k), int):
            return meta[k]
    if isinstance(meta.get("observation"), dict):
        v = meta["observation"].get("step_id")
        if isinstance(v, int):
            return v
    c = find_key(meta, ("step_id",), lambda v: isinstance(v, int) and not isinstance(v, bool))
    if c:
        c.sort(key=lambda pv: len(pv[0]))
        return c[0][1]
    return None


def get_flag(meta, names):
    for name in names:
        c = find_key(meta, (name,), lambda v: isinstance(v, bool))
        if c:
            c.sort(key=lambda pv: len(pv[0]))
            return c[0][1]
    return None


def episode_ended(meta):
    v = get_flag(meta, ("episode_ended",))
    return bool(v)


def task_complete(meta):
    v = get_flag(meta, ("task_complete",))
    return bool(v)


def get_instruction(meta):
    for k in ("instruction", "goal", "language_instruction", "task_instruction"):
        c = find_key(meta, (k,), lambda v: isinstance(v, str) and len(v) > 3)
        if c:
            c.sort(key=lambda pv: len(pv[0]))
            return c[0][1]
    return ""


def _decode_image(data_b64):
    raw = base64.b64decode(data_b64)
    im = Image.open(io.BytesIO(raw))
    im.load()
    return im


def pil_to_arrays(im):
    """Return ('rgb', HxWx3 uint8) or ('depth', HxW float raw units)."""
    mode = im.mode
    if mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        a = np.array(im).astype(np.float64)
        return "depth", a
    if mode in ("RGB", "RGBA", "P", "L", "LA"):
        if mode == "L":
            return "gray", np.array(im)
        return "rgb", np.array(im.convert("RGB"))
    return "rgb", np.array(im.convert("RGB"))


def block_label(b, prev_text):
    parts = []
    for k, v in b.items():
        if k in ("data", "blob"):
            continue
        parts.append(json.dumps(v, default=str) if not isinstance(v, str) else v)
        parts.append(str(k))
    if prev_text and len(prev_text) < 300:
        parts.append(prev_text)
    return " ".join(parts).lower()


def reply_images(reply):
    """Decode image blocks: list of dict(kind, arr, label, index)."""
    out = []
    prev_text = ""
    if not isinstance(reply, dict):
        return out
    for i, b in enumerate(reply.get("content", []) or []):
        if not isinstance(b, dict):
            continue
        t = b.get("type")
        if t == "text":
            prev_text = b.get("text", "")
            continue
        data = None
        if t == "image" and isinstance(b.get("data"), str):
            data = b["data"]
        elif t == "resource" and isinstance(b.get("resource"), dict) and isinstance(b["resource"].get("blob"), str):
            data = b["resource"]["blob"]
        if data is None:
            continue
        try:
            im = _decode_image(data)
            kind, arr = pil_to_arrays(im)
            out.append({"kind": kind, "arr": arr, "label": block_label(b, prev_text), "index": i,
                        "mode": im.mode})
        except Exception as e:
            out.append({"kind": "error", "err": str(e), "index": i, "label": ""})
        prev_text = ""
    return out


def assign_cameras(images, meta_cams=None):
    """Map decoded images to {('front','rgb'): arr, ...} using labels, else order."""
    res = {}
    rgbs = [im for im in images if im["kind"] == "rgb"]
    deps = [im for im in images if im["kind"] == "depth"]
    for group, kind in ((rgbs, "rgb"), (deps, "depth")):
        unl = []
        for im in group:
            lab = im["label"]
            cam = None
            if "wrist" in lab or "hand" in lab:
                cam = "wrist"
            elif "front" in lab or "third" in lab or "base" in lab:
                cam = "front"
            if "depth" in lab and kind == "rgb":
                pass
            if cam and (cam, kind) not in res:
                res[(cam, kind)] = im["arr"]
            else:
                unl.append(im)
        order = ["front", "wrist"]
        for im in unl:
            for cam in order:
                if (cam, kind) not in res:
                    res[(cam, kind)] = im["arr"]
                    break
    return res


def read_manifest(path_str):
    p = Path(path_str)
    if not p.is_absolute():
        p = WS / p
    return json.loads(p.read_text()), p


def resolve_path(path_str, manifest_path=None):
    p = Path(path_str)
    if p.is_absolute() and p.exists():
        return p
    cands = [WS / p]
    if manifest_path is not None:
        cands.append(Path(manifest_path).parent / p)
    for c in cands:
        if c.exists():
            return c
    return cands[0]


def load_file_image(path):
    with Image.open(path) as im:
        im.load()
        return pil_to_arrays(im)


def frame_files(frame, manifest_path=None):
    """{(camera, kind): (path, meters_per_unit)}"""
    res = {}
    for f in frame.get("files", []) or []:
        try:
            cam = str(f.get("camera", "")).lower()
            kind = str(f.get("kind", "")).lower()
            if "wrist" in cam or "hand" in cam:
                cam = "wrist"
            elif cam:
                cam = "front" if ("front" in cam or "third" in cam or "base" in cam) else cam
            res[(cam, kind)] = (resolve_path(f["path"], manifest_path), f.get("meters_per_unit", 0.001))
        except Exception:
            pass
    return res


def find_manifest_path(meta):
    c = find_key(meta, ("manifest_path",), lambda v: isinstance(v, str))
    paths = [v for _, v in c]
    # prefer one under a frame_sequence key
    fs = find_key(meta, ("frame_sequence",), lambda v: isinstance(v, dict))
    for _, v in fs:
        if isinstance(v.get("manifest_path"), str):
            return v["manifest_path"]
    return paths[0] if paths else None


# ---------------- calibration -----------------
def _K_from(v):
    if isinstance(v, dict):
        lv = {str(k).lower(): x for k, x in v.items()}
        if all(k in lv for k in ("fx", "fy", "cx", "cy")):
            return tuple(float(lv[k]) for k in ("fx", "fy", "cx", "cy"))
        for k in ("k", "matrix", "intrinsic_cv", "intrinsics", "camera_matrix", "intrinsic"):
            if k in lv:
                r = _K_from(lv[k])
                if r:
                    return r
        return None
    if isinstance(v, list):
        a = None
        try:
            a = np.array(v, dtype=float)
        except Exception:
            return None
        if a.shape == (3, 3):
            return (a[0, 0], a[1, 1], a[0, 2], a[1, 2])
        if a.shape == (9,):
            a = a.reshape(3, 3)
            return (a[0, 0], a[1, 1], a[0, 2], a[1, 2])
        if a.shape == (4,):
            return tuple(a)
    return None


def _pose_from(v):
    """Return (R, t, convention_str) camera-to-world if recognisable."""
    if isinstance(v, dict):
        lv = {str(k).lower(): x for k, x in v.items()}
        conv = str(lv.get("convention", lv.get("frame_convention", ""))).lower()
        pos = None
        for k in ('position_m', "position", "p", "translation", "pos", "xyz"):
            if is_numlist(lv.get(k), 3):
                pos = np.array(lv[k], dtype=float)
                break
        q = None
        for k in ("quaternion_wxyz", "quat_wxyz", "wxyz", "q", "quaternion", "orientation"):
            if is_numlist(lv.get(k), 4):
                q = lv[k]
                break
        if pos is not None and q is not None:
            from geom import quat_wxyz_to_R
            return quat_wxyz_to_R(q), pos, conv
        for k in ("matrix", "cam2world", "cam2world_gl", "pose", "transform", "model_matrix", "extrinsic"):
            if k in lv:
                r = _pose_from(lv[k])
                if r:
                    return r[0], r[1], conv or r[2]
        return None
    if isinstance(v, list):
        try:
            a = np.array(v, dtype=float)
        except Exception:
            return None
        if a.shape == (4, 4):
            return a[:3, :3], a[:3, 3], ""
        if a.shape == (16,):
            a = a.reshape(4, 4)
            return a[:3, :3], a[:3, 3], ""
        if a.shape == (7,):
            from geom import quat_wxyz_to_R
            return quat_wxyz_to_R(a[3:]), a[:3], ""
    return None


def _cam_norm(name):
    n = str(name).lower()
    if "wrist" in n or "hand" in n:
        return "wrist"
    if "front" in n or "third" in n or "base" in n:
        return "front"
    return n


def get_calibration(meta):
    """{'front': {'K':(fx,fy,cx,cy), 'R':..., 't':..., 'conv':str}, ...}"""
    cal = {}
    for _, v in find_key(meta, ("camera_intrinsics", "intrinsics", "camera_intrinsic")):
        if isinstance(v, dict):
            for cam, kv in v.items():
                K = _K_from(kv)
                if K:
                    cal.setdefault(_cam_norm(cam), {})["K"] = K
    for _, v in find_key(meta, ("camera_poses", "camera_pose", "extrinsics", "camera_extrinsics")):
        if isinstance(v, dict):
            direct = _pose_from(v)
            if direct is None:
                for cam, pv in v.items():
                    r = _pose_from(pv)
                    if r:
                        d = cal.setdefault(_cam_norm(cam), {})
                        d["R"], d["t"], d["conv"] = r
    # cameras listed as dicts with name + intrinsics + pose
    for _, v in find_key(meta, ("cameras",)):
        items = v.items() if isinstance(v, dict) else enumerate(v) if isinstance(v, list) else []
        for cam, cv in items:
            if not isinstance(cv, dict):
                continue
            name = _cam_norm(cv.get("name", cam))
            for kk in ("intrinsics", "camera_intrinsics", "K"):
                if kk in cv:
                    K = _K_from(cv[kk])
                    if K:
                        cal.setdefault(name, {}).setdefault("K", K)
            for kk in ("pose", "camera_pose", "extrinsics"):
                if kk in cv:
                    r = _pose_from(cv[kk])
                    if r:
                        d = cal.setdefault(name, {})
                        if "R" not in d:
                            d["R"], d["t"], d["conv"] = r
    return cal


def depth_scale_hint(meta):
    c = find_key(meta, ("meters_per_unit", "depth_scale", "depth_units_m"),
                 lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0)
    if c:
        return float(c[0][1])
    return 0.001
