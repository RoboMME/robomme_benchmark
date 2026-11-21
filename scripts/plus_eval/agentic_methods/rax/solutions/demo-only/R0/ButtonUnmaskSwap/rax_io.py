"""Defensive parsing of RAX MCP replies (format partly unknown offline).

Everything here tolerates missing fields; nothing raises on odd input.
"""
import base64
import io
import json
import os
import re

import numpy as np
from PIL import Image

WS = "/workspace"


def first_json(reply):
    """JSON metadata of the first parseable text block."""
    try:
        for b in reply.get("content", []) or []:
            if b.get("type") == "text":
                t = b.get("text", "")
                try:
                    return json.loads(t)
                except Exception:
                    continue
        sc = reply.get("structuredContent")
        if isinstance(sc, dict):
            return sc
    except Exception:
        pass
    return {}


def walk(obj, depth=0):
    """Yield every dict/list node, breadth-first (top-level keys first)."""
    from collections import deque
    q = deque([(obj, 0)])
    n = 0
    while q and n < 20000:
        node, d = q.popleft()
        n += 1
        yield node
        if d > 12:
            continue
        if isinstance(node, dict):
            for v in node.values():
                if isinstance(v, (dict, list)):
                    q.append((v, d + 1))
        elif isinstance(node, list):
            for v in node:
                if isinstance(v, (dict, list)):
                    q.append((v, d + 1))


def find_key(obj, names, depth=0):
    """First value for the highest-priority key in `names` found recursively (BFS-ish order)."""
    for name in names:
        name = name.lower()
        for node in walk(obj):
            if isinstance(node, dict):
                for k, v in node.items():
                    if isinstance(k, str) and k.lower() == name:
                        return v
    return None


def _decode_image_bytes(raw):
    im = Image.open(io.BytesIO(raw))
    im.load()
    return im


def _pil_to_array(im):
    mode = im.mode
    if mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        a = np.array(im)
        return "depth16", a.astype(np.float64)
    if mode in ("L", "LA"):
        return "gray", np.array(im.convert("L")).astype(np.float64)
    return "rgb", np.array(im.convert("RGB"))


def _is_wrist_rgb(rgb):
    """Wrist view shows dark gripper fingers at the bottom corners."""
    try:
        h, w = rgb.shape[:2]
        v = rgb.max(axis=2)
        bl = v[int(h * 0.6):, : int(w * 0.3)]
        br = v[int(h * 0.6):, int(w * 0.7):]
        d = 0.5 * ((bl < 70).mean() + (br < 70).mean())
        return d
    except Exception:
        return 0.0


def _label_from(text):
    t = (text or "").lower()
    cam = "front" if "front" in t else ("wrist" if ("wrist" in t or "hand" in t) else None)
    kind = "depth" if "depth" in t else ("rgb" if ("rgb" in t or "color" in t or "colour" in t) else None)
    return cam, kind


def load_file(path):
    p = path if os.path.isabs(path) else os.path.join(WS, path)
    if not os.path.exists(p):
        return None
    try:
        im = Image.open(p)
        im.load()
        return im
    except Exception:
        return None


def _walk_paths(obj, path=(), depth=0, out=None):
    """(keypath, value) for every leaf string/list value."""
    if out is None:
        out = []
    if depth > 10 or len(out) > 5000:
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            _walk_paths(v, path + (str(k),), depth + 1, out)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _walk_paths(v, path + (str(i),), depth + 1, out)
    else:
        out.append((path, obj))
    return out


def _cam_kind_from(text):
    t = text.lower()
    cam = "front" if "front" in t else ("wrist" if ("wrist" in t or "hand" in t) else None)
    kind = "depth" if "depth" in t else "rgb"
    return cam, kind


def file_entries(meta, skip_sequences=True):
    """Camera image files or inline base64 images referenced anywhere in the metadata."""
    out = []
    # 1) dicts that carry camera / kind / path explicitly
    for node in walk(meta):
        if isinstance(node, dict):
            path = node.get("path") or node.get("file") or node.get("uri")
            if isinstance(path, str) and re.search(r"\.(png|jpg|jpeg|npy)$", path, re.I):
                label = " ".join(str(node.get(k, "")) for k in ("camera", "camera_name", "name", "kind", "type")) + " " + path
                cam, kind = _cam_kind_from(label)
                if cam:
                    out.append({"camera": cam, "kind": kind, "path": path, "src": "dict",
                                "mpu": node.get("meters_per_unit") or node.get("depth_scale")})
    # 2) any string leaf that looks like an image path / base64 image, labelled by its key path
    for kp, v in _walk_paths(meta):
        if not isinstance(v, str):
            continue
        keytxt = "/".join(kp)
        if skip_sequences and ("frames" in kp or "frame_sequence" in kp):
            continue
        if re.search(r"\.(png|npy)$", v, re.I):
            cam, kind = _cam_kind_from(keytxt + " " + v)
            if cam:
                out.append({"camera": cam, "kind": kind, "path": v, "src": "leaf", "mpu": None})
        elif len(v) > 2000 and re.match(r"^[A-Za-z0-9+/=\s]+$", v[:200]):
            cam, kind = _cam_kind_from(keytxt)
            if cam:
                out.append({"camera": cam, "kind": kind, "b64": v, "src": "b64", "mpu": None})
    return out


def _load_entry(e):
    try:
        if "b64" in e:
            im = _decode_image_bytes(base64.b64decode(e["b64"]))
            return _pil_to_array(im)
        p = e["path"] if os.path.isabs(e["path"]) else os.path.join(WS, e["path"])
        if not os.path.exists(p):
            return None
        if p.endswith(".npy"):
            a = np.load(p)
            if a.ndim == 2:
                return "depth16" if a.dtype != np.uint8 else "gray", a.astype(np.float64)
            return "rgb", a[..., :3].astype(np.uint8)
        im = Image.open(p)
        im.load()
        return _pil_to_array(im)
    except Exception:
        return None


class Obs:
    """One parsed observation."""

    def __init__(self):
        self.meta = {}
        self.step_id = None
        self.state = None
        self.rgb = {}
        self.depth = {}  # metres, 0 = invalid
        self.ended = False
        self.success = None
        self.instruction = None
        self.notes = []
        self.limits = None


def _to_metres(arr, mpu):
    a = np.asarray(arr, dtype=np.float64)
    if mpu:
        try:
            return a * float(mpu)
        except Exception:
            pass
    # uint16 millimetres expected; detect float metres
    if a.max() < 50:
        return a
    return a * 0.001


def parse_obs(reply, need_depth=True):
    o = Obs()
    meta = first_json(reply)
    o.meta = meta
    # --- scalar fields
    sid = find_key(meta, ["step_id", "step", "elapsed_steps"])
    try:
        o.step_id = int(sid) if sid is not None and not isinstance(sid, (dict, list)) else None
    except Exception:
        o.step_id = None
    st = find_key(meta, ["state", "proprio", "robot_state"])
    if isinstance(st, dict):
        st = find_key(st, ["values", "state", "vector"])
    try:
        st = np.asarray(st, dtype=np.float64).reshape(-1)
        if st.size >= 8:
            o.state = st[:8]
    except Exception:
        pass
    ins = find_key(meta, ["instruction", "language_instruction", "goal", "task_instruction"])
    if isinstance(ins, str):
        o.instruction = ins
    for key in ["episode_ended", "done", "terminated", "terminal"]:
        v = find_key(meta, [key])
        if v is True:
            o.ended = True
    succ = find_key(meta, ["success", "task_complete", "is_success"])
    if isinstance(succ, bool):
        o.success = succ
        if succ:
            o.ended = True
    o.limits = find_key(meta, ["episode_limits", "limits"])
    # --- inline images
    blocks = reply.get("content", []) or []
    imgs = []
    last_text = ""
    for b in blocks:
        try:
            if b.get("type") == "text":
                last_text = b.get("text", "")
                if len(last_text) > 300:
                    last_text = ""
            elif b.get("type") == "image":
                raw = base64.b64decode(b.get("data", ""))
                im = _decode_image_bytes(raw)
                kind, arr = _pil_to_array(im)
                cam, k2 = _label_from(last_text)
                ann = b.get("annotations") or b.get("_meta") or {}
                if isinstance(ann, dict):
                    c3, k3 = _label_from(json.dumps(ann))
                    cam = cam or c3
                    k2 = k2 or k3
                imgs.append({"kind": kind, "arr": arr, "cam": cam, "k2": k2})
                last_text = ""
        except Exception as e:  # noqa
            o.notes.append("imgdecode:" + repr(e)[:80])
    # attachment descriptors in meta, in order, may label the image blocks
    desc = None
    for node in walk(meta):
        if isinstance(node, list) and node and all(isinstance(x, dict) for x in node):
            if all(("camera" in x) for x in node) and len(node) == len(imgs):
                desc = node
                break
    if desc is not None:
        for d, im in zip(desc, imgs):
            im["cam"] = im["cam"] or str(d.get("camera", "")).lower() or None
            k = str(d.get("kind", d.get("type", ""))).lower()
            if "depth" in k:
                im["k2"] = "depth"
            elif k:
                im["k2"] = im["k2"] or "rgb"
    rgbs = [im for im in imgs if im["kind"] == "rgb" and im["k2"] != "depth"]
    deps = [im for im in imgs if im["kind"] in ("depth16", "gray") or im["k2"] == "depth"]
    # assign rgb cameras
    lab = [im for im in rgbs if im["cam"] in ("front", "wrist")]
    if len(lab) >= 2 and len({im["cam"] for im in lab}) == 2:
        for im in lab:
            o.rgb.setdefault(im["cam"], im["arr"])
    elif len(rgbs) >= 2:
        a, b = rgbs[0]["arr"], rgbs[1]["arr"]
        da, db = _is_wrist_rgb(a), _is_wrist_rgb(b)
        if db + 0.05 < da:
            o.rgb["wrist"], o.rgb["front"] = a, b
            o.notes.append("rgb order swapped by finger heuristic")
        else:
            o.rgb["front"], o.rgb["wrist"] = a, b
    elif len(rgbs) == 1:
        o.rgb["front"] = rgbs[0]["arr"]
    # depth inline
    for im in deps:
        if im["cam"] in ("front", "wrist") and im["cam"] not in o.depth:
            if im["kind"] == "depth16":
                o.depth[im["cam"]] = _to_metres(im["arr"], None)
    unl = [im for im in deps if im["cam"] not in ("front", "wrist") and im["kind"] == "depth16"]
    if unl and len(o.depth) < 2:
        order = [c for c in ("front", "wrist") if c not in o.depth]
        for c, im in zip(order, unl):
            o.depth[c] = _to_metres(im["arr"], None)
    # --- files / base64 referenced in meta (current observation)
    if need_depth and (len(o.depth) < 2 or len(o.rgb) < 2):
        for e in file_entries(meta):
            cam = e["camera"]
            if e["kind"] == "depth" and cam in o.depth:
                continue
            if e["kind"] != "depth" and cam in o.rgb:
                continue
            r = _load_entry(e)
            if r is None:
                continue
            k, arr = r
            if e["kind"] == "depth" and k in ("depth16",):
                o.depth[cam] = _to_metres(arr, e.get("mpu"))
                o.notes.append(f"depth {cam} from {e['src']}")
            elif e["kind"] != "depth" and k == "rgb":
                o.rgb[cam] = arr
    # --- frame-sequence manifest: last execution frame
    if need_depth and len(o.depth) < 2:
        mp = find_key(meta, ["manifest_path"])
        if isinstance(mp, str):
            try:
                p = mp if os.path.isabs(mp) else os.path.join(WS, mp)
                man = json.load(open(p))
                frames = man.get("frames", [])
                if frames:
                    fr = sorted(frames, key=lambda f: f.get("frame_index", 0))[-1]
                    fsid = fr.get("step_id")
                    if o.step_id is None or fsid is None or int(fsid) == int(o.step_id):
                        for e in fr.get("files", []):
                            cam = str(e.get("camera", "")).lower()
                            if e.get("kind") == "depth" and cam in ("front", "wrist") and cam not in o.depth:
                                im = load_file(e["path"])
                                if im is not None:
                                    k, arr = _pil_to_array(im)
                                    if k == "depth16":
                                        o.depth[cam] = _to_metres(arr, e.get("meters_per_unit"))
            except Exception as e:  # noqa
                o.notes.append("manifest:" + repr(e)[:80])
    # sanity on depth pairing: wrist depth has near fingers at bottom corners
    if "front" in o.depth and "wrist" in o.depth:
        try:
            def corner_ratio(d):
                h, w = d.shape
                c = np.concatenate([d[int(h * .75):, :int(w * .2)].ravel(), d[int(h * .75):, int(w * .8):].ravel()])
                c = c[c > 0]
                m = d[int(h * .3):int(h * .6), int(w * .3):int(w * .7)]
                m = m[m > 0]
                if c.size == 0 or m.size == 0:
                    return 1.0
                return float(np.median(c) / np.median(m))
            if corner_ratio(o.depth["front"]) + 0.2 < corner_ratio(o.depth["wrist"]):
                o.depth["front"], o.depth["wrist"] = o.depth["wrist"], o.depth["front"]
                o.notes.append("depth swapped by finger heuristic")
        except Exception:
            pass
    return o


def parse_points(reply):
    """robomme_pixel_to_position reply -> list of (xyz or None)."""
    meta = first_json(reply)
    res = None
    for key in ["positions", "points", "results", "selections", "world_points"]:
        v = find_key(meta, [key])
        if isinstance(v, list):
            res = v
            break
    if res is None and isinstance(meta, list):
        res = meta
    out = []
    if res is None:
        return out
    for r in res:
        p = None
        if isinstance(r, dict):
            if r.get("valid") is False:
                out.append(None)
                continue
            for k in ["position", "world", "world_position", "point", "xyz", "position_m", "world_point"]:
                v = r.get(k)
                if isinstance(v, dict):
                    v = [v.get("x"), v.get("y"), v.get("z")]
                if isinstance(v, (list, tuple)) and len(v) == 3:
                    try:
                        p = np.array([float(x) for x in v])
                        break
                    except Exception:
                        p = None
            if p is None and all(k in r for k in ("x", "y", "z")):
                try:
                    p = np.array([float(r["x"]), float(r["y"]), float(r["z"])])
                except Exception:
                    p = None
        elif isinstance(r, (list, tuple)) and len(r) == 3:
            try:
                p = np.array([float(x) for x in r])
            except Exception:
                p = None
        out.append(p)
    return out
