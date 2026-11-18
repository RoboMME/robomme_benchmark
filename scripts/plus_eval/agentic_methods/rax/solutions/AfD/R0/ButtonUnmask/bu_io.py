"""Defensive parsing of RAX / RoboMME-Plus MCP replies.

No live reply sample was available during development (AfD, zero simulator
starts), so every accessor searches the JSON tree for the documented keys and
tolerates several plausible layouts.
"""
import base64
import io
import json
import os
import time

import numpy as np
from PIL import Image

WORKSPACE = "/workspace"

LOG_PATH = None
_T0 = time.time()


def set_log(path):
    global LOG_PATH
    LOG_PATH = path
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except Exception:
        pass


def log(*parts):
    msg = "[%7.2fs] " % (time.time() - _T0) + " ".join(str(p) for p in parts)
    try:
        print(msg, flush=True)
    except Exception:
        pass
    if LOG_PATH:
        try:
            with open(LOG_PATH, "a") as f:
                f.write(msg + "\n")
        except Exception:
            pass


# ----------------------------------------------------------------- JSON search
def walk(obj, path=()):
    """Yield (path, key, value) for every dict entry in a JSON tree."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield path, k, v
            yield from walk(v, path + (k,))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk(v, path + (i,))


def find_all(obj, key):
    return [(p, v) for p, k, v in walk(obj) if k == key]


def _is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _prefer(cands, prefer_words=("observation", "obs", "final", "current")):
    if not cands:
        return None
    def score(c):
        p = [str(x).lower() for x in c[0]]
        has = any(any(w in s for w in prefer_words) for s in p)
        in_steps = any(s in ("steps", "transition", "frames", "history") for s in p)
        return (0 if has else 1, 1 if in_steps else 0, len(p))
    return sorted(cands, key=score)[0][1]


def find_state(meta):
    cands = [(p, v) for p, v in find_all(meta, "state")
             if isinstance(v, list) and len(v) == 8 and all(_is_num(x) for x in v)]
    if not cands:
        # state might be a dict with ee position etc.
        for p, v in find_all(meta, "state"):
            if isinstance(v, dict):
                vals = []
                for k in ("ee_pos", "position", "ee_position", "tcp_pos"):
                    if k in v:
                        vals = list(v[k])
                        break
                rot = None
                for k in ("ee_axis_angle", "orientation", "axis_angle", "rotation"):
                    if k in v:
                        rot = list(v[k])
                        break
                fing = None
                for k in ("fingers", "gripper", "finger_positions", "gripper_qpos"):
                    if k in v:
                        fing = list(v[k])
                        break
                if len(vals) == 3 and rot and len(rot) == 3:
                    fing = fing if fing and len(fing) == 2 else [0.0, 0.0]
                    return [float(x) for x in vals + rot + fing]
        return None
    v = _prefer(cands)
    return [float(x) for x in v]


def find_int(meta, key):
    cands = [(p, v) for p, v in find_all(meta, key) if _is_num(v)]
    v = _prefer(cands)
    return int(v) if v is not None else None


def find_bool_any(meta, keys):
    for k in keys:
        for p, v in find_all(meta, k):
            if v is True:
                return True
    return False


def find_str(meta, key):
    for p, v in find_all(meta, key):
        if isinstance(v, str) and v.strip():
            return v
    return None


def find_manifest_path(meta):
    for p, v in find_all(meta, "manifest_path"):
        if isinstance(v, str) and v:
            return v
    return None


def remaining_steps(meta):
    """Best-effort remaining-step count from episode_limits."""
    for p, v in find_all(meta, "episode_limits"):
        if isinstance(v, dict):
            for k, x in v.items():
                if "remaining" in k.lower() and _is_num(x):
                    return int(x)
            mx = None
            used = None
            for k, x in v.items():
                kl = k.lower()
                if _is_num(x) and ("max" in kl or "limit" in kl or "horizon" in kl):
                    mx = int(x)
                if _is_num(x) and ("used" in kl or "elapsed" in kl or kl in ("step", "steps", "step_id")):
                    used = int(x)
            if mx is not None and used is not None:
                return mx - used
    return None


def max_steps(meta):
    for p, v in find_all(meta, "episode_limits"):
        if isinstance(v, dict):
            for k, x in v.items():
                kl = k.lower()
                if _is_num(x) and ("max" in kl or "limit" in kl or "horizon" in kl) and "remain" not in kl:
                    return int(x)
        if _is_num(v):
            return int(v)
    for k in ("max_steps", "step_limit", "max_episode_steps"):
        x = find_int(meta, k)
        if x:
            return x
    return None


# ----------------------------------------------------------------- images
def _decode_image(b):
    data = b.get("data") or b.get("blob")
    if not data:
        return None
    try:
        raw = base64.b64decode(data)
        im = Image.open(io.BytesIO(raw))
        im.load()
        return im
    except Exception:
        return None


def _img_to_arrays(im):
    """Return (kind_guess, array)."""
    mode = im.mode
    if mode in ("I;16", "I;16B", "I;16L", "I;16N", "I", "F"):
        arr = np.array(im)
        return "depth", arr.astype(np.float64)
    if mode in ("L", "LA", "P") and False:
        return "gray", np.array(im.convert("L"))
    return "rgb", np.array(im.convert("RGB"))


def _label_from_text(s):
    s = (s or "").lower()
    cam = "front" if "front" in s else ("wrist" if ("wrist" in s or "hand" in s) else None)
    kind = "depth" if "depth" in s else ("rgb" if ("rgb" in s or "color" in s or "colour" in s) else None)
    return cam, kind


def load_file_image(path):
    p = path if os.path.isabs(path) else os.path.join(WORKSPACE, path)
    im = Image.open(p)
    im.load()
    return im


class Obs:
    """One parsed observation: rgb/depth per camera, state, step id, meta."""

    def __init__(self):
        self.meta = {}
        self.rgb = {}
        self.depth = {}
        self.depth_scale = {}
        self.state = None
        self.step_id = None
        self.instruction = None
        self.ended = False
        self.complete = False
        self.manifest_path = None
        self.raw_images = []  # list of (label_cam, label_kind, kind_guess, array)
        self.source = ""

    def ok(self):
        return self.state is not None


def parse_result(res, learned_order=None):
    """Parse an MCP tools/call result into an Obs (images may be missing)."""
    ob = Obs()
    if not isinstance(res, dict):
        return ob
    content = res.get("content") or []
    meta = {}
    sc = res.get("structuredContent")
    if isinstance(sc, dict):
        meta = sc
    prev_text = None
    imgs = []
    for b in content:
        if not isinstance(b, dict):
            continue
        t = b.get("type")
        if t == "text":
            txt = b.get("text", "")
            try:
                j = json.loads(txt)
                if isinstance(j, dict):
                    if not meta:
                        meta = j
                    else:
                        meta.setdefault("_extra_text", []).append(j)
                prev_text = None
            except Exception:
                prev_text = txt
        elif t == "image" or (t == "resource" and isinstance(b.get("resource"), dict)):
            bb = b if t == "image" else b["resource"]
            im = _decode_image(bb)
            label = " ".join(str(x) for x in [prev_text or "", json.dumps(b.get("annotations") or {}),
                                               bb.get("uri", ""), bb.get("name", ""), b.get("name", "")])
            if im is not None:
                imgs.append((label, im))
            prev_text = None
    ob.meta = meta
    ob.state = find_state(meta)
    ob.step_id = find_int(meta, "step_id")
    ob.instruction = find_str(meta, "instruction")
    ob.ended = find_bool_any(meta, ["episode_ended", "terminated", "truncated", "done"])
    ob.complete = find_bool_any(meta, ["task_complete", "success"])
    ob.manifest_path = find_manifest_path(meta)

    # descriptor lists in meta that may label the attachments in order
    desc = None
    for p, k, v in walk(meta):
        if isinstance(v, list) and v and all(isinstance(x, dict) and "camera" in x for x in v):
            if len(v) == len(imgs):
                desc = v
                break
    decoded = []
    for i, (label, im) in enumerate(imgs):
        kind_guess, arr = _img_to_arrays(im)
        cam, kind = _label_from_text(label)
        if desc is not None:
            d = desc[i]
            cam = str(d.get("camera", cam or "")).lower() or cam
            kk = str(d.get("kind", d.get("type", d.get("modality", "")))).lower()
            if "depth" in kk:
                kind = "depth"
            elif kk:
                kind = "rgb" if kind_guess == "rgb" else kind_guess
        if kind is None:
            kind = kind_guess
        if kind == "depth" and kind_guess == "rgb":
            # depth delivered as an 8-bit/RGB visualisation: unusable for metric depth
            kind = "depth_vis"
        decoded.append([cam, kind, arr])
        ob.raw_images.append((cam, kind, kind_guess, arr.shape))

    # assign unlabeled by order: first of a kind = front, second = wrist
    for kind in ("rgb", "depth"):
        group = [d for d in decoded if d[1] == kind]
        unl = [d for d in group if d[0] not in ("front", "wrist")]
        if unl:
            used = {d[0] for d in group if d[0] in ("front", "wrist")}
            order = ["front", "wrist"]
            if learned_order and learned_order.get(kind) == "swapped":
                order = ["wrist", "front"]
            for d in unl:
                for c in order:
                    if c not in used:
                        d[0] = c
                        used.add(c)
                        break
    for cam, kind, arr in decoded:
        if cam not in ("front", "wrist"):
            continue
        if kind == "rgb" and cam not in ob.rgb:
            ob.rgb[cam] = arr
        elif kind == "depth" and cam not in ob.depth:
            ob.depth[cam] = arr
    return ob


def load_manifest_frames(manifest_path):
    p = manifest_path if os.path.isabs(manifest_path) else os.path.join(WORKSPACE, manifest_path)
    with open(p) as f:
        man = json.load(f)
    frames = man.get("frames") if isinstance(man, dict) else None
    return man, (frames or [])


def frame_files(frame):
    """{(camera, kind): (path, meters_per_unit)} for one manifest frame."""
    out = {}
    for f in frame.get("files", []) or []:
        if not isinstance(f, dict):
            continue
        cam = str(f.get("camera", "")).lower()
        kind = str(f.get("kind", "")).lower()
        path = f.get("path")
        if cam and kind and path:
            out[(cam, kind)] = (path, f.get("meters_per_unit"))
    return out


def fill_from_manifest(ob, want_step=None):
    """Fill missing rgb/depth of `ob` from the last execution frame of its manifest."""
    if not ob.manifest_path:
        return False
    try:
        man, frames = load_manifest_frames(ob.manifest_path)
    except Exception as e:
        log("manifest read failed", ob.manifest_path, repr(e))
        return False
    if not frames:
        return False
    def key(fr):
        return (fr.get("frame_index", -1) if _is_num(fr.get("frame_index", None)) else -1)
    cand = [fr for fr in frames if str(fr.get("phase", "execution")) != "video_demo"] or frames
    cand = sorted(cand, key=key)
    fr = cand[-1]
    if want_step is not None and _is_num(fr.get("step_id")) and int(fr["step_id"]) != int(want_step):
        # saved frame is not the current one; do not mix
        return False
    files = frame_files(fr)
    got = False
    for (cam, kind), (path, mpu) in files.items():
        if cam not in ("front", "wrist"):
            continue
        try:
            im = load_file_image(path)
        except Exception:
            continue
        kg, arr = _img_to_arrays(im)
        if kind == "rgb" and cam not in ob.rgb and kg == "rgb":
            ob.rgb[cam] = arr
            got = True
        elif kind == "depth" and cam not in ob.depth and kg == "depth":
            ob.depth[cam] = arr
            ob.depth_scale[cam] = mpu
            got = True
    return got


def parse_points(res, n):
    """Parse robomme_pixel_to_position output -> list of (xyz or None, spread or None)."""
    out = [(None, None)] * n
    if not isinstance(res, dict):
        return out
    meta = None
    sc = res.get("structuredContent")
    if isinstance(sc, dict):
        meta = sc
    if meta is None:
        for b in res.get("content") or []:
            if isinstance(b, dict) and b.get("type") == "text":
                try:
                    j = json.loads(b.get("text", ""))
                    meta = j
                    break
                except Exception:
                    continue
    if meta is None:
        return out
    lists = []
    if isinstance(meta, list):
        lists.append(meta)
    for p, k, v in walk(meta):
        if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
            lists.append(v)
    lst = None
    for l in lists:
        if len(l) == n:
            lst = l
            break
    if lst is None:
        return out
    res_list = []
    for d in lst:
        xyz = None
        for k in ("position", "world_position", "position_m", "world", "point", "xyz", "world_point",
                  "position_world", "world_xyz", "point_world"):
            v = d.get(k)
            if isinstance(v, list) and len(v) == 3 and all(_is_num(x) for x in v):
                xyz = [float(x) for x in v]
                break
            if isinstance(v, dict) and all(c in v for c in "xyz"):
                xyz = [float(v["x"]), float(v["y"]), float(v["z"])]
                break
        if xyz is None:
            for p, k, v in walk(d):
                if isinstance(v, list) and len(v) == 3 and all(_is_num(x) for x in v) and \
                        "pixel" not in str(k).lower() and "uv" not in str(k).lower():
                    xyz = [float(x) for x in v]
                    break
        valid = d.get("valid", True)
        if valid is False:
            xyz = None
        spread = d.get("depth_spread_m")
        res_list.append((xyz, spread if _is_num(spread) else None))
    return res_list
