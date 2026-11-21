"""Defensive parsing of RoboMME-Plus MCP replies (RAX adapter).

Nothing here assumes one exact reply layout: keys are searched recursively and every
image source (manifest files labelled by camera, inline image blocks) is tried.
"""
import base64
import io
import json
import os
import time
from pathlib import Path

import numpy as np

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None

WORKSPACE = Path("/workspace")


def log(*a):
    try:
        print("[%7.1fs] " % (time.time() - T0), *a, flush=True)
    except Exception:
        pass


T0 = time.time()


# ----------------------------------------------------------------------------- search
def find_key(obj, key, depth=0, maxdepth=8):
    """First value stored under `key` anywhere in obj (breadth-first)."""
    if depth > maxdepth:
        return None
    queue = [obj]
    seen = 0
    while queue and seen < 20000:
        cur = queue.pop(0)
        seen += 1
        if isinstance(cur, dict):
            if key in cur:
                return cur[key]
            queue.extend(cur.values())
        elif isinstance(cur, list):
            queue.extend(cur[:200])
    return None


def as_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes")
    return False


# ----------------------------------------------------------------------------- reply
class Reply:
    def __init__(self, raw):
        self.raw = raw
        self.meta = {}
        self.texts = []
        self.images = []  # list of (mime, bytes)
        content = []
        if isinstance(raw, dict):
            content = raw.get("content") or []
            sc = raw.get("structuredContent")
            if isinstance(sc, dict):
                self.meta.update(sc)
        for block in content if isinstance(content, list) else []:
            if not isinstance(block, dict):
                continue
            t = block.get("type")
            if t == "text":
                txt = block.get("text", "")
                self.texts.append(txt)
                try:
                    j = json.loads(txt)
                    if isinstance(j, dict):
                        for k, v in j.items():
                            self.meta.setdefault(k, v)
                    elif isinstance(j, list):
                        self.meta.setdefault("_list", j)
                except Exception:
                    pass
            elif t == "image":
                data = block.get("data")
                if isinstance(data, str):
                    try:
                        self.images.append((block.get("mimeType", ""), base64.b64decode(data)))
                    except Exception:
                        pass

    def get(self, key):
        return find_key(self.meta, key)


def call(ctx, tool, **kw):
    raw = ctx.call(tool, **kw)
    return Reply(raw)


# ----------------------------------------------------------------------------- images
def decode_image(b):
    if Image is None:
        return None
    try:
        im = Image.open(io.BytesIO(b))
        im.load()
        return im
    except Exception:
        return None


def load_path(p):
    """Open a published path (relative to /workspace) or an absolute one."""
    if not p:
        return None
    cands = [Path(p)]
    if not os.path.isabs(p):
        cands = [WORKSPACE / p, Path.cwd() / p, Path(p)]
    for c in cands:
        try:
            if c.is_file():
                im = Image.open(c)
                im.load()
                return im
        except Exception:
            continue
    return None


def rgb_array(im):
    if im is None:
        return None
    try:
        a = np.asarray(im.convert("RGB"), dtype=np.uint8)
        return a
    except Exception:
        return None


def depth_array(im, mpu=0.001):
    if im is None:
        return None
    try:
        a = np.asarray(im)
        if a.ndim == 3:
            return None  # an RGB visualisation, not metric depth
        a = a.astype(np.float64)
        if mpu is None:
            mpu = 0.001
        d = a * float(mpu)
        d[a <= 0] = np.nan
        return d
    except Exception:
        return None


# ----------------------------------------------------------------------------- calibration
def _to_K(v):
    try:
        if isinstance(v, dict):
            low = {str(k).lower(): x for k, x in v.items()}
            if all(k in low for k in ("fx", "fy", "cx", "cy")):
                return np.array([[float(low["fx"]), 0, float(low["cx"])],
                                 [0, float(low["fy"]), float(low["cy"])], [0, 0, 1.0]])
            for k in ("k", "matrix", "intrinsic", "intrinsics", "intrinsic_cv", "camera_matrix"):
                if k in low:
                    r = _to_K(low[k])
                    if r is not None:
                        return r
            return None
        a = np.array(v, dtype=np.float64)
        if a.size == 9:
            a = a.reshape(3, 3)
            if abs(a[2, 2] - 1) > 1e-6 and abs(a[2, 2]) > 1e-9:
                a = a / a[2, 2]
            if a[0, 2] == 0 and a[2, 0] != 0:  # transposed
                a = a.T
            return a
        if a.size == 4:
            fx, fy, cx, cy = a.ravel()
            return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    except Exception:
        return None
    return None


def _to_pose(v):
    """Return dict(position=3, quat=4 (as given), matrix=4x4 or None, order hint)."""
    try:
        if isinstance(v, dict):
            low = {str(k).lower(): x for k, x in v.items()}
            pos = None
            for k in ('position_m', "position", "p", "pos", "translation", "t", "xyz"):
                if k in low:
                    pos = np.array(low[k], dtype=np.float64).ravel()[:3]
                    break
            quat, order = None, None
            for k, o in (("quaternion_wxyz", "wxyz"), ("wxyz", "wxyz"), ("quat_wxyz", "wxyz"),
                         ("quaternion_xyzw", "xyzw"), ("xyzw", "xyzw"), ("quat_xyzw", "xyzw"),
                         ("quaternion", None), ("quat", None), ("q", None), ("orientation", None),
                         ("rotation", None)):
                if k in low:
                    arr = np.array(low[k], dtype=np.float64).ravel()
                    if arr.size == 4:
                        quat, order = arr, o
                        break
                    if arr.size == 9:
                        m = np.eye(4)
                        m[:3, :3] = arr.reshape(3, 3)
                        if pos is not None:
                            m[:3, 3] = pos
                        return {"matrix": m, "position": pos, "quat": None, "order": None,
                                "convention": str(low.get("convention", ""))}
            for k in ("matrix", "pose", "transform", "cam2world", "camera_to_world", "extrinsic",
                      "model_matrix", "cam2world_gl"):
                if k in low:
                    arr = np.array(low[k], dtype=np.float64)
                    if arr.size == 16:
                        return {"matrix": arr.reshape(4, 4), "position": None, "quat": None,
                                "order": None, "convention": str(low.get("convention", ""))}
            if pos is not None and quat is not None:
                return {"matrix": None, "position": pos, "quat": quat, "order": order,
                        "convention": str(low.get("convention", ""))}
            return None
        arr = np.array(v, dtype=np.float64)
        if arr.size == 16:
            return {"matrix": arr.reshape(4, 4), "position": None, "quat": None, "order": None,
                    "convention": ""}
        if arr.size == 7:
            return {"matrix": None, "position": arr[:3], "quat": arr[3:], "order": None,
                    "convention": ""}
    except Exception:
        return None
    return None


def _per_camera(v, conv):
    out = {}
    if isinstance(v, dict):
        for k, x in v.items():
            name = str(k).lower()
            cam = "front" if "front" in name or "base" in name or "third" in name else (
                "wrist" if "wrist" in name or "hand" in name else None)
            if cam is None:
                continue
            r = conv(x)
            if r is not None:
                out[cam] = r
    elif isinstance(v, list):
        for x in v:
            if isinstance(x, dict) and "camera" in x:
                name = str(x["camera"]).lower()
                cam = "front" if "front" in name else ("wrist" if "wrist" in name else None)
                if cam:
                    r = conv(x)
                    if r is not None:
                        out[cam] = r
    return out


# ----------------------------------------------------------------------------- observation
class Obs:
    def __init__(self):
        self.step_id = None
        self.state = None
        self.instruction = ""
        self.ended = False
        self.complete = False
        self.rgb = {}
        self.depth = {}
        self.K = {}
        self.pose = {}
        self.remaining = None
        self.manifest = None
        self.reply = None

    @property
    def pos(self):
        return np.array(self.state[:3], dtype=np.float64)

    @property
    def rotvec(self):
        return np.array(self.state[3:6], dtype=np.float64)

    @property
    def grip(self):
        return float(self.state[6] + self.state[7])


def _num_list(v, n):
    try:
        a = np.array(v, dtype=np.float64).ravel()
        if a.size == n and np.all(np.isfinite(a)):
            return a
    except Exception:
        pass
    return None


def _remaining(meta):
    lim = find_key(meta, "episode_limits")
    rem = None
    if isinstance(lim, dict):
        for k in ("remaining_steps", "steps_remaining", "remaining", "remaining_control_steps"):
            if k in lim:
                try:
                    rem = int(lim[k])
                    break
                except Exception:
                    pass
        if rem is None:
            mx = None
            for k in ("max_steps", "step_limit", "max_episode_steps", "limit", "horizon"):
                if k in lim:
                    try:
                        mx = int(lim[k])
                        break
                    except Exception:
                        pass
            used = None
            for k in ("steps_used", "elapsed_steps", "step", "step_id", "used"):
                if k in lim:
                    try:
                        used = int(lim[k])
                        break
                    except Exception:
                        pass
            if mx is not None:
                if used is None:
                    used = find_key(meta, "step_id")
                try:
                    rem = mx - int(used or 0)
                except Exception:
                    rem = None
    return rem


def parse_obs(reply, want_images=True):
    o = Obs()
    o.reply = reply
    m = reply.meta
    st = find_key(m, "state")
    if isinstance(st, dict):
        for k in ("vector", "values", "state"):
            if k in st:
                st = st[k]
                break
    o.state = _num_list(st, 8)
    sid = find_key(m, "step_id")
    try:
        o.step_id = int(sid)
    except Exception:
        o.step_id = None
    ins = find_key(m, "instruction")
    if isinstance(ins, str):
        o.instruction = ins
    o.ended = as_bool(find_key(m, "episode_ended"))
    tc = find_key(m, "task_complete")
    if tc is None:
        tc = find_key(m, "success")
    o.complete = as_bool(tc)
    o.remaining = _remaining(m)
    o.K = _per_camera(find_key(m, "camera_intrinsics") or {}, _to_K)
    o.pose = _per_camera(find_key(m, "camera_poses") or {}, _to_pose)
    if want_images:
        load_images(o)
    return o


def _frame_files(frame):
    files = {}
    for f in frame.get("files", []) or []:
        try:
            cam = str(f.get("camera", "")).lower()
            kind = str(f.get("kind", "")).lower()
            files[(cam, kind)] = (f.get("path"), f.get("meters_per_unit", 0.001))
        except Exception:
            pass
    return files


def read_manifest(reply):
    fs = reply.get("frame_sequence")
    mp = None
    if isinstance(fs, dict):
        mp = fs.get("manifest_path") or find_key(fs, "manifest_path")
    if mp is None:
        mp = reply.get("manifest_path")
    if not isinstance(mp, str):
        return None, None
    for c in (WORKSPACE / mp, Path(mp)):
        try:
            if c.is_file():
                return json.loads(c.read_text()), c
        except Exception:
            continue
    return None, None


def load_images(o):
    reply = o.reply
    # 1) camera-labelled files from the frame manifest, if its last frame is this step
    try:
        man, mpath = read_manifest(reply)
        if isinstance(man, dict):
            o.manifest = man
            frames = [f for f in man.get("frames", []) if isinstance(f, dict)]
            ex = [f for f in frames if f.get("phase", "execution") != "video_demo"]
            if ex:
                ex.sort(key=lambda f: (f.get("frame_index", 0)))
                last = ex[-1]
                ok = True
                if o.step_id is not None and last.get("step_id") is not None:
                    ok = int(last.get("step_id")) == int(o.step_id)
                if ok:
                    files = _frame_files(last)
                    for cam in ("front", "wrist"):
                        if (cam, "rgb") in files:
                            a = rgb_array(load_path(files[(cam, "rgb")][0]))
                            if a is not None:
                                o.rgb[cam] = a
                        if (cam, "depth") in files:
                            p, mpu = files[(cam, "depth")]
                            d = depth_array(load_path(p), mpu)
                            if d is not None:
                                o.depth[cam] = d
    except Exception as e:
        log("manifest read failed:", repr(e))
    # 2) other labelled path hints in the metadata
    try:
        for cam in ("front", "wrist"):
            if cam in o.rgb and cam in o.depth:
                continue
            for key in ("images", "attachments", "files", "cameras"):
                v = find_key(reply.meta, key)
                if isinstance(v, list):
                    for f in v:
                        if not isinstance(f, dict) or "path" not in f:
                            continue
                        c = str(f.get("camera", "")).lower()
                        k = str(f.get("kind", "rgb")).lower()
                        if c != cam:
                            continue
                        if k == "rgb" and cam not in o.rgb:
                            a = rgb_array(load_path(f["path"]))
                            if a is not None:
                                o.rgb[cam] = a
                        if k == "depth" and cam not in o.depth:
                            d = depth_array(load_path(f["path"]), f.get("meters_per_unit", 0.001))
                            if d is not None:
                                o.depth[cam] = d
    except Exception as e:
        log("path hints failed:", repr(e))
    # 3) inline image blocks: RGB first front then wrist; 16-bit images are depth
    try:
        rgbs, deps = [], []
        for mime, b in reply.images:
            im = decode_image(b)
            if im is None:
                continue
            if im.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
                deps.append(im)
            else:
                rgbs.append(im)
        order = ["front", "wrist"]
        for i, im in enumerate(rgbs[:2]):
            if order[i] not in o.rgb:
                a = rgb_array(im)
                if a is not None:
                    o.rgb[order[i]] = a
        for i, im in enumerate(deps[:2]):
            if order[i] not in o.depth:
                d = depth_array(im, 0.001)
                if d is not None:
                    o.depth[order[i]] = d
    except Exception as e:
        log("inline image decode failed:", repr(e))


def describe(reply):
    """Compact description of a reply's structure for the log."""
    def shape(v, d=0):
        if d > 2:
            return "..."
        if isinstance(v, dict):
            return {k: shape(x, d + 1) for k, x in list(v.items())[:30]}
        if isinstance(v, list):
            return "list[%d]" % len(v)
        if isinstance(v, str):
            return "str[%d]" % len(v)
        return type(v).__name__
    try:
        return json.dumps({"meta": shape(reply.meta), "n_images": len(reply.images),
                           "mimes": [m for m, _ in reply.images]})[:3000]
    except Exception:
        return "?"
