"""Defensive parsing of RAX MCP replies (JSON text blocks + image blocks + file paths).

The exact reply layout was never observed before freezing (AfI, R1: the only simulator
start is the rehearsal), so every accessor searches recursively and degrades gracefully.
"""
import base64
import io
import json
import math
import os
from pathlib import Path

import numpy as np

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None

WORKSPACE = Path("/workspace")


# --------------------------------------------------------------------------- JSON
def reply_json(reply):
    """Merge all JSON text blocks of a reply into one dict (first block wins on conflicts)."""
    out = {}
    texts = []
    for block in (reply or {}).get("content", []) or []:
        if block.get("type") == "text":
            t = block.get("text", "")
            texts.append(t)
            try:
                v = json.loads(t)
            except Exception:
                continue
            if isinstance(v, dict):
                for k, val in v.items():
                    out.setdefault(k, val)
    sc = (reply or {}).get("structuredContent")
    if isinstance(sc, dict):
        for k, val in sc.items():
            out.setdefault(k, val)
    out.setdefault("_texts", texts)
    return out


def walk(obj, path=()):
    """Yield (path, key, value) for every dict entry in a nested structure."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield path, k, v
            yield from walk(v, path + (k,))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk(v, path + (i,))


def find_key(obj, names, pred=None):
    names = set(names)
    best = None
    for path, k, v in walk(obj):
        if k in names and (pred is None or pred(v)):
            if best is None or len(path) < len(best[0]):
                best = (path, v)
    return None if best is None else best[1]


def _is_vec(v, n):
    return isinstance(v, (list, tuple)) and len(v) == n and all(isinstance(x, (int, float)) for x in v)


def get_state(j):
    v = find_key(j, ["state"], lambda v: _is_vec(v, 8))
    if v is None:
        v = find_key(j, ["proprio", "proprioception", "robot_state"], lambda v: _is_vec(v, 8))
    return None if v is None else np.array(v, dtype=float)


def get_bool(j, names):
    v = find_key(j, names, lambda v: isinstance(v, bool))
    return v


def episode_ended(j):
    for n in ["episode_ended", "terminated", "done"]:
        v = get_bool(j, [n])
        if v:
            return True
    return False


def task_complete(j):
    v = get_bool(j, ["task_complete", "success"])
    return bool(v)


def get_step_id(j):
    v = find_key(j, ["step_id"], lambda v: isinstance(v, int))
    return v


def get_instruction(j):
    v = find_key(j, ["instruction", "language_instruction", "goal", "task_instruction"],
                 lambda v: isinstance(v, str) and len(v) > 3)
    return v or ""


# --------------------------------------------------------------------------- images
def _decode_png_bytes(data):
    if Image is None:
        return None
    im = Image.open(io.BytesIO(data))
    im.load()
    return im


def _im_to_array(im):
    mode = im.mode
    if mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        a = np.array(im)
        return "depth", a.astype(np.float64)
    if mode in ("RGB", "RGBA", "P", "L", "LA"):
        a = np.array(im.convert("RGB"))
        return "rgb", a
    a = np.array(im)
    if a.dtype == np.uint16:
        return "depth", a.astype(np.float64)
    return "rgb", np.array(im.convert("RGB"))


def load_image_file(path):
    p = Path(path)
    if not p.is_absolute():
        p = WORKSPACE / p
    if p.suffix == ".npy":
        a = np.load(p)
        return ("depth" if a.ndim == 2 else "rgb"), a
    with Image.open(p) as im:
        im.load()
        return _im_to_array(im)


def _camera_from_text(t):
    t = (t or "").lower()
    cam = None
    if "wrist" in t or "hand" in t:
        cam = "wrist"
    elif "front" in t or "third" in t or "base" in t:
        cam = "front"
    kind = None
    if "depth" in t:
        kind = "depth"
    elif "rgb" in t or "color" in t:
        kind = "rgb"
    return cam, kind


def image_blocks(reply):
    """Return list of dicts {kind, camera, array, label} for each image block."""
    out = []
    last_text = ""
    for block in (reply or {}).get("content", []) or []:
        typ = block.get("type")
        if typ == "text":
            last_text = block.get("text", "")
            continue
        if typ != "image":
            continue
        try:
            data = base64.b64decode(block.get("data", ""))
            im = _decode_png_bytes(data)
            kind, arr = _im_to_array(im)
        except Exception:
            continue
        label_src = []
        for key in ("annotations", "_meta", "meta", "name", "title", "camera", "label"):
            if key in block:
                label_src.append(json.dumps(block[key]) if not isinstance(block[key], str) else block[key])
        if len(last_text) < 400:
            label_src.append(last_text)
        cam, k2 = _camera_from_text(" ".join(label_src))
        out.append({"kind": kind, "camera": cam, "array": arr, "label": " ".join(label_src)[:200],
                    "label_kind": k2})
        last_text = ""
    return out


def file_entries(j):
    """Find dicts with camera/kind/path anywhere in the JSON (frame file lists)."""
    out = []
    for path, k, v in walk(j):
        if isinstance(v, dict) and "path" in v and isinstance(v.get("path"), str):
            cam = v.get("camera")
            kind = v.get("kind")
            if cam is None or kind is None:
                c2, k2 = _camera_from_text(v["path"])
                cam = cam or c2
                kind = kind or k2
            out.append({"camera": cam, "kind": kind, "path": v["path"], "ctx": path,
                        "mpu": v.get("meters_per_unit")})
    return out


def manifest_last_execution_files(j):
    """If the reply has a frame_sequence manifest, return the file entries of its last frame."""
    mp = find_key(j, ["manifest_path"], lambda v: isinstance(v, str))
    if not mp:
        return None, None
    p = Path(mp)
    if not p.is_absolute():
        p = WORKSPACE / p
    try:
        man = json.loads(p.read_text())
    except Exception:
        return None, None
    frames = man.get("frames") or []
    if not frames:
        return None, man
    fr = sorted(frames, key=lambda f: f.get("frame_index", 0))
    exe = [f for f in fr if f.get("phase") == "execution"] or fr
    last = exe[-1]
    files = []
    for f in last.get("files", []) or []:
        files.append({"camera": f.get("camera"), "kind": f.get("kind"), "path": f.get("path"),
                      "mpu": f.get("meters_per_unit")})
    return {"files": files, "frame": last}, man


# --------------------------------------------------------------------------- calibration
def _mat(v):
    try:
        a = np.array(v, dtype=float)
    except Exception:
        return None
    return a


def parse_intrinsics(v):
    """Return (fx, fy, cx, cy) from many possible encodings."""
    if v is None:
        return None
    if isinstance(v, dict):
        lower = {str(k).lower(): val for k, val in v.items()}
        if all(k in lower for k in ("fx", "fy", "cx", "cy")):
            return tuple(float(lower[k]) for k in ("fx", "fy", "cx", "cy"))
        for key in ("k", "matrix", "intrinsic", "intrinsics", "intrinsic_cv", "camera_matrix", "k_matrix"):
            if key in lower:
                r = parse_intrinsics(lower[key])
                if r:
                    return r
        for val in v.values():
            if isinstance(val, (list, dict)):
                r = parse_intrinsics(val)
                if r:
                    return r
        return None
    a = _mat(v)
    if a is None:
        return None
    if a.shape == (3, 3):
        return float(a[0, 0]), float(a[1, 1]), float(a[0, 2]), float(a[1, 2])
    if a.shape == (9,):
        a = a.reshape(3, 3)
        return float(a[0, 0]), float(a[1, 1]), float(a[0, 2]), float(a[1, 2])
    if a.shape == (4,):
        return tuple(float(x) for x in a)
    if a.ndim == 3 and a.shape[-2:] == (3, 3):
        a = a[0]
        return float(a[0, 0]), float(a[1, 1]), float(a[0, 2]), float(a[1, 2])
    return None


def quat_wxyz_to_R(q):
    w, x, y, z = [float(t) for t in q]
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def parse_pose(v):
    """Return (R_world_cam, t_world_cam, convention_str or None)."""
    if v is None:
        return None
    if isinstance(v, dict):
        lower = {str(k).lower(): val for k, val in v.items()}
        conv = None
        for key in ("convention", "frame_convention", "camera_convention"):
            if isinstance(lower.get(key), str):
                conv = lower[key].lower()
        pos = None
        for key in ('position_m', "position", "p", "translation", "pos", "xyz", "t"):
            if key in lower and _is_vec(lower[key], 3):
                pos = np.array(lower[key], dtype=float)
                break
        quat = None
        for key in ("quaternion_wxyz", "quat_wxyz", "wxyz", "quaternion", "q", "orientation", "quat"):
            if key in lower and _is_vec(lower[key], 4):
                quat = lower[key]
                break
        if quat is None:
            for key in ("quaternion_xyzw", "quat_xyzw", "xyzw"):
                if key in lower and _is_vec(lower[key], 4):
                    x, y, z, w = lower[key]
                    quat = [w, x, y, z]
        if pos is not None and quat is not None:
            return quat_wxyz_to_R(quat), pos, conv
        for key in ("matrix", "cam2world", "cam2world_gl", "pose", "extrinsic", "transform", "t_world_cam",
                    "world_from_camera", "model_matrix"):
            if key in lower:
                r = parse_pose(lower[key])
                if r:
                    R, t, c = r
                    if c is None and "gl" in key:
                        c = "opengl"
                    return R, t, conv or c
        for key, val in v.items():
            if isinstance(val, (list, dict)):
                r = parse_pose(val)
                if r:
                    return r[0], r[1], conv or r[2]
        return None
    a = _mat(v)
    if a is None:
        return None
    if a.shape == (4, 4):
        return a[:3, :3], a[:3, 3], None
    if a.shape == (16,):
        a = a.reshape(4, 4)
        return a[:3, :3], a[:3, 3], None
    if a.shape == (7,):
        return quat_wxyz_to_R(a[3:]), a[:3], None
    return None


def camera_calibration(j, cam):
    """Return dict(K=(fx,fy,cx,cy), R, t, conv) for a camera, or None."""
    intr_root = find_key(j, ["camera_intrinsics", "intrinsics", "camera_intrinsic"])
    pose_root = find_key(j, ["camera_poses", "camera_pose", "extrinsics", "camera_extrinsics"])
    K = None
    pose = None
    if isinstance(intr_root, dict):
        sub = None
        for k, v in intr_root.items():
            if cam in str(k).lower():
                sub = v
        if sub is None and cam == "front":
            for k, v in intr_root.items():
                if any(s in str(k).lower() for s in ("base", "third", "main", "agentview")):
                    sub = v
        K = parse_intrinsics(sub) if sub is not None else None
    if isinstance(pose_root, dict):
        sub = None
        for k, v in pose_root.items():
            if cam in str(k).lower():
                sub = v
        if sub is None and cam == "front":
            for k, v in pose_root.items():
                if any(s in str(k).lower() for s in ("base", "third", "main", "agentview")):
                    sub = v
        pose = parse_pose(sub) if sub is not None else None
    if K is None or pose is None:
        # look for camera-named dicts containing both
        for path, k, v in walk(j):
            if isinstance(v, dict) and cam in str(k).lower():
                if K is None:
                    kk = find_key(v, ["intrinsics", "camera_intrinsics", "K", "intrinsic"])
                    K = parse_intrinsics(kk) if kk is not None else None
                if pose is None:
                    pp = find_key(v, ["pose", "camera_pose", "extrinsics", "cam2world", "cam2world_gl"])
                    pose = parse_pose(pp) if pp is not None else None
    if K is None or pose is None:
        return None
    R, t, conv = pose
    return {"K": K, "R": np.array(R, dtype=float), "t": np.array(t, dtype=float), "conv": conv}


def depth_scale(j, cam):
    """meters_per_unit for depth (default 0.001)."""
    for path, k, v in walk(j):
        if k == "meters_per_unit" and isinstance(v, (int, float)) and v > 0:
            return float(v)
    return 0.001


# --------------------------------------------------------------------------- observation
class Obs:
    def __init__(self):
        self.json = {}
        self.state = None
        self.step_id = None
        self.rgb = {}
        self.depth = {}       # metres, float64, 0 = invalid
        self.calib = {}
        self.ended = False
        self.complete = False
        self.instruction = ""
        self.notes = []
        self.stale = set()


def build_obs(reply):
    o = Obs()
    j = reply_json(reply)
    o.json = j
    o.state = get_state(j)
    o.step_id = get_step_id(j)
    o.ended = episode_ended(j)
    o.complete = task_complete(j)
    o.instruction = get_instruction(j)
    scale = depth_scale(j, "front")

    # 1) image blocks
    blocks = image_blocks(reply)
    # descriptor list in JSON matching blocks?
    desc = None
    for path, k, v in walk(j):
        if isinstance(v, list) and v and all(isinstance(e, dict) and "camera" in e for e in v) \
                and len(v) == len(blocks):
            desc = v
            break
    rgb_order, depth_order = [], []
    for i, b in enumerate(blocks):
        cam = b["camera"]
        kind = b["kind"]
        if desc is not None:
            cam = str(desc[i].get("camera", cam) or cam).lower()
            dk = desc[i].get("kind")
            if dk in ("rgb", "depth"):
                kind = dk if not (dk == "depth" and b["kind"] == "rgb" and b["array"].ndim == 3) else kind
        if kind == "depth":
            depth_order.append((cam, b))
        else:
            rgb_order.append((cam, b))
    for order, store, conv in ((rgb_order, o.rgb, None), (depth_order, o.depth, scale)):
        unnamed = [b for c, b in order if c not in ("front", "wrist")]
        named = [(c, b) for c, b in order if c in ("front", "wrist")]
        for c, b in named:
            if c not in store:
                store[c] = b["array"] if conv is None else b["array"] * conv
        default_cams = [c for c in ("front", "wrist") if c not in store]
        for c, b in zip(default_cams, unnamed):
            store[c] = b["array"] if conv is None else b["array"] * conv
    o.notes.append("blocks=%d rgb=%s depth=%s" % (len(blocks), list(o.rgb), list(o.depth)))

    # 2) file entries in the reply itself (current frame), then manifest last frame
    need = [(c, k) for c in ("front", "wrist") for k in ("rgb", "depth")
            if (k == "rgb" and c not in o.rgb) or (k == "depth" and c not in o.depth)]
    if need:
        entries = []
        man_last, _ = manifest_last_execution_files(j)
        direct = [e for e in file_entries(j) if "frames" not in [str(x) for x in e["ctx"]]]
        entries.extend(direct)
        if man_last:
            entries.extend(man_last["files"])
        for c, k in need:
            cands = [e for e in entries if (e.get("camera") or "").lower().startswith(c) and e.get("kind") == k]
            if not cands:
                continue
            e = cands[0] if cands[0] in direct else cands[-1]
            if man_last and e in man_last["files"] and o.step_id is not None:
                fsid = man_last["frame"].get("step_id")
                if isinstance(fsid, int) and fsid != o.step_id:
                    o.notes.append("manifest frame step %s != obs step %s: %s/%s stale" % (fsid, o.step_id, c, k))
                    if k == "depth":
                        o.stale.add(c)
                    continue
            try:
                kind, arr = load_image_file(e["path"])
                if k == "depth":
                    if kind != "depth":
                        continue
                    mpu = e.get("mpu") or scale
                    o.depth[c] = arr * float(mpu) if arr.max() > 50 else arr
                else:
                    o.rgb[c] = arr if arr.ndim == 3 else np.stack([arr] * 3, -1)
                o.notes.append("file %s/%s <- %s" % (c, k, e["path"]))
            except Exception as ex:
                o.notes.append("file load fail %s: %r" % (e.get("path"), ex))

    # 3) base64 PNG strings inside the JSON (e.g. {"front_depth_png": "iVBOR..."})
    for path, k, v in walk(j):
        if isinstance(v, str) and len(v) > 200 and v[:6] in ("iVBORw", "/9j/4A"):
            ctxs = " ".join(str(x) for x in path) + " " + str(k)
            c, kd = _camera_from_text(ctxs)
            if c is None:
                continue
            try:
                kind, arr = _im_to_array(_decode_png_bytes(base64.b64decode(v)))
            except Exception:
                continue
            if kind == "depth" and c not in o.depth:
                o.depth[c] = arr * scale
                o.notes.append("b64 depth %s" % c)
            elif kind == "rgb" and c not in o.rgb and kd != "depth":
                o.rgb[c] = arr
    for c in ("front", "wrist"):
        cal = camera_calibration(j, c)
        if cal is not None:
            o.calib[c] = cal
    o.notes.append("calib=%s" % list(o.calib))
    return o


def parse_points_reply(reply, n):
    """Parse robomme_pixel_to_position reply into list of (xyz or None, spread)."""
    j = reply_json(reply)
    best = None
    for path, k, v in walk(j):
        if isinstance(v, list) and len(v) == n and all(isinstance(e, dict) for e in v):
            best = v
            break
    if best is None:
        # maybe top-level list in text
        for t in j.get("_texts", []):
            try:
                v = json.loads(t)
            except Exception:
                continue
            if isinstance(v, list) and len(v) == n:
                best = v
    out = []
    if best is None:
        return [None] * n
    for e in best:
        valid = e.get("valid", True)
        xyz = None
        for key in ("position", "world_position", "point", "xyz", "world", "position_m", "world_point",
                    "world_xyz", "point_world", "xyz_world", "position_world", "world_position_m"):
            if key in e and _is_vec(e[key], 3):
                xyz = np.array(e[key], dtype=float)
                break
        if xyz is None:
            for path, k, v in walk(e):
                ks = str(k).lower()
                if _is_vec(v, 3) and not any(b in ks for b in ("pixel", "uv", "camera", "ray", "origin",
                                                                "normal", "direction")):
                    xyz = np.array(v, dtype=float)
                    break
        spread = e.get("depth_spread_m")
        out.append((xyz if (valid and xyz is not None) else None, spread))
    return out
