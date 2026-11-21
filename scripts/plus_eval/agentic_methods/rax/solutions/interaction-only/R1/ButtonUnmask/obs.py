"""Defensive parsing of RoboMME-Plus MCP replies (JSON metadata + image blocks)."""
import base64
import io
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from geom import Camera, pose_to_T

WS = Path("/workspace")
CAMS = ("front", "wrist")


def text_blocks(reply):
    out = []
    for b in (reply or {}).get("content", []) or []:
        if isinstance(b, dict) and b.get("type") == "text":
            out.append(b.get("text", ""))
    return out


def reply_meta(reply):
    for t in text_blocks(reply):
        try:
            d = json.loads(t)
            if isinstance(d, dict):
                return d
        except Exception:
            continue
    return {}


def walk(obj, path=()):
    yield path, obj
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk(v, path + (str(k),))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk(v, path + (i,))


def find_key(meta, names, types=None):
    """First value whose key (lower) is in names, breadth-first by depth."""
    best = None
    for p, v in walk(meta):
        if p and isinstance(p[-1], str) and p[-1].lower() in names:
            if types is None or isinstance(v, types):
                if best is None or len(p) < len(best[0]):
                    best = (p, v)
    return None if best is None else best[1]


def resolve_path(p, base_dirs=()):
    p = str(p)
    cands = [Path(p), WS / p]
    for b in base_dirs:
        cands.append(Path(b) / p)
    for c in cands:
        try:
            if c.is_file():
                return c
        except Exception:
            pass
    return None


def _cam_of(path_keys, d, p):
    cam = d.get("camera") if isinstance(d, dict) else None
    if isinstance(cam, str):
        for c in CAMS:
            if c in cam.lower():
                return c
    for k in reversed(path_keys):
        if isinstance(k, str):
            for c in CAMS:
                if c in k.lower():
                    return c
    s = str(p).lower()
    for c in CAMS:
        if c in os.path.basename(s):
            return c
    for c in CAMS:
        if c in s:
            return c
    return None


def _kind_of(path_keys, d, p):
    k = d.get("kind") if isinstance(d, dict) else None
    if isinstance(k, str):
        k = k.lower()
        if "depth" in k:
            return "depth"
        if k in ("rgb", "color", "image") or "rgb" in k:
            return "rgb"
    for kk in reversed(path_keys):
        if isinstance(kk, str):
            l = kk.lower()
            if "depth" in l:
                return "depth"
            if "rgb" in l or "color" in l:
                return "rgb"
    s = os.path.basename(str(p)).lower()
    if "depth" in s:
        return "depth"
    if "rgb" in s or "color" in s:
        return "rgb"
    return None


def file_entries(meta):
    """All dicts that reference an image file path, with camera/kind/frame context."""
    out = []
    for p, v in walk(meta):
        if isinstance(v, dict):
            fp = v.get("path") or v.get("file") or v.get("rgb_path") or v.get("depth_path")
            if isinstance(fp, str) and fp.lower().endswith((".png", ".jpg", ".jpeg", ".npy")):
                in_frames = any(isinstance(k, str) and k.lower() in ("frames", "saved_frames", "steps", "sequence")
                                for k in p)
                out.append(dict(keys=p, d=v, path=fp, cam=_cam_of(p, v, fp), kind=_kind_of(p, v, fp),
                                mpu=float(v.get("meters_per_unit", 0.001) or 0.001), in_frames=in_frames))
        elif isinstance(v, str) and p and isinstance(p[-1], str) and v.lower().endswith((".png", ".npy")):
            parent = p[:-1]
            out.append(dict(keys=p, d={}, path=v, cam=_cam_of(p, {}, v), kind=_kind_of(p, {}, v), mpu=0.001,
                            in_frames=any(isinstance(k, str) and k.lower() == "frames" for k in parent)))
    return out


def load_rgb(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB")).copy()


def load_depth(path, mpu=0.001):
    path = Path(path)
    if path.suffix == ".npy":
        a = np.load(path).astype(np.float32)
        if a.ndim == 3:
            a = a[..., 0]
        if a.max() > 50:
            a = a * mpu
        return a
    with Image.open(path) as im:
        a = np.asarray(im)
    a = a.astype(np.float32)
    if a.ndim == 3:
        a = a[..., 0]
    return a * mpu


def decode_block_image(b):
    try:
        raw = base64.b64decode(b.get("data", ""))
        im = Image.open(io.BytesIO(raw))
        im.load()
        return im
    except Exception:
        return None


def _parse_K(v):
    try:
        if isinstance(v, dict):
            for key in ("matrix", "K", "k", "intrinsic_matrix", "intrinsics", "camera_matrix"):
                if key in v:
                    return _parse_K(v[key])
            if all(k in v for k in ("fx", "fy", "cx", "cy")):
                return np.array([[v["fx"], 0, v["cx"]], [0, v["fy"], v["cy"]], [0, 0, 1]], float)
            return None
        a = np.asarray(v, dtype=float)
        if a.shape == (3, 3):
            return a
        if a.size == 9:
            return a.reshape(3, 3)
        if a.size == 4:
            fx, fy, cx, cy = a
            return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], float)
        if a.shape == (3, 4):
            return a[:, :3]
    except Exception:
        return None
    return None


def _parse_T(v):
    try:
        if isinstance(v, dict):
            pos = None
            q = None
            for key in ('position_m', "position", "p", "pos", "xyz", "translation", "t"):
                if key in v:
                    pos = v[key]
                    break
            for key in ("quaternion_wxyz", "wxyz", "quaternion", "quat", "q", "orientation", "rotation"):
                if key in v:
                    q = v[key]
                    break
            if pos is not None and q is not None and len(q) == 4:
                return pose_to_T(pos, q)
            for key in ("matrix", "cam2world", "camera_to_world", "pose", "T", "extrinsic", "extrinsics",
                        "world_from_camera", "model_matrix"):
                if key in v:
                    r = _parse_T(v[key])
                    if r is not None:
                        return r
            return None
        a = np.asarray(v, dtype=float)
        if a.shape == (4, 4):
            return a
        if a.size == 16:
            return a.reshape(4, 4)
        if a.shape == (3, 4):
            T = np.eye(4)
            T[:3] = a
            return T
        if a.size == 7:
            return pose_to_T(a[:3], a[3:])
    except Exception:
        return None
    return None


def _per_camera(meta, keyword, parser):
    out = {}
    for p, v in walk(meta):
        if not p or not isinstance(p[-1], str):
            continue
        k = p[-1].lower()
        if keyword in k:
            if isinstance(v, dict):
                for c in CAMS:
                    for kk, vv in v.items():
                        if c in str(kk).lower() and c not in out:
                            r = parser(vv)
                            if r is not None:
                                out[c] = r
                if not out:
                    c = _cam_of(p, v, "")
                    r = parser(v)
                    if c and r is not None:
                        out[c] = r
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, dict) and isinstance(item.get("camera"), str):
                        c = _cam_of((), item, "")
                        r = parser(item)
                        if c and r is not None and c not in out:
                            out[c] = r
                c = _cam_of(p, {}, "")
                if c and c not in out:
                    r = parser(v)
                    if r is not None:
                        out[c] = r
    return out


def camera_models(meta):
    Ks = _per_camera(meta, "intrinsic", _parse_K)
    Ts = _per_camera(meta, "pose", _parse_T)
    # also nested camera dicts: meta['cameras']['front']['intrinsics'] etc.
    for p, v in walk(meta):
        if isinstance(v, dict) and p and isinstance(p[-1], str):
            c = None
            for cc in CAMS:
                if cc in p[-1].lower():
                    c = cc
            if c is None:
                continue
            for kk, vv in v.items():
                l = str(kk).lower()
                if "intrinsic" in l or l in ("k", "camera_matrix") and c not in Ks:
                    r = _parse_K(vv)
                    if r is not None and c not in Ks:
                        Ks[c] = r
                if ("pose" in l or "extrinsic" in l or "cam2world" in l) and c not in Ts:
                    r = _parse_T(vv)
                    if r is not None:
                        Ts[c] = r
    cams = {}
    for c in CAMS:
        if c in Ks and c in Ts:
            cams[c] = Camera(Ks[c], Ts[c])
    return cams


def parse_state(meta):
    s = meta.get("state") if isinstance(meta, dict) else None
    if isinstance(s, list) and len(s) >= 8 and all(isinstance(x, (int, float)) for x in s[:8]):
        return np.array(s[:8], float)
    if isinstance(s, dict):
        for k in ("values", "vector", "state", "data"):
            if isinstance(s.get(k), list) and len(s[k]) >= 8:
                return np.array(s[k][:8], float)
    def num_list(v, n):
        return isinstance(v, list) and len(v) == n and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v)

    best = None
    for p, v in walk(meta):
        if not p or not any(isinstance(k, str) and "state" in k.lower() for k in p[-3:]):
            continue
        if num_list(v, 8):
            if best is None or len(p) < len(best[0]):
                best = (p, v)
    if best:
        return np.array(best[1], float)
    # dict form: position / orientation / fingers
    for p, v in walk(meta):
        if isinstance(v, dict) and p and isinstance(p[-1], str) and "state" in p[-1].lower():
            pos = rot = fing = None
            for k, x in v.items():
                l = str(k).lower()
                if pos is None and num_list(x, 3) and "pos" in l:
                    pos = x
                elif rot is None and num_list(x, 3) and any(a in l for a in ("axis", "rot", "orient", "ang")):
                    rot = x
                elif fing is None and num_list(x, 2) and any(a in l for a in ("finger", "grip")):
                    fing = x
            if pos is not None and rot is not None:
                return np.array(list(pos) + list(rot) + list(fing or [0.04, 0.04]), float)
    return None


def _bool_key(meta, names):
    for p, v in walk(meta):
        if p and isinstance(p[-1], str) and p[-1].lower() in names and isinstance(v, bool):
            if v:
                return True
    return False


class Obs:
    pass


def parse_obs(reply, want_images=True):
    """Return Obs with meta, step_id, state, instruction, rgb{cam}, depth{cam}, cams{cam}."""
    o = Obs()
    meta = reply_meta(reply)
    o.meta = meta
    sid = meta.get("step_id")
    if not isinstance(sid, int):
        sid = find_key(meta, {"step_id", "step", "elapsed_steps"}, (int,))
    o.step_id = sid
    o.state = parse_state(meta)
    ins = meta.get("instruction")
    if not isinstance(ins, str):
        ins = find_key(meta, {"instruction", "language_instruction", "goal", "task_instruction"}, (str,))
    o.instruction = ins if isinstance(ins, str) else ""
    o.ended = _bool_key(meta, {"episode_ended", "done", "terminated", "truncated", "ended"})
    o.complete = _bool_key(meta, {"task_complete", "success", "is_success"})
    o.rgb, o.depth, o.cams = {}, {}, {}
    o.limits = find_key(meta, {"episode_limits"}, (dict,)) or {}
    fs = find_key(meta, {"frame_sequence"}, (dict,))
    o.manifest_path = fs.get("manifest_path") if isinstance(fs, dict) else None
    if not want_images:
        return o
    try:
        o.cams = camera_models(meta)
    except Exception:
        o.cams = {}
    # files referenced in metadata: prefer entries not inside frame lists
    try:
        ents = file_entries(meta)
    except Exception:
        ents = []
    direct = [e for e in ents if not e["in_frames"]]
    pool = direct if direct else ents
    if not direct and ents:
        # take the last frame's entries (the current observation)
        last_prefix = None
        for e in ents:
            pre = tuple(k for k in e["keys"][:-1])
            # identify frame-level prefix: up to the index after 'frames'
            for i, k in enumerate(e["keys"]):
                if isinstance(k, str) and k.lower() in ("frames", "saved_frames", "steps", "sequence"):
                    if i + 1 < len(e["keys"]):
                        pre = e["keys"][:i + 2]
                    break
            e["prefix"] = pre
            last_prefix = pre
        pool = [e for e in ents if e.get("prefix") == last_prefix]
    for e in pool:
        c, k = e["cam"], e["kind"]
        if c is None or k is None:
            continue
        rp = resolve_path(e["path"])
        if rp is None:
            continue
        try:
            if k == "rgb" and c not in o.rgb:
                o.rgb[c] = load_rgb(rp)
            elif k == "depth" and c not in o.depth:
                o.depth[c] = load_depth(rp, e["mpu"])
        except Exception:
            pass
    # fallback: current frame from the frame_sequence manifest
    try:
        if o.manifest_path and (len(o.rgb) < 2 or len(o.depth) < 2):
            mp = resolve_path(o.manifest_path)
            if mp is not None:
                man = json.loads(mp.read_text())
                frs = man.get("frames", []) if isinstance(man, dict) else []
                if frs:
                    fr = max(frs, key=lambda f: f.get("frame_index", 0) if isinstance(f.get("frame_index"), int) else 0)
                    fsid = fr.get("step_id")
                    if fr.get("phase", "execution") == "execution" and (fsid is None or o.step_id is None or fsid == o.step_id):
                        for e in fr.get("files", []):
                            c = _cam_of((), e, e.get("path", ""))
                            k = _kind_of((), e, e.get("path", ""))
                            rp = resolve_path(e.get("path", ""), [mp.parent])
                            if not c or not k or rp is None:
                                continue
                            if k == "rgb" and c not in o.rgb:
                                o.rgb[c] = load_rgb(rp)
                            elif k == "depth" and c not in o.depth:
                                o.depth[c] = load_depth(rp, float(e.get("meters_per_unit", 0.001) or 0.001))
    except Exception:
        pass
    # inline image blocks
    try:
        blocks = (reply or {}).get("content", []) or []
        rgb_seq, dep_seq = [], []
        hint = None
        for b in blocks:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "text":
                t = b.get("text", "")
                hint = None
                tl = t.lower()[:300]
                if len(t) < 300:
                    for c in CAMS:
                        if c in tl:
                            hint = c
                continue
            if b.get("type") != "image":
                continue
            im = decode_block_image(b)
            if im is None:
                continue
            ann = b.get("annotations") or b.get("_meta") or b.get("meta") or {}
            cam = hint
            for c in CAMS:
                if c in json.dumps(ann).lower() if ann else False:
                    cam = c
            if im.mode in ("I;16", "I", "I;16B", "F"):
                dep_seq.append((cam, np.asarray(im).astype(np.float32) * 0.001))
            else:
                rgb_seq.append((cam, np.asarray(im.convert("RGB")).copy()))
        for seq, store in ((rgb_seq, o.rgb), (dep_seq, o.depth)):
            unl = [a for cam, a in seq if cam is None]
            for cam, a in seq:
                if cam and cam not in store:
                    store[cam] = a
            free = [c for c in CAMS if c not in store]
            for c, a in zip(free, unl):
                store[c] = a
    except Exception:
        pass
    for c, cam in o.cams.items():
        if c in o.depth:
            cam.size = o.depth[c].shape[:2]
        elif c in o.rgb:
            cam.size = o.rgb[c].shape[:2]
    return o


def summarize(obj, depth=0, maxlen=160):
    """Short structural summary for logs."""
    if depth > 4:
        return "..."
    if isinstance(obj, dict):
        return {k: summarize(v, depth + 1) for k, v in list(obj.items())[:40]}
    if isinstance(obj, list):
        if len(obj) > 8:
            return [summarize(obj[0], depth + 1), f"...({len(obj)} items)"]
        return [summarize(v, depth + 1) for v in obj]
    if isinstance(obj, str):
        return obj[:maxlen]
    return obj
