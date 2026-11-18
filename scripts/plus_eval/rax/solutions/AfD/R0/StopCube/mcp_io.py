"""Defensive parsing of RoboMME-Plus MCP replies (format verified only from docs).

Everything here is written to tolerate unknown key names: values are located by
recursive search, images are classified by decoded mode, and camera assignment is
cross-checked against the frame-sequence manifest when one is present.
"""
import base64
import io
import json
import os
import time

import numpy as np
from PIL import Image

WORKSPACE = "/workspace"


# ----------------------------------------------------------------- JSON search
def walk(obj, path=()):
    """Yield (path, key, value) for every dict entry in a nested structure."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield path, k, v
            yield from walk(v, path + (k,))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from walk(v, path + (i,))


def find_key(obj, names, pred=None):
    """Shallowest value whose key is in `names` (case-insensitive) and satisfies pred (BFS)."""
    names = {n.lower() for n in names}
    queue = [obj]
    while queue:
        nxt = []
        for o in queue:
            if isinstance(o, dict):
                for k, v in o.items():
                    if isinstance(k, str) and k.lower() in names and (pred is None or pred(v)):
                        return v
                for v in o.values():
                    if isinstance(v, (dict, list, tuple)):
                        nxt.append(v)
            elif isinstance(o, (list, tuple)):
                for v in o:
                    if isinstance(v, (dict, list, tuple)):
                        nxt.append(v)
        queue = nxt
    return None


def is_num_list(v, n=None):
    if not isinstance(v, (list, tuple)):
        return False
    if n is not None and len(v) != n:
        return False
    return len(v) > 0 and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v)


# ----------------------------------------------------------------- images
def _decode_image(b64):
    raw = base64.b64decode(b64)
    im = Image.open(io.BytesIO(raw))
    im.load()
    return im


def image_to_arrays(im):
    """Return ('rgb', HxWx3 uint8) or ('depth', HxW float32 metres-or-units)."""
    mode = im.mode
    if mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        arr = np.asarray(im).astype(np.float32)
        return "depth16", arr
    if mode in ("L", "LA", "P") and mode != "P":
        return "gray", np.asarray(im.convert("L"))
    return "rgb", np.asarray(im.convert("RGB"))


class Reply:
    """Parsed MCP tool result."""

    def __init__(self, result):
        self.raw = result
        self.meta = {}
        self.texts = []
        self.images = []  # list of dict(kind, arr, label)
        content = result.get("content", []) if isinstance(result, dict) else []
        last_text = ""
        for block in content:
            t = block.get("type")
            if t == "text":
                txt = block.get("text", "")
                self.texts.append(txt)
                try:
                    j = json.loads(txt)
                    if isinstance(j, dict):
                        if not self.meta:
                            self.meta = j
                        else:
                            self.meta.setdefault("_extra", []).append(j)
                except Exception:
                    last_text = txt
            elif t == "image":
                try:
                    im = _decode_image(block.get("data", ""))
                    kind, arr = image_to_arrays(im)
                    self.images.append({"kind": kind, "arr": arr, "label": last_text.lower(),
                                        "mime": block.get("mimeType", "")})
                except Exception as e:  # keep going
                    self.images.append({"kind": "error", "arr": None, "label": str(e)})
                last_text = ""
        if isinstance(result, dict) and isinstance(result.get("structuredContent"), dict) and not self.meta:
            self.meta = result["structuredContent"]

    # -- simple fields
    def get(self, *names, pred=None):
        return find_key(self.meta, names, pred)

    @property
    def step_id(self):
        v = self.get("step_id", pred=lambda x: isinstance(x, int) and not isinstance(x, bool))
        return v

    @property
    def state(self):
        v = self.get("state", pred=lambda x: is_num_list(x, 8))
        if v is None:
            v = self.get("state", "robot_state", "proprio", "proprioception",
                         pred=lambda x: is_num_list(x) and len(x) >= 8)
        if v is None:
            d = self.get("state", "robot_state", pred=lambda x: isinstance(x, dict))
            if isinstance(d, dict):
                pos = find_key(d, ["ee_pos", "tcp_pos", "position", "ee_position", "pos"], lambda x: is_num_list(x, 3))
                rot = find_key(d, ["ee_axis_angle", "axis_angle", "orientation", "rotation", "rotvec"], lambda x: is_num_list(x, 3))
                fin = find_key(d, ["fingers", "gripper", "finger_positions", "gripper_qpos"], lambda x: is_num_list(x, 2))
                if pos is not None:
                    v = list(pos) + list(rot or [0.0, 0.0, 0.0]) + list(fin or [0.0, 0.0])
        return None if v is None else [float(x) for x in v[:8]]

    @property
    def instruction(self):
        v = self.get("instruction", "goal", "language_instruction", pred=lambda x: isinstance(x, str) and len(x) > 3)
        return v

    @property
    def episode_ended(self):
        v = self.get("episode_ended", "done", "terminated", pred=lambda x: isinstance(x, bool))
        return bool(v) if v is not None else False

    @property
    def task_complete(self):
        v = self.get("task_complete", "success", pred=lambda x: isinstance(x, bool))
        return v

    @property
    def manifest_path(self):
        fs = self.get("frame_sequence", pred=lambda x: isinstance(x, dict))
        if isinstance(fs, dict):
            p = fs.get("manifest_path")
            if isinstance(p, str):
                return p
        v = self.get("manifest_path", pred=lambda x: isinstance(x, str))
        return v

    def remaining_steps(self):
        lim = self.get("episode_limits", pred=lambda x: isinstance(x, dict))
        if isinstance(lim, dict):
            for k in ("remaining_steps", "steps_remaining", "remaining"):
                if isinstance(lim.get(k), int):
                    return lim[k]
            mx = None
            for k in ("max_steps", "step_limit", "max_episode_steps", "limit"):
                if isinstance(lim.get(k), int):
                    mx = lim[k]
            used = None
            for k in ("steps_used", "elapsed_steps", "step", "steps"):
                if isinstance(lim.get(k), int):
                    used = lim[k]
            if mx is not None and used is not None:
                return mx - used
        return None


# ----------------------------------------------------------------- manifest
def load_manifest(path):
    if not path:
        return None
    p = path if os.path.isabs(path) else os.path.join(WORKSPACE, path)
    for _ in range(3):
        try:
            with open(p) as f:
                return json.load(f)
        except Exception:
            time.sleep(0.05)
    return None


def manifest_frame_files(frame):
    """{(camera, kind): (abs_path, meters_per_unit)} for one manifest frame."""
    out = {}
    for f in frame.get("files", []) or []:
        cam = str(f.get("camera", "")).lower()
        kind = str(f.get("kind", "")).lower()
        p = f.get("path")
        if not p:
            continue
        ap = p if os.path.isabs(p) else os.path.join(WORKSPACE, p)
        out[(cam, kind)] = (ap, f.get("meters_per_unit", 0.001))
    return out


def load_rgb(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def load_depth_m(path, mpu=0.001):
    with Image.open(path) as im:
        arr = np.asarray(im).astype(np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    d = arr * float(mpu if mpu else 0.001)
    d[arr <= 0] = np.nan
    return d


# ----------------------------------------------------------------- camera assignment
def assign_cameras(reply, mapping_hint=None):
    """Return dict camera->rgb array, camera->depth array (metres) from inline images.

    mapping_hint: list of camera names for rgb images in order (learned at step 0).
    """
    rgbs = [im for im in reply.images if im["kind"] == "rgb" and im["arr"] is not None]
    deps = [im for im in reply.images if im["kind"] == "depth16" and im["arr"] is not None]
    out_rgb, out_dep = {}, {}
    # labels from preceding text blocks
    for lst, out in ((rgbs, out_rgb), (deps, out_dep)):
        for im in lst:
            lab = im.get("label", "")
            if "wrist" in lab and "wrist" not in out:
                out["wrist"] = im["arr"]
            elif ("front" in lab or "third" in lab) and "front" not in out:
                out["front"] = im["arr"]
    if not out_rgb:
        names = mapping_hint or ["front", "wrist"]
        for i, im in enumerate(rgbs[:len(names)]):
            out_rgb[names[i]] = im["arr"]
    if not out_dep:
        names = mapping_hint or ["front", "wrist"]
        for i, im in enumerate(deps[:len(names)]):
            arr = im["arr"].astype(np.float32) * 0.001
            arr[im["arr"] <= 0] = np.nan
            out_dep[names[i]] = arr
    return out_rgb, out_dep


def match_order_with_manifest(reply, manifest):
    """Use the last manifest frame's labelled RGB files to learn inline image order."""
    if not manifest or not manifest.get("frames"):
        return None
    frame = manifest["frames"][-1]
    files = manifest_frame_files(frame)
    ref = {}
    for cam in ("front", "wrist"):
        if (cam, "rgb") in files:
            try:
                ref[cam] = load_rgb(files[(cam, "rgb")][0]).astype(np.int16)
            except Exception:
                pass
    rgbs = [im["arr"] for im in reply.images if im["kind"] == "rgb" and im["arr"] is not None]
    if not ref or not rgbs:
        return None
    order = []
    for arr in rgbs:
        best, bestd = None, 1e18
        for cam, r in ref.items():
            if r.shape != arr.shape:
                continue
            d = float(np.abs(r - arr.astype(np.int16)).mean())
            if d < bestd:
                best, bestd = cam, d
        order.append((best, bestd))
    names = [o[0] for o in order]
    if len(set(names)) == len(names) and all(n is not None for n in names) and all(o[1] < 20 for o in order):
        return names
    # only one camera referenced (or ambiguous): the closest image is that camera, others the rest
    if len(ref) == 1 and len(rgbs) == 2:
        cam = next(iter(ref))
        other = "wrist" if cam == "front" else "front"
        ds = [float(np.abs(ref[cam] - a.astype(np.int16)).mean()) if a.shape == ref[cam].shape else 1e18
              for a in rgbs]
        i = int(np.argmin(ds))
        if ds[i] < 20 and ds[1 - i] > ds[i] + 10:
            out = [other, other]
            out[i] = cam
            return out
    return None


# ----------------------------------------------------------------- pixel_to_position
def parse_points(reply, n):
    """Return list of n dicts {valid, xyz, spread, depth} from a pixel_to_position reply."""
    meta = reply.meta
    cand = None
    for _, k, v in walk(meta):
        if isinstance(v, list) and len(v) == n and all(isinstance(e, dict) for e in v):
            cand = v
            break
    if cand is None and isinstance(meta, list):
        cand = meta
    out = []
    for i in range(n):
        e = cand[i] if cand is not None and i < len(cand) else {}
        xyz = None
        for key in ("position", "world_position", "world", "point", "xyz", "position_m", "world_point"):
            v = e.get(key)
            if is_num_list(v, 3):
                xyz = [float(x) for x in v]
                break
            if isinstance(v, dict):
                vv = [v.get("x"), v.get("y"), v.get("z")]
                if all(isinstance(x, (int, float)) for x in vv):
                    xyz = [float(x) for x in vv]
                    break
        if xyz is None:
            for _, k, v in walk(e):
                if is_num_list(v, 3) and not (isinstance(k, str) and "pix" in k.lower()):
                    xyz = [float(x) for x in v]
                    break
        valid = e.get("valid", xyz is not None)
        if xyz is None:
            valid = False
        spread = find_key(e, ["depth_spread_m", "spread"], lambda x: isinstance(x, (int, float)))
        depth = find_key(e, ["depth_m", "depth"], lambda x: isinstance(x, (int, float)))
        out.append({"valid": bool(valid), "xyz": xyz, "spread": spread, "depth": depth})
    return out
