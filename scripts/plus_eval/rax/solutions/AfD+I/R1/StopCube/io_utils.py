"""Defensive parsing of RAX MCP replies (format never observed offline).

Every function here must tolerate missing or differently-shaped fields."""
import base64
import io
import json
import os
import re

import numpy as np

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None

WORKSPACE = "/workspace"


# ----------------------------------------------------------------- generic walkers
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
    """First value whose key is in names (breadth-first preference: shallow first)."""
    best = None
    for path, k, v in walk(obj):
        if k in names and (pred is None or pred(v)):
            if best is None or len(path) < best[0]:
                best = (len(path), v)
    return None if best is None else best[1]


def is_num_list(v, n=None):
    if not isinstance(v, (list, tuple)):
        return False
    if n is not None and len(v) != n:
        return False
    return len(v) > 0 and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v)


# ----------------------------------------------------------------- images
def decode_image_bytes(data):
    if Image is None:
        return None
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
        return im
    except Exception:
        return None


def pil_to_rgb_array(im):
    if im is None:
        return None
    if im.mode in ("RGB", "RGBA", "P", "L") and im.mode != "L":
        return np.asarray(im.convert("RGB"), np.uint8)
    return None


def load_path_image(rel):
    if not isinstance(rel, str):
        return None
    for p in (rel, os.path.join(WORKSPACE, rel)):
        try:
            if os.path.isfile(p):
                with open(p, "rb") as f:
                    return decode_image_bytes(f.read())
        except Exception:
            pass
    return None


# ----------------------------------------------------------------- reply parsing
class Obs:
    def __init__(self):
        self.meta = {}
        self.texts = []
        self.step_id = None
        self.state = None
        self.instruction = None
        self.ended = False
        self.complete = None
        self.front = None
        self.wrist = None
        self.front_src = None
        self.remaining = None
        self.max_steps = None
        self.raw_image_count = 0
        self.notes = []

    def summary(self):
        st = None if self.state is None else [round(float(x), 4) for x in self.state]
        return {"step_id": self.step_id, "state": st, "ended": self.ended, "complete": self.complete,
                "front": None if self.front is None else list(self.front.shape), "front_src": self.front_src,
                "wrist": None if self.wrist is None else list(self.wrist.shape),
                "remaining": self.remaining, "max_steps": self.max_steps, "images": self.raw_image_count,
                "notes": self.notes[:6]}


def _label_from(obj):
    s = json.dumps(obj, default=str).lower() if not isinstance(obj, str) else obj.lower()
    if "wrist" in s or "hand" in s:
        return "wrist"
    if "front" in s or "third" in s or "base_camera" in s:
        return "front"
    return None


def parse_reply(result):
    o = Obs()
    if not isinstance(result, dict):
        o.notes.append("result not dict")
        return o
    content = result.get("content") or []
    images = []  # (index, label_hint, PIL image)
    last_text = None
    for i, b in enumerate(content):
        if not isinstance(b, dict):
            continue
        t = b.get("type")
        if t == "text":
            txt = b.get("text", "")
            o.texts.append(txt)
            last_text = txt
            try:
                j = json.loads(txt)
                if isinstance(j, dict):
                    if not o.meta:
                        o.meta = j
                    else:
                        o.meta.setdefault("_extra", []).append(j)
            except Exception:
                pass
        elif t == "image":
            data = b.get("data")
            im = None
            if isinstance(data, str):
                try:
                    im = decode_image_bytes(base64.b64decode(data))
                except Exception:
                    im = None
            hint = None
            for key in ("annotations", "_meta", "name", "title", "uri", "camera"):
                if key in b and b[key]:
                    hint = hint or _label_from(b[key])
            if hint is None and last_text is not None and len(last_text) < 200:
                hint = _label_from(last_text)
            images.append((i, hint, im))
        elif t == "resource":
            pass
    sc = result.get("structuredContent")
    if isinstance(sc, dict):
        if not o.meta:
            o.meta = sc
        else:
            o.meta.setdefault("_structured", sc)
    o.raw_image_count = len(images)
    m = o.meta

    # scalar fields
    sid = find_key(m, ("step_id",), lambda v: isinstance(v, int) and not isinstance(v, bool))
    if sid is None:
        sid = find_key(m, ("step", "elapsed_steps", "steps"), lambda v: isinstance(v, int) and not isinstance(v, bool))
    o.step_id = sid
    st = find_key(m, ("state",), lambda v: is_num_list(v) and len(v) >= 8)
    if st is None:
        st = find_key(m, ("proprio", "robot_state", "state_vector"), lambda v: is_num_list(v) and len(v) >= 8)
    if st is not None:
        o.state = np.asarray(st[:8], float)
    ins = find_key(m, ("instruction", "language_instruction", "goal", "task_instruction"),
                   lambda v: isinstance(v, str) and len(v) > 3)
    o.instruction = ins
    for name in ("episode_ended", "terminated", "done", "truncated"):
        v = find_key(m, (name,), lambda v: isinstance(v, bool))
        if v is True:
            o.ended = True
    tc = find_key(m, ("task_complete", "success"), lambda v: isinstance(v, bool))
    o.complete = tc
    rem = find_key(m, ("remaining_steps", "steps_remaining", "remaining"),
                   lambda v: isinstance(v, int) and not isinstance(v, bool))
    o.remaining = rem
    mx = find_key(m, ("max_steps", "step_limit", "max_episode_steps", "horizon"),
                  lambda v: isinstance(v, int) and not isinstance(v, bool))
    o.max_steps = mx

    # ---------------- images: labelled inline first
    rgb_inline = []
    for idx, hint, im in images:
        arr = pil_to_rgb_array(im)
        if arr is None:
            continue
        rgb_inline.append((idx, hint, arr))
    # metadata hints listing cameras in attachment order
    order_hint = None
    for key in ("images", "attachments", "image_order", "cameras"):
        v = find_key(m, (key,))
        if isinstance(v, list) and v:
            labs = [_label_from(x) for x in v]
            if any(labs):
                order_hint = labs
                break
    if order_hint and len(order_hint) >= len(rgb_inline):
        for j, (idx, hint, arr) in enumerate(rgb_inline):
            if hint is None and j < len(order_hint):
                rgb_inline[j] = (idx, order_hint[j], arr)
    for idx, hint, arr in rgb_inline:
        if hint == "front" and o.front is None:
            o.front, o.front_src = arr, "inline-labelled"
        elif hint == "wrist" and o.wrist is None:
            o.wrist = arr
    # ---------------- files listed in metadata (camera/kind/path)
    if o.front is None or o.wrist is None:
        files = []
        for path, k, v in walk(m):
            if isinstance(v, dict) and "camera" in v:
                p = None
                for pk in ("path", "file", "rgb_path", "image_path"):
                    if isinstance(v.get(pk), str):
                        p = v[pk]
                        break
                kind = str(v.get("kind", "rgb")).lower()
                if p and "depth" not in kind and "depth" not in p.lower():
                    files.append((str(v["camera"]).lower(), p))
        for cam, p in reversed(files):   # last listed = newest
            if "front" in cam and o.front is None:
                arr = pil_to_rgb_array(load_path_image(p))
                if arr is not None:
                    o.front, o.front_src = arr, "meta-file"
            if "wrist" in cam and o.wrist is None:
                arr = pil_to_rgb_array(load_path_image(p))
                if arr is not None:
                    o.wrist = arr
    # ---------------- unlabelled inline: use dark-bottom heuristic
    if (o.front is None or o.wrist is None) and rgb_inline:
        from perception import dark_bottom_score
        unl = [(idx, arr) for idx, hint, arr in rgb_inline if hint is None]
        if o.front is None and o.wrist is None and len(unl) >= 2:
            a, b = unl[0][1], unl[1][1]
            if dark_bottom_score(a) <= dark_bottom_score(b):
                o.front, o.wrist = a, b
            else:
                o.front, o.wrist = b, a
            o.front_src = "inline-heuristic"
        elif o.front is None and unl:
            o.front, o.front_src = unl[0][1], "inline-only"
        elif o.wrist is None and unl:
            o.wrist = unl[0][1]
    return o


def frame_sequence_manifest(meta):
    """Path of the frame_sequence manifest in a reply, if any."""
    fs = find_key(meta, ("frame_sequence",))
    if isinstance(fs, dict):
        p = fs.get("manifest_path")
        if isinstance(p, str):
            return p
    p = find_key(meta, ("manifest_path",), lambda v: isinstance(v, str))
    return p


def read_manifest_frames(rel):
    try:
        p = rel if os.path.isabs(rel) else os.path.join(WORKSPACE, rel)
        with open(p) as f:
            return json.load(f).get("frames", [])
    except Exception:
        return []


def manifest_front_images(rel):
    """[(step_id, frame_index, phase, rgb array)] for the front camera of a manifest."""
    out = []
    for fr in read_manifest_frames(rel):
        try:
            files = fr.get("files", [])
            path = None
            for f in files:
                if str(f.get("camera", "")).lower().startswith("front") and str(f.get("kind", "rgb")).lower() == "rgb":
                    path = f.get("path")
            if path is None:
                continue
            arr = pil_to_rgb_array(load_path_image(path))
            if arr is not None:
                out.append((fr.get("step_id"), fr.get("frame_index"), fr.get("phase"), arr))
        except Exception:
            continue
    return out


# ----------------------------------------------------------------- pixel_to_position
def parse_points(result, n):
    """Extract up to n world points (or None) from a pixel_to_position reply, in order."""
    meta = {}
    if isinstance(result, dict):
        for b in result.get("content") or []:
            if isinstance(b, dict) and b.get("type") == "text":
                try:
                    j = json.loads(b.get("text", ""))
                    if isinstance(j, (dict, list)):
                        meta = j
                        break
                except Exception:
                    pass
        if not meta and isinstance(result.get("structuredContent"), (dict, list)):
            meta = result["structuredContent"]
    # candidate lists of per-selection records
    lists = []
    if isinstance(meta, list):
        lists.append(meta)
    for path, k, v in walk(meta):
        if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
            lists.append(v)
    pref = ("world_position", "position_world", "world", "position", "point", "xyz", "world_point",
            "point_world", "position_m", "world_m")

    def rec_point(d):
        if not isinstance(d, dict):
            return None
        valid = d.get("valid", True)
        for k in pref:
            if k in d and is_num_list(d[k], 3):
                return (np.asarray(d[k], float), bool(valid) if isinstance(valid, bool) else True, d)
        for k, v in d.items():
            if is_num_list(v, 3) and "pixel" not in k and "uv" not in k:
                return (np.asarray(v, float), bool(valid) if isinstance(valid, bool) else True, d)
            if isinstance(v, dict) and all(c in v for c in ("x", "y", "z")):
                return (np.asarray([v["x"], v["y"], v["z"]], float), True, d)
        if all(c in d for c in ("x", "y", "z")) and all(isinstance(d[c], (int, float)) for c in ("x", "y", "z")):
            return (np.asarray([d["x"], d["y"], d["z"]], float), bool(valid) if isinstance(valid, bool) else True, d)
        if not valid:
            return (None, False, d)
        return None

    best = None
    for L in lists:
        pts = [rec_point(x) for x in L]
        good = [p for p in pts if p is not None]
        if len(good) == len(L) and (best is None or abs(len(L) - n) < abs(len(best) - n)):
            best = pts
    out = []
    if best is not None:
        for p in best[:n]:
            if p is None or p[0] is None or not p[1] or not np.all(np.isfinite(p[0])):
                out.append(None)
            else:
                out.append(p[0])
    while len(out) < n:
        out.append(None)
    return out, meta


# ----------------------------------------------------------------- instruction
ORD = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8,
       "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14,
       "fifteenth": 15, "sixteenth": 16, "seventeenth": 17, "eighteenth": 18, "nineteenth": 19, "twentieth": 20}
CARD = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
        "ten": 10, "eleven": 11, "twelve": 12, "once": 1, "twice": 2, "thrice": 3}


def parse_count(text):
    if not isinstance(text, str):
        return None
    t = text.lower()
    m = re.search(r"the\s+([a-z0-9]+)\s+time", t)
    if m:
        w = m.group(1)
        if w in ORD:
            return ORD[w]
        m2 = re.match(r"(\d+)(st|nd|rd|th)?$", w)
        if m2:
            return int(m2.group(1))
    m = re.search(r"(\d+)\s*(st|nd|rd|th)\b", t)
    if m:
        return int(m.group(1))
    for w, v in sorted(ORD.items(), key=lambda kv: -len(kv[0])):
        if re.search(r"\b" + w + r"\b", t):
            return v
    m = re.search(r"\b(\d+)\s*(times|time)\b", t)
    if m:
        return int(m.group(1))
    for w, v in CARD.items():
        if w in ("once", "twice", "thrice"):
            if re.search(r"\b" + w + r"\b", t):
                return v
        elif re.search(r"\b" + w + r"\s+times?\b", t):
            return v
    return None
