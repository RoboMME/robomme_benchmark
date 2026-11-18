"""VideoPlaceOrder autonomous controller (RAX, AfD: developed offline, never run live).

Pipeline (transition table):
  OBSERVE   -> parse packet, load conditioning video from frame_sequence manifest
            ok -> ANALYSE ; fail -> (no video) ANALYSE with execution frame only
  ANALYSE   -> video_analysis.solve(color, ordinal) -> answer home pixel (front cam)
            fail -> fallback: home of last detected placement, else nearest target to cube
  LOCATE    -> front-camera depth points: answer target (disc) centre & table z, cube top face
  PICK      -> open, hover above cube, wrist-camera refine (xy, yaw), align yaw, descend,
               close, lift, verify finger opening ; miss -> re-locate cube & retry (<=3)
  PLACE     -> transport at safe height above target, descend, open, retreat
  DONE      -> return (or as soon as any reply reports episode_ended / task_complete)
Every tool call is wrapped; a failure in one step is logged and the pipeline continues
with fallbacks.  Logs: /workspace/output/vpo_log.txt
"""
import base64
import io
import json
import math
import os
import re
import sys
import time
import traceback

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import vision  # noqa: E402
import video_analysis as va  # noqa: E402

VERSION = "vpo-v1"
WS = "/workspace"
OUT = os.path.join(WS, "output")

# ---- tunables (untested live: AfD gives no simulator access) ----
POS_SCALE = 0.05          # m per unit action (documented)
ROT_SCALE = 0.5           # rad per unit action (documented)
V_FAST = 0.5              # max action component in free space (2.5 cm/step)
V_SLOW = 0.2              # near contact (1 cm/step)
HOVER_DZ = 0.13           # hover height above cube top for wrist refine
SAFE_DZ = 0.17            # transport height above table
GRASP_FRAC = 0.5          # grasp at this fraction of cube height above table
PLACE_CLEAR = 0.012       # release with cube bottom this far above target top
CLOSE_STEPS = 15
OPEN_STEPS = 10
MIN_OPENING = 0.004       # finger sum (m) below this after closing = missed grasp
MAX_PICK_TRIES = 3
TIME_BUDGET_S = 1100.0

ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
            "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10, "once": 1, "twice": 2}

T0 = time.time()
_LOG = None


def log(*a):
    global _LOG
    msg = "[%7.1fs] " % (time.time() - T0) + " ".join(str(x) for x in a)
    try:
        print(msg, flush=True)
    except Exception:
        pass
    try:
        if _LOG is None:
            os.makedirs(OUT, exist_ok=True)
            _LOG = open(os.path.join(OUT, "vpo_log.txt"), "a")
        _LOG.write(msg + "\n")
        _LOG.flush()
    except Exception:
        pass


class EpisodeOver(Exception):
    pass


# ----------------------------------------------------------------- reply parsing
def reply_json(reply):
    for b in (reply or {}).get("content", []) or []:
        if b.get("type") == "text":
            try:
                return json.loads(b.get("text", ""))
            except Exception:
                continue
    return {}


def find_all(obj, key, out=None, depth=0):
    if out is None:
        out = []
    if depth > 12:
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                out.append(v)
            find_all(v, key, out, depth + 1)
    elif isinstance(obj, list):
        for v in obj[:200]:
            find_all(v, key, out, depth + 1)
    return out


def find_first(obj, key, default=None):
    if isinstance(obj, dict) and key in obj:
        return obj[key]
    r = find_all(obj, key)
    return r[0] if r else default


def is_vec(v, n):
    return isinstance(v, (list, tuple)) and len(v) == n and all(isinstance(x, (int, float)) for x in v)


def packet_state(pk):
    if isinstance(pk, dict):
        if is_vec(pk.get("state"), 8):
            return np.array(pk["state"], float)
        ob = pk.get("observation")
        if isinstance(ob, dict) and is_vec(ob.get("state"), 8):
            return np.array(ob["state"], float)
    for v in find_all(pk, "state"):
        if is_vec(v, 8):
            return np.array(v, float)
    return None


def ended(pk):
    for k in ("episode_ended", "task_complete"):
        for v in find_all(pk, k):
            if v is True:
                return k
    return None


# ----------------------------------------------------------------- robot I/O
class Robot:
    def __init__(self, ctx):
        self.ctx = ctx
        self.pk = {}
        self.state = None
        self.step_id = None
        self.over = None
        self.front_ref = None
        self.nsteps = 0

    def observe(self):
        rep = self.ctx.call("robomme_observe")
        pk = reply_json(rep)
        self.pk = pk
        self.rep = rep
        st = packet_state(pk)
        if st is not None:
            self.state = st
        sid = pk.get("step_id", find_first(pk, "step_id"))
        if isinstance(sid, (int, float)):
            self.step_id = int(sid)
        e = ended(pk)
        if e:
            self.over = e
        return pk

    def step(self, rows):
        rows = [[float(np.clip(x, -1, 1)) for x in r] for r in rows]
        if self.over:
            raise EpisodeOver(self.over)
        for i in range(0, len(rows), 100):
            rep = self.ctx.call("robomme_step", actions=rows[i:i + 100])
            self.nsteps += len(rows[i:i + 100])
            pk = reply_json(rep)
            e = ended(pk)
            lim = find_first(pk, "episode_limits")
            if lim is not None and (self.nsteps <= len(rows) or self.nsteps // 50 != (self.nsteps - len(rows)) // 50):
                log("limits", json.dumps(lim)[:200])
            if e:
                self.over = e
                log("episode reports", e, json.dumps({k: find_first(pk, k) for k in
                                                     ("episode_ended", "task_complete")}))
                raise EpisodeOver(e)
        self.observe()
        if self.over:
            raise EpisodeOver(self.over)

    @property
    def pos(self):
        return self.state[:3].copy()

    @property
    def opening(self):
        return float(self.state[6] + self.state[7])

    # ---- images of the current observation
    def images(self):
        """Return (front_rgb, wrist_rgb) of the current observation (either may be None)."""
        pk, rep = self.pk, getattr(self, "rep", None)
        front = wrist = None
        # 1) file paths published in the packet for this step
        try:
            seq = pk.get("frame_sequence") or find_first(pk, "frame_sequence")
            if isinstance(seq, dict) and seq.get("manifest_path"):
                man = json.load(open(os.path.join(WS, seq["manifest_path"])))
                fr = sorted(man.get("frames", []), key=lambda f: f.get("frame_index", 0))
                if fr and (self.step_id is None or fr[-1].get("step_id") == self.step_id):
                    files = {(f.get("camera"), f.get("kind")): f.get("path") for f in fr[-1].get("files", [])}
                    if files.get(("front", "rgb")):
                        front = load_rgb(os.path.join(WS, files[("front", "rgb")]))
                    if files.get(("wrist", "rgb")):
                        wrist = load_rgb(os.path.join(WS, files[("wrist", "rgb")]))
        except Exception as ex:
            log("images: manifest path failed", repr(ex))
        if front is not None and wrist is not None:
            return front, wrist
        # 2) inline image blocks
        ims = []
        try:
            for b in (rep or {}).get("content", []) or []:
                if b.get("type") != "image":
                    continue
                try:
                    im = Image.open(io.BytesIO(base64.b64decode(b.get("data", ""))))
                    if im.mode not in ("RGB", "RGBA", "P", "L"):
                        continue
                    a = np.array(im.convert("RGB"))
                    if a.ndim != 3:
                        continue
                    ims.append(a)
                except Exception:
                    continue
        except Exception as ex:
            log("images: blocks failed", repr(ex))
        # drop gray (depth-like) images
        col = [a for a in ims if np.mean(np.abs(a[..., 0].astype(int) - a[..., 2].astype(int))) > 4]
        if not col:
            return front, wrist
        if self.front_ref is not None and front is None:
            d = [np.mean(np.abs(a[40:].astype(int) - self.front_ref[40:].astype(int))) if a.shape == self.front_ref.shape
                 else 1e9 for a in col]
            fi = int(np.argmin(d))
            front = col[fi]
            others = [a for i, a in enumerate(col) if i != fi]
            if wrist is None and others:
                wrist = others[0]
        else:
            if front is None:
                front = col[0]
            if wrist is None and len(col) > 1:
                wrist = col[1]
        return front, wrist

    # ---- depth back-projection
    def pix2world(self, cam, uvs, radius=1):
        """uvs list of (u,v); returns list of np.array(3) or None.
        Primary: robomme_pixel_to_position (read-only, so retried).  Fallback: own
        back-projection from the published depth PNG + calibration of this observation."""
        if not uvs:
            return []
        out = []
        for i in range(0, len(uvs), 32):
            chunk = uvs[i:i + 32]
            sel = [{"camera": cam, "u": float(u), "v": float(v)} for u, v in chunk]
            pts = [None] * len(chunk)
            for attempt in range(3):
                try:
                    rep = self.ctx.call("robomme_pixel_to_position", observation_step_id=int(self.step_id or 0),
                                        selections=sel, depth_window_radius=int(radius))
                    pts = parse_points(reply_json(rep), len(chunk))
                    if any(p is not None for p in pts):
                        break
                except Exception as ex:
                    log("pixel_to_position failed (try %d)" % attempt, repr(ex)[:300])
                    time.sleep(0.5)
            try:
                own = self.backproject(cam, chunk)
            except Exception as ex:
                own = None
                log("backproject failed", repr(ex)[:200])
            if own is not None:
                diffs = [np.linalg.norm(a - b) for a, b in zip(pts, own) if a is not None and b is not None]
                if diffs:
                    log("backproject check %s: median diff %.4f m (n=%d)" % (cam, float(np.median(diffs)), len(diffs)))
                if not any(p is not None for p in pts):
                    log("using own back-projection for", cam)
                    pts = own
            out += pts
        return out

    def depth_and_calib(self, cam):
        """Depth (m) of the current observation for cam, plus K (3x3) and cam->world (4x4, OpenGL)."""
        key = (self.step_id, cam)
        cache = getattr(self, "_dcache", {})
        self._dcache = cache
        if key in cache:
            return cache[key]
        depth = None
        seq = self.pk.get("frame_sequence") or find_first(self.pk, "frame_sequence")
        if isinstance(seq, dict) and seq.get("manifest_path"):
            man = json.load(open(os.path.join(WS, seq["manifest_path"])))
            fr = sorted(man.get("frames", []), key=lambda f: f.get("frame_index", 0))
            if fr and (self.step_id is None or fr[-1].get("step_id") == self.step_id):
                for f in fr[-1].get("files", []):
                    if f.get("camera") == cam and f.get("kind") == "depth":
                        with Image.open(os.path.join(WS, f["path"])) as im:
                            depth = np.array(im).astype(np.float64) * float(f.get("meters_per_unit", 0.001) or 0.001)
        if depth is None:
            cache[key] = None
            return None
        K = parse_K(find_first(self.pk, "camera_intrinsics"), cam)
        T = parse_pose(find_first(self.pk, "camera_poses"), cam)
        cache[key] = None if (K is None or T is None) else (depth, K, T)
        return cache[key]

    def backproject(self, cam, uvs):
        dc = self.depth_and_calib(cam)
        if dc is None:
            return None
        depth, K, T = dc
        H, W = depth.shape[:2]
        out = []
        for u, v in uvs:
            x0, y0 = int(round(u)), int(round(v))
            win = depth[max(0, y0 - 1):y0 + 2, max(0, x0 - 1):x0 + 2]
            win = win[win > 0]
            if win.size == 0:
                out.append(None)
                continue
            d = float(np.median(win))
            pc = np.array([(u - K[0, 2]) / K[0, 0] * d, -(v - K[1, 2]) / K[1, 1] * d, -d, 1.0])
            out.append((T @ pc)[:3])
        return out


def _cam_entry(obj, cam):
    if isinstance(obj, dict):
        if cam in obj:
            return obj[cam]
        for k, v in obj.items():
            if cam in str(k).lower():
                return v
    if isinstance(obj, list):
        for v in obj:
            if isinstance(v, dict) and (v.get("camera") == cam or v.get("name") == cam):
                return v
    return None


def parse_K(obj, cam):
    e = _cam_entry(obj, cam)
    if e is None:
        return None
    if isinstance(e, dict):
        for k in ("K", "matrix", "intrinsic", "intrinsics", "intrinsic_matrix", "camera_matrix"):
            if k in e:
                return parse_K({cam: e[k]}, cam)
        if all(k in e for k in ("fx", "fy", "cx", "cy")):
            return np.array([[e["fx"], 0, e["cx"]], [0, e["fy"], e["cy"]], [0, 0, 1]], float)
        return None
    a = np.array(e, float)
    if a.size == 9:
        return a.reshape(3, 3)
    if a.size == 4:
        return np.array([[a[0], 0, a[2]], [0, a[1], a[3]], [0, 0, 1]], float)
    return None


def quat_wxyz_to_mat(q):
    w, x, y, z = [float(c) for c in q]
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def parse_pose(obj, cam):
    e = _cam_entry(obj, cam)
    if e is None:
        return None
    if isinstance(e, dict):
        p = e.get("position_m", e.get("position")) or e.get("p") or e.get("translation")
        q = e.get("quaternion_wxyz") or e.get("q") or e.get("quaternion")
        if is_vec(p, 3) and is_vec(q, 4):
            T = np.eye(4)
            T[:3, :3] = quat_wxyz_to_mat(q)
            T[:3, 3] = p
            return T
        for k in ("matrix", "cam2world", "cam2world_gl", "pose", "T"):
            if k in e:
                return parse_pose({cam: e[k]}, cam)
        return None
    a = np.array(e, float)
    if a.size == 16:
        return a.reshape(4, 4)
    return None


def load_rgb(p):
    with Image.open(p) as im:
        return np.array(im.convert("RGB"))


POINT_KEYS = ("position", "world_position", "world", "point", "xyz", "position_m", "world_point", "point_world")


def _pt_from_dict(d):
    if d.get("valid") is False:
        return None
    for k in POINT_KEYS:
        v = d.get(k)
        if is_vec(v, 3):
            return np.array(v, float)
        if isinstance(v, dict):
            if all(c in v for c in "xyz"):
                return np.array([v["x"], v["y"], v["z"]], float)
    for k, v in d.items():
        if is_vec(v, 3) and not any(s in k.lower() for s in ("pixel", "uv", "camera", "normal", "ray")):
            return np.array(v, float)
    return None


def parse_points(pk, n):
    # find a list of n dicts
    cands = []

    def rec(o, depth=0):
        if depth > 8:
            return
        if isinstance(o, list) and len(o) == n and all(isinstance(x, dict) for x in o):
            cands.append(o)
        if isinstance(o, dict):
            for v in o.values():
                rec(v, depth + 1)
        elif isinstance(o, list):
            for v in o[:64]:
                rec(v, depth + 1)
    rec(pk)
    for c in cands:
        pts = [_pt_from_dict(d) for d in c]
        if any(p is not None for p in pts):
            return pts
    # list of raw xyz lists
    for v in find_all(pk, "points") + find_all(pk, "positions"):
        if isinstance(v, list) and len(v) == n and all(is_vec(x, 3) or x is None for x in v):
            return [np.array(x, float) if x is not None else None for x in v]
    log("parse_points: unparsed reply", json.dumps(pk)[:400])
    return [None] * n


# ----------------------------------------------------------------- instruction
def parse_instruction(text):
    t = (text or "").lower()
    color = None
    for c in vision.CUBE_COLORS:
        if re.search(r"\b%s\b" % c, t):
            color = c
            break
    ordinal = None
    if re.search(r"(second|2nd)[ -]to[ -]last|penultimate|second last", t):
        ordinal = -2
    elif re.search(r"\b(last|final|most recent)\b", t):
        ordinal = "last"
    else:
        for w, k in ORDINALS.items():
            if re.search(r"\b%s\b" % w, t):
                ordinal = k
                break
        if ordinal is None:
            m = re.search(r"\b(\d+)(st|nd|rd|th)\b", t)
            if m:
                ordinal = int(m.group(1))
    return color, ordinal


# ----------------------------------------------------------------- video
def load_video(pk):
    seq = pk.get("frame_sequence") or find_first(pk, "frame_sequence")
    if not isinstance(seq, dict) or not seq.get("manifest_path"):
        raise RuntimeError("no frame_sequence manifest in packet")
    mp = os.path.join(WS, seq["manifest_path"])
    man = json.load(open(mp))
    frames = sorted(man.get("frames", []), key=lambda f: f.get("frame_index", 0))
    vid = [f for f in frames if f.get("phase") == "video_demo"]
    exe = [f for f in frames if f.get("phase") == "execution"]
    out_ims, out_t = [], []
    last = None

    def front_path(f):
        for e in f.get("files", []):
            if e.get("camera") == "front" and e.get("kind", "rgb") == "rgb":
                return e.get("path")
        return None
    for f in vid:
        p = front_path(f)
        if not p:
            continue
        try:
            out_ims.append(load_rgb(os.path.join(WS, p)))
            out_t.append(float(f.get("timestamp_s", len(out_t) * 0.05)))
        except Exception as ex:
            log("frame load failed", p, repr(ex))
    exe_front = exe_files = None
    if exe:
        exe_files = {(e.get("camera"), e.get("kind")): e for e in exe[-1].get("files", [])}
        p = front_path(exe[-1])
        if p:
            exe_front = load_rgb(os.path.join(WS, p))
    log("video frames", len(out_ims), "exec frame", exe_front is not None, "sampling", json.dumps(man.get("video_sampling", ""))[:200])
    return out_ims, out_t, exe_front, exe_files


def analyse(ims, ts, exe_front, color, ordinal):
    frames = list(ims)
    times = list(ts)
    if exe_front is not None:
        frames.append(exe_front)
        times.append((times[-1] + 0.05) if times else 0.0)
    order = np.argsort(np.asarray(times), kind="stable")
    frames = [frames[i] for i in order]
    times = [times[i] for i in order]
    if color is None:
        best = None
        for c in vision.CUBE_COLORS:
            A = va.Analysis(frames, times, log=log)
            n = len(A.detect_placements(c))
            if best is None or n > best[0]:
                best = (n, c)
        color = best[1]
        log("color guessed from video", color)
    ordn = ordinal
    res = va.solve(frames, times, color, None if ordinal in (None, "last", -2) else ordinal, log=log)
    seq = res.get("sequence", [])
    if ordinal == -2 and len(seq) >= 2:
        tid = seq[-2]["identity"]
        res["answer_home"] = res["final_ident"].index(tid)
    log("homes", [(round(h["x"]), round(h["y"])) for h in res["homes"]])
    log("swaps", [(e["i"], e["j"], round(e["t"], 2)) for e in res["swaps"]])
    log("placements", [(p["h"], round(p["t0"], 2)) for p in res["placements"]])
    log("sequence", [(s["identity"], s["home"], round(s["t"], 2)) for s in seq], "ordinal", ordn,
        "answer_home", res.get("answer_home"))
    return res, color


# ----------------------------------------------------------------- perception in live front image
def mask_pixels(mask, n=24, erode_it=1):
    m = vision.erode(mask, erode_it) if erode_it else mask
    if m.sum() < 3:
        m = mask
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return []
    idx = np.linspace(0, len(xs) - 1, min(n, len(xs))).round().astype(int)
    return [(int(xs[i]), int(ys[i])) for i in idx]


def component_mask(mask, x, y, maxd=12):
    lab, n = vision.label(mask)
    if n == 0:
        return None
    ys, xs = np.nonzero(lab)
    d = (xs - x) ** 2 + (ys - y) ** 2
    i = int(np.argmin(d))
    if d[i] > maxd ** 2:
        return None
    return lab == lab[ys[i], xs[i]]


def find_cube_front(im, color, avoid=None):
    m = vision.cube_mask(im, color)
    m[:va.TABLE_TOP_V] = False
    bl = vision.blobs(m, min_area=8, max_area=4000)
    if not bl:
        return None, None
    b = max(bl, key=lambda b: b["area"])
    return b, component_mask(m, b["x"], b["y"], maxd=3)


def target_point(robot, im, home):
    """Locate the target disc near the answer home: returns (centre xyz, table z)."""
    hx, hy = home["x"], home["y"]
    m = va.target_mask(im)
    comp = component_mask(vision.closing(m, 1), hx, hy, maxd=max(8, home["w"]))
    if comp is not None and comp.sum() >= 8:
        ys, xs = np.nonzero(comp)
        cx, cy = float(xs.mean()), float(ys.mean())
        if abs(cx - hx) > home["w"] or abs(cy - hy) > home["h"] * 1.5:
            comp = None
    pix = []
    if comp is not None:
        pix = mask_pixels(comp, 20, 1)
    # always include the home centre and a small ring
    pix = [(int(round(hx)), int(round(hy)))] + [(int(round(hx + dx)), int(round(hy + dy)))
                                                 for dx, dy in ((-3, 0), (3, 0), (0, -2), (0, 2))] + pix
    pts = [p for p in robot.pix2world("front", pix, radius=1) if p is not None]
    if not pts:
        return None, None
    P = np.array(pts)
    med = np.median(P, axis=0)
    keep = np.linalg.norm(P[:, :2] - med[:2], axis=1) < 0.06
    P = P[keep] if keep.sum() >= 3 else P
    c = np.median(P, axis=0)
    return c, float(np.median(P[:, 2]))


def top_face(pts):
    P = np.array([p for p in pts if p is not None])
    if len(P) == 0:
        return None, None
    zt = np.percentile(P[:, 2], 90)
    T = P[P[:, 2] > zt - 0.008]
    if len(T) < 2:
        T = P
    med = np.median(T, axis=0)
    T = T[np.linalg.norm(T[:, :2] - med[:2], axis=1) < 0.05]
    if len(T) == 0:
        return med, None
    return np.array([np.mean(T[:, 0]), np.mean(T[:, 1]), float(np.median(T[:, 2]))]), T


def square_yaw(T):
    """Yaw (mod 90 deg) of a square from its top-face xy points via min-area rectangle."""
    if T is None or len(T) < 8:
        return None
    xy = T[:, :2] - T[:, :2].mean(0)
    best = None
    for a in np.radians(np.arange(0, 90, 2.0)):
        c, s = math.cos(a), math.sin(a)
        u = xy @ np.array([c, s]); w = xy @ np.array([-s, c])
        area = (u.max() - u.min()) * (w.max() - w.min())
        if best is None or area < best[0]:
            best = (area, a, (u.max() - u.min()), (w.max() - w.min()))
    area, a, l1, l2 = best
    if min(l1, l2) < 0.008 or max(l1, l2) / max(min(l1, l2), 1e-6) > 1.6:
        return None
    return a


# ----------------------------------------------------------------- orientation
def rotvec_to_mat(r):
    r = np.asarray(r, float)
    th = np.linalg.norm(r)
    if th < 1e-9:
        return np.eye(3)
    k = r / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def finger_yaw(state):
    R = rotvec_to_mat(state[3:6])
    ax = R[:, 1]
    return math.atan2(ax[1], ax[0])


def wrap90(a):
    return (a + math.pi / 4) % (math.pi / 2) - math.pi / 4


# ----------------------------------------------------------------- motion
def move_to(robot, goal, grip, vmax=V_FAST, tol=0.006, max_iter=25, chunk=6, label=""):
    goal = np.asarray(goal, float)
    last_err = None
    stall = 0
    for it in range(max_iter):
        err = goal - robot.pos
        dist = float(np.linalg.norm(err))
        if dist < tol:
            break
        if last_err is not None and last_err - dist < 0.002:
            stall += 1
            if stall >= 3:
                log("move_to", label, "stalled at", np.round(robot.pos, 3), "err", round(dist, 4))
                break
        else:
            stall = 0
        last_err = dist
        rows = []
        rem = err.copy()
        for _ in range(chunk):
            v = rem / POS_SCALE
            n = float(np.max(np.abs(v)))
            if n < 1e-4:
                break
            if n > vmax:
                v = v * (vmax / n)
            rows.append([v[0], v[1], v[2], 0, 0, 0, grip])
            rem = rem - v * POS_SCALE
            if np.linalg.norm(rem) < 0.002:
                break
        robot.step(rows)
    err = float(np.linalg.norm(goal - robot.pos))
    log("move_to", label, "goal", np.round(goal, 3), "pos", np.round(robot.pos, 3), "err", round(err, 4))
    return err


def hold(robot, grip, n):
    robot.step([[0, 0, 0, 0, 0, 0, grip]] * n)


def rotate_yaw(robot, delta, grip, max_iter=8):
    """Rotate the gripper about world z by delta (rad), closed loop on measured finger yaw."""
    y0 = finger_yaw(robot.state)
    target = y0 + delta
    sign = 1.0
    for it in range(max_iter):
        e = wrap90(target - finger_yaw(robot.state))
        if abs(e) < math.radians(3):
            break
        before = abs(e)
        rows = []
        rem = e
        for _ in range(4):
            s = float(np.clip(rem / ROT_SCALE, -0.3, 0.3))
            rows.append([0, 0, 0, 0, 0, sign * s, grip])
            rem -= s * ROT_SCALE
            if abs(rem) < 1e-3:
                break
        robot.step(rows)
        after = abs(wrap90(target - finger_yaw(robot.state)))
        if after > before + math.radians(2):
            sign = -sign
            log("rotate_yaw: flipping sign")
    log("rotate_yaw delta %.1f deg, residual %.1f deg" % (math.degrees(delta),
                                                         math.degrees(wrap90(target - finger_yaw(robot.state)))))


# ----------------------------------------------------------------- main pipeline
def main(ctx):
    log("controller", VERSION, "start")
    robot = Robot(ctx)
    try:
        run(robot)
    except EpisodeOver as e:
        log("episode over:", e)
    except Exception:
        log("FATAL", traceback.format_exc())
    log("controller end; steps sent", robot.nsteps)


def run(robot):
    pk = robot.observe()
    instr = pk.get("instruction") or find_first(pk, "instruction") or ""
    log("instruction:", instr, "step_id", robot.step_id, "state", None if robot.state is None else np.round(robot.state, 3).tolist())
    color, ordinal = parse_instruction(instr)
    log("parsed color", color, "ordinal", ordinal)
    ims, ts, exe_front, exe_files = [], [], None, None
    try:
        ims, ts, exe_front, exe_files = load_video(pk)
    except Exception:
        log("load_video failed", traceback.format_exc())
    cur_front, cur_wrist = robot.images()
    if exe_front is None:
        exe_front = cur_front
    if exe_front is not None:
        robot.front_ref = exe_front
        if cur_front is None:
            cur_front = exe_front
    if cur_front is None:
        raise RuntimeError("no front image")

    # ---- ANALYSE
    answer = None
    res = None
    try:
        res, color = analyse(ims, ts, exe_front, color, ordinal)
        if res.get("answer_home") is not None:
            answer = res["homes"][res["answer_home"]]
    except Exception:
        log("analysis failed", traceback.format_exc())
    if color is None:
        color = "red"
    if answer is None:
        # fallback: any target in the live frame (homes from this frame alone)
        homes, _ = va.find_homes([va.target_mask(cur_front)], min_support=0.5)
        if res is not None and res.get("homes"):
            homes = res["homes"]
        if res is not None and res.get("placements"):
            answer = res["homes"][res["placements"][-1]["h"]]
        elif homes:
            answer = homes[0]
        log("fallback answer", None if answer is None else (answer["x"], answer["y"]))
    if answer is None:
        raise RuntimeError("no target found")
    log("ANSWER target pixel", round(answer["x"], 1), round(answer["y"], 1), "elapsed", round(time.time() - T0, 1))

    # ---- LOCATE target and cube (front camera, current observation)
    tgt, table_z = target_point(robot, cur_front, answer)
    log("target world", None if tgt is None else np.round(tgt, 4), "table_z", table_z)
    cube, cube_T, cube_h = locate_cube_front(robot, cur_front, color, table_z)
    if tgt is None or cube is None:
        raise RuntimeError("localization failed: target %s cube %s" % (tgt, cube))
    if table_z is None:
        table_z = cube[2] - 0.04

    # ---- PICK
    grasped = False
    for attempt in range(MAX_PICK_TRIES):
        if time.time() - T0 > TIME_BUDGET_S:
            break
        try:
            grasped, cube_h = pick(robot, cube, cube_h, table_z, color, attempt)
        except EpisodeOver:
            raise
        except Exception:
            log("pick exception", traceback.format_exc())
            grasped = False
        if grasped:
            break
        # re-locate cube from a fresh observation
        try:
            hold(robot, -1, 3)
            move_to(robot, robot.pos + np.array([0, 0, max(0.0, table_z + SAFE_DZ - robot.pos[2])]), -1, label="retreat")
            f, _ = robot.images()
            if f is not None:
                c2, T2, h2 = locate_cube_front(robot, f, color, table_z)
                if c2 is not None:
                    cube, cube_h = c2, h2
        except EpisodeOver:
            raise
        except Exception:
            log("relocate exception", traceback.format_exc())
    log("grasped", grasped)

    # ---- PLACE (also if grasp uncertain: releasing over the target is harmless)
    place(robot, tgt, table_z, cube_h)
    log("done; elapsed", round(time.time() - T0, 1))


def locate_cube_front(robot, im, color, table_z):
    b, comp = find_cube_front(im, color)
    if b is None:
        log("cube not found in front image")
        return None, None, 0.04
    pix = mask_pixels(comp, 24, 1)
    pts = robot.pix2world("front", pix, radius=0)
    c, T = top_face(pts)
    h = 0.04
    if c is not None and table_z is not None:
        h = float(np.clip(c[2] - table_z, 0.02, 0.07))
    log("cube front blob", (round(b["x"], 1), round(b["y"], 1), b["area"]), "top", None if c is None else np.round(c, 4), "h", round(h, 3))
    return c, T, h


def pick(robot, cube, cube_h, table_z, color, attempt):
    grip = -1
    hover = np.array([cube[0], cube[1], table_z + cube_h + HOVER_DZ])
    hold(robot, grip, 4)
    move_to(robot, hover, grip, vmax=V_FAST, tol=0.008, label="hover")
    yaw = None
    # wrist refine
    try:
        _, wrist = robot.images()
        if wrist is not None:
            m = vision.cube_mask(wrist, color)
            bl = vision.blobs(m, min_area=30, max_area=256 * 256)
            if bl:
                b = max(bl, key=lambda b: b["area"])
                comp = component_mask(m, b["x"], b["y"], maxd=3)
                pts = robot.pix2world("wrist", mask_pixels(comp, 32, 2), radius=0)
                c, T = top_face(pts)
                if c is not None and np.linalg.norm(c[:2] - cube[:2]) < 0.06:
                    log("wrist refine: cube top", np.round(c, 4), "shift", np.round(c[:2] - cube[:2], 4))
                    h_w = float(np.clip(c[2] - table_z, 0.02, 0.07))
                    if abs(h_w - cube_h) < 0.02:
                        cube_h = h_w
                    cube = np.array([c[0], c[1], c[2]])
                    yaw = square_yaw(T)
                else:
                    log("wrist refine rejected", None if c is None else np.round(c, 4))
            else:
                log("wrist: no cube blob")
    except EpisodeOver:
        raise
    except Exception:
        log("wrist refine exception", traceback.format_exc())
    if yaw is not None:
        delta = wrap90(yaw - finger_yaw(robot.state))
        log("cube yaw %.1f deg, finger yaw %.1f, delta %.1f" % (math.degrees(yaw), math.degrees(finger_yaw(robot.state)),
                                                               math.degrees(delta)))
        if abs(delta) > math.radians(8):
            rotate_yaw(robot, delta, grip)
    hover = np.array([cube[0], cube[1], table_z + cube_h + HOVER_DZ * 0.6])
    move_to(robot, hover, grip, vmax=V_FAST, tol=0.006, label="hover2")
    gz = table_z + max(GRASP_FRAC * cube_h, 0.012)
    if attempt >= 1:
        gz = table_z + max(0.4 * cube_h, 0.010)
    move_to(robot, [cube[0], cube[1], gz], grip, vmax=V_SLOW, tol=0.005, label="descend")
    hold(robot, +1, CLOSE_STEPS)
    op = robot.opening
    log("after close opening", round(op, 4))
    lift = np.array([robot.pos[0], robot.pos[1], table_z + SAFE_DZ])
    move_to(robot, lift, +1, vmax=V_SLOW * 2, tol=0.01, label="lift")
    op2 = robot.opening
    ok = op2 > MIN_OPENING
    # visual check in wrist/front: cube near gripper
    log("after lift opening", round(op2, 4), "grasp ok" if ok else "MISS")
    return ok, cube_h


def place(robot, tgt, table_z, cube_h):
    grip = +1
    safe_z = max(table_z + SAFE_DZ, robot.pos[2])
    move_to(robot, [tgt[0], tgt[1], safe_z], grip, vmax=V_FAST, tol=0.006, label="transport")
    move_to(robot, [tgt[0], tgt[1], tgt[2] + 0.5 * cube_h + 0.05], grip, vmax=V_FAST * 0.8, tol=0.006, label="pre-place")
    pz = tgt[2] + 0.5 * cube_h + PLACE_CLEAR
    move_to(robot, [tgt[0], tgt[1], pz], grip, vmax=V_SLOW, tol=0.004, label="place")
    hold(robot, -1, OPEN_STEPS)
    move_to(robot, [robot.pos[0], robot.pos[1], robot.pos[2] + 0.10], -1, vmax=V_SLOW * 2, tol=0.01, label="retreat")
    hold(robot, -1, 5)
