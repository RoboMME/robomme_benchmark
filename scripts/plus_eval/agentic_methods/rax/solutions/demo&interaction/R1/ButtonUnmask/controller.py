"""ButtonUnmask controller (RAX, frozen bundle).

Pipeline / transition table
  INIT      observe, parse instruction, find button + table height (wrist cam, home pose)
            -> PRESS (always; on failure to find the button: PRESS uses front fallback / skips)
  PRESS     close gripper, go above button, push down, rise, return home (open gripper)
            -> WATCH
  WATCH     hold at home; after every motion chunk detect coloured cubes in front+wrist
            images, back-project target colours with robomme_pixel_to_position and cluster
            in world xy.  Ends at step >= WATCH_MIN_STEP with no coloured blob for
            WATCH_QUIET steps, or WATCH_MAX_STEP.  -> PICKS
  PICKS     for each target (instruction order; "all ... cubes" = every cluster):
            hover above recorded xy, segment container tops in wrist view, choose the one
            nearest, align yaw (image-space, sign learned), centre, descend, close,
            verify finger opening, lift, (not last) put back + release.
            failure -> retry (max PICK_RETRIES) -> next target.
  END       when the episode reports ended / steps exhausted, return.
Every phase is wrapped in try/except; main() never raises.
"""
import json
import math
import os
import re
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mcp_io  # noqa: E402
import perception as P  # noqa: E402

VERSION = "bu-v2"
OUT_DIR = "/workspace/output"

POS_SCALE = 0.05     # metres per unit action
ROT_SCALE = 0.5      # rad per unit action
V_FAST = 0.03        # m/step free-space transit  (untested live)
V_MED = 0.015        # m/step near objects
V_SLOW = 0.008       # m/step final descent / press
PRESS_ABOVE = 0.05   # m above cap before pushing
PRESS_DEPTH = 0.014  # m commanded below measured cap top
WATCH_MIN_STEP = 140  # demos: last reveal covered by ~frame 135
WATCH_QUIET = 15
WATCH_MAX_STEP = 185
HOVER_ABOVE_TABLE = 0.20
APPROACH_ABOVE_TOP = 0.07
GRASP_BELOW_TOP = 0.025
LIFT_HEIGHT = 0.14
CLUSTER_R = 0.035
MERGE_R = 0.04
MIN_CLUSTER_N = 2
HOLD_MIN_FINGERS = 0.008   # finger sum above which something is held
PICK_RETRIES = 2
YAW_CUM_MAX = math.radians(110)  # keep wrist joint away from its limits


WALL_LIMIT_S = 1080.0   # controller wall limit is 1200 s


class EpisodeEnded(Exception):
    pass


class Log:
    def __init__(self):
        self.t0 = time.time()
        self.f = None
        try:
            os.makedirs(OUT_DIR, exist_ok=True)
            self.f = open(os.path.join(OUT_DIR, "controller_log.txt"), "a", buffering=1)
        except Exception:
            self.f = None

    def __call__(self, *a):
        s = "[%7.1fs] " % (time.time() - self.t0) + " ".join(str(x) for x in a)
        try:
            print(s, flush=True)
        except Exception:
            pass
        try:
            self.f.write(s + "\n")
        except Exception:
            pass

    def dump(self, name, obj):
        try:
            with open(os.path.join(OUT_DIR, name), "w") as f:
                json.dump(obj, f, indent=1, default=str)
        except Exception as ex:
            self(f"dump {name} failed {ex!r}")


def _short(meta, limit=3000):
    try:
        s = json.dumps(meta, default=str)
    except Exception:
        s = str(meta)
    return s[:limit]


# ----------------------------------------------------------------- rotation utils
def rotvec_to_mat(r):
    r = np.asarray(r, float)
    th = np.linalg.norm(r)
    if th < 1e-9:
        return np.eye(3)
    k = r / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def mat_to_rotvec(R):
    c = (np.trace(R) - 1) / 2
    c = min(1.0, max(-1.0, c))
    th = math.acos(c)
    if th < 1e-7:
        return np.zeros(3)
    if th > math.pi - 1e-4:
        # near pi: axis from the diagonal
        A = (R + np.eye(3)) / 2
        k = np.sqrt(np.clip(np.diag(A), 0, None))
        i = int(np.argmax(k))
        k[i] = max(k[i], 1e-9)
        for j in range(3):
            if j != i:
                k[j] = A[i, j] / k[i]
        k = k / np.linalg.norm(k)
        return k * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return w * th


def rotz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


# ----------------------------------------------------------------- robot wrapper
class Robot:
    def __init__(self, ctx, log):
        self.ctx = ctx
        self.log = log
        self.obs = None
        self.grip = -1.0
        self.max_steps = None
        self.calls = 0
        self.on_obs = None          # hook(obs) after every motion reply
        self.R_des = None           # desired orientation (world)
        self.R0 = None              # initial orientation
        self.orient_ok = True       # world-frame orientation correction trusted
        self.orient_err_hist = []
        self.dumped = 0
        self.yaw_resp = 1.0

    # -- calls
    def _record(self, tool, res):
        if self.dumped < 4:
            self.dumped += 1
            self.log.dump("reply_%s_%d.json" % (tool, self.dumped), {
                "meta": mcp_io.meta_of(res),
                "blocks": [{k: (v if k != "data" else "<%d b64 chars>" % len(v)) for k, v in b.items()}
                           for b in res.get("content", [])]})

    def observe(self):
        res = self.ctx.call("robomme_observe")
        self.calls += 1
        self._record("observe", res)
        self.obs = mcp_io.parse_obs(res, self.obs)
        return self.obs

    @property
    def step_id(self):
        return self.obs.step_id if self.obs and self.obs.step_id is not None else 0

    @property
    def pos(self):
        return np.array(self.obs.state[0:3], float)

    @property
    def rotvec(self):
        return np.array(self.obs.state[3:6], float)

    @property
    def fingers(self):
        return float(self.obs.state[6] + self.obs.state[7])

    def remaining(self):
        if self.max_steps is None:
            return 10_000
        return self.max_steps - self.step_id

    def step(self, rows):
        if self.obs is not None and self.obs.ended:
            raise EpisodeEnded()
        if time.time() - self.log.t0 > WALL_LIMIT_S:
            self.log("wall-clock guard reached; stopping")
            raise EpisodeEnded()
        rem = self.remaining()
        if rem <= 0:
            raise EpisodeEnded()
        rows = rows[:max(1, min(100, rem))]
        rows = [[float(max(-1.0, min(1.0, x))) for x in r] for r in rows]
        try:
            res = self.ctx.call("robomme_step", actions=rows)
        except Exception as ex:
            self.log("step error", repr(ex)[:400])
            # do not retry blindly: resync
            self.observe()
            if self.obs.ended:
                raise EpisodeEnded()
            return self.obs
        self.calls += 1
        self._record("step", res)
        prev = self.obs
        self.obs = mcp_io.parse_obs(res, prev)
        if self.obs.step_id is None and prev is not None and prev.step_id is not None:
            self.obs.step_id = prev.step_id + len(rows)
        if self.obs.complete:
            self.log("task_complete reported at step", self.step_id)
        if self.on_obs is not None:
            try:
                self.on_obs(self.obs)
            except Exception as ex:
                self.log("on_obs error", repr(ex), traceback.format_exc()[-600:])
        if self.obs.ended:
            self.log("episode ended at step", self.step_id, "complete", self.obs.complete)
            raise EpisodeEnded()
        return self.obs

    # -- orientation correction (world-frame rotvec deltas, with divergence guard)
    def _rot_cmd(self):
        if self.R_des is None or not self.orient_ok:
            return [0.0, 0.0, 0.0]
        R = rotvec_to_mat(self.rotvec)
        e = mat_to_rotvec(self.R_des @ R.T)
        n = float(np.linalg.norm(e))
        self.orient_err_hist.append(n)
        if len(self.orient_err_hist) > 6:
            h = self.orient_err_hist[-6:]
            if h[-1] > 0.25 and h[-1] > h[0] + 0.08:
                self.log("orientation correction diverging; disabled", ["%.3f" % x for x in h])
                self.orient_ok = False
                return [0.0, 0.0, 0.0]
        if n < 0.01:
            return [0.0, 0.0, 0.0]
        step = min(n, 0.12) * 0.6
        d = e / n * step
        return list(d / ROT_SCALE)

    def yaw_error(self):
        """Signed yaw (rad) from current orientation to R_des about world z."""
        if self.R_des is None:
            return 0.0
        R = rotvec_to_mat(self.rotvec)
        E = self.R_des @ R.T
        return math.atan2(E[1, 0], E[0, 0])

    def hold(self, n, grip=None):
        if grip is not None:
            self.grip = grip
        rows = [[0, 0, 0] + self._rot_cmd() + [self.grip] for _ in range(n)]
        return self.step(rows)

    def gripper(self, cmd, n=8):
        self.grip = cmd
        return self.hold(n)

    def move_to(self, target, vmax=V_MED, tol=0.004, max_steps=80, chunk=6, grip=None,
                stall_ok=False):
        """Closed-loop Cartesian move. Returns (reached, final_error_m)."""
        if grip is not None:
            self.grip = grip
        target = np.asarray(target, float)
        used = 0
        stalls = 0
        last = None
        while used < max_steps:
            err = target - self.pos
            dist = float(np.linalg.norm(err))
            if dist <= tol:
                return True, dist
            nneed = max(1, int(math.ceil(dist / vmax)))
            n = min(chunk, nneed, max_steps - used)
            d = err / nneed
            rot = self._rot_cmd()
            rows = [list(d / POS_SCALE) + rot + [self.grip] for _ in range(n)]
            p0 = self.pos.copy()
            self.step(rows)
            used += n
            moved = float(np.linalg.norm(self.pos - p0))
            expect = float(np.linalg.norm(d)) * n
            if expect > 0.004 and moved < 0.25 * expect:
                stalls += 1
                if stalls >= 2:
                    if not stall_ok:
                        self.log("move stall: target", np.round(target, 3), "pos", np.round(self.pos, 3))
                    return False, float(np.linalg.norm(target - self.pos))
            else:
                stalls = 0
            last = dist
        return False, float(np.linalg.norm(target - self.pos))

    def rotate_yaw(self, dpsi, grip=None, max_rate=0.25):
        """Rotate about world z by dpsi (rad) by moving R_des and servoing.
        Falls back to raw drz rows if world-frame servoing does not converge."""
        if grip is not None:
            self.grip = grip
        if self.R_des is None:
            self.R_des = rotvec_to_mat(self.rotvec)
        y_before = self.yaw_from_start()
        self.R_des = rotz(dpsi) @ self.R_des
        steps = 0
        try:
            steps = self._rotate_yaw_inner(dpsi, max_rate)
        finally:
            if abs(dpsi) > math.radians(5):
                dy = self.yaw_from_start() - y_before
                dy = (dy + math.pi) % (2 * math.pi) - math.pi
                r = dy / dpsi
                self.log("yaw response %.2f (cmd %.1f deg, measured %.1f deg)" % (r, math.degrees(dpsi), math.degrees(dy)))
                if abs(r) > 0.3:
                    self.yaw_resp = 1.0 if r > 0 else -1.0
        return steps

    def _rotate_yaw_inner(self, dpsi, max_rate):
        steps = 0
        if self.orient_ok:
            n0 = None
            probe = True
            while steps < 30:
                e = mat_to_rotvec(self.R_des @ rotvec_to_mat(self.rotvec).T)
                n = float(np.linalg.norm(e))
                if n0 is None:
                    n0 = n
                if n < 0.02:
                    return steps
                if not probe and n > n_prev - 0.02 and n > 0.1:
                    self.log("yaw servo not converging (%.3f -> %.3f); switching to raw drz" % (n_prev, n))
                    self.orient_ok = False
                    break
                rate = 0.08 if probe else max_rate
                k = max(1, int(math.ceil(n / rate)))
                m = min(k, 2 if probe else 3)
                d = e / k
                rows = [[0, 0, 0] + list(d / ROT_SCALE) + [self.grip] for _ in range(m)]
                n_prev = n
                self.step(rows)
                steps += m
                probe = False
            if self.orient_ok:
                return steps
            # undo with raw commands is not attempted; re-base on measurement
            self.R_des = rotvec_to_mat(self.rotvec)
            return steps
        k = max(1, int(math.ceil(abs(dpsi) / max_rate)))
        rows = [[0, 0, 0, 0, 0, dpsi / k / ROT_SCALE, self.grip] for _ in range(k)]
        self.step(rows)
        self.R_des = rotvec_to_mat(self.rotvec)
        return steps + k

    def yaw_from_start(self):
        if self.R0 is None:
            return 0.0
        R = rotvec_to_mat(self.rotvec)
        E = R @ self.R0.T
        return math.atan2(E[1, 0], E[0, 0])


# ----------------------------------------------------------------- instruction
COLOR_WORDS = P.COLOR_NAMES + ["white", "black", "gray", "grey", "brown", "magenta", "violet"]
COLOR_ALIAS = {"violet": "purple", "magenta": "pink", "grey": "gray"}


def parse_targets(instr):
    """Return list of (color, all_flag) in instruction order."""
    s = (instr or "").lower()
    out = []
    pat = re.compile(r"(all\s+(?:of\s+)?(?:the\s+)?containers?\s+(?:that\s+are\s+)?hiding\s+(?:the\s+)?(\w+)\s+cubes?)"
                     r"|(hiding\s+(?:the\s+|a\s+|an\s+)?(\w+)\s+(cubes?))")
    for m in pat.finditer(s):
        if m.group(1):
            c, allf = m.group(2), True
        else:
            c, allf = m.group(4), m.group(5).endswith("s")
        c = COLOR_ALIAS.get(c, c)
        if c in COLOR_WORDS:
            out.append((c, allf))
    if not out:
        for m in re.finditer(r"\b(" + "|".join(COLOR_WORDS) + r")\b", s):
            out.append((COLOR_ALIAS.get(m.group(1), m.group(1)), False))
    return out


# ----------------------------------------------------------------- controller
class Controller:
    def __init__(self, ctx):
        self.log = Log()
        self.rb = Robot(ctx, self.log)
        self.ctx = ctx
        self.targets = []
        self.target_colors = set()
        self.dets = []            # world detections {color,xyz,cam,step}
        self.pending_px = []      # (cam,u,v,color) from video frames, query at current step
        self.table_z = None
        self.button = None        # world xyz of cap top
        self.home = None
        self.last_blob_step = -1
        self.watch_active = True
        self.summary = {"version": VERSION, "events": []}

    def ev(self, *a):
        self.log(*a)
        self.summary["events"].append(" ".join(str(x) for x in a))

    # -------------------------------------------------------- geometry
    def p2w(self, sels, radius=1):
        """sels: list of (camera,u,v). Returns list of (xyz|None, spread)."""
        out = []
        for i in range(0, len(sels), 32):
            chunk = sels[i:i + 32]
            try:
                res = self.ctx.call("robomme_pixel_to_position",
                                    observation_step_id=int(self.rb.step_id),
                                    selections=[{"camera": c, "u": float(u), "v": float(v)} for c, u, v in chunk],
                                    depth_window_radius=int(radius))
                if not getattr(self, "_p2w_dumped", False):
                    self._p2w_dumped = True
                    self.log.dump("reply_pixel_to_position.json", {"meta": mcp_io.meta_of(res)})
                out.extend(mcp_io.parse_points(res, len(chunk)))
            except Exception as ex:
                self.log("p2w error", repr(ex)[:300])
                out.extend([(None, None)] * len(chunk))
        return out

    # -------------------------------------------------------- init
    def init(self):
        o = self.rb.observe()
        self.log("VERSION", VERSION, "step", o.step_id, "notes", o.notes)
        self.log("meta keys", list(o.meta.keys()))
        self.log("instruction:", o.instruction)
        self.log("state:", o.state)
        try:
            st = self.ctx.call("robomme_status")
            sm = mcp_io.meta_of(st)
            self.log.dump("robomme_status.json", sm)
            self.rb.max_steps = self._find_limit(sm) or self._find_limit(o.meta)
        except Exception as ex:
            self.log("status error", repr(ex))
        self.log("max_steps", self.rb.max_steps, "limits", o.limits)
        self.targets = parse_targets(o.instruction)
        self.target_colors = set(c for c, _ in self.targets)
        self.ev("targets", self.targets)
        self.home = self.rb.pos.copy()
        self.rb.R_des = rotvec_to_mat(self.rb.rotvec)
        self.rb.R0 = self.rb.R_des.copy()
        self.rb.on_obs = self.on_obs
        # video frames in the reset manifest (if any)
        try:
            self.scan_video(o)
        except Exception as ex:
            self.log("scan_video error", repr(ex))
        self.locate_button_and_table()
        self.detect_now(o)

    def _find_limit(self, meta):
        for k in ("max_steps", "step_limit", "max_episode_steps", "episode_step_limit", "horizon"):
            v = mcp_io.find_key(meta, k, lambda v: isinstance(v, int) and not isinstance(v, bool) and v > 0)
            if v:
                return int(v)
        lim = mcp_io.find_key(meta, "episode_limits")
        if isinstance(lim, dict):
            for k, v in lim.items():
                if isinstance(v, int) and not isinstance(v, bool) and v > 50 and "step" in k and "remaining" not in k:
                    return int(v)
            rem = lim.get("remaining_steps")
            if isinstance(rem, int):
                return int(rem) + int(self.rb.step_id)
        return None

    def scan_video(self, o):
        if not o.manifest_path:
            self.log("no frame_sequence manifest")
            return
        frames = mcp_io.read_manifest(o.manifest_path)
        vid = [f for f in frames if f.get("phase") == "video_demo"]
        self.log("reset manifest frames", len(frames), "video", len(vid))
        if not vid:
            return
        seen = {}
        for f in vid[::2]:
            files = mcp_io.frame_files(f)
            if ("front", "rgb") not in files:
                continue
            rgb = mcp_io.load_file(files[("front", "rgb")][0], "rgb")
            if rgb is None:
                continue
            for d in P.detect_cubes(rgb, "front", colors=list(self.target_colors & set(P.COLOR_NAMES))):
                key = (d["color"], d["u"] // 4, d["v"] // 4)
                seen.setdefault(key, (d["color"], d["u"], d["v"]))
        for color, u, v in list(seen.values())[:60]:
            self.pending_px.append(("front", u, v, color))
        self.ev("video cube pixels", len(self.pending_px))
        if self.pending_px:
            pts = self.p2w([(c, u, v) for c, u, v, _ in self.pending_px])
            for (c, u, v, color), (xyz, sp) in zip(self.pending_px, pts):
                if xyz is not None:
                    self.dets.append({"color": color, "xyz": xyz, "cam": "video", "step": -1, "w": 0.5})

    def locate_button_and_table(self):
        o = self.rb.obs
        wr = o.rgb.get("wrist")
        fr = o.rgb.get("front")
        sels = []
        b = P.detect_button_wrist(wr) if wr is not None else None
        self.log("button wrist px", b)
        # table: wood pixels in wrist and front images
        tsel = []
        if wr is not None:
            hsv = P.hsv_of(wr)
            wm = P.wood_mask(hsv)
            for (u, v) in [(40, 60), (128, 40), (215, 60), (60, 130), (200, 130), (90, 100), (170, 100)]:
                if wm[v, u]:
                    tsel.append(("wrist", u, v))
        if fr is not None:
            hsv = P.hsv_of(fr)
            wm = P.wood_mask(hsv)
            for (u, v) in [(40, 200), (128, 220), (215, 200), (60, 160), (200, 160)]:
                if wm[v, u]:
                    tsel.append(("front", u, v))
        if b is not None:
            bu, bv = b["u"], b["v"]
            sels = [("wrist", bu, bv), ("wrist", bu - 2, bv), ("wrist", bu + 2, bv),
                    ("wrist", bu, bv - 2), ("wrist", bu, bv + 2)]
        pts = self.p2w(sels + tsel)
        bp = [p for p, _ in pts[:len(sels)] if p is not None]
        tp = [p for p, _ in pts[len(sels):] if p is not None]
        if tp:
            self.table_z = float(np.median([p[2] for p in tp]))
        self.log("table pts", [np.round(p, 3).tolist() for p in tp])
        if bp:
            arr = np.array(bp)
            self.button = np.array([np.median(arr[:, 0]), np.median(arr[:, 1]), np.max(arr[:, 2])])
        if self.table_z is None and self.button is not None:
            self.table_z = float(self.button[2] - 0.03)
        plausible = (self.button is not None and self.table_z is not None and
                     0.003 < self.button[2] - self.table_z < 0.10)
        if not plausible and fr is not None:
            self.log("wrist button implausible/missing", self.button, "-> front fallback")
            bf = P.detect_button_front(fr)
            self.log("button front px", bf)
            if bf is not None:
                pf = self.p2w([("front", bf["u"], bf["v"]), ("front", bf["u"], bf["v"] - 2),
                               ("front", bf["u"] - 2, bf["v"]), ("front", bf["u"] + 2, bf["v"])])
                pf = [p for p, _ in pf if p is not None]
                if pf:
                    arr = np.array(pf)
                    cand = np.array([np.median(arr[:, 0]), np.median(arr[:, 1]), np.max(arr[:, 2])])
                    if self.table_z is None or 0.003 < cand[2] - self.table_z < 0.10:
                        self.button = cand
        self.ev("button", None if self.button is None else np.round(self.button, 4).tolist(),
                "table_z", self.table_z, "home", np.round(self.home, 4).tolist())

    # -------------------------------------------------------- detection hook
    def on_obs(self, o):
        if self.watch_active:
            self.detect_now(o)

    def detect_now(self, o):
        sels, meta = [], []
        any_blob = False
        tc = list(self.target_colors & set(P.COLOR_NAMES))
        for cam in ("front", "wrist"):
            rgb = o.rgb.get(cam)
            if rgb is None:
                continue
            ign = P.wrist_gripper_mask(*rgb.shape[:2]) if cam == "wrist" else None
            ds = P.detect_cubes(rgb, cam, ignore=ign)
            if ds:
                any_blob = True
            for d in ds:
                if d["color"] in tc:
                    sels.append((cam, d["u"], d["v"]))
                    meta.append((cam, d))
        if any_blob:
            self.last_blob_step = self.rb.step_id
        if not sels:
            return
        sels, meta = sels[:32], meta[:32]
        pts = self.p2w(sels, radius=1)
        for (cam, d), (xyz, sp) in zip(meta, pts):
            if xyz is None:
                continue
            if sp is not None and sp > 0.03:
                continue
            if self.table_z is not None and not (-0.015 < xyz[2] - self.table_z < 0.07):
                continue
            if self.button is not None and math.hypot(xyz[0] - self.button[0], xyz[1] - self.button[1]) < 0.025:
                continue
            self.dets.append({"color": d["color"], "xyz": xyz, "cam": cam, "step": self.rb.step_id,
                              "w": 1.0, "area": d["area"]})

    def clusters(self):
        cl = []
        for d in self.dets:
            best, bd = None, 1e9
            for c in cl:
                if c["color"] != d["color"]:
                    continue
                dist = math.hypot(c["x"] - d["xyz"][0], c["y"] - d["xyz"][1])
                if dist < bd:
                    best, bd = c, dist
            if best is not None and bd < CLUSTER_R:
                best["pts"].append(d)
                xs = np.array([p["xyz"] for p in best["pts"]])
                best["x"], best["y"] = float(np.median(xs[:, 0])), float(np.median(xs[:, 1]))
                best["z"] = float(np.median(xs[:, 2]))
            else:
                cl.append({"color": d["color"], "x": d["xyz"][0], "y": d["xyz"][1], "z": d["xyz"][2], "pts": [d]})
        # merge clusters that drifted together
        merged = []
        for c in cl:
            for m in merged:
                if m["color"] == c["color"] and math.hypot(m["x"] - c["x"], m["y"] - c["y"]) < MERGE_R:
                    m["pts"] += c["pts"]
                    xs = np.array([p["xyz"] for p in m["pts"]])
                    m["x"], m["y"] = float(np.median(xs[:, 0])), float(np.median(xs[:, 1]))
                    break
            else:
                merged.append(c)
        for c in merged:
            c["n"] = sum(p["w"] for p in c["pts"])
            c["cams"] = sorted(set(p["cam"] for p in c["pts"]))
        return merged

    # -------------------------------------------------------- press
    def press(self):
        rb = self.rb
        if self.button is None:
            self.ev("PRESS skipped: no button")
            return False
        bx, by, bz = self.button
        above = np.array([bx, by, bz + PRESS_ABOVE])
        # rise slightly first if home is below the 'above' point
        ok, e = rb.move_to(above, vmax=V_FAST, tol=0.006, max_steps=30, chunk=6, grip=1.0)
        self.log("press above reached", ok, "err %.4f" % e, "step", rb.step_id, "fingers %.4f" % rb.fingers)
        ok2, e2 = rb.move_to([bx, by, bz + 0.012], vmax=V_MED, tol=0.004, max_steps=10, chunk=3)
        z_before = rb.pos[2]
        ok3, e3 = rb.move_to([bx, by, bz - PRESS_DEPTH], vmax=V_SLOW, tol=0.003, max_steps=8, chunk=2,
                             stall_ok=True)
        rb.step([[0, 0, -0.15] + rb._rot_cmd() + [1.0] for _ in range(3)])
        zmin = rb.pos[2]
        self.ev("PRESS at step", rb.step_id, "cap_z %.4f" % bz, "tcp_z %.4f" % zmin,
                "below_cap %.4f" % (bz - zmin), "xy_err %.4f" % math.hypot(rb.pos[0] - bx, rb.pos[1] - by))
        rb.move_to(above, vmax=V_MED, tol=0.008, max_steps=8, chunk=4)
        return True

    def go_home(self):
        rb = self.rb
        rb.move_to(self.home, vmax=V_FAST, tol=0.006, max_steps=30, chunk=6, grip=-1.0)
        self.log("home reached step", rb.step_id, "pos", np.round(rb.pos, 3).tolist())

    # -------------------------------------------------------- watch
    def watch(self):
        rb = self.rb
        while True:
            s = rb.step_id
            if s >= WATCH_MAX_STEP:
                break
            if s >= WATCH_MIN_STEP and s - self.last_blob_step >= WATCH_QUIET:
                break
            rb.hold(5, grip=-1.0)
        self.watch_active = False
        cl = self.clusters()
        self.summary["clusters"] = [{k: v for k, v in c.items() if k != "pts"} for c in cl]
        self.ev("WATCH done step", rb.step_id, "last_blob", self.last_blob_step, "clusters",
                [(c["color"], round(c["x"], 3), round(c["y"], 3), round(c["n"], 1), c["cams"]) for c in cl])
        return cl

    # -------------------------------------------------------- picks
    def plan(self, cl):
        plan = []
        used = set()
        cur = self.rb.pos[:2].copy()
        for color, allf in self.targets:
            cs = [c for c in cl if c["color"] == color and id(c) not in used]
            strong = [c for c in cs if c["n"] >= MIN_CLUSTER_N]
            if allf and strong:
                nmax = max(c["n"] for c in strong)
                strong = [c for c in strong if c["n"] >= max(3, 0.2 * nmax)]
            cs = strong or cs
            if not cs:
                self.ev("no cluster for", color)
                plan.append({"color": color, "xy": None})
                continue
            if allf:
                rest = list(cs)
                while rest:
                    c = min(rest, key=lambda c: math.hypot(c["x"] - cur[0], c["y"] - cur[1]))
                    rest.remove(c)
                    used.add(id(c))
                    plan.append({"color": color, "xy": (c["x"], c["y"]), "n": c["n"]})
                    cur = np.array([c["x"], c["y"]])
            else:
                c = max(cs, key=lambda c: c["n"])
                used.add(id(c))
                plan.append({"color": color, "xy": (c["x"], c["y"]), "n": c["n"]})
                cur = np.array([c["x"], c["y"]])
        self.ev("PLAN", [(p["color"], None if p["xy"] is None else (round(p["xy"][0], 3), round(p["xy"][1], 3)),
                          p.get("n")) for p in plan])
        return plan

    def find_container(self, xy, max_dist=0.06, with_axes=False):
        """Segment container tops in current wrist view; pick nearest to xy.
        Returns dict with world top centre, image angle, etc. or None."""
        o = self.rb.obs
        wr = o.rgb.get("wrist")
        if wr is None:
            return None
        tops = P.container_tops(wr, None)
        if not tops:
            self.log("find_container: no tops segmented")
            return None
        tops = sorted(tops, key=lambda t: -t["area"])[:30]
        pts = self.p2w([("wrist", t["u"], t["v"]) for t in tops], radius=1)
        best, bd = None, 1e9
        cands = []
        for t, (xyz, sp) in zip(tops, pts):
            if xyz is None:
                continue
            if self.table_z is not None and xyz[2] - self.table_z < 0.015:
                continue
            if self.button is not None and math.hypot(xyz[0] - self.button[0], xyz[1] - self.button[1]) < 0.03:
                continue
            t["xyz"] = xyz
            cands.append(t)
            dist = math.hypot(xyz[0] - xy[0], xyz[1] - xy[1])
            if dist < bd:
                best, bd = t, dist
        if best is None or bd > max_dist:
            self.log("find_container: none near", np.round(xy, 3).tolist(), "best dist", round(bd, 3),
                     "n", len(cands))
            return None
        best["dist"] = bd
        best["neighbors"] = [c["xyz"] for c in cands if c is not best]
        best["axis_u"] = None
        best["edges"] = None
        if with_axes:
            try:
                u, v = best["u"], best["v"]
                box = __import__("cv2").boxPoints(best["rect"])
                e1 = np.array(box[1] - box[0], float)
                e1 = e1 / max(1e-6, np.linalg.norm(e1))
                L = 7.0
                sel = [("wrist", u - L, v), ("wrist", u + L, v),
                       ("wrist", u - L * e1[0], v - L * e1[1]), ("wrist", u + L * e1[0], v + L * e1[1])]
                sel = [(c, float(min(255, max(0, uu))), float(min(255, max(0, vv)))) for c, uu, vv in sel]
                ax = self.p2w(sel, radius=0)
                if all(p[0] is not None for p in ax):
                    au = np.array(ax[1][0][:2]) - np.array(ax[0][0][:2])
                    ae = np.array(ax[3][0][:2]) - np.array(ax[2][0][:2])
                    if np.linalg.norm(au) > 1e-4 and np.linalg.norm(ae) > 1e-4:
                        au, ae = au / np.linalg.norm(au), ae / np.linalg.norm(ae)
                        best["axis_u"] = au
                        best["edges"] = (ae, np.array([-ae[1], ae[0]]))
            except Exception as ex:
                self.log("axes error", repr(ex))
        return best

    def clearance(self, center, axis, neighbors):
        """Min distance from neighbour centres to the two finger pads along axis."""
        if axis is None:
            return 1.0
        c = np.array(center[:2])
        pads = [c + 0.05 * axis, c - 0.05 * axis]
        dmin = 1.0
        for n in neighbors:
            nn = np.array(n[:2])
            if np.linalg.norm(nn - c) < 0.02:
                continue
            for p in pads:
                dmin = min(dmin, float(np.linalg.norm(nn - p)))
        return dmin

    def choose_yaw_cmd(self, need, flip90):
        """need: image-derived rotation (rad, already signed). Return an equivalent command
        need + k*pi/2 (k parity set by flip90): smallest |cmd| keeping |cumulative yaw| <= YAW_CUM_MAX,
        else the one minimising |cumulative yaw|."""
        cum = self.rb.yaw_from_start()
        r = self.rb.yaw_resp
        cands = [need + k * math.pi / 2 for k in range(-4, 5) if (k % 2 != 0) == bool(flip90)]
        ok = [c for c in cands if abs(cum + r * c) <= YAW_CUM_MAX and abs(c) <= math.radians(100)]
        if ok:
            return min(ok, key=abs)
        return min(cands, key=lambda c: abs(cum + r * c))

    def align_over(self, xy, top_z_guess):
        """Hover above xy, find container, align yaw and centre. Returns target dict or None."""
        rb = self.rb
        hz = (self.table_z if self.table_z is not None else rb.pos[2] - 0.2) + HOVER_ABOVE_TABLE
        hz = max(hz, (top_z_guess or 0) + 0.12)
        if rb.pos[2] < hz - 0.02:
            rb.move_to([rb.pos[0], rb.pos[1], hz], vmax=V_FAST, tol=0.01, max_steps=20, chunk=6, grip=-1.0)
        rb.move_to([xy[0], xy[1], hz], vmax=V_FAST, tol=0.008, max_steps=40, chunk=8, grip=-1.0)
        rb.hold(1)
        t = self.find_container(xy, with_axes=True)
        if t is None:
            for dz, dx, dy in ((0.05, 0, 0), (0.0, 0.04, 0), (0.0, -0.04, 0), (0.0, 0, 0.04), (0.0, 0, -0.04)):
                rb.move_to([xy[0] + dx, xy[1] + dy, hz + dz], vmax=V_MED, tol=0.008, max_steps=10, chunk=5)
                rb.hold(1)
                t = self.find_container(xy, max_dist=0.07, with_axes=True)
                if t is not None:
                    break
        if t is None:
            return None
        self.log("container found", np.round(t["xyz"], 3).tolist(), "dist %.3f" % t["dist"],
                 "angle", None if t["angle"] is None else round(t["angle"], 1), "area", t["area"],
                 "neighbors", len(t["neighbors"]))
        # which box edge will the fingers close along after a <=45 deg alignment?
        flip90 = False
        if t.get("edges") is not None and t.get("axis_u") is not None and t["neighbors"]:
            ea, eb = t["edges"]
            au = t["axis_u"]
            natural, other = (ea, eb) if abs(float(ea @ au)) >= abs(float(eb @ au)) else (eb, ea)
            c_nat = self.clearance(t["xyz"], natural, t["neighbors"])
            c_oth = self.clearance(t["xyz"], other, t["neighbors"])
            self.log("clearance natural %.3f other %.3f" % (c_nat, c_oth))
            if c_oth > c_nat + 0.01 and c_nat < 0.045:
                flip90 = True
        sign = getattr(self, "yaw_sign", -1.0)
        target = t
        self._no_effect = 0
        for it in range(4):
            ang = target["angle"]
            if ang is None:
                break
            need = sign * math.radians(ang)
            cmd = self.choose_yaw_cmd(need, flip90 if it == 0 else False)
            if abs(cmd) < math.radians(3):
                break
            rb.rotate_yaw(cmd)
            rb.hold(1)
            nt = self.find_container(xy, max_dist=0.07)
            if nt is None:
                self.log("lost container after rotation")
                break
            self.log("yaw iter", it, "cmd %.1f deg" % math.degrees(cmd), "angle before %.1f after %s" %
                     (ang, None if nt["angle"] is None else round(nt["angle"], 1)),
                     "cum %.1f" % math.degrees(rb.yaw_from_start()))
            if nt["angle"] is not None and abs(ang) > 8 and abs(nt["angle"]) > abs(ang) + 3:
                sign = -sign
                self.log("yaw sign flipped ->", sign)
            if nt["angle"] is not None and abs(P.wrap90(nt["angle"] - ang)) < 1.5 and abs(cmd) > math.radians(6):
                no_effect = getattr(self, "_no_effect", 0) + 1
                self._no_effect = no_effect
                self.log("rotation had no visible effect", no_effect)
                if no_effect >= 2:
                    break
            target = nt
            if nt["angle"] is not None and abs(nt["angle"]) < 3:
                break
        self.yaw_sign = sign
        return target

    def pick(self, item, last):
        rb = self.rb
        xy = item["xy"]
        for attempt in range(PICK_RETRIES + 1):
            t = self.align_over(xy, None)
            if t is None:
                self.ev("pick", item["color"], "attempt", attempt, "no container found")
                # blind fallback: use recorded xy and table-relative height
                if attempt < PICK_RETRIES:
                    continue
                return False
            top = np.array(t["xyz"])
            cx, cy, tz = float(top[0]), float(top[1]), float(top[2])
            # approach above, refine
            rb.move_to([cx, cy, tz + APPROACH_ABOVE_TOP], vmax=V_MED, tol=0.005, max_steps=30, chunk=6, grip=-1.0)
            rb.hold(1)
            t2 = self.find_container([cx, cy], max_dist=0.03)
            if t2 is not None:
                p2 = np.array(t2["xyz"])
                if abs(p2[2] - tz) < 0.03:
                    cx, cy, tz = float(p2[0]), float(p2[1]), float(max(tz, p2[2]))
                self.log("refined top", round(cx, 4), round(cy, 4), round(tz, 4),
                         "angle", None if t2["angle"] is None else round(t2["angle"], 1))
                if t2["angle"] is not None and abs(t2["angle"]) > 6:
                    rb.rotate_yaw(self.choose_yaw_cmd(self.yaw_sign * math.radians(t2["angle"]), False))
            else:
                self.log("refine: container not re-found; using hover estimate")
            gz = tz - GRASP_BELOW_TOP
            if self.table_z is not None:
                gz = max(gz, self.table_z + 0.012)
            rb.move_to([cx, cy, tz + 0.02], vmax=V_MED, tol=0.004, max_steps=20, chunk=4)
            ok, e = rb.move_to([cx, cy, gz], vmax=V_SLOW, tol=0.004, max_steps=20, chunk=3)
            if not ok and rb.pos[2] > gz + 0.012:
                self.ev("descent blocked at z %.3f (target %.3f); retry" % (rb.pos[2], gz))
                rb.move_to([cx, cy, tz + APPROACH_ABOVE_TOP], vmax=V_MED, tol=0.01, max_steps=15, chunk=5)
                continue
            rb.gripper(1.0, n=12)
            f = rb.fingers
            self.log("closed fingers %.4f" % f)
            if f < HOLD_MIN_FINGERS:
                self.ev("grasp missed (fingers %.4f); retry" % f)
                rb.gripper(-1.0, n=6)
                rb.move_to([cx, cy, tz + APPROACH_ABOVE_TOP], vmax=V_MED, tol=0.01, max_steps=15, chunk=5)
                continue
            rb.move_to([cx, cy, gz + LIFT_HEIGHT], vmax=V_MED, tol=0.008, max_steps=25, chunk=5)
            f2 = rb.fingers
            self.ev("LIFT", item["color"], "step", rb.step_id, "fingers %.4f" % f2, "z %.3f" % rb.pos[2])
            self.check_revealed(item["color"], cx, cy)
            if f2 < HOLD_MIN_FINGERS:
                self.ev("dropped during lift; retry")
                rb.gripper(-1.0, n=4)
                continue
            if last:
                rb.hold(6)
                return True
            # put back
            rb.move_to([cx, cy, gz + 0.006], vmax=V_MED, tol=0.005, max_steps=25, chunk=5)
            rb.gripper(-1.0, n=8)
            rb.move_to([cx, cy, tz + APPROACH_ABOVE_TOP + 0.03], vmax=V_MED, tol=0.01, max_steps=15, chunk=5)
            return True
        return False

    def check_revealed(self, color, cx, cy):
        try:
            fr = self.rb.obs.rgb.get("front")
            if fr is None:
                return
            ds = P.detect_cubes(fr, "front")
            if not ds:
                self.log("reveal check: no coloured cube in front view")
                return
            pts = self.p2w([("front", d["u"], d["v"]) for d in ds][:32])
            seen = []
            for d, (xyz, sp) in zip(ds, pts):
                if xyz is not None and math.hypot(xyz[0] - cx, xyz[1] - cy) < 0.05:
                    seen.append(d["color"])
            self.ev("reveal check under", color, "->", seen)
        except Exception as ex:
            self.log("reveal check error", repr(ex))

    # -------------------------------------------------------- run
    def run(self):
        try:
            self.init()
        except EpisodeEnded:
            return
        except Exception:
            self.log("init error", traceback.format_exc())
            try:
                self.rb.observe()
                if self.home is None:
                    self.home = self.rb.pos.copy()
            except Exception:
                return
        for phase in (self.press, self.go_home):
            try:
                phase()
            except EpisodeEnded:
                return
            except Exception:
                self.log(phase.__name__, "error", traceback.format_exc())
        cl = []
        try:
            cl = self.watch()
        except EpisodeEnded:
            return
        except Exception:
            self.log("watch error", traceback.format_exc())
            self.watch_active = False
            cl = self.clusters()
        plan = self.plan(cl)
        todo = [p for p in plan if p["xy"] is not None]
        for i, item in enumerate(plan):
            if item["xy"] is None:
                continue
            last = item is todo[-1]
            try:
                ok = self.pick(item, last)
                self.ev("pick", item["color"], "ok" if ok else "FAILED", "step", self.rb.step_id)
            except EpisodeEnded:
                return
            except Exception:
                self.log("pick error", traceback.format_exc())
                try:
                    self.rb.observe()
                except Exception:
                    return
                if self.rb.obs.ended:
                    return
        # nothing left: hold briefly so the final state is observed
        try:
            self.rb.hold(5)
        except Exception:
            pass


def main(ctx):
    c = None
    try:
        c = Controller(ctx)
        c.run()
    except Exception:
        try:
            print("fatal", traceback.format_exc(), flush=True)
        except Exception:
            pass
    finally:
        if c is not None:
            try:
                c.summary["final_step"] = c.rb.step_id
                c.summary["ended"] = bool(c.rb.obs and c.rb.obs.ended)
                c.summary["complete"] = bool(c.rb.obs and c.rb.obs.complete)
                c.summary["calls"] = c.rb.calls
                c.summary["elapsed_s"] = time.time() - c.log.t0
                c.log.dump("summary.json", c.summary)
                c.log("done", c.summary.get("final_step"), "ended", c.summary.get("ended"),
                      "complete", c.summary.get("complete"))
            except Exception:
                pass
