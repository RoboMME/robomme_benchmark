"""InsertPeg controller (RoboMME-Plus, RAX native contract).

VERSION v1 (offline-developed, AfD: no simulator start was available).

Idea
----
The episode's conditioning video shows the robot solving the task *in the same layout* that
the execution starts from (verified in all 3 released training recordings: the video's first
frame and the first execution frame show identical object poses; the scene is reset after the
video and the expert then repeats the video's grasp/insert).  The reset frame-sequence manifest
records the robot state for each video frame.  So the controller re-executes the video's
end-effector pose trajectory with closed-loop absolute-pose tracking:

    every control step:  delta = target_pose(video frame k) (-) measured_pose

which, because the adapter targets "measured pose + scaled delta", commands the video pose
directly.  The gripper follows the video's finger-width history (close when the video closed,
open when it opened).  Convergence gates hold the video clock at every stationary point of the
video trajectory (arrival before grasp, before insertion, end) so tracking lag never cuts
corners where it matters.  After the trajectory: hold, push along the insertion direction,
then a small retract/offset/re-insert search ladder until success, episode end, step or wall
budget runs out.

Transition table
----------------
LOAD      -> TRACK (trajectory found)            | FALLBACK (no usable per-frame state)
TRACK     -> FINISH (video clock at last frame)  | DONE (episode ended / task_complete)
            (grasp check failure -> one regrasp from the last gate before the grasp)
FINISH    -> HOLD -> PUSH -> SEARCH ladder -> DONE (success / ended / budget exhausted)
FALLBACK  -> hold the start pose (no motion) and return; logged as a known limitation.
"""
import json
import math
import os
import time
from pathlib import Path

VERSION = "insertpeg-v1"

# ---------------------------------------------------------------- tunables
# All untested live (AfD); values chosen from the documented interface scaling and from the
# expert video's timing (20 Hz, ~400-460 frames, stationary pauses at grasp/insert).
POS_SCALE = 0.05            # m per unit action (documented)
ROT_SCALE = 0.5             # rad per unit action (documented)
CTRL_HZ = 20.0
LEAD_FREE = 2               # frames of look-ahead while the video moves freely
LEAD_PRECISE = 1            # look-ahead inside precision windows
GATE_POS_TOL = 0.003        # m, convergence gate at stationary video points
GATE_ROT_TOL = 0.03         # rad
GATE_FINAL_POS_TOL = 0.002  # m, gate at the final video frame
STALL_WINDOW = 4            # steps over which a gate must still improve
STALL_POS = 0.0003          # m
STALL_ROT = 0.003           # rad
GATE_MAX_WAIT = 25          # steps per gate (normal)
GATE_FINAL_MAX_WAIT = 30    # steps at the final frame
STATIONARY_SPEED = 0.0003   # m/frame: below this the video arm is "stopped"
STATIONARY_ROT = 0.004      # rad/frame
MIN_STATIONARY_RUN = 3      # frames
WIDTH_EVENT = 0.004         # m change of finger sum that marks a gripper event
GRIP_SETTLE_MAX = 20        # steps to wait for our fingers to settle after a command
INSERT_AXIS_LEN = 0.02      # m of final straight approach used as the insertion axis
INSERT_STRAIGHT_TOL = 0.003 # m max deviation from that line
MIN_TRAJ_RANGE = 0.02       # m: video EE must move at least this much to count as a trajectory
WALL_LIMIT_S = 1080.0       # stop issuing motion after this (controller limit 1200 s)
PUSH_DISTANCES = (0.003, 0.006, 0.009)   # m beyond the final pose along insert dir
PUSH_HOLD = 6
BLOCKED_MARGIN = 0.004      # m: EE short of the push target by more than this = blocked
MAX_STEP_M = 0.025          # m: cap on commanded translation per step
MAX_BATCH = 5               # rows per call when tool calls are slow (free motion only)
SEARCH_OFFSETS_MM = ((0, 0), (2, 0), (-2, 0), (0, 2), (0, -2),
                     (2, 2), (-2, -2), (2, -2), (-2, 2), (4, 0), (-4, 0), (0, 4), (0, -4))
SEARCH_BACKOFF = 0.025      # m retract along the insertion direction before re-inserting
OUTPUT_DIR = Path("/workspace/output")


# ---------------------------------------------------------------- logging
_LOG_FH = None
_T0 = time.monotonic()


def log(msg):
    global _LOG_FH
    line = "[%s %7.1fs] %s" % (VERSION, time.monotonic() - _T0, msg)
    try:
        print(line, flush=True)
    except Exception:
        pass
    try:
        if _LOG_FH is None:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            _LOG_FH = open(OUTPUT_DIR / "controller_log.txt", "a")
        _LOG_FH.write(line + "\n")
        _LOG_FH.flush()
    except Exception:
        pass


# ---------------------------------------------------------------- rotation math (stdlib)
def _norm(v):
    return math.sqrt(sum(x * x for x in v))


def rotvec_to_quat(r):
    a = _norm(r)
    if a < 1e-12:
        return (1.0, 0.0, 0.0, 0.0)
    s = math.sin(a / 2.0) / a
    return (math.cos(a / 2.0), r[0] * s, r[1] * s, r[2] * s)


def quat_to_rotvec(q):
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return [0.0, 0.0, 0.0]
    w, x, y, z = w / n, x / n, y / n, z / n
    if w < 0:
        w, x, y, z = -w, -x, -y, -z
    v = math.sqrt(x * x + y * y + z * z)
    if v < 1e-12:
        return [0.0, 0.0, 0.0]
    ang = 2.0 * math.atan2(v, w)
    return [x / v * ang, y / v * ang, z / v * ang]


def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw)


def quat_conj(q):
    return (q[0], -q[1], -q[2], -q[3])


def quat_slerp(a, b, t):
    dot = sum(x * y for x, y in zip(a, b))
    if dot < 0:
        b = tuple(-x for x in b)
        dot = -dot
    if dot > 0.9995:
        q = tuple(x + t * (y - x) for x, y in zip(a, b))
        n = math.sqrt(sum(x * x for x in q))
        return tuple(x / n for x in q)
    th = math.acos(max(-1.0, min(1.0, dot)))
    s = math.sin(th)
    wa = math.sin((1 - t) * th) / s
    wb = math.sin(t * th) / s
    return tuple(wa * x + wb * y for x, y in zip(a, b))


def rot_delta(q_target, q_meas):
    """World-frame rotation vector d with exp(d) * R_meas = R_target."""
    return quat_to_rotvec(quat_mul(q_target, quat_conj(q_meas)))


def rot_angle(q_a, q_b):
    return _norm(rot_delta(q_a, q_b))


def quat_rotate(q, v):
    p = (0.0, v[0], v[1], v[2])
    r = quat_mul(quat_mul(q, p), quat_conj(q))
    return [r[1], r[2], r[3]]


# ---------------------------------------------------------------- reply parsing
def reply_json(result):
    """First JSON text block of an MCP tool result (dict), or {}."""
    try:
        if isinstance(result, dict) and "content" in result:
            for block in result.get("content") or []:
                if isinstance(block, dict) and block.get("type") == "text":
                    try:
                        val = json.loads(block.get("text", ""))
                        if isinstance(val, dict):
                            return val
                    except Exception:
                        continue
        if isinstance(result, dict) and "structuredContent" in result:
            sc = result["structuredContent"]
            if isinstance(sc, dict):
                return sc
        if isinstance(result, dict):
            return result
    except Exception:
        pass
    return {}


def bfs_find(obj, pred, max_nodes=20000):
    """Breadth-first search; returns the first (key, value) with pred(key, value) true."""
    queue = [obj]
    seen = 0
    while queue and seen < max_nodes:
        nxt = []
        for node in queue:
            seen += 1
            if isinstance(node, dict):
                for k, v in node.items():
                    try:
                        if pred(k, v):
                            return k, v
                    except Exception:
                        pass
                    if isinstance(v, (dict, list)):
                        nxt.append(v)
            elif isinstance(node, list):
                for v in node:
                    if isinstance(v, (dict, list)):
                        nxt.append(v)
        queue = nxt
    return None, None


def _is_state(v):
    return (isinstance(v, list) and len(v) >= 8
            and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v[:8])
            and all(math.isfinite(x) for x in v[:8]))


def parse_state(js, deep=False):
    """Current 8-D state from a reply: top level or its observation; BFS only if deep."""
    if not isinstance(js, dict):
        return None
    cands = [js.get("state")]
    for key in ("observation", "final_observation", "obs", "current_observation"):
        sub = js.get(key)
        if isinstance(sub, dict):
            cands.append(sub.get("state"))
    for v in cands:
        if _is_state(v):
            return [float(x) for x in v[:8]]
    if deep:
        k, v = bfs_find(js, lambda k, v: k == "state" and _is_state(v))
        if v is not None:
            return [float(x) for x in v[:8]]
    return None


def parse_flag(js, name):
    """True if any `name` key is true anywhere in the reply."""
    k, v = bfs_find(js, lambda k, v: k == name and v is True)
    return v is True


def parse_int(js, names):
    k, v = bfs_find(js, lambda k, v: k in names and isinstance(v, (int, float))
                    and not isinstance(v, bool))
    return int(v) if v is not None else None


def parse_remaining_steps(js):
    """Best effort: remaining control steps from an `episode_limits` dict, else None.

    Only used for speed planning and logging, never to stop the run (a misparsed key must not
    end an episode early)."""
    k, lim = bfs_find(js, lambda k, v: k == "episode_limits" and isinstance(v, dict))
    if not isinstance(lim, dict):
        return None
    for key in ("remaining_steps", "steps_remaining", "remaining"):
        v = lim.get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return int(v)
    mx = None
    for key in ("max_steps", "step_limit", "max_episode_steps", "steps"):
        v = lim.get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            mx = int(v)
            break
    cur = None
    for key in ("step_id", "steps_taken", "elapsed_steps", "used_steps", "current_step"):
        v = lim.get(key)
        if v is None:
            v = js.get(key) if isinstance(js, dict) else None
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            cur = int(v)
            break
    if mx is not None and cur is not None:
        return mx - cur
    return None


# ---------------------------------------------------------------- robot I/O wrapper
class Robot:
    def __init__(self, ctx):
        self.ctx = ctx
        self.state = None
        self.ended = False
        self.success = False
        self.remaining = None
        self.steps_sent = 0
        self.last_js = {}
        self.call_times = []

    def _absorb(self, js, deep=False):
        self.last_js = js
        st = parse_state(js, deep=deep)
        if st is not None:
            self.state = st
        if parse_flag(js, "task_complete"):
            self.success = True
        if parse_flag(js, "episode_ended"):
            self.ended = True
        rem = parse_remaining_steps(js)
        if rem is not None and -5 <= rem <= 100000:
            self.remaining = rem

    def observe(self):
        res = self.ctx.call("robomme_observe")
        js = reply_json(res)
        self._absorb(js, deep=True)
        return js

    def status(self):
        try:
            res = self.ctx.call("robomme_status")
            js = reply_json(res)
            if parse_flag(js, "task_complete"):
                self.success = True
            if parse_flag(js, "episode_ended"):
                self.ended = True
            rem = parse_remaining_steps(js)
            if rem is not None:
                self.remaining = rem
            return js
        except Exception as e:  # status is advisory
            log("status failed: %r" % (e,))
            return {}

    def step(self, rows):
        """Send rows (list of 7-vectors). Never retries; on error re-reads status/observation."""
        clean = []
        for r in rows:
            rr = []
            for x in r[:7]:
                x = float(x)
                if not math.isfinite(x):
                    x = 0.0
                rr.append(round(max(-1.0, min(1.0, x)), 6))
            while len(rr) < 7:
                rr.append(0.0)
            clean.append(rr)
        clean = clean[:100]
        if not clean:
            return False
        t = time.monotonic()
        try:
            res = self.ctx.call("robomme_step", actions=clean)
        except Exception as e:
            log("robomme_step error: %s" % (str(e)[:300],))
            self.status()
            try:
                self.observe()
            except Exception as e2:
                log("observe after error failed: %s" % (str(e2)[:200],))
                self.ended = True
            return False
        self.call_times.append(time.monotonic() - t)
        self.steps_sent += len(clean)
        js = reply_json(res)
        old_rem = self.remaining
        self._absorb(js)
        if old_rem is not None and parse_remaining_steps(js) is None:
            self.remaining = old_rem - len(clean)
        if parse_state(js) is None and not self.ended:
            # the step reply had no current state at a known place: read the cached observation
            try:
                self.observe()
            except Exception as e:
                log("observe failed: %s" % (str(e)[:200],))
        return True

    def mean_call_time(self):
        if not self.call_times:
            return 0.5
        recent = self.call_times[-20:]
        return sum(recent) / len(recent)


def pose_of(state):
    p = state[0:3]
    q = rotvec_to_quat(state[3:6])
    w = state[6] + state[7]
    return p, q, w


def tracking_row(state, p_t, q_t, grip, pos_gain=1.0):
    p, q, _ = pose_of(state)
    dp = [(p_t[i] - p[i]) * pos_gain / POS_SCALE for i in range(3)]
    nrm = _norm(dp) * POS_SCALE
    if nrm > MAX_STEP_M:
        dp = [x * MAX_STEP_M / nrm for x in dp]
    m = max(abs(x) for x in dp)
    if m > 1.0:
        dp = [x / m for x in dp]
    dr = [x / ROT_SCALE for x in rot_delta(q_t, q)]
    m = max(abs(x) for x in dr)
    if m > 1.0:
        dr = [x / m for x in dr]
    return dp + dr + [float(grip)]


def increment_row(a, b, grip):
    """Open-loop row that moves from video pose a to video pose b."""
    dp = [(b[0][i] - a[0][i]) / POS_SCALE for i in range(3)]
    dr = [x / ROT_SCALE for x in rot_delta(b[1], a[1])]
    for v in (dp, dr):
        m = max(abs(x) for x in v)
        if m > 1.0:
            for i in range(3):
                v[i] /= m
    return dp + dr + [float(grip)]


def errors(state, p_t, q_t):
    p, q, _ = pose_of(state)
    return _norm([p_t[i] - p[i] for i in range(3)]), rot_angle(q_t, q)


# ---------------------------------------------------------------- video trajectory
def find_manifest_path(js):
    k, v = bfs_find(js, lambda k, v: k == "frame_sequence" and isinstance(v, dict))
    cands = []
    if isinstance(v, dict) and isinstance(v.get("manifest_path"), str):
        cands.append(v["manifest_path"])
    k2, v2 = bfs_find(js, lambda k, v: k == "manifest_path" and isinstance(v, str))
    if isinstance(v2, str):
        cands.append(v2)
    out = []
    for c in cands:
        for p in (Path("/workspace") / c, Path(c), Path.cwd() / c):
            if p.is_file() and p not in out:
                out.append(p)
    return out


def newest_run_manifests():
    """Fallback: the newest reset manifests under this run's frames directory."""
    root = Path("/workspace/runtime/autonomous_controller/runs")
    found = []
    try:
        for p in root.glob("*/frames/**/*.json"):
            try:
                found.append((p.stat().st_mtime, p))
            except Exception:
                pass
    except Exception:
        pass
    found.sort(reverse=True)
    return [p for _, p in found[:20]]


def load_video_states(manifest_path):
    m = json.loads(Path(manifest_path).read_text())
    frames = m.get("frames") if isinstance(m, dict) else None
    if not isinstance(frames, list):
        return None, None
    vids = [f for f in frames if isinstance(f, dict) and f.get("phase") == "video_demo"]
    execs = [f for f in frames if isinstance(f, dict) and f.get("phase") == "execution"]
    if not vids:
        return None, None
    vids.sort(key=lambda f: (f.get("frame_index", 0)))
    out = []
    for f in vids:
        st = f.get("state")
        if not _is_state(st):
            k, st = bfs_find(f, lambda k, v: _is_state(v))
        if st is None:
            continue
        ts = f.get("timestamp_s")
        if not isinstance(ts, (int, float)):
            sfi = f.get("source_frame_index")
            ts = (sfi / CTRL_HZ) if isinstance(sfi, (int, float)) else None
        out.append((ts, [float(x) for x in st[:8]], f))
    exec_state = None
    if execs and _is_state(execs[-1].get("state")):
        exec_state = [float(x) for x in execs[-1]["state"][:8]]
    return out, exec_state


def resample(samples):
    """Per-control-step trajectory [(p, q, w)] from (timestamp, state) samples."""
    if any(s[0] is None for s in samples):
        pts = [(float(i), s[1]) for i, s in enumerate(samples)]
    else:
        t0 = samples[0][0]
        pts = [((s[0] - t0) * CTRL_HZ, s[1]) for s in samples]
    # dedupe equal times
    clean = []
    for t, st in pts:
        if clean and t <= clean[-1][0] + 1e-6:
            continue
        clean.append((t, st))
    if not clean:
        return []
    n = int(round(clean[-1][0])) + 1
    traj = []
    j = 0
    for i in range(n):
        while j + 1 < len(clean) and clean[j + 1][0] < i - 1e-9:
            j += 1
        ta, sa = clean[j]
        if j + 1 < len(clean):
            tb, sb = clean[j + 1]
        else:
            tb, sb = ta, sa
        u = 0.0 if tb <= ta else max(0.0, min(1.0, (i - ta) / (tb - ta)))
        pa, qa, wa = pose_of(sa)
        pb, qb, wb = pose_of(sb)
        p = [pa[k] + u * (pb[k] - pa[k]) for k in range(3)]
        q = quat_slerp(qa, qb, u)
        w = wa if u < 0.5 else wb
        traj.append((p, q, w))
    return traj


def analyse(traj):
    """Gripper command per frame, gripper events, stationary gates, precision windows."""
    n = len(traj)
    widths = [t[2] for t in traj]
    # gripper command: close when the width falls WIDTH_EVENT below the running open level,
    # open when it rises WIDTH_EVENT above the running closed level.
    cmd = [-1.0] * n
    events = []  # (frame, +1 close / -1 open)
    state = -1.0
    ref = widths[0]
    # if the video starts closed (tiny width), start with close
    if widths[0] < 0.01:
        state = 1.0
    for i in range(n):
        w = widths[i]
        if state < 0:
            ref = max(ref, w)
            if w < ref - WIDTH_EVENT:
                # the command was issued where the width began to fall
                k = i
                while k > 0 and widths[k - 1] < ref - 2e-4:   # still below the open level
                    k -= 1
                k = max(0, k - 1)
                events.append((k, 1.0))
                state = 1.0
                ref = w
        else:
            ref = min(ref, w)
            if w > ref + WIDTH_EVENT:
                k = i
                while k > 0 and widths[k - 1] > ref + 2e-4:   # still above the closed level
                    k -= 1
                k = max(0, k - 1)
                events.append((k, -1.0))
                state = -1.0
                ref = w
    cur = 1.0 if widths[0] < 0.01 else -1.0
    ev = dict(events)
    for i in range(n):
        if i in ev:
            cur = ev[i]
        cmd[i] = cur
    # stationary runs -> gate at the first frame of each run
    moving = []
    for i in range(n - 1):
        dp = _norm([traj[i + 1][0][k] - traj[i][0][k] for k in range(3)])
        dr = rot_angle(traj[i + 1][1], traj[i][1])
        moving.append(dp > STATIONARY_SPEED or dr > STATIONARY_ROT)
    moving.append(False)
    gates = set()
    i = 0
    while i < n:
        if not moving[i]:
            j = i
            while j < n and not moving[j]:
                j += 1
            if j - i >= MIN_STATIONARY_RUN or j >= n:
                gates.add(i)
            i = j
        else:
            i += 1
    for f, _ in events:
        gates.add(f)
    gates.add(n - 1)
    gates.discard(0)
    # insertion axis: straight-line approach into the final pose.  Walk back from the end until
    # the EE is INSERT_AXIS_LEN away from the final pose; accept if the path is straight.
    insert_dir, dl, seg_start = None, 0.0, n - 1
    pf = traj[n - 1][0]
    for want in (INSERT_AXIS_LEN, INSERT_AXIS_LEN / 2.0):
        j = n - 1
        while j > 0 and _norm([traj[j][0][k] - pf[k] for k in range(3)]) < want:
            j -= 1
        d = [pf[k] - traj[j][0][k] for k in range(3)]
        L = _norm(d)
        if L < want * 0.9:
            continue
        uu = [x / L for x in d]
        dev = 0.0
        for i in range(j, n):
            v = [traj[i][0][k] - traj[j][0][k] for k in range(3)]
            t = sum(v[k] * uu[k] for k in range(3))
            dev = max(dev, _norm([v[k] - t * uu[k] for k in range(3)]))
        if dev < INSERT_STRAIGHT_TOL:
            insert_dir, dl, seg_start = uu, L, j
            break
    return {"cmd": cmd, "events": events, "gates": sorted(gates), "moving": moving,
            "insert_dir": insert_dir, "insert_len": dl, "insert_start": seg_start,
            "widths": widths}


# ---------------------------------------------------------------- controller
class Controller:
    def __init__(self, ctx):
        self.robot = Robot(ctx)
        self.t0 = time.monotonic()

    def wall_left(self):
        return WALL_LIMIT_S - (time.monotonic() - self.t0)

    def out_of_budget(self):
        r = self.robot
        if r.ended or r.success:
            return True
        if self.wall_left() < 10:
            log("wall budget exhausted")
            return True
        return False

    # -- low-level: one closed-loop step toward a pose
    def goto_step(self, p_t, q_t, grip, gain=1.0):
        st = self.robot.state
        if st is None:
            self.robot.observe()
            st = self.robot.state
            if st is None:
                raise RuntimeError("no state")
        row = tracking_row(st, p_t, q_t, grip, gain)
        return self.robot.step([row])

    def converge(self, p_t, q_t, grip, pos_tol, rot_tol, max_steps, settle_grip=False):
        """Hold the target until within tolerance (and fingers settled if asked)."""
        last_w = None
        hist = []
        for k in range(max_steps):
            if self.out_of_budget():
                return False
            st = self.robot.state
            ep, er = errors(st, p_t, q_t)
            w = st[6] + st[7]
            settled = (not settle_grip) or (k >= 4 and last_w is not None and abs(w - last_w) < 3e-4)
            if ep < pos_tol and er < rot_tol and settled and k > 0:
                return True
            hist.append((ep, er))
            if settled and len(hist) >= STALL_WINDOW + 3 and \
                    hist[-1 - STALL_WINDOW][0] - ep < STALL_POS and \
                    hist[-1 - STALL_WINDOW][1] - er < STALL_ROT:
                log("gate stalled ep=%.4f er=%.4f after %d steps" % (ep, er, k))
                return False
            last_w = w
            gain = 1.0 if k < 6 else 1.4
            self.goto_step(p_t, q_t, grip, gain)
        st = self.robot.state
        ep, er = errors(st, p_t, q_t)
        log("gate timeout ep=%.4f er=%.4f" % (ep, er))
        return False

    def run(self):
        r = self.robot
        js = r.observe()
        log("start state=%s" % (r.state,))
        r.status()
        log("remaining steps at start: %s" % (r.remaining,))
        traj, info = self.load_trajectory(js)
        if traj is None:
            log("FALLBACK: no usable per-frame video state; holding still (limitation)")
            return self.fallback()
        return self.track(traj, info)

    def load_trajectory(self, js):
        paths = find_manifest_path(js)
        if not paths:
            log("no manifest path in observe reply; searching run frames")
            paths = newest_run_manifests()
        for p in paths:
            try:
                samples, exec_state = load_video_states(p)
            except Exception as e:
                log("manifest %s unreadable: %r" % (p, e))
                continue
            if not samples or len(samples) < 10:
                log("manifest %s: %s video samples with state" % (p, 0 if not samples else len(samples)))
                continue
            xs = [s[1][0] for s in samples]
            ys = [s[1][1] for s in samples]
            zs = [s[1][2] for s in samples]
            rng = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
            ws = [s[1][6] + s[1][7] for s in samples]
            log("manifest %s: %d video frames, EE range %.3f m, width %.4f..%.4f, exec_state=%s"
                % (p, len(samples), rng, min(ws), max(ws), exec_state))
            if rng < MIN_TRAJ_RANGE:
                log("video state does not move; not a trajectory")
                continue
            traj = resample(samples)
            info = analyse(traj)
            log("traj frames=%d events=%s gates=%s insert_dir=%s len=%.3f start=%d"
                % (len(traj), info["events"], info["gates"],
                   None if info["insert_dir"] is None else [round(x, 3) for x in info["insert_dir"]],
                   info["insert_len"], info["insert_start"]))
            if r_state := self.robot.state:
                p0, q0, _ = traj[0]
                ep, er = errors(r_state, p0, q0)
                log("start vs video frame0: pos err %.4f rot err %.4f" % (ep, er))
            return traj, info
        return None, None

    def fallback(self):
        # Hold the measured pose (no blind motion without a trajectory). Returns quickly.
        return False

    def plan_speed(self, n_left):
        """Frames advanced per step outside precision windows (1 normally, more if steps short)."""
        r = self.robot
        if r.remaining is None:
            return 1
        budget = r.remaining - 60
        if budget >= n_left * 1.15:
            return 1
        if budget >= n_left * 0.6:
            return 2
        return 3

    def track(self, traj, info):
        r = self.robot
        n = len(traj)
        cmd = info["cmd"]
        gates = info["gates"]
        events = dict(info["events"])
        close_frames = [f for f, c in info["events"] if c > 0]
        first_close = close_frames[0] if close_frames else None
        # precision windows: around gates and gripper events, and the whole insertion approach
        precise = [False] * n
        for g in gates + [f for f, _ in info["events"]]:
            for i in range(max(0, g - 12), min(n, g + 4)):
                precise[i] = True
        if first_close is not None:
            # after grasp, the transport+insert: keep every frame from the insertion segment on
            for i in range(max(info["insert_start"] - 25, 0), n):
                precise[i] = True
        # if the execution does not start where the video started, go there first
        ep0, er0 = errors(r.state, traj[0][0], traj[0][1])
        if ep0 > 0.01 or er0 > 0.05:
            log("start differs from video frame 0 (%.4f m, %.3f rad): converging" % (ep0, er0))
            self.converge(traj[0][0], traj[0][1], cmd[0], GATE_POS_TOL, GATE_ROT_TOL, 60)
        speed = self.plan_speed(n)
        log("speed=%d (remaining=%s, frames=%d)" % (speed, r.remaining, n))
        gate_set = set(gates)
        k = 0
        regrasp_used = False
        last_gate_before_close = max([g for g in gates if first_close is not None and g < first_close - 2] or [0])
        while k < n - 1:
            if self.out_of_budget():
                return r.success
            # gate: the video was stationary here (or a gripper event) -> converge first
            if k in gate_set:
                p_t, q_t, _ = traj[k]
                final = (k == n - 1)
                grip_now = cmd[k]
                ok = self.converge(p_t, q_t, cmd[max(0, k - 1)], GATE_POS_TOL, GATE_ROT_TOL,
                                   GATE_MAX_WAIT)
                if k in events:
                    # issue the gripper change at the converged pose and wait for the fingers
                    ok2 = self.converge(p_t, q_t, grip_now, GATE_POS_TOL, GATE_ROT_TOL,
                                        GRIP_SETTLE_MAX, settle_grip=True)
                    w = r.state[6] + r.state[7]
                    log("gripper event frame %d cmd %+.0f -> width %.4f (video after %.4f) ok=%s/%s"
                        % (k, grip_now, w, info["widths"][min(n - 1, k + 15)], ok, ok2))
                    if grip_now > 0 and w < 0.003 and info["widths"][min(n - 1, k + 15)] > 0.008 \
                            and not regrasp_used:
                        regrasp_used = True
                        log("grasp looks empty -> open and redo from frame %d" % last_gate_before_close)
                        self.converge(p_t, q_t, -1.0, 1.0, 1.0, 8, settle_grip=True)
                        k = last_gate_before_close
                        # lift back to that gate pose with the gripper open
                        p_g, q_g, _ = traj[k]
                        self.converge(p_g, q_g, -1.0, GATE_POS_TOL, GATE_ROT_TOL, GATE_MAX_WAIT)
                        k += 1
                        continue
                else:
                    log("gate frame %d ok=%s step=%d remaining=%s" % (k, ok, r.steps_sent, r.remaining))
            if self.out_of_budget():
                return r.success
            # slow tool calls: send short open-loop batches in free-motion segments
            if not precise[k] and k > 0 and \
                    r.mean_call_time() * (n - k) / max(1, speed) > self.wall_left() - 240:
                rows, kk = [], k
                st = r.state
                while len(rows) < MAX_BATCH and kk < n - 1 and not precise[kk] \
                        and (kk + 1) not in gate_set:
                    nk = min(n - 1, kk + speed)
                    if not rows:
                        rows.append(tracking_row(st, traj[nk][0], traj[nk][1], cmd[kk]))
                    else:
                        rows.append(increment_row(traj[kk], traj[nk], cmd[kk]))
                    kk = nk
                if len(rows) >= 2:
                    r.step(rows)
                    k = kk
                    continue
            # one tracking step
            if precise[k]:
                lead, adv = LEAD_PRECISE, 1
            else:
                lead, adv = LEAD_FREE, speed
            # never jump over a gate
            nxt = k + adv
            for g in range(k + 1, min(n - 1, k + adv) + 1):
                if g in gate_set:
                    nxt = g
                    break
            tgt = min(n - 1, nxt + lead - 1)
            for g in range(k + 1, tgt + 1):
                if g in gate_set:
                    tgt = g
                    break
            p_t, q_t, _ = traj[tgt]
            self.goto_step(p_t, q_t, cmd[k])
            k = nxt
            if r.steps_sent % 40 == 0:
                ep, er = errors(r.state, traj[k][0], traj[k][1])
                log("k=%d/%d step=%d err=%.4f/%.3f w=%.4f rem=%s call=%.2fs"
                    % (k, n, r.steps_sent, ep, er, r.state[6] + r.state[7], r.remaining,
                       r.mean_call_time()))
        # final frame
        p_f, q_f, _ = traj[n - 1]
        g_f = cmd[n - 1]
        ok = self.converge(p_f, q_f, g_f, GATE_FINAL_POS_TOL, GATE_ROT_TOL, GATE_FINAL_MAX_WAIT)
        ep, er = errors(r.state, p_f, q_f)
        log("trajectory done ok=%s ep=%.4f er=%.4f success=%s ended=%s steps=%d"
            % (ok, ep, er, r.success, r.ended, r.steps_sent))
        if r.success or r.ended:
            return r.success
        return self.finish(traj, info)

    def finish(self, traj, info):
        r = self.robot
        n = len(traj)
        p_f, q_f, _ = traj[n - 1]
        g_f = info["cmd"][n - 1]
        u = info["insert_dir"]
        if g_f < 0:
            # video released the peg at the end: nothing safe to push with; hold and check.
            log("video ends with open gripper; holding")
            self.converge(p_f, q_f, g_f, 0.001, 0.01, 20)
            r.status()
            return r.success
        if u is None:
            log("no insertion direction; holding")
            self.converge(p_f, q_f, g_f, 0.001, 0.01, 20)
            r.status()
            return r.success
        # push further along the insertion direction; measure how deep the EE actually gets
        def depth():
            p = r.state[0:3]
            return sum((p[i] - p_f[i]) * u[i] for i in range(3))
        best_depth = depth()
        for d in PUSH_DISTANCES:
            if self.out_of_budget():
                return r.success
            p_t = [p_f[i] + u[i] * d for i in range(3)]
            self.converge(p_t, q_f, g_f, 0.0015, GATE_ROT_TOL, PUSH_HOLD)
            best_depth = max(best_depth, depth())
            log("push %.3f: depth %.4f success=%s" % (d, depth(), r.success))
        r.status()
        if r.success or r.ended:
            return r.success
        blocked = best_depth < PUSH_DISTANCES[-1] - BLOCKED_MARGIN
        log("after push: best depth %.4f blocked=%s" % (best_depth, blocked))
        if not blocked:
            # the arm reached past the video's final pose: the peg went in as far as the video's;
            # do not pull it out. Hold slightly inside and finish.
            p_t = [p_f[i] + u[i] * PUSH_DISTANCES[0] for i in range(3)]
            self.converge(p_t, q_f, g_f, 0.0005, 0.005, 20)
            r.status()
            return r.success
        # search ladder: retract, offset perpendicular to u, re-insert
        a = [0.0, 0.0, 1.0] if abs(u[2]) < 0.9 else [1.0, 0.0, 0.0]
        e1 = [u[1] * a[2] - u[2] * a[1], u[2] * a[0] - u[0] * a[2], u[0] * a[1] - u[1] * a[0]]
        l1 = _norm(e1)
        e1 = [x / l1 for x in e1]
        e2 = [u[1] * e1[2] - u[2] * e1[1], u[2] * e1[0] - u[0] * e1[2], u[0] * e1[1] - u[1] * e1[0]]
        back = SEARCH_BACKOFF
        results = [((0, 0), best_depth)]
        for (o1, o2) in SEARCH_OFFSETS_MM:
            if self.out_of_budget():
                break
            off = [(o1 * e1[i] + o2 * e2[i]) / 1000.0 for i in range(3)]
            p_back = [p_f[i] - u[i] * back for i in range(3)]
            self.converge(p_back, q_f, g_f, 0.002, GATE_ROT_TOL, 15)
            p_back_off = [p_back[i] + off[i] for i in range(3)]
            self.converge(p_back_off, q_f, g_f, 0.0015, GATE_ROT_TOL, 10)
            # slow re-insert: 3 mm per step
            reach = PUSH_DISTANCES[-1]
            steps = int((back + reach) / 0.003) + 1
            for s_ in range(1, steps + 1):
                if self.out_of_budget():
                    break
                d = -back + (back + reach) * s_ / steps
                p_t = [p_f[i] + u[i] * d + off[i] for i in range(3)]
                self.goto_step(p_t, q_f, g_f)
            if self.out_of_budget():
                break
            p_t = [p_f[i] + u[i] * reach + off[i] for i in range(3)]
            self.converge(p_t, q_f, g_f, 0.0015, GATE_ROT_TOL, 6)
            dep = depth()
            results.append(((o1, o2), dep))
            log("search offset %s mm: depth %.4f success=%s" % ((o1, o2), dep, r.success))
            if r.success or r.ended:
                break
            if dep >= reach - BLOCKED_MARGIN:
                log("offset %s not blocked; staying" % ((o1, o2),))
                break
        if not (r.success or r.ended) and not self.out_of_budget():
            (o1, o2), dep = max(results, key=lambda t: t[1])
            log("ladder done; best offset %s depth %.4f" % ((o1, o2), dep))
            last = results[-1][0]
            if (o1, o2) != last:
                off = [(o1 * e1[i] + o2 * e2[i]) / 1000.0 for i in range(3)]
                p_back = [p_f[i] - u[i] * back for i in range(3)]
                self.converge(p_back, q_f, g_f, 0.002, GATE_ROT_TOL, 15)
                self.converge([p_back[i] + off[i] for i in range(3)], q_f, g_f, 0.0015,
                              GATE_ROT_TOL, 10)
                p_t = [p_f[i] + u[i] * PUSH_DISTANCES[-1] + off[i] for i in range(3)]
                self.converge(p_t, q_f, g_f, 0.0015, GATE_ROT_TOL, 25)
        r.status()
        return r.success


def main(ctx):
    log("controller start")
    c = Controller(ctx)
    try:
        ok = c.run()
        log("controller end success=%s ended=%s steps=%d" % (c.robot.success, c.robot.ended,
                                                            c.robot.steps_sent))
        return ok
    except Exception as e:
        import traceback
        log("controller exception: %r\n%s" % (e, traceback.format_exc()))
        return False
