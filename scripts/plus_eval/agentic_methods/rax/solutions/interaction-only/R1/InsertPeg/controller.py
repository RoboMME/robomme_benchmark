"""InsertPeg controller (RoboMME-Plus, RAX).  VERSION below.

Method: the conditioning video carries the robot state of a demonstration in the
same workspace. We reproduce that end-effector trajectory (position, orientation,
gripper) with a closed-loop pose servo, retargeted by planar transforms when the
depth comparison between the video's first frame and the execution frame shows the
peg or the box moved. Fallback without a usable video trajectory: a generic
depth-based grasp and side insertion.

Phase / transition table
  SETUP     -> REPLAY (demo ok)        | FALLBACK (no demo) | END (no manifest/state)
  REPLAY    -> PUSH (end, still holding) | END (episode ended / complete / budget)
  PUSH      -> END
  FALLBACK  -> END
Every phase has a step cap and the global wall/step guards in Robot.
"""
import glob
import os
import time
import traceback

import numpy as np

import mcpio
from mcpio import log, log_exc, reply_json, strip_images, save_json
from geom import rotvec_to_mat, mat_to_rotvec, rot_angle, rotz, SE2, wrap
import vision
import video
import scene

VERSION = "insertpeg-v1"

# ---- tunables (untested live: R1 budget, no exploration; values from the public
#      interface description: 0.05 m and 0.5 rad per unit action)
POS_SCALE = 0.05
ROT_SCALE = 0.5
GAIN_P = 1.0
GAIN_R = 1.0
WALL_LIMIT_S = 1000.0          # controller limit 1200 s, keep margin
CONVERGE_TOL_P = 0.003
CONVERGE_TOL_R = np.deg2rad(2.0)
CONVERGE_MAX = 25
APPROACH_MAX = 120
GRIP_SETTLE_MAX = 12
PUSH_MAX_M = 0.025


# =====================================================================  robot I/O
class EpisodeOver(Exception):
    pass


class Robot:
    def __init__(self, ctx):
        self.ctx = ctx
        self.state = None
        self.ended = False
        self.complete = False
        self.steps = 0
        self.call_t = []
        self.errors = 0
        self.consec_err = 0
        self.limit = None
        self.remaining = None
        self.rot_mode = "world"      # or "body"
        self.votes = 0.0
        self.t_start = time.time()

    # -- wall
    def wall_left(self):
        return WALL_LIMIT_S - (time.time() - mcpio._T0)

    def avg_call(self):
        if not self.call_t:
            return 0.3
        return float(np.median(self.call_t[-30:]))

    def _absorb(self, j):
        st = mcpio.get_state(j)
        if st is not None:
            self.state = np.array(st, dtype=float)
        if mcpio.episode_ended(j):
            self.ended = True
        if mcpio.task_complete(j):
            self.complete = True
        r = mcpio.steps_remaining(j)
        if r is not None:
            self.remaining = r
        lim = mcpio.step_limit(j)
        if lim is not None:
            self.limit = lim
        return st is not None

    def observe(self):
        res = self.ctx.call("robomme_observe")
        j = reply_json(res)
        self._absorb(j)
        return res, j

    def status(self):
        res = self.ctx.call("robomme_status")
        j = reply_json(res)
        self._absorb(j)
        return j

    def send(self, rows):
        if self.ended or self.complete:
            raise EpisodeOver()
        if self.wall_left() < 15:
            log("wall budget exhausted")
            raise EpisodeOver()
        rows = [[float(max(-1.0, min(1.0, x))) for x in r] for r in rows]
        if self.remaining is not None:
            rows = rows[:max(1, self.remaining)]
        t0 = time.time()
        try:
            res = self.ctx.call("robomme_step", actions=rows)
        except Exception as e:
            self.errors += 1
            log("step error #%d: %s" % (self.errors, str(e)[:400]))
            # do not resend blindly: refresh state; the loop recomputes from it
            try:
                self.status()
                self.observe()
            except Exception:
                log_exc("refresh after step error")
            self.consec_err += 1
            if self.consec_err >= 6 or self.errors >= 60:
                raise EpisodeOver()
            return False
        self.consec_err = 0
        self.call_t.append(time.time() - t0)
        j = reply_json(res)
        if self.steps < 2:
            save_json("step_reply", strip_images(res))
        got = self._absorb(j)
        self.steps += len(rows)
        if self.remaining is not None and mcpio.steps_remaining(j) is None:
            self.remaining -= len(rows)
        if not got and not self.ended:
            try:
                self.observe()
            except Exception:
                log_exc("observe after step")
        if self.ended or self.complete:
            log("episode ended=%s complete=%s at step %d" % (self.ended, self.complete, self.steps))
        return True

    # -- pose helpers
    @property
    def pos(self):
        return self.state[:3].copy()

    @property
    def R(self):
        return rotvec_to_mat(self.state[3:6])

    @property
    def width(self):
        return float(self.state[6] + self.state[7])

    def rot_cmd(self, R_t):
        R_c = self.R
        if self.rot_mode == "world":
            e = mat_to_rotvec(R_t @ R_c.T)
        else:
            e = mat_to_rotvec(R_c.T @ R_t)
        return e

    def servo(self, p_t, R_t, grip, max_dp=None, max_dr=None):
        """One closed-loop step toward (p_t, R_t). Returns errors before the step."""
        p = self.pos
        dp = (np.asarray(p_t) - p) * GAIN_P
        ep = float(np.linalg.norm(np.asarray(p_t) - p))
        if max_dp is not None and np.linalg.norm(dp) > max_dp:
            dp = dp / np.linalg.norm(dp) * max_dp
        a = dp / POS_SCALE
        m = np.max(np.abs(a))
        if m > 1.0:
            a = a / m
        er_vec = self.rot_cmd(R_t)
        er = float(np.linalg.norm(er_vec))
        dr = er_vec * GAIN_R
        if max_dr is not None and np.linalg.norm(dr) > max_dr:
            dr = dr / np.linalg.norm(dr) * max_dr
        b = dr / ROT_SCALE
        nb = np.linalg.norm(b)
        if nb > 1.0:
            b = b / nb
        R_before = self.R
        cmd_world = b * ROT_SCALE if self.rot_mode == "world" else R_before @ (b * ROT_SCALE)
        row = list(a) + list(b) + [float(grip)]
        ok = self.send([row])
        if ok and np.linalg.norm(cmd_world) > 0.04 and self.state is not None:
            self._vote(R_before, self.R, b * ROT_SCALE)
        return ep, er

    def _vote(self, R0, R1, c):
        """Which frame convention explains the measured rotation better?"""
        aw = mat_to_rotvec(R1 @ R0.T)
        ab = mat_to_rotvec(R0.T @ R1)
        nc = np.linalg.norm(c) + 1e-9
        if self.rot_mode == "world":
            fit_cur = np.dot(aw, c) / nc
            fit_alt = np.dot(ab, c) / nc
        else:
            fit_cur = np.dot(ab, c) / nc
            fit_alt = np.dot(aw, c) / nc
        self.votes += np.sign(fit_alt - fit_cur) if abs(fit_alt - fit_cur) > 0.3 * nc else 0.0
        if self.votes >= 3:
            self.rot_mode = "body" if self.rot_mode == "world" else "world"
            self.votes = 0
            log("rotation convention switched to", self.rot_mode)
        elif self.votes < -3:
            self.votes = -3

    def converge(self, p_t, R_t, grip, max_steps=CONVERGE_MAX, tol_p=CONVERGE_TOL_P,
                 tol_r=CONVERGE_TOL_R, max_dp=None, max_dr=None, tag=""):
        ep = er = None
        for i in range(max_steps):
            ep = float(np.linalg.norm(np.asarray(p_t) - self.pos))
            er = float(np.linalg.norm(self.rot_cmd(R_t)))
            if ep < tol_p and er < tol_r:
                return True, i, ep, er
            self.servo(p_t, R_t, grip, max_dp=max_dp, max_dr=max_dr)
        ep = float(np.linalg.norm(np.asarray(p_t) - self.pos))
        er = float(np.linalg.norm(self.rot_cmd(R_t)))
        log("converge%s: not within tol after %d steps (ep=%.4f er=%.2fdeg)" % (
            tag, max_steps, ep, np.rad2deg(er)))
        return False, max_steps, ep, er


# =====================================================================  setup
def find_manifest(j_list):
    for j in j_list:
        for mp in mcpio.manifest_paths(j):
            full = video.resolve(mp)
            if os.path.exists(full):
                return mp
    # fallback: newest manifest under runtime frames
    cands = []
    for pat in ("/workspace/runtime/autonomous_controller/runs/*/frames/**/manifest*.json",
                "/workspace/runtime/**/manifest*.json"):
        try:
            cands += glob.glob(pat, recursive=True)
        except Exception:
            pass
    cands = [c for c in cands if "demonstrations" not in c]
    if cands:
        cands.sort(key=lambda c: os.path.getmtime(c))
        return cands[-1]
    return None


def calibrate(robot, obs_j, step_id, dE):
    """Fit world = M [u d, v d, d, 1] from pixel_to_position samples on the front camera."""
    H, W = dE.shape
    sels = []
    us = np.linspace(20, W - 21, 7)
    vs = np.linspace(20, H - 21, 6)
    for v in vs:
        for u in us:
            uu, vv = int(round(u)), int(round(v))
            if np.isfinite(dE[vv, uu]):
                sels.append((uu, vv))
    sels = sels[:32]
    cal_json = vision.calib_from_json(obs_j, "front")
    samples = []
    if step_id is not None and len(sels) >= 8:
        try:
            res = robot.ctx.call("robomme_pixel_to_position", observation_step_id=int(step_id),
                                 depth_window_radius=0,
                                 selections=[{"camera": "front", "u": float(u), "v": float(v)} for u, v in sels])
            j = reply_json(res)
            save_json("p2p_reply", j)
            pts = vision.parse_p2p(j, len(sels))
            by_uv = {}
            for pt, raw in pts:
                if isinstance(raw, dict) and isinstance(raw.get("u"), (int, float)) and isinstance(raw.get("v"), (int, float)):
                    by_uv[(int(round(raw["u"])), int(round(raw["v"])))] = pt
            if len(by_uv) >= 8:
                for (u, v) in sels:
                    pt = by_uv.get((u, v))
                    if pt is not None:
                        samples.append((u, v, np.array(pt)))
            else:
                for (u, v), (pt, raw) in zip(sels, pts):
                    if pt is not None:
                        samples.append((u, v, np.array(pt)))
        except Exception:
            log_exc("pixel_to_position")
    fit = vision.fit_calib(samples, dE) if len(samples) >= 8 else None
    if fit is not None:
        log("calib fit: %s rms=%.5f n=%d" % (fit[0].source, fit[1], len(samples)))
    if cal_json is not None and samples:
        A = np.array([[u * dE[v, u], v * dE[v, u], dE[v, u], 1.0] for (u, v, P) in samples])
        Y = np.array([P for (u, v, P) in samples])
        rj = float(np.sqrt(np.mean(np.sum((A @ cal_json.M.T - Y) ** 2, axis=1))))
        log("calib json rms vs samples=%.5f" % rj)
        cal_json.resid = rj
    if fit is not None and fit[1] < 0.004:
        return fit[0]
    if cal_json is not None and (cal_json.resid is None or cal_json.resid < 0.01):
        return cal_json
    if fit is not None and fit[1] < 0.02:
        return fit[0]
    return cal_json


# =====================================================================  replay
def smoothstep(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def gripper_flip(demo, T_peg):
    """Parallel-jaw symmetry: for a top-down grasp of a peg rotated by more than ~100 deg,
    grasp with the gripper turned by pi about the vertical (same physical grasp, smaller
    wrist rotation). Returns body-frame flip matrix F or None."""
    hold = demo.grasp_hold()
    if hold is None or abs(T_peg.th) < np.deg2rad(100):
        return None
    Rg = demo.R[hold["closed"]]
    if np.max(np.abs(Rg[2, :])) < 0.97:      # no body axis vertical: not a top-down grasp
        return None
    return Rg.T @ rotz(np.pi) @ Rg


def build_targets(demo, T_peg, T_box):
    n = demo.n
    hold = demo.grasp_hold()
    P, R = [], []
    ident = T_peg.is_identity() and T_box.is_identity()
    F = None if ident else gripper_flip(demo, T_peg)
    if F is not None:
        log("using gripper flip (peg rotated %.0f deg)" % np.rad2deg(T_peg.th))
    for k in range(n):
        if ident or hold is None:
            T = SE2() if ident else T_peg
        else:
            kc, kr = hold["closed"], hold["release"]
            if k <= kc:
                T = T_peg
            elif k < kr:
                w = smoothstep((k - kc) / max(1.0, 0.4 * (kr - kc)))
                T = SE2.blend(T_peg, T_box, w)
            else:
                T = T_box
        if T.is_identity():
            P.append(demo.P[k].copy())
            Rk = demo.R[k].copy()
        else:
            P.append(T.apply_pos(demo.P[k]))
            Rk = T.apply_rot(demo.R[k])
        if F is not None and (hold is None or k < hold["release"] or hold["release"] >= n - 1):
            Rk = Rk @ F
        R.append(Rk)
    return np.array(P), R


def find_insert_start(demo, P, R):
    """Start index of the final straight, constant-orientation segment of the object hold."""
    hold = demo.grasp_hold()
    if hold is None:
        return None
    kc, kr = hold["closed"], min(hold["release"], demo.n - 1)
    ke = kr
    while ke > kc + 2 and np.linalg.norm(P[ke] - P[ke - 1]) < 2e-4:
        ke -= 1
    m = ke - 1
    while m > kc and np.linalg.norm(P[ke] - P[m]) < 0.01:
        m -= 1
    if m <= kc:
        return None
    d = (P[ke] - P[m]) / np.linalg.norm(P[ke] - P[m])
    i = m
    while i > kc + 1:
        v = P[ke] - P[i - 1]
        nv = np.linalg.norm(v)
        if nv < 1e-6 or np.dot(v / nv, d) < 0.995 or rot_angle(R[i - 1].T @ R[ke]) > np.deg2rad(2):
            break
        i -= 1
    if np.linalg.norm(P[ke] - P[i]) < 0.02:
        return None
    return i, ke, d


TRACE = []


def trace(robot, k, p_t):
    try:
        if len(TRACE) < 3000 and robot.state is not None:
            TRACE.append("%d,%d,%s,%s,%.4f,%s" % (robot.steps, k, ",".join("%.4f" % x for x in p_t),
                                                 ",".join("%.4f" % x for x in robot.state[:3]), robot.width, robot.rot_mode))
    except Exception:
        pass


def write_trace():
    try:
        with open(os.path.join(mcpio.OUT_DIR, "trace.csv"), "w") as fh:
            fh.write("exec_step,demo_k,tx,ty,tz,x,y,z,width,rot_mode\n")
            fh.write("\n".join(TRACE))
    except Exception:
        pass


def replay(robot, demo, P, R):
    n = demo.n
    cmd = demo.cmd
    hold = demo.grasp_hold()
    k0 = demo.first_move
    ev_idx = set(i for _, i in demo.events)
    closed_idx = set(h["closed"] for h in demo.holds)
    # precise segments: around gripper events and the last 45% of the object hold
    precise = np.zeros(n, dtype=bool)
    for _, i in demo.events:
        precise[max(0, i - 30):min(n, i + 20)] = True
    if hold is not None:
        a = hold["closed"] + int(0.55 * (hold["release"] - hold["closed"]))
        precise[a:hold["release"] + 5] = True
    log("replay: n=%d k0=%d precise=%d steps limit=%s remaining=%s" % (
        n, k0, int(precise.sum()), robot.limit, robot.remaining))
    # approach the first demo pose
    grip0 = cmd[k0]
    ok, used, ep, er = robot.converge(P[k0], R[k0], grip0, max_steps=APPROACH_MAX, tol_p=0.004,
                                      tol_r=np.deg2rad(3), max_dp=0.03, max_dr=0.2, tag="[approach]")
    log("approach done ok=%s steps=%d ep=%.4f er=%.2fdeg" % (ok, used, ep, np.rad2deg(er)))
    ins = find_insert_start(demo, P, R)
    k_pre = ins[0] if ins is not None else None
    if ins is not None:
        precise[max(0, ins[0] - 10):] = True
        log("insertion segment: demo %d..%d length %.3f m dir %s" % (
            ins[0], ins[1], np.linalg.norm(P[ins[1]] - P[ins[0]]), np.round(ins[2], 3)))
    k = k0
    stride = 1
    max_err = 0.0
    grasp_checked = False
    while k < n - 1:
        # budget-aware stride
        left_demo = (n - 1 - k)
        if robot.remaining is not None:
            need = left_demo + 60
            stride = 2 if robot.remaining < need else 1
        if robot.wall_left() < left_demo * robot.avg_call() * 1.2 + 30:
            stride = max(stride, 2)
        s = 1 if precise[k] else stride
        kn = min(n - 1, k + s)
        # do not jump over an event
        for i in range(k + 1, kn + 1):
            if i in ev_idx:
                kn = i
                break
        if kn in ev_idx and cmd[kn] != cmd[k]:
            # settle on the pose before switching the gripper
            okc, used, ep, er = robot.converge(P[kn], R[kn], cmd[k], max_steps=40, tol_p=0.0012,
                                               tol_r=np.deg2rad(1.5),
                                               tag="[pre-%s@%d]" % ("close" if cmd[kn] > 0 else "open", kn))
            log("gripper %s at demo step %d (exec step %d) ep=%.4f er=%.2fdeg width=%.4f" % (
                "close" if cmd[kn] > 0 else "open", kn, robot.steps, ep, np.rad2deg(er), robot.width))
        if k_pre is not None and k < k_pre <= kn:
            kn = k_pre
            okc, used, ep, er = robot.converge(P[kn], R[kn], cmd[kn], max_steps=40, tol_p=0.0010,
                                               tol_r=np.deg2rad(1.0), tag="[pre-insert]")
            log("pre-insert pose at demo %d: ok=%s steps=%d ep=%.4f er=%.2fdeg width=%.4f" % (
                kn, okc, used, ep, np.rad2deg(er), robot.width))
        ep, er = robot.servo(P[kn], R[kn], cmd[kn])
        trace(robot, kn, P[kn])
        max_err = max(max_err, ep)
        if kn in closed_idx:
            # wait for the fingers to stop before lifting
            prev = robot.width
            for _ in range(GRIP_SETTLE_MAX):
                robot.servo(P[kn], R[kn], cmd[kn])
                if abs(robot.width - prev) < 0.0003:
                    break
                prev = robot.width
            h = next(h for h in demo.holds if h["closed"] == kn)
            log("grasp settled: my width=%.4f demo width=%.4f (exec step %d)" % (robot.width, h["width"], robot.steps))
            if h["object"] and robot.width < 0.6 * h["width"] - 0.002:
                log("WARNING: grasp looks empty (width %.4f vs demo %.4f)" % (robot.width, h["width"]))
        if (kn - k0) % 25 == 0:
            log("replay k=%d/%d step=%d ep=%.4f er=%.2fdeg width=%.4f rem=%s rot=%s" % (
                kn, n, robot.steps, ep, np.rad2deg(er), robot.width, robot.remaining, robot.rot_mode))
        k = kn
    # finish on the final pose
    robot.converge(P[n - 1], R[n - 1], cmd[n - 1], max_steps=10, tag="[final]")
    log("replay finished at exec step %d max_err=%.4f" % (robot.steps, max_err))


def push(robot, demo, P, R):
    """If the demo ends while holding, press a little further along the final motion."""
    n = demo.n
    if demo.cmd[n - 1] < 0:
        return
    end = P[n - 1]
    d = None
    for m in range(3, n):
        v = end - P[n - 1 - m]
        if np.linalg.norm(v) > 0.03:
            d = v / np.linalg.norm(v)
            break
    if d is None:
        return
    log("push along", np.round(d, 3))
    for s in np.arange(0.004, PUSH_MAX_M + 1e-9, 0.003):
        robot.converge(end + d * s, R[n - 1], 1.0, max_steps=4, tol_p=0.0015, tag="[push]")
    for _ in range(6):
        robot.servo(end + d * PUSH_MAX_M, R[n - 1], 1.0)


# =====================================================================  fallback
def fallback(robot, calib, dE, rgbE):
    """Generic: grasp the elongated object, align it to the box, push sideways."""
    if calib is None or dE is None:
        log("fallback: no calibration/depth, nothing to do")
        return
    PE = calib.backproject(dE)
    tz = vision.table_height(PE)
    if tz is None:
        log("fallback: no table")
        return
    labE, sizes = scene._comps(PE, tz, robot.pos)
    stats = []
    for i in range(len(sizes)):
        st = vision.comp_stats(PE, rgbE, labE == i, tz)
        if st is not None and st["n"] >= 30:
            stats.append(st)
    for st in stats:
        log("fallback comp", scene._fmt(st))
    pegs = [s for s in stats if s["height"] < 0.07 and s["length"] > 2.0 * max(0.01, s["width"]) and s["length"] < 0.4]
    if not pegs:
        log("fallback: no peg-like object")
        return
    peg = max(pegs, key=lambda s: s["length"] / max(s["width"], 0.01))
    boxes = [s for s in stats if s is not peg and s["height"] > 0.03 and s["n"] > 60]
    if not boxes:
        log("fallback: no box")
        return
    box = min(boxes, key=lambda s: np.linalg.norm(s["c_top"] - peg["c_top"]))
    ax = peg["axis"]
    to_box = box["c_top"] - peg["c_top"]
    if np.dot(ax, to_box) < 0:
        ax = -ax                     # ax points from peg toward the box ("head" side)
    r = max(0.01, peg["height"] / 2)
    grasp_xy = peg["c_top"] - ax * 0.25 * peg["length"]
    grasp = np.array([grasp_xy[0], grasp_xy[1], tz + r])
    R0 = robot.R
    # yaw: gripper closing axis (body y) perpendicular to the peg axis
    by = R0[:, 1]
    cur = np.arctan2(by[1], by[0])
    want = np.arctan2(ax[1], ax[0]) + np.pi / 2
    dyaw = min([wrap(want - cur), wrap(want + np.pi - cur)], key=abs)
    Rg = rotz(dyaw) @ R0
    above = grasp + np.array([0, 0, 0.12])
    robot.converge(above, Rg, -1, max_steps=80, max_dp=0.03, max_dr=0.2, tol_p=0.005, tag="[fb-above]")
    robot.converge(grasp, Rg, -1, max_steps=50, max_dp=0.015, tol_p=0.003, tag="[fb-down]")
    for _ in range(12):
        robot.servo(grasp, Rg, 1)
    log("fallback grasp width=%.4f" % robot.width)
    # carry: peg axis toward the box direction measured from the grasp point
    hole_z = tz + max(box["height"] * 0.5, r)
    d = box["c_top"] - grasp_xy
    d = d / (np.linalg.norm(d) + 1e-9)
    yaw_now = np.arctan2(ax[1], ax[0])
    yaw_want = np.arctan2(d[1], d[0])
    Rc = rotz(wrap(yaw_want - yaw_now)) @ Rg
    half_box = 0.5 * max(box["length"], box["width"])
    lead = 0.75 * peg["length"]           # head extends beyond the grasp point
    pre_xy = box["c_top"] - d * (half_box + lead + 0.05)
    lift = grasp + np.array([0, 0, 0.12])
    robot.converge(lift, Rg, 1, max_steps=40, max_dp=0.02, tol_p=0.006, tag="[fb-lift]")
    pre = np.array([pre_xy[0], pre_xy[1], hole_z])
    robot.converge(pre + np.array([0, 0, 0.08]), Rc, 1, max_steps=80, max_dp=0.02, max_dr=0.15, tol_p=0.005, tag="[fb-pre-high]")
    robot.converge(pre, Rc, 1, max_steps=40, max_dp=0.01, tol_p=0.003, tag="[fb-pre]")
    for s in np.arange(0.005, 0.10, 0.005):
        robot.converge(pre + np.append(d * s, 0.0), Rc, 1, max_steps=3, tol_p=0.002, tag="[fb-insert]")
    log("fallback done")


# =====================================================================  main
def run(ctx):
    log("controller", VERSION, "start")
    robot = Robot(ctx)
    st_j = {}
    try:
        st_j = robot.status()
        save_json("status_reply", st_j)
    except Exception:
        log_exc("status")
    obs_res, obs_j = None, {}
    for attempt in range(3):          # read-only call: safe to retry
        try:
            obs_res, obs_j = robot.observe()
            break
        except Exception:
            log_exc("observe attempt %d" % attempt)
            time.sleep(2.0)
    if obs_res is None:
        log("no observation; abort")
        return
    save_json("observe_reply", strip_images(obs_res))
    log("instruction:", mcpio.get_instruction(obs_j) or mcpio.get_instruction(st_j))
    log("state:", None if robot.state is None else np.round(robot.state, 4).tolist(),
        "limit:", robot.limit, "remaining:", robot.remaining)
    if robot.ended:
        log("episode already ended")
        return
    if robot.state is None:
        log("no state available; abort")
        return
    step_id = mcpio.get_step_id(obs_j)
    mpath = find_manifest([obs_j, st_j])
    log("manifest:", mpath)
    vid, exe, mfull = [], [], None
    if mpath:
        try:
            m, mfull = video.load_manifest(mpath)
            vid, exe = video.split_frames(m)
            log("frames: video=%d execution=%d sampling=%s" % (len(vid), len(exe), m.get("video_sampling")))
        except Exception:
            log_exc("manifest")
    demo = None
    try:
        demo = video.Demo(vid) if vid else None
        log(demo.summary() if demo else "no video frames")
    except Exception:
        log_exc("demo")
        demo = None
    # execution-frame depth & calibration
    dE = rgbE = calib = None
    try:
        fr = exe[-1] if exe else None
        if fr is not None:
            ff = video.frame_files(fr, mfull)
            if ("front", "depth") in ff:
                dE = vision.load_depth(*ff[("front", "depth")])
            if ("front", "rgb") in ff:
                rgbE = vision.load_rgb(ff[("front", "rgb")][0])
        if dE is not None:
            calib = calibrate(robot, obs_j, step_id if step_id is not None else 0, dE)
        log("calib:", None if calib is None else (calib.source, calib.resid))
    except Exception:
        log_exc("calibration")
    if demo is not None and demo.ok and demo.grasp_hold() is not None:
        T_peg, T_box = SE2(), SE2()
        try:
            if calib is not None and dE is not None:
                ffV = video.frame_files(vid[0], mfull)
                dV = vision.load_depth(*ffV[("front", "depth")])
                rgbV = vision.load_rgb(ffV[("front", "rgb")][0]) if ("front", "rgb") in ffV else None
                eeV = demo.S_raw[0][:3]
                res = scene.analyse(calib, dV, rgbV, dE, rgbE, demo, eeV, robot.pos)
                T_peg, T_box = res["T_peg"], res["T_box"]
                log("scene diag:", res["diag"], "T_peg:", T_peg, "T_box:", T_box)
        except Exception:
            log_exc("scene analysis")
            T_peg, T_box = SE2(), SE2()
        P, R = build_targets(demo, T_peg, T_box)
        replay(robot, demo, P, R)
        if not (robot.ended or robot.complete):
            push(robot, demo, P, R)
    elif demo is not None and demo.ok:
        # robot moves in the video but no object grasp detected: replay anyway
        log("demo without detected grasp: plain replay")
        P, R = build_targets(demo, SE2(), SE2())
        replay(robot, demo, P, R)
    else:
        fallback(robot, calib, dE, rgbE)


def main(ctx):
    try:
        run(ctx)
    except EpisodeOver:
        log("episode over / budget reached")
    except Exception:
        log_exc("main")
    write_trace()
    log("controller exit")
