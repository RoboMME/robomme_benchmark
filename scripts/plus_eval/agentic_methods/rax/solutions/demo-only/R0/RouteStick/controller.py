"""RouteStick controller (demo-only, developed offline from the three released demos).

Pipeline (each step logs; failures degrade to the next fallback, never crash):
  S0 observe      robomme_observe + robomme_status -> state, step_id, frame_sequence manifest
  S1 parse video  front frames of phase video_demo -> visited target indices + side per move
  S2 perceive     execution front frame -> target pixels -> world points (pixel_to_position),
                  table height, obstacle footprints, 'near' (towards front camera) direction
  S3 approach     travel above the first target at a safe height, descend until contact
  S4 route        for each move i->j: half-ellipse detour on the demonstrated side at contact
                  height, arrive on target j, check it turned red (log), continue
  S5 finish       hold until the episode ends (or a few steps), return
"""
import json
import math
import os
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp_io import (log, reply_meta, get_state, get_step_id, episode_ended, task_complete,  # noqa: E402
                    manifest_path, read_manifest, resolve_file, load_rgb, load_depth,
                    inline_images, pick_front, find_key, find_all)
from vision import detect_targets, red_counts  # noqa: E402
from video_parse import parse_video, wood_mask  # noqa: E402
from geometry import (tool_backproject, parse_calibration, own_backproject, rotvec_to_R,  # noqa: E402
                      R_to_rotvec, detour_offset, ellipse_path)

VERSION = "routestick-v1"

# ---------------------------------------------------------------- tunables (untested live: demo-only)
POS_SCALE = 0.05          # m per unit action (documented)
ROT_SCALE = 0.5           # rad per unit action (documented)
V_ROUTE = 0.008           # m/step along detours (expert ~0.004-0.006)
V_TRAVEL = 0.025          # m/step in free space above obstacles
V_DESCEND_FAST = 0.015
V_DESCEND_SLOW = 0.004
CLEAR_R = 0.040           # min lateral distance stick path <-> obstacle surface point
EXTRA_FAR = 0.02          # hidden back face allowance when passing on the far side
C_MIN = 0.050             # min detour bulge
C_MAX = 0.12
PRESS = 0.002             # command this far below contact height while drawing
ARRIVE_TOL = 0.004
CHUNK = 3
OUT_DIR = "/workspace/output"


class Robot:
    def __init__(self, ctx):
        self.ctx = ctx
        self.state = None
        self.step_id = None
        self.ended = False
        self.complete = False
        self.last_result = None
        self.R0 = None
        self.max_steps = None
        self.steps_sent = 0
        self.front_ref = None
        self.resp = []        # achieved/commanded displacement ratios in free motion

    def observe(self):
        res = self.ctx.call("robomme_observe")
        self._absorb(res)
        return res

    def _absorb(self, res):
        self.last_result = res
        meta = reply_meta(res)
        st = get_state(meta)
        if st is not None:
            self.state = np.array(st, float)
        sid = get_step_id(meta)
        if sid is not None:
            self.step_id = sid
        if episode_ended(meta):
            self.ended = True
        if task_complete(meta):
            self.complete = True
        return meta

    def response_ratio(self):
        if len(self.resp) < 2:
            return 1.0
        return float(np.clip(np.median(self.resp[-10:]), 0.1, 1.2))

    @property
    def pos(self):
        return self.state[:3].copy()

    def rot_correction(self):
        if self.R0 is None or self.state is None:
            return np.zeros(3)
        R = rotvec_to_R(self.state[3:6])
        err = R_to_rotvec(self.R0 @ R.T)
        en = float(np.linalg.norm(err))
        if en > 0.35 and not getattr(self, "_rot_disabled", False):
            log("orientation error", round(en, 3), "-> disabling orientation correction")
            self._rot_disabled = True
        if getattr(self, "_rot_disabled", False) or en < 0.04:
            return np.zeros(3)
        return np.clip(0.5 * err / ROT_SCALE, -0.2, 0.2)

    def send(self, deltas):
        """deltas: list of 3-vectors (metres) -> action rows; first row carries an
        orientation correction."""
        if self.ended:
            return None
        rows = []
        rc = self.rot_correction()
        for k, d in enumerate(deltas):
            a = np.clip(np.asarray(d, float) / POS_SCALE, -1.0, 1.0)
            r = rc if k == 0 else np.zeros(3)
            rows.append([float(a[0]), float(a[1]), float(a[2]), float(r[0]), float(r[1]), float(r[2]), -1.0])
        if self.max_steps is not None and self.step_id is not None:
            left = self.max_steps - self.step_id
            rows = rows[:max(1, left)]   # the environment, not this estimate, ends the episode
        try:
            res = self.ctx.call("robomme_step", actions=rows)
        except Exception as e:
            log("robomme_step error:", repr(e)[:400])
            # do not retry blindly: re-read state
            try:
                self.observe()
            except Exception as e2:
                log("observe after error failed:", repr(e2)[:200])
            return None
        self.steps_sent += len(rows)
        self._absorb(res)
        return res

    def front_image(self):
        ims = inline_images(self.last_result) if self.last_result is not None else []
        return pick_front(ims, self.front_ref)


# ---------------------------------------------------------------- helpers
def find_initial_manifest(meta):
    p = manifest_path(meta)
    if p:
        return p
    root = "/workspace/runtime/autonomous_controller/runs"
    best, bt = None, -1
    for dp, dn, fn in os.walk(root):
        if "manifest.json" in fn:
            fp = os.path.join(dp, "manifest.json")
            try:
                t = os.path.getmtime(fp)
                with open(fp) as f:
                    txt = f.read(200000)
                if "video_demo" in txt and t > bt:
                    best, bt = fp, t
            except Exception:
                pass
    return best


def parse_limits(meta):
    lim = find_key(meta, "episode_limits")
    cands = []
    if isinstance(lim, dict):
        for k, v in lim.items():
            kl = k.lower()
            if isinstance(v, int) and "step" in kl and ("max" in kl or "limit" in kl) \
                    and "call" not in kl and "row" not in kl and "per" not in kl:
                cands.append(v)
    for k in ("max_steps", "step_limit", "max_episode_steps", "episode_step_limit"):
        v = find_key(meta, k)
        if isinstance(v, int):
            cands.append(v)
    cands = [c for c in cands if c > 150]   # ignore per-call row limits (100)
    return min(cands) if cands else None


def target_world_points(robot, step_id, Timg, depth_info, calib):
    pix = [(x, y) for (x, y) in Timg]
    W = tool_backproject(robot.ctx, step_id, pix, "front", 1)
    if (any(w is None for w in W)) and depth_info is not None and calib is not None:
        own = own_backproject(depth_info, calib, pix, 1)
        W = [w if w is not None else o for w, o in zip(W, own)]
    return W


def backproject(robot, step_id, pix, depth_info, calib, radius=1):
    W = tool_backproject(robot.ctx, step_id, pix, "front", radius) if pix else []
    if pix and any(w is None for w in W) and depth_info is not None and calib is not None:
        own = own_backproject(depth_info, calib, pix, radius)
        W = [w if w is not None else o for w, o in zip(W, own)]
    return W


def img_frame(Pi, Pj):
    d = np.array(Pj, float) - np.array(Pi, float)
    L = float(np.linalg.norm(d))
    u = d / max(L, 1e-6)
    n = np.array([-u[1], u[0]])
    if n[1] < 0:
        n = -n
    return u, n, L


# ---------------------------------------------------------------- perception of the execution scene
def perceive_scene(robot, front, depth_m, calib, n_expected):
    info = {}
    Timg = detect_targets(front, top_exclude=20)
    if n_expected and len(Timg) != n_expected:
        log(f"exec target count {len(Timg)} != video {n_expected}")
    info["targets_img"] = Timg
    sid = robot.step_id if robot.step_id is not None else 0
    W = backproject(robot, sid, Timg, depth_m, calib, 1)
    info["targets_world"] = [None if w is None else w.tolist() for w in W]
    wood = wood_mask(front)
    # table samples: wood pixels just below / above each target in the image
    tab_pix = []
    for k, P in enumerate(Timg):
        if k + 1 < len(Timg):
            u, n, L = img_frame(P, Timg[k + 1])
        elif k > 0:
            u, n, L = img_frame(Timg[k - 1], P)
        else:
            u, n, L = np.array([1.0, 0]), np.array([0, 1.0]), 30.0
        for off in (7, 10, -7):
            q = np.array(P) + off * n
            qi = (int(round(q[0])), int(round(q[1])))
            if 0 <= qi[0] < front.shape[1] and 0 <= qi[1] < front.shape[0] and wood[qi[1], qi[0]]:
                tab_pix.append((k, off, (float(q[0]), float(q[1]))))
    TW = backproject(robot, sid, [p for _, _, p in tab_pix], depth_m, calib, 1)
    zs = [w[2] for w in TW if w is not None]
    tz = [w[2] for w in W if w is not None]
    table_z = float(np.median(zs)) if zs else (float(np.median(tz)) - 0.003 if tz else None)
    info["table_z"] = table_z
    # near direction per gap from table samples below (+) and above (-) the line
    near_votes = {}
    for (k, off, p), w in zip(tab_pix, TW):
        if w is None:
            continue
        for g in (k - 1, k):
            if g < 0 or g + 1 >= len(W) or W[g] is None or W[g + 1] is None:
                continue
            Wa, Wb = W[g][:2], W[g + 1][:2]
            d = Wb - Wa
            perp = np.array([-d[1], d[0]]) / max(np.linalg.norm(d), 1e-6)
            s = float((w[:2] - W[k][:2]) @ perp)
            vote = np.sign(s) * np.sign(off)
            near_votes.setdefault(g, []).append(vote)
    info["near_votes"] = {g: v for g, v in near_votes.items()}
    # obstacles: non-wood, non-target pixels between consecutive targets
    nonwood = ~wood
    nonwood[:22] = False
    h, w_ = front.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w_]
    for (x, y) in Timg:
        nonwood &= ((xx - x) ** 2 + (yy - y) ** 2 > 49)
    foot = []
    obstacles = []
    from vision import label
    lab, nl = label(nonwood)
    for g in range(len(Timg) - 1):
        P, Q = np.array(Timg[g]), np.array(Timg[g + 1])
        u, n, L = img_frame(P, Q)
        v = np.stack([xx - P[0], yy - P[1]], -1)
        t = (v @ u) / L
        s = v @ n
        central = (t > 0.25) & (t < 0.75) & (np.abs(s) < 0.45 * L) & nonwood
        ids = np.unique(lab[central])
        ids = [i for i in ids if i > 0]
        mask = np.isin(lab, ids) & (t > -0.2) & (t < 1.2) & (s > -2.5 * L) & (s < 1.0 * L)
        # erode by one pixel: silhouette pixels mix obstacle and background depth
        er = mask.copy()
        er[1:] &= mask[:-1]; er[:-1] &= mask[1:]; er[:, 1:] &= mask[:, :-1]; er[:, :-1] &= mask[:, 1:]
        if er.sum() >= 10:
            mask = er
        ys, xs = np.nonzero(mask)
        pts = []
        if len(xs):
            sel = np.linspace(0, len(xs) - 1, min(64, len(xs))).astype(int)
            pix = [(float(xs[i]), float(ys[i])) for i in sel]
            OW = backproject(robot, sid, pix, depth_m, calib, 0)
            for o in OW:
                if o is not None and (table_z is None or o[2] > table_z + 0.006):
                    pts.append(o)
        obstacles.append(dict(gap=g, n_pix=int(len(xs)), n_pts=len(pts),
                              top_z=float(max(p[2] for p in pts)) if pts else None))
        foot.extend(pts)
    info["obstacles"] = obstacles
    info["footprints"] = [p.tolist() for p in foot]
    return info


# ---------------------------------------------------------------- motion primitives
def move_linear(robot, goal, v, tol=0.003, max_steps=200):
    """Straight-line closed-loop move of the tip to goal (xyz)."""
    goal = np.asarray(goal, float)
    used = 0
    stall = 0
    while used < max_steps and not robot.ended:
        p = robot.pos
        err = goal - p
        dist = float(np.linalg.norm(err))
        if dist < tol:
            return True
        nsteps = min(CHUNK, max(1, int(math.ceil(dist / v))))
        deltas = []
        cur = p.copy()
        for k in range(nsteps):
            e = goal - cur
            dn = np.linalg.norm(e)
            step = e if dn <= v else e / dn * v
            deltas.append(step)
            cur = cur + step
        before = p
        robot.send(deltas)
        used += len(deltas)
        moved = float(np.linalg.norm(robot.pos - before))
        commanded = float(sum(np.linalg.norm(d) for d in deltas))
        if dist > 2 * commanded and commanded > 1e-4:
            robot.resp.append(moved / commanded)
        if moved < 0.2 * robot.response_ratio() * min(dist, v * len(deltas)):
            stall += 1
            if stall >= 4:
                log("move_linear stalled at", np.round(robot.pos, 4).tolist(), "goal", np.round(goal, 4).tolist())
                return False
        else:
            stall = 0
    return float(np.linalg.norm(goal - robot.pos)) < 3 * tol


def descend_to_contact(robot, z_floor, z_slow_start, z_accept):
    """Descend vertically: fast to z_slow_start, then slowly until the tip stops
    (contact) or z_floor is reached.  A stall above z_accept is only accepted
    after it persists.  Returns the contact height."""
    p = robot.pos
    if p[2] > z_slow_start:
        move_linear(robot, [p[0], p[1], z_slow_start], V_DESCEND_FAST, tol=0.003, max_steps=60)
    stalls = 0
    for _ in range(40):
        if robot.ended:
            break
        z0 = robot.pos[2]
        if z0 <= z_floor:
            log("descend reached floor", round(z0, 4))
            return z0
        robot.send([[0, 0, -V_DESCEND_SLOW], [0, 0, -V_DESCEND_SLOW]])
        dz = z0 - robot.pos[2]
        if dz < 0.25 * robot.response_ratio() * 2 * V_DESCEND_SLOW:
            stalls += 1
            if robot.pos[2] < z_accept or stalls >= 4:
                log("contact detected at z", round(robot.pos[2], 4), "dz", round(dz, 5), "stalls", stalls)
                return robot.pos[2]
        else:
            stalls = 0
    return robot.pos[2]


def follow_path(robot, path_xy, z_low, v, max_steps):
    """Follow xy waypoints at constant tip height z_low (pressing slightly)."""
    k = 0
    used = 0
    n = len(path_xy)
    stall = 0
    while used < max_steps and not robot.ended:
        p = robot.pos
        # advance the progress index to the nearest waypoint ahead
        look = path_xy[k:min(n, k + 8)]
        dists = np.linalg.norm(look - p[:2], axis=1)
        k = k + int(np.argmin(dists))
        if k >= n - 1 and float(np.linalg.norm(path_xy[-1] - p[:2])) < ARRIVE_TOL:
            return True
        deltas = []
        cur = p[:2].copy()
        for j in range(CHUNK):
            idx = min(n - 1, k + 1 + j)
            tgt = path_xy[idx]
            e = tgt - cur
            dn = float(np.linalg.norm(e))
            lim = v * 1.5
            if dn > lim:
                e = e / dn * lim
            dz = (z_low - p[2]) if j == 0 else 0.0
            deltas.append(np.array([e[0], e[1], np.clip(dz, -0.01, 0.01)]))
            cur = cur + e
            if idx == n - 1 and float(np.linalg.norm(path_xy[-1] - cur)) < 1e-4:
                break
        before = p
        robot.send(deltas)
        used += len(deltas)
        moved = float(np.linalg.norm(robot.pos[:2] - before[:2]))
        if moved < 0.15 * robot.response_ratio() * float(sum(np.linalg.norm(d[:2]) for d in deltas)):
            stall += 1
            if stall >= 5:
                log("follow_path stalled at", np.round(robot.pos, 4).tolist())
                return False
        else:
            stall = 0
    return False


def settle_xy(robot, goal_xy, z_low, max_chunks=4):
    """Fine xy alignment over a target at drawing height (z only pressed)."""
    for _ in range(max_chunks):
        if robot.ended:
            return False
        p = robot.pos
        e = np.asarray(goal_xy, float) - p[:2]
        if float(np.linalg.norm(e)) < ARRIVE_TOL:
            return True
        n = float(np.linalg.norm(e))
        e1 = e if n <= 0.006 else e / n * 0.006
        rest = e - e1
        n2 = float(np.linalg.norm(rest))
        e2 = rest if n2 <= 0.006 else rest / n2 * 0.006
        dz = float(np.clip(z_low - p[2], -0.01, 0.01))
        robot.send([[e1[0], e1[1], dz], [e2[0], e2[1], 0.0]])
    return float(np.linalg.norm(np.asarray(goal_xy) - robot.pos[:2])) < 2 * ARRIVE_TOL


def red_at(robot, Timg, k, base):
    fr = robot.front_image()
    if fr is None:
        return None
    try:
        c = red_counts(fr, [Timg[k]], 8.0)[0] - base[k]
        return int(c)
    except Exception:
        return None


# ---------------------------------------------------------------- main pipeline
def run(ctx, logbook):
    t_start = time.time()
    robot = Robot(ctx)
    res = robot.observe()
    meta0 = reply_meta(res)
    logbook["instruction"] = find_key(meta0, "instruction")
    try:
        st = ctx.call("robomme_status")
        smeta = reply_meta(st)
        robot.max_steps = parse_limits(smeta) or parse_limits(meta0)
        if robot.ended is False and episode_ended(smeta):
            robot.ended = True
    except Exception as e:
        log("robomme_status failed", repr(e)[:200])
        robot.max_steps = parse_limits(meta0)
    log(VERSION, "state", None if robot.state is None else np.round(robot.state, 4).tolist(),
        "step_id", robot.step_id, "max_steps", robot.max_steps)
    if robot.state is None:
        raise RuntimeError("no state in observation")
    robot.R0 = rotvec_to_R(robot.state[3:6])

    # ---------------- S1: conditioning video
    mpath = find_initial_manifest(meta0)
    logbook["manifest"] = mpath
    video_fronts, exec_front, exec_depth = [], None, None
    video_states = []
    if mpath:
        man, frames = read_manifest(mpath)
        for fr in frames:
            f = fr["files"].get(("front", "rgb"))
            if f is None:
                continue
            if fr["phase"] == "video_demo":
                try:
                    video_fronts.append(load_rgb(resolve_file(f["path"], mpath)))
                    video_states.append(fr.get("state"))
                except Exception as e:
                    log("load video frame failed", repr(e)[:120])
            else:
                try:
                    exec_front = load_rgb(resolve_file(f["path"], mpath))
                    fd = fr["files"].get(("front", "depth"))
                    if fd is not None:
                        exec_depth = load_depth(resolve_file(fd["path"], mpath), fd.get("mpu", 0.001))
                except Exception as e:
                    log("load exec frame failed", repr(e)[:120])
    log("video frames", len(video_fronts), "exec frame", exec_front is not None, "t", round(time.time() - t_start, 1))
    try:
        vs = np.array([v[:3] for v in video_states if isinstance(v, list) and len(v) >= 3], float)
        if len(vs):
            logbook["video_state_std"] = vs.std(0).tolist()
            log("video state std", np.round(vs.std(0), 4).tolist(), "n", len(vs))
    except Exception:
        pass
    if exec_front is None:
        exec_front = pick_front(inline_images(res), None)
    robot.front_ref = exec_front
    if not video_fronts:
        raise RuntimeError("no conditioning video frames found")
    parsed = parse_video(video_fronts)
    visits = parsed["visits"]
    moves = parsed["moves"]
    logbook["video"] = dict(targets=parsed["targets"], visits=visits, onsets=parsed["onsets"],
                            moves=moves, debug=parsed["debug"])
    log("video visits", visits, "sides", [m[2] for m in moves], "conf", [round(m[3], 2) for m in moves],
        "t", round(time.time() - t_start, 1))
    if not visits:
        raise RuntimeError("no visits parsed from the video")

    # ---------------- S2: perceive execution scene
    calib = parse_calibration(meta0, "front")
    scene = perceive_scene(robot, exec_front, exec_depth, calib, len(parsed["targets"]))
    logbook["scene"] = {k: v for k, v in scene.items() if k != "footprints"}
    logbook["n_footprints"] = len(scene["footprints"])
    Timg = scene["targets_img"]
    TW = [None if w is None else np.array(w) for w in scene["targets_world"]]
    log("exec targets img", [tuple(round(c) for c in p) for p in Timg])
    log("exec targets world", [None if w is None else np.round(w, 3).tolist() for w in TW])
    log("table_z", scene["table_z"], "obstacles", scene["obstacles"], "t", round(time.time() - t_start, 1))
    ntar = len(Timg)
    good = [k for k, w in enumerate(TW) if w is not None]
    if ntar >= 3 and 2 <= len(good) < ntar:
        # targets are collinear and evenly spaced in the world: fill gaps by a linear fit over index
        A = np.array([[k, 1.0] for k in good])
        Y = np.array([TW[k] for k in good])
        coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
        for k in range(ntar):
            if TW[k] is None:
                TW[k] = coef[0] * k + coef[1]
        log("filled missing target points by line fit", [k for k in range(ntar) if k not in good])
        logbook["filled_targets"] = [k for k in range(ntar) if k not in good]
    if ntar < 2 or any(w is None for w in TW):
        raise RuntimeError("target localisation failed")
    table_z = scene["table_z"] if scene["table_z"] is not None else float(np.median([w[2] for w in TW]))
    foot = [np.array(p) for p in scene["footprints"]]
    foot_xy = [p[:2] for p in foot]
    top_z = max([p[2] for p in foot], default=table_z + 0.15)
    # global near direction fallback: majority of per-gap votes
    def near_vec(a, b):
        lo, hi = min(a, b), max(a, b)   # votes are relative to the lo->hi perpendicular
        d = TW[hi][:2] - TW[lo][:2]
        perp = np.array([-d[1], d[0]]) / max(np.linalg.norm(d), 1e-6)
        votes = []
        for g in range(min(a, b), max(a, b)):
            votes += scene["near_votes"].get(g, [])
        if not votes:
            for g, v in scene["near_votes"].items():
                votes += v
        sgn = np.sign(sum(votes)) if votes else 0
        if sgn == 0 and calib is not None:
            cam = calib[2][:2]
            M = (TW[a][:2] + TW[b][:2]) / 2
            sgn = np.sign((cam - M) @ perp) or 1
        if sgn == 0:
            sgn = 1
        return perp * sgn

    # map video target indices to execution indices (same left-to-right order)
    nvid = len(parsed["targets"])

    def mapi(i):
        if nvid == ntar:
            return i
        return int(min(ntar - 1, max(0, round(i * (ntar - 1) / max(1, nvid - 1)))))

    plan = []
    for (i, j, side, conf) in moves:
        a, b = mapi(i), mapi(j)
        if a == b:
            continue
        plan.append((a, b, side))
    start = mapi(visits[0])
    logbook["plan"] = dict(start=start, moves=plan)
    log("plan start", start, "moves", plan)

    # step budget -> speed
    v_route = V_ROUTE
    if robot.max_steps:
        est_len = sum(0.35 for _ in plan)
        left = robot.max_steps - (robot.step_id or 0) - 120
        if left > 0 and est_len / v_route > 0.8 * left:
            v_route = min(0.03, est_len / (0.8 * left))
            log("speeding up route to", round(v_route, 4))

    base_red = red_counts(exec_front, Timg, 8.0)

    # ---------------- S3: approach first target
    z_travel = max(robot.pos[2], top_z + 0.10, table_z + 0.20)
    p = robot.pos
    if p[2] < z_travel - 0.01:
        move_linear(robot, [p[0], p[1], z_travel], V_TRAVEL, tol=0.005, max_steps=40)
    S = TW[start]
    move_linear(robot, [S[0], S[1], z_travel], V_TRAVEL, tol=0.004, max_steps=120)
    move_linear(robot, [S[0], S[1], table_z + 0.04], V_DESCEND_FAST, tol=0.003, max_steps=80)
    z_contact = descend_to_contact(robot, z_floor=table_z - 0.015, z_slow_start=table_z + 0.03,
                                   z_accept=table_z + 0.012)
    if z_contact < table_z + 0.012:
        z_low = min(z_contact, table_z + 0.002) - PRESS   # TCP is the stick tip
    else:
        z_low = z_contact - PRESS                         # tip below TCP: trust contact
    log("response ratio", round(robot.response_ratio(), 3), "samples", len(robot.resp))
    log("start reached", np.round(robot.pos, 4).tolist(), "z_contact", round(z_contact, 4),
        "red", red_at(robot, Timg, start, base_red), "step", robot.step_id, "t", round(time.time() - t_start, 1))
    logbook["z_contact"] = z_contact

    # ---------------- S4: route
    for mi, (a, b, side) in enumerate(plan):
        if robot.ended:
            break
        Pa, Pb = TW[a][:2].copy(), TW[b][:2].copy()
        # start from where the tip actually is
        cur = robot.pos[:2]
        if np.linalg.norm(cur - Pa) < 0.03:
            Pa = cur
        nv = near_vec(a, b)
        sv = nv if side == "near" else -nv
        extra = EXTRA_FAR if side == "far" else 0.0
        c, c_raw = detour_offset(Pa, Pb, sv, foot_xy, CLEAR_R, C_MIN, C_MAX, extra)
        path, plen = ellipse_path(Pa, Pb, sv, c, v_route)
        if robot.max_steps and robot.step_id is not None:
            left = robot.max_steps - robot.step_id - 15
            moves_left = len(plan) - mi
            ratio = robot.response_ratio()
            need = moves_left * (plen / (v_route * ratio) + 14)
            if need > 0.9 * left and left > 0:
                v_new = min(0.03, v_route * need / (0.9 * left))
                if v_new > v_route * 1.05:
                    log(f"budget: {left} steps left for {moves_left} moves -> v_route {v_route:.4f}->{v_new:.4f}")
                    v_route = v_new
                    path, plen = ellipse_path(Pa, Pb, sv, c, v_route)
        log(f"move {mi}: {a}->{b} side {side} c={c:.3f} (raw {c_raw:.3f}) len={plen:.3f} steps~{len(path)}")
        ok = follow_path(robot, path, z_low, v_route, max_steps=int(len(path) * 3 + 30))
        # fine arrival on the target centre
        settle_xy(robot, TW[b][:2], z_low)
        if not robot.ended:
            robot.send([[0, 0, z_low - robot.pos[2]]])
        r = red_at(robot, Timg, b, base_red)
        if r is not None and r < 6 and not robot.ended:
            # nudge: press a little more and re-check
            robot.send([[0, 0, -0.003], [0, 0, -0.003]])
            r = red_at(robot, Timg, b, base_red)
        log(f"  arrived {ok} pos {np.round(robot.pos, 4).tolist()} red {r} step {robot.step_id} ended {robot.ended}"
            f" t {round(time.time() - t_start, 1)}")
        logbook.setdefault("moves_done", []).append(dict(a=a, b=b, side=side, c=c, ok=ok, red=r,
                                                         step=robot.step_id, pos=robot.pos.tolist()))

    # ---------------- S5: finish
    hold = 0
    while not robot.ended and hold < 10:
        robot.send([[0, 0, 0]])
        hold += 1
    log("done: ended", robot.ended, "complete", robot.complete, "step", robot.step_id,
        "t", round(time.time() - t_start, 1))
    logbook["final"] = dict(ended=robot.ended, complete=robot.complete, step=robot.step_id)


def main(ctx):
    logbook = {"version": VERSION}
    try:
        run(ctx, logbook)
    except Exception as e:
        logbook["error"] = traceback.format_exc()
        log("controller error:", traceback.format_exc())
    finally:
        try:
            os.makedirs(OUT_DIR, exist_ok=True)
            with open(os.path.join(OUT_DIR, "routestick_log.json"), "w") as f:
                json.dump(logbook, f, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o), indent=1)
        except Exception:
            pass
