"""ButtonUnmask controller v1.

Phases:
  P0 locate button (front heightmap, step 0)
  P1 go above button, watch scene (track cubes by colour) until 1st containers or step>=46
  P2 press button (descend until blocked), verify (TCP stops above table), rise
  P3 hover high, watch until all target colours hidden by containers (or timeout)
  P4 for each target colour in instruction order: refine container with wrist view, grasp, lift,
     put back, release.  Stop when episode ends.
Transitions: any phase -> stop when episode_ended. Exceptions logged; never retry an uncertain motion blindly.
"""
import math
import time
import traceback

import numpy as np
import cv2

import rlib
import percep

VERSION = "bu-v2"
GAIN = 2.6              # measured: arm achieves ~35-40% of commanded delta per step
VMAX = 0.016            # m/step achievable
HOVER_BTN_Z = 0.25     # v2: high enough that wrist cam also watches the table
PRESS_START_STEP = 46   # ep1: containers at 44, press 48-54 worked; deadline 90
PRESS_LATEST = 70
HOVER_HIGH = (-0.04, 0.0, 0.42)
GRASP_Z = 0.048         # TCP z for container grasp (top 0.073), ep1 worked
APPROACH_Z = 0.20
LIFT_Z = 0.22
PLACE_Z = 0.052
WAIT_MAX_STEP = 300
MATCH_R = 0.035

T0 = time.time()


def log(*a):
    print(f"[{VERSION} t={time.time() - T0:6.1f}]", *a, flush=True)


class Ended(Exception):
    pass


def yaw_of(obs):
    R, _ = cv2.Rodrigues(np.asarray(obs.state[3:6], float))
    return math.atan2(R[1, 0], R[0, 0])


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Robot:
    def __init__(self, ctx):
        self.ctx = ctx
        self.obs = rlib.observe(ctx)
        self.watch = None  # callback(obs)

    def step(self, rows):
        if self.obs.remaining is not None and self.obs.remaining <= 0:
            raise Ended()
        o = rlib.step(self.ctx, rows)
        self.obs = o
        if self.watch:
            try:
                self.watch(o)
            except Exception:
                log("watch error", traceback.format_exc())
        if o.ended:
            log("EPISODE ENDED step", o.step_id, "task_complete", o.task_complete)
            raise Ended()
        return o

    def hold(self, grip, n):
        return self.step([[0, 0, 0, 0, 0, 0, grip]] * n)

    def servo(self, target, grip, yaw=None, tol=0.004, ytol=0.03, max_calls=25, chunk=5, vmax=VMAX):
        target = np.asarray(target, float)
        for _ in range(max_calls):
            o = self.obs
            err = target - o.pos
            dist = float(np.linalg.norm(err))
            yerr = 0.0 if yaw is None else wrap(yaw - yaw_of(o))
            if dist < tol and abs(yerr) < ytol:
                return True
            est = max(1, int(math.ceil(dist / vmax)), int(math.ceil(abs(yerr) / 0.15)))
            n = min(chunk, est)
            v = err / est
            cmd = np.clip(v * GAIN / rlib.POS_SCALE, -1, 1)
            rz = float(np.clip(yerr / est * GAIN / rlib.ROT_SCALE, -1, 1))
            self.step([[cmd[0], cmd[1], cmd[2], 0, 0, rz, grip]] * n)
        log("servo: not converged", np.round(self.obs.pos, 4).tolist(), "target", np.round(target, 4).tolist())
        return False


class Tracker:
    """Keeps last-seen position of each coloured cube and current containers."""

    def __init__(self, button_xy):
        self.button_xy = button_xy
        self.last = {}        # colour -> (xy, step)
        self.tracks = []      # dicts colour, xy, step
        self.containers = []  # list of xy from latest good frame
        self.n_cont_hist = []
        self.last_change = 0
        self.first_container_step = None

    def update(self, o):
        excl = [(self.button_xy, 0.04)]
        cams = ['front']
        if o.pos[2] > 0.20:
            cams.append('wrist')
        else:
            excl.append((o.pos[:2], 0.08))
        dets = percep.detect(o, tuple(cams), exclude=excl)
        conts = [d['xy'] for d in dets if d['kind'] == 'container' and d['ztop'] < 0.10]
        cubes = [d for d in dets if d['kind'] == 'cube' and d['ztop'] < 0.05]
        for c in cubes:
            self.last[c['color']] = (c['xy'], o.step_id)
            for t in self.tracks:
                if t['color'] == c['color'] and np.linalg.norm(t['xy'] - c['xy']) < 0.025:
                    t['xy'], t['step'] = c['xy'], o.step_id
                    break
            else:
                self.tracks.append(dict(color=c['color'], xy=c['xy'], step=o.step_id))
        if len(conts) != (len(self.containers) if self.containers is not None else -1):
            self.last_change = o.step_id
        if conts and self.first_container_step is None:
            self.first_container_step = o.step_id
        self.containers = conts
        self.cubes_now = [(c['color'], c['xy']) for c in cubes]
        log(f"track s{o.step_id} cams={cams} cubes={[(c, np.round(xy, 3).tolist()) for c, xy in self.cubes_now]} "
            f"n_cont={len(conts)}")

    def hidden(self, color):
        ts = [t for t in self.tracks if t['color'] == color]
        if not ts:
            return False
        for t in ts:
            xy = t['xy']
            visible = any(c == color and np.linalg.norm(x - xy) < 0.02 for c, x in self.cubes_now)
            cov = any(np.linalg.norm(c - xy) < MATCH_R for c in self.containers)
            if visible or not cov:
                return False
        return True

    def containers_for(self, color):
        out = []
        for t in sorted([t for t in self.tracks if t['color'] == color], key=lambda t: -t['step']):
            if not self.containers:
                break
            d = [np.linalg.norm(c - t['xy']) for c in self.containers]
            i = int(np.argmin(d))
            if d[i] < MATCH_R and not any(np.linalg.norm(self.containers[i] - u) < 0.01 for u in out):
                out.append(self.containers[i])
        return out

    def container_for(self, color):
        if color not in self.last or not self.containers:
            return None
        xy = self.last[color][0]
        d = [np.linalg.norm(c - xy) for c in self.containers]
        i = int(np.argmin(d))
        return self.containers[i] if d[i] < MATCH_R else None


def find_button(o):
    dets = percep.detect(o, ('front',))
    cands = [d for d in dets if 0.015 < d['ztop'] < 0.06]
    if not cands:
        return None
    # the button is the only object at reset; prefer the one closest to the robot side
    cands.sort(key=lambda d: -d['area'])
    return np.array(cands[0]['xy'])


def wrist_container(o, expect_xy, r=0.045):
    """Refine container (top face) centre and yaw from the wrist cloud."""
    P = o.cloud('wrist')
    z = P[..., 2]
    m = (z > 0.058) & (z < 0.095) & (o.depth['wrist'] > 0.10)
    n, lab = cv2.connectedComponents(m.astype(np.uint8))
    best = None
    for k in range(1, n):
        pts = P[lab == k]
        if len(pts) < 30:
            continue
        top = pts[pts[:, 2] > np.max(pts[:, 2]) - 0.01]
        c = top[:, :2].mean(0)
        d = np.linalg.norm(c - expect_xy)
        if d < r and (best is None or d < best[0]):
            rect = cv2.minAreaRect(top[:, :2].astype(np.float32))
            best = (d, np.array(rect[0], float), rect[1], rect[2], len(top))
    if best is None:
        return None
    d, c, size, ang, npt = best
    yaw = math.radians(((ang + 45) % 90) - 45)
    return dict(xy=c, yaw=yaw, size=size, n=npt, dist=d)


def press_button(rb, bxy):
    rb.servo([bxy[0], bxy[1], 0.06], 1, yaw=0.0, tol=0.004)
    zs = []
    for _ in range(3):
        rb.step([[0, 0, -0.35, 0, 0, 0, 1]] * 4)
        zs.append(rb.obs.pos[2])
    zmin = min(zs)
    rb.hold(1, 2)
    log("press: zmin", round(float(zmin), 4), "xy", np.round(rb.obs.pos[:2], 4).tolist())
    rb.servo([bxy[0], bxy[1], 0.10], 1, tol=0.01)
    return zmin


def pick_container(rb, color, cxy):
    log(f"pick {color}: container guess {np.round(cxy, 3).tolist()}")
    rb.servo([cxy[0], cxy[1], APPROACH_Z], -1, yaw=0.0, tol=0.006)
    ref = wrist_container(rb.obs, cxy)
    if ref is None:
        log("wrist refine failed; using tracker xy")
        gxy, gyaw = np.asarray(cxy), 0.0
    else:
        log("wrist refine", np.round(ref['xy'], 4).tolist(), "yaw deg", round(math.degrees(ref['yaw']), 1),
            "size", np.round(ref['size'], 3).tolist(), "dist", round(float(ref['dist']), 3))
        gxy, gyaw = ref['xy'], ref['yaw']
        if ref['dist'] > 0.02:  # recentre once and re-measure
            rb.servo([gxy[0], gxy[1], APPROACH_Z], -1, yaw=0.0, tol=0.006)
            ref2 = wrist_container(rb.obs, gxy, r=0.03)
            if ref2 is not None:
                gxy, gyaw = ref2['xy'], ref2['yaw']
    rb.servo([gxy[0], gxy[1], 0.12], -1, yaw=gyaw, tol=0.004)
    rb.servo([gxy[0], gxy[1], GRASP_Z], -1, yaw=gyaw, tol=0.004, vmax=0.010)
    rb.hold(1, 10)
    g = rb.obs.grip
    log("closed grip", round(g, 4), "pos", np.round(rb.obs.pos, 4).tolist())
    rb.servo([gxy[0], gxy[1], LIFT_Z], 1, yaw=gyaw, tol=0.01, vmax=0.012)
    rb.hold(1, 6)
    g2 = rb.obs.grip
    log("lifted grip", round(g2, 4), "pos", np.round(rb.obs.pos, 4).tolist())
    held = g2 > 0.01
    # put it back where it was
    rb.servo([gxy[0], gxy[1], PLACE_Z + 0.03], 1, yaw=gyaw, tol=0.006, vmax=0.012)
    rb.servo([gxy[0], gxy[1], PLACE_Z], 1, yaw=gyaw, tol=0.005, vmax=0.008)
    rb.hold(-1, 8)
    rb.servo([gxy[0], gxy[1], APPROACH_Z], -1, yaw=gyaw, tol=0.01)
    return held, gxy


def main(ctx):
    log("start")
    rb = Robot(ctx)
    o = rb.obs
    instr = o.meta.get("instruction", "")
    tfull = percep.parse_targets_full(instr)
    targets = [c for c, _ in tfull]
    log("instruction:", instr, "targets:", targets, "limit", o.limit)
    try:
        bxy = find_button(o)
        log("button xy", None if bxy is None else np.round(bxy, 4).tolist())
        if bxy is None:
            bxy = np.array([-0.2, -0.02])
        tr = Tracker(bxy)
        tr.update(o)
        rb.watch = tr.update
        # P1: go above the button and watch
        rb.servo([bxy[0], bxy[1], HOVER_BTN_Z], 1, yaw=0.0, tol=0.006)
        # refine button with wrist (top-down view from above it)
        try:
            P = rb.obs.cloud('wrist')
            m = (P[..., 2] > 0.02) & (P[..., 2] < 0.045) & (np.hypot(P[..., 0] - bxy[0], P[..., 1] - bxy[1]) < 0.035)
            if m.sum() > 20:
                nb = np.median(P[m][:, :2], 0)
                log("button wrist refine", np.round(nb, 4).tolist(), "n", int(m.sum()))
                if np.linalg.norm(nb - bxy) < 0.03:
                    bxy = nb
                    tr.button_xy = bxy
        except Exception:
            log("button refine error", traceback.format_exc())
        while rb.obs.step_id < PRESS_START_STEP and (tr.first_container_step is None or rb.obs.step_id < 40):
            rb.hold(1, min(5, PRESS_START_STEP - rb.obs.step_id))
        # P2: press
        zmin = press_button(rb, bxy)
        if zmin < 0.015 and rb.obs.step_id < PRESS_LATEST:
            log("press may have missed (reached table); retry")
            zmin = press_button(rb, bxy)
        # P3: hover high and watch until targets hidden
        rb.servo(HOVER_HIGH, -1, yaw=0.0, tol=0.01)
        while rb.obs.step_id < WAIT_MAX_STEP:
            allh = all(tr.hidden(c) for c in targets)
            if allh and rb.obs.step_id - tr.last_change >= 10 and not tr.cubes_now:
                break
            if allh and rb.obs.step_id - tr.last_change >= 40:
                break
            rb.hold(-1, 5)
        log("wait done step", rb.obs.step_id, "hidden", {c: tr.hidden(c) for c in targets},
            "tracks", [(t['color'], np.round(t['xy'], 3).tolist(), t['step']) for t in tr.tracks])
        rb.watch = None
        # P4: pick in order
        plan = []
        for c, want_all in tfull:
            cs = tr.containers_for(c)
            if not cs:
                plan.append((c, None))
            elif want_all:
                plan.extend((c, x) for x in cs)
            else:
                plan.append((c, cs[0]))
        log("plan", [(c, None if x is None else np.round(x, 3).tolist()) for c, x in plan])
        used = []
        for c, cxy in plan:
            if cxy is None:
                # fallback: a container not matched to any seen colour
                others = [x for x in tr.containers if all(np.linalg.norm(x - tr.last[k][0]) > MATCH_R for k in tr.last)]
                if not others:
                    log("no container for", c, "- skipping")
                    continue
                cxy = others[0]
            for attempt in range(2):
                held, gxy = pick_container(rb, c, cxy)
                if held:
                    break
                log("grasp seemed to miss; retry")
            used.append(gxy)
        log("all picks done; holding")
        while True:
            rb.hold(-1, 20)
    except Ended:
        log("ended; final step", rb.obs.step_id, "task_complete", rb.obs.task_complete)
    except Exception:
        log("FATAL", traceback.format_exc())
        try:
            while not rb.obs.ended:
                rb.hold(-1, 20)
        except Exception:
            pass
    log("main returns")
