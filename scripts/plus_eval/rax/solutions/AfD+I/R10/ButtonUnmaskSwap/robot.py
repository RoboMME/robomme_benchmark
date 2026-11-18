import json, time, numpy as np
from rlib import parse
from tracker import Tracker, init_tracker, detect_cubes, top_mask

class Robot:
    def __init__(self, ctx, log=print):
        self.ctx = ctx; self.log = log
        self.meta = None; self.imgs = None
        self.tracker = None; self.ended = False; self.complete = False
        self.frames = []  # (step, depth front) for offline replay
        self.step_id = 0
        self.use_wrist = True
        self.observe()

    def observe(self):
        r = self.ctx.call("robomme_observe")
        self._ingest(r)

    def _ingest(self, r):
        meta, imgs = parse(r)
        if "camera_poses" in meta:
            self.calib = {k: meta[k] for k in ("camera_intrinsics", "camera_poses")}
        else:
            meta.update(self.calib)
        self.meta, self.imgs = meta, imgs
        self.state = np.array(meta["state"])
        self.step_id = meta.get("step_id", self.step_id)
        tr = meta.get("transition") or {}
        if tr.get("task_complete") or any(s.get("task_complete") for s in tr.get("steps", [])):
            self.complete = True
        if meta.get("episode_ended") or tr.get("episode_ended"):
            self.ended = True
        lim = meta.get("episode_limits")
        if lim: self.remaining = lim.get("remaining_episode_steps")

    @property
    def ee(self):
        return self.state[:3]

    def front(self):
        return self.imgs[("front", "rgb")], self.imgs[("front", "depth")]

    def step(self, row):
        if self.ended:
            return False
        row = [float(np.clip(x, -1, 1)) for x in row]
        r = self.ctx.call("robomme_step", actions=[row])
        self._ingest(r)
        if self.tracker is not None and ("front", "depth") in self.imgs:
            self.tracker.update(self.meta, self.imgs[("front", "depth")], ee=self.ee,
                                wdepth=self.imgs.get(("wrist", "depth")) if self.use_wrist else None)
        return not self.ended

    def move_to(self, target, grip=-1.0, tol=0.004, max_steps=60, gain=1.0, vmax=0.8, rot=None):
        target = np.array(target, float)
        for i in range(max_steps):
            err = target - self.ee
            if np.linalg.norm(err) < tol:
                return True
            a = np.clip(err / 0.05 * gain, -vmax, vmax)
            row = list(a) + [0, 0, 0] + [grip]
            if rot is not None:
                row[3:6] = list(rot(self))
            if not self.step(row):
                return False
        return False

    def hold(self, n, grip=-1.0):
        for i in range(n):
            if not self.step([0, 0, 0, 0, 0, 0, grip]):
                return False
        return True

def rotvec_to_R(rv):
    rv = np.asarray(rv, float); th = np.linalg.norm(rv)
    if th < 1e-9: return np.eye(3)
    k = rv / th; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K

def ee_yaw(state):
    R = rotvec_to_R(state[3:6]); return np.arctan2(R[1, 0], R[0, 0])

def wrap(a, period=2*np.pi):
    return (a + period / 2) % period - period / 2

def container_yaw(meta, depth, xy):
    from rlib import backproject
    P = backproject(meta, depth)
    m = (P[..., 2] > 0.066) & (P[..., 2] < 0.09) & (np.linalg.norm(P[..., :2] - xy, axis=-1) < 0.04)
    q = P[m][:, :2]
    if len(q) < 30: return None, xy, len(q)
    c = q.mean(0); best = None
    for a in np.linspace(0, np.pi / 2, 91)[:-1]:
        Rm = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]]); r = (q - c) @ Rm.T
        ext = r.max(0) - r.min(0); area = ext[0] * ext[1]
        if best is None or area < best[0]:
            best = (area, a, (r.max(0) + r.min(0)) / 2 @ Rm)
    return best[1], c + best[2], len(q)

def yaw_controller(target_psi):
    def f(R):
        e = wrap(target_psi - ee_yaw(R.state), np.pi / 2 * 4)
        return [0, 0, float(np.clip(e / 0.5 * 0.8, -0.6, 0.6))]
    return f

def choose_psi(cur_psi, cont_yaw):
    # finger axis must align with a face normal: psi = cont_yaw mod 90deg, nearest to current
    base = cont_yaw
    cands = [base + k * np.pi / 2 for k in range(-4, 5)]
    return min(cands, key=lambda c: abs(wrap(c - cur_psi)))

def place_back(R, xy, z_release=0.066):
    R.move_to([xy[0], xy[1], z_release], grip=1, tol=0.004, vmax=0.4, max_steps=40)
    R.hold(6, grip=-1)
    R.move_to([xy[0], xy[1], 0.16], grip=-1, tol=0.01, vmax=0.6, max_steps=30)

def pick(R, xy, log=print, lift_z=0.17, grasp_z=0.062):
    """Grasp the container whose top is near xy and lift it. Returns (ok, grasp_xy)."""
    R.move_to([xy[0], xy[1], 0.16], grip=-1, tol=0.01, vmax=0.8, max_steps=60)
    rgb, dep = R.front()
    cy, c, n = container_yaw(R.meta, dep, xy)
    if cy is None:
        log(f"pick: top not visible near {xy}, n={n}"); cy, c = 0.0, np.array(xy)
    psi = choose_psi(ee_yaw(R.state), cy)
    log(f"pick: cont yaw {np.degrees(cy):.1f} c {c.round(3)} n {n} psi {np.degrees(psi):.1f}")
    rot = yaw_controller(psi)
    R.move_to([c[0], c[1], 0.12], grip=-1, tol=0.004, vmax=0.5, rot=rot, max_steps=50)
    for _ in range(15):
        if abs(wrap(psi - ee_yaw(R.state), 2*np.pi)) < np.radians(4): break
        R.step([0, 0, 0] + rot(R) + [-1])
    R.move_to([c[0], c[1], grasp_z], grip=-1, tol=0.003, vmax=0.4, rot=rot, max_steps=40)
    R.hold(8, grip=1)
    fs = R.state[6] + R.state[7]
    R.move_to([c[0], c[1], lift_z], grip=1, tol=0.01, vmax=0.5, max_steps=40)
    fs2 = R.state[6] + R.state[7]
    log(f"pick: finger sum after close {fs:.4f} after lift {fs2:.4f} step {R.step_id} complete {R.complete}")
    return (0.025 < fs2 < 0.05), c
