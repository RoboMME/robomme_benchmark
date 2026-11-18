"""Live tracking session, pick and place of containers (front-camera RGB-D, per-step tracking)."""
import numpy as np
from robot import log, current_yaw, down_R, R_to_rotvec, rotvec_to_R, POS_SCALE, ROT_SCALE, wrap_half_pi
from track import white_mask, fused_white_points
from perceive import rect_fit

# ---- tunables (tested in ep1, 5/5 grasps after fixes) ----
HOVER_Z = 0.16          # approach height above table (container top 0.074)
GRASP_Z = 0.035         # TCP height for grasp
LIFT_Z = 0.22           # lift height (pick counted; ep1 0.213)
PLACE_Z = 0.042         # TCP height to release a carried container
OCCL_FRAC = 0.6         # visible point count below this fraction of reference -> occluded
STILL_MM = 0.6          # per-step motion below this (mm) counts as still
STILL_STEPS = 6         # consecutive still steps -> containers have stopped
HELD_OPEN = (0.03, 0.075)  # finger opening range meaning a container is held (ep1: 0.0555)
TOP_BIAS = np.array([-0.008, 0.0])   # all-points centroid is biased toward the camera (+x)


LOW_TCP_Z = 0.12        # below this the fingers are in the container height band
FINGER_EXCL_R = 0.045   # exclude points around the TCP when low (fingers / target under the hand)

def obs_views(obs):
    v = {}
    for cam in ('front', 'wrist'):
        rgb = obs.rgb(cam); P = obs.points(cam)
        if rgb is not None and P is not None:
            pos = np.array((obs.poses.get(cam) or {}).get('position_m', [0, 0, 10]), float)
            v[cam] = (rgb, P, pos)
    return v

def obs_points(obs, exclude_low_tcp=True):
    st = obs.state
    if exclude_low_tcp and st is not None and st[2] < LOW_TCP_Z:
        return fused_white_points(obs_views(obs), exclude_xy=st[:2], exclude_r=FINGER_EXCL_R)
    return fused_white_points(obs_views(obs))


def target_info(pts, C, idx, radius=0.04):
    xy = pts[:, :2]
    out = {'n': 0, 'n_top': 0, 'centre': C[idx] + TOP_BIAS, 'yaw': None}
    if len(xy) == 0:
        return out
    d = np.sqrt(((xy[:, None] - C[None]) ** 2).sum(-1))
    a = d.argmin(1); dm = d[np.arange(len(xy)), a]
    sel = (a == idx) & (dm < radius)
    top = sel & (pts[:, 2] > 0.062)
    out.update(n=int(sel.sum()), n_top=int(top.sum()))
    if top.sum() >= 12:
        c, yaw, wh = rect_fit(xy[top])
        out.update(centre=c, yaw=float(yaw), wh=wh)
    return out


def choose_grip_yaw(box_yaw, centre, others):
    """Two finger axes are possible (box_yaw, box_yaw+90). Fingers open along gripper local y,
    i.e. along world direction yaw+90deg. Prefer the axis whose finger corridor is clearest."""
    best = None
    for cand in [box_yaw, box_yaw + np.pi / 2]:
        g = wrap_half_pi(cand)
        ax = np.array([np.cos(g + np.pi / 2), np.sin(g + np.pi / 2)])
        clear = 1.0
        for o in others:
            v = o - centre
            along = abs(v @ ax); perp = abs(v @ np.array([-ax[1], ax[0]]))
            if perp < 0.045:
                clear = min(clear, along)
        if best is None or clear > best[0] + 1e-3:
            best = (clear, g)
    return best[1], best[0]


def yaw_err(target_yaw, state):
    return wrap_half_pi(target_yaw - current_yaw(state))


def row_toward(state, target, yaw, g, vmax=1.0, rmax=1.0):
    err = np.asarray(target, float) - state[:3]
    dp = np.clip(err / POS_SCALE, -vmax, vmax)
    Rc = rotvec_to_R(state[3:6])
    cy = current_yaw(state)
    ty = cy + wrap_half_pi(yaw - cy) if yaw is not None else cy
    rerr = R_to_rotvec(down_R(ty) @ Rc.T)
    dr = np.clip(rerr / ROT_SCALE, -rmax, rmax)
    return list(dp) + list(dr) + [g], err, rerr


class Session:
    """Live state: robot + tracker + time base + motion monitor."""
    def __init__(self, rb, tr, t_exec0):
        self.rb, self.tr, self.t0 = rb, tr, t_exec0
        self.held = None
        self.static = False
        self.still_run = 0
        self.ref_counts = None
        self.static_step = None
        self.obstacles = None

    def t(self):
        return self.t0 + float(self.rb.obs.step_id or 0)

    def track(self):
        pts = obs_points(self.rb.obs)
        excl = self.rb.state[:2] if self.held is not None else None
        prevC = self.tr.C.copy()
        t_prev = self.tr.t
        self.tr.update(pts[:, :2], self.t(), exclude_xy=excl, exclude_r=0.05 if excl is not None else 0.0)
        cnt = self.tr.last_counts
        if self.ref_counts is None:
            self.ref_counts = np.maximum(cnt.astype(float), 1.0)
        occl = cnt < OCCL_FRAC * self.ref_counts
        if self.static:
            self.tr.C[occl] = prevC[occl]
            self.tr.V[:] = 0
        else:
            self.tr.C[occl] = prevC[occl] + self.tr.V[occl] * max(0.0, self.tr.t - t_prev)
        vis = ~occl & ~self.tr.frozen
        self.ref_counts[vis] = 0.9 * self.ref_counts[vis] + 0.1 * cnt[vis]
        # motion monitor on visible free containers
        dt = max(1e-3, self.tr.t - t_prev)
        if vis.any() and not self.static:
            sp = np.sqrt(((self.tr.C[vis] - prevC[vis]) ** 2).sum(-1)).max() / dt * 1000
            self.still_run = self.still_run + 1 if sp < STILL_MM else 0
            if self.still_run >= STILL_STEPS and vis.sum() >= max(2, 0.5 * (~self.tr.frozen).sum()):
                self.static = True
                self.static_step = self.rb.obs.step_id
                self.tr.V[:] = 0
                log(f'motion: containers static from step {self.static_step}')
        return pts

    def step1(self, row):
        self.rb.step([row])
        return self.track()


def rise(S, z, g, n=15):
    pts = None
    for _ in range(n):
        s = S.rb.state
        row, err, _ = row_toward(s, [s[0], s[1], z], None, g)
        if abs(err[2]) < 0.015:
            break
        pts = S.step1(row)
    return pts


def pick(S, idx, max_steps=160, lead=2.0, max_attempts=3):
    """Approach, descend, close and lift container idx. Returns dict(ok, why, steps)."""
    rb, tr = S.rb, S.tr
    used = 0
    attempts = 0
    pts = obs_points(rb.obs)
    while attempts < max_attempts and used < max_steps:
        attempts += 1
        phase = 'approach'
        grip_yaw = None
        zhist = []
        close_n = 0
        close_xy = None
        lock_c = None
        stalled = False
        while used < max_steps:
            s = rb.state
            info = target_info(pts, tr.C, idx)
            c = info['centre'] + (0 if S.static else tr.V[idx] * lead)
            # below LOW_TCP_Z the target's own points are excluded (fingers): keep the last centre seen from above
            if phase != 'approach' and s[2] < LOW_TCP_Z and lock_c is not None:
                c = lock_c
            elif info['yaw'] is not None:
                lock_c = c.copy()
            others = [tr.C[j] for j in range(len(tr.C)) if j != idx]
            if info['yaw'] is not None and phase == 'approach':
                grip_yaw, clear = choose_grip_yaw(info['yaw'], info['centre'], others)
            exy = float(np.linalg.norm(c - s[:2]))
            if phase == 'approach':
                row, err, rerr = row_toward(s, [c[0], c[1], HOVER_Z], grip_yaw, -1)
                if exy < 0.012 and abs(s[2] - HOVER_Z) < 0.03 and np.linalg.norm(rerr) < 0.08 and grip_yaw is not None:
                    phase = 'descend'; zhist = []
                    log(f'pick[{idx}] descend st={rb.obs.step_id} gy={np.rad2deg(grip_yaw):.1f}')
            if phase == 'descend':
                yerr = abs(yaw_err(grip_yaw, s))
                zt = GRASP_Z if (exy < 0.015 and yerr < 0.06) else max(s[2], 0.11)
                row, err, rerr = row_toward(s, [c[0], c[1], zt], grip_yaw, -1, vmax=0.8)
                zhist.append(s[2])
                if s[2] < GRASP_Z + 0.008 and exy < 0.014:
                    phase = 'close'
                    log(f'pick[{idx}] close st={rb.obs.step_id} exy={exy:.3f}')
                elif len(zhist) > 6 and s[2] < 0.09 and zhist[-6] - s[2] < 0.004:
                    log(f'pick[{idx}] descent stall z={s[2]:.3f} attempt {attempts}')
                    stalled = True
                    break
            if phase == 'close':
                if close_xy is None:
                    close_xy = s[:2].copy()
                row, err, rerr = row_toward(s, [close_xy[0], close_xy[1], GRASP_Z], None, 1, vmax=0.5)
                close_n += 1
                if close_n > 8:
                    break
            pts = S.step1(row)
            used += 1
        if stalled:
            p = rise(S, HOVER_Z, -1); used += 12
            pts = p if p is not None else obs_points(rb.obs)
            continue
        if phase != 'close':
            break
        s = rb.state
        opening = float(s[6] + s[7])
        log(f'pick[{idx}] closed st={rb.obs.step_id} opening={opening:.4f}')
        if not (HELD_OPEN[0] < opening < HELD_OPEN[1]):
            # missed: open, rise, re-approach
            for _ in range(4):
                s = rb.state
                row, _, _ = row_toward(s, s[:3], None, -1)
                pts = S.step1(row); used += 1
            p = rise(S, HOVER_Z, -1); used += 12
            pts = p if p is not None else obs_points(rb.obs)
            continue
        tr.frozen[idx] = True
        S.held = idx
        for _ in range(30):
            s = rb.state
            row, err, _ = row_toward(s, [s[0], s[1], LIFT_Z], None, 1)
            if abs(err[2]) < 0.01:
                break
            S.step1(row); used += 1
        s = rb.state
        opening = float(s[6] + s[7])
        ok = HELD_OPEN[0] < opening < HELD_OPEN[1]
        log(f'pick[{idx}] lifted z={s[2]:.3f} opening={opening:.4f} ok={ok}')
        if not ok:
            S.held = None
            tr.frozen[idx] = False
            tr.C[idx] = s[:2].copy()
            p = rise(S, HOVER_Z, -1)
            pts = p if p is not None else obs_points(rb.obs)
            continue
        return {'ok': True, 'why': 'lifted', 'steps': used, 'attempts': attempts}
    return {'ok': False, 'why': 'attempts/steps exhausted', 'steps': used, 'attempts': attempts}


def free_spot(S):
    tr = S.tr
    best = None
    others = [tr.C[j] for j in range(len(tr.C)) if j != S.held]
    obst = S.obstacles if getattr(S, 'obstacles', None) is not None else np.zeros((0, 2))
    for x in (-0.22, -0.15, -0.08, -0.01, 0.06):
        for y in (-0.30, -0.26, 0.26, 0.30):
            p = np.array([x, y])
            d = min(np.linalg.norm(p - o) for o in others) if others else 1.0
            if len(obst):
                d = min(d, float(np.sqrt(((obst - p) ** 2).sum(-1)).min()) - 0.05)
            if best is None or d > best[0]:
                best = (d, p)
    return best[1], best[0]


def place(S):
    """Put the held container down at a free spot away from the others and release."""
    rb, tr = S.rb, S.tr
    idx = S.held
    spot, clear = free_spot(S)
    log(f'place[{idx}] spot={np.round(spot,3).tolist()} clearance={clear:.3f}')
    used = 0
    for z, tol in ((LIFT_Z, 0.015), (0.06, 0.008), (PLACE_Z, 0.006)):
        for _ in range(25):
            s = rb.state
            row, err, _ = row_toward(s, [spot[0], spot[1], z], None, 1)
            if np.linalg.norm(err) < tol:
                break
            S.step1(row); used += 1
    for _ in range(6):
        s = rb.state
        row, _, _ = row_toward(s, s[:3], None, -1)
        S.step1(row); used += 1
    s = rb.state
    tr.C[idx] = s[:2].copy(); tr.V[idx] = 0; tr.frozen[idx] = True
    S.held = None
    rise(S, HOVER_Z, -1, n=20)
    log(f'place[{idx}] done at {np.round(tr.C[idx],3).tolist()} steps={used}')
    return {'ok': True, 'steps': used}
