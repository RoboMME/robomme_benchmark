"""Moving-target detection and tracking placement."""
import numpy as np
from robo import all_points, hsv, components, log

PURPLE = (125, 166)
SEEN_PURPLE = [False]


def detect_target_cam(R, cam):
    if (cam, "rgb") not in R.imgs or (cam, "depth") not in R.imgs:
        return None
    rgb = R.imgs[(cam, "rgb")]
    P = all_points(R.meta, cam, R.imgs[(cam, "depth")])
    z = P[..., 2]
    h = hsv(rgb)
    purple = (h[..., 0] >= PURPLE[0]) & (h[..., 0] <= PURPLE[1]) & (h[..., 1] > 60)
    flat = (z > 0.003) & (z < 0.014) & (P[..., 0] > -0.45)
    comps = components(flat, 8)
    for c in comps:
        npur = int((purple & c["mask"]).sum())
        if npur >= 4:
            SEEN_PURPLE[0] = True
            pts = P[c["mask"]]
            return np.nanmedian(pts, 0), c["area"], npur
    if not SEEN_PURPLE[0]:
        # fallback for a non-purple target: largest flat object
        for c in comps:
            if c["area"] >= 30:
                pts = P[c["mask"]]
                return np.nanmedian(pts, 0), c["area"], 0
    return None


def detect_target(R, prefer="front"):
    order = [prefer, "wrist" if prefer == "front" else "front"]
    for cam in order:
        d = detect_target_cam(R, cam)
        if d is not None and d[1] >= (25 if cam == "front" else 40):
            return d[0], cam, d[1]
    for cam in order:
        d = detect_target_cam(R, cam)
        if d is not None:
            return d[0], cam, d[1]
    return None


class TargetTrack:
    def __init__(self):
        self.hist = []  # (step, x, y)

    def add(self, step, p):
        if self.hist and self.hist[-1][0] == step:
            self.hist[-1] = (step, p[0], p[1])
        else:
            self.hist.append((step, float(p[0]), float(p[1])))
        self.hist = self.hist[-40:]

    def predict(self, step, window=8):
        """Predicted xy at `step` and velocity (m/step)."""
        if not self.hist:
            return None, np.zeros(2)
        H = np.array(self.hist[-window:])
        H = H[H[:, 0] >= step - 30]
        if len(H) == 0:
            H = np.array(self.hist[-1:])
        if len(H) < 3 or H[-1, 0] - H[0, 0] < 2:
            return np.array(H[-1, 1:3]), np.zeros(2)
        t = H[:, 0] - H[-1, 0]
        deg = 2 if len(H) >= 6 else 1
        out = []; vel = []
        for k in (1, 2):
            c = np.polyfit(t, H[:, k], deg)
            tt = step - H[-1, 0]
            out.append(np.polyval(c, tt))
            vel.append(np.polyval(np.polyder(c), tt))
        v = np.array(vel)
        if np.linalg.norm(v) > 0.03:  # implausible
            v = v / np.linalg.norm(v) * 0.03
        return np.array(out), v

    def speed(self):
        if len(self.hist) < 4:
            return 0.0
        H = np.array(self.hist[-8:])
        dt = H[-1, 0] - H[0, 0]
        if dt <= 0:
            return 0.0
        return float(np.linalg.norm(H[-1, 1:3] - H[0, 1:3]) / dt)


def observe_target(R, track):
    d = detect_target(R)
    if d is not None:
        track.add(int(R.meta.get("step_id", 0)), d[0])
    return d


def place_tracking(R, track, grasp_z, obstacles=(), hover_z=0.10, max_steps=220,
                   CLOSE=1.0, OPEN=-1.0, resp=0.4):
    """Hold the grasped cube, follow the (possibly moving) target and release on it.
    Returns (released, release_xy, target_xy_at_release)."""
    place_z = grasp_z + 0.004
    n = 0
    released = False
    open_steps = 0
    rel_info = (None, None)
    aligned = 0
    while n < max_steps and not R.ended:
        step = int(R.meta.get("step_id", 0))
        observe_target(R, track)
        cur, v = track.predict(step)
        if cur is None:
            # nothing seen yet: rise and look
            p = R.state[:3]
            R.move_to([p[0], p[1], max(p[2], 0.2)], CLOSE, tol=0.01, max_steps=3)
            n += 3
            continue
        lead = 2.0
        aim, v_aim = track.predict(step + lead)
        tcp = R.state[:3]
        exy = aim - tcp[:2]
        ecur = cur - tcp[:2]
        near_obst = any(np.linalg.norm(np.asarray(o[:2]) - cur) < 0.055 for o in obstacles)
        if released:
            zt = place_z + 0.03 if open_steps < 6 else 0.14
            grip = OPEN
            open_steps += 1
        else:
            grip = CLOSE
            if np.linalg.norm(ecur) < 0.025 and not near_obst:
                zt = place_z
            else:
                zt = hover_z
        dxy = exy / 0.05 + v_aim / (0.05 * resp) * 0.8
        dz = (zt - tcp[2]) / 0.05
        r = np.clip(R.rot_err() / 0.5 * 0.8, -0.4, 0.4)
        if not released and abs(tcp[2] - place_z) < 0.006 and np.linalg.norm(ecur) < 0.007:
            aligned += 1
        else:
            aligned = 0 if not released else aligned
        if not released and aligned >= 1:
            released = True
            grip = OPEN
            rel_info = (tcp[:2].copy(), cur.copy())
            log("track: release at", np.round(tcp, 4), "target", np.round(cur, 4), "speed", round(track.speed(), 4))
        R.step([[dxy[0], dxy[1], dz, r[0], r[1], r[2], grip]])
        n += 1
        if released and open_steps >= 10 and R.state[2] > 0.1:
            break
    return released, rel_info


def place_v3(R, model, grasp_z, obstacles=(), hover_z=0.10, max_steps=200,
             CLOSE=1.0, OPEN=-1.0, resp=0.4, lead=2.0):
    """Follow the model-predicted target with the grasped cube, descend and release on it.
    Returns (released, tcp_xy_at_release, predicted_target_xy_at_release)."""
    place_z = grasp_z + 0.004
    n = 0
    while n < max_steps and not R.ended:
        step = int(R.meta.get("step_id", 0))
        cur, kind = model.predict(step)
        if cur is None:
            p = R.state[:3]
            R.move_to([p[0], p[1], 0.32], CLOSE, tol=0.01, max_steps=4)
            n += 4
            continue
        aim, _ = model.predict(step + lead)
        a1, _ = model.predict(step + lead + 1)
        a0, _ = model.predict(step + lead - 1)
        v = (np.asarray(a1) - np.asarray(a0)) / 2.0
        tcp = R.state[:3]
        ecur = np.asarray(cur) - tcp[:2]
        near_obst = any(np.linalg.norm(np.asarray(o[:2]) - np.asarray(cur)) < 0.055 for o in obstacles)
        if np.linalg.norm(ecur) < 0.02 and not near_obst:
            zt = place_z
        elif np.linalg.norm(ecur) < 0.04 and not near_obst:
            zt = min(hover_z, max(place_z + 0.03, tcp[2]))
        else:
            zt = hover_z
        if abs(tcp[2] - place_z) < 0.006 and np.linalg.norm(ecur) < 0.008:
            log("place: release tcp", np.round(tcp, 4), "target", np.round(cur, 4), kind,
                "v", np.round(v, 4), "step", step)
            R.step([[0, 0, 0, 0, 0, 0, OPEN]] * 6)
            return True, tcp[:2].copy(), np.asarray(cur).copy()
        exy = np.asarray(aim) - tcp[:2]
        dxy = exy / 0.05 + v / (0.05 * resp) * 0.7
        dz = (zt - tcp[2]) / 0.05
        r = np.clip(R.rot_err() / 0.5 * 0.8, -0.4, 0.4)
        R.step([[dxy[0], dxy[1], dz, r[0], r[1], r[2], CLOSE]])
        n += 1
    return False, None, None
