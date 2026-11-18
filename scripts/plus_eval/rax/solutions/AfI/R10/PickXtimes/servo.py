import math
import numpy as np
from robot import log, Ended
from scene import detect


class Tracker:
    def __init__(self, name):
        self.name = name
        self.obs = []  # (t, x, y)

    def add(self, t, xy):
        self.obs.append((float(t), float(xy[0]), float(xy[1])))
        if len(self.obs) > 400:
            self.obs = self.obs[-400:]

    def last(self):
        return self.obs[-1] if self.obs else None

    def predict(self, t, window=10):
        if not self.obs:
            return None
        tl = self.obs[-1][0]
        pts = [o for o in self.obs if o[0] >= tl - window]
        # collapse duplicate times
        d = {}
        for o in pts:
            d[o[0]] = o
        pts = sorted(d.values())
        P = np.array(pts)
        if len(P) < 3 or P[-1, 0] - P[0, 0] < 2:
            return P[-1, 1:3].copy()
        T = P[:, 0] - tl
        deg = 2 if (len(P) >= 5 and T[-1] - T[0] >= 6) else 1
        h = min(t - tl, 8.0)
        out = []
        for j in (1, 2):
            c = np.polyfit(T, P[:, j], deg)
            v = np.polyval(c, h)
            out.append(v)
        out = np.array(out)
        # clamp extrapolation speed (m/step)
        base = P[-1, 1:3]
        step = out - base
        if h > 0 and np.linalg.norm(step) / h > 0.015:
            step = step / np.linalg.norm(step) * 0.015 * h
        return base + step

    def speed(self, window=10):
        if len(self.obs) < 2:
            return 0.0
        a, b = self.obs[0], self.obs[-1]
        pts = [o for o in self.obs if o[0] >= b[0] - window]
        a = pts[0]
        if b[0] - a[0] < 1:
            return 0.0
        return math.hypot(b[1] - a[1], b[2] - a[2]) / (b[0] - a[0])


def observe_obj(rob, kind, color, tracker, gate=0.05, cams=("front", "wrist")):
    """Detect kind ('target'|'cube') in the latest images and add to tracker. Returns xy or None."""
    pred = None
    if tracker.obs and rob.step_id - tracker.obs[-1][0] <= 6:
        pred = tracker.predict(rob.step_id)
    best = None
    for cam in cams:
        try:
            d = detect(rob.meta, rob.imgs, cam, color)
        except Exception as e:
            log("detect err", e)
            continue
        if kind not in d:
            continue
        o = d[kind]
        if kind == "target" and o["n"] < 40:
            continue
        if kind == "cube" and o["n"] < 25:
            continue
        xy = np.array(o["xy"])
        if pred is not None and np.linalg.norm(xy - pred) > gate:
            continue
        if best is None:
            best = (cam, xy, o)
        elif np.linalg.norm(best[1] - xy) < 0.02:
            # average front and wrist estimates (their occlusion biases differ)
            best = (best[0] + "+" + cam, (best[1] + xy) / 2, best[2])
            break
    if best is not None:
        tracker.add(rob.step_id, best[1])
        return best
    return None


def servo(rob, kind, color, tracker, z_target, grip, max_calls=60, rows=2, zspeed=0.025,
          xy_speed=0.03, xy_gate_descend=0.012, xy_tol=0.004, cams=("front", "wrist"), finish_rows=0,
          finish_grip=None, extra_after=0, yaw=None, gain=0.5, settle_calls=15):
    """Follow tracked object's predicted xy (velocity feed-forward + proportional correction),
    descending to z_target once aligned; then run finish_rows with finish_grip while following."""
    from robot import wrap
    reached = False
    descending = False
    at_bottom = 0
    best_err = 1.0
    for it in range(max_calls):
        observe_obj(rob, kind, color, tracker, cams=cams)
        if not tracker.obs:
            log("servo: no observation of", kind)
            return False
        t = rob.step_id
        pos = rob.pos
        p0 = tracker.predict(t)
        err_xy = float(np.linalg.norm(p0 - pos[:2]))
        if abs(pos[2] - z_target) < 0.006:
            at_bottom += 1
            best_err = min(best_err, err_xy)
            if err_xy < xy_tol or at_bottom >= settle_calls:
                reached = True
                break
        if err_xy < xy_gate_descend:
            descending = True
        elif err_xy > 0.02:
            descending = False
        if descending:
            z_goal = z_target
        elif err_xy > 0.04:
            z_goal = max(pos[2], z_target + 0.04)
        else:
            z_goal = pos[2]
        cur = pos.copy()
        rws = []
        for k in range(rows):
            pa = tracker.predict(t + k)
            pb = tracker.predict(t + k + 1)
            v = pb - pa
            e = pb - cur[:2]
            dxy = v + gain * e
            n = np.linalg.norm(dxy)
            if n > xy_speed:
                dxy = dxy / n * xy_speed
            dz = float(np.clip(z_goal - cur[2], -zspeed, zspeed))
            rz = 0.0
            if yaw is not None and k == 0:
                ey = wrap(yaw - rob.finger_yaw(), math.pi / 2)
                rz = float(np.clip(ey / 0.5, -0.5, 0.5))
            rws.append([dxy[0] / 0.05, dxy[1] / 0.05, dz / 0.05, 0, 0, rz, grip])
            cur = cur + np.array([dxy[0], dxy[1], dz])
        rob.step(rws)
    if not reached:
        log("servo %s: not reached z=%.3f" % (kind, rob.pos[2]))
    log("servo %s: err at finish %.4f (best %.4f) bottom_calls=%d step=%s" % (
        kind, err_xy, best_err, at_bottom, rob.step_id))
    if finish_rows:
        done = 0
        while done < finish_rows:
            observe_obj(rob, kind, color, tracker, cams=cams)
            t = rob.step_id
            cur = rob.pos.copy()
            rws = []
            for k in range(min(2, finish_rows - done)):
                pa = tracker.predict(t + k)
                pb = tracker.predict(t + k + 1)
                dxy = (pb - pa) + gain * (pb - cur[:2])
                n = np.linalg.norm(dxy)
                if n > xy_speed:
                    dxy = dxy / n * xy_speed
                dz = float(np.clip(z_target - cur[2], -0.01, 0.01))
                rws.append([dxy[0] / 0.05, dxy[1] / 0.05, dz / 0.05, 0, 0, 0, finish_grip])
                cur = cur + np.array([dxy[0], dxy[1], dz])
            rob.step(rws)
            done += len(rws)
    return reached
