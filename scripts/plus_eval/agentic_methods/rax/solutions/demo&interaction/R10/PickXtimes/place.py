"""Tracking placement onto a (possibly moving) target."""
import math
import numpy as np
import robo, skills
from robo import log

Z_PLACE = skills.GRASP_Z + 0.006

class TargetEstimator:
    def __init__(self, trk, model, static_mode=False):
        self.trk, self.model = trk, model
        self.static_mode = static_mode

    def recent(self, now, window=8):
        return [h for h in self.trk.hist if h[0] >= now - window]

    def static(self, now, window=25):
        rec = [h for h in self.trk.hist if h[0] >= now - window]
        if len(rec) < 3:
            return None
        a = np.array(rec)
        return (np.ptp(a[:, 1]) + np.ptp(a[:, 2])) < 0.004

    def predict(self, now, ahead):
        """Return (xy, quality) for time now+ahead."""
        H = self.trk.hist
        if not H:
            return None, "none"
        if self.static_mode:
            a = np.array(H[-40:])
            return np.median(a[:, 1:], 0), "static_mode"
        st = self.static(now)
        rec = self.recent(now)
        if st:
            a = np.array([h for h in H if h[0] >= now - 25])
            return np.median(a[:, 1:], 0), "static"
        if self.model is not None and self.model.P is not None:
            meas = rec[-3:] if rec else [H[-1]]
            xy, c = self.model.predict(now + ahead, meas)
            if c > 0.01 and len(meas) > 1:
                xy, c = self.model.predict(now + ahead, meas[-1:])
            if c < 0.02:
                return xy, f"model(c={c:.3f},age={now - meas[-1][0]})"
        # no path model: local polynomial extrapolation of recent measurements
        a = np.array([h for h in H if h[0] >= H[-1][0] - 16])
        age = now - H[-1][0]
        if len(a) >= 2 and a[-1, 0] > a[0, 0]:
            tq = now + ahead
            dt = min(tq - a[-1, 0], 10)
            if len(a) >= 5 and a[-1, 0] - a[0, 0] >= 8:
                tt = a[:, 0] - a[-1, 0]
                cx = np.polyfit(tt, a[:, 1], 2); cy = np.polyfit(tt, a[:, 2], 2)
                return np.array([np.polyval(cx, dt), np.polyval(cy, dt)]), f"cv2(age={age})"
            v = (a[-1, 1:] - a[0, 1:]) / (a[-1, 0] - a[0, 0])
            return a[-1, 1:] + v * dt, f"cv(age={age})"
        return np.array(H[-1][1:]), "last"

def place_tracked(r, est, gain=2.2, lead=2.0, max_steps=140, tol_xy=0.007):
    """Carry held cube over the target, follow it, descend and release."""
    used = 0
    released = False
    z_carry = max(r.state[2], 0.10)
    rel_info = None
    descending = False
    good = 0
    while used < max_steps and not r.ended:
        now = r.step_id
        p, q = est.predict(now, lead)
        if p is None:
            log("place: no target estimate")
            return False, None
        cur = r.state[:3]
        exy = p - cur[:2]
        # error relative to where the target is now (for release decision)
        pn, _ = est.predict(now, 0)
        dnow = np.linalg.norm(pn - cur[:2])
        if dnow < 0.015:
            descending = True
        elif dnow > 0.035:
            descending = False
        z = cur[2]
        stale = q.startswith("cv") and ("age=" in q) and int(q.split("age=")[1].rstrip(")")) > 12
        if stale:
            descending = False
            zc = 0.16
        elif descending:
            zc = max(Z_PLACE, z - 0.02)
        else:
            zc = max(z, 0.07) if dnow < 0.08 else max(z, z_carry * 0.8)
        d = np.array([exy[0] * gain, exy[1] * gain, (zc - z) * 1.5]) / 0.05
        m = np.max(np.abs(d))
        if m > 1:
            d = d / m
        if z <= Z_PLACE + 0.006 and dnow < tol_xy:
            good += 1
        else:
            good = 0
        if good >= 1:
            rel_info = dict(step=now, tcp=cur.copy(), pred=pn.copy(), q=q)
            log(f"place: release at step {now} tcp={np.round(cur,3)} pred={np.round(pn,3)} err={dnow:.4f} {q}")
            for k in range(4):
                p2, _ = est.predict(r.step_id, lead)
                e2 = (p2 - r.state[:2]) * gain / 0.05
                e2 = np.clip(e2, -1, 1)
                r.step([[e2[0], e2[1], 0] + r.rot_fix(r.state) + [-1]])
            r.grip = -1
            released = True
            break
        r.step([list(d) + r.rot_fix(r.state) + [1]])
        used += 1
        if used % 10 == 0:
            log(f"place: step {now} tcp={np.round(cur,3)} pred_now={np.round(pn,3)} err={dnow:.4f} {q}")
    if not released:
        log("place: timeout, releasing in place")
        r.gripper(-1, n=4)
    # lift straight up
    r.move_to([r.state[0], r.state[1], skills.HOVER_Z], grip=-1, speed=0.8, max_steps=30)
    return released, rel_info
