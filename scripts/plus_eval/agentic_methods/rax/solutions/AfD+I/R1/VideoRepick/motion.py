"""Motion primitives on top of Robot: closed-loop Cartesian moves with orientation hold."""
import numpy as np
from mcp_util import log
from geometry import axang_to_R, R_to_axang
from robot import EpisodeOver

POS_SCALE = 0.05
ROT_SCALE = 0.5


class Motion:
    def __init__(self, robot):
        self.r = robot
        self.R_target = None
        self.R_base = None
        self.rot_ok = True
        self.rot_err_hist = []

    def pos(self):
        return np.array(self.r.state[:3], float)

    def fingers(self):
        s = self.r.state
        return float(s[6] + s[7])

    def set_orientation_from_state(self):
        self.R_target = axang_to_R(self.r.state[3:6])
        self.R_base = self.R_target.copy()

    def set_yaw(self, yaw):
        """Target orientation = Rz(yaw) * R_target_base."""
        if self.R_base is None:
            return
        c, s = np.cos(yaw), np.sin(yaw)
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        self.R_target = Rz @ self.R_base

    def rot_delta(self):
        if self.R_target is None or not self.rot_ok:
            return np.zeros(3), 0.0
        Rc = axang_to_R(self.r.state[3:6])
        e = R_to_axang(self.R_target @ Rc.T)
        return e, float(np.linalg.norm(e))

    def move_to(self, target, grip, tol=0.004, max_chunks=25, max_step=0.02, label='', min_z=None):
        target = np.array(target, float)
        if min_z is not None:
            target[2] = max(target[2], min_z)
        stall = 0
        prev = None
        for it in range(max_chunks):
            p = self.pos()
            d = target - p
            dist = float(np.linalg.norm(d))
            e, en = self.rot_delta()
            if dist < tol and en < 0.04:
                return True
            k = int(np.clip(np.ceil(max(dist / max_step, en / 0.12)), 1, 6))
            dp = d / k / POS_SCALE
            dr = (e / k / ROT_SCALE) if en > 0.03 else np.zeros(3)
            dr = np.clip(dr, -0.3, 0.3)
            row = list(np.clip(dp, -1, 1)) + list(dr) + [grip]
            self.r.step([row] * k)
            if self.R_target is not None and en > 0.1:
                _, en2 = self.rot_delta()
                self.rot_err_hist.append((en, en2))
                if len(self.rot_err_hist) >= 3 and all(b > a + 0.02 for a, b in self.rot_err_hist[-3:]):
                    log('rotation correction diverging; disabled')
                    self.rot_ok = False
            q = self.pos()
            if prev is not None and np.linalg.norm(q - prev) < 0.001 and dist > tol:
                stall += 1
                if stall >= 2:
                    log('move_to %s stalled at %s (target %s)' % (label, np.round(q, 3).tolist(), np.round(target, 3).tolist()))
                    return False
            else:
                stall = 0
            prev = q
        log('move_to %s max chunks; at %s target %s' % (label, np.round(self.pos(), 3).tolist(), np.round(target, 3).tolist()))
        return float(np.linalg.norm(target - self.pos())) < 3 * tol

    def gripper(self, value, n=8, max_total=24):
        """Command gripper and wait until fingers settle."""
        last = self.fingers()
        used = 0
        while used < max_total:
            self.r.step([[0, 0, 0, 0, 0, 0, value]] * n)
            used += n
            f = self.fingers()
            if abs(f - last) < 0.0008:
                break
            last = f
            n = 4
        return self.fingers()

    def hold(self, grip, n=3):
        self.r.step([[0, 0, 0, 0, 0, 0, grip]] * n)
