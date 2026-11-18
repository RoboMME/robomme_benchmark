"""Closed-loop tracking of possibly moving targets / cubes."""
import math
import numpy as np
from perception import find_objects, rect_yaw, wrap90, finger_yaw
from motion import log, GAIN, ROT_GAIN


class Track:
    def __init__(self, name):
        self.name = name
        self.obs = []  # (step, x, y)

    def add(self, step, xy):
        self.obs.append((step, float(xy[0]), float(xy[1])))
        self.obs = self.obs[-40:]

    def predict(self, step, window=18):
        """Return (position at `step`, velocity at `step`) from recent observations."""
        if not self.obs:
            return None, np.zeros(2)
        last = self.obs[-1][0]
        rec = [o for o in self.obs if o[0] >= last - window]
        a = np.array(rec)
        dt = min(step - last, 25)
        if len(a) >= 3 and a[-1, 0] - a[0, 0] >= 4:
            t = a[:, 0] - last
            px = np.polyfit(t, a[:, 1], 1); py = np.polyfit(t, a[:, 2], 1)
            v = np.array([px[0], py[0]])
            if np.linalg.norm(v) < 0.0007 and np.ptp(a[:, 1]) < 0.01 and np.ptp(a[:, 2]) < 0.01:
                return a[:, 1:].mean(0), np.zeros(2)
            if len(a) >= 5 and a[-1, 0] - a[0, 0] >= 8:
                qx = np.polyfit(t, a[:, 1], 2); qy = np.polyfit(t, a[:, 2], 2)
                acc = np.array([qx[0], qy[0]]) * 2
                if np.linalg.norm(acc) < 0.0012:
                    tt = dt
                    p = np.array([np.polyval(qx, tt), np.polyval(qy, tt)])
                    vv = np.array([2 * qx[0] * tt + qx[1], 2 * qy[0] * tt + qy[1]])
                    return p, np.clip(vv, -0.012, 0.012)
            v = np.clip(v, -0.012, 0.012)
            return np.array([px[1], py[1]]) + v * dt, v
        return np.array(self.obs[-1][1:]), np.zeros(2)

    def age(self, step):
        return step - self.obs[-1][0] if self.obs else 10 ** 6


def detect(R, color, holding):
    """Detect target and cube from the latest frame. Returns dict of xy per source."""
    out = {"target": None, "cube": None, "cube_yaw": None, "cube_top": None}
    step = R.meta.get("step_id", 0)
    tf = None; tw = None
    for cam in ("wrist", "front"):
        try:
            res, _ = find_objects(R.meta, R.imgs, cam, color)
        except Exception as e:
            continue
        t = res["target"]
        if t is not None:
            if cam == "front" and t["n"] >= 50:
                tf = t
            if cam == "wrist" and not holding and t["n"] >= 60:
                tw = t
        if res["cube"] and not holding:
            c = res["cube"][0]
            if cam == "wrist" and c["n"] >= 40 and out["cube"] is None:
                out["cube"] = c["xy"]; out["cube_top"] = c["top"]
                if len(c["top_pts"]) > 30:
                    out["cube_yaw"] = rect_yaw(c["top_pts"][:, :2])
            elif cam == "front" and out["cube"] is None and c["n"] >= 40:
                out["cube_front"] = c["xy"]; out["cube_top_front"] = c["top"]
    if tw is not None:
        out["target"] = tw["xy"]; out["target_z"] = tw["z"]
    elif tf is not None:
        out["target"] = tf["xy"]; out["target_z"] = tf["z"]
    return out


def servo_rows(R, goal_xyz, vff, n, max_v=0.019, rz=0.0):
    """Rows driving EE toward goal (moving with velocity vff) over n steps."""
    p = R.pos
    rows = []
    for k in range(n):
        g = np.array(goal_xyz, float).copy()
        g[:2] = clamp_xy(g[:2] + vff * (k + 1))
        d = g - p
        # move fraction toward goal each step plus feed-forward
        step = d * 0.6
        step[:2] += vff
        nrm = np.linalg.norm(step)
        if nrm > max_v:
            step = step * max_v / nrm
        nxt = p + step
        nxt[:2] = clamp_xy(nxt[:2]); step = nxt - p
        p = nxt
        rows.append([step[0] / GAIN, step[1] / GAIN, step[2] / GAIN, 0, 0, rz / 0.5 if k == 0 else 0.0, R.grip])
    return rows


CUBE_HALF = 0.02
RING_R = 0.0365  # measured: front-view near-edge (97th pct x) minus ring centre


def _det_target(R, color, holding):
    """Target centre from the ring extents; only when most of the ring is visible."""
    cams = ("front",) if holding else ("wrist", "front")
    for cam in cams:
        try:
            res, _ = find_objects(R.meta, R.imgs, cam, color)
        except Exception:
            continue
        t = res["target"]
        if t is None or t["n"] < 40:
            continue
        yw = t["ymax"] - t["ymin"]; xw = t["xmax"] - t["xmin"]
        if yw < 0.06:
            continue
        if xw >= 0.066:
            x = (t["xmax"] + t["xmin"]) / 2
        elif cam == "front" and xw >= 0.03:
            # far half hidden behind the gripper/cube: the near (camera-side) edge stays visible
            x = t["xmax"] - RING_R
        else:
            continue
        xy = np.array([x, (t["ymax"] + t["ymin"]) / 2])
        return xy, t["z"], cam
    return None


BASE = np.array([-0.615, 0.0])
REACH = 0.74
XMAX = 0.12


def safe(xy, margin=0.0):
    xy = np.asarray(xy, float)
    return np.linalg.norm(xy - BASE) <= REACH - margin and xy[0] <= XMAX - margin


def clamp_xy(xy):
    xy = np.array(xy, float)
    d = xy - BASE; r = np.linalg.norm(d)
    if r > REACH:
        xy = BASE + d * REACH / r
    xy[0] = min(xy[0], XMAX)
    return xy


def _det_cube(R, color):
    for cam in ("wrist", "front"):
        try:
            res, _ = find_objects(R.meta, R.imgs, cam, color)
        except Exception:
            continue
        if res["cube"]:
            c = res["cube"][0]
            if c["n"] >= 30 and c["top"] < 0.07:
                yaw = rect_yaw(c["top_pts"][:, :2]) if (cam == "wrist" and len(c["top_pts"]) > 30) else None
                return c["xy"], c["top"], yaw, cam
    return None


def servo_grasp(R, color, trk, hover=0.10, max_iter=70, trkT=None):
    from motion import rotate_yaw
    R.grip = -1.0
    rotated = False
    top = 0.041
    it = 0
    phase = "hover"
    zg = None
    while it < max_iter and not R.ended:
        it += 1
        step = R.meta.get("step_id", 0)
        if trkT is not None:
            update_target(R, color, trkT, holding=False)
        d = _det_cube(R, color) if R.pos[2] > 0.038 else None
        if d is not None:
            prev, _ = trk.predict(step)
            if prev is None or np.linalg.norm(d[0] - prev) < 0.06 or trk.age(step) > 20:
                trk.add(step, d[0]); top = 0.7 * top + 0.3 * d[1] if d[1] > 0.03 else top
        pred, v = trk.predict(step)
        if pred is None:
            if it > 10:
                log("servo_grasp: no cube"); return False
            R.hold(2); continue
        err = np.linalg.norm(pred - R.pos[:2])
        if phase == "hover":
            if err < 0.012 and not rotated:
                if d is not None and d[2] is not None:
                    dyaw = wrap90(d[2] - finger_yaw(R.state))
                    if abs(dyaw) > 0.05:
                        rotate_yaw(R, dyaw, max_iter=3)
                rotated = True
                continue
            if rotated and err < 0.008 and abs(R.pos[2] - hover) < 0.03:
                phase = "descend"; zg = max(top - CUBE_HALF - 0.003, 0.012)
                log(f"servo_grasp: descend at step {step} pos={R.pos.round(3)} pred={pred.round(3)} v={v.round(4)} top={top:.3f}")
                continue
            R.step(servo_rows(R, [pred[0], pred[1], hover], v, 2))
        else:
            z = R.pos[2]
            if z <= zg + 0.004:
                # close while following the cube
                R.grip = 1.0
                for k in range(4):
                    pred, v = trk.predict(R.meta.get("step_id", 0))
                    if np.linalg.norm(v) < 1e-4:
                        R.step([[0, 0, 0, 0, 0, 0, 1.0]] * 2)
                    else:
                        R.step(servo_rows(R, [pred[0], pred[1], R.pos[2]], v, 2))
                w = R.state[6] + R.state[7]
                log(f"servo_grasp: closed width={w:.4f} pos={R.pos.round(3)} step={R.meta.get('step_id')}")
                return w > 0.02
            gz = max(zg, z - 0.022)
            R.step(servo_rows(R, [pred[0], pred[1], gz], v, 2))
    log("servo_grasp: iteration limit")
    return False


def update_target(R, color, trk, holding):
    step = R.meta.get("step_id", 0)
    d = _det_target(R, color, holding)
    if d is None:
        return False
    prev, _ = trk.predict(step)
    if prev is None or np.linalg.norm(d[0] - prev) < 0.06 or trk.age(step) > 20:
        trk.add(step, d[0])
        return True
    return False


def servo_place(R, color, trk, tz=0.009, hover=0.075, max_iter=90):
    phase = "hover"
    tol = 0.008
    it = 0
    zp = tz + CUBE_HALF + 0.007
    while it < max_iter and not R.ended:
        it += 1
        step = R.meta.get("step_id", 0)
        update_target(R, color, trk, holding=True)
        pred, v = trk.predict(step)
        if pred is None:
            if it > 15:
                log("servo_place: no target"); return False
            R.hold(2); continue
        err = np.linalg.norm(pred - R.pos[:2])
        fut, _ = trk.predict(step + 12)
        ok_ws = safe(pred, 0.01) and safe(fut, 0.01)
        if phase == "hover":
            tol = 0.008 if it < 25 else (0.012 if it < 45 else 0.018)
            if err < tol and abs(R.pos[2] - hover) < 0.02 and ok_ws:
                phase = "descend"
                log(f"servo_place: descend step {step} pos={R.pos.round(3)} pred={pred.round(3)} v={v.round(4)} age={trk.age(step)}")
                continue
            R.step(servo_rows(R, [pred[0], pred[1], hover], v, 2))
        else:
            z = R.pos[2]
            if z <= zp + 0.004:
                if err > max(0.012, tol):
                    R.step(servo_rows(R, [pred[0], pred[1], zp], v, 1)); continue
                # release while following the target, then keep following with open fingers
                R.grip = -1.0
                errs = []
                for k in range(7):
                    step = R.meta.get("step_id", 0)
                    if k > 0:
                        update_target(R, color, trk, holding=True)
                    pred, v = trk.predict(step)
                    errs.append(round(float(np.linalg.norm(pred - R.pos[:2])), 4))
                    zz = zp if k < 5 else zp + 0.03
                    R.step(servo_rows(R, [pred[0], pred[1], zz], v, 2))
                log(f"servo_place: released pos={R.pos.round(3)} pred={pred.round(3)} v={v.round(4)} step={R.meta.get('step_id')} age={trk.age(step)} errs={errs}")
                return True
            gz = max(zp, z - 0.03) if err < 0.015 else z
            R.step(servo_rows(R, [pred[0], pred[1], gz], v, 2))
    log("servo_place: iteration limit")
    return False
