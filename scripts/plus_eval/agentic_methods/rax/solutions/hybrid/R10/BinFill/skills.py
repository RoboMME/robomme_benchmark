import math, numpy as np
import vision as V

OBS_POSE = np.array([-0.06, 0.0, 0.52])
SAFE_Z = 0.14


def wrap_yaw(th):
    # choose equivalent (mod pi/2) closest to 0
    th = (th + math.pi / 4) % (math.pi / 2) - math.pi / 4
    return th


def scan(r, log=print):
    r.move_to(OBS_POSE, yaw=0.0, max_iter=15)
    r.hold(2)
    det = V.detect(r.meta, r.imgs, "wrist")
    return det


def pick(r, cube, log=print, neighbor=None):
    x, y = cube["xy"]; ztop = cube["ztop"]
    yaw = wrap_yaw(cube["yaw"])
    if neighbor is not None:
        # fingers close along gripper y axis = (-sin yaw, cos yaw); keep it perpendicular to neighbour direction
        nd = np.array(neighbor) - np.array([x, y]); nd /= (np.linalg.norm(nd) + 1e-9)
        def along(a):
            return abs(-math.sin(a) * nd[0] + math.cos(a) * nd[1])
        alt = yaw + (math.pi / 2 if yaw < 0 else -math.pi / 2)
        if along(alt) < along(yaw) - 0.2:
            log("using alternative yaw %.2f instead of %.2f (neighbour)" % (alt, yaw))
            yaw = alt
    r.move_to([x, y, SAFE_Z], yaw=yaw, grip=-1.0)
    r.move_to([x, y, SAFE_Z - 0.06], yaw=yaw, grip=-1.0, tol=0.004)
    zg = max(ztop - 0.022, 0.012)
    r.move_to([x, y, zg], yaw=yaw, grip=-1.0, tol=0.004, max_d=0.02)
    r.hold(10, grip=1.0)
    f = r.fingers()
    r.move_to([x, y, SAFE_Z], grip=1.0, max_d=0.03)
    f2 = r.fingers()
    log("pick: fingers after close %.4f after lift %.4f pos %s" % (f, f2, np.round(r.pos, 3)))
    return f2 > 0.01, f2


def place_in_bin(r, binxy, log=print):
    x, y = binxy
    r.move_to([x, y, SAFE_Z], grip=1.0)
    r.move_to([x, y, 0.11], grip=1.0, tol=0.006)
    f = r.fingers()
    held = f > 0.01
    log("over bin fingers %.4f pos %s" % (f, np.round(r.pos, 3)))
    r.hold(8, grip=-1.0)
    r.move_to([x, y, SAFE_Z], grip=-1.0)
    return held


def press(r, button, log=print):
    x, y = button["capxy"]; zc = button["zmax"]
    r.move_to([x, y, SAFE_Z], yaw=0.0, grip=1.0)
    r.move_to([x, y, zc + 0.03], grip=1.0, tol=0.004)
    log("press at", round(x, 3), round(y, 3), "cap", round(zc, 3))
    for dz in [0.0, -0.01, -0.02, -0.025]:
        r.move_to([x, y, zc + dz], grip=1.0, tol=0.003, max_d=0.01, max_iter=4)
        if r.complete:
            break
    log("pressed; pos", np.round(r.pos, 3), "complete", r.complete)
    r.move_to([x, y, zc + 0.08], grip=1.0, max_iter=4)


def match_velocity(c1, c2, dt):
    """Estimate common belt velocity (m/step) from two detections dt steps apart."""
    if dt <= 0:
        return np.zeros(2)
    ds = []
    for a in c2:
        best, bd = None, 0.05
        for b in c1:
            if b["color"] != a["color"]:
                continue
            d = math.hypot(a["xy"][0] - b["xy"][0], a["xy"][1] - b["xy"][1])
            if d < bd:
                best, bd = b, d
        if best is not None:
            ds.append(np.array(a["xy"]) - np.array(best["xy"]))
    if len(ds) < 1:
        return np.zeros(2)
    v = np.median(np.array(ds), 0) / dt
    if np.linalg.norm(v) < 0.0004:
        v = np.zeros(2)
    return v


def scan2(r, gap=6, log=print):
    d1 = scan(r, log)
    s1 = r.meta["step_id"]
    r.hold(gap)
    d2 = V.detect(r.meta, r.imgs, "wrist")
    s2 = r.meta["step_id"]
    v = match_velocity(d1["cubes"], d2["cubes"], s2 - s1)
    return d2, v, s2


def relocate(r, color, pred, maxd=0.035):
    """Detect cube of colour nearest to predicted xy in current wrist view."""
    try:
        det = V.detect(r.meta, r.imgs, "wrist")
    except Exception:
        return None
    best, bd = None, maxd
    for c in det["cubes"]:
        if c["color"] != color:
            continue
        d = math.hypot(c["xy"][0] - pred[0], c["xy"][1] - pred[1])
        if d < bd:
            best, bd = c, d
    return best


def track_pick(r, cube, v, t0, log=print, neighbor=None, lead=2.0):
    """Pick a cube moving with constant velocity v (m/step), observed at xy at step t0."""
    color = cube["color"]
    p0 = np.array(cube["xy"], float); ztop = cube["ztop"]
    v = np.asarray(v, float)
    yaw = wrap_yaw(cube["yaw"])
    if neighbor is not None:
        nd = np.array(neighbor) - p0; nd /= (np.linalg.norm(nd) + 1e-9)
        along = lambda a: abs(-math.sin(a) * nd[0] + math.cos(a) * nd[1])
        alt = yaw + (math.pi / 2 if yaw < 0 else -math.pi / 2)
        if along(alt) < along(yaw) - 0.2:
            yaw = alt
    base = V and None
    from robot import rotvec_to_R, Rz
    tgtR = Rz(yaw) @ rotvec_to_R([math.pi, 0, 0])
    pred = lambda t: p0 + v * (t - t0)
    zg = max(ztop - 0.022, 0.012)
    hover = zg + 0.075

    def go(z, grip, n_rows=3, maxd=0.04):
        t = r.meta["step_id"]
        rows = []
        pos = r.pos.copy()
        rv = r.rot_delta(tgtR)
        for k in range(n_rows):
            tgt = np.r_[pred(t + k + 1 + lead), z]
            d = tgt - pos
            nd_ = np.linalg.norm(d[:2])
            if nd_ > maxd:
                d[:2] *= maxd / nd_
            d[2] = np.clip(d[2], -maxd, maxd)
            rows.append(list(np.clip(d / 0.05, -1, 1)) + list(np.clip(rv / 0.5 / n_rows, -0.6, 0.6)) + [grip])
            pos = pos + d * 0.6  # rough expectation
        r.step(rows)

    # phase A: approach hover
    for it in range(25):
        t = r.meta["step_id"]
        err = np.r_[pred(t) - r.pos[:2], hover - r.pos[2]]
        if np.linalg.norm(err[:2]) < 0.012 and abs(err[2]) < 0.02:
            break
        go(hover, -1.0, n_rows=3, maxd=0.045)
    # visual refinement at hover
    for it in range(3):
        c = relocate(r, color, pred(r.meta["step_id"]))
        if c is None:
            break
        p0 = np.array(c["xy"]); t0 = r.meta["step_id"]
        go(hover, -1.0, n_rows=2)
    log("track: hover pos %s pred %s" % (np.round(r.pos, 3), np.round(pred(r.meta["step_id"]), 3)))
    # phase B: descend
    for it in range(8):
        go(zg, -1.0, n_rows=2, maxd=0.03)
        if r.pos[2] < zg + 0.006:
            break
    log("track: grasp pos %s pred %s" % (np.round(r.pos, 3), np.round(pred(r.meta["step_id"]), 3)))
    # close while following
    for it in range(4):
        go(zg, 1.0, n_rows=2, maxd=0.02)
    # lift
    for it in range(4):
        t = r.meta["step_id"]
        r.step([[0, 0, 0.8, 0, 0, 0, 1.0]] * 2)
    r.move_to([r.pos[0], r.pos[1], SAFE_Z], grip=1.0, max_iter=4)
    f = r.fingers()
    log("track pick: fingers %.4f pos %s" % (f, np.round(r.pos, 3)))
    return f > 0.01, f
