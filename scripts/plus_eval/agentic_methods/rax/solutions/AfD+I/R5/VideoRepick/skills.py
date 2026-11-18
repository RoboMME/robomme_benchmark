import numpy as np
from geom import depth_to_world, world_to_pixel

def seg_components(rgb, P, zmin=0.008, zmax=0.25, sat_min=35, min_px=15):
    from geom import label
    rgbi = rgb.astype(np.int32)
    sat = rgbi.max(-1) - rgbi.min(-1)
    z = P[..., 2]
    m = np.isfinite(z) & (z > zmin) & (z < zmax) & (sat > sat_min)
    lab, n = label(m)
    out = []
    for i in range(1, n + 1):
        yy, xx = np.nonzero(lab == i)
        if len(xx) < min_px:
            continue
        W = P[yy, xx]
        out.append(dict(n=len(xx), u=xx.mean(), v=yy.mean(), W=W,
                        center=np.array([(W[:, 0].min() + W[:, 0].max()) / 2, (W[:, 1].min() + W[:, 1].max()) / 2]),
                        mean=W.mean(0), zmax=float(np.percentile(W[:, 2], 95))))
    return out

def localize_near(robot, xy, cam="wrist", max_dist=0.06, log=print):
    """Re-localize the object nearest xy using the given camera's depth. Returns (center_xy, zmax, comp) or None."""
    meta, content = robot.observe()
    ims = robot.images(meta, content)
    K = meta["camera_intrinsics"][cam]["matrix"]; pose = meta["camera_poses"][cam]
    P = depth_to_world(ims[(cam, "depth")], K, pose)
    comps = seg_components(ims[(cam, "rgb")], P)
    best = None
    for c in comps:
        # distance from xy to nearest object point
        d = np.min(np.linalg.norm(c["W"][:, :2] - np.asarray(xy)[None], axis=1))
        if best is None or d < best[0]:
            best = (d, c)
    if best is None or best[0] > max_dist:
        log(f"localize_near: none near {np.round(xy,3)} (best {None if best is None else round(best[0],3)})")
        return None
    c = best[1]
    # if the component is large (merged neighbours), restrict to points within 3cm of xy-anchor
    W = c["W"]
    d = np.linalg.norm(W[:, :2] - np.asarray(xy)[None], axis=1)
    if c["n"] > 0 and (W[:, 0].ptp() > 0.06 or W[:, 1].ptp() > 0.06):
        W = W[d < 0.03]
    center = np.array([(W[:, 0].min() + W[:, 0].max()) / 2, (W[:, 1].min() + W[:, 1].max()) / 2])
    ext = np.array([W[:, 0].ptp(), W[:, 1].ptp()])
    zmax = float(np.percentile(W[:, 2], 95))
    log(f"localize_near {np.round(xy,3)} -> center {center.round(3)} ext {ext.round(3)} zmax {zmax:.3f} n {len(W)} dist {best[0]:.3f}")
    return center, zmax, ext

def grasp_z(zmax):
    return float(np.clip(zmax * 0.5, 0.012, 0.03))

def pick(robot, xy, log=print, hover=0.10):
    """Pick the object near xy. Returns True if something held."""
    robot.move_to([xy[0], xy[1], 0.25], grip=-1, tol=0.006)
    loc = localize_near(robot, xy, log=log)
    if loc is None:
        return False, xy
    c, zmax, ext = loc
    robot.move_to([c[0], c[1], 0.18], grip=-1, tol=0.005)
    loc2 = localize_near(robot, c, log=log, max_dist=0.04)
    if loc2 is not None:
        c, zmax, ext = loc2
    robot.move_to([c[0], c[1], hover], grip=-1, tol=0.004)
    gz = grasp_z(zmax)
    robot.move_to([c[0], c[1], gz], grip=-1, tol=0.003, max_iter=8)
    robot.set_grip(1, 12)
    robot.last_gz = gz
    robot.move_to([c[0], c[1], 0.15], grip=1, tol=0.006)
    gap = robot.state[6] + robot.state[7]
    held = gap > 0.004
    log(f"pick at {c.round(3)} gz {gz:.3f} gap {gap:.4f} held {held}")
    return held, c

def capture_finger_mask(robot, meta=None, content=None):
    """Pixels occupied by the open fingers (seen at home pose where the table is far)."""
    try:
        if meta is None:
            meta, content = robot.observe()
        d = robot.images(meta, content)[("wrist", "depth")]
        m = (d > 0) & (d < 0.2)
        try:
            from scipy import ndimage
            m = ndimage.binary_dilation(m, iterations=3)
        except Exception:
            pass
        if m.sum() < 20000:
            robot.finger_mask = m
    except Exception:
        pass

def _near_obj_depth(robot):
    """10th-percentile wrist depth of non-finger pixels above the table (held / just released object)."""
    meta, content = robot.observe()
    ims = robot.images(meta, content)
    d = ims[("wrist", "depth")]
    z = float(robot.state[2]); tab = z + 0.067
    m = (d > 0) & (d < tab - 0.012)
    fm = getattr(robot, "finger_mask", None)
    if fm is not None and fm.shape == m.shape:
        m &= ~fm
    else:
        m &= d > 0.04
    reg = np.zeros(d.shape, bool); reg[60:, 20:236] = True
    m &= reg
    if m.sum() < 40:
        return None, int(m.sum())
    return float(np.percentile(d[m], 10)), int(m.sum())

def place(robot, xy, z=None, log=print):
    if z is None:
        z = float(getattr(robot, "last_gz", 0.02)) + 0.003
    robot.move_to([xy[0], xy[1], 0.10], grip=1, tol=0.005)
    robot.move_to([xy[0], xy[1], z], grip=1, tol=0.003, max_iter=8)
    robot.set_grip(-1, 12)
    offsets = [(0.0, 0.0), (0.03, 0.0), (-0.03, 0.0), (0.0, 0.02), (0.0, -0.02)]
    released = False
    base = np.array(robot.state[:2])
    for i, (ox, oy) in enumerate(offsets):
        try:
            if i > 0:
                log(f"release recovery {i}: lower and slide by ({ox},{oy})")
                robot.move_to([base[0], base[1], z], grip=-1, tol=0.004, max_iter=6)
                robot.move_to([base[0] + ox, base[1] + oy, z], grip=-1, tol=0.004, max_iter=6)
                robot.hold(4, -1)
            d0, n0 = _near_obj_depth(robot)
            if d0 is None:
                released = True; log(f"release check: nothing near fingers (n={n0})"); break
            cur = robot.state[:3].copy()
            robot.move_to([cur[0], cur[1], cur[2] + 0.012], grip=-1, tol=0.003, max_iter=4)
            d1, n1 = _near_obj_depth(robot)
            dz = robot.state[2] - cur[2]
            if d1 is None or d1 - d0 > 0.5 * max(dz, 0.008):
                released = True
                log(f"release check ok: depth {d0:.3f}->{d1 if d1 is None else round(d1,3)} lift {dz:.3f}")
                break
            log(f"release check: object follows gripper (depth {d0:.3f}->{d1:.3f}, lift {dz:.3f})")
        except Exception as e:
            if type(e).__name__ == "EpisodeEnded":
                raise
            log("release check error", repr(e)); released = True; break
    robot.move_to([robot.state[0], robot.state[1], 0.15], grip=-1, tol=0.006)
    log(f"placed at {np.round(xy,3)} released={released}")
    return released

def find_button(robot, cam="front", near=None, log=print):
    from geom import label
    meta, content = robot.observe()
    ims = robot.images(meta, content)
    K = meta["camera_intrinsics"][cam]["matrix"]; pose = meta["camera_poses"][cam]
    P = depth_to_world(ims[(cam, "depth")], K, pose)
    rgb = ims[(cam, "rgb")].astype(np.int32)
    sat = rgb.max(-1) - rgb.min(-1); val = rgb.max(-1)
    m = (P[..., 2] > 0.004) & (P[..., 2] < 0.06) & (sat < 30) & (val > 120)
    lab, n = label(m)
    cands = []
    for i in range(1, n + 1):
        yy, xx = np.nonzero(lab == i)
        if len(xx) < 10:
            continue
        W = P[yy, xx]
        ext = np.ptp(W[:, :2], axis=0)
        if ext.max() > 0.12:
            continue  # robot base / large structure
        if W[:, 0].mean() < -0.45:
            continue  # robot base region
        if ext.max() < 0.015 or ext.min() < 0.008:
            continue  # thin edge / fragment
        # require the robot-base exclusion: button lies on the table in front of the base
        c = np.array([(W[:, 0].min() + W[:, 0].max()) / 2, (W[:, 1].min() + W[:, 1].max()) / 2])
        cands.append(dict(c=c, n=len(xx), ext=ext, zmax=float(np.percentile(W[:, 2], 95))))
    log(f"button cands ({cam}): " + "; ".join(f"{d['c'].round(3)} n{d['n']} ext{d['ext'].round(3)} z{d['zmax']:.3f}" for d in cands))
    if not cands:
        return None
    if near is not None:
        cands.sort(key=lambda d: np.linalg.norm(d["c"] - near))
    else:
        cands.sort(key=lambda d: -d["n"])
    return cands[0]

def press_button(robot, log=print):
    b = getattr(robot, "button_hint", None) or find_button(robot, "front", log=log)
    if b is None:
        log("no button in front view; scanning with wrist")
        robot.move_to([-0.15, 0.0, 0.40], grip=1, tol=0.01)
        b = find_button(robot, "wrist", log=log)
    if b is None:
        log("no button found"); return False
    c = b["c"]
    robot.move_to([c[0], c[1], 0.20], grip=1, tol=0.006)
    b2 = find_button(robot, "wrist", near=c, log=log)
    if b2 is not None and np.linalg.norm(b2["c"] - c) < 0.04:
        c = b2["c"]
    top = b["zmax"]
    robot.move_to([c[0], c[1], top + 0.05], grip=1, tol=0.004)
    for z in (top - 0.005, top - 0.012, top - 0.02):
        robot.move_to([c[0], c[1], max(z, 0.005)], grip=1, tol=0.003, max_iter=5)
        if robot.complete:
            break
    robot.move_to([c[0], c[1], top + 0.06], grip=1, tol=0.006)
    log(f"pressed button at {c.round(3)} top {top:.3f} complete {robot.complete}")
    return True
