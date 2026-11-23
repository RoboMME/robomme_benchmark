import math
import numpy as np

def wrap90(t, ref=0.0):
    """Choose t + k*pi/2 closest to ref."""
    k = round((ref - t) / (math.pi / 2))
    return t + k * math.pi / 2

def pick_container(rb, xy, yaw, grasp_z=0.035, hover_z=0.16, lift_z=0.20, log=print, place_back=True,
                   get_frame=None, attempts=2):
    """Hover, re-centre on the fresh top surface, descend (retry if blocked), close, lift.
    Retries the whole grasp once if the gripper closed on nothing."""
    from robot import yaw_of
    x, y = map(float, xy)
    width = 0.0; held = False
    for att in range(attempts):
        if rb.ended: break
        s = rb.state(); cur_yaw = yaw_of(s[3:6])
        yaw_t = wrap90(yaw, cur_yaw)
        rb.move_to([x, y, hover_z], yaw=yaw_t, speed=0.8, grip=-1.0, tol=0.006)
        for tries in range(2):
            if get_frame is not None and not rb.ended:
                try:
                    fr = get_frame()
                    xn, yn, n = refine_top(fr, (x, y))
                    if xn is not None:
                        log(f"refine: {np.round([x, y], 3)} -> {np.round(xn, 3)} yaw {None if yn is None else round(math.degrees(yn))} cells {n}")
                        x, y = float(xn[0]), float(xn[1])
                        if yn is not None: yaw_t = wrap90(yn, yaw_of(rb.state()[3:6]))
                        rb.move_to([x, y, hover_z], yaw=yaw_t, speed=0.5, tol=0.004)
                    else:
                        log(f"refine: no reliable top near {np.round([x, y], 3)} (cells {n})")
                except Exception as e:
                    log('refine failed', repr(e))
            rb.move_to([x, y, grasp_z + 0.04], yaw=yaw_t, speed=0.5, tol=0.004)
            rb.move_to([x, y, grasp_z], yaw=yaw_t, speed=0.3, tol=0.003)
            z = rb.state()[2]
            if z <= grasp_z + 0.012 or rb.ended: break
            log(f"descent blocked at z {z:.3f} (xy {np.round(rb.state()[:2], 3)}); rising to re-centre")
            rb.move_to([x, y, hover_z], yaw=None, speed=0.6, tol=0.006)
        rb.gripper(1.0, n=10)
        s = rb.state(); width = s[6] + s[7]
        log(f"closed at {np.round(s[:3],3)} width {width:.4f}")
        if width < 0.01 and not rb.ended:
            log("closed on nothing; reopening and retrying")
            rb.gripper(-1.0, n=6)
            rb.move_to([x, y, hover_z], yaw=None, speed=0.6, tol=0.008)
            continue
        rb.move_to([x, y, lift_z], yaw=None, speed=0.4, tol=0.008)
        s = rb.state(); log(f"lifted at {np.round(s[:3],3)} width {s[6]+s[7]:.4f} ended={rb.ended} complete={rb.complete}")
        held = (s[6] + s[7]) > 0.01
        if rb.ended: break
        if not held:
            log("lost the container during lift; retrying")
            rb.gripper(-1.0, n=6)
            continue
        if place_back:
            rb.move_to([x, y, grasp_z + 0.01], yaw=None, speed=0.4, tol=0.006)
            rb.gripper(-1.0, n=8)
            rb.move_to([x, y, hover_z], yaw=None, speed=0.6, tol=0.01)
        break
    return held, width

def refine_top(fr, xy, radius=0.028, iters=6, zlo=0.062, zhi=0.10, max_shift=0.03, min_cells=40):
    """Mean-shift the container top centre near xy using fresh top-surface cells.
    Returns (xy_new, yaw or None, n_cells) or (None, None, n) when evidence is weak."""
    from tracker import frame_points, cells, rect_fit
    p, _ = frame_points(fr, zlo, zhi)
    c = cells(p)
    if len(c) == 0: return None, None, 0
    x0 = np.asarray(xy, float); x = x0.copy()
    for _ in range(iters):
        sel = np.linalg.norm(c - x, axis=1) < radius
        if sel.sum() < min_cells: return None, None, int(sel.sum())
        xn = c[sel].mean(0)
        if np.linalg.norm(xn - x) < 1e-4: x = xn; break
        x = xn
    if np.linalg.norm(x - x0) > max_shift: return None, None, int(sel.sum())
    sel = np.linalg.norm(c - x, axis=1) < 0.034
    yaw = None
    if sel.sum() >= min_cells:
        t, a, b = rect_fit(c[sel]); yaw = t
    return x, yaw, int(sel.sum())
