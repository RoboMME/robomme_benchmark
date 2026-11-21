"""Task-level skills for SwingXtimes."""
import numpy as np
from scipy.spatial.transform import Rotation as R
from robo import log
from percep import cube_estimate, color_mask, components

DOWN = R.from_rotvec([-np.pi, 0, 0])


def wrap90(a):
    return (a + np.pi / 4) % (np.pi / 2) - np.pi / 4


CUBE_COLOR = ["red"]


def locate_cube(rb, cams=("wrist", "front")):
    o = rb.observe()
    for cam in cams:
        est = cube_estimate(o, cam, CUBE_COLOR[0])
        if est is not None and est[3] >= 3:
            return est, cam, o
    return None, None, o


def grasp_cube(rb, approach_z=0.13, grasp_dz=0.02):
    rb.grip = -1.0
    rb.down_yaw = DOWN
    est, cam, o = locate_cube(rb, ("front", "wrist"))
    if est is None:
        log("cube not found")
        return False
    c, ztop, yaw, n = est
    log("cube coarse", cam, c, ztop, yaw)
    rb.move_to([c[0], c[1], approach_z], tol=0.006, max_steps=80, chunk=4)
    # refine with wrist
    est, cam, o = locate_cube(rb, ("wrist", "front"))
    if est is not None:
        c, ztop, yaw, n = est
        log("cube refine", cam, c, ztop, yaw, n)
    dyaw = wrap90(yaw) if yaw is not None else 0.0
    target_rot = R.from_rotvec([0, 0, dyaw]) * DOWN
    rb.rotate_to(target_rot, max_steps=30)
    rb.move_to([c[0], c[1], approach_z], tol=0.004, max_steps=30, chunk=3)
    est2, cam2, o = locate_cube(rb, ("wrist",))
    if est2 is not None:
        c, ztop = est2[0], est2[1]
        log("cube refine2", c, ztop)
    gz = max(ztop - grasp_dz, 0.012)
    rb.move_to([c[0], c[1], gz + 0.04], tol=0.005, max_steps=40, chunk=3)
    rb.move_to([c[0], c[1], gz], tol=0.003, max_steps=40, chunk=2, vmax=0.5)
    rb.hold(10, grip=1.0)
    rb.move_to([c[0], c[1], gz + 0.08], tol=0.01, max_steps=40, chunk=3)
    held = rb.state[6] + rb.state[7]
    log("after lift, finger sum", held, "state", rb.state)
    return held > 0.01


def visit(rb, xy, z_low, z_trav=0.09, dwell=2):
    """Carry the held cube over target xy and lower it to z_low (EE z)."""
    s0 = rb.step_id
    p = rb.state[:3]
    if p[2] < z_trav - 0.02:
        rb.move_to([p[0], p[1], z_trav], tol=0.02, max_steps=25, chunk=3)
    # travel; begin descending a bit before arriving
    rb.move_to([xy[0], xy[1], z_trav], tol=0.02, max_steps=60, chunk=3)
    rb.move_to([xy[0], xy[1], z_low], tol=0.006, max_steps=40, chunk=3)
    if dwell:
        rb.hold(dwell)
    held = rb.state[6] + rb.state[7]
    log("visit", np.round(xy, 3), "EE", np.round(rb.state[:3], 3), "steps", rb.step_id - s0, "grip", round(held, 4))
    return held > 0.01


def target_lit(rb, xy, cams=("front", "wrist")):
    """Count red pixels lying on the table plane (z<0.016) within 4 cm of target xy."""
    o = rb.observe()
    best = 0
    for cam in cams:
        m = color_mask(o.rgb(cam), "red")
        P = o.deproject_mask(cam, m)
        if len(P) == 0:
            continue
        d = np.linalg.norm(P[:, :2] - np.asarray(xy)[None], axis=1)
        k = int(np.sum((P[:, 2] < 0.016) & (d < 0.045)))
        best = max(best, k)
    return best


def visit2(rb, xy, z_low, z_trav=0.09, dwell=2, probe=False):
    s0 = rb.step_id
    p = rb.state[:3]
    if p[2] < z_trav - 0.02:
        rb.move_to([p[0], p[1], z_trav], tol=0.02, max_steps=25, chunk=3)
    rb.move_to([xy[0], xy[1], z_trav], tol=0.015 if not probe else 0.006, max_steps=60, chunk=3)
    lit_high = target_lit(rb, xy) if probe else None
    rb.move_to([xy[0], xy[1], z_low], tol=0.006, max_steps=40, chunk=3)
    if dwell:
        rb.hold(dwell)
    lit = target_lit(rb, xy)
    held = rb.state[6] + rb.state[7]
    log("visit", np.round(xy, 3), "EE", np.round(rb.state[:3], 3), "steps", rb.step_id - s0,
        "grip", round(held, 4), "lit_high", lit_high, "lit", lit)
    return held > 0.01, lit
