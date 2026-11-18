import math
import numpy as np
from perception import *
from motion import Robot, log

CUBE_HALF = 0.02


def locate_cube(R, color="green", cam="wrist"):
    res, _ = find_objects(R.meta, R.imgs, cam, color)
    if not res["cube"]:
        return None
    c = res["cube"][0]
    yaw = rect_yaw(c["top_pts"][:, :2]) if len(c["top_pts"]) > 20 else 0.0
    return dict(xy=c["xy"], top=c["top"], yaw=yaw, n=c["n"])


def grasp_cube(R, color="green", hover=0.12):
    """Assumes the cube is roughly visible. Returns True if gripper holds something."""
    R.gripper(-1, 6)
    c = locate_cube(R, color, "wrist") or locate_cube(R, color, "front")
    if c is None:
        log("grasp: cube not found"); return False
    # approach over cube and refine
    R.move_to([c["xy"][0], c["xy"][1], hover], speed=0.025)
    c2 = locate_cube(R, color, "wrist")
    if c2 is not None and np.linalg.norm(c2["xy"] - c["xy"]) < 0.1:
        c = c2
    dyaw = wrap90(c["yaw"] - finger_yaw(R.state))
    log(f"grasp: cube xy={c['xy'].round(3)} top={c['top']:.3f} yaw={math.degrees(c['yaw']):.1f} dyaw={math.degrees(dyaw):.1f}")
    from motion import rotate_yaw
    rotate_yaw(R, dyaw)
    # refine once more after rotation
    c3 = locate_cube(R, color, "wrist")
    if c3 is not None and np.linalg.norm(c3["xy"] - c["xy"]) < 0.05:
        c = c3
    zg = max(c["top"] - CUBE_HALF - 0.002, 0.012)
    R.move_to([c["xy"][0], c["xy"][1], zg + 0.04], speed=0.02)
    R.move_to([c["xy"][0], c["xy"][1], zg], speed=0.01, max_iter=4)
    R.gripper(1, 12)
    w = R.state[6] + R.state[7]
    log(f"grasp: closed width={w:.4f} pos={R.pos.round(3)}")
    return w > 0.02


def lift(R, z=0.15):
    p = R.pos
    R.move_to([p[0], p[1], z], speed=0.02, tol=0.01)
    w = R.state[6] + R.state[7]
    return w > 0.02


def locate_target(R):
    for cam in ("wrist", "front"):
        res, _ = find_objects(R.meta, R.imgs, cam)
        if res["target"] is not None and res["target"]["n"] >= 15:
            return res["target"], cam
    return None, None


def place_on_target(R, tgt_xy, tgt_z, carry_z=0.15):
    p = R.pos
    R.move_to([tgt_xy[0], tgt_xy[1], carry_z], speed=0.025, tol=0.006)
    zp = tgt_z + CUBE_HALF + 0.006
    R.move_to([tgt_xy[0], tgt_xy[1], zp + 0.04], speed=0.02)
    R.move_to([tgt_xy[0], tgt_xy[1], zp], speed=0.01, max_iter=4)
    R.gripper(-1, 10)
    log(f"place: released at {R.pos.round(3)} target {np.round(tgt_xy,3)}")
