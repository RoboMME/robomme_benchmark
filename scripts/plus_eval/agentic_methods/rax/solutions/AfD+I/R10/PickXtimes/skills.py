"""Pick / place / press skills."""
import math
import numpy as np
import robo, percep
from robo import log

HOVER_Z = 0.13
GRASP_Z = 0.018
CUBE = 0.04
LIFT_Z = 0.10

def wrap45(a):
    return (a + math.pi / 4) % (math.pi / 2) - math.pi / 4

def locate_cube(r, guess_xy=None):
    m = r.observe()
    c = percep.find_cube(m)
    return c

def pick(r, cube_xy, tries=3, color="green"):
    """Grasp the cube near cube_xy and lift to HOVER_Z. Returns True if held."""
    xy = np.asarray(cube_xy, float)
    for attempt in range(tries):
        if r.ended:
            return False
        r.move_to([xy[0], xy[1], HOVER_Z], grip=-1, speed=1.0, max_steps=60)
        r.hold(2)
        m = r.observe()
        c = percep.find_cube(m, cams=("wrist",), color=color)
        if c is None:
            c = percep.find_cube(m, color=color)
        if c is None:
            log("pick: cube not visible at hover")
            return False
        xy = c["xy"]; yaw = wrap45(c["yaw"] or 0.0)
        log(f"pick attempt {attempt}: cube xy={np.round(xy,3)} ztop={c['ztop']:.3f} yaw={yaw:.2f} cam={c['cam']} n={c['n']}")
        # keep current yaw if within symmetric equivalence
        cur = r.yaw()
        tgt = cur + wrap45(yaw - cur)
        if abs(tgt) > 2.0:
            tgt = yaw
        r.yaw_d = tgt
        r.move_to([xy[0], xy[1], HOVER_Z - 0.03], grip=-1, speed=0.8, max_steps=40)
        r.rotate_to(tgt)
        zg = max(0.012, c["ztop"] - CUBE + GRASP_Z) if c["ztop"] < 0.07 else GRASP_Z
        r.move_to([xy[0], xy[1], zg + 0.04], grip=-1, speed=0.8, max_steps=40)
        r.move_to([xy[0], xy[1], zg], grip=-1, speed=0.4, max_steps=40, tol=0.003)
        r.gripper(1, n=10)
        w = float(r.state[6] + r.state[7])
        log(f"pick: closed width={w:.4f} tcp={np.round(r.state[:3],3)}")
        r.move_to([xy[0], xy[1], LIFT_Z], grip=1, speed=0.6, max_steps=60)
        w2 = float(r.state[6] + r.state[7])
        log(f"pick: lifted width={w2:.4f}")
        if 0.025 < w2 < 0.05:
            return True
        # missed: reopen and re-locate
        r.gripper(-1, n=6)
        m = r.observe()
        c2 = percep.find_cube(m, color=color)
        if c2 is not None:
            xy = c2["xy"]
    return False

def place(r, target_xy, carry_z=HOVER_Z):
    t = np.asarray(target_xy, float)
    r.move_to([t[0], t[1], carry_z], grip=1, speed=0.6, max_steps=80)
    r.move_to([t[0], t[1], GRASP_Z + 0.04], grip=1, speed=0.5, max_steps=40)
    r.move_to([t[0], t[1], GRASP_Z + 0.006], grip=1, speed=0.25, max_steps=30, tol=0.003)
    r.gripper(-1, n=6)
    r.move_to([t[0], t[1], HOVER_Z], grip=-1, speed=0.8, max_steps=40)

def press_button(r, bxy, bztop, extra=0.0):
    b = np.asarray(bxy, float)
    r.move_to([b[0], b[1], HOVER_Z], grip=1, speed=1.0, max_steps=60)
    r.gripper(1, n=6)
    r.move_to([b[0], b[1], bztop + 0.03], grip=1, speed=0.6, max_steps=40)
    z0 = r.state[2]
    for k in range(14):
        if r.ended:
            break
        r.step([[0, 0, -0.2, 0, 0, 0, 1]] * 2)
        log(f"press: z={r.state[2]:.4f}")
        if r.state[2] < bztop - 0.02 - extra:
            break
    r.move_to([b[0], b[1], HOVER_Z], grip=1, speed=0.6, max_steps=30)
