import math, numpy as np
from lib import log
import percep

GRASP_Z = 0.045
SAFE_Z = 0.17

def grip_yaw(ang):
    a = ang % (math.pi / 2)
    return a if a <= math.pi / 4 else a - math.pi / 2

def pick_and_replace(rb, cont, lift_z=SAFE_Z, grasp_z=GRASP_Z, psi=None):
    """Grasp container dict (xy, yaw), lift it, place it back, release. Returns dict of results."""
    x, y = cont["xy"]
    if psi is None:
        psi = grip_yaw(cont["yaw"])
    res = {"xy": [x, y], "psi": psi}
    e, u = rb.move_to([x, y, SAFE_Z], grip=-1.0, yaw=psi, max_steps=40, tol=0.006, rmax=0.6)
    log("above", rb.step_id, np.round(rb.state[:3], 4), "e", round(e, 4))
    e, u = rb.move_to([x, y, grasp_z + 0.04], grip=-1.0, yaw=psi, max_steps=25, tol=0.004, vmax=0.6)
    e, u = rb.move_to([x, y, grasp_z], grip=-1.0, yaw=psi, max_steps=20, tol=0.003, vmax=0.3)
    log("at grasp", rb.step_id, np.round(rb.state[:3], 4), "yaw", round(math.degrees(__import__('lib').cur_yaw(rb.state[3:6])), 1))
    rb.hold(10, 1.0)
    width = float(rb.state[6] + rb.state[7])
    res["width_closed"] = width
    log("closed width", round(width, 4))
    e, u = rb.move_to([x, y, lift_z], grip=1.0, yaw=psi, max_steps=40, tol=0.006, vmax=0.6)
    rb.hold(6, 1.0)
    width2 = float(rb.state[6] + rb.state[7])
    res["width_lifted"] = width2
    res["holding"] = width2 > 0.02
    log("lifted", rb.step_id, np.round(rb.state[:3], 4), "width", round(width2, 4), "complete", rb.complete)
    return res

def place_back(rb, xy, psi, place_z=GRASP_Z + 0.012):
    x, y = xy
    rb.move_to([x, y, place_z + 0.03], grip=1.0, yaw=psi, max_steps=30, tol=0.005, vmax=0.6)
    rb.move_to([x, y, place_z], grip=1.0, yaw=psi, max_steps=15, tol=0.003, vmax=0.3)
    rb.hold(6, -1.0)
    rb.move_to([x, y, SAFE_Z], grip=-1.0, yaw=psi, max_steps=30, tol=0.008, vmax=0.8)
    log("placed back", rb.step_id, np.round(rb.state[:3], 4))
