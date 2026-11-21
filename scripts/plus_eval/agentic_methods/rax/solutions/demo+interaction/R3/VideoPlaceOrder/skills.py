import numpy as np
from robot import backproject
from perception import cube_mask
from scipy import ndimage

def locate_cube_world(robot, color):
    rgb = robot.imgs[('front', 'rgb')][..., :3]; dep = robot.imgs[('front', 'depth')]
    m = cube_mask(rgb, color)
    lab, n = ndimage.label(m)
    if n == 0: return None
    sizes = ndimage.sum(m, lab, range(1, n + 1)); k = int(np.argmax(sizes)) + 1
    vs, us = np.nonzero(lab == k)
    P = backproject(robot.meta, dep, 'front', us, vs)
    if len(P) < 5: return None
    top = P[:, 2].max()
    T = P[P[:, 2] > top - 0.008]
    return np.array([T[:, 0].mean(), T[:, 1].mean(), top])

def pick_place(robot, color, target_xy, log=print, hover=0.14):
    robot.observe()
    c = locate_cube_world(robot, color)
    log(f"cube world {c}")
    if c is None: return False
    z_grasp = max(0.015, c[2] - 0.022)
    robot.move_to([c[0], c[1], hover], -1, tol=0.006)
    robot.move_to([c[0], c[1], z_grasp + 0.04], -1, tol=0.004)
    # refine with wrist-independent re-observation
    robot.move_to([c[0], c[1], z_grasp], -1, tol=0.003, max_steps=30)
    robot.grip(1, 12)
    log(f"after close fingers {robot.state[6:]}")
    robot.move_to([c[0], c[1], hover], 1, tol=0.008)
    if robot.ended: return robot.success
    fing = robot.state[6:].sum()
    log(f"lifted, finger sum {fing:.4f}")
    robot.move_to([target_xy[0], target_xy[1], hover], 1, tol=0.006)
    robot.move_to([target_xy[0], target_xy[1], c[2] + 0.025], 1, tol=0.004, max_steps=40)
    robot.grip(-1, 10)
    robot.move_to([target_xy[0], target_xy[1], hover], -1, tol=0.01, max_steps=20)
    return robot.success
