import numpy as np
from lib import *
from percep import *

HOVER_Z = 0.16
GRASP_DZ = 0.022     # grasp TCP this far below cube top
LIFT_Z = 0.15


def refine(ctx, xy, yaw0=0.0):
    """Hover over xy with the wrist camera, return refined (xy, top_z, yaw)."""
    o, _ = servo(ctx, [xy[0], xy[1], HOVER_Z], yaw=yaw0, grip=-1)
    for k in range(2):
        pts = cube_top_points(o, 'wrist', xy)
        if len(pts) < 20:
            return o, None
        xy = pts[:, :2].mean(0)
        yaw = square_yaw(pts[:, :2]); top = float(np.median(pts[:, 2]))
        log('refine', k, 'xy', xy.round(4), 'top %.3f yaw %.1f n %d' % (top, np.degrees(yaw), len(pts)))
        if k == 0:
            o, _ = servo(ctx, [xy[0], xy[1], HOVER_Z - 0.04], yaw=yaw0, grip=-1, obs=o)
    return o, (xy, top, yaw)


def pick(ctx, xy):
    o, r = refine(ctx, xy)
    if r is None:
        return o, False, None
    xy, top, yaw = r
    gz = max(top - GRASP_DZ, 0.012)
    o, _ = servo(ctx, [xy[0], xy[1], top + 0.05], yaw=yaw, grip=-1, obs=o)
    o, _ = servo(ctx, [xy[0], xy[1], gz], yaw=yaw, grip=-1, obs=o, max_iter=25)
    log('at grasp', o.state[:3].round(4), 'yaw', round(np.degrees(ee_yaw(o.state)), 1))
    o = hold(ctx, 1.0, 12)
    log('closed fingers', o.state[6:8].round(4))
    o, _ = servo(ctx, [xy[0], xy[1], LIFT_Z], yaw=yaw, grip=1.0, obs=o)
    width = o.state[6] + o.state[7]
    held = width > 0.01
    log('lifted', o.state[:3].round(3), 'width %.4f held %s' % (width, held))
    return o, held, (xy, top, yaw)


def place(ctx, xy, top, yaw):
    o, _ = servo(ctx, [xy[0], xy[1], top + 0.03], yaw=yaw, grip=1.0)
    o, _ = servo(ctx, [xy[0], xy[1], top - GRASP_DZ + 0.006], yaw=yaw, grip=1.0, obs=o, max_iter=20)
    o = hold(ctx, -1.0, 8)
    o, _ = servo(ctx, [xy[0], xy[1], HOVER_Z], yaw=yaw, grip=-1.0, obs=o)
    return o
