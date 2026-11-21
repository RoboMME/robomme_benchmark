import numpy as np
from robot import move_to, hold, log
from live import obs_points, top_components

SCAN_Z = 0.30
HOVER_Z = 0.15
GRASP_Z = 0.035
LIFT_Z = 0.16
OPEN_G = -0.45

def s_yaw(s):
    from scipy.spatial.transform import Rotation as Rot
    R = Rot.from_rotvec(s[3:6]) * Rot.from_rotvec([np.pi, 0, 0]).inv()
    return float(R.as_euler('zyx')[0])

def wrist_components(rb):
    m = rb.observe()
    return top_components(obs_points(m, ('wrist',)), min_pts=40), m

def nearest(comps, xy, maxd):
    if not comps: return None
    d = [np.linalg.norm(c['xy'] - xy) for c in comps]
    i = int(np.argmin(d))
    return comps[i] if d[i] < maxd else None

def choose_yaw(c, comps):
    """yaw for gripper: fingers close along gripper y axis. choose of yaw, yaw+90 the one with
    more clearance from neighbours along the finger axis."""
    best = None
    for yaw in (c['yaw'], c['yaw'] + np.pi / 2):
        ax = np.array([-np.sin(yaw), np.cos(yaw)])  # gripper y in world at this yaw (assumed)
        clr = 1.0
        for o in comps:
            if o is c: continue
            v = o['xy'] - c['xy']; dist = np.linalg.norm(v)
            if dist < 1e-6 or dist > 0.12: continue
            along = abs(v @ ax) / dist
            clr = min(clr, dist / (0.3 + along))
        if best is None or clr > best[0]: best = (clr, yaw)
    yaw = best[1]
    yaw = (yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4 if False else yaw
    while yaw > np.pi / 2: yaw -= np.pi
    while yaw < -np.pi / 2: yaw += np.pi
    return float(yaw)

def locate(rb, guess_xy, maxd=0.04):
    s = rb.state()
    cam_off = np.zeros(2)
    move_to(rb, [guess_xy[0], guess_xy[1], SCAN_Z], s_yaw(s), -1, max_steps=50, tol=0.01)
    comps, m = wrist_components(rb)
    c = nearest(comps, np.asarray(guess_xy), maxd)
    if c is None: c = nearest(comps, np.asarray(guess_xy), 0.08)
    log('scan comps', [(np.round(o['xy'], 3).tolist(), o['n']) for o in comps], 'pick', None if c is None else c['xy'].round(3))
    if c is None: return None, comps
    move_to(rb, [c['xy'][0], c['xy'][1], HOVER_Z], s_yaw(rb.state()), OPEN_G, max_steps=50, tol=0.006)
    comps2, m = wrist_components(rb)
    c2 = nearest(comps2, c['xy'], 0.025)
    log('hover refine', None if c2 is None else (c2['xy'].round(3), round(np.rad2deg(c2['yaw'])), c2['n']))
    if c2 is not None: c = c2; comps = comps2
    return c, comps

def pick_and_show(rb, guess_xy, put_back=True):
    c, comps = locate(rb, guess_xy)
    if c is None: return {'ok': False, 'why': 'not found'}
    yaw = choose_yaw(c, comps)
    xy = c['xy']
    move_to(rb, [xy[0], xy[1], HOVER_Z], yaw, OPEN_G, max_steps=40, tol=0.004, rtol=0.02)
    move_to(rb, [xy[0], xy[1], 0.09], yaw, OPEN_G, max_steps=30, tol=0.004, rtol=0.02)
    move_to(rb, [xy[0], xy[1], GRASP_Z], yaw, OPEN_G, max_steps=30, tol=0.004, rtol=0.02, vmax=0.5)
    hold(rb, 1, 10)
    st = rb.state(); log('grip fingers', st[6:8].round(4), 'z', round(st[2], 3))
    move_to(rb, [xy[0], xy[1], LIFT_Z], yaw, 1, max_steps=30, tol=0.008)
    hold(rb, 1, 4)
    st = rb.state(); held = st[6] + st[7] > 0.01
    log('after lift fingers', st[6:8].round(4), 'z', round(st[2], 3), 'held', held)
    if put_back:
        move_to(rb, [xy[0], xy[1], GRASP_Z + 0.008], yaw, 1, max_steps=40, tol=0.005, vmax=0.6)
        hold(rb, OPEN_G, 6)
        move_to(rb, [xy[0], xy[1], HOVER_Z], yaw, OPEN_G, max_steps=30, tol=0.01)
    return {'ok': held, 'xy': xy.tolist(), 'yaw': yaw}
