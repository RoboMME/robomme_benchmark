import numpy as np
from robot import reply_images, servo_pose, finger_yaw, wrap
from tracker import white_world_points

def top_face(m, ims, xy, rad=0.035, cams=('wrist', 'front')):
    r1 = _top_face(m, ims, xy, rad, cams)
    if r1 is None:
        return None
    r2 = _top_face(m, ims, r1['xy'], 0.029, cams)
    return r2 if r2 is not None else r1

def _top_face(m, ims, xy, rad, cams):
    pts = []
    for cam in cams:
        if (cam, 'depth') not in ims:
            continue
        K = m['camera_intrinsics'][cam]['matrix']; P = m['camera_poses'][cam]
        Pw = white_world_points(ims[(cam, 'rgb')], ims[(cam, 'depth')], K, P, zmin=0.03, zmax=0.11)
        pts.append(Pw[np.linalg.norm(Pw[:, :2] - np.asarray(xy), axis=1) < rad])
    if not pts:
        return None
    Pw = np.concatenate(pts)
    if len(Pw) < 20:
        return None
    ztop = np.percentile(Pw[:, 2], 95)
    top = Pw[Pw[:, 2] > ztop - 0.008]
    if len(top) < 10:
        return None
    xy, (w, h), ang = min_area_rect(top[:, :2])
    return {'xy': xy, 'z': float(ztop), 'size': (w, h), 'ang': ang, 'n': len(top)}

def min_area_rect(Q):
    best = None
    for a in np.radians(np.arange(0, 90, 1.0)):
        u = np.array([np.cos(a), np.sin(a)]); w = np.array([-u[1], u[0]])
        pu, pw = Q @ u, Q @ w
        lo_u, hi_u = np.percentile(pu, [2, 98]); lo_w, hi_w = np.percentile(pw, [2, 98])
        area = (hi_u - lo_u) * (hi_w - lo_w)
        if best is None or area < best[0]:
            c = u * (lo_u + hi_u) / 2 + w * (lo_w + hi_w) / 2
            best = (area, c, (hi_u - lo_u, hi_w - lo_w), a)
    return best[1], best[2], best[3]

def choose_yaw(face_ang, cur_yaw):
    cands = [face_ang + k * np.pi / 2 for k in range(-4, 5)]
    return min(cands, key=lambda a: abs(wrap(a - cur_yaw)) if abs(a - cur_yaw) < 2.5 else 9)

def pick_container(robot, ctx, xy_guess, log=print, grasp_z=0.045, lift_z=0.20):
    """Hover, refine top face, align yaw, descend, close, lift. Returns dict."""
    s = robot.state
    servo_pose(robot, [xy_guess[0], xy_guess[1], 0.17], -1, yaw=finger_yaw(s), chunk=5)
    m, ims = reply_images(ctx.call('robomme_observe')); robot._upd(m)
    tf = top_face(m, ims, xy_guess)
    log(f'top_face {tf}')
    if tf is None or np.linalg.norm(tf['xy'] - np.asarray(xy_guess)) > 0.04:
        return {'ok': False, 'why': 'no top face', 'tf': tf}
    yaw = choose_yaw(tf['ang'], finger_yaw(robot.state))
    xy = tf['xy']
    servo_pose(robot, [xy[0], xy[1], 0.13], -1, yaw=yaw, chunk=5)
    gz = min(grasp_z, tf['z'] - 0.025)
    servo_pose(robot, [xy[0], xy[1], gz], -1, yaw=yaw, chunk=4, vmax=0.6)
    robot.step([[0, 0, 0, 0, 0, 0, 1]] * 12)
    servo_pose(robot, [xy[0], xy[1], lift_z], 1, yaw=yaw, chunk=5)
    width = float(robot.state[6] + robot.state[7])
    return {'ok': width > 0.01, 'width': width, 'xy': xy, 'yaw': yaw}

def place_at(robot, xy, z=0.075, yaw=None):
    yaw = finger_yaw(robot.state) if yaw is None else yaw
    servo_pose(robot, [xy[0], xy[1], 0.20], 1, yaw=yaw, chunk=5)
    servo_pose(robot, [xy[0], xy[1], z], 1, yaw=yaw, chunk=4, vmax=0.6)
    robot.step([[0, 0, 0, 0, 0, 0, -1]] * 8)
    servo_pose(robot, [xy[0], xy[1], 0.20], -1, yaw=yaw, chunk=5)
