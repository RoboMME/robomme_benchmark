"""Manipulation skills: grasp, peek (lift-aside-look), restore, pick."""
import numpy as np
from perception import (frame_from_reply, live_containers, load_frame, cube_detections, in_table)
from motion import Robot, yaw_of_state

GRASP_Z = 0.045
CARRY_Z = 0.10     # container bottom ~5.5cm above table: clears cubes, below pick threshold
SAFE_Z = 0.16
LOOK_Z = 0.30
HOME = [-0.06, 0.0, 0.50]


def observe_frame(rb):
    reply = rb.ctx.call('robomme_observe')
    f = frame_from_reply(reply)
    rb.last = {k: v for k, v in f.items() if k != 'arrays'}
    rb._lim(rb.last)
    return f


def obstacle_points(frame):
    """xy of all elevated points (containers, pillars, etc.) from both cameras."""
    pts = []
    for cam in ('front', 'wrist'):
        try:
            rgb, P = load_frame(frame, cam)
        except Exception:
            continue
        with np.errstate(invalid='ignore'):
            ok = in_table(P) & (P[..., 2] > 0.012)
        pts.append(P[ok][:, :2])
    return np.concatenate(pts) if pts else np.zeros((0, 2))


def choose_aside(frame, xy, exclude_r=0.045, dist=0.10, log=print):
    """Pick a free spot about `dist` from xy, away from other elevated objects."""
    obs = obstacle_points(frame)
    if len(obs):
        obs = obs[np.linalg.norm(obs - xy, axis=1) > exclude_r]
        obs = obs[::4]
    best = None
    for d in (dist, dist + 0.03, dist + 0.06):
        for a in np.deg2rad(np.arange(0, 360, 20)):
            p = xy + d * np.array([np.cos(a), np.sin(a)])
            # workspace limits (reachable, in view)
            if not (-0.30 < p[0] < 0.17 and -0.30 < p[1] < 0.30):
                continue
            clr = np.min(np.linalg.norm(obs - p, axis=1)) if len(obs) else 1.0
            # path clearance: sample along segment
            seg = [xy + t * (p - xy) for t in (0.4, 0.55, 0.7, 0.85)]
            pclr = min(np.min(np.linalg.norm(obs - s, axis=1)) if len(obs) else 1.0 for s in seg)
            score = min(clr, pclr + 0.005)
            if best is None or score > best[0]:
                best = (score, p)
        if best is not None and best[0] > 0.065:
            break
    log(f'aside: {best[1].round(3)} clearance {best[0]:.3f}')
    return best[1], best[0]


def grasp_at(rb, xy, yaw, log=print):
    X, Y = xy
    rb.move_to([X, Y, SAFE_Z], yaw, -1, max_iter=60, tol=0.012, vmax=1.0, gain=1.5)
    rb.move_to([X, Y, GRASP_Z], yaw, -1, max_iter=40, tol=0.004, vmax=0.8, gain=1.3)
    rb.hold(1, 9)
    s = rb.state()
    width = s[6] + s[7]
    ok = 0.035 < width < 0.075
    log(f'grasp at {np.round(xy,3)} width {width:.4f} ok={ok}')
    return ok


def carry_to(rb, xy, z=CARRY_Z):
    s = rb.state(); yaw = yaw_of_state(s)
    rb.move_to([s[0], s[1], z], yaw, 1, max_iter=25, tol=0.008, vmax=0.5)
    rb.move_to([xy[0], xy[1], z], yaw, 1, max_iter=50, tol=0.004, vmax=0.5)


def lower_release(rb, z=0.05):
    s = rb.state(); yaw = yaw_of_state(s)
    rb.move_to([s[0], s[1], z], yaw, 1, max_iter=25, tol=0.005, vmax=0.4)
    rb.hold(-1, 6)
    rb.move_to([s[0], s[1], SAFE_Z], yaw, -1, max_iter=25, tol=0.015, vmax=1.0, gain=1.5)


def look_cube(rb, xy, log=print):
    """Move wrist camera above xy and detect a colored cube near xy. Returns (color, cube_xy)."""
    s = rb.state(); yaw = yaw_of_state(s)
    # wrist camera is offset (0.0465, 0.02) from the TCP in the yaw frame: put the camera over the spot
    c, sn = np.cos(yaw), np.sin(yaw)
    off = np.array([c * 0.0465 - sn * 0.02, sn * 0.0465 + c * 0.02])
    t = np.array(xy) - off
    rb.move_to([t[0], t[1], LOOK_Z], yaw, -1, max_iter=40, tol=0.015, vmax=1.0, gain=1.5)
    f = observe_frame(rb)
    best = None
    for cam in ('wrist', 'front'):
        rgb, P = load_frame(f, cam)
        for c in cube_detections(rgb, P):
            d = np.hypot(c['x'] - xy[0], c['y'] - xy[1])
            if d < 0.05 and c['n'] >= 8 and (best is None or c['n'] > best[2]):
                best = (c['color'], np.array([c['x'], c['y']]), c['n'], cam)
    if best is None:
        # look for dark / white cubes: non-table pixels at cube height near xy
        rgb, P = load_frame(f, 'wrist')
        with np.errstate(invalid='ignore'):
            near = (np.hypot(P[..., 0] - xy[0], P[..., 1] - xy[1]) < 0.04) & (P[..., 2] > 0.015) & (P[..., 2] < 0.045)
        if near.sum() > 20:
            col = rgb[near].mean(0)
            name = 'black' if col.max() < 70 else ('white' if col.min() > 150 else 'unknown')
            p = P[near][:, :2].mean(0)
            log(f'look: achromatic object near spot mean rgb {col.round(0)} -> {name}')
            return name, p, f
        log('look: nothing under container')
        return None, None, f
    log(f'look: {best[0]} at {best[1].round(3)} ({best[3]}, n={best[2]})')
    return best[0], best[1], f
