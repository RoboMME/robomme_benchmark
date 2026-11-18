import os
import numpy as np
from tracker import points_from_obs

def box_yaw(pts, center, r=0.032):
    p = pts[np.linalg.norm(pts - center, axis=1) < r]
    if len(p) < 20:
        return None, center
    best = (1e9, 0.0)
    for deg in np.arange(0, 90, 2.0):
        t = np.radians(deg)
        a = p @ np.array([np.cos(t), np.sin(t)])
        b = p @ np.array([-np.sin(t), np.cos(t)])
        ra = np.percentile(a, 98) - np.percentile(a, 2)
        rb = np.percentile(b, 98) - np.percentile(b, 2)
        if ra * rb < best[0]:
            best = (ra * rb, t)
    t = best[1]
    a = p @ np.array([np.cos(t), np.sin(t)]); b = p @ np.array([-np.sin(t), np.cos(t)])
    ca = (np.percentile(a, 98) + np.percentile(a, 2)) / 2
    cb = (np.percentile(b, 98) + np.percentile(b, 2)) / 2
    c = ca * np.array([np.cos(t), np.sin(t)]) + cb * np.array([-np.sin(t), np.cos(t)])
    return t, c

def nearest_equiv(target, current, period=np.pi / 2):
    d = (target - current + period / 2) % period - period / 2
    return current + d

Z_ABOVE, Z_GRASP, Z_LIFT = 0.16, 0.045, 0.17

def localize(pts, guess):
    c = np.asarray(guess, float)
    for _ in range(3):
        p = pts[np.linalg.norm(pts - c, axis=1) < 0.04]
        if len(p) < 25:
            break
        c = p.mean(0)
    th, cc = box_yaw(pts, c)
    if th is None:
        return 0.0, np.asarray(guess, float), False
    if np.linalg.norm(cc - np.asarray(guess)) > 0.05:
        return th, np.asarray(guess, float), False
    return th, cc, True

def pick_and_return(rb, xy_guess, log=print, z_grasp=Z_GRASP, tries=3):
    """Grasp container near xy_guess, lift, put it back down where it was, release, retreat."""
    guess = np.asarray(xy_guess, float)
    held, c = False, guess
    for attempt in range(tries):
        if rb.ended:
            break
        pts = points_from_obs(rb.meta, rb.imgs)
        th, c, ok = localize(pts, guess)
        yaw = nearest_equiv(th, rb.yaw())
        log(f"pick try{attempt}: center={np.round(c,3).tolist()} ok={ok} boxyaw={np.degrees(th):.0f} toolyaw={np.degrees(yaw):.0f}")
        rb.move_to([c[0], c[1], Z_ABOVE], yaw=yaw, max_steps=80)
        zg = z_grasp
        if attempt == 0 and os.environ.get("VUS_TEST_MISS"):
            zg = 0.13  # dev-only: force a miss to exercise the retry path
        rb.move_to([c[0], c[1], zg], yaw=yaw, max_steps=60, vmax=0.4)
        rb.hold(10, 1.0)
        fingers = float(rb.state[6] + rb.state[7])
        rb.move_to([c[0], c[1], Z_LIFT], yaw=yaw, grip=1.0, max_steps=60, vmax=0.5)
        fingers_up = float(rb.state[6] + rb.state[7])
        held = 0.03 < fingers_up < 0.078
        log(f"pick: fingers closed={fingers:.4f} after lift={fingers_up:.4f} held={held} step={rb.meta['step_id']} ended={rb.ended}")
        if rb.ended:
            return dict(held=held, c=c)
        if held:
            break
        # missed: open, back off high, re-observe and retry near the same place
        rb.hold(5, -1.0)
        rb.move_to([c[0], c[1], 0.30], grip=-1.0, max_steps=40)
        rb.hold(2, -1.0)
        guess = c
    if not held:
        return dict(held=False, c=c)
    rb.hold(3, 1.0)
    # put back
    rb.move_to([c[0], c[1], z_grasp + 0.006], yaw=yaw, grip=1.0, max_steps=60, vmax=0.4)
    rb.hold(6, -1.0)
    rb.move_to([c[0], c[1], Z_ABOVE], yaw=yaw, grip=-1.0, max_steps=60)
    return dict(held=True, c=c)
