import numpy as np
from scipy.spatial.transform import Rotation as Rot
from percep import gather
from motion import move_to, hold
from core import log

DOWN = np.array([-np.pi, 0.0, 0.0])


def refine(obs, xy, r=0.05):
    """Refine container center & yaw from top-surface points near xy."""
    P, C = gather(obs, 0.064, 0.085)
    m = np.hypot(P[:, 0] - xy[0], P[:, 1] - xy[1]) < r
    Q = P[m]
    if len(Q) < 30:
        return xy, None, len(Q)
    X = Q[:, :2]
    best = None
    c0 = X.mean(0)
    for a in np.linspace(0, np.pi / 2, 91)[:-1]:
        R = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
        Y = (X - c0) @ R.T
        area = np.ptp(Y[:, 0]) * np.ptp(Y[:, 1])
        if best is None or area < best[0]:
            lo, hi = Y.min(0), Y.max(0)
            ctr = c0 + ((lo + hi) / 2) @ R
            best = (area, a, ctr, np.ptp(Y[:, 0]), np.ptp(Y[:, 1]))
    _, a, ctr, w, h = best
    return ctr, a, len(Q)


def yaw_target(a, base=DOWN):
    d = (a + np.pi / 4) % (np.pi / 2) - np.pi / 4
    return (Rot.from_rotvec([0, 0, d]) * Rot.from_rotvec(base)).as_rotvec(), d


def pick_container(ctx, st, xy, tracker=None, zgrasp=0.045, zlift=0.20, put_back=True):
    obs = st["obs"]
    move_to(ctx, st, [xy[0], xy[1], 0.22], -1, tracker, tgt_rot=DOWN)
    ctr, a, n = refine(st["obs"], xy)
    log("refine1", np.round(xy, 3), "->", np.round(ctr, 3), a, n)
    rot, d = (yaw_target(a) if a is not None else (DOWN, 0.0))
    move_to(ctx, st, [ctr[0], ctr[1], 0.16], -1, tracker, tgt_rot=rot)
    ctr2, a2, n2 = refine(st["obs"], ctr)
    log("refine2", np.round(ctr2, 3), a2, n2)
    if a2 is not None and np.linalg.norm(ctr2 - ctr) < 0.03:
        ctr = ctr2
        rot, d = yaw_target(a2)
    move_to(ctx, st, [ctr[0], ctr[1], 0.12], -1, tracker, tgt_rot=rot)
    move_to(ctx, st, [ctr[0], ctr[1], zgrasp], -1, tracker, tgt_rot=rot, maxstep=0.02, max_calls=15)
    hold(ctx, st, 10, 1)
    width = st["obs"].state[6] + st["obs"].state[7]
    log("grip width", round(float(width), 4), "z", round(float(st["obs"].state[2]), 3))
    move_to(ctx, st, [ctr[0], ctr[1], zlift], 1, None, tgt_rot=rot, maxstep=0.02, max_calls=20)
    width2 = st["obs"].state[6] + st["obs"].state[7]
    log("lifted width", round(float(width2), 4), "z", round(float(st["obs"].state[2]), 3), "step", st["obs"].step)
    ok = width2 > 0.03
    if put_back:
        move_to(ctx, st, [ctr[0], ctr[1], zgrasp + 0.01], 1, None, tgt_rot=rot, maxstep=0.02, max_calls=15)
        hold(ctx, st, 6, -1)
        move_to(ctx, st, [ctr[0], ctr[1], 0.2], -1, None, tgt_rot=rot)
    return ok, ctr
