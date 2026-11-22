import numpy as np
from core import step, log

MAXSTEP = 0.03  # m per control step


def rot_err(cur_aa, tgt_aa):
    from scipy.spatial.transform import Rotation as R
    e = R.from_rotvec(tgt_aa) * R.from_rotvec(cur_aa).inv()
    return e.as_rotvec()


def move_to(ctx, st, target, grip, tracker=None, tol=0.006, max_calls=40, maxstep=MAXSTEP,
            tgt_rot=None, chunk=5):
    """Move TCP to target (world xyz). st: dict with 'obs'. Returns final obs."""
    obs = st["obs"]
    for _ in range(max_calls):
        cur = obs.state[:3]
        r = np.asarray(target) - cur
        dist = np.linalg.norm(r)
        rot = np.zeros(3)
        if tgt_rot is not None:
            rot = rot_err(obs.state[3:6], tgt_rot)
        if dist < tol and np.linalg.norm(rot) < 0.03:
            break
        n = max(1, int(np.ceil(dist / maxstep)), int(np.ceil(np.linalg.norm(rot) / 0.25)))
        k = min(chunk, n)
        d = r / n
        dr = rot / n
        row = [d[0] / 0.05, d[1] / 0.05, d[2] / 0.05, dr[0] / 0.5, dr[1] / 0.5, dr[2] / 0.5, grip]
        obs = step(ctx, [row] * k)
        st["obs"] = obs
        if tracker is not None:
            tracker.update(obs)
        if obs.ended:
            break
    return obs


def hold(ctx, st, n, grip, tracker=None, chunk=5):
    obs = st["obs"]
    done = 0
    while done < n and not obs.ended:
        k = min(chunk, n - done)
        obs = step(ctx, [[0, 0, 0, 0, 0, 0, grip]] * k)
        done += k
        st["obs"] = obs
        if tracker is not None:
            tracker.update(obs)
    return obs
