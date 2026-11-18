import json, math, time
import numpy as np

def packet(result):
    for b in result.get("content", []):
        if b.get("type") == "text":
            try:
                return json.loads(b["text"])
            except Exception:
                pass
    return {}

def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

def rotvec_to_R(v):
    v = np.asarray(v, float); th = np.linalg.norm(v)
    if th < 1e-12:
        return np.eye(3)
    k = v / th; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K

def R_to_rotvec(R):
    c = max(-1.0, min(1.0, (np.trace(R) - 1) / 2)); th = math.acos(c)
    if th < 1e-9:
        return np.zeros(3)
    if th > math.pi - 1e-4:
        # near pi: use symmetric part
        A = (R + np.eye(3)) / 2
        i = int(np.argmax(np.diag(A))); k = A[:, i] / math.sqrt(max(A[i, i], 1e-12))
        return k * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return w * th

def rot_err(cur_rv, tgt_rv):
    """world-frame rotvec taking current to target: R_t R_c^T"""
    return R_to_rotvec(rotvec_to_R(tgt_rv) @ rotvec_to_R(cur_rv).T)

def make_row(cur, tgt, grip, pgain=1.0, rgain=1.0, pmax=1.0, rmax=1.0):
    dp = (np.asarray(tgt[:3]) - np.asarray(cur[:3])) / 0.05 * pgain
    n = np.linalg.norm(dp)
    if n > pmax:
        dp = dp * pmax / n
    dr = rot_err(cur[3:6], tgt[3:6]) / 0.5 * rgain
    n = np.linalg.norm(dr)
    if n > rmax:
        dr = dr * rmax / n
    return [float(x) for x in dp] + [float(x) for x in dr] + [float(grip)]
