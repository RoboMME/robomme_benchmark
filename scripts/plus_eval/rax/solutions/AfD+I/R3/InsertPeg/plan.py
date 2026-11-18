"""Video analysis and geometric planning (pure functions, offline-testable)."""
import numpy as np
from motion import rotvec_to_mat, mat_to_rotvec, rot_err

CAM_OFF = np.array([0.0465, -0.02, -0.0674])   # wrist camera position in EE frame (verified constant)


def analyze_video(S):
    """S: (N,8) video states. Returns plan dict with grasp, lift, pre-insert and insertion info."""
    S = np.asarray(S, float)
    wid = S[:, 6] + S[:, 7]
    w0 = wid[0]
    closed = np.nonzero(wid < w0 - 0.01)[0]
    if len(closed) == 0:
        raise RuntimeError("video: gripper never closes")
    ci = int(closed[0])
    # grasp pose: the lowest-z frame among the few frames before closing
    lo = max(0, ci - 8)
    gi = lo + int(np.argmin(S[lo:ci + 1, 2]))
    gi = min(gi, ci)
    end = len(S) - 1
    zf = S[end, 2]
    rf = S[end, 3:6]
    pre = end
    for k in range(end, ci, -1):
        if abs(S[k, 2] - zf) < 0.002 and np.linalg.norm(rot_err(rf, S[k, 3:6])) < 0.06:
            pre = k
        else:
            break
    d = S[end, :2] - S[pre, :2]
    L = float(np.linalg.norm(d))
    if L < 0.02:
        raise RuntimeError("video: no insertion segment found (L=%.3f)" % L)
    u = d / L
    # least-squares direction over the insertion segment (robust to endpoint noise)
    xy = S[pre:end + 1, :2]
    if len(xy) >= 4:
        c = xy - xy.mean(0)
        ev = np.linalg.svd(c, full_matrices=False)[2][0]
        if ev @ u < 0:
            ev = -ev
        if ev @ u > np.cos(np.radians(10)):
            u = ev / np.linalg.norm(ev)
    lift_z = float(np.max(S[ci:pre + 1, 2]))
    return dict(close_idx=ci, grasp_idx=gi, grasp=S[gi, :6].copy(), closed_width=float(wid[min(end, ci + 15)]),
                pre_idx=pre, pre=S[pre, :6].copy(), end=S[end, :6].copy(), u=u, L=L, lift_z=lift_z)


def frame_uw(u):
    u = np.asarray(u, float)[:2]; u = u / np.linalg.norm(u)
    w = np.array([-u[1], u[0]])
    return u, w


def uwz_to_xyz(a, wv, z, u):
    u, w = frame_uw(u)
    xy = a * u + wv * w
    return np.array([xy[0], xy[1], z])


def view_pose(F, u, R_ins_rotvec, tilt_rad, dist):
    """EE pose so the wrist camera looks at world point F along a direction tilted from straight down toward +u."""
    u2, w2 = frame_uw(u)
    U3 = np.array([u2[0], u2[1], 0.0]); W3 = np.array([w2[0], w2[1], 0.0])
    v = -np.cos(tilt_rad) * np.array([0, 0, 1.0]) + np.sin(tilt_rad) * U3
    R_ins = rotvec_to_mat(R_ins_rotvec)
    R_view = rotvec_to_mat(W3 * (-tilt_rad)) @ R_ins
    # check camera axis = R_view @ [0,0,1] (EE z) approx v
    C = np.asarray(F) - dist * v
    p = C - R_view @ CAM_OFF
    return p, mat_to_rotvec(R_view), v, C


def view_pose_offaxis(F, u, R_view_rotvec, beta_rad, dist):
    """EE pose (orientation fixed = R_view) so the wrist camera sees world point F along a line of sight
    descending at beta from vertical toward +u, from distance dist. Returns (EE pos, rotvec, C, pixel(u,v))."""
    u2, w2 = frame_uw(u)
    U3 = np.array([u2[0], u2[1], 0.0])
    d = np.sin(beta_rad) * U3 - np.cos(beta_rad) * np.array([0, 0, 1.0])
    C = np.asarray(F, float) - dist * d
    R = rotvec_to_mat(R_view_rotvec)
    p = C - R @ CAM_OFF
    # projection: camera axes x=EEy, y=EEx (image up), looking along EEz
    Rc = R @ np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1.0]])
    q = Rc.T @ (np.asarray(F) - C)
    px = 128 + 128 * q[0] / (-q[2]); py = 128 - 128 * q[1] / (-q[2])
    return p, np.asarray(R_view_rotvec, float), C, (float(px), float(py))
