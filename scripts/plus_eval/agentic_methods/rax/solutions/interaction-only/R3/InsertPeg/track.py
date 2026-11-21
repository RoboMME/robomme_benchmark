"""Closed-loop tracking of the conditioning video's end-effector state trajectory."""
import json, time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R

WS = Path("/workspace")
POS_SCALE = 0.05
ROT_SCALE = 0.5


def meta(reply):
    for b in reply.get("content", []):
        if b.get("type") == "text":
            try:
                return json.loads(b["text"])
            except Exception:
                pass
    return {}


def load_video(manifest_path):
    m = json.loads((WS / manifest_path).read_text())
    frames = sorted(m["frames"], key=lambda f: f["frame_index"])
    vid = [f for f in frames if f["phase"] == "video_demo"]
    ex = [f for f in frames if f["phase"] != "video_demo"]
    return m, vid, ex


def row_to(cur, tgt, grip, pos_gain=1.0, rot_gain=1.0, max_pos=1.0, max_rot=1.0):
    cur = np.asarray(cur, float); tgt = np.asarray(tgt, float)
    dp = (tgt[:3] - cur[:3]) * pos_gain / POS_SCALE
    n = np.linalg.norm(dp)
    if n > max_pos:
        dp = dp * max_pos / n
    rc = R.from_rotvec(cur[3:6]); rt = R.from_rotvec(tgt[3:6])
    dr = (rt * rc.inv()).as_rotvec() * rot_gain / ROT_SCALE
    n = np.linalg.norm(dr)
    if n > max_rot:
        dr = dr * max_rot / n
    return [float(x) for x in np.concatenate([np.clip(dp, -1, 1), np.clip(dr, -1, 1), [grip]])]


def pose_err(cur, tgt):
    cur = np.asarray(cur, float); tgt = np.asarray(tgt, float)
    pe = float(np.linalg.norm(tgt[:3] - cur[:3]))
    re = float((R.from_rotvec(tgt[3:6]) * R.from_rotvec(cur[3:6]).inv()).magnitude())
    return pe, re


class Robot:
    def __init__(self, ctx, log):
        self.ctx = ctx; self.log = log
        self.state = None; self.ended = False; self.complete = False
        self.remaining = None; self.step_id = 0; self.last = {}

    def observe(self):
        r = self.ctx.call("robomme_observe")
        m = meta(r); self._upd(m); return m

    def _upd(self, m):
        if not m:
            return
        self.last = m
        st = m.get("state") or (m.get("observation") or {}).get("state")
        if st:
            self.state = st
        if "step_id" in m:
            self.step_id = m["step_id"]
        if m.get("episode_ended"):
            self.ended = True
        if m.get("task_complete"):
            self.complete = True
        lim = m.get("episode_limits") or {}
        if "remaining_episode_steps" in lim:
            self.remaining = lim["remaining_episode_steps"]

    def step(self, rows):
        if self.ended:
            return {}
        r = self.ctx.call("robomme_step", actions=rows)
        m = meta(r); self._upd(m)
        return m


def grip_cmd(vstate, closed_thresh=0.074):
    return 1.0 if (vstate[6] + vstate[7]) < closed_thresh else -1.0


def track_segment(robot, states, i0, i1, grip=None, stride=1, lead=1, log=print):
    """Follow states[i0..i1] one step per (stride) video frame, targeting a frame `lead` ahead."""
    n = len(states)
    errs = []
    for i in range(i0, i1 + 1, stride):
        if robot.ended:
            break
        j = min(i + lead, n - 1, i1)
        tgt = states[j]
        g = grip if grip is not None else grip_cmd(tgt)
        row = row_to(robot.state, tgt, g)
        robot.step([row])
        pe, re = pose_err(robot.state, states[min(i + lead, n - 1, i1)])
        errs.append(pe)
    return errs


def converge(robot, tgt, grip, tol_p=0.002, tol_r=0.02, max_steps=30, log=print):
    for k in range(max_steps):
        pe, re = pose_err(robot.state, tgt)
        if pe < tol_p and re < tol_r:
            return True, k, pe, re
        if robot.ended:
            break
        robot.step([row_to(robot.state, tgt, grip)])
    pe, re = pose_err(robot.state, tgt)
    return (pe < tol_p and re < tol_r), max_steps, pe, re


def hold(robot, grip, n):
    tgt = list(robot.state)
    for k in range(n):
        if robot.ended:
            break
        robot.step([row_to(robot.state, tgt, grip)])


def backproject(depth_mm, K, pose, mask):
    """World points for masked pixels. OpenGL camera: looks along -Z, +Y up."""
    vs, us = np.nonzero(mask)
    d = depth_mm[vs, us].astype(float) * 0.001
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    # pixel centre convention: try u+0.5
    xc = (us - cx) / fx * d
    yc = -(vs - cy) / fy * d
    zc = -d
    P = np.stack([xc, yc, zc], 1)
    w, x, y, z = pose["quaternion_wxyz"]
    Rm = R.from_quat([x, y, z, w]).as_matrix()
    return P @ Rm.T + np.asarray(pose["position_m"])


def load_frame_arrays(frame):
    from PIL import Image
    files = {(f["camera"], f["kind"]): f["path"] for f in frame["files"]}
    out = {}
    for k, p in files.items():
        out[k] = np.array(Image.open(WS / p))
    return out


def peg_top_points(frame_arrays, frame_meta, max_depth_mm=90, finger_row=200):
    d = frame_arrays[("wrist", "depth")].astype(int)
    mask = (d > 0) & (d < max_depth_mm)
    mask[finger_row:, :] = False
    if mask.sum() < 200:
        return None
    med = np.median(d[mask])
    mask &= (np.abs(d - med) <= 2)
    return backproject(d, frame_meta["camera_intrinsics"]["wrist"]["matrix"], frame_meta["camera_poses"]["wrist"], mask)


def peg_line(P):
    """PCA line in xy of the peg top surface: centroid, unit axis, extent along axis, mean z."""
    c = P.mean(0)
    Q = P[:, :2] - c[:2]
    w, V = np.linalg.eigh(Q.T @ Q)
    a = V[:, -1]
    t = Q @ a
    n = np.array([-a[1], a[0]])
    s = Q @ n
    return dict(c=c, axis=a, tmin=float(t.min()), tmax=float(t.max()), smin=float(np.percentile(s, 2)), smax=float(np.percentile(s, 98)), z=float(np.median(P[:, 2])))


def peg_tip_frame(frame_arrays, frame_meta, ee_pos):
    """Return (tip_xy, unit axis pointing to tip, z_top, n_points) for the held peg, or None."""
    P = peg_top_points(frame_arrays, frame_meta)
    if P is None:
        return None
    L = peg_line(P)
    a = L["axis"]; c = L["c"]
    e1 = c[:2] + a * L["tmin"]; e2 = c[:2] + a * L["tmax"]
    ee = np.asarray(ee_pos[:2])
    if np.linalg.norm(e1 - ee) > np.linalg.norm(e2 - ee):
        tip, ax = e1, -a
    else:
        tip, ax = e2, a
    return dict(tip=tip, axis=ax, z=L["z"], n=len(P), width=L["smax"] - L["smin"], length=L["tmax"] - L["tmin"])


def pose_mat(state):
    T = np.eye(4)
    T[:3, :3] = R.from_rotvec(state[3:6]).as_matrix()
    T[:3, 3] = state[:3]
    return T


def mat_state(T, grip_vals):
    rv = R.from_matrix(T[:3, :3]).as_rotvec()
    return list(T[:3, 3]) + list(rv) + list(grip_vals)


def correction(live, video, ee_state):
    """Body-frame correction C so that EE_target' = EE_target @ C moves the live peg onto the video peg."""
    th = np.arctan2(video["axis"][1], video["axis"][0]) - np.arctan2(live["axis"][1], live["axis"][0])
    th = (th + np.pi) % (2 * np.pi) - np.pi
    D = np.eye(4)
    Rz = R.from_rotvec([0, 0, th]).as_matrix()
    D[:3, :3] = Rz
    p_live = np.array([live["tip"][0], live["tip"][1], live["z"]])
    p_vid = np.array([video["tip"][0], video["tip"][1], video["z"]])
    D[:3, 3] = p_vid - Rz @ p_live
    E = pose_mat(ee_state)
    return np.linalg.inv(E) @ D @ E, th, p_vid - p_live


def apply_corr(state, C):
    T = pose_mat(state) @ C
    return mat_state(T, state[6:8])


def peg_top_line(frame_arrays, frame_meta, ee_state, finger_row=200, top_row=56):
    """Fit the held peg's top surface as a world line x = x0 + a*(y - y_ref), z_top.
    Uses wrist depth pixels near the median depth of the closest large surface (the peg top)."""
    d = frame_arrays[("wrist", "depth")].astype(int)
    K = frame_meta["camera_intrinsics"]["wrist"]["matrix"]; pose = frame_meta["camera_poses"]["wrist"]
    P = backproject(d, K, pose, np.ones_like(d, bool)).reshape(256, 256, 3)
    ee = np.asarray(ee_state[:3])
    zt = P[..., 2] - ee[2]
    rows = np.arange(256)[:, None] * np.ones((1, 256))
    # top surface lies ~1.0-1.2 cm above the EE; restrict to a band and laterally near EE
    m = (d > 0) & (zt > 0.004) & (zt < 0.018) & (rows < finger_row) & (rows >= top_row)
    lat = np.abs(P[..., 0] - ee[0])
    m &= lat < 0.06
    Q = P[m]
    if len(Q) < 300:
        return None
    zmed = np.median(Q[:, 2])
    Q = Q[np.abs(Q[:, 2] - zmed) < 0.0025]
    if len(Q) < 300:
        return None
    # PCA in xy
    c = Q[:, :2].mean(0); X = Q[:, :2] - c
    w, V = np.linalg.eigh(X.T @ X); ax = V[:, -1]
    if ax[1] < 0:
        ax = -ax
    t = X @ ax; n = np.array([ax[1], -ax[0]]); s = X @ n
    # centre line: midpoint of lateral extents
    s_lo, s_hi = np.percentile(s, 1), np.percentile(s, 99)
    centre = c + n * (s_lo + s_hi) / 2
    return dict(centre=centre, axis=ax, width=float(s_hi - s_lo), z=float(np.median(Q[:, 2])),
                tmin=float(t.min()), tmax=float(t.max()), n=int(len(Q)), c=c)


def line_x_at(L, y):
    """x of the centre line at world y."""
    ax = L["axis"]; c = L["centre"]
    return float(c[0] + ax[0] / ax[1] * (y - c[1]))


def peg_pitch(frame_arrays, frame_meta, ee_state, finger_row=200, top_row=56):
    """Slope dz/dy (world) of the peg top surface and lateral roll dz/dx."""
    d = frame_arrays[("wrist", "depth")].astype(int)
    P = backproject(d, frame_meta["camera_intrinsics"]["wrist"]["matrix"], frame_meta["camera_poses"]["wrist"], np.ones_like(d, bool)).reshape(256, 256, 3)
    ee = np.asarray(ee_state[:3]); zt = P[..., 2] - ee[2]
    rows = np.arange(256)[:, None] * np.ones((1, 256))
    m = (d > 0) & (zt > 0.004) & (zt < 0.018) & (rows < finger_row) & (rows >= top_row) & (np.abs(P[..., 0] - ee[0]) < 0.06)
    Q = P[m]
    zmed = np.median(Q[:, 2]); Q = Q[np.abs(Q[:, 2] - zmed) < 0.0025]
    A = np.stack([Q[:, 0], Q[:, 1], np.ones(len(Q))], 1)
    coef = np.linalg.lstsq(A, Q[:, 2], rcond=None)[0]
    return coef  # z = a x + b y + c
