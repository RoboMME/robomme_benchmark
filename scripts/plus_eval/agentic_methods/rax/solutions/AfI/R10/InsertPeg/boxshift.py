"""Estimate the box displacement between the conditioning video and the live scene by
registering the wrist depth image at (nearly) the same end-effector pose."""
import numpy as np


def quat_to_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def backproject(d_mm, cam_pose, K):
    R = quat_to_mat(cam_pose["quaternion_wxyz"])
    t = np.array(cam_pose["position_m"])
    v, u = np.mgrid[0:d_mm.shape[0], 0:d_mm.shape[1]]
    z = d_mm * 0.001
    pc = np.stack([(u - K[0][2]) * z / K[0][0], -(v - K[1][2]) * z / K[1][1], -z], -1)
    return pc @ R.T + t


def project(P, cam_pose, K):
    R = quat_to_mat(cam_pose["quaternion_wxyz"])
    t = np.array(cam_pose["position_m"])
    pc = (P - t) @ R
    z = -pc[..., 2]
    u = pc[..., 0] / np.maximum(z, 1e-6) * K[0][0] + K[0][2]
    v = -pc[..., 1] / np.maximum(z, 1e-6) * K[1][1] + K[1][2]
    return u, v, z


def estimate_shift(demo_d, demo_cam, live_d, live_cam, K, hand_mask, dmin=40, dmax=118,
                   grid=0.004, step=0.0005):
    """Return (shift_world[3], cost0, cost_best, n_pts). Box points from the demo depth
    (excluding the hand/peg mask and far floor) are translated by candidate world shifts,
    projected into the live camera and compared with live depth."""
    msk = (demo_d > dmin) & (demo_d < dmax) & (~hand_mask)
    P = backproject(demo_d.astype(float), demo_cam, K)[msk]
    if len(P) < 50:
        return None
    live = live_d.astype(float)
    H, W = live.shape

    def cost(sh):
        u, v, z = project(P + sh, live_cam, K)
        ui = np.round(u).astype(int)
        vi = np.round(v).astype(int)
        ok = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        lz = live[vi[ok], ui[ok]] * 0.001
        r = np.abs(lz - z[ok])
        good = (lz > dmin * 0.001) & ~hand_mask[vi[ok], ui[ok]]
        r = np.minimum(r[good], 0.01)
        return float(r.mean()) if len(r) else 1.0

    c0 = cost(np.zeros(3))
    best = (c0, np.zeros(3))
    # coarse-to-fine search over a 3-D grid
    centre = np.zeros(3)
    for g, st in ((grid, 0.001), (0.0012, 0.0003), (0.0004, 0.0001)):
        rng = np.arange(-g, g + 1e-9, st)
        for dx in rng:
            for dy in rng:
                for dz in rng:
                    sh = centre + np.array([dx, dy, dz])
                    c = cost(sh)
                    if c < best[0]:
                        best = (c, sh)
        centre = best[1].copy()
    return best[1], c0, best[0], int(len(P))
