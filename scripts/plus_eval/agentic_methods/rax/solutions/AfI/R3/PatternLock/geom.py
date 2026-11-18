import numpy as np

def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]])

def backproject(depth_m, K, pose):
    """depth_m HxW metres (0 invalid); returns HxWx3 world points (OpenGL cam: -Z forward, +Y up)."""
    H, W = depth_m.shape
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    u, v = np.meshgrid(np.arange(W) + 0.0, np.arange(H) + 0.0)
    d = depth_m
    xc = (u - cx) / fx * d
    yc = -(v - cy) / fy * d
    zc = -d
    P = np.stack([xc, yc, zc], -1)
    R = quat_to_R(pose["quaternion_wxyz"])
    t = np.asarray(pose["position_m"])
    return P @ R.T + t


def project(p, K, pose):
    R = quat_to_R(pose["quaternion_wxyz"])
    t = np.asarray(pose["position_m"])
    c = R.T @ (np.asarray(p) - t)
    d = -c[2]
    return (K[0][2] + K[0][0] * c[0] / d, K[1][2] - K[1][1] * c[1] / d)


def rotvec_to_R(r):
    r = np.asarray(r, dtype=float)
    th = float(np.linalg.norm(r))
    if th < 1e-12:
        return np.eye(3)
    k = r / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * Kx @ Kx


def R_to_rotvec(R):
    c = (np.trace(R) - 1) / 2
    c = min(1.0, max(-1.0, c))
    th = float(np.arccos(c))
    if th < 1e-9:
        return np.zeros(3)
    if th > np.pi - 1e-4:
        # near pi: use symmetric part
        M = (R + np.eye(3)) / 2
        k = np.sqrt(np.clip(np.diag(M), 0, None))
        i = int(np.argmax(k))
        k = M[:, i] / np.sqrt(M[i, i])
        return k * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(th))
    return w * th


def label(mask):
    """4-connected component labelling without scipy. Returns (labels, n)."""
    H, W = mask.shape
    lab = np.zeros((H, W), dtype=np.int32)
    n = 0
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys, xs):
        if lab[y0, x0]:
            continue
        n += 1
        stack = [(y0, x0)]
        lab[y0, x0] = n
        while stack:
            y, x = stack.pop()
            for yy, xx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                if 0 <= yy < H and 0 <= xx < W and mask[yy, xx] and not lab[yy, xx]:
                    lab[yy, xx] = n
                    stack.append((yy, xx))
    return lab, n
