"""Convention-free camera geometry.

A camera with linear depth d along its optical axis back-projects pixel (u, v) to
world = A @ [u*d, v*d, d] + b, which is linear in (u*d, v*d, d, 1).  We fit the 3x4
matrix M=[A|b] from pixel_to_position samples (authoritative tool) and our own depth
reading; a fixed pixel-offset or depth-scale convention is absorbed by the fit.
For a moved camera rigidly attached to the end effector, M_t = T_rel * M_0 where
T_rel = T_ee(t) * T_ee(0)^-1.
"""
import numpy as np

def axis_angle_to_R(v):
    v = np.asarray(v, float)
    th = np.linalg.norm(v)
    if th < 1e-12:
        return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K

def R_to_axis_angle(R):
    R = np.asarray(R, float)
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    th = np.arccos(c)
    if th < 1e-9:
        return np.zeros(3)
    if th > np.pi - 1e-4:
        # near pi: axis from the symmetric part
        M = (R + np.eye(3)) / 2
        k = np.sqrt(np.clip(np.diag(M), 0, None))
        i = int(np.argmax(k))
        k[i] = np.sqrt(M[i, i])
        for j in range(3):
            if j != i:
                k[j] = M[i, j] / k[i]
        k /= np.linalg.norm(k)
        return k * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(th))
    return w * th

def quat_wxyz_to_R(q):
    w, x, y, z = [float(a) for a in q]
    n = np.sqrt(w*w + x*x + y*y + z*z)
    w, x, y, z = w/n, x/n, y/n, z/n
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

def fit_M(uv, d, world):
    """uv: Nx2 (u=column, v=row), d: N depths (any consistent scale), world: Nx3."""
    uv = np.asarray(uv, float); d = np.asarray(d, float); world = np.asarray(world, float)
    X = np.stack([uv[:, 0] * d, uv[:, 1] * d, d, np.ones_like(d)], 1)
    M, *_ = np.linalg.lstsq(X, world, rcond=None)
    M = M.T  # 3x4
    pred = X @ M.T
    res = np.linalg.norm(pred - world, axis=1)
    return M, res

def M_from_intrinsics_pose(K, pos, quat_wxyz, convention='opengl'):
    """Back-projection matrix from intrinsics and a camera-to-world pose.
    OpenGL: camera looks along -Z, +Y up, image row 0 at top."""
    K = np.asarray(K, float).reshape(3, 3)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    R = quat_wxyz_to_R(quat_wxyz)
    t = np.asarray(pos, float)
    if str(convention).lower().startswith('opengl') or 'gl' in str(convention).lower():
        # x_c = (u-cx) d/fx ; y_c = -(v-cy) d/fy ; z_c = -d
        C = np.array([[1/fx, 0, -cx/fx], [0, -1/fy, cy/fy], [0, 0, -1]])
    else:  # opencv: x right, y down, z forward
        C = np.array([[1/fx, 0, -cx/fx], [0, 1/fy, -cy/fy], [0, 0, 1]])
    A = R @ C
    return np.concatenate([A, t[:, None]], 1)

def backproject(M, u, v, d):
    u = np.asarray(u, float); v = np.asarray(v, float); d = np.asarray(d, float)
    X = np.stack([u * d, v * d, d], -1)
    return X @ M[:, :3].T + M[:, 3]

def project(M, P):
    """Inverse of backproject: world points Nx3 -> (u, v, d)."""
    A = M[:, :3]; b = M[:, 3]
    Ai = np.linalg.inv(A)
    q = (np.asarray(P, float) - b) @ Ai.T
    d = q[..., 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        return q[..., 0] / d, q[..., 1] / d, d

def ee_T(state):
    T = np.eye(4)
    T[:3, :3] = axis_angle_to_R(state[3:6])
    T[:3, 3] = state[0:3]
    return T

def moved_M(M0, state0, state_t):
    """Back-projection for an EE-mounted camera after the EE moved from state0 to state_t."""
    Trel = ee_T(state_t) @ np.linalg.inv(ee_T(state0))
    A = Trel[:3, :3] @ M0[:, :3]
    b = Trel[:3, :3] @ M0[:, 3] + Trel[:3, 3]
    return np.concatenate([A, b[:, None]], 1)

def table_depth_along(M, u, v, z0):
    """Depth d at which the ray of pixel (u,v) meets plane z=z0."""
    A = M[:, :3]; b = M[:, 3]
    r = np.stack([np.asarray(u, float), np.asarray(v, float), np.ones_like(np.asarray(u, float))], -1) @ A.T
    with np.errstate(divide='ignore', invalid='ignore'):
        return (z0 - b[2]) / r[..., 2]
