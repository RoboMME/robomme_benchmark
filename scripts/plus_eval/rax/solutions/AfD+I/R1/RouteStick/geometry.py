"""Rotation helpers and an independent pinhole back-projection (OpenGL camera convention)."""
import numpy as np


def rotvec_to_mat(r):
    r = np.asarray(r, float)
    th = np.linalg.norm(r)
    if th < 1e-9:
        return np.eye(3)
    k = r / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def mat_to_rotvec(R):
    R = np.asarray(R, float)
    c = (np.trace(R) - 1) / 2
    c = min(1.0, max(-1.0, c))
    th = np.arccos(c)
    if th < 1e-7:
        return np.zeros(3)
    if np.pi - th < 1e-4:
        # near pi: axis from the largest diagonal
        M = (R + np.eye(3)) / 2
        i = int(np.argmax(np.diag(M)))
        k = M[:, i] / np.sqrt(max(M[i, i], 1e-12))
        return k / np.linalg.norm(k) * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(th))
    return w * th


def rot_error_world(r_ref, r_cur):
    """World-frame rotation vector d with exp(d) @ R_cur = R_ref."""
    return mat_to_rotvec(rotvec_to_mat(r_ref) @ rotvec_to_mat(r_cur).T)


def quat_wxyz_to_mat(q):
    w, x, y, z = [float(t) for t in q]
    n = np.sqrt(w * w + x * x + y * y + z * z) + 1e-12
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def backproject(u, v, depth_m, K, cam_pos, cam_quat_wxyz, convention='opengl'):
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    xc = (u - cx) / fx * depth_m
    yc = (v - cy) / fy * depth_m
    if convention == 'opengl':
        pc = np.array([xc, -yc, -depth_m])
    else:  # opencv
        pc = np.array([xc, yc, depth_m])
    R = quat_wxyz_to_mat(cam_quat_wxyz)
    return R @ pc + np.asarray(cam_pos, float)
