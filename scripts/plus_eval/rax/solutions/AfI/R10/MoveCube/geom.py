import numpy as np
from scipy.spatial.transform import Rotation as Rot

def cam_params(meta, cam):
    K = np.array(meta["camera_intrinsics"][cam]["matrix"], dtype=float)
    pose = meta["camera_poses"][cam]
    p = np.array(pose["position_m"], dtype=float)
    w, x, y, z = pose["quaternion_wxyz"]
    R = Rot.from_quat([x, y, z, w]).as_matrix()
    return K, R, p

def deproject(meta, cam, depth, u, v, r=1):
    """Pixel (u=col, v=row) -> world point using depth map (meters)."""
    K, R, p = cam_params(meta, cam)
    H, W = depth.shape
    ui, vi = int(round(u)), int(round(v))
    win = depth[max(0, vi - r):vi + r + 1, max(0, ui - r):ui + r + 1]
    win = win[win > 0]
    if win.size == 0:
        return None
    d = float(np.median(win))
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    pc = np.array([(u - cx) * d / fx, -(v - cy) * d / fy, -d])
    return R @ pc + p

def deproject_all(meta, cam, depth):
    """World xyz for every pixel (H,W,3); invalid depth -> nan."""
    K, R, p = cam_params(meta, cam)
    H, W = depth.shape
    vv, uu = np.mgrid[0:H, 0:W].astype(float)
    d = depth.astype(float)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    pc = np.stack([(uu - cx) * d / fx, -(vv - cy) * d / fy, -d], -1)
    pw = pc @ R.T + p
    pw[d <= 0] = np.nan
    return pw

def project(meta, cam, X):
    K, R, p = cam_params(meta, cam)
    pc = R.T @ (np.asarray(X) - p)
    d = -pc[2]
    u = K[0, 0] * pc[0] / d + K[0, 2]
    v = -K[1, 1] * pc[1] / d + K[1, 2]
    return u, v, d
