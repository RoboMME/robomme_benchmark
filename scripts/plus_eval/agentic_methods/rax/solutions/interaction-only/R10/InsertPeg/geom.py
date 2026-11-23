import numpy as np
from scipy.spatial.transform import Rotation as R

def backproject(depth_mm, K, pose):
    """depth uint16 mm (H,W) -> world points (H,W,3), valid mask."""
    d = depth_mm.astype(float) * 0.001
    H, W = d.shape
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    u, v = np.meshgrid(np.arange(W) + 0.0, np.arange(H) + 0.0)
    pc = np.stack([(u - cx) * d / fx, -(v - cy) * d / fy, -d], -1)
    q = pose["quaternion_wxyz"]
    rot = R.from_quat([q[1], q[2], q[3], q[0]])
    pw = rot.apply(pc.reshape(-1, 3)).reshape(H, W, 3) + np.asarray(pose["position_m"])
    return pw, d > 0
