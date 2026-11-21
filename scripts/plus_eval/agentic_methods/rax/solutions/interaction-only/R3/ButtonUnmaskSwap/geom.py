import numpy as np

def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

def backproject(u, v, d, K, pose):
    """u,v pixel, d depth (m) along optical axis; pose: dict position_m, quaternion_wxyz (OpenGL)."""
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    p = np.array([(u - cx) / fx * d, -(v - cy) / fy * d, -d])
    R = quat_to_R(pose["quaternion_wxyz"])
    return R @ p + np.array(pose["position_m"])

def project(Pw, K, pose):
    R = quat_to_R(pose["quaternion_wxyz"])
    p = R.T @ (np.asarray(Pw) - np.array(pose["position_m"]))
    d = -p[2]
    u = K[0][0] * p[0] / d + K[0][2]
    v = -K[1][1] * p[1] / d + K[1][2]
    return u, v, d
