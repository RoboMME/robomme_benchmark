import numpy as np

def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

def project(P, K, pose):
    """world points (N,3) -> pixel (u,v) using OpenGL camera convention."""
    R = quat_to_R(pose['quaternion_wxyz']); c = np.array(pose['position_m'])
    pc = (np.asarray(P) - c) @ R  # R^T (p-c)
    d = -pc[:, 2]
    u = K[0][0]*pc[:, 0]/d + K[0][2]
    v = -K[1][1]*pc[:, 1]/d + K[1][2]
    return np.c_[u, v], d

def backproject(u, v, d, K, pose):
    R = quat_to_R(pose['quaternion_wxyz']); c = np.array(pose['position_m'])
    pc = np.stack([(u-K[0][2])*d/K[0][0], -(v-K[1][2])*d/K[1][1], -d], -1)
    return pc @ R.T + c
