import numpy as np

def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])

class Cam:
    def __init__(self, intr, pose):
        K = np.array(intr["matrix"], float)
        self.fx, self.fy, self.cx, self.cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        self.R = quat_to_R(pose["quaternion_wxyz"])
        self.t = np.array(pose["position_m"], float)

    def backproject(self, u, v, d):
        """u,v pixel arrays (col,row), d depth (m) -> world Nx3"""
        u = np.asarray(u, float); v = np.asarray(v, float); d = np.asarray(d, float)
        pc = np.stack([(u - self.cx) * d / self.fx, -(v - self.cy) * d / self.fy, -d], -1)
        return pc @ self.R.T + self.t

    def project(self, P):
        P = np.atleast_2d(P)
        pc = (P - self.t) @ self.R
        d = -pc[:, 2]
        u = pc[:, 0] / d * self.fx + self.cx
        v = -pc[:, 1] / d * self.fy + self.cy
        return np.stack([u, v, d], -1)
