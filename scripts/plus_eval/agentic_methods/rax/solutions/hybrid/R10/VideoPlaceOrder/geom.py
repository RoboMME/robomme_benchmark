import numpy as np, math
def quat_to_mat(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
class Cam:
    def __init__(self, intr, pose):
        K = np.array(intr['matrix'], float)
        self.fx, self.fy, self.cx, self.cy = K[0,0], K[1,1], K[0,2], K[1,2]
        self.R = quat_to_mat(pose['quaternion_wxyz']); self.t = np.array(pose['position_m'], float)
    def ray(self, u, v):
        d = np.array([(u-self.cx)/self.fx, -(v-self.cy)/self.fy, -1.0])
        return self.R @ d
    def deproject(self, u, v, depth):
        d = np.array([(u-self.cx)*depth/self.fx, -(v-self.cy)*depth/self.fy, -depth])
        return self.R @ d + self.t
    def to_plane(self, u, v, z):
        r = self.ray(u, v); s = (z - self.t[2]) / r[2]
        return self.t + s * r
    def project(self, p):
        c = self.R.T @ (np.asarray(p, float) - self.t)
        return np.array([self.cx + self.fx*c[0]/(-c[2]), self.cy - self.fy*c[1]/(-c[2])])
