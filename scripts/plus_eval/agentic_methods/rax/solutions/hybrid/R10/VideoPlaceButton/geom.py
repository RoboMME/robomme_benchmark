"""Front-camera pixel <-> table-plane geometry (ray/plane intersection, no depth needed)."""
import numpy as np
from scipy.spatial.transform import Rotation as Rr

class PlaneCam:
    def __init__(self, K, pose, z=0.009):
        self.K = np.array(K['matrix'], float)
        q = pose['quaternion_wxyz']
        self.R = Rr.from_quat([q[1], q[2], q[3], q[0]])
        self.p = np.array(pose['position_m'], float)
        self.z = z

    def to_world(self, uv):
        uv = np.atleast_2d(np.asarray(uv, float))
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        d = np.stack([(uv[:, 0] - cx) / fx, -(uv[:, 1] - cy) / fy, -np.ones(len(uv))], 1)
        dw = self.R.apply(d)
        s = (self.z - self.p[2]) / dw[:, 2]
        return self.p + dw * s[:, None]
