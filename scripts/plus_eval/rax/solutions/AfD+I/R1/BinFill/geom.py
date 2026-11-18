import math
import numpy as np


def rotvec_to_R(rv):
    rv = np.asarray(rv, float)
    th = float(np.linalg.norm(rv))
    if th < 1e-9:
        return np.eye(3)
    k = rv / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def quat_wxyz_to_R(q):
    w, x, y, z = [float(a) for a in q]
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def wrap_pi(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def wrap_quarter(a):
    """Wrap to (-pi/4, pi/4]."""
    q = math.pi / 2
    return (a + q / 2) % q - q / 2


def yaw_of(v):
    return math.atan2(v[1], v[0])
