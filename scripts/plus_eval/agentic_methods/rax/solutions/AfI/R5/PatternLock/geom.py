import numpy as np

def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])

def backproject(u, v, d, K, pose):
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    pc = np.array([(u-cx)*d/fx, -(v-cy)*d/fy, -d])
    R = quat_to_R(pose['quaternion_wxyz'])
    return R @ pc + np.array(pose['position_m'])

def project(p, K, pose):
    R = quat_to_R(pose['quaternion_wxyz'])
    pc = R.T @ (np.asarray(p) - np.array(pose['position_m']))
    d = -pc[2]
    u = K[0][0]*pc[0]/d + K[0][2]
    v = -K[1][1]*pc[1]/d + K[1][2]
    return u, v, d


def button_points(rgb, dep_m, K, pose, hmin=0.008, hmax=0.04):
    """Raised button tops: back-project all pixels, keep those 8-40 mm above table, label blobs."""
    from scipy import ndimage
    v, u = np.mgrid[0:dep_m.shape[0], 0:dep_m.shape[1]]
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    d = dep_m
    pc = np.stack([(u - cx) * d / fx, -(v - cy) * d / fy, -d], -1).reshape(-1, 3)
    R = quat_to_R(pose['quaternion_wxyz'])
    W = (pc @ R.T + np.array(pose['position_m'])).reshape(d.shape + (3,))
    mask = (d > 0) & (W[..., 2] > hmin) & (W[..., 2] < hmax)
    lab, n = ndimage.label(mask)
    pts = []
    for i in range(1, n + 1):
        sel = lab == i
        if sel.sum() < 3:
            continue
        pts.append(W[sel].mean(0))
    return np.array(pts)
