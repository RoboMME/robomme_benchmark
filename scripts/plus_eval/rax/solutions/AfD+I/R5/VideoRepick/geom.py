import numpy as np

def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]])

def depth_to_world(depth_m, K, pose):
    """depth_m HxW meters -> HxWx3 world points (OpenGL camera convention)."""
    H, W = depth_m.shape
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    u, v = np.meshgrid(np.arange(W) + 0.0, np.arange(H) + 0.0)
    d = depth_m
    cam = np.stack([(u - cx) * d / fx, -(v - cy) * d / fy, -d], -1)
    R = quat_to_R(pose["quaternion_wxyz"])
    return cam @ R.T + np.array(pose["position_m"])

def world_to_pixel(p, K, pose):
    R = quat_to_R(pose["quaternion_wxyz"])
    c = R.T @ (np.asarray(p, float) - np.array(pose["position_m"]))
    d = -c[2]
    u = c[0] * K[0][0] / d + K[0][2]
    v = -c[1] * K[1][1] / d + K[1][2]
    return u, v, d

def label(mask):
    """Connected components (4-neighbour); scipy if available, BFS fallback."""
    try:
        from scipy import ndimage
        return ndimage.label(mask)
    except Exception:
        H, W = mask.shape
        lab = np.zeros((H, W), np.int32); n = 0
        ys, xs = np.nonzero(mask)
        for y0, x0 in zip(ys, xs):
            if lab[y0, x0]:
                continue
            n += 1; st = [(y0, x0)]; lab[y0, x0] = n
            while st:
                y, x = st.pop()
                for yy, xx in ((y+1, x), (y-1, x), (y, x+1), (y, x-1)):
                    if 0 <= yy < H and 0 <= xx < W and mask[yy, xx] and not lab[yy, xx]:
                        lab[yy, xx] = n; st.append((yy, xx))
        return lab, n
