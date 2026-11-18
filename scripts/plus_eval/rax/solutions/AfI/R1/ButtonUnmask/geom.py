"""Geometry helpers: rotations, camera back-projection conventions, height maps."""
import math
import numpy as np


def quat_wxyz_to_R(q):
    w, x, y, z = [float(a) for a in q]
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def axang_to_R(v):
    v = np.asarray(v, dtype=float)
    th = float(np.linalg.norm(v))
    if th < 1e-9:
        return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)


def R_to_axang(R):
    R = np.asarray(R, dtype=float)
    c = (np.trace(R) - 1) / 2
    c = min(1.0, max(-1.0, c))
    th = math.acos(c)
    if th < 1e-6:
        return np.zeros(3)
    if th > math.pi - 1e-3:
        # near pi: use the symmetric part
        M = (R + np.eye(3)) / 2
        k = np.sqrt(np.clip(np.diag(M), 0, None))
        i = int(np.argmax(k))
        if k[i] < 1e-9:
            return np.zeros(3)
        k = M[:, i] / k[i]
        k = k / (np.linalg.norm(k) or 1)
        return k * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return w * th


def Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def wrap_pi(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


# ---------------------------------------------------------------- cameras
CONVENTIONS = []
for _flip in ("", "u", "v", "uv"):
    for _gl in ("gl", "cv"):
        for _off in (0.0, 0.5):
            for _dk in ("z", "range"):
                for _inv in (False, True):
                    CONVENTIONS.append((_gl, _off, _dk, _inv, _flip))


class Camera:
    """Intrinsics K (3x3) and pose T (4x4, as supplied)."""

    def __init__(self, K, T, conv=("gl", 0.0, "z", False, ""), size=(256, 256)):
        self.size = size
        self.K = np.asarray(K, dtype=float)
        self.T = np.asarray(T, dtype=float)
        self.conv = conv

    def cam2world(self):
        return np.linalg.inv(self.T) if self.conv[3] else self.T

    def backproject(self, us, vs, d, conv=None):
        """us = column, vs = row (float arrays), d = depth (m). Returns Nx3 world."""
        conv = conv or self.conv
        gl, off, dk, inv = conv[:4]
        flip = conv[4] if len(conv) > 4 else ""
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        us = np.asarray(us, dtype=float)
        vs = np.asarray(vs, dtype=float)
        if "u" in flip:
            us = self.size[1] - 1 - us
        if "v" in flip:
            vs = self.size[0] - 1 - vs
        us = us + off
        vs = vs + off
        d = np.asarray(d, dtype=float)
        a = (us - cx) / fx
        b = (vs - cy) / fy
        if gl == "gl":
            dirs = np.stack([a, -b, -np.ones_like(a)], -1)
        else:
            dirs = np.stack([a, b, np.ones_like(a)], -1)
        if dk == "range":
            dirs = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)
        pc = dirs * d[..., None]
        T = np.linalg.inv(self.T) if inv else self.T
        return pc @ T[:3, :3].T + T[:3, 3]

    def project(self, P, conv=None):
        """World points -> (u, v) pixel coords (approximate inverse of backproject)."""
        conv = conv or self.conv
        gl, off, dk, inv = conv[:4]
        flip = conv[4] if len(conv) > 4 else ""
        T = np.linalg.inv(self.T) if inv else self.T
        Ti = np.linalg.inv(T)
        pc = np.asarray(P, float) @ Ti[:3, :3].T + Ti[:3, 3]
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        if gl == "gl":
            z = -pc[:, 2]
            u = pc[:, 0] / z * fx + cx - off
            v = -pc[:, 1] / z * fy + cy - off
        else:
            z = pc[:, 2]
            u = pc[:, 0] / z * fx + cx - off
            v = pc[:, 1] / z * fy + cy - off
        if "u" in flip:
            u = self.size[1] - 1 - u
        if "v" in flip:
            v = self.size[0] - 1 - v
        return np.stack([u, v], 1)

    def position(self):
        return self.cam2world()[:3, 3]


def pose_to_T(position, quat_wxyz):
    T = np.eye(4)
    T[:3, :3] = quat_wxyz_to_R(quat_wxyz)
    T[:3, 3] = np.asarray(position, dtype=float)
    return T


# ---------------------------------------------------------------- 2-D helpers
def min_area_rect(xy):
    """Return (center(2), (L, W) with L>=W, angle_of_L_axis rad) of a 2-D point set."""
    xy = np.asarray(xy, dtype=float)
    if len(xy) < 3:
        c = xy.mean(0) if len(xy) else np.zeros(2)
        return c, (0.0, 0.0), 0.0
    try:
        import cv2
        (cx, cy), (w, h), ang = cv2.minAreaRect(xy.astype(np.float32))
        a = math.radians(ang)
        if w >= h:
            L, W, axis = w, h, a
        else:
            L, W, axis = h, w, a + math.pi / 2
        return np.array([cx, cy]), (float(L), float(W)), wrap_pi(axis)
    except Exception:
        c = xy.mean(0)
        cov = np.cov((xy - c).T)
        ev, evec = np.linalg.eigh(cov)
        major = evec[:, 1]
        ang = math.atan2(major[1], major[0])
        rot = np.array([[math.cos(ang), math.sin(ang)], [-math.sin(ang), math.cos(ang)]])
        p = (xy - c) @ rot.T
        mn, mx = p.min(0), p.max(0)
        cc = (mn + mx) / 2
        center = c + cc @ rot
        return center, (float(mx[0] - mn[0]), float(mx[1] - mn[1])), wrap_pi(ang)


def label(mask, connectivity=8):
    try:
        from scipy import ndimage
        st = np.ones((3, 3)) if connectivity == 8 else None
        lab, n = ndimage.label(mask, structure=st)
        return lab, n
    except Exception:
        pass
    try:
        import cv2
        n, lab = cv2.connectedComponents(mask.astype(np.uint8), connectivity=connectivity)
        return lab, n - 1
    except Exception:
        pass
    # pure-python flood fill fallback
    H, W = mask.shape
    lab = np.zeros((H, W), dtype=np.int32)
    n = 0
    for i in range(H):
        for j in range(W):
            if mask[i, j] and lab[i, j] == 0:
                n += 1
                stack = [(i, j)]
                lab[i, j] = n
                while stack:
                    a, b = stack.pop()
                    for da in (-1, 0, 1):
                        for db in (-1, 0, 1):
                            x, y = a + da, b + db
                            if 0 <= x < H and 0 <= y < W and mask[x, y] and lab[x, y] == 0:
                                lab[x, y] = n
                                stack.append((x, y))
    return lab, n


def rgb_to_hsv(rgb):
    """rgb uint8 HxWx3 -> h (deg), s, v in [0,1]."""
    f = rgb.astype(np.float32) / 255.0
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    mx = f.max(-1)
    mn = f.min(-1)
    d = mx - mn
    s = np.where(mx > 1e-6, d / np.maximum(mx, 1e-6), 0)
    h = np.zeros_like(mx)
    dd = np.where(d > 1e-6, d, 1)
    hr = ((g - b) / dd) % 6
    hg = (b - r) / dd + 2
    hb = (r - g) / dd + 4
    h = np.where(mx == r, hr, np.where(mx == g, hg, hb)) * 60.0
    h = np.where(d > 1e-6, h, 0)
    return h, s, mx
