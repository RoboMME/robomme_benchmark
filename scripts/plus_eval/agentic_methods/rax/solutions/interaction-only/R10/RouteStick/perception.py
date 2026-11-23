"""Disc (target) detection in the front RGB-D view -> world xy."""
import json
from pathlib import Path
import numpy as np
from PIL import Image

WS = Path("/workspace")


def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def backproject(depth_m, K, pose):
    H, W = depth_m.shape
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    d = depth_m
    pc = np.stack([(u - cx) * d / fx, -(v - cy) * d / fy, -d], -1)
    R = quat_to_R(pose["quaternion_wxyz"])
    pw = pc @ R.T + np.array(pose["position_m"])
    return pw


def load_frame(frame, cam="front"):
    files = {(f["camera"], f["kind"]): f["path"] for f in frame["files"]}
    rgb = np.asarray(Image.open(WS / files[(cam, "rgb")]).convert("RGB"), dtype=np.float32)
    dep = np.asarray(Image.open(WS / files[(cam, "depth")]), dtype=np.float32) * 0.001
    K = frame["camera_intrinsics"][cam]["matrix"]
    pose = frame["camera_poses"][cam]
    return rgb, dep, K, pose


def detect_discs(frame, cam="front"):
    rgb, dep, K, pose = load_frame(frame, cam)
    P = backproject(dep, K, pose)
    valid = dep > 0
    mx = rgb.max(-1)
    mn = rgb.min(-1)
    sat = (mx - mn) / np.maximum(mx, 1)
    # table plane height: median z of saturated (orange wood) pixels
    table = valid & (sat > 0.35)
    zt = float(np.median(P[..., 2][table])) if table.sum() > 100 else 0.0
    dz = P[..., 2] - zt
    cand = valid & (sat < 0.15) & (mx > 120) & (dz > 0.006) & (dz < 0.03) & (P[..., 0] > -0.45)
    pts = P[cand][:, :2]
    # greedy clustering in xy
    clusters = []
    if len(pts):
        rem = pts.copy()
        while len(rem):
            c = rem[0]
            for _ in range(3):
                m = np.linalg.norm(rem - c, axis=1) < 0.025
                c = rem[m].mean(0)
            m = np.linalg.norm(rem - c, axis=1) < 0.025
            clusters.append((c, int(m.sum())))
            rem = rem[~m]
    discs = [c for c, n in clusters if n >= 6]
    return np.array(discs), zt, clusters


def fit_rigid(A, B):
    """Least-squares 2D rigid transform with B ~ R A + t."""
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    t = cb - R @ ca
    return R, t


def match_sets(A, B):
    """Match two disc sets: try ordering along principal axis, fall back to NN.
    Returns (Am, Bm) matched arrays."""
    if len(A) == len(B) and len(A) >= 2:
        allp = np.vstack([A, B])
        c = allp.mean(0)
        _, _, Vt = np.linalg.svd(allp - c)
        ax = Vt[0]
        ia = np.argsort((A - c) @ ax)
        ib = np.argsort((B - c) @ ax)
        return A[ia], B[ib]
    # nearest-neighbour mutual matches after centroid alignment
    Ac = A - A.mean(0) + B.mean(0)
    pairs = []
    for i, a in enumerate(Ac):
        j = int(np.argmin(np.linalg.norm(B - a, axis=1)))
        i2 = int(np.argmin(np.linalg.norm(Ac - B[j], axis=1)))
        if i2 == i:
            pairs.append((i, j))
    if not pairs:
        return A[:0], B[:0]
    return A[[p[0] for p in pairs]], B[[p[1] for p in pairs]]


def order_discs(D, axis):
    s = D @ axis
    return D[np.argsort(s)]


class RowWarp:
    """Map xy points from the video's disc row to the execution's disc row.

    Each point is written as (interval i, fraction f along the row axis, signed normal
    offset n in metres) relative to consecutive video discs and rebuilt on the
    corresponding execution discs.  Requires equal disc counts (>=2)."""

    def __init__(self, Dv, De):
        allv = Dv - Dv.mean(0)
        _, _, Vt = np.linalg.svd(allv)
        ax = Vt[0]
        if ax[1] < 0:
            ax = -ax
        Dv = order_discs(Dv, ax)
        _, _, Vt2 = np.linalg.svd(De - De.mean(0))
        ax2 = Vt2[0]
        if ax2 @ ax < 0:
            ax2 = -ax2
        De = order_discs(De, ax2)
        self.Dv, self.De, self.ax = Dv, De, ax

    @staticmethod
    def _seg(D, i):
        a, b = D[i], D[i + 1]
        L = np.linalg.norm(b - a)
        u = (b - a) / L
        nrm = np.array([-u[1], u[0]])
        return a, L, u, nrm

    def __call__(self, p):
        Dv, De = self.Dv, self.De
        m = len(Dv)
        best = None
        for i in range(m - 1):
            a, L, u, nrm = self._seg(Dv, i)
            f = (p - a) @ u / L
            if (0 <= f <= 1) or (i == 0 and f < 0) or (i == m - 2 and f > 1):
                cost = 0
            else:
                cost = min(abs(f), abs(f - 1))
            if best is None or cost < best[0]:
                best = (cost, i, f, (p - a) @ nrm)
        _, i, f, n = best
        a2, L2, u2, n2 = self._seg(De, i)
        return a2 + f * L2 * u2 + n * n2


def warp_path(W, warp):
    out = W.copy()
    for k in range(len(W)):
        out[k, :2] = warp(W[k, :2])
    return out
