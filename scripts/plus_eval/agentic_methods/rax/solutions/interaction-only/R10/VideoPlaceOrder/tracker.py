"""Track visually identical targets through the conditioning video (front camera)."""
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation as R
from pathlib import Path

WS = Path("/workspace")

def target_mask(im):
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    return (r > 120) & (b > 150) & (g < 130) & (b - g > 60)

def detect(im, min_px=12, dilate=3):
    m = target_mask(im)
    lab, n = ndimage.label(ndimage.binary_dilation(m, iterations=dilate))
    out = []
    for j in range(1, n + 1):
        sel = m & (lab == j)
        c = int(sel.sum())
        if c >= min_px:
            vs, us = np.nonzero(sel)
            out.append((float(us.mean()), float(vs.mean()), c))
    return out

def project(xyz, K, pose):
    w, x, y, z = pose["quaternion_wxyz"]
    rot = R.from_quat([x, y, z, w])
    pc = rot.inv().apply(np.array(xyz) - np.array(pose["position_m"]))
    d = -pc[2]
    u = K[0][2] + K[0][0] * pc[0] / d
    v = K[1][2] - K[1][1] * pc[1] / d
    return np.array([u, v])

def track(frames, log=print, gate=32.0):
    """frames: manifest frames sorted by frame_index (video + execution).
    Returns dict frame_index -> array (N,2) of track pixel positions, and N."""
    dets = []
    for f in frames:
        p = next(x["path"] for x in f["files"] if x["camera"] == "front" and x["kind"] == "rgb")
        im = np.array(Image.open(WS / p).convert("RGB")).astype(int)
        dets.append(detect(im))
    counts = [len(d) for d in dets]
    # number of targets: most common count among frames (robust to occlusion)
    vals, cnts = np.unique(counts, return_counts=True)
    N = int(vals[np.argmax(cnts)])
    # also take max count of frames with full detection consistent
    log("target count mode", N, "count hist", dict(zip(vals.tolist(), cnts.tolist())))
    start = next(i for i, d in enumerate(dets) if len(d) == N)
    pos = np.array([[d[0], d[1]] for d in dets[start]], float)
    vel = np.zeros_like(pos)
    hist = {}
    # backfill frames before start with initial positions
    for i in range(start + 1):
        hist[frames[i]["frame_index"]] = pos.copy()
    lost = np.zeros(N, int)
    for i in range(start + 1, len(frames)):
        d = dets[i]
        pred = pos + vel
        if d:
            D = np.array([[x[0], x[1]] for x in d])
            C = np.linalg.norm(pred[:, None, :] - D[None, :, :], axis=2)
            ri, ci = linear_sum_assignment(C)
            matched = set()
            for a, b in zip(ri, ci):
                if C[a, b] < gate:
                    nv = D[b] - pos[a]
                    vel[a] = 0.5 * vel[a] + 0.5 * nv
                    pos[a] = D[b]
                    matched.add(a)
                    lost[a] = 0
            for a in range(N):
                if a not in matched:
                    pos[a] = pos[a] + 0.5 * vel[a]
                    vel[a] *= 0.5
                    lost[a] += 1
        else:
            pos = pos + 0.5 * vel
            vel *= 0.5
            lost += 1
        hist[frames[i]["frame_index"]] = pos.copy()
    return hist, N
