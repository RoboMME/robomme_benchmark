"""Grid detection and pattern extraction for PatternLock (pure numpy/PIL)."""
import json
from pathlib import Path
import numpy as np
from PIL import Image

WS = Path("/workspace")


def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def backproject(depth_m, K, pose):
    H, W = depth_m.shape
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    v, u = np.mgrid[0:H, 0:W]
    d = depth_m
    cam = np.stack([(u-cx)*d/fx, -(v-cy)*d/fy, -d], -1)
    R = quat_to_R(pose['quaternion_wxyz'])
    return cam @ R.T + np.array(pose['position_m'])


def _label(mask):
    """4-connected component labelling without scipy."""
    try:
        from scipy.ndimage import label
        return label(mask)
    except Exception:
        pass
    H, W = mask.shape
    lab = np.zeros((H, W), int)
    n = 0
    for i in range(H):
        for j in range(W):
            if mask[i, j] and not lab[i, j]:
                n += 1
                stack = [(i, j)]
                lab[i, j] = n
                while stack:
                    a, b = stack.pop()
                    for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        c, d = a+da, b+db
                        if 0 <= c < H and 0 <= d < W and mask[c, d] and not lab[c, d]:
                            lab[c, d] = n
                            stack.append((c, d))
    return lab, n


def detect_buttons(depth_m, K, pose, zmin=0.004, zmax=0.05, xmax_far=-0.6, min_px=5):
    """Return list of (x, y, ztop, npx) for raised blobs on the table seen by a camera."""
    P = backproject(depth_m, K, pose)
    valid = depth_m > 0
    mask = valid & (P[..., 2] > zmin) & (P[..., 2] < zmax) & (P[..., 0] > xmax_far)
    lab, n = _label(mask)
    out = []
    for i in range(1, n+1):
        sel = lab == i
        cnt = int(sel.sum())
        if cnt < min_px:
            continue
        p = P[sel]
        out.append((float(p[:, 0].mean()), float(p[:, 1].mean()), float(p[:, 2].max()), cnt))
    return out


def _cluster_1d(vals, tol):
    vals = sorted(vals)
    groups = [[vals[0]]]
    for v in vals[1:]:
        if v - groups[-1][-1] < tol:
            groups[-1].append(v)
        else:
            groups.append([v])
    return [float(np.mean(g)) for g in groups], [len(g) for g in groups]


def fit_grid(buttons, tol=0.03):
    """Fit a regular grid: returns dict with xs (rows), ys (cols), centers[(i,j)] -> (x,y), ztop."""
    if len(buttons) < 4:
        raise ValueError("too few buttons: %d" % len(buttons))
    xs, xc = _cluster_1d([b[0] for b in buttons], tol)
    ys, yc = _cluster_1d([b[1] for b in buttons], tol)
    # keep clusters with support
    xs = [x for x, c in zip(xs, xc) if c >= 2]
    ys = [y for y, c in zip(ys, yc) if c >= 2]
    centers = {}
    for b in buttons:
        i = int(np.argmin([abs(b[0]-x) for x in xs]))
        j = int(np.argmin([abs(b[1]-y) for y in ys]))
        if abs(b[0]-xs[i]) < tol and abs(b[1]-ys[j]) < tol:
            centers[(i, j)] = (b[0], b[1])
    # fill missing using row/col means
    for i in range(len(xs)):
        for j in range(len(ys)):
            centers.setdefault((i, j), (xs[i], ys[j]))
    ztop = float(np.median([b[2] for b in buttons]))
    return {"xs": xs, "ys": ys, "centers": centers, "ztop": ztop, "n_detected": len(buttons)}


def load_frame(frame, cam="front"):
    files = {(f["camera"], f["kind"]): f["path"] for f in frame["files"]}
    rgb = np.array(Image.open(WS / files[(cam, "rgb")]).convert("RGB"))
    dep = np.array(Image.open(WS / files[(cam, "depth")])).astype(float) * 0.001
    return rgb, dep, frame["camera_intrinsics"][cam]["matrix"], frame["camera_poses"][cam]


def grid_from_frame(frame):
    rgb, dep, K, pose = load_frame(frame, "front")
    return fit_grid(detect_buttons(dep, K, pose))


def extract_pattern(video_frames, grid, z_touch_margin=0.02, snap_radius=0.03):
    """From video TCP states, list grid cells visited while the tool is low.

    Returns (cells, z_low, info)."""
    st = np.array([f["state"][:3] for f in video_frames])
    zmin = float(st[:, 2].min())
    low = st[:, 2] < zmin + z_touch_margin
    cells = []
    dwell = {}
    for k, p in enumerate(st):
        if not low[k]:
            continue
        best, bd = None, 1e9
        for c, (cx, cy) in grid["centers"].items():
            d = np.hypot(p[0]-cx, p[1]-cy)
            if d < bd:
                best, bd = c, d
        if bd < snap_radius:
            dwell[best] = dwell.get(best, 0) + 1
            if not cells or cells[-1] != best:
                cells.append(best)
    return cells, zmin, {"dwell": {str(k): v for k, v in dwell.items()}, "n_low": int(low.sum())}


def grid_buttons(frame):
    """Detected button blobs (x, y, ztop, n) that look like grid buttons."""
    rgb, dep, K, pose = load_frame(frame, "front")
    b = detect_buttons(dep, K, pose)
    b = [x for x in b if x[3] < 400 and 0.008 < x[2] < 0.03]
    return b


def top_centers(frame):
    """Top-face centroids of buttons (less biased than full blobs)."""
    rgb, dep, K, pose = load_frame(frame, "front")
    P = backproject(dep, K, pose)
    mask = (dep > 0) & (P[..., 2] > 0.004) & (P[..., 2] < 0.05) & (P[..., 0] > -0.6)
    lab, n = _label(mask)
    out = []
    for i in range(1, n + 1):
        p = P[lab == i]
        if len(p) < 5 or len(p) > 400:
            continue
        zt = p[:, 2].max()
        if not (0.008 < zt < 0.03):
            continue
        top = p[p[:, 2] > zt - 0.002]
        out.append((float(top[:, 0].mean()), float(top[:, 1].mean()), float(zt), len(p)))
    return out


def grid_index(points):
    """Assign (row, col) indices to ~25 grid points. Returns dict idx->xy, theta, spacing, centroid."""
    P = np.array([p[:2] for p in points])
    c = P.mean(0)
    # nearest-neighbour directions
    angs = []
    dists = []
    for i in range(len(P)):
        d = np.linalg.norm(P - P[i], axis=1)
        d[i] = 1e9
        j = int(np.argmin(d))
        v = P[j] - P[i]
        angs.append(np.arctan2(v[1], v[0]))
        dists.append(d[j])
    s = float(np.median(dists))
    z = np.exp(4j * np.array(angs)).mean()
    theta = float(np.angle(z) / 4.0)  # in (-45, 45] deg
    Rm = np.array([[np.cos(-theta), -np.sin(-theta)], [np.sin(-theta), np.cos(-theta)]])
    Q = (P - c) @ Rm.T
    rows = np.round(Q[:, 0] / s)
    cols = np.round(Q[:, 1] / s)
    rows = rows - rows.min()
    cols = cols - cols.min()
    idx = {}
    for k in range(len(P)):
        idx[(int(rows[k]), int(cols[k]))] = P[k]
    return {"idx": idx, "theta": theta, "spacing": s, "centroid": c,
            "nrows": int(rows.max()) + 1, "ncols": int(cols.max()) + 1}


def fit_similarity(src, dst, with_scale=True):
    """2D similarity (scale*rotation+translation) least squares mapping src->dst (Umeyama)."""
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    ms, md = src.mean(0), dst.mean(0)
    A, B = src - ms, dst - md
    H = A.T @ B / len(src)
    U, S, Vt = np.linalg.svd(H)
    Dm = np.eye(2)
    if np.linalg.det(Vt.T @ U.T) < 0:
        Dm[1, 1] = -1
    Rr = Vt.T @ Dm @ U.T
    sc = float(np.trace(np.diag(S) @ Dm) / (A ** 2).sum(1).mean()) if with_scale else 1.0
    R = sc * Rr
    t = md - R @ ms
    resid = np.linalg.norm((src @ R.T + t) - dst, axis=1)
    return R, t, resid


def dwell_nodes(video_frames, grid, slow_mps=0.035, min_frames=2, snap=0.025, zmargin=0.012):
    """Pattern nodes = places where the demo tool (nearly) stopped while low, snapped to grid nodes.

    Speeds use frame timestamps, so subsampled videos work.  Returns (nodes, zmin)."""
    st = np.array([f["state"][:3] for f in video_frames])
    ts = np.array([float(f.get("timestamp_s", i / 20.0)) for i, f in enumerate(video_frames)])
    zmin = float(st[:, 2].min())
    dt = np.maximum(np.diff(ts), 1e-3)
    sp = np.r_[1.0, np.linalg.norm(np.diff(st[:, :2], axis=0), axis=1) / dt]
    low = st[:, 2] < zmin + zmargin
    groups, cur = [], []
    for k in range(len(st)):
        if low[k] and sp[k] < slow_mps:
            cur.append(k)
        else:
            if len(cur) >= min_frames:
                groups.append(cur)
            cur = []
    if len(cur) >= min_frames:
        groups.append(cur)
    nodes = []
    for g in groups:
        p = st[g, :2].mean(0)
        best, bd = None, 1e9
        for k, c in grid["idx"].items():
            d = float(np.linalg.norm(p - c))
            if d < bd:
                best, bd = k, d
        if bd > snap:
            continue
        if nodes and nodes[-1]["idx"] == best:
            nodes[-1]["end"] = g[-1]
            continue
        nodes.append({"idx": best, "point": p, "start": g[0], "end": g[-1], "dist": bd})
    return nodes, zmin


def red_onsets(video_frames, grid_pts, stride=1, min_px=6, rad=0.025):
    """Order in which grid buttons first turn red in the front view of the video.

    grid_pts: dict idx -> xy of the video grid.  Returns list of (idx, frame_pos)."""
    seen = {}
    keys = list(grid_pts.keys())
    C = np.array([grid_pts[k] for k in keys])
    for pos in range(0, len(video_frames), stride):
        f = video_frames[pos]
        try:
            rgb, dep, K, pose = load_frame(f, "front")
        except Exception:
            continue
        r = rgb.astype(int)
        red = (r[..., 0] > 150) & (r[..., 1] < 80) & (r[..., 2] < 80) & (dep > 0)
        if red.sum() < min_px:
            continue
        P = backproject(dep, K, pose)[red][:, :2]
        d = np.linalg.norm(P[:, None, :] - C[None, :, :], axis=2)
        j = d.argmin(1)
        ok = d[np.arange(len(j)), j] < rad
        cnt = np.bincount(j[ok], minlength=len(keys))
        for jj in np.nonzero(cnt >= min_px)[0]:
            k = keys[jj]
            if k not in seen:
                seen[k] = pos
    return sorted(seen.items(), key=lambda kv: kv[1])
