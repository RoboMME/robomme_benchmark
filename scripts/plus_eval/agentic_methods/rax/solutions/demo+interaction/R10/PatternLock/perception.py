import json, math
import numpy as np
from PIL import Image

def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])

def backproject(depth_m, K, pose):
    """depth_m HxW metres -> HxWx3 world points (nan where invalid)."""
    H, W = depth_m.shape
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    d = depth_m
    cam = np.stack([(u-cx)*d/fx, -(v-cy)*d/fy, -d], -1)
    R = quat_to_R(pose["quaternion_wxyz"])
    pts = cam @ R.T + np.array(pose["position_m"])
    pts[d <= 0] = np.nan
    return pts

def load_depth(path, scale=0.001):
    return np.asarray(Image.open(path)).astype(np.float64) * scale

def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB")).astype(np.int32)

def frame_files(f):
    return {(x['camera'], x['kind']): x['path'] for x in f['files']}

def detect_nodes(P, zmin=0.007, zmax=0.035):
    """Return list of world xyz centroids of raised targets from a world point map."""
    from scipy import ndimage
    z = P[..., 2]
    ok = np.isfinite(z) & (z > zmin) & (z < zmax)
    ok &= (P[..., 0] > -0.75) & (P[..., 0] < 0.35) & (np.abs(P[..., 1]) < 0.5)
    lab, n = ndimage.label(ok)
    out = []
    for i in range(1, n + 1):
        mk = lab == i
        cnt = int(mk.sum())
        if cnt < 4:
            continue
        pts = P[mk]
        # top surface points only
        ztop = np.nanmax(pts[:, 2])
        top = pts[pts[:, 2] > ztop - 0.002]
        if len(top) < 3:
            top = pts
        out.append((np.nanmean(top[:, 0]), np.nanmean(top[:, 1]), ztop, cnt))
    return out

def fit_grid(nodes, n=5):
    """nodes: list of (x,y,z,cnt). Returns dict with theta, spacing, origin, and index map.
    Labels rows along rotated +x (toward front camera), cols along rotated +y."""
    xy = np.array([[a[0], a[1]] for a in nodes])
    if len(xy) < 6:
        raise ValueError('too few nodes %d' % len(xy))
    # nearest neighbour vectors
    D = np.linalg.norm(xy[:, None] - xy[None], axis=-1)
    np.fill_diagonal(D, np.inf)
    nn = D.min(1)
    sp = float(np.median(nn))
    angs = []
    for i in range(len(xy)):
        for j in range(len(xy)):
            if i != j and D[i, j] < 1.3 * sp:
                v = xy[j] - xy[i]
                angs.append(math.atan2(v[1], v[0]))
    angs = np.array(angs)
    th = math.atan2(np.mean(np.sin(4 * angs)), np.mean(np.cos(4 * angs))) / 4.0
    c, s = math.cos(th), math.sin(th)
    R = np.array([[c, -s], [s, c]])  # grid->world
    g = xy @ R  # world->grid coords (u along rotated x, v along rotated y)
    return dict(theta=th, spacing=sp, R=R, g=g, xy=xy)

def label_grid(grid, n=5):
    """Assign (row,col) to each node using a regular lattice fit; returns dict (r,c)->world xy
    and the lattice function. Uses robust centre: lattice of n x n with spacing."""
    g, sp = grid['g'], grid['spacing']
    # robust lattice offset: median of residual mod spacing
    off = np.array([
        math.atan2(np.mean(np.sin(2*np.pi*g[:, k]/sp)), np.mean(np.cos(2*np.pi*g[:, k]/sp))) / (2*np.pi) * sp
        for k in range(2)])
    idx = np.round((g - off) / sp).astype(int)
    # choose 5x5 window covering most nodes
    best = None
    for r0 in range(idx[:, 0].min() - n, idx[:, 0].max() + 1):
        for c0 in range(idx[:, 1].min() - n, idx[:, 1].max() + 1):
            inside = ((idx[:, 0] >= r0) & (idx[:, 0] < r0 + n) & (idx[:, 1] >= c0) & (idx[:, 1] < c0 + n)).sum()
            if best is None or inside > best[0]:
                best = (inside, r0, c0)
    _, r0, c0 = best
    # refine spacing and offset with least squares on inliers
    rows, cols = idx[:, 0] - r0, idx[:, 1] - c0
    inl = (rows >= 0) & (rows < n) & (cols >= 0) & (cols < n)
    A = np.stack([rows[inl], cols[inl]], 1).astype(float)
    # g = origin + sp*[r,c]  (independent per axis)
    su = np.polyfit(A[:, 0], g[inl, 0], 1); sv = np.polyfit(A[:, 1], g[inl, 1], 1)
    R = grid['R']
    def node_world(r, c):
        gg = np.array([su[1] + su[0]*r, sv[1] + sv[0]*c])
        return R @ gg
    resid = []
    for k in np.nonzero(inl)[0]:
        resid.append(np.linalg.norm(node_world(rows[k], cols[k]) - grid['xy'][k]))
    return dict(node_world=node_world, n_inliers=int(inl.sum()), n_total=len(g),
                spacing_rc=(float(su[0]), float(sv[0])), resid_max=float(max(resid)), theta=grid['theta'])

def grid_from_frame(f, root='/workspace'):
    fl = frame_files(f)
    d = load_depth(root + '/' + fl[('front', 'depth')])
    P = backproject(d, f['camera_intrinsics']['front']['matrix'], f['camera_poses']['front'])
    nodes = detect_nodes(P)
    gr = fit_grid(nodes)
    lab = label_grid(gr)
    lab['nodes'] = nodes
    lab['ztop'] = float(np.median([a[2] for a in nodes]))
    return lab

def extract_pattern(frames, grid, rad_frac=0.35, zband=0.012):
    """frames: video_demo frames (with state). Returns list of (node, pen_down_before) entries."""
    zs = np.array([f['state'][2] for f in frames])
    zlow = float(np.min(zs))
    sp = float(np.mean(grid['spacing_rc']))
    allnodes = [(r, c) for r in range(5) for c in range(5)]
    W = np.array([grid['node_world'](r, c) for r, c in allnodes])
    seq = []
    lifted = True
    cur = None
    for f in frames:
        x, y, z = f['state'][:3]
        down = z < zlow + zband
        if not down:
            lifted = True if cur is not None or not seq else lifted
            cur = None
            continue
        dd = np.linalg.norm(W - np.array([x, y]), axis=1)
        k = int(np.argmin(dd))
        if dd[k] < rad_frac * sp:
            node = allnodes[k]
            if node != cur:
                if not seq or seq[-1][0] != node:
                    seq.append((node, lifted))
                cur = node
                lifted = False
        else:
            cur = None
    return seq, zlow


def red_mask(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return (r > 150) & (g < 70) & (b < 70) & (r - g > 100)

def world_to_pixel(pw, K, pose):
    R = quat_to_R(pose["quaternion_wxyz"])
    pc = R.T @ (np.asarray(pw, float) - np.array(pose["position_m"]))
    d = -pc[2]
    u = K[0][0] * pc[0] / d + K[0][2]
    v = -K[1][1] * pc[1] / d + K[1][2]
    return u, v

def video_offsets(frames, grid, pattern, zlow, zband=0.012):
    """For each pattern node, TCP xy minus node xy at the end of the dwell (lowest speed frame
    while within the node's radius). Returns median offset."""
    sp = float(np.mean(grid['spacing_rc']))
    offs = []
    for node, _ in pattern:
        w = grid['node_world'](*node)
        best = None
        for i in range(1, len(frames)):
            x, y, z = frames[i]['state'][:3]
            if z > zlow + zband:
                continue
            dd = np.hypot(x - w[0], y - w[1])
            if dd > 0.3 * sp:
                continue
            px, py = frames[i - 1]['state'][:2]
            v = np.hypot(x - px, y - py)
            if best is None or v < best[0] - 1e-5:
                best = (v, x - w[0], y - w[1])
        if best is not None:
            offs.append(best[1:])
    if not offs:
        return np.zeros(2), []
    o = np.median(np.array(offs), 0)
    return o, offs

def stick_tip_offset(f, root='/workspace'):
    """Lowest visible stick point (wrist depth) minus TCP, world frame."""
    fl = frame_files(f)
    d = load_depth(root + '/' + fl[('wrist', 'depth')])
    P = backproject(d, f['camera_intrinsics']['wrist']['matrix'], f['camera_poses']['wrist'])
    cz = f['camera_poses']['wrist']['position_m'][2]
    mk = (d > 0) & (d < cz * 0.8)
    if mk.sum() < 50:
        return None
    pts = P[mk]
    i = int(np.argmin(pts[:, 2]))
    return pts[i] - np.array(f['state'][:3])

def extract_dwells(frames, grid, zband=0.012, vthr=0.0006, min_len=3, rad_frac=0.4):
    """Node visits = runs of >=min_len consecutive near-stationary low frames near a node."""
    zs = np.array([f['state'][2] for f in frames])
    zlow = float(np.min(zs))
    sp = float(np.mean(grid['spacing_rc']))
    allnodes = [(r, c) for r in range(5) for c in range(5)]
    W = np.array([grid['node_world'](r, c) for r, c in allnodes])
    seq = []; dwz = []; offs = []
    run = []
    ts = [f.get('timestamp_s', 0) for f in frames]
    mdt = float(np.median(np.diff(ts))) if len(ts) > 2 else 0.05
    min_len_eff = [max(2, int(round(min_len * 0.05 / mdt))) if mdt > 0 else min_len]
    def flush(run):
        if len(run) < min_len_eff[0]:
            return
        xy = np.mean([frames[i]['state'][:2] for i in run], 0)
        dd = np.linalg.norm(W - xy, axis=1); k = int(np.argmin(dd))
        if dd[k] > rad_frac * sp:
            return
        node = allnodes[k]
        if seq and seq[-1] == node:
            return
        seq.append(node); dwz.append(float(np.mean([frames[i]['state'][2] for i in run])))
        offs.append(xy - W[k])
    for i in range(1, len(frames)):
        s = frames[i]['state']; p = frames[i - 1]['state']
        v = math.sqrt((s[0]-p[0])**2 + (s[1]-p[1])**2 + (s[2]-p[2])**2)
        dt = float(frames[i].get('timestamp_s', 0) - frames[i - 1].get('timestamp_s', 0))
        scale = max(1.0, dt / 0.05) if dt > 0 else 1.0
        if s[2] < zlow + zband and v < vthr * scale:
            run.append(i)
        else:
            flush(run); run = []
    flush(run)
    return seq, dwz, offs, zlow
