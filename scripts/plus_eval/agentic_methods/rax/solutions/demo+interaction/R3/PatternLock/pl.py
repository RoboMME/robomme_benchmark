"""PatternLock helpers: decoding, geometry, grid fitting, video pattern extraction, motion."""
import base64, io, json, math, time
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage

WS = Path('/workspace')

# ---------------------------------------------------------------- decoding
def decode(reply):
    content = reply.get('content', [])
    meta = {}
    for b in content:
        if b.get('type') == 'text':
            try:
                meta = json.loads(b['text']); break
            except Exception:
                pass
    imgs = {}
    for att in meta.get('attachments', []) or []:
        idx = att.get('content_index')
        if idx is None or idx >= len(content):
            continue
        blk = content[idx]
        data = blk.get('data')
        if not data:
            continue
        im = Image.open(io.BytesIO(base64.b64decode(data)))
        if att.get('kind') == 'depth':
            arr = np.asarray(im).astype(np.float64) * 0.001
        else:
            arr = np.asarray(im.convert('RGB'))
        imgs[(att.get('camera'), att.get('kind'))] = arr
    return meta, imgs

def quat_to_R(q):
    w, x, y, z = q
    n = math.sqrt(w*w + x*x + y*y + z*z) or 1.0
    w, x, y, z = w/n, x/n, y/n, z/n
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

def backproject(depth, K, pose, us, vs):
    """Pixels (arrays) -> world points (N,3); NaN for invalid depth."""
    us = np.asarray(us, float); vs = np.asarray(vs, float)
    ui = np.clip(np.round(us).astype(int), 0, depth.shape[1]-1)
    vi = np.clip(np.round(vs).astype(int), 0, depth.shape[0]-1)
    d = depth[vi, ui].astype(float)
    K = np.asarray(K, float)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    pc = np.stack([(us-cx)*d/fx, -(vs-cy)*d/fy, -d], 1)
    R = quat_to_R(pose['quaternion_wxyz']); t = np.asarray(pose['position_m'], float)
    pw = pc @ R.T + t
    pw[d <= 0] = np.nan
    return pw

def world_map(depth, K, pose):
    H, W = depth.shape
    vs, us = np.mgrid[0:H, 0:W]
    return backproject(depth, K, pose, us.ravel(), vs.ravel()).reshape(H, W, 3)

# ---------------------------------------------------------------- color masks
def red_mask(rgb):
    a = rgb.astype(int); r, g, b = a[..., 0], a[..., 1], a[..., 2]
    return (r > 130) & (g < 80) & (b < 80) & (r - g > 90)

def white_mask(rgb):
    a = rgb.astype(int); r, g, b = a[..., 0], a[..., 1], a[..., 2]
    return (r > 150) & (g > 150) & (b > 150) & (np.abs(r - b) < 45) & (np.abs(r - g) < 45)

# ---------------------------------------------------------------- grid detection
def blob_candidates(rgb, depth, K, pose, min_px=4, max_px=400):
    """White compact blobs -> list of dict(u,v,px,world)."""
    m = white_mask(rgb) | red_mask(rgb)
    lab, n = ndimage.label(m)
    out = []
    if n == 0:
        return out
    idx = range(1, n+1)
    cms = ndimage.center_of_mass(m, lab, idx); szs = ndimage.sum(m, lab, idx)
    sls = ndimage.find_objects(lab)
    W = world_map(depth, K, pose)
    for k, (cm, sz, sl) in enumerate(zip(cms, szs, sls)):
        if sz < min_px or sz > max_px:
            continue
        h = sl[0].stop - sl[0].start; w = sl[1].stop - sl[1].start
        if max(h, w) > 3.0 * max(1, min(h, w)) or sz < 0.3 * h * w:
            continue
        pts = W[lab == k+1]
        pts = pts[np.isfinite(pts).all(1)]
        if len(pts) == 0:
            continue
        out.append(dict(u=cm[1], v=cm[0], px=int(sz), world=np.median(pts, 0)))
    return out

def fit_grid(points_xy, n=5, spacing_hint=None):
    """Fit an n x n square lattice to candidate xy points (N,2).
    Returns dict(origin, e1, e2, spacing, angle, nodes (n,n,2), inliers, resid) or None.
    Indexing: node[i][j] = origin + i*e1 + j*e2 where e1 ~ +x rotated, e2 ~ +y rotated,
    angle in (-45, 45] deg."""
    P = np.asarray(points_xy, float)
    if len(P) < 8:
        return None
    from scipy.spatial import cKDTree
    tree = cKDTree(P)
    dd, ii = tree.query(P, k=min(5, len(P)))
    nn = dd[:, 1]
    s = spacing_hint or float(np.median(nn))
    # direction histogram mod 90 deg from near-neighbour vectors
    angs = []
    for a in range(len(P)):
        for j in range(1, dd.shape[1]):
            if abs(dd[a, j] - s) < 0.25 * s:
                v = P[ii[a, j]] - P[a]
                angs.append(math.atan2(v[1], v[0]))
    if not angs:
        return None
    angs = np.array(angs)
    # circular mean of 4*angle
    c = np.mean(np.cos(4*angs)); sn = np.mean(np.sin(4*angs))
    th = math.atan2(sn, c) / 4.0  # in (-45,45]
    best = None
    for _ in range(3):
        R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        Q = P @ R  # coordinates in rotated frame (columns e1,e2)
        # choose integer offsets: cluster coordinates
        best = None
        for q0 in [Q]:
            # robust origin: try anchors at each point
            for a in range(len(Q)):
                rel = (Q - Q[a]) / s
                idx = np.round(rel)
                err = np.linalg.norm(rel - idx, axis=1)
                ok = err < 0.3
                if ok.sum() < 8:
                    continue
                ii0 = idx[ok]
                # pick n x n window with most inliers
                for i0 in range(int(ii0[:, 0].min()), int(ii0[:, 0].max()) - n + 2):
                    for j0 in range(int(ii0[:, 1].min()), int(ii0[:, 1].max()) - n + 2):
                        inw = ok & (idx[:, 0] >= i0) & (idx[:, 0] < i0+n) & (idx[:, 1] >= j0) & (idx[:, 1] < j0+n)
                        cells = set(map(tuple, idx[inw].astype(int)))
                        score = len(cells) - 0.01 * err[inw].sum()
                        if best is None or score > best[0]:
                            best = (score, a, i0, j0, inw.copy(), idx.copy())
        if best is None:
            return None
        score, a, i0, j0, inw, idx = best
        # least squares refine origin, e1, e2
        I = idx[inw, 0] - i0; J = idx[inw, 1] - j0
        A = np.zeros((2*inw.sum(), 6))
        A[0::2, 0] = 1; A[0::2, 2] = I; A[0::2, 4] = J
        A[1::2, 1] = 1; A[1::2, 3] = I; A[1::2, 5] = J
        bvec = P[inw].reshape(-1)
        sol, *_ = np.linalg.lstsq(A, bvec, rcond=None)
        o = sol[0:2]; e1 = sol[2:4]; e2 = sol[4:6]
        s = 0.5 * (np.linalg.norm(e1) + np.linalg.norm(e2))
        th = math.atan2(e1[1], e1[0])
        th = (th + math.pi/4) % (math.pi/2) - math.pi/4
    nodes = np.zeros((n, n, 2))
    for i in range(n):
        for j in range(n):
            nodes[i, j] = o + i*e1 + j*e2
    pred = o + np.outer(I, e1) + np.outer(J, e2)
    resid = float(np.sqrt(np.mean(np.sum((pred - P[inw])**2, 1))))
    # canonicalise so that e1 is the axis closest to +x, e2 closest to +y
    nodes, e1, e2, o = canonical(nodes)
    return dict(origin=o, e1=e1, e2=e2, spacing=float(s), angle=float(math.degrees(math.atan2(e1[1], e1[0]))),
                nodes=nodes, ncells=int(round(score)), resid=resid, inliers=inw)

def canonical(nodes):
    n = nodes.shape[0]
    best = None
    for k in range(4):
        N = np.rot90(nodes, k, axes=(0, 1))
        for flip in (False, True):
            M = np.transpose(N, (1, 0, 2)) if flip else N
            e1 = (M[-1, 0] - M[0, 0]) / (n-1); e2 = (M[0, -1] - M[0, 0]) / (n-1)
            a1 = math.atan2(e1[1], e1[0]); a2 = math.atan2(e2[1], e2[0])
            # want e1 ~ +x, e2 ~ +y
            cost = abs(a1) + abs((a2 - math.pi/2 + math.pi) % (2*math.pi) - math.pi)
            if best is None or cost < best[0]:
                best = (cost, M.copy(), e1, e2)
    _, M, e1, e2 = best
    return M, e1, e2, M[0, 0].copy()

def grid_from_view(rgb, depth, K, pose, table_z=None):
    cands = blob_candidates(rgb, depth, K, pose)
    if not cands:
        return None, cands
    W = np.array([c['world'] for c in cands])
    if table_z is not None:
        keep = np.abs(W[:, 2] - table_z) < 0.05
        cands = [c for c, k in zip(cands, keep) if k]
        W = W[keep]
    g = fit_grid(W[:, :2])
    if g is not None:
        g['z'] = float(np.median(W[g['inliers'], 2]))
        g['cands'] = cands
    return g, cands

def nearest_node(g, xy, max_frac=0.35):
    N = g['nodes'].reshape(-1, 2)
    d = np.linalg.norm(N - np.asarray(xy)[None, :2], axis=1)
    k = int(d.argmin())
    if d[k] > max_frac * g['spacing']:
        return None, float(d[k])
    n = g['nodes'].shape[0]
    return (k // n, k % n), float(d[k])

# ---------------------------------------------------------------- reset sequence
def load_sequence(manifest_rel):
    man = json.loads((WS / manifest_rel).read_text())
    frames = sorted(man['frames'], key=lambda f: f['frame_index'])
    return man, frames

def frame_images(fr, cams=('front',), kinds=('rgb', 'depth')):
    files = {(f['camera'], f['kind']): f['path'] for f in fr['files']}
    out = {}
    for c in cams:
        for k in kinds:
            p = files.get((c, k))
            if p is None:
                continue
            im = Image.open(WS / p)
            out[(c, k)] = (np.asarray(im).astype(np.float64) * 0.001) if k == 'depth' else np.asarray(im.convert('RGB'))
    return out

def grid_from_frame(fr, cam='front'):
    im = frame_images(fr, (cam,))
    return grid_from_view(im[(cam, 'rgb')], im[(cam, 'depth')],
                          fr['camera_intrinsics'][cam]['matrix'], fr['camera_poses'][cam])

def extract_pattern(frames, log=print, cam='front', min_px=3):
    """Return (grid, pattern list of (i,j), details)."""
    vids = [f for f in frames if f.get('phase') == 'video_demo']
    if not vids:
        return None, [], {}
    g, cands = grid_from_frame(vids[0], cam)
    if g is None:
        return None, [], {'error': 'grid'}
    log(f"video grid: cells={g['ncells']} spacing={g['spacing']:.4f} angle={g['angle']:.1f} resid={g['resid']:.4f} z={g['z']:.3f} ncand={len(cands)}")
    base = None
    seq = []; events = []
    for fi, fr in enumerate(vids):
        im = frame_images(fr, (cam,))
        rgb, dep = im[(cam, 'rgb')], im[(cam, 'depth')]
        rm = red_mask(rgb)
        if base is None:
            base = ndimage.binary_dilation(rm, iterations=2)
        rm = rm & ~base
        vs, us = np.nonzero(rm)
        lit = {}
        if len(us):
            P = backproject(dep, fr['camera_intrinsics'][cam]['matrix'], fr['camera_poses'][cam], us, vs)
            ok = np.isfinite(P).all(1) & (np.abs(P[:, 2] - g['z']) < 0.04)
            for p in P[ok]:
                node, d = nearest_node(g, p[:2], 0.4)
                if node is not None:
                    lit[node] = lit.get(node, 0) + 1
        lit_nodes = [k for k, c in lit.items() if c >= min_px]
        tip = fr.get('state', [None]*3)[:3]
        for k in sorted(lit_nodes, key=lambda k: -lit[k]):
            if not seq or seq[-1] != k:
                if k in seq:
                    continue
                seq.append(k); events.append(dict(frame=fi, ts=fr.get('timestamp_s'), node=k, px=lit[k], tip=tip))
    for e in events:
        log(f"  event frame={e['frame']} ts={e['ts']} node={e['node']} px={e['px']} tip={np.round(e['tip'],3).tolist() if e['tip'][0] is not None else None}")
    return g, seq, {'events': events}

# ---------------------------------------------------------------- rotation utils
def rotvec_to_R(r):
    r = np.asarray(r, float); th = np.linalg.norm(r)
    if th < 1e-12:
        return np.eye(3)
    k = r / th; Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th)*Kx + (1-math.cos(th))*Kx@Kx

def R_to_rotvec(R):
    c = (np.trace(R) - 1) / 2; c = min(1.0, max(-1.0, c)); th = math.acos(c)
    if th < 1e-9:
        return np.zeros(3)
    if th > math.pi - 1e-4:
        # near pi: use eigen decomposition
        w, v = np.linalg.eigh((R + np.eye(3)) / 2)
        k = v[:, np.argmax(w)]
        return k * th
    return th / (2*math.sin(th)) * np.array([R[2, 1]-R[1, 2], R[0, 2]-R[2, 0], R[1, 0]-R[0, 1]])

# ---------------------------------------------------------------- robot
class Robot:
    def __init__(self, ctx, log=print):
        self.ctx = ctx; self.log = log
        self.meta = None; self.imgs = None; self.ended = False; self.complete = False
        self.steps_sent = 0
        self.R_ref = None

    def observe(self):
        rep = self.ctx.call('robomme_observe')
        self.meta, self.imgs = decode(rep)
        self._upd(self.meta)
        return self.meta

    def _upd(self, meta):
        if meta.get('episode_ended'):
            self.ended = True
        tr = meta.get('transition') or {}
        for s in tr.get('steps', []) or []:
            if s.get('task_complete'):
                self.complete = True
            if s.get('episode_ended'):
                self.ended = True
        if meta.get('task_complete'):
            self.complete = True

    @property
    def state(self):
        return np.asarray(self.meta['state'], float)

    @property
    def remaining(self):
        return (self.meta.get('episode_limits') or {}).get('remaining_episode_steps', 10**6)

    def step(self, rows):
        rows = [[float(max(-1.0, min(1.0, x))) for x in r] for r in rows]
        rep = self.ctx.call('robomme_step', actions=rows)
        meta, imgs = decode(rep)
        if 'state' in meta:
            self.meta, self.imgs = meta, imgs
        else:
            self.meta.update(meta)
        self._upd(meta)
        self.steps_sent += len(rows)
        return meta

    def rot_cmd(self, gain=0.8, lim=0.3):
        if self.R_ref is None:
            return [0.0, 0.0, 0.0]
        Rc = rotvec_to_R(self.state[3:6])
        e = R_to_rotvec(self.R_ref @ Rc.T)
        return list(np.clip(e / 0.5 * gain, -lim, lim))

    def move_to(self, target, vmax=0.02, tol=0.004, max_steps=80, chunk=1, gain=1.0, settle_tol=None):
        """P-control the TCP to target xyz (world m). vmax in m/step. Returns final error."""
        target = np.asarray(target, float)
        n = 0
        while n < max_steps and not self.ended and self.remaining > 0:
            p = self.state[:3]
            e = target - p
            dist = float(np.linalg.norm(e))
            if dist < tol:
                break
            d = e * gain
            nd = np.linalg.norm(d)
            if nd > vmax:
                d = d / nd * vmax
            rows = []
            for _ in range(chunk):
                rows.append(list(d / 0.05) + self.rot_cmd() + [0.0])
            self.step(rows)
            n += chunk
        return float(np.linalg.norm(target - self.state[:3])), n

    def line_to(self, target, speed=0.012, tol=0.004, max_steps=120):
        """Move along the straight segment from current position to target, constant speed (m/step)."""
        target = np.asarray(target, float)
        start = self.state[:3].copy()
        L = float(np.linalg.norm(target - start))
        if L < 1e-6:
            return self.move_to(target, tol=tol)
        u = (target - start) / L
        s = 0.0; n = 0
        while n < max_steps and not self.ended and self.remaining > 0:
            s = min(L, s + speed)
            carrot = start + u * s
            p = self.state[:3]
            # cross-track correction toward the carrot
            d = carrot - p
            nd = np.linalg.norm(d)
            if nd > 2.5*speed:
                d = d / nd * 2.5*speed
            self.step([list(d / 0.05) + self.rot_cmd() + [0.0]])
            n += 1
            if s >= L and np.linalg.norm(target - self.state[:3]) < tol:
                break
        e, m = self.move_to(target, vmax=speed, tol=tol, max_steps=20)
        return e, n + m

def red_near(meta, imgs, xy, radius, cams=('front', 'wrist'), zref=None):
    """Count red pixels whose world xy lies within radius of xy, per camera."""
    out = {}
    for cam in cams:
        rgb = imgs.get((cam, 'rgb')); dep = imgs.get((cam, 'depth'))
        if rgb is None or dep is None:
            continue
        vs, us = np.nonzero(red_mask(rgb))
        if len(us) == 0:
            out[cam] = 0; continue
        P = backproject(dep, meta['camera_intrinsics'][cam]['matrix'], meta['camera_poses'][cam], us, vs)
        ok = np.isfinite(P).all(1)
        if zref is not None:
            zlo, zhi = zref if isinstance(zref, (tuple, list)) else (zref - 0.03, zref + 0.03)
            ok &= (P[:, 2] > zlo) & (P[:, 2] < zhi)
        d = np.linalg.norm(P[:, :2] - np.asarray(xy)[None], axis=1)
        out[cam] = int((ok & (d < radius)).sum())
    return out

def press(robot, xy, z_travel, z_press_min, radius, zref, log=print, max_steps=14, thr=(20, 60)):
    """At xy: descend toward z_press_min step by step until the node is seen red. Returns (lit, info)."""
    xy = np.asarray(xy, float)
    lit = False; info = {}
    z_cmd = z_travel
    for k in range(max_steps):
        if robot.ended or robot.remaining <= 1:
            break
        z_cmd = max(z_press_min, z_cmd - 0.003)
        p = robot.state[:3]
        tgt = np.r_[xy, z_cmd]
        d = tgt - p
        dxy = d[:2]; n = np.linalg.norm(dxy)
        if n > 0.01:
            dxy = dxy / n * 0.01
        robot.step([[dxy[0]/0.05, dxy[1]/0.05, max(-0.4, min(0.4, d[2]/0.05))] + robot.rot_cmd() + [0.0]])
        info = red_near(robot.meta, robot.imgs, xy, radius, zref=zref)
        log(f"     k={k} z={robot.state[2]:.4f} zcmd={z_cmd:.4f} red={info}")
        if info.get('front', 0) >= thr[0] or info.get('wrist', 0) >= thr[1]:
            lit = True
            break
    log(f"   press xy={np.round(xy,4).tolist()} lit={lit} k={k} z={robot.state[2]:.4f} zcmd={z_cmd:.4f} red={info}")
    return lit, info

# ---------------------------------------------------------------- depth-based button detection
def depth_candidates(depth, K, pose, min_px=3, max_px=600):
    """Raised button tops: pixels 8-30 mm above the dominant table plane."""
    W = world_map(depth, K, pose)
    z = W[..., 2]
    fin = np.isfinite(z)
    if fin.sum() < 100:
        return [], None
    zz = z[fin]
    hist, edges = np.histogram(zz, bins=400, range=(-0.5, 0.5))
    table = float(edges[np.argmax(hist)] + 0.00125)
    sel = zz[np.abs(zz - table) < 0.004]
    if len(sel):
        table = float(np.median(sel))
    h = z - table
    m = fin & (h > 0.007) & (h < 0.035)
    lab, n = ndimage.label(m)
    out = []
    if n == 0:
        return out, table
    idx = range(1, n + 1)
    szs = ndimage.sum(m, lab, idx)
    sls = ndimage.find_objects(lab)
    for k, (sz, sl) in enumerate(zip(szs, sls)):
        if sz < min_px or sz > max_px:
            continue
        sub = lab[sl] == k + 1
        pts = W[sl][sub]
        pts = pts[np.isfinite(pts).all(1)]
        if len(pts) < min_px:
            continue
        top = pts[pts[:, 2] > np.percentile(pts[:, 2], 30) - 1e-9]
        ext = np.ptp(pts[:, :2], axis=0)
        if ext.max() > 0.05:
            continue
        out.append(dict(px=int(sz), world=np.r_[np.median(top[:, :2], 0), np.median(top[:, 2])], ext=ext))
    return out, table

def grid_from_depth(depth, K, pose):
    cands, table = depth_candidates(depth, K, pose)
    if len(cands) < 8:
        return None, cands
    Wp = np.array([c['world'] for c in cands])
    g = fit_grid(Wp[:, :2])
    if g is not None:
        g['z'] = float(np.median(Wp[g['inliers'], 2])); g['table'] = table; g['cands'] = cands
    return g, cands

# ---------------------------------------------------------------- stick tip
def stick_tip(depth, K, pose, tcp, min_comp=30):
    """Lowest point of the stick seen in a (wrist) depth image, world frame. Returns (xyz, npx) or (None, 0)."""
    W = world_map(depth, K, pose); z = W[..., 2]
    tcp = np.asarray(tcp, float)
    m = np.isfinite(z) & (z > 0.05) & (np.linalg.norm(W[..., :2] - tcp[None, None, :2], axis=2) < 0.15) & (z < tcp[2] + 0.2)
    lab, n = ndimage.label(m)
    if n == 0:
        return None, 0
    best = None
    for k in range(1, n + 1):
        sel = lab == k
        cnt = int(sel.sum())
        if cnt < min_comp:
            continue
        P = W[sel]
        zmin = float(P[:, 2].min())
        if best is None or zmin < best[0]:
            best = (zmin, P, cnt)
    if best is None:
        return None, 0
    zmin, P, cnt = best
    lo = P[P[:, 2] < zmin + 0.004]
    return np.r_[np.median(lo[:, :2], 0), zmin], cnt
