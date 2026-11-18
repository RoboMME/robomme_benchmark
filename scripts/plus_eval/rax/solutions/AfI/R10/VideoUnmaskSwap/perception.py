"""Perception for VideoUnmaskSwap: cube colors at video start, container tracking through video."""
import json, os
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

WS = '/workspace'

def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])

def backproject(depth_m, K, pose):
    K = K['matrix'] if isinstance(K, dict) else K
    H, W = depth_m.shape
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    d = depth_m
    pc = np.stack([(u-cx)*d/fx, -(v-cy)*d/fy, -d], -1)
    R = quat_to_R(pose['quaternion_wxyz']); t = np.array(pose['position_m'])
    P = pc @ R.T + t
    P[d <= 0] = np.nan
    return P

def load_frame(frame, cam='front'):
    if 'arrays' in frame:
        rgb = frame['arrays'][(cam, 'rgb')].astype(int)
        dep = frame['arrays'][(cam, 'depth')].astype(float) / 1000.0
        return rgb, backproject(dep, frame['camera_intrinsics'][cam], frame['camera_poses'][cam])
    files = {(x['camera'], x['kind']): x['path'] for x in frame['files']}
    def p(path):
        return path if os.path.isabs(path) else os.path.join(WS, path)
    rgb = np.array(Image.open(p(files[(cam, 'rgb')])).convert('RGB')).astype(int)
    dep = np.array(Image.open(p(files[(cam, 'depth')]))).astype(float) / 1000.0
    P = backproject(dep, frame['camera_intrinsics'][cam], frame['camera_poses'][cam])
    return rgb, P

TABLE = dict(xmin=-0.45, xmax=0.30, ymin=-0.40, ymax=0.40)

def in_table(P):
    with np.errstate(invalid='ignore'):
        return (P[..., 0] > TABLE['xmin']) & (P[..., 0] < TABLE['xmax']) & \
               (P[..., 1] > TABLE['ymin']) & (P[..., 1] < TABLE['ymax'])

def white_mask(rgb):
    mx = rgb.max(-1); mn = rgb.min(-1)
    return (mn > 110) & (mx - mn < 50)

def container_points(rgb, P, zlo=None, zhi=0.12):
    """xy points of container tops (white, upper part)."""
    with np.errstate(invalid='ignore'):
        z = P[..., 2]
        ok = white_mask(rgb) & in_table(P) & (z > 0.012) & (z < zhi)
    pts = P[ok]
    if zlo is None and len(pts):
        ztop = np.percentile(pts[:, 2], 95)
        zlo = ztop - 0.03
    if len(pts):
        pts = pts[pts[:, 2] > zlo]
    return pts

def grid_blobs(pts, res=0.004, min_cells=6):
    if len(pts) == 0:
        return []
    ix = np.floor((pts[:, 0] - TABLE['xmin']) / res).astype(int)
    iy = np.floor((pts[:, 1] - TABLE['ymin']) / res).astype(int)
    H = int((TABLE['xmax'] - TABLE['xmin']) / res) + 1
    W = int((TABLE['ymax'] - TABLE['ymin']) / res) + 1
    g = np.zeros((H, W), bool)
    g[ix.clip(0, H-1), iy.clip(0, W-1)] = True
    g = ndimage.binary_closing(g, iterations=1)
    lab, n = ndimage.label(g)
    out = []
    for i in range(1, n+1):
        cells = np.argwhere(lab == i)
        if len(cells) < min_cells:
            continue
        c = cells.mean(0)
        out.append(dict(x=TABLE['xmin'] + (c[0]+0.5)*res, y=TABLE['ymin'] + (c[1]+0.5)*res, cells=len(cells)))
    return out

COLORS = {'red': (220, 30, 30), 'green': (30, 200, 30), 'blue': (30, 30, 220),
          'yellow': (220, 220, 30), 'orange': (230, 140, 20), 'purple': (150, 40, 180)}

def classify_color(rgb_px):
    r, g, b = [int(v) for v in rgb_px]
    mx = max(r, g, b); mn = min(r, g, b)
    if mx < 50 or mx - mn < 60:
        return None
    # hue
    import colorsys
    h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
    h *= 360
    if h < 15 or h >= 330: return 'red'
    if h < 45: return 'orange'
    if h < 70: return 'yellow'
    if h < 170: return 'green'
    if h < 260: return 'blue'
    return 'purple'

def cube_detections(rgb, P):
    """Colored cubes above the table (excluding table texture via height)."""
    with np.errstate(invalid='ignore'):
        z = P[..., 2]
        cand = in_table(P) & (z > 0.006) & (z < 0.08)
    rgbf = rgb.astype(float)
    mx = rgbf.max(-1); mn = rgbf.min(-1)
    sat = (mx - mn) / np.maximum(mx, 1)
    cand &= (sat > 0.45) & (mx > 60)
    labels = np.full(rgb.shape[:2], '', dtype=object)
    import colorsys
    hsv = np.zeros(rgb.shape[:2])
    r, g, b = rgbf[..., 0], rgbf[..., 1], rgbf[..., 2]
    # hue computation vectorised
    d = np.maximum(mx - mn, 1e-6)
    h = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60
    out = []
    for name, (lo, hi) in {'red': (-1, 15), 'red2': (330, 361), 'orange': (15, 45), 'yellow': (45, 70),
                           'green': (70, 170), 'blue': (170, 260), 'purple': (260, 330)}.items():
        m = cand & (h >= lo) & (h < hi)
        lab, n = ndimage.label(m)
        for i in range(1, n+1):
            s = lab == i
            if s.sum() < 4:
                continue
            p = P[s]
            out.append(dict(color=name.rstrip('2'), x=float(np.nanmean(p[:, 0])), y=float(np.nanmean(p[:, 1])),
                            z=float(np.nanmax(p[:, 2])), n=int(s.sum())))
    return out


def track_containers(frames, log=print):
    """Track containers through video_demo frames + execution frame.
    Returns dict with initial cube colors per container, final positions."""
    vid = [f for f in frames if f['phase'] == 'video_demo']
    vid.sort(key=lambda f: f['frame_index'])
    exe = [f for f in frames if f['phase'] == 'execution']
    seq = vid + exe[:1]
    # 1. cubes in early frames: first frame with cube detections and no containers
    cubes = None
    start = None
    counts = []
    for i, f in enumerate(seq):
        rgb, P = load_frame(f)
        pts = container_points(rgb, P)
        if len(pts) < 200:
            cd = cube_detections(rgb, P)
            if cd:
                cubes = cd
            continue
        b = [q for q in grid_blobs(pts) if q['cells'] >= 60]
        counts.append((i, len(b), b))
        if len(counts) >= 12:
            break
    # K: mode of big-blob count over the first frames after containers appear
    from collections import Counter
    K = Counter([c[1] for c in counts]).most_common(1)[0][0]
    # choose first frame with K big blobs that all have ~ normal size
    start_i, b0 = None, None
    for i, k, b in counts:
        if k == K:
            start_i, b0 = i, b
            break
    centers = np.array([[q['x'], q['y']] for q in b0])
    log(f'track: K={K} start frame {start_i} cubes={[(c["color"], round(c["x"],3), round(c["y"],3)) for c in (cubes or [])]}')
    # assign cubes to containers by nearest (cube visible before drop)
    assign = {}
    if cubes:
        cubes = [c for c in cubes if c['n'] >= 15]
        C = np.array([[c['x'], c['y']] for c in cubes])
        D = np.linalg.norm(C[:, None, :] - centers[None, :, :], axis=-1)
        r, c = linear_sum_assignment(D)
        for ri, ci in zip(r, c):
            if D[ri, ci] < 0.05:
                assign[int(ci)] = cubes[ri]['color']
            else:
                log(f'track: cube {cubes[ri]["color"]} far from containers {D[ri,ci]:.3f}')
    # 2. track
    vel = np.zeros_like(centers)
    traj = [centers.copy()]
    min_sep_hist = []
    for i in range(start_i + 1, len(seq)):
        rgb, P = load_frame(seq[i])
        pts = container_points(rgb, P)[:, :2]
        pred = centers + vel
        c = pred.copy()
        for it in range(4):
            if len(pts) == 0:
                break
            d = np.linalg.norm(pts[:, None, :] - c[None, :, :], axis=-1)
            a = d.argmin(1)
            dm = d.min(1)
            newc = c.copy()
            for k in range(K):
                sel = (a == k) & (dm < 0.045)
                if sel.sum() >= 8:
                    newc[k] = pts[sel].mean(0)
            c = newc
        vel = 0.5 * vel + 0.5 * (c - centers)
        centers = c
        traj.append(centers.copy())
        dd = np.linalg.norm(centers[:, None] - centers[None], axis=-1) + np.eye(K)
        min_sep_hist.append(dd.min())
    # final consistency check with blobs on last frame
    rgb, P = load_frame(seq[-1])
    bl = [q for q in grid_blobs(container_points(rgb, P)) if q['cells'] >= 60]
    log(f'track: final blobs {len(bl)}, min sep during tracking {min(min_sep_hist) if min_sep_hist else -1:.3f}')
    return dict(K=K, start=start_i, assign=assign, final=centers, traj=np.array(traj),
                min_sep=min_sep_hist, final_blobs=bl, cubes=cubes)


def fused_top_points(frame, zlo=0.060, zhi=0.090, cams=('front', 'wrist')):
    out = []
    for cam in cams:
        try:
            rgb, P = load_frame(frame, cam)
        except Exception:
            continue
        with np.errstate(invalid='ignore'):
            ok = white_mask(rgb) & in_table(P) & (P[..., 2] > zlo) & (P[..., 2] < zhi)
        out.append(P[ok][:, :2])
    return np.concatenate(out) if out else np.zeros((0, 2))


def grid_detect(pts, res=0.003):
    if len(pts) == 0:
        return [], None
    ix = np.floor((pts[:, 0] - TABLE['xmin']) / res).astype(int)
    iy = np.floor((pts[:, 1] - TABLE['ymin']) / res).astype(int)
    H = int((TABLE['xmax'] - TABLE['xmin']) / res) + 1
    W = int((TABLE['ymax'] - TABLE['ymin']) / res) + 1
    g = np.zeros((H, W), bool)
    g[ix.clip(0, H-1), iy.clip(0, W-1)] = True
    g = ndimage.binary_closing(g, iterations=1)
    g = ndimage.binary_opening(g, iterations=1)
    lab, n = ndimage.label(g)
    blobs = []
    for i in range(1, n+1):
        cells = np.argwhere(lab == i)
        if len(cells) < 15:
            continue
        xy = np.stack([TABLE['xmin'] + (cells[:, 0]+0.5)*res, TABLE['ymin'] + (cells[:, 1]+0.5)*res], 1)
        blobs.append(xy)
    return blobs, res


def detections(frame):
    blobs, _ = grid_detect(fused_top_points(frame))
    dets = [(b.mean(0), len(b)) for b in blobs]
    # merge fragments closer than 3 cm
    merged = True
    while merged and len(dets) > 1:
        merged = False
        for i in range(len(dets)):
            for j in range(i+1, len(dets)):
                if np.linalg.norm(dets[i][0] - dets[j][0]) < 0.03:
                    c = (dets[i][0]*dets[i][1] + dets[j][0]*dets[j][1]) / (dets[i][1] + dets[j][1])
                    dets[i] = (c, dets[i][1] + dets[j][1]); dets.pop(j); merged = True
                    break
            if merged:
                break
    return dets


def track_v2(frames, log=print, gate=0.035, full_cells=150):
    vid = sorted([f for f in frames if f['phase'] == 'video_demo'], key=lambda f: f['frame_index'])
    exe = [f for f in frames if f['phase'] == 'execution']
    seq = vid + exe[:1]
    cubes = None; first = None
    from collections import Counter
    cand = []
    for i, f in enumerate(seq):
        dets = detections(f)
        if sum(c for _, c in dets) < 300:
            rgb, P = load_frame(f)
            cd = [c for c in cube_detections(rgb, P) if c['n'] >= 15]
            if cd:
                cubes = cd
            continue
        cand.append((i, dets))
        if len(cand) >= 15:
            break
    # number of containers: max over first frames of count of full detections, or cubes
    K = max(Counter([len(d) for _, d in cand]).most_common(1)[0][0], len(cubes or []))
    # start frame: first candidate with exactly K detections
    start = None
    for i, d in cand:
        if len(d) == K:
            start = i; break
    if start is None:
        start, d = cand[0]
    dets = detections(seq[start])
    pos = np.array([c for c, _ in dets])
    if len(pos) < K and cubes:
        # add hidden containers at unmatched cube positions
        C = np.array([[c['x'], c['y']] for c in cubes])
        dd = np.linalg.norm(C[:, None] - pos[None], axis=-1).min(1)
        extra = C[np.argsort(-dd)[:K - len(pos)]]
        pos = np.concatenate([pos, extra])
    K = len(pos)
    vel = np.zeros_like(pos)
    hidden = np.zeros(K, bool)
    assign = {}
    if cubes:
        C = np.array([[c['x'], c['y']] for c in cubes])
        D = np.linalg.norm(C[:, None, :] - pos[None, :, :], axis=-1)
        r, c = linear_sum_assignment(D)
        for ri, ci in zip(r, c):
            if D[ri, ci] < 0.06:
                assign[int(ci)] = cubes[ri]['color']
    log(f'track_v2: K={K} start={start} assign={assign}')
    traj = [pos.copy()]; events = []
    for i in range(start + 1, len(seq)):
        dets = detections(seq[i])
        if not dets:
            traj.append(pos.copy()); continue
        Dp = np.array([c for c, _ in dets]); Dn = np.array([n for _, n in dets])
        pred = pos + vel
        vis = np.where(~hidden)[0]
        used_d = set(); matched = {}
        if len(vis):
            cost = np.linalg.norm(pred[vis][:, None] - Dp[None], axis=-1)
            r, c = linear_sum_assignment(cost)
            for ri, ci in zip(r, c):
                if cost[ri, ci] < gate:
                    matched[vis[ri]] = ci; used_d.add(ci)
        # unmatched tracks -> hidden ; unmatched dets -> hidden tracks
        un_t = [k for k in range(K) if k not in matched]
        un_d = [j for j in range(len(dets)) if j not in used_d]
        if un_t and un_d:
            cost = np.linalg.norm(pos[un_t][:, None] - Dp[un_d][None], axis=-1)
            r, c = linear_sum_assignment(cost)
            for ri, ci in zip(r, c):
                if cost[ri, ci] < 0.20:
                    k = un_t[ri]; j = un_d[ci]
                    if hidden[k] or cost[ri, ci] < 0.06:
                        matched[k] = j
                        if hidden[k]:
                            events.append((i, 'reacquire', k, round(float(cost[ri, ci]), 3)))
        for k in range(K):
            if k in matched:
                j = matched[k]; new = Dp[j]
                if not hidden[k] and Dn[j] >= full_cells:
                    vel = vel.copy(); vel[k] = 0.6*vel[k] + 0.4*(new - pos[k])
                else:
                    vel[k] = 0.0 * vel[k]
                pos[k] = new; hidden[k] = False
            else:
                if not hidden[k]:
                    events.append((i, 'hide', k, pos[k].round(3).tolist()))
                hidden[k] = True; vel[k] = 0
        traj.append(pos.copy())
    log(f'track_v2: events={events[:50]}')
    return dict(K=K, start=start, assign=assign, final=pos, traj=np.array(traj), events=events,
                hidden=hidden, cubes=cubes)


def frame_from_reply(reply):
    """Build a frame dict with decoded arrays from an MCP observe/step reply."""
    import base64, io
    blocks = reply['content'] if isinstance(reply, dict) and 'content' in reply else reply
    meta = None
    for b in blocks:
        if isinstance(b, dict) and b.get('type') == 'text':
            try:
                meta = json.loads(b['text']); break
            except Exception:
                pass
    arrays = {}
    for a in meta.get('attachments', []):
        b = blocks[a['content_index']]
        data = b.get('data')
        if data is None:
            continue
        im = Image.open(io.BytesIO(base64.b64decode(data)))
        arr = np.array(im.convert('RGB')) if a['kind'] == 'rgb' else np.array(im)
        arrays[(a['camera'], a['kind'])] = arr
    meta['arrays'] = arrays
    return meta


def blob_yaw(xy):
    """Yaw (rad, in [-pi/4, pi/4)) of a square top face from its xy cells via min-area rectangle."""
    c = xy - xy.mean(0)
    best = None
    for a in np.arange(0, 90, 1.0):
        t = np.deg2rad(a)
        q = c @ np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        ar = np.ptp(q[:, 0]) * np.ptp(q[:, 1])
        if best is None or ar < best[0]:
            best = (ar, a)
    a = best[1]
    if a >= 45:
        a -= 90
    return np.deg2rad(a)


def live_containers(frame, cams=('front', 'wrist')):
    blobs, _ = grid_detect(fused_top_points(frame, cams=cams))
    out = []
    for b in blobs:
        out.append(dict(xy=b.mean(0), n=len(b), yaw=blob_yaw(b) if len(b) >= 60 else None))
    return out


class Tracker:
    """Incremental container tracker (Hungarian on top-face detections, hidden-track reacquisition)."""

    def __init__(self, pos, colors, gate=0.035, full_cells=150, log=print):
        self.pos = np.array(pos, float)
        self.K = len(self.pos)
        self.vel = np.zeros_like(self.pos)
        self.hidden = np.zeros(self.K, bool)
        self.colors = dict(colors)          # track idx -> color
        self.gate, self.full_cells, self.log = gate, full_cells, log
        self.events = []
        self.reacq_dist = np.zeros(self.K)  # cumulative suspicious reacquire distance (uncertainty)
        self.n = 0

    def update(self, dets):
        self.n += 1
        if not dets:
            return
        Dp = np.array([c for c, _ in dets]); Dn = np.array([n for _, n in dets])
        pred = self.pos + self.vel
        vis = np.where(~self.hidden)[0]
        used = set(); matched = {}
        if len(vis):
            cost = np.linalg.norm(pred[vis][:, None] - Dp[None], axis=-1)
            r, c = linear_sum_assignment(cost)
            for ri, ci in zip(r, c):
                if cost[ri, ci] < self.gate:
                    matched[vis[ri]] = ci; used.add(ci)
        un_t = [k for k in range(self.K) if k not in matched]
        un_d = [j for j in range(len(dets)) if j not in used]
        if un_t and un_d:
            cost = np.linalg.norm(self.pos[un_t][:, None] - Dp[un_d][None], axis=-1)
            r, c = linear_sum_assignment(cost)
            for ri, ci in zip(r, c):
                k = un_t[ri]; j = un_d[ci]
                if cost[ri, ci] < 0.20 and (self.hidden[k] or cost[ri, ci] < 0.06):
                    matched[k] = j
                    if self.hidden[k] and cost[ri, ci] > 0.03:
                        self.reacq_dist[k] += cost[ri, ci]
                        self.events.append((self.n, 'reacquire', k, round(float(cost[ri, ci]), 3)))
        for k in range(self.K):
            if k in matched:
                j = matched[k]; new = Dp[j]
                if not self.hidden[k] and Dn[j] >= self.full_cells:
                    self.vel[k] = 0.6 * self.vel[k] + 0.4 * (new - self.pos[k])
                else:
                    self.vel[k] = 0
                self.pos[k] = new; self.hidden[k] = False
            else:
                self.hidden[k] = True; self.vel[k] = 0


def init_from_video(frames, log=print):
    """Find cubes and the container drop; return (Tracker, video frame list after start, cubes)."""
    vid = sorted([f for f in frames if f['phase'] == 'video_demo'], key=lambda f: f['frame_index'])
    exe = [f for f in frames if f['phase'] == 'execution']
    seq = vid + exe[:1]
    cubes = None
    cand = []
    for i, f in enumerate(seq):
        dets = detections(f)
        if sum(c for _, c in dets) < 300:
            cd = []
            for cam in ('wrist', 'front'):
                try:
                    rgb, P = load_frame(f, cam)
                    cd += [dict(c, cam=cam) for c in cube_detections(rgb, P) if c['n'] >= 12]
                    cd += [dict(c, cam=cam) for c in achro_cube_detections(rgb, P) if 20 <= c['n'] <= 900]
                except Exception:
                    pass
            if cd:
                cubes = merge_cubes(cd)
            continue
        cand.append((i, dets))
        if len(cand) >= 6:
            break
    if not cand:
        return None, seq, cubes
    # first frame where containers just appeared: they sit on cube positions
    i0, dets0 = cand[0]
    pos = [c for c, _ in dets0]
    colors = {}
    if cubes:
        C = np.array([[c['x'], c['y']] for c in cubes])
        P0 = np.array(pos)
        D = np.linalg.norm(C[:, None] - P0[None], axis=-1)
        r, c = linear_sum_assignment(D)
        assigned_c = set()
        for ri, ci in zip(r, c):
            if D[ri, ci] < 0.04:
                colors[int(ci)] = cubes[ri]['color']; assigned_c.add(ri)
        # cubes without a visible container (occluded) -> add hidden container track at cube position
        for ri in range(len(cubes)):
            if ri not in assigned_c and D[ri].min() > 0.04:
                pos.append(C[ri]); colors[len(pos) - 1] = cubes[ri]['color']
    tr = Tracker(pos, colors, log=log)
    for k in range(len(pos)):
        if k >= len(dets0):
            tr.hidden[k] = True
    log(f'init: start frame {i0}, K={tr.K}, colors={colors}, cubes={[(c["color"], round(c["x"],3), round(c["y"],3)) for c in (cubes or [])]}')
    return tr, seq[i0 + 1:], cubes


def achro_cube_detections(rgb, P):
    """White / black cubes on the table (pre-drop frames only; containers would also be white)."""
    with np.errstate(invalid='ignore'):
        z = P[..., 2]
        cand = in_table(P) & (z > 0.008) & (z < 0.045)
    rgbf = rgb.astype(float); mx = rgbf.max(-1); mn = rgbf.min(-1)
    out = []
    for name, m in (('white', (mn > 150) & (mx - mn < 30)), ('black', mx < 70)):
        lab, n = ndimage.label(cand & m)
        for i in range(1, n + 1):
            sel = lab == i
            if sel.sum() < 6:
                continue
            p = P[sel]
            out.append(dict(color=name, x=float(np.nanmean(p[:, 0])), y=float(np.nanmean(p[:, 1])),
                            z=float(np.nanmax(p[:, 2])), n=int(sel.sum())))
    return out


def merge_cubes(cd, r=0.05):
    """Merge cube detections from two cameras (same color within r)."""
    out = []
    for c in sorted(cd, key=lambda c: (c.get('cam') != 'wrist', -c['n'])):
        dup = False
        for o in out:
            if o['color'] == c['color'] and np.hypot(o['x'] - c['x'], o['y'] - c['y']) < r:
                dup = True; break
        if not dup:
            out.append(dict(c))
    return out
