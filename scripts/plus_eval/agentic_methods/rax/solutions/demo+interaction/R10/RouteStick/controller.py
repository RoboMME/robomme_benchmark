"""RouteStick controller: replay the conditioning video's route (target visit order and
arc side around each obstacle) on the live target layout."""
import json, os, sys, time, traceback
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rsutil import packet, images, Cam
from route import parse_route

WS = '/workspace'
OUT = os.path.join(WS, 'output')
LOG = []

def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)

def label(mask):
    """4-connected components; returns list of (ys, xs)."""
    H, W = mask.shape
    seen = np.zeros_like(mask, bool)
    comps = []
    for y0, x0 in zip(*np.nonzero(mask)):
        if seen[y0, x0]:
            continue
        st = [(y0, x0)]; seen[y0, x0] = True; ys = []; xs = []
        while st:
            y, x = st.pop(); ys.append(y); xs.append(x)
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                yy, xx = y + dy, x + dx
                if 0 <= yy < H and 0 <= xx < W and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True; st.append((yy, xx))
        comps.append((np.array(ys), np.array(xs)))
    return comps

def detect_targets(rgb, depth, cam):
    """White/grey discs lying on the table in front-camera RGB-D. Returns list of world xy sorted by y."""
    rgb = rgb.astype(int)
    W = cam.world_map(depth)
    z = W[..., 2]
    mn = rgb.min(-1); rg = rgb.max(-1) - mn
    mask = (mn > 150) & (rg < 40) & (z > -0.01) & (z < 0.03) & (W[..., 0] > -0.45)
    mask &= ~np.isnan(z)
    out = []
    for ys, xs in label(mask):
        if len(xs) < 6:
            continue
        P = W[ys, xs]
        c = np.nanmedian(P, 0)
        out.append((c[:2], len(xs)))
    # merge close blobs
    merged = []
    for c, n in sorted(out, key=lambda t: -t[1]):
        if all(np.linalg.norm(c - m) > 0.05 for m, _ in merged):
            merged.append((c, n))
    pts = sorted([m for m, _ in merged], key=lambda p: p[1])
    return pts

def detect_frame(f):
    """Detect targets with both cameras of a published frame and merge (front + wrist)."""
    found = {}
    for cn in ('front', 'wrist'):
        try:
            rgb, dep, cam = load_frame(f, cn)
            found[cn] = detect_targets(rgb, dep, cam)
        except Exception as e:
            log('detect', cn, 'failed', e)
            found[cn] = []
    pts = [np.asarray(p) for p in found.get('front', [])]
    for p in found.get('wrist', []):
        if all(np.linalg.norm(p - q) > 0.04 for q in pts):
            pts.append(np.asarray(p))
    log('detect counts front', len(found.get('front', [])), 'wrist', len(found.get('wrist', [])), 'merged', len(pts))
    return sorted(pts, key=lambda p: p[1])

def stick_tip_z(f):
    """Lowest world z of the drawing stick seen by the wrist camera (stick pixels: near, grey)."""
    rgb, dep, cam = load_frame(f, 'wrist')
    W = cam.world_map(dep)
    rgbi = rgb.astype(int); mn = rgbi.min(-1); rg = rgbi.max(-1) - mn
    stick = (dep > 0) & (dep < 0.3) & (rg < 30)
    if stick.sum() < 50:
        return None
    z = W[stick][:, 2]
    z = z[~np.isnan(z)]
    return float(np.mean(np.sort(z)[:15])), int(stick.sum())

def load_frame(f, cam_name):
    files = {(x['camera'], x['kind']): x['path'] for x in f['files']}
    rgb = np.asarray(Image.open(os.path.join(WS, files[(cam_name, 'rgb')])).convert('RGB'))
    dep = np.asarray(Image.open(os.path.join(WS, files[(cam_name, 'depth')]))).astype(float) * 0.001
    cam = Cam(f['camera_intrinsics'][cam_name], f['camera_poses'][cam_name])
    return rgb, dep, cam

def rotvec_to_R(r):
    r = np.asarray(r, float); th = np.linalg.norm(r)
    if th < 1e-9:
        return np.eye(3)
    k = r / th; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K

def R_to_rotvec(R):
    c = np.clip((np.trace(R) - 1) / 2, -1, 1); th = np.arccos(c)
    if th < 1e-9:
        return np.zeros(3)
    if th > np.pi - 1e-4:
        # near pi: axis from the largest diagonal
        A = (R + np.eye(3)) / 2
        k = np.sqrt(np.clip(np.diag(A), 0, None))
        i = int(np.argmax(k)); 
        k = A[:, i] / max(np.sqrt(A[i, i]), 1e-9)
        return k / np.linalg.norm(k) * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(th))
    return w * th

YAW = {'on': False, 'dir': None}

def rot_cmd(state):
    """Rotation action keeping the tool pointing down and the hand's finger axis along YAW['dir'] (mod pi)."""
    if not YAW['on'] or YAW['dir'] is None:
        return [0.0, 0.0, 0.0]
    R = rotvec_to_R(state[3:6])
    t = np.asarray(YAW['dir'], float); t = t / (np.linalg.norm(t) + 1e-9)
    best = None
    for sgn in (1, -1):
        y = sgn * np.array([t[0], t[1], 0.0])
        zc = np.array([0, 0, -1.0]); x = np.cross(y, zc)
        Rd = np.stack([x, y, zc], 1)
        e = R_to_rotvec(Rd @ R.T)
        if best is None or np.linalg.norm(e) < np.linalg.norm(best):
            best = e
    a = np.clip(best / 0.5, -0.35, 0.35)
    return [float(a[0]), float(a[1]), float(a[2])]

class Robot:
    def __init__(self, ctx):
        self.ctx = ctx
        self.ended = False
        self.complete = False
        self.state = None
        self.step_id = 0
        self.remaining = None
        self.ncalls = 0

    def observe(self):
        r = self.ctx.call('robomme_observe')
        pk = packet(r)
        self._upd(pk)
        return r, pk

    def _upd(self, pk):
        self.state = np.array(pk['state'], float)
        self.step_id = pk.get('step_id', self.step_id)
        lim = pk.get('episode_limits') or {}
        self.remaining = lim.get('remaining_episode_steps', self.remaining)
        if pk.get('episode_ended'):
            self.ended = True
        tr = pk.get('transition') or {}
        if tr.get('task_complete'):
            self.complete = True
        if tr.get('episode_ended'):
            self.ended = True
        for s in tr.get('steps', []) or []:
            if s.get('task_complete'):
                self.complete = True
            if s.get('episode_ended'):
                self.ended = True
        if pk.get('task_complete'):
            self.complete = True

    def step(self, rows):
        if self.ended:
            return None
        rows = [[float(np.clip(v, -1, 1)) for v in r] for r in rows]
        if self.remaining is not None:
            rows = rows[:max(0, int(self.remaining))]
            if not rows:
                self.ended = True
                return None
        r = self.ctx.call('robomme_step', actions=rows)
        self.ncalls += 1
        pk = packet(r)
        self._upd(pk)
        if self.complete:
            log('TASK COMPLETE at step', self.step_id)
        return pk

def go_to(rb, target, max_steps=80, tol=0.004, vmax=0.6, rows_per_call=1):
    """Closed-loop straight move of TCP to target xyz."""
    target = np.asarray(target, float)
    for _ in range(max_steps):
        if rb.ended:
            return False
        d = target - rb.state[:3]
        if np.linalg.norm(d) < tol:
            return True
        a = np.clip(d / 0.05, -vmax, vmax)
        rb.step([[a[0], a[1], a[2]] + rot_cmd(rb.state) + [0]] * rows_per_call)
    return np.linalg.norm(target - rb.state[:3]) < 3 * tol

def follow(rb, pts, vmax=0.6, lookahead=1):
    """Follow a dense list of waypoints, one control row per waypoint (closed loop)."""
    for i, p in enumerate(pts):
        if rb.ended:
            return
        if YAW['on']:
            q = np.asarray(pts[min(i + 2, len(pts) - 1)]); q0 = np.asarray(pts[max(i - 2, 0)])
            if np.linalg.norm((q - q0)[:2]) > 1e-4:
                YAW['dir'] = (q - q0)[:2]
        d = np.asarray(p) - rb.state[:3]
        a = np.clip(d / 0.05, -vmax, vmax)
        rb.step([[a[0], a[1], a[2]] + rot_cmd(rb.state) + [0]])

def hold(rb, target, n):
    """Hold the TCP on target for n control steps with gentle corrective deltas."""
    target = np.asarray(target, float)
    for _ in range(n):
        if rb.ended:
            return
        d = target - rb.state[:3]
        a = np.clip(d / 0.05, -0.2, 0.2)
        rb.step([[a[0], a[1], a[2]] + rot_cmd(rb.state) + [0]])

def arc_points(A, B, side, h, z, n):
    """Half-ellipse from A to B (xy), bulging toward world +x if side>0 else -x, peak offset h."""
    A = np.asarray(A, float); B = np.asarray(B, float)
    ch = B - A; L = np.linalg.norm(ch)
    u = ch / L
    nrm = np.array([-u[1], u[0]])
    if np.sign(nrm[0]) != np.sign(side):
        nrm = -nrm
    pts = []
    for i in range(1, n + 1):
        s = i / n
        th = np.pi * s
        p = A + u * (L / 2) * (1 - np.cos(th)) + nrm * h * np.sin(th)
        pts.append([p[0], p[1], z])
    return pts

def plan(pk0, mpath=None):
    seqd = pk0.get('frame_sequence') or {}
    mpath = mpath or seqd.get('manifest_path')
    man = json.load(open(os.path.join(WS, mpath)))
    frames = sorted(man['frames'], key=lambda f: f['frame_index'])
    vid = [f for f in frames if f['phase'] == 'video_demo']
    exe = [f for f in frames if f['phase'] != 'video_demo']
    S = np.array([f['state'] for f in vid], float)
    T = np.array([f.get('timestamp_s', i / 20.0) for i, f in enumerate(vid)], float)
    vt = []
    try:
        vt = detect_frame(vid[0])
    except Exception as e:
        log('video target detect failed', e)
    route = parse_route(S, T, vt if len(vt) >= 3 else None)
    log('route seq', route['seq'], 'sides', [m['side'] for m in route['moves']],
        'h', [round(m['h'], 3) for m in route['moves']], 'z', round(route['z_draw'], 4))
    ef = exe[-1] if exe else frames[-1]
    lt = detect_frame(ef)
    # tool geometry can differ between the video and the live episode: compare stick tips
    dz = 0.0
    try:
        tv = stick_tip_z(vid[0]); tl = stick_tip_z(ef)
        log('stick tip video', tv, 'live', tl, 'tcp video', round(vid[0]['state'][2], 4), 'tcp live', round(ef['state'][2], 4))
        if tv and tl:
            dz = (tl[0] - ef['state'][2]) - (tv[0] - vid[0]['state'][2])
            dz = float(np.clip(dz, -0.08, 0.12))
    except Exception as e:
        log('stick measure failed', e)
    route['z_draw_video'] = route['z_draw']
    route['z_draw'] = route['z_draw'] - dz
    route['dz'] = dz
    log('stick dz', round(dz, 4), 'z_draw live TCP', round(route['z_draw'], 4))
    log('video targets', [np.round(p, 3).tolist() for p in vt])
    log('live targets', [np.round(p, 3).tolist() for p in lt])
    # index of each route target within the video layout
    rt = [np.array(t) for t in route['targets']]
    live_pts = []
    if len(vt) == len(lt) and len(vt) >= 2:
        for t in rt:
            k = int(np.argmin([np.linalg.norm(t - v) for v in vt]))
            if np.linalg.norm(t - vt[k]) > 0.05:
                log('WARN route target far from video disc', t, vt[k])
            # same layout as the video: the demonstrated dwell point is the most accurate target
            live_pts.append(t if np.linalg.norm(t - lt[k]) < 0.015 else lt[k])
    else:
        log('WARN target count mismatch video', len(vt), 'live', len(lt))
        # fallback: best global translation aligning route targets to live discs (inlier count)
        if len(lt) >= 1:
            best = (-1, 0.0, np.zeros(2))
            for t in rt:
                for v in lt:
                    off = v - t
                    dists = [min(np.linalg.norm(r + off - w) for w in lt) for r in rt]
                    inl = sum(d < 0.03 for d in dists)
                    score = (inl, -sum(min(d, 0.03) for d in dists))
                    if score > best[:2]:
                        best = (inl, score[1], off)
            off = best[2]
            log('fallback offset', np.round(off, 3), 'inliers', best[0], '/', len(rt))
            for t in rt:
                k = int(np.argmin([np.linalg.norm(t + off - v) for v in lt]))
                if np.linalg.norm(t + off - lt[k]) < 0.04:
                    live_pts.append(lt[k])
                else:
                    live_pts.append(t + off)
        else:
            live_pts = rt
    return route, [np.asarray(p, float) for p in live_pts]

def run(ctx, speed=0.010, dwell=16, hscale=1.0, first_move=0, skip_start=False, mpath=None):
    rb = Robot(ctx)
    r0, pk0 = rb.observe()
    t0 = time.time()
    route, live = plan(pk0, mpath)
    z = route['z_draw']
    seq = route['seq']
    if not seq:
        log('no route found'); return rb
    A = live[seq[0]]
    start = rb.state[:3].copy()
    log('start state', np.round(start, 3), 'first target', np.round(A, 3))
    short = route.get('dz', 0.0) > 0.02
    hover = z + (0.16 if short else 0.08)
    if short and len(seq) > 1:
        # short stick: the hand must pass between tall blocks; align its finger axis with the path tangent
        YAW['on'] = True
        B = live[seq[1]]
        ch = B - A
        YAW['dir'] = np.array([-ch[1], ch[0]])  # arc leaves the target perpendicular to the chord
        log('short stick: yaw control on, hover', round(hover, 3))
    if not skip_start:
        go_to(rb, [A[0], A[1], hover], max_steps=150, tol=0.006)
        for _ in range(25):
            if rb.ended or np.linalg.norm(rot_cmd(rb.state)) < 0.01:
                break
            rb.step([[0, 0, 0] + rot_cmd(rb.state) + [0]])
        go_to(rb, [A[0], A[1], z], max_steps=60, tol=0.003, vmax=0.4)
        hold(rb, [A[0], A[1], z], dwell)
    log('at first target', np.round(rb.state[:3], 3), 'step', rb.step_id)
    for k, mv in enumerate(route['moves']):
        if k < first_move:
            continue
        if rb.ended:
            break
        P = live[mv['frm']]; Q = live[mv['to']]
        h = float(np.clip(mv['h'], 0.07, 0.12)) * hscale
        L = np.linalg.norm(Q - P)
        arclen = np.pi * np.sqrt((h * h + (L / 2) ** 2) / 2)
        n = max(10, int(np.ceil(arclen / speed)))
        pts = arc_points(rb.state[:2], Q, mv['side'], h, z, n)
        follow(rb, pts)
        go_to(rb, [Q[0], Q[1], z], max_steps=15, tol=0.003, vmax=0.3)
        hold(rb, [Q[0], Q[1], z], dwell)
        log('move', k, mv['frm'], '->', mv['to'], 'side', mv['side'], 'at', np.round(rb.state[:3], 3),
            'err', round(float(np.linalg.norm(rb.state[:2] - Q)), 4), 'step', rb.step_id, 't', round(time.time() - t0, 1))
    if not rb.ended and seq:
        # final target: keep holding a little longer so the last visit registers
        Q = live[seq[-1]]
        hold(rb, [Q[0], Q[1], z], 30)
    log('done: ended', rb.ended, 'complete', rb.complete, 'step', rb.step_id, 'calls', rb.ncalls)
    return rb

def main(ctx):
    os.makedirs(OUT, exist_ok=True)
    try:
        rb = run(ctx)
        if not rb.ended:
            # hold still; do not end the episode with spurious motion
            log('route finished without episode end')
    except Exception:
        log('ERROR', traceback.format_exc())
    finally:
        try:
            with open(os.path.join(OUT, 'controller_log.txt'), 'w') as f:
                f.write('\n'.join(LOG))
        except Exception:
            pass
