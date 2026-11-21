"""Robot helper library for PickXtimes (observation parsing, deprojection, motion)."""
import base64, io, json, math, time
import numpy as np
from PIL import Image

LOG = []

def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)

def _parse(reply):
    meta = None
    imgs = []
    for b in reply.get('content', []):
        if b.get('type') == 'text' and meta is None:
            try:
                meta = json.loads(b['text'])
            except Exception:
                pass
        elif b.get('type') == 'image':
            imgs.append(b)
    return meta, imgs

def _decode(block):
    return Image.open(io.BytesIO(base64.b64decode(block['data'])))

class Obs:
    def __init__(self, meta, imgs, all_blocks):
        self.meta = meta
        self.state = np.array(meta['state'], dtype=float)
        self.step_id = meta.get('step_id')
        self.rgb = {}
        self.depth = {}
        for att in meta.get('attachments', []):
            blk = all_blocks[att['content_index']]
            im = _decode(blk)
            if att['kind'] == 'rgb':
                self.rgb[att['camera']] = np.array(im.convert('RGB'))
            else:
                d = np.array(im).astype(np.float32)
                mpu = meta['depth'][att['camera']].get('meters_per_unit', 0.001)
                self.depth[att['camera']] = d * mpu
        self.K = {c: np.array(v['matrix']) for c, v in meta['camera_intrinsics'].items()}
        self.pose = meta['camera_poses']
        self.ended = bool(meta.get('episode_ended', False))
        self.limits = meta.get('episode_limits', {})

def make_obs(reply):
    meta, _ = _parse(reply)
    return Obs(meta, None, reply['content'])

def observe(ctx):
    return make_obs(ctx.call('robomme_observe'))

def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])

def deproject(obs, cam, u, v, d=None):
    """pixel (u=col, v=row) -> world point. OpenGL camera: looks -Z, +Y up."""
    if d is None:
        d = float(obs.depth[cam][int(round(v)), int(round(u))])
    K = obs.K[cam]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    xc = (u - cx) / fx * d   # pixel-centre convention, verify
    yc = (v - cy) / fy * d
    p_cam = np.array([xc, -yc, -d])
    R = quat_to_R(obs.pose[cam]['quaternion_wxyz'])
    t = np.array(obs.pose[cam]['position_m'])
    return R @ p_cam + t

def deproject_many(obs, cam, u, v, d):
    K = obs.K[cam]
    pc = np.stack([(u - K[0, 2]) / K[0, 0] * d, -(v - K[1, 2]) / K[1, 1] * d, -d], 1)
    R = quat_to_R(obs.pose[cam]['quaternion_wxyz'])
    t = np.array(obs.pose[cam]['position_m'])
    return pc @ R.T + t

def project(obs, cam, pw):
    R = quat_to_R(obs.pose[cam]['quaternion_wxyz'])
    t = np.array(obs.pose[cam]['position_m'])
    pc = R.T @ (np.asarray(pw) - t)
    d = -pc[2]
    K = obs.K[cam]
    u = pc[0] / d * K[0, 0] + K[0, 2]
    v = -pc[1] / d * K[1, 1] + K[1, 2]
    return u, v, d

# ---------------- colour segmentation ----------------
def color_mask(rgb, color):
    r = rgb[..., 0].astype(int); g = rgb[..., 1].astype(int); b = rgb[..., 2].astype(int)
    # cubes are saturated primaries (e.g. 231,3,3); strict ratios reject wood/table textures
    if color == 'green':
        return (g > 80) & (r < 0.3 * g) & (b < 0.3 * g)
    if color == 'red':
        return (r > 80) & (g < 0.3 * r) & (b < 0.3 * r)
    if color == 'blue':
        return (b > 80) & (r < 0.3 * b) & (g < 0.3 * b)
    if color == 'yellow':
        return (r > 80) & (g > 80) & (np.abs(r - g) < 0.2 * r) & (b < 0.3 * r)
    if color == 'purple':  # target rings (magenta/purple)
        return (r > 120) & (b > 120) & (g < r - 40) & (g < b - 40)
    raise ValueError(color)

def blobs(mask, min_px=4):
    """connected components (4-neigh) -> list of (n, cu, cv, pixels)"""
    H, W = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out = []
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys, xs):
        if seen[y0, x0]:
            continue
        stack = [(y0, x0)]; seen[y0, x0] = True; pix = []
        while stack:
            y, x = stack.pop(); pix.append((y, x))
            for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                yy, xx = y+dy, x+dx
                if 0 <= yy < H and 0 <= xx < W and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True; stack.append((yy, xx))
        if len(pix) >= min_px:
            p = np.array(pix)
            out.append((len(pix), p[:, 1].mean(), p[:, 0].mean(), p))
    out.sort(key=lambda b: -b[0])
    return out

def locate(obs, cam, color, min_px=4, zmin=0.012):
    """Largest blob of colour whose median height exceeds zmin (a cube, not table texture).
    Return (world_point_median, npx, (u,v), pts) or None."""
    bl = blobs(color_mask(obs.rgb[cam], color), min_px)
    for n, cu, cv, pix in bl[:6]:
        d = obs.depth[cam][pix[:, 0], pix[:, 1]]
        ok = d > 0
        if ok.sum() < max(3, min_px // 2):
            continue
        pts = deproject_many(obs, cam, pix[ok][:, 1].astype(float), pix[ok][:, 0].astype(float), d[ok])
        med = np.median(pts, axis=0)
        if med[2] < zmin:
            continue
        return med, n, (cu, cv), pts
    return None

# ---------------- motion ----------------
class Robot:
    def __init__(self, ctx):
        self.ctx = ctx
        self.last = None
        self.ended = False
        self.complete = False
        self.grip = -1.0
        self.steps = 0
        self.on_obs = None     # callback(obs) after every reply

    def step(self, rows):
        if self.ended:
            return self.last
        rows = [[float(np.clip(x, -1, 1)) for x in r] for r in rows]
        rem = None
        if self.last is not None:
            rem = self.last.limits.get('remaining_episode_steps')
        if rem is not None:
            rows = rows[:max(0, rem)]
            if not rows:
                self.ended = True
                return self.last
        out = None
        for i in range(0, len(rows), 100):
            reply = self.ctx.call('robomme_step', actions=rows[i:i+100])
            out = make_obs(reply)
            self.steps += len(rows[i:i+100])
            self.last = out
            if self.on_obs is not None:
                try:
                    self.on_obs(out)
                except Exception as e:
                    log('[on_obs error]', repr(e))
            m = out.meta
            if m.get('task_complete') or (m.get('transition') or {}).get('task_complete'):
                self.complete = True
            if out.ended:
                self.ended = True
                log('EPISODE ENDED', json.dumps({k: m.get(k) for k in ('task_complete','episode_ended','step_id')}))
                break
        return out

    def observe(self):
        self.last = observe(self.ctx)
        if self.last.ended:
            self.ended = True
        return self.last

    def hold(self, n, grip=None):
        g = self.grip if grip is None else grip
        self.grip = g
        return self.step([[0, 0, 0, 0, 0, 0, g]] * n)

    def move_to(self, target, grip=None, tol=0.004, max_iter=12, boost=2.0, vmax=0.012, nmax=15):
        """Move TCP to world target with batched rows; re-measure and correct.
        Measured response: commanded per-step delta d gives ~0.4-0.65 d actual (lag)."""
        g = self.grip if grip is None else grip
        self.grip = g
        target = np.asarray(target, float)
        o = self.last if self.last is not None else self.observe()
        for it in range(max_iter):
            pos = o.state[:3]
            err = target - pos
            dist = np.linalg.norm(err)
            if dist < tol:
                break
            n = int(np.clip(math.ceil(dist / vmax), 1, nmax))
            d = err / n * boost
            m = np.linalg.norm(d)
            if m > 0.04:
                d = d * 0.04 / m
            d = d / 0.05
            rows = [[d[0], d[1], d[2], 0, 0, 0, g]] * n
            o = self.step(rows)
            if self.ended:
                break
        return o

def square_angle(pix_uv):
    """pix_uv: Nx2 (u,v). Angle (deg, in [-45,45)) of square edges vs image u axis, via min-area bbox."""
    p = np.asarray(pix_uv, float)
    p = p - p.mean(0)
    best = None
    for a in np.arange(0, 90, 1.0):
        t = math.radians(a)
        c, s = math.cos(t), math.sin(t)
        x = p[:, 0] * c + p[:, 1] * s
        y = -p[:, 0] * s + p[:, 1] * c
        area = (x.max() - x.min()) * (y.max() - y.min())
        if best is None or area < best[0]:
            best = (area, a)
    a = best[1]
    if a >= 45:
        a -= 90
    return a

def blob_angle(obs, cam, color):
    bl = blobs(color_mask(obs.rgb[cam], color), 4)
    if not bl:
        return None
    pix = bl[0][3]
    return square_angle(np.stack([pix[:, 1], pix[:, 0]], 1))

def yaw_of(state):
    """Rotation angle about world z of the gripper relative to the reset pose (axis-angle [pi,0,0])."""
    rv = np.asarray(state[3:6], float)
    th = np.linalg.norm(rv)
    if th < 1e-9:
        Rm = np.eye(3)
    else:
        k = rv / th
        Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        Rm = np.eye(3) + math.sin(th) * Kx + (1 - math.cos(th)) * Kx @ Kx
    # gripper x axis in world
    x = Rm[:, 0]
    return math.degrees(math.atan2(x[1], x[0]))

def locate_top(obs, cam, color, min_px=4):
    """Top-face centre of the largest blob of colour (world), plus top z and pixel count."""
    r = locate(obs, cam, color, min_px)
    if r is None:
        return None
    _, n, uv, pts = r
    zt = np.percentile(pts[:, 2], 90)
    top = pts[pts[:, 2] > zt - 0.006]
    c = np.median(top, axis=0)
    return c, n, uv, pts

def _wrap(a):
    return (a + 180) % 360 - 180

def rotvec_to_R(rv):
    rv = np.asarray(rv, float)
    th = np.linalg.norm(rv)
    if th < 1e-12:
        return np.eye(3)
    k = rv / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * Kx + (1 - math.cos(th)) * Kx @ Kx

def R_to_rotvec(Rm):
    c = (np.trace(Rm) - 1) / 2
    c = float(np.clip(c, -1, 1))
    th = math.acos(c)
    if th < 1e-9:
        return np.zeros(3)
    if th > math.pi - 1e-4:
        # axis from the largest diagonal
        M = (Rm + np.eye(3)) / 2
        i = int(np.argmax(np.diag(M)))
        k = M[:, i] / math.sqrt(max(M[i, i], 1e-12))
        return k * th
    w = np.array([Rm[2, 1] - Rm[1, 2], Rm[0, 2] - Rm[2, 0], Rm[1, 0] - Rm[0, 1]]) / (2 * math.sin(th))
    return w * th

R_DOWN = np.diag([1.0, -1.0, -1.0])

def down_R(yaw_deg):
    t = math.radians(yaw_deg)
    Rz = np.array([[math.cos(t), -math.sin(t), 0], [math.sin(t), math.cos(t), 0], [0, 0, 1]])
    return Rz @ R_DOWN

def orient_err(state, yaw_deg):
    """World-frame rotation vector taking current orientation to down-pointing at yaw."""
    return R_to_rotvec(down_R(yaw_deg) @ rotvec_to_R(state[3:6]).T)

def rotate_yaw(robot, target_deg, tol=1.5, max_iter=10):
    """Rotate to a down-pointing gripper with the given yaw (also removes tilt)."""
    o = robot.last if robot.last is not None else robot.observe()
    for it in range(max_iter):
        e = orient_err(o.state, target_deg)
        ang = math.degrees(np.linalg.norm(e))
        if ang < tol:
            break
        n = int(np.clip(math.ceil(ang / 4.0), 1, 15))
        d = e / n * 2.0 / 0.5
        m = np.abs(d).max()
        if m > 0.8:
            d = d * 0.8 / m
        o = robot.step([[0, 0, 0, d[0], d[1], d[2], robot.grip]] * n)
        if robot.ended:
            break
    return o

# ---------------- task primitives ----------------
HOVER_Z = 0.15        # tested ep1
GRASP_Z = 0.020       # TCP z for grasp of 4cm cube (centre ~0.02). tested ep1 (0.0193 reached)
PLACE_Z = 0.036       # TCP z when releasing on target (cube bottom just above target)
CLOSED_MIN = 0.012    # per-finger: below this = missed (closed on nothing gives ~0?)
CLOSED_MAX = 0.030

def _wrap90(a):
    return (a + 45) % 90 - 45

def finger_dir(yaw_deg):
    t = math.radians(yaw_deg)
    return np.array([-math.sin(t), math.cos(t)])

def clearance(yaw_deg, cube_xy, others):
    """Smallest along-finger distance to another cube lying in the finger corridor (bigger is better)."""
    f = finger_dir(yaw_deg)
    best = 1.0
    for o in others or []:
        r = np.asarray(o[:2], float) - np.asarray(cube_xy[:2], float)
        along = abs(float(r @ f))
        across = abs(float(r[0] * f[1] - r[1] * f[0]))
        if across < 0.05:
            best = min(best, along)
    return best

def choose_yaw(cur_yaw, img_angle, cube_xy, others):
    ya = _wrap90(cur_yaw - img_angle)        # in [-45,45)
    yb = ya - 90 if ya > 0 else ya + 90      # in (-90, 90]
    ca, cb = clearance(ya, cube_xy, others), clearance(yb, cube_xy, others)
    if ca >= 0.09 or ca >= cb:
        return ya, ca, cb
    return yb, ca, cb

def grasp_cube(R, color, xy_hint, tag='', others=None):
    """Hover over xy_hint, locate cube top with wrist, align yaw, descend, close, lift. Returns dict."""
    R.move_to([xy_hint[0], xy_hint[1], HOVER_Z], grip=-1)
    o = R.hold(2, grip=-1)
    c = locate_top(o, 'wrist', color)
    if c is None:
        return {'ok': False, 'why': 'cube not seen in wrist', 'step': o.step_id}
    a = blob_angle(o, 'wrist', color)
    cur = yaw_of(o.state)
    if a is None:
        a = 0.0
    yt, ca, cb = choose_yaw(cur, a, c[0][:2], [p for p in (others or [])
                                              if np.linalg.norm(np.asarray(p[:2]) - c[0][:2]) > 0.02])
    rotate_yaw(R, yt)
    o = R.hold(2)
    c2 = locate_top(o, 'wrist', color)
    if c2 is not None and np.linalg.norm(c2[0][:2] - c[0][:2]) < 0.05:
        c = c2
    cx, cy, ctop = c[0]
    R.move_to([cx, cy, 0.08], grip=-1)
    o = R.last
    c3 = locate_top(o, 'wrist', color)
    if c3 is not None and np.linalg.norm(c3[0][:2] - np.array([cx, cy])) < 0.03:
        cx, cy, ctop = c3[0]
    gz = float(np.clip(ctop - 0.02, 0.012, 0.03))
    R.move_to([cx, cy, gz], grip=-1, tol=0.003)
    o = R.hold(2, grip=-1)
    o = R.hold(12, grip=1)
    f = o.state[6:8]
    lift_step = o.step_id
    R.move_to([cx, cy, HOVER_Z], grip=1)
    o = R.hold(2, grip=1)
    f2 = o.state[6:8]
    ok = (f2.min() > CLOSED_MIN) and (f2.max() < CLOSED_MAX)
    log(f'[grasp{tag}] cube=({cx:.4f},{cy:.4f},{ctop:.4f}) img_angle={a:.1f} yaw {cur:.1f}->{yt:.1f} '
        f'clear=({ca:.3f},{cb:.3f}) fingers_closed={f.round(4)} after_lift={f2.round(4)} ok={ok} step={o.step_id}')
    return {'ok': bool(ok), 'xy': (cx, cy), 'fingers': f2.tolist(), 'step': o.step_id, 'lift_step': lift_step,
            'why': None if ok else 'fingers out of range'}

def place_cube(R, color, target_xy, tag=''):
    tx, ty = target_xy
    R.move_to([tx, ty, HOVER_Z], grip=1)
    R.move_to([tx, ty, PLACE_Z], grip=1, tol=0.003)
    o = R.hold(2, grip=1)
    o = R.hold(8, grip=-1)
    R.move_to([tx, ty, HOVER_Z], grip=-1)
    o = R.hold(2, grip=-1)
    c = locate_top(o, 'wrist', color)
    if c is None:
        log(f'[place{tag}] cube not seen after release step={o.step_id}')
        return {'ok': False, 'why': 'not seen', 'step': o.step_id}
    d = float(np.linalg.norm(c[0][:2] - np.array(target_xy)))
    ok = d < 0.03 and c[0][2] < 0.06
    log(f'[place{tag}] cube=({c[0][0]:.4f},{c[0][1]:.4f},{c[0][2]:.4f}) dist_to_target={d:.4f} ok={ok} step={o.step_id}')
    return {'ok': bool(ok), 'dist': d, 'cube': c[0].tolist(), 'step': o.step_id}
