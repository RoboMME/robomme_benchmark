import json, time, numpy as np
from scipy.spatial.transform import Rotation as Rot

R0 = Rot.from_rotvec([np.pi, 0, 0])   # gripper pointing down, yaw 0

class Robot:
    def __init__(self, ctx, log=print):
        self.ctx = ctx; self.log = log
        self.last = None; self.ended = False; self.complete = None

    def _parse(self, res):
        txt = next(b['text'] for b in res['content'] if b.get('type') == 'text')
        return json.loads(txt)

    def observe(self):
        p = self._parse(self.ctx.call('robomme_observe'))
        self.last = p
        if p.get('episode_ended'): self.ended = True
        return p

    def state(self):
        return np.array(self.last['state'])

    def remaining(self):
        return self.last.get('episode_limits', {}).get('remaining_episode_steps', 0)

    def step(self, rows):
        if self.ended: return None
        rows = [[float(np.clip(v, -1, 1)) for v in r] for r in rows]
        res = self.ctx.call('robomme_step', actions=rows)
        self.raw = res
        p = self._parse(res)
        self.last = p
        self.nsteps = getattr(self, 'nsteps', 0) + len(rows)
        tr = p.get('transition') or {}
        for s in (tr.get('steps') or []):
            if isinstance(s, dict) and s.get('task_complete'):
                self.complete = True
            if isinstance(s, dict) and s.get('episode_ended'):
                self.ended = True
        if tr.get('task_complete') or p.get('task_complete'): self.complete = True
        if tr.get('episode_ended') or p.get('episode_ended'):
            self.ended = True
        if self.ended:
            self.log(f"episode ended at step {p.get('step_id')}; task_complete={tr.get('task_complete')} flag={self.complete}")
        return p

    def move_to(self, xyz, yaw=None, grip=-1.0, tol=0.004, max_iter=12, vmax=0.5, chunk=4):
        """Move TCP to xyz (world); yaw = gripper yaw rad about world z (R = Rz(yaw) Rx(pi))."""
        xyz = np.asarray(xyz, float)
        Rdes = None if yaw is None else Rot.from_rotvec([0, 0, yaw]) * R0
        for it in range(max_iter):
            if self.ended: return False
            s = self.state()
            dp = xyz - s[:3]
            rv = np.zeros(3)
            if Rdes is not None:
                rv = (Rdes * Rot.from_rotvec(s[3:6]).inv()).as_rotvec()
            if np.linalg.norm(dp) < tol and np.linalg.norm(rv) < 0.03:
                return True
            # steps needed
            n = int(np.ceil(max(np.abs(dp).max() / (0.05 * vmax), np.abs(rv).max() / (0.5 * 0.6), 1)))
            k = min(n, chunk)
            rows = []
            for _ in range(k):
                d = dp / n; r = rv / n
                rows.append(list(d / 0.05) + list(r / 0.5) + [grip])
            self.step(rows)
        return False

    def grip(self, g, n=8):
        s = self.state()
        return self.step([[0, 0, 0, 0, 0, 0, g]] * n)

def grasp_yaw(box_yaw, cur_yaw=0.0):
    """Choose gripper yaw aligned with box faces closest to cur_yaw."""
    cands = [box_yaw + k * np.pi / 2 for k in range(-4, 5)]
    return min(cands, key=lambda c: abs(c - cur_yaw))

def cur_yaw(state):
    R = Rot.from_rotvec(state[3:6]) * R0.inv()
    return R.as_rotvec()[2]

def observe_full(ctx, res=None):
    """Return (packet, {(cam,kind): ndarray}); decode `res` if given, else observe."""
    import base64, io
    from PIL import Image
    if res is None:
        res = ctx.call('robomme_observe')
    blocks = res['content']
    p = json.loads(next(b['text'] for b in blocks if b.get('type') == 'text'))
    imgs = {}
    for a in p.get('attachments', []):
        b = blocks[a['content_index']]
        im = Image.open(io.BytesIO(base64.b64decode(b['data'])))
        arr = np.asarray(im)
        if a['kind'] == 'depth':
            arr = arr.astype(np.float64) * p['depth'][a['camera']].get('meters_per_unit', 0.001)
        else:
            arr = np.asarray(im.convert('RGB'))
        imgs[(a['camera'], a['kind'])] = arr
    return p, imgs

def wrist_boxes(p, imgs, zlo=0.062, zhi=0.085, cell=0.003):
    """Detect container top faces in wrist RGB-D -> list of (center_xy, yaw, wh, n)."""
    from geom import deproject
    from track import box_yaw_width
    from scipy import ndimage
    W = deproject(imgs[('wrist', 'depth')], p['camera_intrinsics']['wrist']['matrix'], p['camera_poses']['wrist'])
    z = W[..., 2]
    sel = (z > zlo) & (z < zhi) & np.isfinite(z)
    P = W[sel][:, :2]
    if len(P) < 30: return []
    x0, y0 = P.min(0) - 0.01
    gi = ((P[:, 0] - x0) / cell).astype(int); gj = ((P[:, 1] - y0) / cell).astype(int)
    g = np.zeros((gi.max() + 2, gj.max() + 2), bool); g[gi, gj] = True
    lab, n = ndimage.label(g)
    out = []
    pl = lab[gi, gj]
    for k in range(1, n + 1):
        m = pl == k
        if m.sum() < 60: continue
        Q = P[m]; yaw, wh = box_yaw_width(Q)
        out.append((Q.mean(0), yaw, wh, int(m.sum())))
    return out

def choose_yaw(box_yaw, cur, xy, others, reach=0.05):
    """Among face-aligned yaws (box_yaw + k*90deg) pick one whose finger positions
    clear the other containers best; prefer small rotation from cur when clearance is ample."""
    best = None
    for k in range(-4, 5):
        y = box_yaw + k * np.pi / 2
        if abs(y - cur) > np.pi / 2 + 1e-6: continue
        d = np.array([np.sin(y), -np.cos(y)])   # closing direction (gripper y axis)
        fingers = [xy + reach * d, xy - reach * d]
        clr = min([np.linalg.norm(f - o) for f in fingers for o in others] or [1.0])
        score = min(clr, 0.075) - 0.01 * abs(y - cur)
        if best is None or score > best[0]:
            best = (score, y, clr)
    return best[1], best[2]

def pick_container(rb, ctx, xy, log=print, zg=0.045, lift=0.17, place_back=True, max_snap=0.035, others=()):
    """Approach container near xy, refine with wrist depth, grasp, lift, place back, release.
    Returns dict with result info."""
    xy = np.asarray(xy, float)
    info = {'target': xy.tolist()}
    s = rb.state()
    if s[2] < 0.14:
        rb.move_to([s[0], s[1], lift], yaw=None, grip=-1)
    rb.move_to([xy[0], xy[1], lift], yaw=cur_yaw(rb.state()), grip=-1)
    yaw = cur_yaw(rb.state())
    for r in range(2):
        p, imgs = observe_full(ctx)
        bx = wrist_boxes(p, imgs)
        full = [b for b in bx if b[3] >= 500 and min(b[2]) > 0.03] or bx
        if not full:
            log('no wrist boxes; using tracked xy'); break
        best = min(full, key=lambda b: np.linalg.norm(b[0] - xy))
        dist = np.linalg.norm(best[0] - xy)
        log(f'  wrist refine {r}: box {np.round(best[0],3)} yaw {np.rad2deg(best[1]):.0f} n={best[3]} dist={dist:.3f}')
        if dist > max_snap:
            log('  refine too far; keep tracked xy'); break
        xy = best[0]
        oth = [b[0] for b in bx if np.linalg.norm(b[0] - xy) > 0.03] + [o for o in others if np.linalg.norm(o - xy) > 0.03]
        yaw, clr = choose_yaw(best[1], cur_yaw(rb.state()), xy, oth)
        log(f'  yaw {np.rad2deg(yaw):.0f} finger clearance {clr:.3f}')
        rb.move_to([xy[0], xy[1], lift - 0.02 * (r + 1)], yaw=yaw, grip=-1)
    info['grasp_xy'] = xy.tolist(); info['yaw'] = float(yaw)
    rb.move_to([xy[0], xy[1], 0.11], yaw=yaw, grip=-1)
    rb.move_to([xy[0], xy[1], zg], yaw=yaw, grip=-1, vmax=0.3)
    info['down'] = rb.state()[:3].tolist()
    rb.grip(1, 10)
    rb.move_to([xy[0], xy[1], lift], yaw=yaw, grip=1, vmax=0.3)
    s = rb.state(); info['finger'] = float(s[6] + s[7]); info['lift_z'] = float(s[2])
    info['held'] = bool(0.035 < s[6] + s[7] < 0.075)
    log(f'  lifted: opening {info["finger"]:.3f} z {s[2]:.3f} held={info["held"]} ended={rb.ended} complete={rb.complete}')
    if rb.ended: return info
    for _ in range(4): rb.step([[0, 0, 0, 0, 0, 0, 1]] * 2)   # hold briefly so lift registers
    if place_back and not rb.ended:
        rb.move_to([xy[0], xy[1], zg + 0.01], yaw=yaw, grip=1, vmax=0.3)
        rb.grip(-1, 8)
        s = rb.state(); rb.move_to([s[0], s[1], lift], yaw=yaw, grip=-1)
    return info
