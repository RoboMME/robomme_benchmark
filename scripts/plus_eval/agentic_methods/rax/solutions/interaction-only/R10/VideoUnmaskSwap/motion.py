import json, math, time
import numpy as np

class Robot:
    def __init__(self, ctx, log=print):
        self.ctx = ctx; self.log = log
        self.last = None; self.ended = False; self.complete = False
        self.steps_used = 0

    def _meta(self, res):
        for b in res.get('content', []):
            if b.get('type') == 'text':
                try:
                    return json.loads(b['text'])
                except Exception:
                    pass
        return {}

    def observe(self):
        m = self._meta(self.ctx.call('robomme_observe'))
        self.last = m
        return m

    def state(self):
        return np.array(self.last['state'])

    def step(self, rows):
        rows = [[float(max(-1, min(1, v))) for v in r] for r in rows]
        if self.ended:
            return self.last
        for i in range(0, len(rows), 100):
            m = self._meta(self.ctx.call('robomme_step', actions=rows[i:i+100]))
            self.last = m
            tr = m.get('transition') or {}
            if m.get('task_complete') or tr.get('task_complete'):
                self.complete = True
            for s in (tr.get('steps') or []) + (m.get('steps') or []):
                if isinstance(s, dict) and s.get('task_complete'):
                    self.complete = True
            if m.get('episode_ended'):
                self.ended = True
                self.log(f"episode ended; complete={self.complete}")
                break
        return self.last

    def step_scene_points(self, row):
        """execute one row and return merged white world points from the reply's cameras"""
        res = self.ctx.call('robomme_step', actions=[[float(max(-1, min(1, v))) for v in row]])
        m = self._meta(res); self.last = m
        tr = m.get('transition') or {}
        if m.get('task_complete') or tr.get('task_complete'): self.complete = True
        if m.get('episode_ended'): self.ended = True
        return scene_points_from_reply(res, m)

    def remaining(self):
        try:
            return int(self.last['episode_limits']['remaining_episode_steps'])
        except Exception:
            return 10**6

def yaw_of(state):
    """gripper yaw about world z from axis-angle (pointing-down gripper)."""
    rv = np.array(state[3:6]); th = np.linalg.norm(rv)
    if th < 1e-9: R = np.eye(3)
    else:
        k = rv / th; Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        R = np.eye(3) + math.sin(th) * Kx + (1 - math.cos(th)) * Kx @ Kx
    x = R[:, 0]
    return math.atan2(x[1], x[0])

def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi

def rotvec_to_R(rv):
    rv = np.asarray(rv, float); th = np.linalg.norm(rv)
    if th < 1e-9: return np.eye(3)
    k = rv / th; Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * Kx + (1 - math.cos(th)) * Kx @ Kx

def R_to_rotvec(R):
    c = max(-1.0, min(1.0, (np.trace(R) - 1) / 2)); th = math.acos(c)
    if th < 1e-6: return np.zeros(3)
    if th > math.pi - 1e-3:
        # near pi: axis from diagonal
        A = (R + np.eye(3)) / 2; i = int(np.argmax(np.diag(A)))
        k = A[:, i] / math.sqrt(max(A[i, i], 1e-12)); return k * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return w * th

def Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

R_DOWN = None

def target_R(yaw):
    return Rz(yaw) @ R_DOWN

def rot_error(state, yaw):
    """world-frame rotation vector taking current orientation to Rz(yaw)*R_DOWN"""
    Rc = rotvec_to_R(state[3:6])
    return R_to_rotvec(target_R(yaw) @ Rc.T)

def move_to(robot, target, grip=-1.0, yaw=0.0, tol=0.004, max_iter=8, speed=1.0, rtol=0.03, max_rows=60):
    """closed-loop translate and hold absolute yaw (gripper pointing down) to world target."""
    target = np.array(target, dtype=float)
    for it in range(max_iter):
        if robot.ended: return False
        s = robot.state(); p = s[:3]
        d = target - p; dist = np.linalg.norm(d)
        er = rot_error(s, yaw); ang = np.linalg.norm(er)
        if dist < tol and ang < rtol:
            return True
        n = int(max(math.ceil(dist / (0.05 * speed)), math.ceil(ang / 0.25), 1))
        n = min(n, max_rows)
        rows = []
        for i in range(n):
            v = d / n / 0.05
            r = er / n / 0.5
            rows.append([v[0], v[1], v[2], r[0], r[1], r[2], grip])
        robot.step(rows)
        robot.step([[0, 0, 0, 0, 0, 0, grip]] * 2)
    s = robot.state()
    return np.linalg.norm(target - s[:3]) < 3 * tol and np.linalg.norm(rot_error(s, yaw)) < 3 * rtol

def init_down(state):
    global R_DOWN
    R = rotvec_to_R(state[3:6])
    # remove any yaw: keep pointing-down frame with yaw_of==0
    a = yaw_of(state)
    R_DOWN = Rz(-a) @ R

def gripper(robot, value, n=12):
    s = robot.state()
    robot.step([[0, 0, 0, 0, 0, 0, value]] * n)

def observe_images(robot):
    """returns meta, dict camera->(rgb uint8 HxWx3, depth_m HxW)"""
    import base64, io
    from PIL import Image
    res = robot.ctx.call('robomme_observe')
    meta = robot._meta(res); robot.last = meta
    out = {}
    for a in meta.get('attachments', []):
        b = res['content'][a['content_index']]
        im = Image.open(io.BytesIO(base64.b64decode(b['data'])))
        arr = np.asarray(im)
        cam = a['camera']; out.setdefault(cam, {})
        if a['kind'] == 'rgb':
            out[cam]['rgb'] = np.asarray(im.convert('RGB'))
        else:
            out[cam]['depth'] = arr.astype(np.float64) * meta['depth'][cam].get('meters_per_unit', 0.001)
    return meta, out

def decode_reply(res, meta):
    import base64, io
    from PIL import Image
    out = {}
    for a in meta.get('attachments', []):
        b = res['content'][a['content_index']]
        im = Image.open(io.BytesIO(base64.b64decode(b['data'])))
        cam = a['camera']; out.setdefault(cam, {})
        if a['kind'] == 'rgb':
            out[cam]['rgb'] = np.asarray(im.convert('RGB'))
        else:
            out[cam]['depth'] = np.asarray(im).astype(np.float64) * meta['depth'][cam].get('meters_per_unit', 0.001)
    return out

def scene_points_from_ims(meta, ims, white=True):
    from geom import backproject
    from track import white_filter
    pts = []
    for cam, d in ims.items():
        if 'depth' not in d or 'rgb' not in d: continue
        pw, ok = backproject(d['depth'], meta['camera_intrinsics'][cam], meta['camera_poses'][cam])
        sel = ok & (pw[..., 2] > 0.008) & (pw[..., 2] < 0.13) & (np.abs(pw[..., 1]) < 0.4) & (pw[..., 0] > -0.45) & (pw[..., 0] < 0.35)
        P = pw[sel]; C = d['rgb'][sel].astype(np.int16)
        pts.append(white_filter(P, C) if white else P)
    return np.concatenate(pts) if pts else np.zeros((0, 3))

def scene_points_from_reply(res, meta):
    return scene_points_from_ims(meta, decode_reply(res, meta))

def front_points(robot):
    """(meta, merged white points from all cameras, images) for the current observation"""
    res = robot.ctx.call('robomme_observe')
    meta = robot._meta(res); robot.last = meta
    ims = decode_reply(res, meta)
    return meta, scene_points_from_ims(meta, ims), ims

def grasp_yaw(cyaw, cur=0.0):
    """choose gripper yaw equivalent to container yaw mod 90deg closest to cur"""
    best = None
    for k in range(-4, 5):
        y = cyaw + k * math.pi / 2
        if abs(y) > math.radians(100): continue
        if best is None or abs(wrap(y - cur)) < abs(wrap(best - cur)): best = y
    return best

def place_down(robot, xy, yaw, z_release=0.05, z_up=0.16):
    ok = move_to(robot, [xy[0], xy[1], z_release + 0.03], grip=1, yaw=yaw)
    ok = move_to(robot, [xy[0], xy[1], z_release], grip=1, yaw=yaw, max_iter=4) and ok
    gripper(robot, -1, 10)
    move_to(robot, [xy[0], xy[1], z_up], grip=-1, yaw=yaw)
    return ok

def pick_container(robot, xy, log=print, z_grasp=0.04, z_lift=0.18, search_r=0.05):
    """locate container near xy in current front obs, grasp it and lift. returns (ok, info)"""
    from track import container_geometry
    s = robot.state()
    if s[2] < 0.14:
        move_to(robot, [s[0], s[1], 0.16], grip=-1, yaw=yaw_of(s) if R_DOWN is not None else 0.0)
    meta, P, _ = front_points(robot)
    g = container_geometry(P, np.array(xy, float)[None], radius=search_r)[0]
    c = np.array(g['xy']); y = grasp_yaw(g['yaw'], 0.0)
    log(f"pick: target {np.round(xy,3).tolist()} refined {np.round(c,3).tolist()} n={g['n']} yaw={y:.3f}")
    move_to(robot, [c[0], c[1], 0.16], grip=-1, yaw=y)
    # re-refine with the wrist camera now looking down near the target
    meta, P, _ = front_points(robot)
    g2 = container_geometry(P, c[None], radius=0.04)[0]
    if g2['n'] >= 30 and np.linalg.norm(np.array(g2['xy']) - c) < 0.03:
        c = np.array(g2['xy']); y = grasp_yaw(g2['yaw'], y)
        log(f"pick: re-refined {np.round(c,3).tolist()} n={g2['n']} yaw={y:.3f}")
    move_to(robot, [c[0], c[1], 0.10], grip=-1, yaw=y)
    move_to(robot, [c[0], c[1], z_grasp], grip=-1, yaw=y, max_iter=5)
    gripper(robot, 1, 15)
    move_to(robot, [c[0], c[1], z_lift], grip=1, yaw=y)
    st = robot.state(); gap = st[6] + st[7]
    ok = gap > 0.01 and st[2] > z_lift - 0.03
    log(f"pick: gap={gap:.4f} z={st[2]:.3f} ok={ok}")
    return ok, {'xy': c.tolist(), 'yaw': y, 'gap': float(gap)}

def revealed_color(robot, xy, r=0.035):
    from track import classify_rgb, COLORS
    from geom import backproject
    meta, ims = observe_images(robot)
    pw, ok = backproject(ims['front']['depth'], meta['camera_intrinsics']['front'], meta['camera_poses']['front'])
    sel = ok & (pw[..., 2] > 0.003) & (pw[..., 2] < 0.05) & (np.linalg.norm(pw[..., :2] - np.array(xy), axis=-1) < r)
    lab = classify_rgb(ims['front']['rgb'][sel].astype(np.int16))
    cnt = {COLORS[k]: int((lab == k).sum()) for k in range(6)}
    best = max(cnt, key=cnt.get)
    return (best if cnt[best] >= 5 else None), cnt
