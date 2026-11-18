import json, numpy as np
from pathlib import Path
from PIL import Image

WS = Path('/workspace')

def meta(reply):
    for b in reply.get('content', []):
        if b.get('type') == 'text':
            try:
                return json.loads(b['text'])
            except Exception:
                pass
    return {}

class Robot:
    def __init__(self, ctx, log=print):
        self.ctx, self.log = ctx, log
        self.last = None
        self.ended = False
        self.step_id = 0
        self.frames = []  # list of (step_id, manifest frame dict) from motion sequences

    def observe(self):
        m = meta(self.ctx.call('robomme_observe'))
        self._upd(m)
        return m

    def _upd(self, m):
        if not m:
            return
        self.last = m
        self.ended = bool(m.get('episode_ended', self.ended))
        self.step_id = m.get('step_id', self.step_id)

    @property
    def state(self):
        return np.array(self.last['state'])

    def step(self, rows):
        rows = [[float(np.clip(x, -1, 1)) for x in r] for r in rows]
        out = None
        for i in range(0, len(rows), 100):
            if self.ended:
                break
            m = meta(self.ctx.call('robomme_step', actions=rows[i:i+100]))
            self._upd(m)
            self._collect(m)
            out = m
        return out

    def _collect(self, m):
        fs = m.get('frame_sequence') or {}
        mp = fs.get('manifest_path')
        if not mp:
            return
        try:
            man = json.loads((WS / mp).read_text())
            for f in man.get('frames', []):
                self.frames.append(f)
        except Exception as e:
            self.log(f'frame manifest read failed: {e}')

def load_img(path):
    return np.array(Image.open(WS / path).convert('RGB'))

def load_depth(path):
    return np.array(Image.open(WS / path)).astype(float) * 0.001

def frame_files(f):
    return {(x['camera'], x['kind']): x['path'] for x in f.get('files', [])}

def obs_images(m, cam):
    """rgb, depth, K, pose for current observation's cam using frame_sequence last frame."""
    fs = m.get('frame_sequence') or {}
    man = json.loads((WS / fs['manifest_path']).read_text())
    f = man['frames'][-1]
    files = frame_files(f)
    rgb = load_img(files[(cam, 'rgb')])
    dep = load_depth(files[(cam, 'depth')]) if (cam, 'depth') in files else None
    K = f['camera_intrinsics'][cam]['matrix'] if 'camera_intrinsics' in f else m['camera_intrinsics'][cam]['matrix']
    pose = f['camera_poses'][cam] if 'camera_poses' in f else m['camera_poses'][cam]
    return rgb, dep, K, pose

GRIP_OPEN, GRIP_CLOSE = -1.0, 1.0

def move_rows(cur, target, grip, vmax=0.02):
    """Rows moving from cur toward target with per-step translation <= vmax metres."""
    d = np.asarray(target) - np.asarray(cur)
    dist = np.linalg.norm(d)
    n = max(1, int(np.ceil(dist / vmax)))
    step = d / n / 0.05
    return [[step[0], step[1], step[2], 0, 0, 0, grip] for _ in range(n)]

def goto(robot, target, grip, vmax=0.02, tol=0.006, max_iter=4, settle=3):
    for it in range(max_iter):
        cur = robot.state[:3]
        if np.linalg.norm(np.asarray(target) - cur) < tol:
            break
        rows = move_rows(cur, target, grip, vmax) + [[0, 0, 0, 0, 0, 0, grip]] * settle
        robot.step(rows)
        if robot.ended:
            break
    return np.linalg.norm(np.asarray(target) - robot.state[:3])

GAIN = 0.35  # measured: arm achieves ~35% of commanded per-step delta (ep1)

def servo(robot, target, grip, tol=0.004, chunk=6, vmax=1.0, max_chunks=12, rot=(0, 0, 0)):
    target = np.asarray(target, float)
    for _ in range(max_chunks):
        e = target - robot.state[:3]
        if np.linalg.norm(e) < tol or robot.ended:
            break
        a = e / (chunk * GAIN * 0.05)
        n = np.max(np.abs(a))
        if n > vmax:
            a = a * vmax / n
        robot.step([[a[0], a[1], a[2], rot[0], rot[1], rot[2], grip]] * chunk)
    return float(np.linalg.norm(target - robot.state[:3]))

import base64, io
def reply_images(reply):
    """dict (camera, kind) -> np array from a step/observe reply's content blocks."""
    m = meta(reply)
    blocks = reply.get('content', [])
    out = {}
    for a in m.get('attachments', []):
        b = blocks[a['content_index']]
        im = Image.open(io.BytesIO(base64.b64decode(b['data'])))
        if a['kind'] == 'rgb':
            out[(a['camera'], 'rgb')] = np.array(im.convert('RGB'))
        else:
            out[(a['camera'], 'depth')] = np.array(im).astype(float) * m['depth'][a['camera']].get('meters_per_unit', 0.001)
    return m, out

def rotvec_to_R(v):
    v = np.asarray(v, float); th = np.linalg.norm(v)
    if th < 1e-12:
        return np.eye(3)
    k = v / th; Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * Kx @ Kx

def R_to_rotvec(R):
    c = np.clip((np.trace(R) - 1) / 2, -1, 1); th = np.arccos(c)
    if th < 1e-9:
        return np.zeros(3)
    if th > np.pi - 1e-4:
        # axis from the symmetric part
        A = (R + np.eye(3)) / 2
        k = np.sqrt(np.clip(np.diag(A), 0, None))
        i = int(np.argmax(k)); k = A[:, i] / np.sqrt(max(A[i, i], 1e-12))
        return k / np.linalg.norm(k) * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(th))
    return w * th

def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def finger_yaw(state):
    R = rotvec_to_R(state[3:6])
    return float(np.arctan2(R[1, 1], R[0, 1]))

_BASE = rotvec_to_R([np.pi, 0, 0])
_Y0 = float(np.arctan2(_BASE[1, 1], _BASE[0, 1]))

def target_rot(yaw):
    """Gripper pointing down with finger axis at world yaw `yaw` (rad)."""
    return Rz(yaw - _Y0) @ _BASE

def rot_err(state, yaw):
    """World-frame rotation vector taking the current orientation to the target."""
    cur = rotvec_to_R(state[3:6])
    return R_to_rotvec(target_rot(yaw) @ cur.T)

def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def servo_pose(robot, target, grip, yaw=None, tol=0.004, rtol=0.03, chunk=5, vmax=1.0, max_chunks=15):
    target = np.asarray(target, float)
    for _ in range(max_chunks):
        s = robot.state
        e = target - s[:3]
        re = rot_err(s, yaw if yaw is not None else finger_yaw(s)) if yaw is not None else np.zeros(3)
        if (np.linalg.norm(e) < tol and np.linalg.norm(re) < rtol) or robot.ended:
            break
        a = e / (chunk * GAIN * 0.05)
        n = np.max(np.abs(a))
        if n > vmax:
            a = a * vmax / n
        ra = np.clip(re / (chunk * 0.5 * 0.5), -0.6, 0.6)
        robot.step([[a[0], a[1], a[2], ra[0], ra[1], ra[2], grip]] * chunk)
    return float(np.linalg.norm(target - robot.state[:3]))
