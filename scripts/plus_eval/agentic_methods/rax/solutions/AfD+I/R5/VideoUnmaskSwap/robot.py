import json, base64, io, time
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as Rot

LOG = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); LOG.append(s)

class Robot:
    def __init__(self, ctx):
        self.ctx = ctx; self.ended = False; self.complete = False; self.last = None

    def _parse(self, res):
        blocks = res.get('content', [])
        meta = json.loads(blocks[0]['text'])
        imgs = {}
        for a in meta.get('attachments', []):
            b = blocks[a['content_index']]
            if b.get('type') == 'image':
                im = Image.open(io.BytesIO(base64.b64decode(b['data'])))
                imgs[(a['camera'], a['kind'])] = np.asarray(im)
        meta['_imgs'] = imgs
        return meta

    def observe(self):
        m = self._parse(self.ctx.call('robomme_observe'))
        self.last = m; self._check(m); return m

    def _check(self, m):
        if m.get('episode_ended'): self.ended = True
        for s in m.get('steps', []) or []:
            if isinstance(s, dict):
                if s.get('episode_ended'): self.ended = True
                if s.get('task_complete'): self.complete = True
        if m.get('task_complete'): self.complete = True

    def step(self, rows):
        if self.ended: return self.last
        rows = [[float(np.clip(v, -1, 1)) for v in r] for r in rows]
        m = self._parse(self.ctx.call('robomme_step', actions=rows))
        self.last = m; self._check(m)
        return m

    def state(self):
        return np.array(self.last['state'], float)

    def remaining(self):
        return self.last.get('episode_limits', {}).get('remaining_episode_steps', 9999)

def down_rot(yaw):
    # gripper pointing down (rot x by pi) then yaw about world z
    return Rot.from_euler('z', yaw) * Rot.from_rotvec([np.pi, 0, 0])

def rot_err(cur_rv, target_R):
    cur = Rot.from_rotvec(cur_rv)
    return (target_R * cur.inv()).as_rotvec()

def move_to(rb, target, yaw, grip, max_steps=60, tol=0.004, rtol=0.03, chunk=4, gain=1.0, vmax=1.0):
    """closed-loop move of TCP to target xyz with yaw, gripper command grip (-1/1)."""
    target = np.asarray(target, float); tR = down_rot(yaw)
    used = 0
    while used < max_steps and not rb.ended:
        s = rb.state(); e = target - s[:3]; re = rot_err(s[3:6], tR)
        if np.linalg.norm(e) < tol and np.linalg.norm(re) < rtol: return True
        rows = []
        # plan chunk rows open loop from measured error, splitting the remaining error
        n = min(chunk, max_steps - used)
        for k in range(n):
            frac = 1.0 / max(1, (n - k))
            dp = np.clip(e * gain / 0.05, -vmax, vmax)
            dr = np.clip(re / 0.5, -1, 1)
            rows.append(list(dp) + list(dr) + [grip])
            e = e - dp * 0.05; re = re - dr * 0.5
            if np.linalg.norm(e) < 1e-4 and np.linalg.norm(re) < 1e-3: break
        rb.step(rows); used += len(rows)
    s = rb.state()
    return np.linalg.norm(target - s[:3]) < tol * 3

def hold(rb, grip, n):
    s = rb.state()
    rows = [[0, 0, 0, 0, 0, 0, grip]] * n
    rb.step(rows)
