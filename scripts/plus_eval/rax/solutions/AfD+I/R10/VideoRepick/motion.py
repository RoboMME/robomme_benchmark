import json, time
import numpy as np

def packet(reply):
    if isinstance(reply, dict) and 'content' in reply:
        content = reply['content']
    else:
        content = reply
    for b in content:
        if b.get('type') == 'text':
            try:
                return json.loads(b['text'])
            except Exception:
                continue
    return {}

class Robot:
    def __init__(self, ctx, log=print):
        self.ctx = ctx; self.log = log
        self.last = None; self.ended = False; self.complete = False
        self.steps_used = 0

    def observe(self):
        p = packet(self.ctx.call('robomme_observe'))
        self.last = p
        return p

    def state(self):
        if self.last is None or 'state' not in self.last:
            self.observe()
        return np.array(self.last['state'], float)

    def step(self, rows):
        rows = [[float(np.clip(x, -1, 1)) for x in r] for r in rows]
        p = packet(self.ctx.call('robomme_step', actions=rows))
        if 'state' in p:
            self.last = p
        else:
            # final observation may be nested
            self.last = dict(self.last or {}, **p)
        if p.get('episode_ended'):
            self.ended = True
        for s in p.get('steps', []) or []:
            if isinstance(s, dict):
                if s.get('task_complete'):
                    self.complete = True
                if s.get('episode_ended'):
                    self.ended = True
        if p.get('task_complete'):
            self.complete = True
        return p

    def remaining(self):
        lim = (self.last or {}).get('episode_limits', {})
        return lim.get('remaining_episode_steps', 10**6)

    def move_to(self, target, grip=-1.0, yaw=None, tol=0.004, max_steps=80, gain=0.9, vmax=0.6, chunk=4):
        """Closed-loop move of EE to target xyz; optional yaw target (rad about world z, relative to down pose)."""
        target = np.asarray(target, float)
        used = 0
        while used < max_steps and not self.ended:
            s = self.state()
            err = target - s[:3]
            yerr = 0.0
            if yaw is not None:
                yerr = wrap(yaw - ee_yaw(s))
            if np.linalg.norm(err) < tol and abs(yerr) < 0.03:
                return True
            rows = []
            n = min(chunk, max_steps - used)
            # plan n rows from the current error (open loop within chunk, conservative)
            for k in range(n):
                e = err * (1 - k / max(n, 1) * 0.0)
                a = np.clip(e / 0.05 * gain / n * (1 if n == 1 else 1), -vmax, vmax)
                ry = np.clip(yerr / 0.5 * gain / n, -0.5, 0.5) if yaw is not None else 0.0
                rows.append([a[0], a[1], a[2], 0.0, 0.0, ry, grip])
            # rows above split the remaining error across n steps
            self.step(rows)
            used += n
        return False

    def hold(self, n, grip):
        return self.step([[0, 0, 0, 0, 0, 0, grip]] * n)

def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def rotvec_to_R(rv):
    th = np.linalg.norm(rv)
    if th < 1e-9:
        return np.eye(3)
    k = rv / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K

def ee_yaw(s):
    R = rotvec_to_R(np.asarray(s[3:6]))
    x = R[:, 0]
    return float(np.arctan2(x[1], x[0]))

def live_frame(reply):
    """Convert an observe reply into a frame-like dict with in-memory arrays."""
    import base64, io
    from PIL import Image
    content = reply['content'] if isinstance(reply, dict) and 'content' in reply else reply
    p = packet(reply)
    arrays = {}
    for a in p.get('attachments', []):
        b = content[a['content_index']]
        data = base64.b64decode(b['data'])
        with Image.open(io.BytesIO(data)) as im:
            if a['kind'] == 'depth':
                arrays[(a['camera'], 'depth')] = np.asarray(im).astype(np.float32) * 0.001
            else:
                arrays[(a['camera'], 'rgb')] = np.asarray(im.convert('RGB'))
    p['_arrays'] = arrays
    return p
