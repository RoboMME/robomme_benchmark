import json
import numpy as np
from scipy.spatial.transform import Rotation as Rot

def parse(reply):
    """Return JSON dict from MCP reply (list of content blocks or dict)."""
    if isinstance(reply, dict) and 'content' in reply:
        reply = reply['content']
    if isinstance(reply, list):
        for b in reply:
            if isinstance(b, dict) and b.get('type') == 'text':
                try:
                    return json.loads(b['text'])
                except Exception:
                    continue
    if isinstance(reply, dict):
        return reply
    raise RuntimeError('cannot parse reply')

DOWN = Rot.from_rotvec([np.pi, 0, 0])

def target_rot(yaw):
    return Rot.from_euler('z', yaw) * DOWN

class Robot:
    def __init__(self, ctx, log=print):
        self.ctx, self.log = ctx, log
        self.last = None
        self.ended = False
        self.task_complete = False
        self.steps_used = 0
        self.remaining = None

    def observe(self):
        r = parse(self.ctx.call('robomme_observe'))
        self.last = r
        self._lim(r)
        return r

    def _lim(self, r):
        lim = r.get('episode_limits') or {}
        if 'remaining_episode_steps' in lim:
            self.remaining = lim['remaining_episode_steps']
        if r.get('episode_ended'):
            self.ended = True
        for s in r.get('steps', []) or []:
            if isinstance(s, dict):
                if s.get('task_complete'):
                    self.task_complete = True
                if s.get('episode_ended'):
                    self.ended = True
        if r.get('task_complete'):
            self.task_complete = True

    def state(self):
        if self.last is None or 'state' not in self.last:
            self.observe()
        return np.array(self.last['state'], float)

    def step(self, rows):
        if self.ended:
            return self.last
        rows = [[float(np.clip(v, -1, 1)) for v in row] for row in rows]
        r = parse(self.ctx.call('robomme_step', actions=rows))
        self.last = r
        self.steps_used += len(rows)
        self._lim(r)
        if 'state' not in r:
            self.observe()
        return r

    def move_to(self, xyz, yaw, grip, tol=0.004, max_iter=40, gain=1.0, vmax=0.6, chunk=1, rtol=0.03):
        """Closed-loop move of TCP to xyz with gripper yaw; grip -1 open, +1 close."""
        xyz = np.array(xyz, float)
        Rt = target_rot(yaw)
        for it in range(max_iter):
            if self.ended:
                return False
            s = self.state()
            e = xyz - s[:3]
            Rc = Rot.from_rotvec(s[3:6])
            re = (Rt * Rc.inv()).as_rotvec()
            if np.linalg.norm(e) < tol and np.linalg.norm(re) < rtol:
                return True
            dp = np.clip(gain * e / 0.05, -vmax, vmax)
            dr = np.clip(re / 0.5, -0.6, 0.6)
            self.step([list(dp) + list(dr) + [grip]] * chunk)
        return False

    def hold(self, grip, n):
        s = self.state()
        return self.step([[0, 0, 0, 0, 0, 0, grip]] * n)

def yaw_of_state(s):
    R = Rot.from_rotvec(s[3:6])
    Rz = R * DOWN.inv()
    return Rz.as_euler('xyz')[2]
