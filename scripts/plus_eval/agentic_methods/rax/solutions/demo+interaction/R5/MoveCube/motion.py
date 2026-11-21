import json, math
import numpy as np

def _meta(blocks):
    for b in blocks:
        if isinstance(b, dict) and b.get('type') == 'text':
            try:
                return json.loads(b['text'])
            except Exception:
                pass
    return {}

class Robot:
    def __init__(self, ctx, log=print):
        self.ctx = ctx; self.log = log
        self.last = None; self.ended = False; self.complete = False
        self.state = None; self.steps_left = None; self.jumped = False; self.ik_errors = 0

    def observe(self):
        r = self.ctx.call('robomme_observe')
        blocks = r if isinstance(r, list) else r.get('content', r)
        m = _meta(blocks)
        self._upd(m); self.blocks = blocks; return m

    def images(self):
        """decode attachments of the last observe() -> dict {(camera, kind): np.array}"""
        import base64, io
        from PIL import Image
        out = {}
        for a in self.last.get('attachments', []):
            b = self.blocks[a['content_index']]
            im = Image.open(io.BytesIO(base64.b64decode(b['data'])))
            if a['kind'] == 'rgb':
                out[(a['camera'], 'rgb')] = np.array(im.convert('RGB')).astype(np.float32)
            else:
                out[(a['camera'], 'depth')] = np.array(im).astype(np.float32) * 0.001
        return out

    def _upd(self, m):
        if 'state' in m: self.state = np.array(m['state'], float)
        lim = m.get('episode_limits') or {}
        if 'remaining_episode_steps' in lim: self.steps_left = lim['remaining_episode_steps']
        if m.get('episode_ended'): self.ended = True
        tr = m.get('transition') or {}
        if tr.get('task_complete'): self.complete = True
        if tr.get('episode_ended'): self.ended = True
        for s in (m.get('steps', []) or []) + (tr.get('steps', []) or []):
            if isinstance(s, dict):
                if s.get('task_complete'): self.complete = True
                if s.get('episode_ended'): self.ended = True
        if m.get('task_complete'): self.complete = True
        self.last = m

    def step(self, rows):
        if self.ended: return None
        rows = [[float(np.clip(v, -1, 1)) for v in r] for r in rows]
        out = None
        self.jumped = False
        for i in range(0, len(rows), 100):
            before = self.last.get('step_id', 0) if self.last else 0
            if self.steps_left is not None and self.steps_left <= 0:
                self.ended = True; break
            chunk = rows[i:i+100]
            r = self.ctx.call('robomme_step', actions=chunk)
            m = _meta(r if isinstance(r, list) else r.get('content', r))
            self._upd(m); out = m
            adv = (m.get('step_id', before) or before) - before
            if adv > len(chunk) + 3:
                self.ik_errors = getattr(self, 'ik_errors', 0) + 1; self.jumped = True
                self.log('IK penalty: step advanced', adv, 'for', len(chunk), 'rows; total', self.ik_errors)
                break
            if self.ended: break
        return out

    def pix2pos(self, sels, step_id=None, radius=1):
        if step_id is None: step_id = self.last.get('step_id', 0)
        r = self.ctx.call('robomme_pixel_to_position', observation_step_id=step_id,
                          selections=[{'camera': c, 'u': float(u), 'v': float(v)} for c, u, v in sels],
                          depth_window_radius=radius)
        m = _meta(r if isinstance(r, list) else r.get('content', r))
        return [np.array(s['position_m']) if s.get('valid') else None for s in m['selections']]

    def move_to(self, target, grip, max_steps=60, tol=0.004, speed=0.6, yaw=None, chunk=4, rot_rate=0.12):
        """Closed-loop move of the TCP to target (world m). grip -1 open/+1 close.
        yaw: optional desired rotation about world z (rad) relative to the downward pose."""
        target = np.array(target, float); n = 0
        while n < max_steps and not self.ended:
            p = self.state[:3]; d = target - p
            rot = np.zeros(3)
            if yaw is not None:
                rot = self.rot_err(yaw)
            if np.linalg.norm(d) < tol and np.linalg.norm(rot) < 0.03: break
            k = min(chunk, max_steps - n)
            rows = []
            v = d / 0.05
            nv = np.max(np.abs(v))
            if nv > speed: v = v * speed / nv
            # take k steps but don't overshoot: each step 1/k of remaining if small
            if nv <= speed: k = 1
            r = rot / 0.5
            rn = np.max(np.abs(r))
            if rn > rot_rate: r = r * rot_rate / rn
            for _ in range(k):
                rows.append(list(v) + list(r) + [grip])
            self.step(rows); n += k
            if self.jumped: break
        return self.state[:3]

    def rot_err(self, yaw):
        """axis-angle (world) from current orientation to the down-pointing pose with given yaw."""
        from scipy.spatial.transform import Rotation as R
        cur = R.from_rotvec(self.state[3:6])
        tgt = R.from_euler('z', yaw) * R.from_rotvec([math.pi, 0, 0])
        return (tgt * cur.inv()).as_rotvec()

    def grip(self, g, n=10):
        p = self.state[:3]
        return self.step([[0, 0, 0, 0, 0, 0, g]] * n)
