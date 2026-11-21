import numpy as np
from scipy.spatial.transform import Rotation as Rot
from mcpio import parse

class Robot:
    def __init__(self, ctx, log=print):
        self.ctx, self.log = ctx, log
        self.meta, self.imgs = parse(ctx.call('robomme_observe'))
        self.grip = -1.0
        self.ended = False
        self.task_complete = False

    @property
    def state(self): return np.array(self.meta['state'], float)
    @property
    def pos(self): return self.state[:3]
    def yaw(self):
        R = Rot.from_rotvec(self.state[3:6]).as_matrix()
        return float(np.arctan2(R[1, 0], R[0, 0]))
    def remaining(self):
        return self.meta.get('episode_limits', {}).get('remaining_episode_steps', 0)

    def step(self, rows):
        if self.ended: return
        r = self.ctx.call('robomme_step', actions=[list(map(float, row)) for row in rows])
        self.meta, self.imgs = parse(r)
        tr = self.meta.get('transition') or {}
        steps = (self.meta.get('steps') or []) + (tr.get('steps') or [])
        if tr.get('task_complete'): self.task_complete = True
        if tr.get('episode_ended'): self.ended = True
        for s in steps:
            if s.get('task_complete'): self.task_complete = True
        if self.meta.get('task_complete'): self.task_complete = True
        if self.ended or self.meta.get('episode_ended') or any(s.get('episode_ended') for s in steps):
            self.ended = True
            self.log(f'EPISODE ENDED task_complete={self.task_complete} step={self.meta.get("step_id")}')

    def observe(self):
        self.meta, self.imgs = parse(self.ctx.call('robomme_observe'))

    def move_to(self, target, yaw=None, tol=0.004, max_steps=80, speed=1.0, yaw_tol=0.03):
        target = np.asarray(target, float)
        for k in range(max_steps):
            if self.ended: return False
            d = target - self.pos
            row = np.zeros(7); row[6] = self.grip
            row[:3] = np.clip(d / 0.05, -speed, speed)
            dyaw = 0.0
            if yaw is not None:
                dyaw = (yaw - self.yaw() + np.pi) % (2 * np.pi) - np.pi
                # rotation about world z; keep gripper pointing down
                row[5] = np.clip(dyaw / 0.5, -1, 1)
            # keep tool pointing down: correct roll/pitch
            R = Rot.from_rotvec(self.state[3:6])
            zaxis = R.as_matrix()[:, 2]
            err = np.cross(zaxis, np.array([0, 0, -1.0]))
            row[3:5] = np.clip(err[:2] / 0.5, -0.3, 0.3)
            if np.linalg.norm(d) < tol and abs(dyaw) < yaw_tol:
                return True
            self.step([row])
        return False

    def set_gripper(self, g, n=12):
        self.grip = g
        row = [0, 0, 0, 0, 0, 0, g]
        self.step([row] * n)
