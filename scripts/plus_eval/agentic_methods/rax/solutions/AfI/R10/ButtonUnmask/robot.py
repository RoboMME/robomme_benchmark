import json, math, time

def jtext(res):
    for b in res.get("content", []):
        if b.get("type") == "text":
            try:
                return json.loads(b["text"])
            except Exception:
                pass
    return {}

class Robot:
    def __init__(self, ctx, log=print):
        self.ctx = ctx
        self.log = log
        self.last = None
        self.ended = False
        self.complete = False
        self.step_id = 0
        self.grip = -1.0
        self.on_frame = None
        self.ik_fail = False

    def observe(self):
        res = self.ctx.call("robomme_observe")
        p = jtext(res)
        if self.on_frame is not None and "state" in p:
            try:
                self.on_frame(res)
            except Exception as e:
                self.log("on_frame error: %r" % (e,))
        self.last = p
        self.step_id = p.get("step_id", self.step_id)
        return p

    def state(self):
        if self.last is None or "state" not in self.last:
            self.observe()
        return self.last["state"]

    def step(self, rows):
        out = None
        for i in range(0, len(rows), 100):
            if self.ended:
                break
            res = self.ctx.call("robomme_step", actions=rows[i:i+100])
            r = jtext(res)
            before = self.step_id
            if self.on_frame is not None and "state" in r:
                try:
                    self.on_frame(res)
                except Exception as e:
                    self.log("on_frame error: %r" % (e,))
            out = r
            if "state" in r:
                self.last = r
            self.step_id = r.get("step_id", self.step_id)
            if self.step_id - before > len(rows[i:i+100]) + 3:
                self.ik_fail = True
                self.log("step jump %d -> %d (native penalty)" % (before, self.step_id))
            for s in r.get("steps", []) or []:
                if s.get("task_complete"):
                    self.complete = True
            if r.get("task_complete") or (r.get("transition") or {}).get("task_complete"):
                self.complete = True
            if r.get("episode_ended"):
                self.ended = True
        return out

    def move_to(self, target, grip=None, max_steps=60, tol=0.004, vmax=1.0, gain=2.0):
        """Closed-loop move of EE to world target (x,y,z). grip: -1 open, +1 close.
        Measured response: ~40% of the commanded delta per step, max ~2 cm/step."""
        if grip is not None:
            self.grip = grip
        n = 0
        best = None; stuck = 0
        while n < max_steps and not self.ended:
            s = self.state()
            d = [target[k] - s[k] for k in range(3)]
            dist = math.sqrt(sum(x * x for x in d))
            if dist < tol:
                break
            if best is not None and dist > best - 0.001:
                stuck += 1
                if stuck >= 4:
                    self.log("move_to: no progress, abort at dist %.3f" % dist)
                    break
            else:
                stuck = 0
            best = dist if best is None else min(best, dist)
            if self.ik_fail:
                self.log("move_to: IK penalty detected, abort")
                self.ik_fail = False
                break
            if dist > 0.06:
                m = max(abs(x) for x in d)
                cmd = [vmax * x / m for x in d]
                rows = [[cmd[0], cmd[1], cmd[2], 0, 0, 0, self.grip]] * 2
            else:
                cmd = [max(-vmax, min(vmax, gain * x / 0.05)) for x in d]
                rows = [[cmd[0], cmd[1], cmd[2], 0, 0, 0, self.grip]]
            self.step(rows)
            n += len(rows)
        return self.state()

    def hold(self, n, grip=None):
        if grip is not None:
            self.grip = grip
        return self.step([[0, 0, 0, 0, 0, 0, self.grip]] * n)

import base64, io
import numpy as np
from PIL import Image

def decode_images(res):
    """Return {(camera, kind): ndarray} from an MCP result with attachments."""
    meta = jtext(res)
    blocks = res.get("content", [])
    out = {}
    for a in meta.get("attachments", []) or []:
        i = a.get("content_index")
        if i is None or i >= len(blocks):
            continue
        b = blocks[i]
        data = b.get("data")
        if not data:
            continue
        im = Image.open(io.BytesIO(base64.b64decode(data)))
        if a["kind"] == "depth":
            arr = np.array(im).astype(np.float64) * 0.001
        else:
            arr = np.array(im.convert("RGB"))
        out[(a["camera"], a["kind"])] = arr
    return meta, out

def rotvec_to_R(rv):
    rv = np.asarray(rv, float)
    th = np.linalg.norm(rv)
    if th < 1e-9:
        return np.eye(3)
    k = rv / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * Kx + (1 - math.cos(th)) * Kx @ Kx

def finger_angle(state):
    R = rotvec_to_R(state[3:6])
    y = R[:, 1]
    return math.atan2(y[1], y[0])

def wrap(a, period):
    return (a + period / 2) % period - period / 2

def rotate_yaw(robot, target_angle, period=math.pi / 2, tol=0.03, max_steps=20):
    """Rotate about world z so finger axis angle matches target modulo period."""
    for _ in range(max_steps):
        s = robot.state()
        e = wrap(target_angle - finger_angle(s), period)
        if abs(e) < tol:
            return e
        c = max(-1, min(1, e / 0.5))
        robot.step([[0, 0, 0, 0, 0, c, robot.grip]])
    return wrap(target_angle - finger_angle(robot.state()), period)
