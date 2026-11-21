import json, time
import numpy as np

LOG = []

def log(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s)
    print(s, flush=True)

def packet(reply):
    for b in reply.get("content", []):
        if b.get("type") == "text":
            try:
                return json.loads(b["text"])
            except Exception:
                pass
    return {}

import base64, io
from PIL import Image

def decode_images(reply, meta):
    out = {}
    blocks = reply.get("content", [])
    for a in meta.get("attachments", []):
        i = a["content_index"]
        if i < len(blocks) and blocks[i].get("type") == "image":
            im = Image.open(io.BytesIO(base64.b64decode(blocks[i]["data"])))
            arr = np.asarray(im)
            if a["kind"] == "depth":
                arr = arr.astype(np.float64) * meta["depth"][a["camera"]].get("meters_per_unit", 0.001)
            else:
                arr = arr[..., :3]
            out[(a["camera"], a["kind"])] = arr
    return out

class Robot:
    def __init__(self, ctx):
        self.ctx = ctx
        self.last = None
        self.ended = False
        self.complete = False
        self.grip = -1.0
        self.on_step = None
        self.single = False

    def observe(self):
        r = self.ctx.call("robomme_observe")
        p = packet(r)
        self.last = p
        self.imgs = decode_images(r, p)
        return p

    def world(self, cam="wrist"):
        from percep import backproject
        d = self.imgs[(cam, "depth")]
        return backproject(d, self.last["camera_intrinsics"][cam]["matrix"], self.last["camera_poses"][cam])

    def state(self):
        return np.array(self.last["state"])

    def remaining(self):
        return self.last.get("episode_limits", {}).get("remaining_episode_steps", 0)

    def step(self, rows):
        if self.ended:
            return self.last
        rows = [[float(np.clip(x, -1, 1)) for x in r] for r in rows]
        rem = self.remaining()
        if rem is not None and rem <= 0:
            self.ended = True
            return self.last
        rows = rows[:max(1, min(100, rem))]
        if self.single and len(rows) > 1:
            for row in rows:
                self.step([row])
                if self.ended:
                    break
            return self.last
        r = self.ctx.call("robomme_step", actions=rows)
        p = packet(r)
        self.last = p
        try:
            self.imgs = decode_images(r, p)
        except Exception:
            pass
        tr = p.get("transition", {})
        for s in tr.get("steps", []):
            if s.get("task_complete"):
                self.complete = True
        if self.on_step is not None and len(rows) == 1:
            try:
                self.on_step(self)
            except Exception as e:
                log("on_step error", repr(e))
        if p.get("episode_ended") or tr.get("episode_ended"):
            self.ended = True
            log("EPISODE ENDED step", p.get("step_id"), "complete", self.complete, json.dumps(tr.get("steps", [])[-1:]))
        return p

    def wait(self, n, grip=None):
        g = self.grip if grip is None else grip
        return self.step([[0, 0, 0, 0, 0, 0, g]] * n)

    def set_grip(self, g, n=8):
        self.grip = g
        return self.wait(n, g)

    def move_to(self, target, tol=0.004, vmax=0.02, max_iter=8, grip=None):
        """Move TCP to target xyz (world) keeping orientation; chunked open-loop + correction."""
        g = self.grip if grip is None else grip
        target = np.asarray(target, float)
        for it in range(max_iter):
            if self.ended:
                return False
            pos = self.state()[:3]
            err = target - pos
            dist = np.linalg.norm(err)
            if dist < tol:
                return True
            n = int(np.ceil(dist / vmax))
            n = max(1, min(n, 40))
            d = err / n / 0.05
            rot = self.rot_correction()
            rows = [[d[0], d[1], d[2], rot[0], rot[1], rot[2], g]] + [[d[0], d[1], d[2], 0, 0, 0, g]] * (n - 1)
            self.step(rows)
        pos = self.state()[:3]
        return np.linalg.norm(target - pos) < tol * 3

    def rotmat(self):
        rv = self.state()[3:6]
        th = np.linalg.norm(rv)
        if th < 1e-9:
            return np.eye(3)
        k = rv / th
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K

    def finger_angle(self):
        """World yaw (rad) of the finger opening axis (tool y), modulo pi."""
        y = self.rotmat()[:, 1]
        return float(np.arctan2(y[1], y[0]))

    def rotate_fingers_to(self, ang, tol=0.03, max_iter=8):
        g = self.grip
        for it in range(max_iter):
            cur = self.finger_angle()
            d = (ang - cur + np.pi / 2) % np.pi - np.pi / 2
            if abs(d) < tol:
                return True
            n = int(min(20, max(1, np.ceil(abs(d) / 0.1))))
            step = d / n / 0.5
            self.step([[0, 0, 0, 0, 0, step, g]] * n)
        return False

    def rot_correction(self):
        """Small world-frame rotation delta to bring gripper to pointing straight down (axis-angle [pi,0,0]-ish), keep yaw."""
        rv = self.state()[3:6]
        th = np.linalg.norm(rv)
        if th < 1e-6:
            return np.zeros(3)
        k = rv / th
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        R = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K
        z = R[:, 2]  # tool z axis in world; want [0,0,-1]
        want = np.array([0, 0, -1.0])
        ax = np.cross(z, want)
        s = np.linalg.norm(ax)
        if s < 1e-4:
            return np.zeros(3)
        ang = np.arcsin(min(1.0, s))
        corr = ax / s * ang
        return np.clip(corr / 0.5, -0.3, 0.3)
