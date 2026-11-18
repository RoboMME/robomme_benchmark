"""Robot helpers for RoboMME-Plus RAX (observe, deproject, closed-loop moves)."""
import base64, io, json, math, time
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R

LOG = []

def log(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s)
    print(s, flush=True)


def _meta(reply):
    for b in reply.get("content", []):
        if b.get("type") == "text":
            try:
                return json.loads(b["text"])
            except Exception:
                pass
    return {}


def _decode_images(reply, meta):
    out = {}
    content = reply.get("content", [])
    for att in meta.get("attachments", []):
        idx = att.get("content_index")
        if idx is None or idx >= len(content):
            continue
        blk = content[idx]
        data = blk.get("data")
        if not data:
            continue
        img = Image.open(io.BytesIO(base64.b64decode(data)))
        arr = np.array(img)
        if att["kind"] == "depth":
            arr = arr.astype(np.float32) * 0.001
        else:
            arr = np.array(img.convert("RGB"))
        out[(att["camera"], att["kind"])] = arr
    return out


class Obs:
    def __init__(self, reply):
        self.meta = _meta(reply)
        self.imgs = _decode_images(reply, self.meta)
        self.state = np.array(self.meta.get("state", [0] * 8), dtype=float)
        self.step_id = self.meta.get("step_id")
        self.intr = self.meta.get("camera_intrinsics", {})
        self.poses = self.meta.get("camera_poses", {})

    def rgb(self, cam):
        return self.imgs.get((cam, "rgb"))

    def depth(self, cam):
        return self.imgs.get((cam, "depth"))

    def deproject(self, cam, u, v, d=None):
        """pixel (u=col, v=row) -> world xyz using OpenGL convention."""
        K = np.array(self.intr[cam]["matrix"], dtype=float)
        if d is None:
            dep = self.depth(cam)
            d = float(dep[int(round(v)), int(round(u))])
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        pc = np.array([(u - cx) * d / fx, -(v - cy) * d / fy, -d])
        p = self.poses[cam]
        w, x, y, z = p["quaternion_wxyz"]
        rot = R.from_quat([x, y, z, w])
        return rot.apply(pc) + np.array(p["position_m"], dtype=float)

    def deproject_mask(self, cam, mask):
        """All valid pixels of mask -> Nx3 world points."""
        dep = self.depth(cam)
        vs, us = np.nonzero(mask & (dep > 0))
        if len(us) == 0:
            return np.zeros((0, 3))
        d = dep[vs, us]
        K = np.array(self.intr[cam]["matrix"], dtype=float)
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        pc = np.stack([(us - cx) * d / fx, -(vs - cy) * d / fy, -d], axis=1)
        p = self.poses[cam]
        w, x, y, z = p["quaternion_wxyz"]
        rot = R.from_quat([x, y, z, w])
        return rot.apply(pc) + np.array(p["position_m"], dtype=float)


class Robot:
    def __init__(self, ctx):
        self.ctx = ctx
        self.ended = False
        self.complete = False
        self.state = None
        self.step_id = 0
        self.remaining = None
        self.grip = -1.0
        self.down_yaw = None  # desired orientation (scipy Rotation) to hold

    def observe(self):
        o = Obs(self.ctx.call("robomme_observe"))
        self.state = o.state
        self.step_id = o.step_id
        lim = o.meta.get("episode_limits") or {}
        self.remaining = lim.get("remaining_episode_steps", self.remaining)
        if o.meta.get("episode_ended"):
            self.ended = True
        return o

    def step(self, rows):
        if self.ended:
            return None
        rows = [[float(np.clip(v, -1, 1)) for v in r] for r in rows]
        reply = self.ctx.call("robomme_step", actions=rows)
        m = _meta(reply)
        if "state" in m:
            self.state = np.array(m["state"], dtype=float)
        elif "observation" in m and "state" in m["observation"]:
            self.state = np.array(m["observation"]["state"], dtype=float)
        self.step_id = m.get("step_id", self.step_id)
        lim = m.get("episode_limits") or {}
        self.remaining = lim.get("remaining_episode_steps", self.remaining)
        tr = m.get("transition") or {}
        for s in (tr.get("steps") or []) + (m.get("steps") or []):
            if s.get("task_complete"):
                self.complete = True
            if s.get("episode_ended"):
                self.ended = True
        if m.get("episode_ended") or tr.get("episode_ended"):
            self.ended = True
        if m.get("task_complete") or tr.get("task_complete"):
            self.complete = True
        self.last_meta = m
        return m

    # orientation helpers
    def cur_rot(self):
        return R.from_rotvec(self.state[3:6])

    def rot_cmd(self):
        if self.down_yaw is None:
            return np.zeros(3)
        err = (self.down_yaw * self.cur_rot().inv()).as_rotvec()
        return np.clip(err / 0.5, -0.6, 0.6)

    def move_to(self, target, tol=0.004, max_steps=120, chunk=3, vmax=1.0, gain=1.0, settle=True):
        """Closed-loop translate EE to target xyz (world), holding orientation & gripper."""
        target = np.asarray(target, dtype=float)
        used = 0
        while used < max_steps and not self.ended:
            err = target - self.state[:3]
            dist = np.linalg.norm(err)
            rerr = np.linalg.norm(self.rot_cmd()) * 0.5
            if dist < tol and rerr < 0.06:
                break
            n = int(min(chunk, max_steps - used))
            # spread the displacement over n steps but no more than vmax per row
            per = err / 0.05 / max(n, 1) * gain
            mx = np.max(np.abs(per))
            if mx > vmax:
                per = per * (vmax / mx)
            rows = []
            rc = self.rot_cmd() / max(n, 1)
            for _ in range(n):
                rows.append(list(per) + list(rc) + [self.grip])
            self.step(rows)
            used += n
        return np.linalg.norm(target - self.state[:3]), used

    def hold(self, n, grip=None):
        if grip is not None:
            self.grip = grip
        rows = [[0, 0, 0] + list(self.rot_cmd() / max(n, 1)) + [self.grip] for _ in range(n)]
        return self.step(rows)

    def rotate_to(self, rot, max_steps=40, tol=0.02):
        self.down_yaw = rot
        used = 0
        while used < max_steps and not self.ended:
            err = (rot * self.cur_rot().inv()).as_rotvec()
            if np.linalg.norm(err) < tol:
                break
            rc = np.clip(err / 0.5, -1, 1)
            self.step([[0, 0, 0] + list(rc) + [self.grip]])
            used += 1
        return used
