import json, time, base64, io
import numpy as np

LOG = []

def parse(reply):
    meta = json.loads(reply["content"][0]["text"])
    imgs = {}
    for a in meta.get("attachments", []):
        blk = reply["content"][a["content_index"]]
        imgs[(a["camera"], a["kind"])] = blk.get("data")
    return meta, imgs

def decode_rgb(b64):
    from PIL import Image
    return np.array(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB"))

def decode_depth(b64):
    from PIL import Image
    return np.array(Image.open(io.BytesIO(base64.b64decode(b64)))).astype(np.float32) * 0.001

class Robot:
    def __init__(self, ctx, log=print):
        self.ctx = ctx
        self.log = log
        self.meta = None
        self.imgs = None
        self.ended = False
        self.complete = False
        self.observe()

    def observe(self):
        r = self.ctx.call("robomme_observe")
        self.meta, self.imgs = parse(r)
        return self.meta

    @property
    def state(self):
        return np.array(self.meta["state"], dtype=float)

    @property
    def step_id(self):
        return self.meta["step_id"]

    def remaining(self):
        return self.meta["episode_limits"]["remaining_episode_steps"]

    def step(self, rows):
        rows = [[float(np.clip(v, -1, 1)) for v in r] for r in rows]
        r = self.ctx.call("robomme_step", actions=rows)
        self.meta, self.imgs = parse(r)
        tr = self.meta.get("transition") or {}
        if tr.get("task_complete") or self.meta.get("task_complete"):
            self.complete = True
        if tr.get("episode_ended"):
            self.ended = True
        for s in tr.get("steps", []):
            if s.get("task_complete"):
                self.complete = True
            if s.get("episode_ended"):
                self.ended = True
        if self.meta.get("episode_ended"):
            self.ended = True
        return self.meta

    def rgb(self, cam):
        return decode_rgb(self.imgs[(cam, "rgb")])

    def depth(self, cam):
        return decode_depth(self.imgs[(cam, "depth")])

    def deproject(self, cam, u, v, d=None):
        """u,v pixel arrays -> world points (N,3), using current calibration."""
        K = np.array(self.meta["camera_intrinsics"][cam]["matrix"])
        pose = self.meta["camera_poses"][cam]
        p = np.array(pose["position_m"]); w, x, y, z = pose["quaternion_wxyz"]
        R = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                      [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                      [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
        u = np.asarray(u, float); v = np.asarray(v, float)
        if d is None:
            D = self.depth(cam)
            d = D[np.clip(v.astype(int), 0, 255), np.clip(u.astype(int), 0, 255)]
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        pc = np.stack([(u-cx)*d/fx, -(v-cy)*d/fy, -d], -1)
        return pc @ R.T + p

    def move_to(self, target, grip, tol=0.006, max_steps=60, gain=1.0, chunk=1, vmax=1.0):
        target = np.asarray(target, float)
        n = 0
        while n < max_steps and not self.ended:
            err = target - self.state[:3]
            if np.linalg.norm(err) < tol:
                return True
            a = np.clip(err / 0.05 * gain, -vmax, vmax)
            # keep orientation pointing down: correct rotation error
            rot = self.rot_correction()
            row = list(a) + list(rot) + [grip]
            self.step([row] * chunk)
            n += chunk
        return np.linalg.norm(target - self.state[:3]) < tol

    def rot_correction(self):
        # desired orientation: rotvec (pi,0,0) (gripper down). Return small world-frame correction.
        from scipy.spatial.transform import Rotation as Rr
        cur = Rr.from_rotvec(self.state[3:6])
        des = Rr.from_rotvec([np.pi, 0, 0]) * Rr.from_rotvec([0, 0, 0])
        d = (des * cur.inv()).as_rotvec()
        return list(np.clip(d / 0.5, -0.3, 0.3))

    def hold(self, n, grip):
        rows = [[0, 0, 0, 0, 0, 0, grip]] * n
        return self.step(rows)
