"""Motion primitives for ButtonUnmask. All use ctx.call (MCP)."""
import json, base64, io, math, time
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as Rot

BASE_ROT = Rot.from_rotvec([math.pi, 0, 0])   # gripper pointing down, yaw 0 (fingers close along world y)

class EpisodeEnded(Exception):
    pass

def meta(res):
    for b in res["content"]:
        if b.get("type") == "text":
            return json.loads(b["text"])
    return {}

def arrays(res):
    m = meta(res); out = {}
    for a in m.get("attachments", []):
        blk = res["content"][a["content_index"]]
        im = Image.open(io.BytesIO(base64.b64decode(blk["data"])))
        if a["kind"] == "depth":
            out[(a["camera"], "depth")] = np.array(im).astype(float) * 0.001
        else:
            out[(a["camera"], "rgb")] = np.array(im.convert("RGB"))
    return m, out

class Robot:
    def __init__(self, ctx, log=print):
        self.ctx = ctx; self.log = log
        self.last = None; self.step_id = 0; self.ended = False; self.remaining = None
        self.task_complete = None

    def _update(self, m):
        self.last = m
        self.step_id = m.get("step_id", self.step_id)
        lim = m.get("episode_limits") or {}
        self.remaining = lim.get("remaining_episode_steps", self.remaining)
        if m.get("task_complete") is not None:
            self.task_complete = m.get("task_complete")
        tr = m.get("transition") or {}
        if m.get("episode_ended") or tr.get("episode_ended"):
            self.ended = True
        return m

    def observe(self):
        res = self.ctx.call("robomme_observe")
        m, A = arrays(res)
        self._update(m)
        self.last_arrays = A
        return m, A

    def step(self, rows):
        if self.ended:
            raise EpisodeEnded()
        rows = [[float(np.clip(v, -1, 1)) for v in r] for r in rows]
        res = self.ctx.call("robomme_step", actions=rows)
        m, A = arrays(res)
        self._update(m)
        self.last_arrays = A
        if self.ended:
            raise EpisodeEnded()
        return m

    @property
    def state(self):
        return self.last["state"]

    def rot_err(self, target_rot):
        cur = Rot.from_rotvec(self.state[3:6])
        return (target_rot * cur.inv()).as_rotvec()

    def goto(self, target, yaw=None, grip=1.0, tol=0.006, rtol=0.03, max_steps=60, chunk=3, gain=1.0, vmax=1.0):
        """Closed-loop move of TCP to target xyz with yaw (rad about world z, None = keep yaw 0 down)."""
        if self.last is None:
            self.observe()
        tr = Rot.from_rotvec([0, 0, yaw or 0.0]) * BASE_ROT
        n = 0
        while n < max_steps:
            p = np.array(self.state[:3]); e = np.array(target) - p
            re = self.rot_err(tr)
            if np.linalg.norm(e) < tol and np.linalg.norm(re) < rtol:
                break
            rows = []
            k = chunk if np.linalg.norm(e) > 0.03 else 1
            for _ in range(k):
                d = np.clip(e / 0.05 * gain, -vmax, vmax)
                r = np.clip(re / 0.5 * 0.8, -1, 1)
                rows.append([d[0], d[1], d[2], r[0], r[1], r[2], grip])
            self.step(rows); n += k
        return np.linalg.norm(np.array(target) - np.array(self.state[:3]))

    def hold(self, n, grip):
        """Hold pose n steps while commanding the gripper."""
        rows = [[0, 0, 0, 0, 0, 0, grip]] * n
        return self.step(rows)
