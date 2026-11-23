import json, base64, io, time
import numpy as np
from PIL import Image

def parse(result):
    blocks = result["content"]
    meta = json.loads(next(b["text"] for b in blocks if b.get("type") == "text"))
    imgs = {}
    for a in meta.get("attachments", []):
        b = blocks[a["content_index"]]
        if b.get("type") != "image":
            continue
        im = Image.open(io.BytesIO(base64.b64decode(b["data"])))
        arr = np.asarray(im)
        if a["kind"] == "depth":
            arr = arr.astype(np.float64) * meta.get("depth", {}).get(a["camera"], {}).get("meters_per_unit", 0.001)
        else:
            arr = np.asarray(im.convert("RGB")).astype(np.int16)
        imgs[(a["camera"], a["kind"])] = arr
    meta["_imgs"] = imgs
    return meta

class Robot:
    def __init__(self, ctx, log=print):
        self.ctx = ctx; self.log = log
        self.ended = False; self.complete = False; self.last = None
        self.home_rot = None

    def observe(self):
        m = parse(self.ctx.call("robomme_observe"))
        self.last = m
        self.ended = self.ended or bool(m.get("episode_ended"))
        return m

    def state(self):
        return np.array(self.last["state"])

    def step(self, rows):
        if self.ended:
            return self.last
        m = parse(self.ctx.call("robomme_step", actions=[[float(np.clip(v, -1, 1)) for v in r] for r in rows]))
        self.last = m
        tr = m.get("transition") or {}
        if tr.get("task_complete") or any(s.get("task_complete") for s in tr.get("steps", [])):
            self.complete = True
        if m.get("episode_ended"):
            self.ended = True
            self.log(f"EPISODE ENDED step {m.get('step_id')} complete={self.complete}")
        return m

    def remaining(self):
        return self.last["episode_limits"]["remaining_episode_steps"]

    def rot_row(self, target_rotvec):
        """rotation delta (world axis-angle) from current to target, scaled to action units."""
        from scipy.spatial.transform import Rotation as R
        cur = R.from_rotvec(self.state()[3:6]); tgt = R.from_rotvec(target_rotvec)
        d = (tgt * cur.inv()).as_rotvec()
        return np.clip(d / 0.5, -1, 1)

    def move_to(self, xyz, grip, rot=None, tol=0.004, max_steps=60, vmax=0.5, chunk=1):
        """closed-loop translate to xyz (world m). grip -1 open / +1 close. vmax in action units (x0.05 m)."""
        xyz = np.asarray(xyz, float)
        for i in range(max_steps):
            if self.ended:
                return False
            s = self.state()
            err = xyz - s[:3]
            rerr = np.zeros(3) if rot is None else self.rot_row(rot)
            if np.linalg.norm(err) < tol and np.linalg.norm(rerr) < 0.02:
                return True
            a = err / 0.05
            n = np.abs(a).max()
            if n > vmax:
                a = a * vmax / n
            self.step([list(a) + list(np.clip(rerr, -0.3, 0.3)) + [grip]] * chunk)
        return False

    def hold(self, grip, n):
        for _ in range(max(1, n // 5)):
            if self.ended:
                return
            self.step([[0, 0, 0, 0, 0, 0, grip]] * min(5, n))
