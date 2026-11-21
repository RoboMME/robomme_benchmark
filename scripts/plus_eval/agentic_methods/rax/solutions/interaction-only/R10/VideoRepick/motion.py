"""Motion primitives over the official 7-D action rows."""
import json
import numpy as np
from scipy.spatial.transform import Rotation as Rot

POS_SCALE = 0.05
ROT_SCALE = 0.5
MAX_ROW = 0.6          # fraction of 0.05 m per step (3 cm/step max)


class Ended(Exception):
    pass


class Robot:
    def __init__(self, ctx, log=print):
        self.ctx = ctx
        self.log = log
        self.last = None
        self.ended = False
        self.complete = False
        self.target_rot = Rot.from_rotvec([np.pi, 0, 0])

    def _packet(self, result):
        txt = next(b["text"] for b in result["content"] if b.get("type") == "text")
        return json.loads(txt)

    def observe(self):
        res = self.ctx.call("robomme_observe")
        p = self._packet(res)
        self.last = p
        self.last_res = res
        return p

    def images(self):
        """Decode current RGB and depth images -> dict[(camera, kind)] = ndarray."""
        import base64, io
        from PIL import Image
        out = {}
        blocks = self.last_res["content"]
        for a in self.last.get("attachments", []):
            b = blocks[a["content_index"]]
            data = base64.b64decode(b["data"])
            with Image.open(io.BytesIO(data)) as im:
                if a["kind"] == "rgb":
                    out[(a["camera"], "rgb")] = np.array(im.convert("RGB"))
                else:
                    mpu = self.last["depth"][a["camera"]].get("meters_per_unit", 0.001)
                    out[(a["camera"], "depth")] = np.array(im).astype(np.float64) * mpu
        return out

    def state(self):
        return np.array(self.last["state"], float)

    def step(self, rows):
        if self.ended:
            raise Ended()
        rows = [[float(np.clip(x, -1, 1)) for x in r] for r in rows]
        res = self.ctx.call("robomme_step", actions=rows)
        p = self._packet(res)
        self._check_end(p)
        # step reply contains final observation fields
        if "state" in p and "attachments" in p:
            self.last = p
            self.last_res = res
        elif not self.ended:
            self.observe()
        return p

    def _check_end(self, p):
        ended = bool(p.get("episode_ended"))
        tc = False
        steps = list(p.get("steps", []) or []) + list((p.get("transition") or {}).get("steps", []) or [])
        if (p.get("transition") or {}).get("episode_ended"):
            ended = True
        for s in steps:
            if s.get("task_complete"):
                tc = True
            if s.get("episode_ended"):
                ended = True
        if p.get("task_complete") or (p.get("transition") or {}).get("task_complete"):
            tc = True
        if tc:
            self.complete = True
        if ended:
            self.ended = True
            self.log(f"episode ended complete={self.complete}")

    def rot_err(self):
        cur = Rot.from_rotvec(self.state()[3:6])
        return (self.target_rot * cur.inv()).as_rotvec()

    def move_to(self, target, grip=-1.0, tol=0.004, max_calls=6, speed=MAX_ROW, settle=0):
        target = np.asarray(target, float)
        for _ in range(max_calls):
            s = self.state()
            d = target - s[:3]
            dist = np.linalg.norm(d)
            if dist < tol:
                break
            n = int(np.ceil(dist / (speed * POS_SCALE)))
            n = max(1, min(n, 60))
            rows = []
            for i in range(n):
                frac = d / n / POS_SCALE
                re = self.rot_err() / ROT_SCALE if i == 0 else np.zeros(3)
                re = np.clip(re, -0.5, 0.5)
                rows.append(list(frac) + list(re) + [grip])
            self.step(rows)
        if settle:
            self.hold(settle, grip)
        return np.linalg.norm(target - self.state()[:3])

    def hold(self, n, grip):
        self.step([[0, 0, 0, 0, 0, 0, grip]] * n)

    def gripper(self, grip, n=10):
        self.hold(n, grip)

    def opening(self):
        s = self.state()
        return float(s[6] + s[7])
