"""Motion helpers over ctx.call (RoboMME-Plus official 7-D action rows)."""
import json, base64, io, time
import numpy as np

class EpisodeEnded(Exception):
    pass

class Robot:
    def __init__(self, ctx, log=print):
        self.ctx = ctx; self.log = log
        self.state = None; self.ended = False; self.complete = False
        self.step_id = 0; self.remaining = None; self.grip = -1.0
        self.last = None

    def _parse(self, res):
        content = res["content"] if isinstance(res, dict) else res
        meta = None
        for b in content:
            if b.get("type") == "text":
                try:
                    meta = json.loads(b["text"]); break
                except Exception:
                    pass
        return meta, content

    def observe(self):
        res = self.ctx.call("robomme_observe")
        meta, content = self._parse(res)
        self._update(meta)
        self.last = (meta, content)
        return meta, content

    def images(self, meta, content):
        out = {}
        from PIL import Image
        for a in meta.get("attachments", []):
            b = content[a["content_index"]]
            data = base64.b64decode(b["data"])
            im = Image.open(io.BytesIO(data))
            arr = np.array(im)
            if a["kind"] == "depth":
                arr = arr.astype(np.float64) * 0.001
            else:
                arr = np.array(im.convert("RGB"))
            out[(a["camera"], a["kind"])] = arr
        return out

    def _update(self, meta):
        if not meta:
            return
        if "state" in meta and meta["state"] is not None:
            self.state = np.array(meta["state"], float)
        if "step_id" in meta:
            self.step_id = meta["step_id"]
        lim = meta.get("episode_limits") or {}
        if "remaining_episode_steps" in lim:
            self.remaining = lim["remaining_episode_steps"]
        if meta.get("episode_ended"):
            self.ended = True
        tr = meta.get("transition") or {}
        if tr.get("episode_ended"):
            self.ended = True
        if tr.get("task_complete"):
            self.complete = True
        for s in (meta.get("steps", []) or []) + (tr.get("steps", []) or []):
            if isinstance(s, dict):
                if s.get("task_complete"):
                    self.complete = True
                if s.get("episode_ended"):
                    self.ended = True
        if meta.get("task_complete"):
            self.complete = True

    def step(self, rows):
        if self.ended:
            raise EpisodeEnded()
        rows = [[float(np.clip(v, -1, 1)) for v in r] for r in rows]
        res = self.ctx.call("robomme_step", actions=rows)
        meta, content = self._parse(res)
        self._update(meta)
        self.last = (meta, content)
        if self.ended:
            raise EpisodeEnded()
        return meta

    def move_to(self, target, grip=None, tol=0.004, max_iter=12, chunk=6, gain=1.0, vmax=1.0):
        """Closed-loop translate the TCP to target xyz (world) keeping orientation."""
        if grip is None:
            grip = self.grip
        target = np.asarray(target, float)
        for _ in range(max_iter):
            err = target - self.state[:3]
            dist = np.linalg.norm(err)
            if dist < tol:
                return True
            n = int(min(chunk, max(1, np.ceil(dist / (0.05 * vmax)))))
            rows = []
            # plan n rows each moving err/n (re-measured after chunk)
            per = err / n / 0.05 * gain
            mx = np.abs(per).max()
            if mx > vmax:
                per = per / mx * vmax
            for i in range(n):
                rows.append([per[0], per[1], per[2], 0, 0, 0, grip])
            self.step(rows)
        return np.linalg.norm(target - self.state[:3]) < tol * 3

    def hold(self, n, grip=None):
        if grip is None:
            grip = self.grip
        self.grip = grip
        self.step([[0, 0, 0, 0, 0, 0, grip]] * n)

    def set_grip(self, g, n=10):
        self.grip = g
        self.hold(n, g)

    def rotate_z(self, dyaw, grip=None):
        """Rotate about world z by dyaw radians (in chunks)."""
        if grip is None:
            grip = self.grip
        steps = int(np.ceil(abs(dyaw) / 0.25)) or 1
        per = dyaw / steps / 0.5
        self.step([[0, 0, 0, 0, 0, per, grip]] * steps)
