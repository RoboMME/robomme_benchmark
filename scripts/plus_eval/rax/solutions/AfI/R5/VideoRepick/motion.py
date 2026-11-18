"""Closed-loop end-effector motion over robomme_step."""
import json
import numpy as np


def rotvec_to_mat(r):
    r = np.asarray(r, float); th = np.linalg.norm(r)
    if th < 1e-9:
        return np.eye(3)
    k = r / th; Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * Kx @ Kx


def mat_to_rotvec(R):
    c = (np.trace(R) - 1) / 2; c = np.clip(c, -1, 1); th = np.arccos(c)
    if th < 1e-7:
        return np.zeros(3)
    if th > np.pi - 1e-4:
        # near pi: axis from diagonal
        A = (R + np.eye(3)) / 2
        k = np.sqrt(np.clip(np.diag(A), 0, None))
        i = int(np.argmax(k)); k = A[:, i] / np.sqrt(A[i, i])
        return k * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(th))
    return w * th


def down_mat(yaw=0.0):
    """Gripper pointing down (rotation pi about x) then yaw about world z."""
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    return Rz @ rotvec_to_mat([np.pi, 0, 0])


def meta(reply):
    for b in reply.get("content", []):
        if b.get("type") == "text":
            return json.loads(b["text"])
    return {}


class Robot:
    def __init__(self, ctx, log=print):
        self.ctx = ctx; self.log = log
        self.last = None; self.ended = False; self.complete = False
        self.state = None; self.step_id = 0; self.remaining = None

    def observe(self):
        r = self.ctx.call("robomme_observe"); m = meta(r)
        self._absorb(m)
        self.last_reply = r
        return m

    def observe_frame(self):
        """Observe and decode the attached images into a frame dict with in-memory arrays."""
        import base64, io
        from PIL import Image
        r = self.ctx.call("robomme_observe"); m = meta(r)
        self._absorb(m)
        content = r.get("content", [])
        arrays = {}
        for att in m.get("attachments", []):
            idx = att.get("content_index")
            if idx is None or idx >= len(content):
                continue
            blk = content[idx]
            if blk.get("type") != "image":
                continue
            im = Image.open(io.BytesIO(base64.b64decode(blk["data"])))
            if att.get("kind") == "rgb":
                arrays[(att["camera"], "rgb")] = np.asarray(im.convert("RGB"))
            else:
                scale = (m.get("depth", {}).get(att["camera"], {}) or {}).get("meters_per_unit", 0.001)
                arrays[(att["camera"], "depth")] = np.asarray(im).astype(np.float64) * scale
        return dict(arrays=arrays, camera_intrinsics=m.get("camera_intrinsics"), camera_poses=m.get("camera_poses"),
                    state=m.get("state"), step_id=m.get("step_id"))

    def _absorb(self, m):
        if "state" in m:
            self.state = np.array(m["state"], float)
        if "step_id" in m:
            self.step_id = m["step_id"]
        lim = m.get("episode_limits") or {}
        if "remaining_episode_steps" in lim:
            self.remaining = lim["remaining_episode_steps"]
        if m.get("episode_ended"):
            self.ended = True
        tr = m.get("transition") or {}
        if tr.get("task_complete"):
            self.complete = True
        if tr.get("episode_ended"):
            self.ended = True
        for s in (m.get("steps", []) or []) + (tr.get("steps", []) or []):
            if isinstance(s, dict):
                if s.get("task_complete"):
                    self.complete = True
                if s.get("episode_ended"):
                    self.ended = True
        if m.get("task_complete"):
            self.complete = True
        self.last = m

    def step(self, rows):
        if self.ended:
            return self.last
        rows = [[float(np.clip(x, -1, 1)) for x in r] for r in rows]
        r = self.ctx.call("robomme_step", actions=rows); m = meta(r)
        self._absorb(m); return m

    def grip(self):
        return self.state[6] + self.state[7]

    def move_to(self, target, R_target=None, gripper=-1.0, tol=0.004, max_steps=120, chunk=4, gain=1.0,
                vmax=1.0, rtol=0.03):
        """Servo TCP to `target` (xyz) with orientation R_target (default: keep current)."""
        target = np.asarray(target, float)
        if R_target is None:
            R_target = rotvec_to_mat(self.state[3:6])
        used = 0
        while used < max_steps and not self.ended:
            pos = self.state[:3]; err = target - pos
            Rc = rotvec_to_mat(self.state[3:6]); rerr = mat_to_rotvec(R_target @ Rc.T)
            if np.linalg.norm(err) < tol and np.linalg.norm(rerr) < rtol:
                return True
            n = min(chunk, max_steps - used)
            # plan n rows from predicted path (assume each row achieves its delta)
            rows = []; p = pos.copy(); rr = rerr.copy()
            for _ in range(n):
                e = target - p
                d = np.clip(e / 0.05 * gain, -vmax, vmax)
                rv = np.clip(rr / 0.5, -0.5, 0.5)
                rows.append(list(d) + list(rv) + [gripper])
                p = p + d * 0.05; rr = rr - rv * 0.5
            self.step(rows); used += n
        return np.linalg.norm(target - self.state[:3]) < tol * 3

    def hold(self, n, gripper):
        return self.step([[0, 0, 0, 0, 0, 0, gripper]] * n)
