"""Perception + motion helpers for BinFill (observation-driven, no layout constants)."""
import base64, io, json, time
import numpy as np
from PIL import Image

LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s)
    print(s, flush=True)


def _quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def parse(result):
    """Decode an MCP result into dict with meta, rgb/depth arrays."""
    content = result["content"]
    meta = json.loads(content[0]["text"])
    obs = {"meta": meta, "rgb": {}, "depth": {}}
    for att in meta.get("attachments", []):
        blk = content[att["content_index"]]
        data = base64.b64decode(blk["data"])
        im = Image.open(io.BytesIO(data))
        if att["kind"] == "rgb":
            obs["rgb"][att["camera"]] = np.array(im.convert("RGB"))
        else:
            obs["depth"][att["camera"]] = np.array(im).astype(np.float32) * 0.001
    return obs


class Cam:
    def __init__(self, meta, cam):
        K = np.array(meta["camera_intrinsics"][cam]["matrix"])
        self.fx, self.fy, self.cx, self.cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        p = meta["camera_poses"][cam]
        self.t = np.array(p["position_m"])
        self.R = _quat_to_R(p["quaternion_wxyz"])

    def to_world(self, u, v, d):
        u = np.asarray(u, float); v = np.asarray(v, float); d = np.asarray(d, float)
        pc = np.stack([(u - self.cx) * d / self.fx, -(v - self.cy) * d / self.fy, -d], -1)
        return pc @ self.R.T + self.t

    def project(self, pw):
        pc = (np.asarray(pw) - self.t) @ self.R
        d = -pc[..., 2]
        u = pc[..., 0] * self.fx / d + self.cx
        v = -pc[..., 1] * self.fy / d + self.cy
        return u, v, d


def hsv(rgb):
    import cv2
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)  # H 0..180


def color_masks(rgb):
    """Boolean masks for cube colours. OpenCV hue 0..180."""
    h = hsv(rgb)
    H, S, V = h[..., 0].astype(int), h[..., 1].astype(int), h[..., 2].astype(int)
    sat = (S > 120) & (V > 50)
    m = {
        "red": sat & ((H <= 4) | (H >= 172)),
        "orange": sat & (H >= 17) & (H <= 25),
        "yellow": sat & (H >= 27) & (H <= 38),
        "green": sat & (H >= 45) & (H <= 75),
        "cyan": sat & (H >= 82) & (H <= 98),
        "blue": sat & (H >= 108) & (H <= 130),
        "magenta": sat & (H >= 140) & (H <= 165),
    }
    return m


def blobs(mask, min_area=6):
    import cv2
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    out = []
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        if a >= min_area:
            out.append({"area": int(a), "uv": (float(cent[i][0]), float(cent[i][1])),
                        "mask": lab == i, "bbox": stats[i, :4].tolist()})
    return out


def blob_world(obs, cam, blob):
    c = Cam(obs["meta"], cam)
    d = obs["depth"][cam]
    vs, us = np.nonzero(blob["mask"] & (d > 0))
    if len(us) == 0:
        return None
    pts = c.to_world(us, vs, d[vs, us])
    zt = np.percentile(pts[:, 2], 90)
    top = pts[pts[:, 2] > zt - 0.006]
    if len(top) < 3:
        top = pts
    return np.array([np.median(top[:, 0]), np.median(top[:, 1]), zt])


class Robot:
    def __init__(self, ctx):
        self.ctx = ctx
        self.obs = None
        self.ended = False
        self.complete = False
        self.step_id = 0
        self.remaining = None

    def observe(self):
        if self.ended and self.obs is not None:
            return self.obs
        r = self.ctx.call("robomme_observe")
        self.obs = parse(r)
        m = self.obs["meta"]
        self.step_id = m.get("step_id", self.step_id)
        lim = m.get("episode_limits") or {}
        if "remaining_episode_steps" in lim:
            self.remaining = lim["remaining_episode_steps"]
        if m.get("episode_ended"):
            self.ended = True
        return self.obs

    @property
    def state(self):
        return np.array(self.obs["meta"]["state"])

    def step(self, rows):
        if self.ended:
            return None
        rows = [[float(np.clip(x, -1, 1)) for x in r] for r in rows]
        for i in range(0, len(rows), 100):
            r = self.ctx.call("robomme_step", actions=rows[i:i + 100])
            self.obs = parse(r)
            m = self.obs["meta"]
            self.step_id = m.get("step_id", self.step_id)
            lim = m.get("episode_limits") or {}
            if "remaining_episode_steps" in lim:
                self.remaining = lim["remaining_episode_steps"]
            for s in m.get("steps", []) or []:
                if isinstance(s, dict) and s.get("task_complete"):
                    self.complete = True
            if m.get("task_complete") or (m.get("transition") or {}).get("task_complete"):
                self.complete = True
            if m.get("episode_ended"):
                self.ended = True
                log("EPISODE ENDED complete=", self.complete, "step", self.step_id)
                return self.obs
        return self.obs

    def grip_open(self):
        s = self.state
        return s[6] + s[7]

    def move_to(self, target, grip, tol=0.004, max_steps=60, speed=1.0, rot_target=None):
        """Closed-loop translate the TCP to target (world m). grip -1 open / +1 close."""
        target = np.asarray(target, float)
        for k in range(max_steps):
            if self.ended:
                return False
            p = self.state[:3]
            err = target - p
            if np.linalg.norm(err) < tol:
                return True
            a = np.clip(err / 0.05, -speed, speed)
            row = [a[0], a[1], a[2], 0, 0, 0, grip]
            if rot_target is not None:
                row[3:6] = list(rot_err(self.state[3:6], rot_target) / 0.5)
            # send several identical-ish steps when far away
            n = 1
            dist = np.linalg.norm(err)
            if dist > 0.06:
                n = 3
            elif dist > 0.02:
                n = 2
            rows = []
            for j in range(n):
                frac = 1.0 / n
                rows.append(row)
            # when sending multiple steps, scale per-step so we don't overshoot
            if n > 1:
                a2 = np.clip(err / (0.05 * n), -speed, speed)
                rows = [[a2[0], a2[1], a2[2], 0, 0, 0, grip] for _ in range(n)]
            self.step(rows)
        return np.linalg.norm(target - self.state[:3]) < tol * 3

    def hold(self, grip, n):
        self.step([[0, 0, 0, 0, 0, 0, grip]] * n)


def rotvec_to_R(rv):
    th = np.linalg.norm(rv)
    if th < 1e-9:
        return np.eye(3)
    k = rv / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def R_to_rotvec(R):
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R).as_rotvec()


def rot_err(cur_rv, tgt_rv):
    Rc = rotvec_to_R(np.asarray(cur_rv)); Rt = rotvec_to_R(np.asarray(tgt_rv))
    return R_to_rotvec(Rt @ Rc.T)


def front_points(obs):
    c = Cam(obs["meta"], "front")
    d = obs["depth"]["front"]
    vv, uu = np.mgrid[0:d.shape[0], 0:d.shape[1]]
    return c.to_world(uu, vv, d)


def find_fixtures(obs, ee_xy=None):
    """Return (bin, button) dicts from the front view, via low-saturation raised blobs.
    bin: has a dark interior; button: small raised low-sat blob without dark interior."""
    P = front_points(obs)
    z = P[..., 2]
    h = hsv(obs["rgb"]["front"])
    S, V = h[..., 1].astype(int), h[..., 2].astype(int)
    dark = (V < 35) & (z > -0.02) & (z < 0.09)
    m = (((z > 0.02) & (z < 0.09) & (S < 90)) | dark) & (P[..., 0] > -0.5)
    cands = []
    for b in blobs(m, 15):
        pts = P[b["mask"]]
        xy = pts[:, :2]
        ext = np.ptp(xy, 0)
        ndark = int((b["mask"] & dark).sum())
        cands.append({"area": b["area"], "uv": b["uv"], "ext": ext, "ndark": ndark,
                      "zmax": float(np.percentile(pts[:, 2], 95)),
                      "center": (xy.min(0) + xy.max(0)) / 2, "mean": xy.mean(0), "mask": b["mask"]})
    bins = [c for c in cands if c["ndark"] > 20 and 0.05 < max(c["ext"]) < 0.25]
    btns = [c for c in cands if c["ndark"] <= 20 and max(c["ext"]) < 0.12 and c["zmax"] < 0.06]
    binc = None
    if bins:
        binc = max(bins, key=lambda c: c["ndark"])
        # interior dark region centroid gives centre (more robust than rim extents)
        dm = binc["mask"] & dark
        dpts = P[dm]
        rim = P[binc["mask"] & ~dark]
        binc["center"] = np.array([np.median(dpts[:, 0]), np.median(dpts[:, 1])]) if len(dpts) > 10 else binc["center"]
        binc["rim_center"] = (rim[:, :2].min(0) + rim[:, :2].max(0)) / 2 if len(rim) else binc["center"]
    btn = max(btns, key=lambda c: c["area"]) if btns else None
    return binc, btn, cands
