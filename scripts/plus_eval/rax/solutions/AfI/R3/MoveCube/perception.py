"""Observation parsing, depth back-projection and color segmentation."""
import base64, io, json
import numpy as np
from PIL import Image


def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class Obs:
    def __init__(self, meta, rgb, depth):
        self.meta = meta
        self.rgb = rgb      # cam -> HxWx3 uint8
        self.depth = depth  # cam -> HxW float metres (0 invalid)
        self.state = np.array(meta["state"], dtype=float)
        self.step_id = meta.get("step_id")
        self.ended = bool(meta.get("episode_ended", False))

    def K(self, cam):
        return np.array(self.meta["camera_intrinsics"][cam]["matrix"], dtype=float)

    def pose(self, cam):
        p = self.meta["camera_poses"][cam]
        return quat_to_R(p["quaternion_wxyz"]), np.array(p["position_m"], dtype=float)

    def cloud(self, cam):
        """World points HxWx3 for every pixel (nan where invalid)."""
        K = self.K(cam)
        d = self.depth[cam]
        H, W = d.shape
        u, v = np.meshgrid(np.arange(W) + 0.0, np.arange(H) + 0.0)
        # pixel centre convention: test showed integer pixel coords match the tool
        x = (u - K[0, 2]) / K[0, 0] * d
        y = -(v - K[1, 2]) / K[1, 1] * d
        z = -d
        P = np.stack([x, y, z], -1)
        R, t = self.pose(cam)
        Pw = P @ R.T + t
        Pw[d <= 0] = np.nan
        return Pw


def parse_reply(reply):
    content = reply["content"]
    meta = json.loads(content[0]["text"])
    imgs = [b for b in content[1:] if b.get("type") == "image"]
    rgb, depth = {}, {}
    att = meta.get("attachments")
    if att:
        for a in att:
            b = content[a["content_index"]]
            im = Image.open(io.BytesIO(base64.b64decode(b["data"])))
            arr = np.array(im)
            if a["kind"] == "rgb":
                rgb[a["camera"]] = arr[..., :3].astype(np.uint8)
            else:
                depth[a["camera"]] = arr.astype(np.float32) * 0.001
    return Obs(meta, rgb, depth)


def observe(ctx):
    return parse_reply(ctx.call("robomme_observe"))


def hsv(rgb):
    a = rgb.astype(np.float32) / 255.0
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx = a.max(-1); mn = a.min(-1); d = mx - mn + 1e-9
    h = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60
    s = np.where(mx > 0, d / (mx + 1e-9), 0)
    return h, s, mx


def masks(rgb):
    """Boolean masks for the task's flat-shaded object colours (ratio tests survive shading)."""
    a = rgb.astype(np.float32)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    rr = r + 1.0
    m = {}
    m["red"] = (r > 90) & (g < 0.25 * r) & (b < 0.25 * r)
    m["purple"] = (b > 120) & (b > r) & (r > 1.3 * g) & (g < 0.7 * b)
    m["teal"] = (g > 50) & (np.abs(b / (g + 1) - 1.02) < 0.08) & (r / (g + 1) > 0.35) & (r / (g + 1) < 0.62)
    m["pink"] = (r > 130) & (np.abs(g / rr - 0.69) < 0.06) & (np.abs(b / rr - 0.57) < 0.06)
    m["shaft"] = (r > 150) & (b > r) & (b - r < 35) & (g > 0.95 * r)
    return m


def _largest_component(mask, min_px=3):
    from scipy import ndimage
    lab, n = ndimage.label(ndimage.binary_dilation(mask, iterations=2))
    lab = lab * mask
    if n == 0:
        return mask & False
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    k = int(np.argmax(sizes)) + 1
    if sizes[k - 1] < min_px:
        return mask & False
    return lab == k


def world_points(o, cam, key, largest=True, zmax=0.25):
    m = masks(o.rgb[cam])[key]
    if largest:
        m = _largest_component(m)
    P = o.cloud(cam)[m]
    P = P[np.isfinite(P).all(1)]
    P = P[(P[:, 2] > -0.02) & (P[:, 2] < zmax)]
    return P


def scene(o, cams=("wrist", "front")):
    """Locate cube, target and hook in world frame from the current observation."""
    out = {}
    for key in ["red", "purple", "teal", "pink"]:
        pts = []
        for cam in cams:
            if cam in o.rgb:
                P = world_points(o, cam, key)
                if len(P) >= 3:
                    pts.append((cam, P))
        out[key] = pts
    res = {"raw": out}
    # cube: top face points from each camera
    cube = None
    for cam, P in out["red"]:
        top = P[P[:, 2] > P[:, 2].max() - 0.008]
        c = np.median(top, 0)
        if cube is None:
            cube = c
        res.setdefault("cube_by_cam", {})[cam] = c
    res["cube_top"] = cube
    tgt = None
    for cam, P in out["purple"]:
        c = np.median(P, 0)
        res.setdefault("target_by_cam", {})[cam] = c
        if tgt is None:
            tgt = c
    res["target"] = tgt
    h = find_hook(o, cams)
    if h is not None:
        res["hook"] = h
        return res
    best = None
    for cam in []:
        pk = [P for c, P in out["pink"] if c == cam]
        tl = [P for c, P in out["teal"] if c == cam]
        if pk and tl and len(pk[0]) >= 25 and len(tl[0]) >= 15:
            if best is None or len(pk[0]) > len(best[1]):
                best = (cam, pk[0], tl[0])
    if best is not None:
        cam, pink, teal = best
        handle = np.median(teal, 0)
        head = pink.mean(0)
        s = head[:2] - handle[:2]
        L = np.linalg.norm(s)
        s = s / L
        perp = np.array([-s[1], s[0]])  # +90 deg
        side = np.mean((pink[:, :2] - handle[:2]) @ perp)
        c = perp if side > 0 else -perp
        res["hook"] = dict(cam=cam, handle=handle, head=head, s=s, c=c, length=L, chirality=1 if side > 0 else -1,
                           n_pink=len(pink), n_teal=len(teal),
                           pink_s=np.ptp((pink[:, :2] - handle[:2]) @ s), pink_c=np.ptp((pink[:, :2] - handle[:2]) @ c))
    return res


def obs_from_manifest_frame(frame, root="/workspace"):
    """Build an Obs from a saved frame record (offline tests / conditioning video)."""
    import os
    rgb, depth = {}, {}
    for f in frame["files"]:
        p = os.path.join(root, f["path"])
        arr = np.array(Image.open(p))
        if f["kind"] == "rgb":
            rgb[f["camera"]] = arr[..., :3].astype(np.uint8)
        else:
            depth[f["camera"]] = arr.astype(np.float32) * 0.001
    return Obs(frame, rgb, depth)


def cube_estimate(o, cams=("front", "wrist")):
    """Cube top-face centre: prefer the camera with the most top-face pixels; average if they agree."""
    ests = []
    for cam in cams:
        if cam not in o.rgb:
            continue
        P = world_points(o, cam, "red", zmax=0.3)
        if len(P) < 8:
            continue
        top = P[P[:, 2] > P[:, 2].max() - 0.008]
        if len(top) < 5:
            continue
        ests.append((len(top), cam, np.median(top, 0), P))
    if not ests:
        return None
    ests.sort(key=lambda e: -e[0])
    best = ests[0][2]
    if len(ests) > 1 and np.linalg.norm(ests[1][2][:2] - best[:2]) < 0.012 and ests[1][0] > 0.4 * ests[0][0]:
        best = (best + ests[1][2]) / 2
    return best


def target_estimate(o, cams=("wrist", "front")):
    ests = []
    for cam in cams:
        if cam not in o.rgb:
            continue
        P = world_points(o, cam, "purple")
        if len(P) >= 5:
            ests.append((len(P), np.median(P, 0)))
    if not ests:
        return None
    ests.sort(key=lambda e: -e[0])
    return ests[0][1]


def obstacle_points(o, cams=("front", "wrist"), exclude_xy=(), zmin=0.012, zmax=0.3):
    """World points that stand above the table and are not task objects.
    exclude_xy: list of (centre_xy, radius) discs to drop (cube, target, ...)."""
    from scipy import ndimage
    out = []
    for cam in cams:
        if cam not in o.rgb:
            continue
        ms = masks(o.rgb[cam])
        known = ms["red"] | ms["purple"]
        known = ndimage.binary_dilation(known, iterations=2)
        P = o.cloud(cam)[~known]
        P = P[np.isfinite(P).all(1)]
        keep = (P[:, 2] > zmin) & (P[:, 2] < zmax) & (P[:, 0] > -0.56) & (P[:, 0] < 0.35) & (np.abs(P[:, 1]) < 0.55)
        P = P[keep]
        out.append(P)
    P = np.concatenate(out) if out else np.zeros((0, 3))
    for c, r in exclude_xy:
        P = P[np.linalg.norm(P[:, :2] - np.asarray(c)[:2], axis=1) > r]
    # voxel-thin for speed
    if len(P) > 6000:
        P = P[np.random.default_rng(0).choice(len(P), 6000, replace=False)]
    return P


def seg_dist(P, a, b):
    """xy distance of points P (N,2) to segment a-b."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    d = b - a
    L2 = float(d @ d) + 1e-12
    t = np.clip(((P - a) @ d) / L2, 0, 1)
    proj = a + t[:, None] * d
    return np.linalg.norm(P - proj, axis=1)


def components(o, cam, key, min_px=4):
    from scipy import ndimage
    m = masks(o.rgb[cam])[key]
    lab, n = ndimage.label(ndimage.binary_dilation(m, iterations=2))
    lab = lab * m
    P = o.cloud(cam)
    comps = []
    for k in range(1, n + 1):
        mm = lab == k
        if mm.sum() < min_px:
            continue
        Q = P[mm]
        Q = Q[np.isfinite(Q).all(1)]
        Q = Q[(Q[:, 2] > -0.02) & (Q[:, 2] < 0.3)]
        if len(Q) < min_px:
            continue
        comps.append(dict(n=len(Q), P=Q, c=np.median(Q, 0), diag=float(np.linalg.norm(np.ptp(Q[:, :2], 0)))))
    return comps


def find_hook(o, cams=("wrist", "front")):
    """The L-hook: a pink head with a large extent (>=5.5cm) paired with a teal handle."""
    best = None
    for cam in cams:
        if cam not in o.rgb:
            continue
        pinks = [c for c in components(o, cam, "pink", 6) if c["diag"] >= 0.055 and c["diag"] < 0.16]
        teals = [c for c in components(o, cam, "teal", 6) if c["diag"] < 0.12]
        if not pinks or not teals:
            continue
        sm = masks(o.rgb[cam])["shaft"]
        SP = o.cloud(cam)[sm]
        SP = SP[np.isfinite(SP).all(1)]
        SP = SP[(SP[:, 2] > -0.02) & (SP[:, 2] < 0.1)][:, :2]
        for pk in pinks:
            for tl in teals:
                a, b = tl["c"][:2], pk["c"][:2]
                d = np.linalg.norm(b - a)
                if d < 0.07 or d > 0.45:
                    continue
                cov = 0.0
                if len(SP):
                    ts = np.linspace(0.2, 0.8, 13)
                    hits = [np.min(np.linalg.norm(SP - (a + t * (b - a)), axis=1)) < 0.02 for t in ts]
                    cov = float(np.mean(hits))
                score = cov * 100 + pk["n"] * 0.1 - d * 10
                if best is None or score > best[0]:
                    best = (score, cam, pk, tl, cov)
    if best is None:
        return None
    _, cam, pk, tl, cov = best
    pink = pk["P"]
    handle = tl["c"]
    head = pink.mean(0)
    s = head[:2] - handle[:2]
    L = float(np.linalg.norm(s))
    s = s / L
    perp = np.array([-s[1], s[0]])
    side = np.mean((pink[:, :2] - handle[:2]) @ perp)
    c = perp if side > 0 else -perp
    return dict(cam=cam, handle=handle, head=head, s=s, c=c, length=L, chirality=1 if side > 0 else -1,
                n_pink=len(pink), n_teal=tl["n"], pink_diag=pk["diag"], shaft_cov=cov, teal_P=tl["P"], pink_P=pink)
