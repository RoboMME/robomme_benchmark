import math
import numpy as np
import cv2
from robot import decode_images, rotate_yaw, finger_angle, wrap, jtext
from percep import quat_to_R

TOP_Z_MIN = 0.064     # top face of container ~0.073 (base step at 0.05)
GRASP_Z = 0.063
HOVER_Z = 0.20
LIFT_Z = 0.25         # pick must lift container well above table (>0.1 m)
PLACE_Z = 0.064

def wrist_cloud(ctx):
    res = ctx.call("robomme_observe")
    meta, ims = decode_images(res)
    d = ims[("wrist", "depth")]
    K = meta["camera_intrinsics"]["wrist"]["matrix"]
    pose = meta["camera_poses"]["wrist"]
    H, W = d.shape
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    fx, cx, fy, cy = K[0][0], K[0][2], K[1][1], K[1][2]
    pc = np.stack([(us - cx) * d / fx, -(vs - cy) * d / fy, -d], -1).reshape(-1, 3)
    P = pc @ quat_to_R(pose["quaternion_wxyz"]).T + np.array(pose["position_m"])
    return P[d.reshape(-1) > 0], meta

def top_clusters(P, zmin=TOP_Z_MIN, zmax=0.11, cell=0.003):
    sel = P[(P[:, 2] > zmin) & (P[:, 2] < zmax)]
    if len(sel) < 20:
        return []
    x0, y0 = sel[:, 0].min(), sel[:, 1].min()
    ix = ((sel[:, 0] - x0) / cell).astype(int)
    iy = ((sel[:, 1] - y0) / cell).astype(int)
    grid = np.zeros((iy.max() + 1, ix.max() + 1), np.uint8)
    grid[iy, ix] = 1
    grid = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab = cv2.connectedComponents(grid, connectivity=8)
    ids = lab[iy, ix]
    out = []
    for k in range(1, n):
        pts = sel[ids == k]
        if len(pts) < 20:
            continue
        (cx, cy), (w, h), ang = cv2.minAreaRect(pts[:, :2].astype(np.float32))
        out.append(dict(c=(float(cx), float(cy)), size=(float(w), float(h)), ang=float(ang),
                        topz=float(np.percentile(pts[:, 2], 90)), n=len(pts)))
    return out

def locate_container(ctx, xy, maxdist=0.04, log=print):
    """Container top face nearest xy, from the wrist depth cloud. Returns dict or None."""
    P, meta = wrist_cloud(ctx)
    cl = top_clusters(P)
    best = None
    for c in cl:
        dd = math.hypot(c["c"][0] - xy[0], c["c"][1] - xy[1])
        if max(c["size"]) > 0.07 or max(c["size"]) < 0.02:
            continue
        if best is None or dd < best[0]:
            best = (dd, c)
    if best is None or best[0] > maxdist:
        log("locate: none near", xy, "clusters", [(round(c["c"][0], 3), round(c["c"][1], 3), round(max(c["size"]), 3)) for c in cl])
        return None
    c = best[1]
    log("locate: centre %.4f %.4f size %.4f %.4f ang %.1f topz %.4f n %d d %.4f" % (c["c"][0], c["c"][1], c["size"][0], c["size"][1], c["ang"], c["topz"], c["n"], best[0]))
    return c

def pick_container(R, xy, log=print):
    ctx = R.ctx
    s = R.state()
    if s[2] < 0.14:
        R.move_to([s[0], s[1], 0.16], grip=-1, tol=0.01, max_steps=30)
    R.move_to([xy[0], xy[1], HOVER_Z], grip=-1, tol=0.006)
    loc = locate_container(ctx, xy, log=log)
    cx, cy = xy
    if loc is not None:
        cx, cy = loc["c"]
        rotate_yaw(R, math.radians(loc["ang"]))
    R.move_to([cx, cy, 0.12], grip=-1, tol=0.004)
    loc2 = locate_container(ctx, (cx, cy), maxdist=0.02, log=log)
    if loc2 is not None:
        cx, cy = loc2["c"]
        rotate_yaw(R, math.radians(loc2["ang"]))
        R.move_to([cx, cy, 0.12], grip=-1, tol=0.004, max_steps=20)
    s = R.state()
    log("yaw residual deg %.1f" % math.degrees(wrap(math.radians(loc2["ang"] if loc2 else (loc["ang"] if loc else 0)) - finger_angle(s), math.pi / 2)))
    R.move_to([cx, cy, GRASP_Z], grip=-1, tol=0.004, max_steps=40)
    R.hold(8, grip=1)
    R.move_to([cx, cy, LIFT_Z], grip=1, tol=0.008, max_steps=60)
    R.hold(4, grip=1)
    s = R.state()
    opening = s[6] + s[7]
    held = opening > 0.01
    log("pick: opening %.4f held %s z %.3f step %d" % (opening, held, s[2], R.step_id))
    return held, (cx, cy)

def place_back(R, xy, log=print):
    R.move_to([xy[0], xy[1], PLACE_Z + 0.01], grip=1, tol=0.005, max_steps=60)
    R.hold(6, grip=-1)
    s = R.state()
    R.move_to([s[0], s[1], 0.16], grip=-1, tol=0.01, max_steps=30)

def map_containers(ctx, log=print):
    """Container top-face centres seen in the current wrist view."""
    P, meta = wrist_cloud(ctx)
    out = []
    for c in top_clusters(P):
        if 0.028 < max(c["size"]) < 0.05 and 0.064 < c["topz"] < 0.085:
            out.append(c["c"])
    log("map: %d containers" % len(out), [tuple(round(v, 3) for v in c) for c in out])
    return out

def free_spot(occupied, near, button, clearance=0.105, log=print):
    best = None
    for x in np.arange(-0.22, 0.061, 0.02):
        for y in np.arange(-0.30, 0.301, 0.02):
            if math.hypot(x + 0.615, y) > 0.70:
                continue
            if math.hypot(x - button[0], y - button[1]) < 0.11:
                continue
            if any(math.hypot(x - o[0], y - o[1]) < clearance for o in occupied):
                continue
            d = math.hypot(x - near[0], y - near[1])
            if best is None or d < best[0]:
                best = (d, (float(x), float(y)))
    if best is None:
        if clearance > 0.08:
            return free_spot(occupied, near, button, clearance - 0.01, log)
        return None
    log("free spot", tuple(round(v, 3) for v in best[1]), "dist", round(best[0], 3), "clearance", clearance)
    return best[1]

def place_at(R, xy, log=print):
    s = R.state()
    R.move_to([xy[0], xy[1], 0.22], grip=1, tol=0.01, max_steps=60)
    R.move_to([xy[0], xy[1], PLACE_Z + 0.012], grip=1, tol=0.005, max_steps=60)
    R.hold(6, grip=-1)
    s = R.state()
    R.move_to([s[0], s[1], 0.16], grip=-1, tol=0.01, max_steps=30)
