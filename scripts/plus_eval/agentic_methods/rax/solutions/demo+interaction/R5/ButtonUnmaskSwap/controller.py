"""ButtonUnmaskSwap controller: track containers from reset, press both buttons, wait for
shuffle to stop, then lift the containers hiding the instructed colours in order."""
import sys, os, re, time, json, math, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy import ndimage
from obs import world_points
from core import (band_grid, low_objects, static_mask, WorldTracker, BlobTracker, cells_to_xy, RES, X0, Y0)
from robot import Robot, Ended

COLS = ("red", "green", "blue", "yellow", "purple", "orange", "black", "white", "gray")
SAFE_Z = 0.22
LIFT = 0.20
HOME = None


def colour_of_rgb(rgb):
    r, g, b = [int(x) for x in rgb]
    mx, mn = max(r, g, b), min(r, g, b)
    if mx - mn < 30:
        return "black" if mx < 90 else ("white" if mx > 190 else "gray")
    if r > 150 and g < 80 and b < 80: return "red"
    if g > 150 and r < 100 and b < 100: return "green"
    if b > 150 and r < 80 and g < 80: return "blue"
    if r > 150 and g > 200 and b < 80: return "yellow"
    if r > 130 and b > 150 and g < 120: return "purple"
    if r > 150 and 100 < g < 200 and b < 80: return "orange"
    return "other"


def parse_order(instr):
    found = []
    for m in re.finditer(r"hiding the (\w+) cube", instr):
        found.append(m.group(1).lower())
    return found


def cube_colours(robot, cubes):
    """Colour label of each low cluster from projected pixel colours (wrist then front)."""
    meta, imgs = robot.meta, robot.imgs
    out = []
    pts = {}
    for cam in ("wrist", "front"):
        pw, uv = world_points(meta, imgs, cam)
        s = (pw[:, 2] > 0.012) & (pw[:, 2] < 0.06)
        pts[cam] = (pw[s], uv[s], imgs[(cam, "rgb")])
    for (xy, n) in cubes:
        votes = {}
        for cam, (pw, uv, im) in pts.items():
            d = np.linalg.norm(pw[:, :2] - np.array(xy)[None], axis=1)
            sel = d < 0.015
            for (u, v) in uv[sel][::2]:
                c = colour_of_rgb(im[v, u])
                votes[c] = votes.get(c, 0) + 1
        votes.pop("other", None)
        out.append(max(votes, key=votes.get) if votes else "unknown")
    return out


def button_caps(robot, buttons):
    """Refine button centres using the top (cap) points."""
    meta, imgs = robot.meta, robot.imgs
    res = []
    for (xy, n) in buttons:
        acc = []
        for cam in ("wrist", "front"):
            pw, uv = world_points(meta, imgs, cam)
            d = np.linalg.norm(pw[:, :2] - np.array(xy)[None], axis=1)
            s = (d < 0.05) & (pw[:, 2] > 0.012) & (pw[:, 2] < 0.07)
            if s.sum():
                zt = np.percentile(pw[s, 2], 90)
                s2 = s & (pw[:, 2] > zt - 0.004)
                acc.append(pw[s2])
        P = np.concatenate(acc) if acc else np.zeros((0, 3))
        if len(P) >= 10:
            res.append((float(np.median(P[:, 0])), float(np.median(P[:, 1])), float(np.percentile(P[:, 2], 90))))
        else:
            res.append((xy[0], xy[1], 0.034))
    return res


def init_tracking(robot, seeds, static):
    """Wait for the containers to appear over the cubes, then start the tracker."""
    T = BlobTracker(seeds, static=static)
    for k in range(40):
        G = band_grid(robot.meta, robot.imgs)
        if static is not None:
            G = G & ~static
        if G.sum() > 60 * len(seeds):
            break
        robot.step([0, 0, 0, 0, 0, 0, -1])
    robot.tracker = T
    for _ in range(3):  # converge on current frame
        T.update(band_grid(robot.meta, robot.imgs))
    robot.log(ev="track_init", step=robot.step_id, c=T.c)
    return T


def press_button(robot, cap):
    x, y, zc = cap
    robot.move_to([x, y, 0.15], grip=1.0, tol=0.01, max_steps=40)
    robot.move_to([x, y, zc + 0.03], grip=1.0, tol=0.006, max_steps=25, vmax=0.5)
    # push down to below the cap
    for _ in range(12):
        d = np.array([x, y, zc - 0.025]) - robot.pos
        a = np.clip(d / 0.05 * 0.9, -0.4, 0.4)
        robot.step([a[0], a[1], a[2], 0, 0, 0, 1.0])
    zmin = robot.pos[2]
    robot.move_to([x, y, 0.15], grip=1.0, tol=0.015, max_steps=30)
    robot.log(ev="press", step=robot.step_id, cap=cap, zmin=zmin)


def container_yaw(robot, xy):
    """Estimate container yaw (mod 90deg) from its top-face points."""
    meta, imgs = robot.meta, robot.imgs
    acc = []
    for cam in ("wrist", "front"):
        pw, _ = world_points(meta, imgs, cam)
        d = np.linalg.norm(pw[:, :2] - np.array(xy)[None], axis=1)
        s = (d < 0.045) & (pw[:, 2] > 0.05) & (pw[:, 2] < 0.12)
        acc.append(pw[s])
    P = np.concatenate(acc)
    if len(P) < 30:
        return None, None, None
    zt = np.percentile(P[:, 2], 95)
    top = P[P[:, 2] > zt - 0.012][:, :2]
    if len(top) < 20:
        top = P[:, :2]
    c = top.mean(0)
    Q = top - c
    best = None
    for deg in np.arange(0, 90, 2.0):
        t = math.radians(deg)
        R = np.array([[math.cos(t), math.sin(t)], [-math.sin(t), math.cos(t)]])
        q = Q @ R.T
        lo = np.percentile(q, 3, axis=0); hi = np.percentile(q, 97, axis=0)
        area = np.prod(hi - lo)
        if best is None or area < best[0]:
            best = (area, t, (lo + hi) / 2 @ R)
    return best[1], float(zt), c + best[2]


def wrap45(a):
    return (a + math.pi / 4) % (math.pi / 2) - math.pi / 4


def pick_container(robot, xy, colour):
    robot.move_to([xy[0], xy[1], SAFE_Z], grip=-1.0, tol=0.015, max_steps=40, vmax=0.8)
    yaw, ztop, cen = container_yaw(robot, xy)
    if cen is not None and len(np.shape(cen)) == 1 and np.linalg.norm(cen - xy) < 0.03:
        xy = cen
    cur = robot.yaw()
    tgt = cur + wrap45((yaw if yaw is not None else cur) - cur)
    ztop = ztop if ztop is not None else 0.073
    ztop = float(np.clip(ztop, 0.06, 0.11))
    robot.log(ev="pick_plan", step=robot.step_id, colour=colour, xy=xy, yaw=yaw, ztop=ztop, tgt=tgt)
    robot.move_to([xy[0], xy[1], ztop + 0.05], grip=-1.0, tol=0.008, max_steps=30, yaw=tgt, vmax=0.7,
                  yaw_tol=0.05)
    zg = max(ztop - 0.035, 0.035)
    robot.move_to([xy[0], xy[1], zg], grip=-1.0, tol=0.006, max_steps=20, yaw=tgt, vmax=0.5, yaw_tol=0.08)
    robot.hold(6, grip=1.0)
    fingers = float(robot.state[6] + robot.state[7])
    robot.move_to([xy[0], xy[1], zg + LIFT], grip=1.0, tol=0.01, max_steps=25, vmax=0.6)
    robot.hold(2, grip=1.0)
    fingers2 = float(robot.state[6] + robot.state[7])
    robot.log(ev="lifted", step=robot.step_id, colour=colour, fingers=fingers, fingers_after=fingers2,
              ee=robot.pos)
    # set it back down roughly where it was and release
    robot.move_to([xy[0], xy[1], zg + 0.015], grip=1.0, tol=0.01, max_steps=15, vmax=0.5)
    robot.hold(4, grip=-1.0)
    robot.move_to([robot.pos[0], robot.pos[1], ztop + 0.08], grip=-1.0, tol=0.02, max_steps=10, vmax=0.8)


def wait_still(robot, max_step, still_need=12, thresh=0.0006):
    still = 0
    while robot.step_id < max_step:
        robot.step([0, 0, 0, 0, 0, 0, -1])
        sp = np.linalg.norm(robot.tracker.v, axis=1).max()
        still = still + 1 if sp < thresh else 0
        if still >= still_need:
            break
    robot.log(ev="still", step=robot.step_id)


def main(ctx, scene=None, robot=None):
    t0 = time.time()
    os.makedirs("output", exist_ok=True)
    if robot is None:
        robot = Robot(ctx)
    instr = robot.meta.get("instruction", "")
    order = parse_order(instr) or ["blue", "green", "red"]
    robot.log(ev="start", instr=instr, order=order, state=robot.state, step=robot.step_id)
    try:
        home = robot.pos.copy()
        cur = (robot.meta, robot.imgs)
        if scene is not None:
            robot.meta, robot.imgs = scene
        static = static_mask(robot.meta, robot.imgs)
        cubes, buttons = low_objects(robot.meta, robot.imgs)
        cols = cube_colours(robot, cubes)
        caps = button_caps(robot, buttons)
        robot.log(ev="scene", cubes=[c[0] for c in cubes], cols=cols, buttons=caps, static=int(static.sum()))
        robot.meta, robot.imgs = cur
        seeds = [c[0] for c in cubes]
        T = init_tracking(robot, seeds, static)
        targets = {}
        for col in order:
            idx = [i for i, c in enumerate(cols) if c == col]
            targets[col] = idx[0] if idx else None
        robot.log(ev="targets", targets=targets)
        # press buttons, nearest first
        caps_sorted = sorted(caps, key=lambda c: np.linalg.norm(np.array(c[:2]) - robot.pos[:2]))
        for cap in caps_sorted[:2]:
            press_button(robot, cap)
        robot.move_to(home, grip=-1.0, tol=0.02, max_steps=40)
        wait_still(robot, max_step=700)
        final = T.c.copy()
        robot.log(ev="final_tracks", c=final, step=robot.step_id)
        robot.tracker = None   # freeze: picking disturbs containers
        for col in order:
            k = targets.get(col)
            if k is None:
                robot.log(ev="no_target", colour=col)
                continue
            pick_container(robot, final[k].copy(), col)
        robot.hold(5)
    except Ended:
        pass
    except Exception as e:
        robot.log(ev="error", err=repr(e), tb=traceback.format_exc())
        print("controller error", repr(e))
    robot.log(ev="done", step=robot.step_id, ended=robot.ended, task_complete=robot.task_complete,
              wall=time.time() - t0)
