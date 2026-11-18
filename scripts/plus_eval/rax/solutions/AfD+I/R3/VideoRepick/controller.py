"""VideoRepick controller: track the picked object through the conditioning video,
pick & place it twice, then press the button."""
import json, sys, time, traceback
from pathlib import Path
import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from robot import Robot, deproject, project, world_map, log, LOG, WS  # noqa: E402
from tracker import Seg, rgb2hsv, hue_dist  # noqa: E402
from perception import find_target  # noqa: E402

OUT = WS / "output"
SAFE_Z = 0.15
GRASP_DZ = 0.02     # TCP height above table for grasp/place


def save_json(name, obj):
    try:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / name).write_text(json.dumps(obj, default=lambda o: np.asarray(o).tolist()))
    except Exception as e:  # never crash on logging
        log("[save] failed", e)


def obj_mask(img, hue):
    seg = Seg(); seg.obj_hue = hue
    h, s, v = rgb2hsv(img)
    m = (s > 0.3) & (v > 0.12)
    if hue is not None:
        m &= hue_dist(h, hue) < 35
    # exclude the table hue (dominant)
    gh = np.median(h[s > 0.2]) if (s > 0.2).any() else 20.0
    m &= hue_dist(h, gh) > 25
    m &= ~((v > 0.72) & (s < 0.5))
    return ndimage.binary_opening(m, iterations=1)


def wrist_refine(R, est_xy, hue, radius=0.05, min_px=15):
    """Using the current wrist RGB-D, return refined object (x, y, top_z) of the object
    blob cluster nearest est_xy (components merged when within 2 cm)."""
    meta = R.meta
    img = R.imgs[("wrist", "rgb")]; dep = R.imgs[("wrist", "depth")]
    W = world_map(meta, dep, "wrist")
    m = obj_mask(img, hue)
    m[200:, :90] = False; m[190:, 215:] = False  # gripper fingers region (bottom corners)
    m &= ~np.isnan(W).any(-1)
    m &= np.nan_to_num(W[..., 2], nan=-1) > 0.004
    lab, n = ndimage.label(m)
    comps = []
    for k in range(1, n + 1):
        sel = lab == k
        if sel.sum() < min_px:
            continue
        P = W[sel]
        comps.append((np.median(P[:, :2], 0), P))
    if not comps:
        log("[refine] no object components")
        return None
    est = np.asarray(est_xy, float)[:2]
    # union-find merge by centroid distance
    parent = list(range(len(comps)))
    def f(i):
        while parent[i] != i:
            i = parent[i]
        return i
    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            if np.linalg.norm(comps[i][0] - comps[j][0]) < 0.02:
                parent[f(i)] = f(j)
    groups = {}
    for i in range(len(comps)):
        groups.setdefault(f(i), []).append(i)
    best = None
    for g, idx in groups.items():
        P = np.concatenate([comps[i][1] for i in idx])
        c = np.median(P[:, :2], 0)
        d = float(np.linalg.norm(c - est))
        if best is None or d < best[0]:
            best = (d, c, P)
    d, c, P = best
    if d > radius:
        log(f"[refine] nearest object {d:.3f} m from estimate; reject")
        return None
    top = float(np.percentile(P[:, 2], 95))
    log(f"[refine] obj xy={c.round(3).tolist()} top={top:.3f} n={len(P)} d={d:.3f} groups={len(groups)}")
    return np.array([c[0], c[1], top])


TRANSIT_Z = 0.25


def transit(R, xy, grip):
    """Lift to a safe height, then move laterally (avoids tall occluders such as pillars)."""
    pos = R.state[:3]
    if np.linalg.norm(pos[:2] - np.asarray(xy)[:2]) < 0.03:
        return
    z = max(pos[2], TRANSIT_Z)
    if pos[2] < TRANSIT_Z - 0.01:
        R.move_to([pos[0], pos[1], z], grip=grip)
    R.move_to([xy[0], xy[1], z], grip=grip)


def pick(R, xy, hue, label="pick"):
    """Approach above xy, refine with wrist camera twice, descend and grasp. Returns grasp xyz or None."""
    xy = np.asarray(xy, float)[:2]
    transit(R, xy, -1)
    R.move_to([xy[0], xy[1], 0.22], grip=-1)
    r = wrist_refine(R, xy, hue, radius=0.05)
    if r is None:
        return None
    xy = r[:2]
    R.move_to([xy[0], xy[1], 0.13], grip=-1)
    r2 = wrist_refine(R, xy, hue, radius=0.025)
    if r2 is not None:
        xy = r2[:2]; top = r2[2]
    else:
        top = r[2]
    gz = float(np.clip(top * 0.5, 0.012, 0.03))
    R.move_to([xy[0], xy[1], gz + 0.06], grip=-1)
    R.move_to([xy[0], xy[1], gz], grip=-1, speed=0.5)
    R.gripper(1, n=10)
    R.move_to([xy[0], xy[1], SAFE_Z], grip=1, speed=0.6)
    op = R.opening()
    log(f"[{label}] grasp at {np.round([xy[0], xy[1], gz], 3).tolist()} opening after lift={op:.4f}")
    if op < 0.004:
        return None
    return np.array([xy[0], xy[1], gz])


def place(R, xyz, label="place"):
    x, y, gz = xyz
    R.move_to([x, y, gz + 0.05], grip=1)
    R.move_to([x, y, gz + 0.008], grip=1, speed=0.4)
    R.gripper(-1, n=8)
    R.move_to([x, y, SAFE_Z], grip=-1, speed=0.6)
    log(f"[{label}] released at {np.round(xyz, 3).tolist()} step={R.meta.get('step_id')}")


def _button_candidates(frac, W, min_a=6, max_a=900):
    lab, n = ndimage.label(frac)
    out = []
    for k in range(1, n + 1):
        sel = lab == k
        a = int(sel.sum())
        if a < min_a or a > max_a:
            continue
        ys, xs = np.nonzero(sel)
        c = None
        if W is not None:
            P = W[sel]; P = P[~np.isnan(P).any(1)]
            if len(P) == 0:
                continue
            c = np.median(P, 0)
            if c[2] > 0.06 or c[2] < -0.02:
                continue
        out.append((a, c, (float(ys.mean()), float(xs.mean()))))
    return out


def geometric_button(meta, rgb, depth, cam, min_area):
    """Low (median z 8-26 mm), compact, unsaturated blob on the table."""
    W = world_map(meta, depth, cam)
    z = np.nan_to_num(W[..., 2], nan=1.0)
    m = (z > 0.006) & (z < 0.06)
    if cam == "front":
        m[:25] = False
    else:
        m[190:, :100] = False; m[190:, 200:] = False
    lab, n = ndimage.label(m)
    hh, ss, vv = rgb2hsv(rgb)
    best = None
    for k in range(1, n + 1):
        sel = lab == k
        a = int(sel.sum())
        if a < min_area or a > 1500:
            continue
        ys, xs = np.nonzero(sel)
        hgt, wid = ys.ptp() + 1, xs.ptp() + 1
        if min(hgt, wid) / max(hgt, wid) < 0.4:
            continue
        P = W[sel]
        mz = float(np.median(P[:, 2]))
        if not (0.008 <= mz <= 0.026):
            continue
        sat = float(np.median(ss[sel]))
        if best is None or sat < best[0]:
            best = (sat, np.median(P, 0), (float(ys.mean()), float(xs.mean())), a, mz)
    if best is not None and best[0] < 0.45:
        log(f"[button] geometric {cam} px={np.round(best[2], 1).tolist()} area={best[3]} medz={best[4]:.3f} sat={best[0]:.2f} world={best[1].round(3).tolist()}")
        return best[1]
    return None


def find_button(meta, front, fdepth, video=None, wrist=None):
    """Button = white base + gray cap on the table. Lighting can tint it, so use the fraction
    of conditioning-video frames in which each pixel is bright and unsaturated."""
    W = world_map(meta, fdepth, "front") if fdepth is not None else None
    if W is not None:
        r = geometric_button(meta, front, fdepth, "front", 60)
        if r is not None:
            return r
    if wrist is not None:
        r = geometric_button(meta, wrist[0], wrist[1], "wrist", 30)
        if r is not None:
            return r
    tries = []
    if video:
        acc = None
        for f in video[:: max(1, len(video) // 60)]:
            h, s, v = rgb2hsv(f)
            g = (s < 0.18) & (v > 0.75)
            acc = g.astype(np.float32) if acc is None else acc + g
        tries.append(acc / max(1, len(video[:: max(1, len(video) // 60)])) > 0.15)
    h, s, v = rgb2hsv(front)
    tries.append((s < 0.15) & (v > 0.78))
    tries.append((s < 0.55) & (v > 0.82))
    for frac in tries:
        frac = frac.copy(); frac[:30] = False
        if W is not None:
            z = np.nan_to_num(W[..., 2], nan=1.0)
            frac &= z < 0.06
        cands = _button_candidates(frac, W)
        if cands:
            a, c, px = max(cands, key=lambda t: t[0])
            if c is None and W is not None:
                continue
            log(f"[button] front px={np.round(px, 1).tolist()} area={a} world={None if c is None else c.round(3).tolist()}")
            return c if W is not None else px
    log("[button] not found")
    return None


def wrist_button(R, est):
    img = R.imgs[("wrist", "rgb")]; dep = R.imgs[("wrist", "depth")]
    W = world_map(R.meta, dep, "wrist")
    h, s, v = rgb2hsv(img)
    m = (s < 0.15) & (v > 0.7)
    m[180:, :100] = False; m[180:, 200:] = False
    m &= ~np.isnan(W).any(-1)
    P = W[m]
    if len(P) < 20:
        return None
    P = P[(P[:, 2] > 0.0) & (P[:, 2] < 0.08)]
    d = np.linalg.norm(P[:, :2] - np.asarray(est[:2])[None], axis=1)
    sel = P[d < 0.05]
    if len(sel) < 20:
        return None
    c = np.median(sel, 0); top = float(np.percentile(sel[:, 2], 95))
    log(f"[button] wrist refine xy={c[:2].round(3).tolist()} top={top:.3f}")
    return np.array([c[0], c[1], top])


def press_button(R, b):
    transit(R, b[:2], 1)
    R.move_to([b[0], b[1], 0.2], grip=1)
    r = wrist_button(R, b)
    if r is not None:
        b = r
    R.move_to([b[0], b[1], b[2] + 0.06], grip=1)
    # descend in small steps until blocked or well below top
    target = b[2] - 0.02
    for _ in range(30):
        if R.ended:
            break
        z = R.state[2]
        if z <= target:
            break
        R.step([[0, 0, -0.3, 0, 0, 0, 1]] * 2)
    log(f"[button] pressed, z={R.state[2]:.3f} step={R.meta.get('step_id')} ended={R.ended}")
    if not R.ended:
        R.move_to([b[0], b[1], b[2] + 0.08], grip=1, max_steps=20)


def main(ctx):
    try:
        _main(ctx)
    except Exception:
        log("[main] unexpected error", traceback.format_exc())
    save_json("controller_log.json", LOG)


def _main(ctx):
    t0 = time.time()
    R = Robot(ctx)
    meta = R.observe()
    log("[main] instruction:", meta.get("instruction"), "limit", meta.get("episode_limits"))
    front0 = R.imgs[("front", "rgb")].copy(); fdep0 = R.imgs[("front", "depth")].copy()
    # button first (used to exclude its purple-lit pixels from object tracking)
    try:
        from perception import load_sequence
        vidF, _, _ = load_sequence(meta)
        button = find_button(meta, front0, fdep0, video=vidF[:-1], wrist=(R.imgs[("wrist", "rgb")], R.imgs[("wrist", "depth")]))
    except Exception:
        log("[main] button search failed", traceback.format_exc()); button = None; vidF = None
    hue = None
    target_xy = None
    wres = None
    try:
        from tracker_world import identify_world
        wres = identify_world(meta, log=log, exclude=None if button is None else button[:2])
    except Exception:
        log("[main] world tracking failed", traceback.format_exc()); wres = None
    log(f"[main] world tracking done in {time.time() - t0:.1f}s")
    res = None
    try:
        res = find_target(meta)
    except Exception:
        log("[main] 2D tracking failed", traceback.format_exc()); res = None
    log(f"[main] tracking done in {time.time() - t0:.1f}s")
    xy2d = None
    if res is not None:
        rr, cc = res["target_rc"]
        p = deproject(meta, fdep0, "front", cc, rr, win=2)
        hue = res.get("hue")
        if p is not None:
            cam = np.array(meta["camera_poses"]["front"]["position_m"], float)
            dvec = p[:2] - cam[:2]; dvec /= (np.linalg.norm(dvec) + 1e-9)
            xy2d = p[:2] + 0.015 * dvec   # front view sees the near face; shift to centre
        save_json("tracking.json", res)
    if wres is not None:
        save_json("wtracking.json", wres)
        target_xy = np.array(wres["target_xy"], float)
        if hue is None:
            hue = wres.get("hue")
        if xy2d is not None:
            log(f"[main] world vs 2D target distance {np.linalg.norm(target_xy - xy2d):.3f} m")
    else:
        target_xy = xy2d
    if hue is None:
        try:
            s = Seg(); s.calibrate([front0]); hue = s.obj_hue
        except Exception:
            hue = None
    log("[main] target_xy", None if target_xy is None else target_xy.round(3).tolist(), "hue", hue)
    if target_xy is None:
        log("[main] no target; aborting manipulation")
        return
    cur = target_xy
    alt = xy2d if (xy2d is not None and np.linalg.norm(xy2d - target_xy) > 0.03) else None
    for rep in range(2):
        g = None
        for attempt in range(3):
            if R.ended:
                break
            src = cur if not (rep == 0 and attempt == 1 and alt is not None) else alt
            g = pick(R, src, hue, label=f"pick{rep}.{attempt}")
            if g is not None:
                break
            R.move_to([cur[0], cur[1], SAFE_Z], grip=-1)
        if g is None or R.ended:
            log("[main] pick failed rep", rep)
            break
        place(R, g, label=f"place{rep}")
        cur = g[:2]
    if not R.ended and button is not None:
        press_button(R, button)
    # wait a few steps for the evaluation to register
    for _ in range(3):
        if R.ended:
            break
        R.step([[0, 0, 0, 0, 0, 0, 1]] * 5)
    log(f"[main] done; ended={R.ended} complete={R.complete} step={R.meta.get('step_id')} t={time.time() - t0:.1f}s")
    save_json("controller_log.json", LOG)
