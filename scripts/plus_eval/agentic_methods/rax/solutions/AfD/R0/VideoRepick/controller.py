"""VideoRepick controller (RAX, RoboMME-Plus).

Pipeline (each step logs to /workspace/output/videorepick_log.txt):
  observe -> load conditioning video manifest -> calibrate cameras (pixel_to_position fit)
  -> table height, object colour -> analyse video (identify picked object by depth vacancy,
  track all objects in world XY with front + wrist) -> select target at execution frame
  -> locate button -> pick/place target N times (N from instruction, default 2) -> press button.
Transitions: any perception failure falls back to a simpler estimate (2-D front tracking,
then nearest-to-video-pick); motion failures retry the grasp with re-localisation (<=3);
episode end (EpisodeOver) stops all calls.
"""
import os, re, sys, time, traceback
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from io_utils import (log, reply_json, find, get_state, get_step_id, manifest_path, load_manifest,
                      load_rgb, load_depth, episode_over)
from geom import backproject
import video as V
from video import zmap, table_height
from motion import Robot, EpisodeOver
from tracker import grid_clusters

VERSION = 'v2l'
LAST = {}
DEBUG = {}

def parse_count(instr):
    s = (instr or '').lower()
    if 'thrice' in s or 'three times' in s or '3 times' in s:
        return 3
    if 'twice' in s or 'two times' in s or '2 times' in s:
        return 2
    if 'once' in s or 'one time' in s:
        return 1
    if 'four times' in s or '4 times' in s:
        return 4
    return 2

def frame_images(fr):
    f = fr['files']
    out = {'state': fr.get('state'), 't': fr.get('ts')}
    for cam in ('front', 'wrist'):
        try:
            out[cam + '_rgb'] = load_rgb(f, cam)
        except Exception:
            out[cam + '_rgb'] = None
        try:
            out[cam + '_depth'] = load_depth(f, cam)
        except Exception:
            out[cam + '_depth'] = None
    return out

def current_frame(robot):
    """Images of the current observation, with the robot state recorded for that very frame."""
    d = None
    try:
        d = robot.refresh()
    except EpisodeOver:
        raise
    except Exception:
        d = None
    mp = manifest_path(d) if d else None
    if mp is None:
        mp = manifest_path(robot.packet)
    if mp is None:
        return None
    try:
        frames, _ = load_manifest(mp)
    except Exception as e:
        log('current_frame manifest failed', repr(e)[:200]); return None
    if not frames:
        return None
    ex = [f for f in frames if 'exec' in str(f.get('phase', '')).lower()] or frames
    cur = robot.state
    best = ex[-1]
    if cur is not None:
        with_state = [f for f in ex if f.get('state') is not None and len(f['state']) >= 3]
        if with_state:
            best = min(with_state, key=lambda f: float(np.linalg.norm(np.asarray(f['state'][:3], float) - cur[:3])))
    fr = frame_images(best)
    st = best.get('state')
    fr['state'] = np.asarray(st, float) if st is not None and len(st) >= 6 else robot.state.copy()
    if cur is not None and st is not None and len(st) >= 3:
        gap = float(np.linalg.norm(np.asarray(st[:3], float) - cur[:3]))
        if gap > 0.01:
            log(f'current_frame: closest frame state is {gap*100:.1f} cm from current state (using frame state)')
    return fr

def object_clusters(scene, fr, use_wrist=True):
    P, _ = V.fused_points(scene, fr, use_wrist=use_wrist)
    if len(P) < 6:
        return []
    out = []
    for idx in grid_clusters(P[:, :2], 0.004, 8):
        Q = P[idx]
        ztop = float(np.percentile(Q[:, 2], 95))
        top = Q[Q[:, 2] > scene.z0 + 0.6 * (ztop - scene.z0)]
        xy = np.mean(top[:, :2], 0) if len(top) >= 4 else np.mean(Q[:, :2], 0)
        out.append({'xy': xy, 'ztop': ztop, 'n': len(idx), 'ext': float(np.ptp(Q[:, :2], 0).max()), 'pts': Q})
    # split merged clusters with k-means using the object diameter
    res = []
    for c in out:
        k = int(round(c['ext'] / max(scene.D, 0.015)))
        if c['ext'] > 1.6 * scene.D and k >= 2:
            from tracker import kmeans
            cc, lab = kmeans(c['pts'][:, :2], min(k, 4))
            for j in range(len(cc)):
                Q = c['pts'][lab == j]
                if len(Q) < 6:
                    continue
                ztop = float(np.percentile(Q[:, 2], 95))
                res.append({'xy': Q[:, :2].mean(0), 'ztop': ztop, 'n': len(Q), 'ext': float(np.ptp(Q[:, :2], 0).max()), 'pts': Q})
        else:
            res.append(c)
    return res


def neighbour_points(cl, txy, D, nmax=800):
    if not cl:
        return np.zeros((0, 2))
    P = np.concatenate([c['pts'][:, :2] for c in cl])
    P = P[np.linalg.norm(P - np.asarray(txy)[:2], axis=1) > 0.6 * D]
    if len(P) > nmax:
        P = P[np.random.default_rng(0).choice(len(P), nmax, replace=False)]
    return P

def local_centre(scene, P, start, D, iters=8):
    """Mean-shift on object points (Nx3) from `start`: centre of the object under it, robust to
    neighbouring objects merged into the same cluster. Returns a cluster-like dict or None."""
    if P is None or len(P) < 6:
        return None
    c = np.asarray(start, float)[:2].copy()
    rad = 0.55 * D
    for _ in range(iters):
        d = np.linalg.norm(P[:, :2] - c, axis=1)
        sel = d < rad
        if sel.sum() < 6:
            return None
        cn = P[sel, :2].mean(0)
        if np.linalg.norm(cn - c) < 0.0005:
            c = cn; break
        c = cn
    d = np.linalg.norm(P[:, :2] - c, axis=1); Q = P[d < rad]
    ztop = float(np.percentile(Q[:, 2], 95))
    # refine on the top-surface band: side faces are seen from one side only (biased towards the
    # cameras); the top is seen symmetrically
    try:
        z0 = float(getattr(scene, 'z0', np.nan))
        hh = ztop - z0 if np.isfinite(z0) else 0.03
        band = float(np.clip(0.25 * hh, 0.005, 0.015))
        T = P[P[:, 2] > ztop - band]
        ct = c.copy()
        for _ in range(iters):
            selt = np.linalg.norm(T[:, :2] - ct, axis=1) < rad
            if selt.sum() < 10:
                ct = None; break
            cn = T[selt, :2].mean(0)
            if np.linalg.norm(cn - ct) < 0.0005:
                ct = cn; break
            ct = cn
        if ct is not None and np.linalg.norm(ct - c) < 0.4 * D:
            c = ct
            d = np.linalg.norm(P[:, :2] - c, axis=1); Q = P[d < rad]
    except Exception:
        pass
    return {'xy': c, 'ztop': ztop, 'n': int(len(Q)), 'ext': float(min(np.ptp(Q[:, :2], 0).max(), D)), 'pts': Q}

def relocalise(scene, robot, xy, radius=0.04):
    relocalise.others = None
    try:
        fr = current_frame(robot)
        if fr is None:
            return None
        cl = object_clusters(scene, fr)
        if not cl:
            return None
        P = np.concatenate([c['pts'] for c in cl])
        c = local_centre(scene, P, xy, scene.D)
        ok = c is not None and float(np.linalg.norm(c['xy'] - xy)) < radius
        # neighbours are everything away from the XY we will actually grasp at
        relocalise.others = neighbour_points(cl, c['xy'] if ok else xy, scene.D)
        update_centres(scene, cl, P)
        if c is not None:
            dd = float(np.linalg.norm(c['xy'] - xy))
            if dd >= radius:
                log(f'relocalise: nearest object centre moved {dd*100:.1f} cm (> {radius*100:.1f}); keeping previous estimate')
            if dd < radius:
                log(f'relocalised target {np.round(c["xy"],3).tolist()} (moved {dd*100:.1f} cm) ztop {c["ztop"]-scene.z0:.3f}')
                return c
    except Exception as e:
        log('relocalise error', repr(e)[:200])
    return None


relocalise.others = None
NBR = {'centres': np.zeros((0, 2))}   # XY centres of other objects (neighbours do not move)

def update_centres(scene, cl, P=None):
    """Refine cluster centres on their top band and merge with the stored set: new centres
    replace stored ones within 0.5 D; stored ones not seen now are kept (occluded in this view)."""
    try:
        if not cl:
            return
        if P is None:
            P = np.concatenate([c['pts'] for c in cl])
        new = []
        for c in cl:
            lc = local_centre(scene, P, c['xy'], scene.D)
            new.append(lc['xy'] if lc is not None else np.asarray(c['xy'], float)[:2])
        new = np.array(new).reshape(-1, 2)
        old = NBR['centres']
        keep = [o for o in old if len(new) == 0 or np.linalg.norm(new - o, axis=1).min() > 0.5 * scene.D]
        NBR['centres'] = np.vstack([new] + ([np.array(keep)] if keep else []))
    except Exception as e:
        log('update_centres error', repr(e)[:150])

def merge_others(old, new, xy, D, nmax=1200):
    """Union of neighbour clouds (neighbours do not move; each view misses different parts),
    without points within 0.6 D of the grasp XY."""
    parts = [np.asarray(a, float)[:, :2] for a in (old, new) if a is not None and len(a) > 0]
    if not parts:
        return old
    P = np.concatenate(parts)
    P = P[np.linalg.norm(P - np.asarray(xy, float)[:2], axis=1) > 0.6 * D]
    if len(P) > nmax:
        P = P[np.random.default_rng(1).choice(len(P), nmax, replace=False)]
    return P

# ---------------------------------------------------------------- gripper yaw
def finger_axis(scene, robot):
    """World XY unit vector of the finger closing direction (wrist image u-axis; EE y as fallback)."""
    from geom import axis_angle_to_R
    R = axis_angle_to_R(robot.state[3:6])
    ey = R[:2, 1] / max(np.linalg.norm(R[:2, 1]), 1e-9)
    a = None
    try:
        M = scene.M_for('wrist', robot.state)
        if M is not None:
            a = M[:2, 0] / max(np.linalg.norm(M[:2, 0]), 1e-9)
    except Exception:
        a = None
    if a is None or not np.all(np.isfinite(a)):
        return ey
    ang = np.degrees(np.arccos(np.clip(abs(float(a @ ey)), 0, 1)))
    if ang > 25:
        log(f'finger axis: wrist-u and EE-y differ by {ang:.0f} deg; using wrist-u')
    return a

def yaw_cost(axis, c, Pn, half_gap):
    """Number of neighbour surface points (XY, target removed) inside the inflated finger
    footprints (finger ~1.2 cm thick along the closing axis, ~2 cm wide; 4 mm margin), plus
    half weight for neighbour points inside the closing gap."""
    if len(Pn) == 0:
        return 0.0
    perp = np.array([-axis[1], axis[0]])
    d = Pn - c
    al = np.abs(d @ axis); pe = np.abs(d @ perp)
    f_al = half_gap + 0.006
    in_f = (np.abs(al - f_al) < 0.010) & (pe < 0.014)
    in_gap = (al < f_al - 0.006) & (pe < 0.012)
    return float(in_f.sum() + 0.5 * in_gap.sum())

def disc_violation(axis, c, Cn, half_gap, rn, margin=0.004, f_len=0.012, f_half=0.011):
    """Sum (mm) of clearance shortfalls between the two finger rectangles (closing-axis extent
    [half_gap, half_gap+f_len], half width f_half) and neighbour discs (centre Cn, radius rn)."""
    if Cn is None or len(Cn) == 0:
        return 0.0
    perp = np.array([-axis[1], axis[0]])
    d = Cn - c
    al = d @ axis; pe = d @ perp
    tot = 0.0
    for sgn in (1.0, -1.0):
        a = sgn * al
        da = np.maximum(np.maximum(half_gap - a, a - (half_gap + f_len)), 0.0)
        dp = np.maximum(np.abs(pe) - f_half, 0.0)
        clear = np.hypot(da, dp) - rn
        tot += float(np.maximum(margin - clear, 0.0).sum()) * 1000.0
    return tot

def choose_yaw(scene, robot, c, Pn, half_gap, tried=()):
    """Pn: XY points of other objects. Returns (relative yaw rad, cost).
    Points are reduced to 4 mm occupancy cells (uniform density); the cost of each yaw is the
    max over +-5 deg so the centre of a free window is preferred over its edge."""
    a0 = finger_axis(scene, robot)
    Pn = np.zeros((0, 2)) if Pn is None else np.asarray(Pn, float).reshape(-1, 2)
    if len(Pn):
        Pn = Pn[np.linalg.norm(Pn - c, axis=1) < half_gap + 0.04]
    if len(Pn):
        Pn = (np.unique(np.floor(Pn / 0.004).astype(int), axis=0) + 0.5) * 0.004
    Cn = np.asarray(NBR.get('centres', np.zeros((0, 2))), float).reshape(-1, 2)
    if len(Cn):
        dc = np.linalg.norm(Cn - c, axis=1)
        Cn = Cn[(dc > 0.6 * scene.D) & (dc < half_gap + 0.03 + scene.D)]
    rn = 0.5 * scene.D
    degs = list(range(-90, 90, 5))
    raw = []
    for deg in degs:
        th = np.radians(deg)
        Rz = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        ax = Rz @ a0
        raw.append(yaw_cost(ax, c, Pn, half_gap) + 2.0 * disc_violation(ax, c, Cn, half_gap, rn))
    raw = np.array(raw); n = len(raw)
    sm = np.maximum(np.maximum(raw, np.roll(raw, 1)), np.roll(raw, -1))   # axis symmetric: circular in 180 deg
    best = None; c0 = None
    for k, deg in enumerate(degs):
        cst = sm[k] + 1.0 * abs(deg) / 90.0
        for t in tried:
            dd = abs(((deg - t) + 90) % 180 - 90)
            if dd < 25:
                cst += 30 * (1 - dd / 25)
        if deg == 0:
            c0 = cst
        if best is None or cst < best[0] - 1e-9:
            best = (cst, deg)
    log(f'yaw: axis0 {np.degrees(np.arctan2(a0[1], a0[0])):.0f} deg; current cost {c0:.1f} -> best {best[1]} deg cost {best[0]:.1f} ({len(Pn)} neighbour cells, {len(Cn)} discs, half gap {half_gap:.3f})')
    if DEBUG.get('yaw'):
        DEBUG.setdefault('yaw_data', []).append((a0.copy(), np.asarray(c, float).copy(), Pn.copy(), half_gap))
    if c0 <= best[0] + 0.5:
        return 0.0, c0
    return float(np.radians(best[1])), best[0]

def apply_yaw(robot, th):
    if abs(th) < 1e-3:
        return
    from geom import axis_angle_to_R
    Rz = axis_angle_to_R([0, 0, th])
    R = axis_angle_to_R(robot.state[3:6])
    robot.rotate_to(Rz @ R, label='yaw')

# ---------------------------------------------------------------- manipulation
def pick(robot, scene, tgt, others=None, attempts=4):
    from geom import R_to_axis_angle, axis_angle_to_R
    xy = np.array(tgt['xy'], float); ztop = tgt['ztop']; xy0 = xy.copy()
    z0 = scene.z0
    h = float(np.clip(ztop - z0, 0.012, 0.12))
    ext = float(np.clip(tgt.get('ext', scene.D), 0.012, 0.06))
    yaw_abs = getattr(robot, 'yaw_abs', 0.0)
    mem = getattr(robot, 'grasp_mem', None)
    if mem is None:
        mem = robot.grasp_mem = []
    near = [m for m in mem if np.linalg.norm(m['xy'] - xy) < 0.012]
    # absolute yaws (deg, relative to start) of attempts blocked by collisions here, this or earlier picks
    tried = [m['yaw_deg'] for m in near if not m['ok']]
    good = [m for m in near if m['ok']]
    shrink = 0.0
    for att in range(attempts):
        if time.time() - LAST.get('t0', time.time()) > 1030:
            log('wall-clock guard: abandoning pick attempts'); break
        z_safe = z0 + max(0.16, h + 0.10)
        z_grasp = z0 + float(np.clip(0.5 * h, 0.012, max(0.012, h - 0.008)))
        w_lo = max(0.025, ext + 0.006)
        widths = sorted({float(np.clip(ext + m - shrink, w_lo, 0.08)) for m in (0.020, 0.014, 0.009)}, reverse=True)
        log(f'pick attempt {att}: xy {np.round(xy,3).tolist()} h {h:.3f} ext {ext:.3f} widths {np.round(widths,3).tolist()} z_grasp {z_grasp-z0:.3f}')
        cur = robot.pos
        if cur[2] < z_safe - 0.01:
            robot.move_to([cur[0], cur[1], z_safe], label='up')
        robot.move_to([xy[0], xy[1], z_safe], vmax=0.03, label='above')
        # fresh neighbour cloud from directly above (wrist looks down; earlier clouds may be stale or occluded)
        relocalise(scene, robot, xy, radius=0.0)
        others = merge_others(others, relocalise.others, xy, scene.D)
        try:
            best = None
            if att == 0 and good:
                # the grasp that worked at this spot before: same neighbours, same yaw and width
                g = good[-1]
                d = (g['yaw_deg'] - np.degrees(yaw_abs) + 90.0) % 180.0 - 90.0
                best = (0.0, float(np.radians(d)), g['width'])
                log(f'  reusing successful grasp here: yaw {g["yaw_deg"]:.0f} deg (abs), width {g["width"]:.3f}')
            for wi, w in enumerate(widths if best is None else []):
                th, cst = choose_yaw(scene, robot, xy, others, w / 2, [t - np.degrees(yaw_abs) for t in tried])
                score = cst + 2.0 * wi
                if best is None or score < best[0]:
                    best = (score, th, w)
            _, th, width = best
            log(f'  chosen width {width:.3f} yaw {np.degrees(th):.0f} deg')
            apply_yaw(robot, th)
            yaw_abs += th; robot.yaw_abs = yaw_abs
        except EpisodeOver:
            raise
        except Exception as e:
            log('yaw selection failed', repr(e)[:200]); width = widths[0]
        robot.open_to(width); robot.last_width = width
        robot.move_to([xy[0], xy[1], z_grasp + 0.04], vmax=0.02, label='pre')
        robot.move_to([xy[0], xy[1], z_grasp], vmax=0.01, tol=0.004, label='grasp')
        zerr = robot.pos[2] - z_grasp
        if zerr > 0.012:
            log(f'  descent blocked {zerr*100:.1f} cm above grasp height (collision); retrying with another yaw/narrower gap')
            tried.append(np.degrees(yaw_abs))
            mem.append(dict(xy=xy.copy(), yaw_deg=float(np.degrees(yaw_abs)), width=float(width), ok=False))
            good = []
            shrink += 0.008
            p = robot.pos
            robot.move_to([p[0], p[1], z_grasp + 0.06], label='back-up')
            relocalise(scene, robot, xy, radius=0.0)   # neighbours only; target XY kept (occluded view)
            others = merge_others(others, relocalise.others, xy, scene.D)
            continue
        robot.set_gripper(1.0, 12)
        f = robot.fingers
        log(f'  fingers after close {f:.4f}')
        if f > 0.004:
            xy_g = robot.pos[:2].copy()
            robot.move_to([xy[0], xy[1], z0 + h + 0.10], vmax=0.02, label='lift')
            robot.hold(4)
            f2 = robot.fingers
            log(f'  fingers after lift {f2:.4f}')
            if f2 > 0.003:
                # held-object offset from the EE: the fingers centre it along the closing axis,
                # the perpendicular offset of the grasp remains
                try:
                    a = finger_axis(scene, robot); pa = np.array([-a[1], a[0]])
                    # along the closing axis the jaws centre a free object; if it sticks it keeps
                    # its offset - take half of that component (bounded error either way)
                    dxy = xy - xy_g
                    robot.hold_off = pa * float(dxy @ pa) + 0.5 * a * float(dxy @ a)
                except Exception:
                    robot.hold_off = np.zeros(2)
                mem.append(dict(xy=xy.copy(), yaw_deg=float(np.degrees(yaw_abs)), width=float(width), ok=True))
                return True, robot.pos[:2].copy(), z_grasp
        # failed: open, back up, relocalise
        robot.set_gripper(-1.0, 8)
        p = robot.pos
        robot.move_to([p[0], p[1], z_safe], label='retreat')
        c = relocalise(scene, robot, xy, radius=0.25 * scene.D)
        if relocalise.others is not None:
            others = relocalise.others
        pattern = np.array([[0.005, 0], [-0.005, 0], [0, 0.005], [0, -0.005]])
        if c is not None and np.linalg.norm(c['xy'] - xy) > 0.002:
            xy = c['xy']; h = float(np.clip(c['ztop'] - z0, 0.012, 0.12)); ext = float(np.clip(c.get('ext', ext), 0.012, 0.06))
        else:
            xy = xy0 + pattern[att % 4]
            log(f'  no better estimate; trying offset {pattern[att % 4].tolist()}')
    return False, xy, None

def place(robot, scene, xy, z_grasp):
    z0 = scene.z0
    robot.move_to([xy[0], xy[1], z_grasp + 0.006], vmax=0.015, label='place')
    rel = robot.pos[:2].copy()
    w_held = robot.fingers
    robot.open_to(max(getattr(robot, 'last_width', 0.08), w_held + 0.012), n=10)
    robot.placed_xy = rel + getattr(robot, 'hold_off', np.zeros(2))
    if robot.fingers < w_held + 0.006:
        log(f'  release not confirmed (fingers {robot.fingers:.3f} vs held {w_held:.3f}); opening fully')
        robot.set_gripper(-1.0, 10)
    log(f'  released: fingers {robot.fingers:.3f}')
    p = robot.pos
    robot.move_to([p[0], p[1], z0 + 0.16], vmax=0.025, label='retreat')
    return True

def press_button(robot, scene, button):
    bxy, btop = np.array(button[0], float), float(button[1])
    log(f'press button at {np.round(bxy,3).tolist()} top {btop-scene.z0:.3f}')
    offsets = [np.zeros(2), np.array([0.006, 0]), np.array([-0.006, 0]), np.array([0, 0.006]), np.array([0, -0.006])]
    robot.grip = 1.0
    for k, off in enumerate(offsets):
        xy = bxy + off
        p = robot.pos
        if p[2] < btop + 0.06:
            robot.move_to([p[0], p[1], btop + 0.08], label='btn-up')
        robot.move_to([xy[0], xy[1], btop + 0.08], vmax=0.03, label='btn-above')
        robot.set_gripper(1.0, 6)
        robot.move_to([xy[0], xy[1], btop + 0.015], vmax=0.015, label='btn-pre')
        for dz in (-0.006, -0.012, -0.018):
            robot.move_to([xy[0], xy[1], btop + dz], vmax=0.006, tol=0.003, max_calls=8, label='btn-press')
            robot.hold(3)
        robot.move_to([xy[0], xy[1], btop + 0.06], vmax=0.02, label='btn-release')
        log(f'  press {k} done, episode continues')
    return False

# ---------------------------------------------------------------- main
def run(ctx):
    t0 = time.time(); LAST['t0'] = t0
    log(f'VideoRepick controller {VERSION}')
    obs = reply_json(ctx.call('robomme_observe'))
    instr = find(obs, 'instruction') or ''
    log('instruction:', instr)
    count = parse_count(instr)
    state0 = get_state(obs); step_id = get_step_id(obs) or 0
    log('state0', None if state0 is None else np.round(state0, 4).tolist(), 'step', step_id)
    robot = Robot(ctx, obs)
    mp = manifest_path(obs)
    frames, _ = load_manifest(mp) if mp else ([], None)
    vid = [f for f in frames if 'video' in f['phase'].lower()]
    execs = [f for f in frames if 'exec' in f['phase'].lower()]
    ex = execs[-1] if execs else (frames[-1] if frames else None)
    log(f'manifest frames {len(frames)} video {len(vid)} exec {len(execs)}')
    exf = frame_images(ex)
    exf['state'] = state0
    # calibration
    Mf = V.calibrate(ctx, step_id, exf['front_depth'], 'front', obs) if exf['front_depth'] is not None else None
    Mw = V.calibrate(ctx, step_id, exf['wrist_depth'], 'wrist', obs) if exf['wrist_depth'] is not None else None
    if Mf is None:
        Mf = V.M_from_packet(obs, 'front')
    if Mw is None:
        Mw = V.M_from_packet(obs, 'wrist')
    z0 = None
    if Mf is not None:
        th = table_height(zmap(Mf, exf['front_depth'])[2]); z0 = th[0] if th else None
    zw = None
    if Mw is not None:
        thw = table_height(zmap(Mw, exf['wrist_depth'])[2]); zw = thw[0] if thw else None
    log(f'table z front {z0} wrist {zw}')
    if z0 is None:
        z0 = zw if zw is not None else 0.0
    if zw is not None and abs(zw - z0) > 0.015:
        log('wrist calibration inconsistent with front; disabling wrist'); Mw = None
    scene = V.Scene(Mf, Mw, state0, z0)
    scene.cm = V.learn_scene_colour(scene, [(exf['front_rgb'], exf['front_depth'], Mf), (exf['wrist_rgb'], exf['wrist_depth'], Mw)])
    if scene.cm is None:
        raise RuntimeError('no colour model')
    log('colour model', scene.cm.to_dict())
    # load video
    vframes = []
    for f in vid:
        fi = frame_images(f)
        vframes.append(fi)
    log(f'loaded {len(vframes)} video frames in {time.time()-t0:.1f}s')
    res = {'ok': False}
    scene.deadline = t0 + 420.0      # first control step must come < 600 s after start
    try:
        if vframes:
            res = V.analyse_video(scene, vframes, exf)
    except Exception:
        log('analyse_video crashed:', traceback.format_exc()[-1500:])
    log(f'analysis done in {time.time()-t0:.1f}s ok={res.get("ok")}')
    # execution-frame objects
    cl = object_clusters(scene, exf)
    log(f'exec clusters {len(cl)}: ' + ', '.join(f'{np.round(c["xy"],3).tolist()}' for c in cl))
    if not cl:
        raise RuntimeError('no objects at execution frame')
    if res.get('ok'):
        fxy = res['final_xy']
        d = [np.linalg.norm(c['xy'] - fxy) for c in cl]
        j = int(np.argmin(d))
        Pall = np.concatenate([c['pts'] for c in cl])
        lc = local_centre(scene, Pall, fxy, scene.D)
        if lc is not None and np.linalg.norm(lc['xy'] - fxy) < 0.8 * scene.D:
            tgt = lc
            log(f'target (mean-shift from track) at {np.round(lc["xy"],3).tolist()} dist to track {np.linalg.norm(lc["xy"]-fxy)*100:.1f} cm; nearest cluster {j} {d[j]*100:.1f} cm')
        else:
            log(f'target cluster {j} at {np.round(cl[j]["xy"],3).tolist()} dist to track {d[j]*100:.1f} cm (mean-shift failed)')
            tgt = cl[j]
    else:
        log('video analysis failed; choosing object nearest to the image centre of objects (low confidence)')
        cen = np.mean([c['xy'] for c in cl], 0)
        tgt = min(cl, key=lambda c: np.linalg.norm(c['xy'] - cen))
    # button
    hover = None
    try:
        sts = [f['state'] for f in vframes[len(vframes) // 2:] if f.get('state') is not None]
        if sts:
            hover = np.median(np.array(sts)[:, :2], 0)
    except Exception:
        pass
    button = V.find_button(scene, [('front', exf['front_rgb'], exf['front_depth'], Mf), ('wrist', exf['wrist_rgb'], exf['wrist_depth'], Mw)], hover, ee_xy=state0[:2])
    if button is None and hover is not None:
        log('button not found; using hover xy'); button = (hover, z0 + 0.03)
    log(f'perception total {time.time()-t0:.1f}s; count={count}')
    # manipulation
    xy = tgt['xy']; cur = tgt
    others = neighbour_points(cl, tgt['xy'], scene.D)
    update_centres(scene, cl)
    LAST['res'] = res
    for k in range(count):
        if time.time() - t0 > 950:
            log('wall-clock guard: skipping remaining picks'); break
        ok, xy, zg = pick(robot, scene, cur, others)
        if not ok:
            log(f'pick {k} failed after retries'); 
            continue
        place(robot, scene, xy, zg)
        dr = getattr(robot, 'placed_xy', xy)
        c = relocalise(scene, robot, dr, radius=0.5 * scene.D)
        others = merge_others(others, relocalise.others, dr, scene.D)
        # dead reckoning of the placed object is preferred (the arm occludes it from above);
        # vision only nudges it when close
        if c is not None and np.linalg.norm(c['xy'] - dr) < 0.25 * scene.D:
            nxy = 0.3 * c['xy'] + 0.7 * dr
        else:
            nxy = dr
        log(f'placed object estimate {np.round(nxy,3).tolist()} (dead reckoning {np.round(dr,3).tolist()})')
        cur = {'xy': nxy, 'ztop': cur['ztop'], 'ext': cur.get('ext', scene.D)}
    if button is not None:
        press_button(robot, scene, button)
    log('sequence finished; holding')
    robot.hold(5)

def main(ctx):
    try:
        run(ctx)
    except EpisodeOver:
        log('episode over; stopping')
    except Exception:
        log('controller error:', traceback.format_exc()[-3000:])
