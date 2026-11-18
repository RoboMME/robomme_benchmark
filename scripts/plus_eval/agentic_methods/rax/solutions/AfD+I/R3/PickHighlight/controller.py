"""PickHighlight controller.

Plan: hold still and watch the front camera for the white highlight discs that appear
under some cubes early in the episode; remember their world positions; press the
button; for each remembered cube, locate it precisely with the wrist camera, grasp it
top-down, lift it, put it back down and release; finally press the button again.
"""
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from perception import (WORKSPACE, components, dilate, load_frame_view, table_height,  # noqa: E402
                        white_mask)
from robot import EpisodeEnded, Robot  # noqa: E402

OUT = Path("/workspace/output")
CFG = dict(
    watch_max_steps=90,      # steps to watch for the highlight
    watch_chunk=3,
    disc_min_area=5,
    scan_dz=0.24,            # wrist-scan height above table
    approach_dz=0.10,
    lift_dz=0.14,
    press_depth=0.012,       # how far below the detected button top to push
    hold_top=4,
)

T0 = time.time()
_logf = None


def log(msg):
    global _logf
    line = f"[{time.time() - T0:7.1f}s] {msg}"
    print(line, flush=True)
    try:
        if _logf is None:
            OUT.mkdir(parents=True, exist_ok=True)
            _logf = open(OUT / "controller_log.txt", "a")
        _logf.write(line + "\n")
        _logf.flush()
    except Exception:
        pass


def save_img(arr, name):
    try:
        from PIL import Image
        OUT.mkdir(parents=True, exist_ok=True)
        Image.fromarray(arr).save(OUT / name)
    except Exception:
        pass


def wrap(a, period=2 * math.pi):
    return (a + period / 2) % period - period / 2


# ====================================================================== perception

def scene_objects_mask(view, tz, zmin=0.008, zmax=0.09):
    W = view.world()
    z = W[..., 2]
    ok = np.isfinite(z)
    m = np.zeros(z.shape, bool)
    m[ok] = (z[ok] > tz + zmin) & (z[ok] < tz + zmax)
    return m


def detect_cubes(view, tz, exclude_xy=None):
    """Cube top-face centres (world xy) from a top-down-ish view, colour-agnostic."""
    W = view.world()
    m = scene_objects_mask(view, tz, zmin=0.02, zmax=0.09)
    cubes = []
    for ys, xs in components(m, min_area=10):
        p = W[ys, xs]
        p = p[np.isfinite(p[:, 2])]
        if len(p) < 10:
            continue
        ext = np.ptp(p[:, :2], axis=0)
        if ext.max() > 0.085:
            continue  # robot base / large merged structure
        topz = np.percentile(p[:, 2], 95)
        tp = p[p[:, 2] > topz - 0.008]
        c = np.mean(tp[:, :2], axis=0)
        if exclude_xy is not None and np.linalg.norm(c - exclude_xy) < 0.035:
            continue
        cubes.append(dict(xy=c, top=float(topz), n=len(p)))
    return cubes


def assign_white(view, mask, cubes, maxd=0.045):
    """Count new-white pixels nearest to each cube centre."""
    counts = np.zeros(len(cubes))
    if not cubes or mask.sum() == 0:
        return counts
    W = view.world()
    ys, xs = np.nonzero(mask)
    p = W[ys, xs][:, :2]
    p = p[np.isfinite(p[:, 0])]
    C = np.array([c["xy"] for c in cubes])
    d = np.linalg.norm(p[:, None, :] - C[None, :, :], axis=2)
    j = np.argmin(d, axis=1)
    ok = d[np.arange(len(p)), j] < maxd
    for k in j[ok]:
        counts[k] += 1
    return counts


def find_button(view, tz):
    """White blob at table level present in the reference frame -> (xy, top_z)."""
    wm = white_mask(view.rgb, lo=170, spread=45)
    W = view.world()
    z = W[..., 2]
    ok = np.isfinite(z)
    near = np.zeros_like(wm)
    near[ok] = z[ok] < tz + 0.06
    best = None
    for ys, xs in components(wm & near, min_area=4):
        pts = W[ys, xs]
        pts = pts[np.isfinite(pts[:, 2])]
        if len(pts) < 4:
            continue
        ext = np.ptp(pts[:, :2], axis=0).max()
        if ext > 0.15:
            continue
        score = len(pts)
        if best is None or score > best[0]:
            best = (score, ys, xs, pts)
    if best is None:
        return None
    _, ys, xs, pts = best
    # Include the grey cap: all object points within the blob's bounding region.
    c = np.median(pts[:, :2], axis=0)
    allp = W.reshape(-1, 3)
    allp = allp[np.isfinite(allp[:, 2])]
    d = np.linalg.norm(allp[:, :2] - c, axis=1)
    loc = allp[d < 0.03]
    top = float(np.percentile(loc[:, 2], 97)) if len(loc) else float(np.max(pts[:, 2]))
    cap = loc[loc[:, 2] > top - 0.006] if len(loc) else pts
    cxy = np.median(cap[:, :2], axis=0) if len(cap) >= 3 else c
    return dict(xy=cxy, top=top, n=len(pts))


def new_white(view, ref_white_dil, tz):
    wm = white_mask(view.rgb, lo=190, spread=40)
    z = view.world()[..., 2]
    ok = np.isfinite(z)
    low = np.zeros_like(wm)
    low[ok] = z[ok] < tz + 0.022
    return wm & ~ref_white_dil & low


def disc_centres(view, mask, tz, min_area):
    """Circle centre of each disc blob in world xy (corrects for occlusion by the cube)."""
    W = view.world()
    fwd = -view.R[:, 2]
    f = np.array([fwd[0], fwd[1]])
    f /= np.linalg.norm(f) + 1e-9
    l = np.array([-f[1], f[0]])
    blobs = []
    for ys, xs in components(mask, min_area=min_area):
        pts = W[ys, xs]
        pts = pts[np.isfinite(pts[:, 0])][:, :2]
        if len(pts) >= min_area:
            blobs.append(pts)
    if not blobs:
        return []
    widths = [np.percentile(p @ l, 97) - np.percentile(p @ l, 3) for p in blobs]
    wmed = float(np.median(widths))
    split = []
    for p, w in zip(blobs, widths):
        k = int(round(w / max(wmed, 0.01))) if len(blobs) >= 2 else 1
        if w > 1.5 * wmed and k >= 2 and len(p) >= 2 * k * min_area:
            k = min(k, 3)
            # k-means along the lateral axis seeded at quantiles
            s_ = p @ l
            cen = np.quantile(s_, [(i + 0.5) / k for i in range(k)])
            for _ in range(10):
                lab = np.argmin(np.abs(s_[:, None] - cen[None, :]), axis=1)
                cen = np.array([s_[lab == i].mean() if np.any(lab == i) else cen[i] for i in range(k)])
            for i in range(k):
                if np.sum(lab == i) >= min_area:
                    split.append(p[lab == i])
        else:
            split.append(p)
    out = []
    for pts in split:
        s = pts @ l
        t = pts @ f
        s_lo, s_hi = np.percentile(s, 3), np.percentile(s, 97)
        r = max(0.005, (s_hi - s_lo) / 2)
        if r > 0.06:
            continue
        t_near = np.percentile(t, 3)
        cs = (s_lo + s_hi) / 2
        ct = t_near + r
        c = cs * l + ct * f
        out.append(dict(xy=c, r=r, n=len(pts), raw=np.mean(pts, axis=0)))
    return out


def merge_discs(found):
    """Merge disc detections from several frames (same disc within 2.5 cm)."""
    clusters = []
    for d in found:
        for c in clusters:
            if np.linalg.norm(c["xy"] - d["xy"]) < 0.025:
                c["all"].append(d)
                ws = np.array([e["n"] for e in c["all"]], float)
                c["xy"] = np.average(np.array([e["xy"] for e in c["all"]]), axis=0, weights=ws)
                break
        else:
            clusters.append(dict(xy=d["xy"].copy(), all=[d]))
    return clusters


def wrist_locate(view, tz, guess, max_dist=0.06):
    """Locate the cube nearest `guess` in a top-down wrist view.

    Returns dict(xy, top, size, edge_angle, comp_pts, other_pts) or None.
    """
    W = view.world()
    m = scene_objects_mask(view, tz, zmin=0.006, zmax=0.09)
    comps = components(m, min_area=15)
    if not comps:
        return None
    allobj = []
    cands = []
    for ys, xs in comps:
        pts = W[ys, xs]
        pts = pts[np.isfinite(pts[:, 2])]
        allobj.append(pts)
        topz = np.percentile(pts[:, 2], 95)
        tp = pts[pts[:, 2] > topz - 0.008]
        c = np.mean(tp[:, :2], axis=0)
        # Merged blobs: points of the top surface nearest the guess.
        d = np.linalg.norm(tp[:, :2] - guess, axis=1)
        cands.append((float(np.min(d)), tp, pts, topz))
    i = int(np.argmin([c[0] for c in cands]))
    dmin, tp, pts, topz = cands[i]
    if dmin > max_dist:
        return None
    sel = tp
    # Size and orientation from the top face.
    try:
        import cv2
        rect = cv2.minAreaRect((sel[:, :2] * 1000).astype(np.float32))
        (cx, cy), (w, h), ang = rect
        size = min(w, h) / 1000.0
        long_side = max(w, h) / 1000.0
        if long_side > 1.45 * size and size > 0.01:
            # Two touching cubes: mean-shift on the part nearest the guess.
            c = np.asarray(guess, float)
            for _ in range(6):
                dd = np.linalg.norm(sel[:, :2] - c, axis=1)
                sub = sel[dd < 0.6 * long_side / 2 + 0.3 * size]
                if len(sub) < 10:
                    break
                c = np.mean(sub[:, :2], axis=0)
            dd = np.linalg.norm(sel[:, :2] - c, axis=1)
            sub = sel[dd < 0.75 * size]
            if len(sub) >= 10:
                sel = sub
                rect = cv2.minAreaRect((sel[:, :2] * 1000).astype(np.float32))
                (cx, cy), (w, h), ang = rect
        xy = np.array([cx, cy]) / 1000.0
        edge = math.radians(ang)
        size = min(w, h) / 1000.0
    except Exception:
        xy = np.mean(sel[:, :2], axis=0)
        cov = np.cov((sel[:, :2] - xy).T)
        ev, evec = np.linalg.eigh(cov)
        edge = math.atan2(evec[1, 1], evec[0, 1])
        size = 0.04
    others = [p for j, p in enumerate(allobj) if j != i]
    other_pts = np.concatenate(others, axis=0) if others else np.zeros((0, 3))
    return dict(xy=xy, top=float(topz), size=float(size), edge=float(edge), comp_pts=pts,
                other_pts=other_pts, dmin=dmin)


def choose_grasp_yaw(loc, cur_yaw):
    """Pick the finger-axis yaw (aligned to a cube edge) with best clearance, nearest current."""
    best = None
    s = max(0.02, min(0.06, loc["size"]))
    xy = loc["xy"]
    own = loc["comp_pts"][:, :2]
    others = loc["other_pts"][:, :2] if len(loc["other_pts"]) else np.zeros((0, 2))
    # include points of the own component that are outside this cube's footprint
    for k in range(4):
        a = loc["edge"] + k * math.pi / 2
        ax = np.array([math.cos(a), math.sin(a)])
        pr = np.array([-ax[1], ax[0]])
        rel_o = own - xy
        outside = (np.abs(rel_o @ ax) > s / 2 + 0.004) | (np.abs(rel_o @ pr) > s / 2 + 0.004)
        pts = np.concatenate([others, own[outside]], axis=0) if len(others) else own[outside]
        rel = pts - xy
        along = np.abs(rel @ ax)
        perp = np.abs(rel @ pr)
        blocked = int(np.sum((along > s / 2 + 0.001) & (along < s / 2 + 0.03) & (perp < 0.015)))
        turn = abs(wrap(a - cur_yaw))
        if turn > math.pi / 2 + 1e-6:
            continue  # the gripper is symmetric: only consider turns within +-90 deg
        score = blocked * 1.0 + turn * 2.0
        if best is None or score < best[0]:
            best = (score, a, blocked, turn)
    return best[1], best[2]


# ====================================================================== skills

class Task:
    def __init__(self, ctx):
        self.r = Robot(ctx, log)
        self.tz = None
        self.button = None
        self.targets = []
        self.home = None

    # ---------------------------------------------------------- startup / highlight
    def read_initial_sequence(self):
        seq = (self.r.meta or {}).get("frame_sequence") or {}
        mp = seq.get("manifest_path")
        frames = []
        if mp and (WORKSPACE / mp).exists():
            try:
                man = json.loads((WORKSPACE / mp).read_text())
                frames = man.get("frames", [])
            except Exception as e:
                log(f"manifest read failed: {e}")
        vids = [f for f in frames if f.get("phase") == "video_demo"]
        log(f"initial sequence: {len(frames)} frames, {len(vids)} video_demo")
        return vids

    def watch_highlight(self):
        """Hold still at home; accumulate new-white (highlight disc) pixels per cube."""
        r = self.r
        w0 = r.views["wrist"]
        f0 = r.views["front"]
        bxy = self.button["xy"] if self.button else None
        cubes = detect_cubes(w0, self.tz, bxy)
        try:
            for c in detect_cubes(f0, self.tz, bxy):
                if c["xy"][0] < self.home[0] - 0.35:
                    continue
                if all(np.linalg.norm(c["xy"] - e["xy"]) > 0.035 for e in cubes):
                    log(f"front-only cube added at {np.round(c['xy'], 3).tolist()}")
                    cubes.append(c)
        except Exception as e:
            log(f"front cube detection failed {e!r}")
        log(f"cubes detected (wrist): {len(cubes)} {[np.round(c['xy'], 3).tolist() for c in cubes]}")
        refw = dilate(white_mask(w0.rgb, lo=170, spread=45), 2)
        reff = dilate(white_mask(f0.rgb, lo=170, spread=45), 2)
        acc_w = np.zeros(len(cubes))
        acc_f = np.zeros(len(cubes))
        counts = []
        seen = False
        quiet = 0
        used = 0
        best_n = 0
        while used < CFG["watch_max_steps"]:
            r.hold(CFG["watch_chunk"], grip=-1.0)
            used += CFG["watch_chunk"]
            vw, vf = r.views["wrist"], r.views["front"]
            mw = new_white(vw, refw, self.tz)
            mf = new_white(vf, reff, self.tz)
            n = int(mw.sum()) + int(mf.sum())
            counts.append((int(mw.sum()), int(mf.sum())))
            if n >= 20:
                seen = True
                quiet = 0
                acc_w += assign_white(vw, mw, cubes)
                acc_f += assign_white(vf, mf, cubes)
                if n > best_n:
                    best_n = n
                    save_img(vw.rgb, "highlight_wrist.png")
                    save_img(vf.rgb, "highlight_front.png")
            elif seen:
                quiet += 1
                if quiet >= 3:
                    break
        log(f"highlight pixel counts per check (wrist, front): {counts}")
        log(f"per-cube scores wrist {acc_w.astype(int).tolist()} front {acc_f.astype(int).tolist()}")
        if not cubes:
            return []
        sw = acc_w / max(acc_w.max(), 1)
        sf = acc_f / max(acc_f.max(), 1)
        score = np.where(acc_w.max() > 0, sw, 0) + np.where(acc_f.max() > 0, sf, 0)
        nsrc = int(acc_w.max() > 0) + int(acc_f.max() > 0)
        if nsrc == 0:
            return []
        score = score / nsrc
        keep = [c["xy"] for c, sc in zip(cubes, score) if sc >= 0.3]
        log(f"combined scores {np.round(score, 2).tolist()}")
        return keep

    # ---------------------------------------------------------- yaw
    def rotate_to(self, yaw_abs, tries=3):
        r = self.r
        for _ in range(tries):
            cur = r.finger_yaw()
            if cur is None:
                return
            err = wrap(yaw_abs - cur)
            if abs(err) < 0.03:
                return
            r.move_to(r.pos, yaw=err, max_steps=20, tol=0.01)

    # ---------------------------------------------------------- button
    def press_button(self, label):
        r = self.r
        b = self.button
        x, y = b["xy"]
        top = b["top"]
        log(f"press_button[{label}] at {np.round(b['xy'], 3).tolist()} top {top:.3f}")
        r.grip = 1.0
        if not b.get("refined"):
            r.move_to([x, y, self.tz + CFG["scan_dz"]], vmax=0.8, tol=0.008, max_steps=80)
            r.hold(2)
            try:
                w = r.views["wrist"]
                W = w.world().reshape(-1, 3)
                W = W[np.isfinite(W[:, 2])]
                d = np.linalg.norm(W[:, :2] - b["xy"], axis=1)
                loc = W[(d < 0.028) & (W[:, 2] > self.tz + 0.015) & (W[:, 2] < self.tz + 0.09)]
                if len(loc) > 20:
                    ztop = float(np.percentile(loc[:, 2], 98))
                    cap = loc[loc[:, 2] > ztop - 0.004]
                    nxy = np.median(cap[:, :2], axis=0)
                    if np.linalg.norm(nxy - b["xy"]) < 0.03 and len(cap) >= 5:
                        log(f"button refined {np.round(b['xy'], 4).tolist()} -> {np.round(nxy, 4).tolist()} top {ztop:.4f} (n={len(cap)})")
                        b["xy"] = nxy
                        b["top"] = ztop
                        b["refined"] = True
                        x, y = nxy
                        top = ztop
                save_img(w.rgb, f"button_scan_{label}.png")
            except Exception as e:
                log(f"button refine failed {e!r}")
        r.move_to([x, y, top + 0.08], vmax=0.7, tol=0.006, max_steps=80)
        r.hold(6, grip=1.0)
        r.move_to([x, y, top + 0.02], vmax=0.3, tol=0.004, max_steps=40)
        z_before = r.pos[2]
        r.move_to([x, y, top - CFG["press_depth"]], vmax=0.15, tol=0.003, max_steps=30,
                  stall_steps=6)
        log(f"press reached z {r.pos[2]:.4f} (target {top - CFG['press_depth']:.4f})")
        r.hold(3)
        r.move_to([x, y, top + 0.08], vmax=0.6, tol=0.008, max_steps=40)

    # ---------------------------------------------------------- pick one cube
    def scan(self, guess):
        r = self.r
        r.grip = -1.0
        r.move_to([guess[0], guess[1], self.tz + CFG["scan_dz"]], vmax=0.8, tol=0.008,
                  max_steps=80)
        r.hold(2)
        w = r.views.get("wrist")
        loc = wrist_locate(w, self.tz, guess) if w is not None else None
        return loc

    def pick(self, guess, idx):
        r = self.r
        for attempt in range(3):
            loc = self.scan(guess)
            if loc is None:
                log(f"cube {idx}: wrist locate failed near {np.round(guess, 3).tolist()}")
                if attempt == 0:
                    continue
                return False
            save_img(r.views["wrist"].rgb, f"scan_{idx}_{attempt}.png")
            # Re-scan centred over the measured cube for accuracy.
            if np.linalg.norm(loc["xy"] - guess) > 0.015:
                loc2 = self.scan(loc["xy"])
                if loc2 is not None and np.linalg.norm(loc2["xy"] - loc["xy"]) < 0.03:
                    loc = loc2
            cube_h = max(0.015, loc["top"] - self.tz)
            yaw, blocked = choose_grasp_yaw(loc, r.finger_yaw())
            log(f"cube {idx} att {attempt}: xy {np.round(loc['xy'], 4).tolist()} top {loc['top']:.3f} "
                f"h {cube_h:.3f} size {loc['size']:.3f} yaw {math.degrees(yaw):.1f} blocked {blocked} "
                f"dguess {np.linalg.norm(loc['xy'] - guess):.3f}")
            x, y = loc["xy"]
            self.rotate_to(yaw)
            r.move_to([x, y, self.tz + CFG["approach_dz"]], vmax=0.6, tol=0.005, max_steps=50)
            gz = self.tz + max(0.010, cube_h * 0.5)
            r.move_to([x, y, gz + 0.03], vmax=0.4, tol=0.004, max_steps=30)
            r.move_to([x, y, gz], vmax=0.2, tol=0.003, max_steps=30, stall_steps=6)
            r.set_gripper(1.0, n=10)
            open_closed = r.opening
            r.move_to([x, y, self.tz + CFG["lift_dz"]], vmax=0.4, tol=0.006, max_steps=50)
            r.hold(CFG["hold_top"])
            held = r.opening > 0.008
            log(f"cube {idx}: opening after close {open_closed:.4f}, after lift {r.opening:.4f}, "
                f"held={held}")
            save_img(r.views["wrist"].rgb, f"lift_{idx}_{attempt}.png")
            save_img(r.views["front"].rgb, f"liftfront_{idx}_{attempt}.png")
            if held:
                # Put it back down where it was and release.
                r.move_to([x, y, gz + 0.012], vmax=0.4, tol=0.004, max_steps=40, stall_steps=6)
                r.set_gripper(-1.0, n=8)
                r.move_to([x, y, self.tz + CFG["approach_dz"]], vmax=0.6, tol=0.008, max_steps=40)
                return True
            r.set_gripper(-1.0, n=6)
            guess = loc["xy"]
        return False

    # ---------------------------------------------------------- main
    def run(self):
        r = self.r
        r.observe()
        log(f"instruction: {r.meta.get('instruction')}")
        log(f"state0: {np.round(r.state, 4).tolist()} remaining {r.remaining}")
        self.home = r.pos
        front = r.views["front"]
        save_img(front.rgb, "front0.png")
        self.tz = table_height(front)
        log(f"table z {self.tz:.4f}")
        self.button = find_button(front, self.tz)
        log(f"button: {None if self.button is None else (np.round(self.button['xy'], 4).tolist(), round(self.button['top'], 4))}")
        targets = self.watch_highlight()
        log(f"targets ({len(targets)}): {[np.round(t, 3).tolist() for t in targets]}")
        if self.button is None:
            self.button = find_button(r.views["front"], self.tz)
        if self.button is not None:
            self.press_button("start")
        else:
            log("no button found!")
        if not targets:
            # Highlight may follow the press: watch again.
            targets = self.watch_highlight()
            log(f"targets after press ({len(targets)}): {[np.round(t, 3).tolist() for t in targets]}")
        # Order: greedy nearest from the button.
        cur = np.array(self.button["xy"]) if self.button else r.pos[:2]
        order = []
        left = list(targets)
        while left:
            j = int(np.argmin([np.linalg.norm(t - cur) for t in left]))
            order.append(left.pop(j))
            cur = order[-1]
        results = []
        for i, t in enumerate(order):
            if r.remaining is not None and r.remaining < 260:
                log("step budget low; skipping remaining picks")
                break
            ok = self.pick(np.asarray(t), i)
            results.append(ok)
            log(f"pick {i} result {ok}; steps used {r.steps_used} remaining {r.remaining}")
        log(f"pick results {results}")
        if self.button is not None:
            self.press_button("stop")
        r.hold(10)
        log(f"final: ended={r.ended} complete={r.complete} steps={r.steps_used}")


def main(ctx):
    from task3 import Task3
    t = Task3(ctx)
    try:
        t.run()
    except EpisodeEnded:
        log(f"episode ended: complete={t.r.complete}")
    except Exception as e:
        log("ERROR " + repr(e))
        log(traceback.format_exc())
        # Last resort: make sure the stop press still happens if possible.
        try:
            if t.button is not None and not t.r.ended and t.T is not None and t.T.picked.any():
                t.press_button("stop-after-error")
        except Exception as e2:
            log("ERROR in recovery " + repr(e2))
    finally:
        log(f"calls {t.r.n_calls} call time {t.r.t_calls:.1f}s total {time.time() - T0:.1f}s")
