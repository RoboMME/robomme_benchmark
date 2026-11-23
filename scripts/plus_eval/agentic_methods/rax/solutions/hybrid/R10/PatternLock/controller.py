"""PatternLock controller: read the conditioning video, find the visited-target sequence on the
5x5 grid (targets turn red when touched), locate the grid in the live scene and drag the stick
through the same grid indices."""
import json
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
from pattern import extract_pattern, grid_from_frame  # noqa: E402
from geom import WS  # noqa: E402

OUT = WS / "output"
DRAW_Z = 0.070       # TCP height while dragging (as in the conditioning video)
HOVER_Z = 0.12
MAX_STEP = 0.4       # action units (x0.05 m) per row -> 2 cm per step
LOGF = None


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    try:
        with open(OUT / "controller_log.txt", "a") as f:
            f.write(msg + "\n")
    except Exception:
        pass


def packet(reply):
    for block in reply.get("content", []) if isinstance(reply, dict) else reply:
        if block.get("type") == "text":
            try:
                return json.loads(block["text"])
            except Exception:
                continue
    return {}


class Robot:
    def __init__(self, ctx):
        self.ctx = ctx
        self.ended = False
        self.complete = False
        self.state = None
        self.remaining = None
        self.last = None
        self.step_id = 0

    def observe(self):
        p = packet(self.ctx.call("robomme_observe"))
        self._update(p)
        return p

    def _update(self, p):
        self.last = p
        if "state" in p:
            self.state = np.array(p["state"], float)
        if isinstance(p.get("step_id"), int):
            self.step_id = p["step_id"]
        lim = p.get("episode_limits") or {}
        if "remaining_episode_steps" in lim:
            self.remaining = lim["remaining_episode_steps"]
        if p.get("episode_ended"):
            self.ended = True
        for st in p.get("steps", []) or []:
            if isinstance(st, dict):
                if st.get("task_complete"):
                    self.complete = True
                if st.get("episode_ended"):
                    self.ended = True
        if p.get("task_complete"):
            self.complete = True
        tr = p.get("transition")
        if isinstance(tr, dict):
            if tr.get("task_complete"):
                self.complete = True
            if tr.get("episode_ended"):
                self.ended = True

    def left(self):
        """Steps left, assuming the observed hard cap of STEP_CAP steps per episode."""
        r = self.remaining if self.remaining is not None else STEP_CAP
        return int(min(r, STEP_CAP - self.step_id))

    def step(self, rows):
        if self.ended:
            return None
        if self.remaining is not None:
            rows = rows[: max(0, int(self.remaining))]
        if not rows:
            return None
        p = packet(self.ctx.call("robomme_step", actions=[[float(x) for x in r] for r in rows]))
        self._update(p)
        return p

    def move_to(self, target, max_step=MAX_STEP, tol=0.004, max_iter=8):
        target = np.asarray(target, float)
        for it in range(max_iter):
            if self.ended:
                return False
            pos = self.state[:3]
            d = target - pos
            dist = np.linalg.norm(d)
            if dist < tol:
                return True
            n = int(np.ceil(dist / (0.05 * max_step)))
            n = max(1, min(n, 40))
            per = d / n / 0.05
            rows = [[per[0], per[1], per[2], 0.0, 0.0, 0.0, 0.0] for _ in range(n)]
            self.step(rows)
        return np.linalg.norm(target - self.state[:3]) < 3 * tol


def decode_obs(reply):
    """Return (packet, {(camera,kind): np.array})."""
    import base64, io
    from PIL import Image
    blocks = reply.get("content", []) if isinstance(reply, dict) else reply
    pk = packet(reply)
    imgs = {}
    for att in pk.get("attachments", []):
        ci = att.get("content_index")
        if ci is None or ci >= len(blocks):
            continue
        b = blocks[ci]
        if b.get("type") != "image":
            continue
        im = Image.open(io.BytesIO(base64.b64decode(b["data"])))
        arr = np.asarray(im)
        if att.get("kind") == "depth":
            arr = arr.astype(np.float64) * 0.001
        else:
            arr = np.asarray(im.convert("RGB")).astype(np.int32)
        imgs[(att.get("camera"), att.get("kind"))] = arr
    return pk, imgs


def red_at(ctx, rob, xy, radius=0.03):
    """Count red pixels whose world point is within radius of xy in the live front view."""
    from geom import world_points
    from pattern import red_mask
    reply = ctx.call("robomme_observe")
    pk, imgs = decode_obs(reply)
    rob._update(pk)
    total = 0
    for cam in ("front", "wrist"):
        rgb = imgs.get((cam, "rgb"))
        dep = imgs.get((cam, "depth"))
        if rgb is None or dep is None:
            continue
        P = world_points(dep, pk["camera_intrinsics"][cam]["matrix"], pk["camera_poses"][cam])
        mk = red_mask(rgb) & (dep > 0)
        if mk.sum() == 0:
            continue
        pts = P[mk]
        d = np.linalg.norm(pts[:, :2] - np.asarray(xy)[None], axis=1)
        c = int(((d < radius) & (pts[:, 2] < 0.04)).sum())
        total += c if cam == "front" else c // 8
    return total


def video_calibration(man, gv, seq):
    """Touch offset (TCP xy - detected disc centre) and drawing height from the video robot state."""
    m = json.loads((WS / man).read_text())
    vf = sorted([f for f in m["frames"] if f["phase"] == "video_demo"], key=lambda f: f["frame_index"])
    FI = [f["frame_index"] for f in vf]
    S = np.array([f.get("state", [0] * 8)[:3] for f in vf], float)
    offs, zs = [], []
    for k, fi in seq:
        if fi not in FI or k not in gv["matched"] or len(S) < 3:
            continue
        i = FI.index(fi)
        w = list(range(max(1, i - 15), min(len(S) - 1, i + 4)))
        if not w:
            continue
        sp = [np.linalg.norm(S[j + 1] - S[j - 1]) for j in w]
        st = S[w[int(np.argmin(sp))]]
        off = np.array(st[:2]) - np.asarray(gv["matched"][k])
        if np.linalg.norm(off) < 0.012 and 0.03 < st[2] < 0.12:
            offs.append(off)
            zs.append(st[2])
    if len(offs) >= 1:
        return np.median(np.array(offs), 0), float(np.median(zs)), len(offs)
    return None, None, 0


def seg_clear(a, b, pts):
    a, b = np.asarray(a), np.asarray(b)
    ab = b - a
    L2 = float(ab @ ab) + 1e-12
    best = 1e9
    for p in pts:
        t = np.clip((p - a) @ ab / L2, 0, 1)
        best = min(best, float(np.linalg.norm(a + t * ab - p)))
    return best


def plan_path(a, b, others, s):
    """Waypoints from a to b (excluding a) keeping clear of other disc centres."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    def clear(path):
        c = 1e9
        prev = a
        for w in path:
            c = min(c, seg_clear(prev, w, others))
            prev = w
        return c
    direct = [b]
    if clear(direct) >= 0.28 * s:
        return direct, clear(direct)
    d = b - a
    n = np.linalg.norm(d) + 1e-12
    perp = np.array([-d[1], d[0]]) / n
    mid = (a + b) / 2
    cands = [direct]
    for t in (0.5, -0.5, 0.35, -0.35):
        cands.append([mid + t * s * perp, b])
        cands.append([a + t * s * perp, b + t * s * perp, b])
    best = max(cands, key=lambda c: (round(clear(c) / (0.05 * s)), -len(c)))
    return best, clear(best)


TIP_MARGIN = 0.003
STEP_CAP = 450
PROBE_DWELL = 7
Z_FLOOR = 0.024


def hex_offsets(spacing, radius):
    pts = []
    n = int(radius / spacing) + 2
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            x = spacing * (i + 0.5 * j)
            y = spacing * (np.sqrt(3) / 2 * j)
            r = np.hypot(x, y)
            if 1e-9 < r <= radius + 1e-9:
                pts.append((r, np.arctan2(y, x), x, y))
    pts.sort()
    return [(x, y) for _, _, x, y in pts]


def search_touch(ctx, rob, xy, z0, centre, log, offsets, step_budget):
    """xy search around a target until it lights red. Returns the successful (dx, dy) or None."""
    start = rob.step_id
    for dx, dy in offsets:
        if rob.ended or rob.step_id - start > step_budget - 11:
            return None
        rob.move_to([xy[0] + dx, xy[1] + dy, z0], max_step=0.12, tol=0.0012, max_iter=4)
        # the expert holds still ~5-6 steps before a disc lights; dwell longer than that
        rob.step([[0, 0, 0, 0, 0, 0, 0]] * PROBE_DWELL)
        cnt = red_at(ctx, rob, centre)
        log("  search", round(dx, 4), round(dy, 4), "red", cnt, "remaining", rob.remaining, "ended", rob.ended)
        if cnt > 3:
            return (dx, dy)
    return None


DEFAULT_OFF = np.array([-0.0002, 0.0023])
DWELL = 5


def run(ctx):
    t0 = time.time()
    rob = Robot(ctx)
    p = rob.observe()
    log("start state", np.round(rob.state, 4).tolist(), "remaining", rob.remaining)
    seq_desc = p.get("frame_sequence") or {}
    man = seq_desc.get("manifest_path")
    if not man:
        log("no frame sequence manifest; abort")
        return
    gv, seq, _ = extract_pattern(man, log=log)
    log("video pattern", [k for k, f in seq], "t=%.1f" % (time.time() - t0))
    off, dz, ncal = video_calibration(man, gv, seq)
    if off is None:
        off, dz = DEFAULT_OFF.copy(), DRAW_Z
    dz = float(np.clip(dz, 0.064, 0.074))
    log("calibration offset", np.round(off, 4).tolist(), "draw_z", round(dz, 4), "n", ncal)
    m = json.loads((WS / man).read_text())
    ex = [f for f in m["frames"] if f["phase"] == "execution"]
    fr = ex[-1] if ex else m["frames"][-1]
    ge = grid_from_frame(fr, log)
    # stick geometry: video stick vs live stick (tool geometry may differ); measured as the lowest
    # robot point in the front depth image relative to TCP, against a robot-free background frame.
    corr_xy, corr_z = np.zeros(2), 0.0
    tip_exec_z = None
    ray = None
    dz_video = dz
    try:
        from stick import tip_from_front, wrist_stick
        from geom import load_frame
        from pattern import frame_points
        vf = sorted([f for f in m["frames"] if f["phase"] == "video_demo"], key=lambda f: f["frame_index"])
        tv = None
        wv = None
        if vf:
            _, bgv, _ = frame_points(vf[0])
            res = []
            cand = [f for f in vf if 0.10 < f["state"][2] < 0.25]
            for f in cand[:: max(1, len(cand) // 25)]:
                _, dv, _ = frame_points(f)
                r = tip_from_front(dv, bgv, f["camera_intrinsics"]["front"]["matrix"], f["camera_poses"]["front"], f["state"][:3])
                if r:
                    res.append(r["tip_rel"])
            if len(res) >= 3:
                tv = np.median(np.array(res), 0)
            wres = []
            for f in cand[:: max(1, len(cand) // 12)]:
                _, dw = load_frame(f, "wrist")
                if dw is None:
                    continue
                r = wrist_stick(dw, f["camera_intrinsics"]["wrist"]["matrix"], f["camera_poses"]["wrist"], f["state"][:3])
                if r:
                    wres.append(np.r_[r["tip_rel"], r["axis"]])
            if len(wres) >= 3:
                wv = np.median(np.array(wres), 0)
        log("video tip_rel", None if tv is None else np.round(tv, 4).tolist(),
            "wrist", None if wv is None else np.round(wv, 4).tolist())
        _, bge, _ = frame_points(fr)
        res = []
        wres_e = []
        cx, cy = ge["center"]
        for hz, dxh in ((0.16, 0.0), (0.20, 0.06), (0.14, -0.04)):
            rob.move_to([cx + dxh, cy, hz], max_step=0.7, tol=0.006)
            for rep in range(2):
                reply = ctx.call("robomme_observe")
                pk, imgs = decode_obs(reply)
                rob._update(pk)
                dep = imgs.get(("front", "depth"))
                if dep is None:
                    continue
                r = tip_from_front(dep, bge, pk["camera_intrinsics"]["front"]["matrix"], pk["camera_poses"]["front"], rob.state[:3])
                if r:
                    res.append(r["tip_rel"])
                dw = imgs.get(("wrist", "depth"))
                if dw is not None:
                    r = wrist_stick(dw, pk["camera_intrinsics"]["wrist"]["matrix"], pk["camera_poses"]["wrist"], rob.state[:3])
                    if r:
                        wres_e.append(np.r_[r["tip_rel"], r["axis"]])
            if len(res) >= 2:
                break
        te = np.median(np.array(res), 0) if res else None
        we = np.median(np.array(wres_e), 0) if len(wres_e) >= 2 else None
        log("exec tip_rel", None if te is None else np.round(te, 4).tolist(), "n", len(res),
            "wrist", None if we is None else np.round(we, 4).tolist(), len(wres_e))
        if tv is None:
            tv = np.array([0.0052, -0.003, 0.0027])
        if wv is not None and we is not None:
            ray = (wv, we)
            corr_z = float(we[2] - wv[2])
            tip_exec_z = float(we[2])
        elif te is not None:
            corr_z = float(te[2] - tv[2])
            corr_xy = te[:2] - tv[:2]
            tip_exec_z = float(te[2])
            if abs(corr_z) > 0.12 or np.linalg.norm(corr_xy) > 0.08:
                log("stick correction implausible; ignoring", corr_z, corr_xy)
                corr_xy, corr_z = np.zeros(2), 0.0
    except Exception:
        log("stick geometry error\n" + traceback.format_exc())
    dz = dz - corr_z - TIP_MARGIN - 0.3 * max(corr_z, 0.0)
    dz = max(dz, Z_FLOOR)
    if ray is not None:
        try:
            from stick import hit_rel
            wv, we = ray
            ztop = float(ge.get("top_z", 0.0164) or 0.0164)
            hv = hit_rel(wv[:3], wv[3:], dz_video, ztop)
            he = hit_rel(we[:3], we[3:], dz, ztop)
            corr_xy = he - hv
            log("ray model: hit video", np.round(hv, 4).tolist(), "hit exec", np.round(he, 4).tolist(), "ztop", round(ztop, 4))
            if np.linalg.norm(corr_xy) > 0.08:
                log("ray correction implausible; ignoring")
                corr_xy = np.zeros(2)
        except Exception:
            log("ray model error\n" + traceback.format_exc())
    log("stick correction xy", np.round(corr_xy, 4).tolist(), "z", round(corr_z, 4), "mode", "ray" if ray is not None else "front")
    off = off - corr_xy
    targets = []
    for k, _f in seq:
        xy = ge["matched"].get(k, ge["lattice"][k])
        targets.append((k, np.array(xy, float) + off))
    log("targets", [(k, np.round(xy, 4).tolist()) for k, xy in targets])
    if not targets:
        return
    calibrated = False
    log("draw z", round(dz, 4), "floor", Z_FLOOR)
    k0, xy0 = targets[0]
    rob.move_to([xy0[0], xy0[1], HOVER_Z], max_step=0.7, tol=0.006)
    for n in range(len(targets)):
        k, xy = targets[n]
        if rob.ended:
            break
        if n == 0:
            rob.move_to([xy[0], xy[1], dz + 0.008], max_step=0.5, tol=0.004)
        else:
            # drag at drawing height, detouring around non-pattern discs
            prev_c = targets[n - 1][1] - off
            others = [np.asarray(v) for kk, v in ge["matched"].items() if kk != k and kk != targets[n - 1][0]]
            path, clr = plan_path(prev_c, xy - off, others, ge["spacing"])
            log("segment", targets[n - 1][0], "->", k, "waypoints", len(path), "clearance %.3f" % clr)
            for w in path[:-1]:
                rob.move_to([w[0] + off[0], w[1] + off[1], dz], max_step=0.35, tol=0.008, max_iter=3)
            rob.move_to([xy[0], xy[1], dz], max_step=0.35, tol=0.006, max_iter=3)
        rob.move_to([xy[0], xy[1], dz], max_step=0.08, tol=0.0012, max_iter=5)
        if rob.ended:
            break
        rob.step([[0, 0, 0, 0, 0, 0, 0]] * DWELL)
        cnt = red_at(ctx, rob, xy - off)
        if cnt <= 3 and not rob.ended and not rob.complete:
            rob.step([[0, 0, 0, 0, 0, 0, 0]] * 3)
            cnt = red_at(ctx, rob, xy - off)
        log("target", n, k, "pos", np.round(rob.state[:3], 4).tolist(), "red", cnt,
            "remaining", rob.remaining, "step", rob.step_id, "ended", rob.ended, "complete", rob.complete)
        if cnt > 3:
            calibrated = True
        elif not rob.ended:
            left = len(targets) - n - 1
            spare = rob.left() - 28 * left - 8
            if not calibrated:
                offs = hex_offsets(0.005, 0.0125)
                budget = max(0, spare)
            else:
                offs = hex_offsets(0.004, 0.0045)
                budget = max(0, min(spare, 50))
            log("  search budget", budget, "left", rob.left())
            found = search_touch(ctx, rob, xy, dz, xy - off, log, offs, budget)
            if found:
                fxy = np.array(found[:2])
                calibrated = True
                off = off + fxy
                targets = [(kk, vv + fxy) for kk, vv in targets]
                log("adopted correction", np.round(fxy, 4).tolist(), "off", np.round(off, 4).tolist())
    if not rob.ended:
        rob.step([[0, 0, 0, 0, 0, 0, 0]] * 5)
    log("done ended=%s complete=%s remaining=%s t=%.1f" % (rob.ended, rob.complete, rob.remaining, time.time() - t0))


def main(ctx):
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        run(ctx)
    except Exception:
        log("controller exception:\n" + traceback.format_exc())
