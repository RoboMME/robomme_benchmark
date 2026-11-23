"""VideoUnmask controller v1: colours from conditioning video -> container lift sequence."""
import base64, io, json, os, re, sys, time, traceback
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from motion import Robot, packet, gripper_yaw, rotate_to_yaw, wrap90  # noqa: E402
from video_cubes import cube_clusters  # noqa: E402
from perception import (cube_tracks, find_containers, backproject, components, COLOR_HUES)  # noqa: E402

OUT = Path("/workspace/output")
COLORS = ["red", "orange", "yellow", "green", "blue", "purple"]
LIFT_Z = 0.20
GRASP_BELOW_TOP = 0.03


def log_factory():
    OUT.mkdir(parents=True, exist_ok=True)
    f = open(OUT / "controller_log.txt", "a")

    def log(*a):
        s = " ".join(str(x) for x in a)
        print(s)
        f.write(s + "\n"); f.flush()
    return log


def parse_targets(instr):
    s = instr.lower()
    hits = []
    for c in COLORS:
        for m in re.finditer(r"\b" + c + r"\b", s):
            hits.append((m.start(), c))
    hits.sort()
    out = []
    for _, c in hits:
        out.append(c)
    return out


def decode_images(reply, pkt):
    blocks = reply["content"]
    imgs = {}
    for a in pkt.get("attachments", []):
        b = blocks[a["content_index"]]
        if b.get("type") == "image":
            imgs[(a["camera"], a["kind"])] = Image.open(io.BytesIO(base64.b64decode(b["data"])))
    return imgs


def live_view(ctx, cam):
    reply = ctx.call("robomme_observe")
    pkt = packet(reply)
    imgs = decode_images(reply, pkt)
    rgb = imgs[(cam, "rgb")].convert("RGB")
    hsv = np.asarray(rgb.convert("HSV")).astype(int)
    depth = np.asarray(imgs[(cam, "depth")]).astype(float) * 0.001
    return pkt, np.asarray(rgb).astype(int), hsv, depth


def container_yaw(ctx_unused, cont):
    return cont.get("yaw", 0.0)


def rect_fit(pts2):
    c = pts2.mean(0); q = pts2 - c
    best = None
    for a in np.radians(np.arange(0, 90, 1.0)):
        R = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
        r = q @ R.T; ext = r.max(0) - r.min(0)
        if best is None or ext[0] * ext[1] < best[0]:
            best = (ext[0] * ext[1], a, ext, (r.max(0) + r.min(0)) / 2, R)
    _, a, ext, mid, R = best
    return c + R.T @ mid, float(a), ext


def kmeans(pts, k, iters=20):
    idx = np.linspace(0, len(pts) - 1, k).astype(int)
    order = np.argsort(pts[:, 0] + pts[:, 1])
    cent = pts[order[idx]].copy()
    for _ in range(iters):
        lab = np.argmin(((pts[:, None, :] - cent[None]) ** 2).sum(-1), 1)
        for j in range(k):
            if (lab == j).any():
                cent[j] = pts[lab == j].mean(0)
    return lab


def containers_from_arrays(hsv, depth, K, pose, log):
    H, W = depth.shape
    vv, uu = np.mgrid[0:H, 0:W]
    P = backproject(uu.ravel(), vv.ravel(), depth.ravel(), K, pose).reshape(H, W, 3)
    s, v = hsv[..., 1], hsv[..., 2]
    table = (depth > 0) & (P[..., 2] < 0.03) & (P[..., 0] > -0.4)
    tz = float(np.median(P[..., 2][table])) if table.sum() > 100 else 0.0
    m = (depth > 0) & (P[..., 2] > tz + 0.04) & (P[..., 2] < tz + 0.15) & (P[..., 0] > -0.40) & (P[..., 0] < 0.17) \
        & (np.abs(P[..., 1]) < 0.40) & (s < 90)
    raw = []
    for ys, xs in components(m, 20):
        pts = P[ys, xs]
        zt = np.percentile(pts[:, 2], 90)
        raw.append(pts[pts[:, 2] > zt - 0.006])
    merged = []
    for top in raw:
        c = np.median(top[:, :2], 0)
        for i, g in enumerate(merged):
            if np.linalg.norm(np.median(g[:, :2], 0) - c) < 0.03:
                merged[i] = np.vstack([g, top]); break
        else:
            merged.append(top)
    parts = []
    for top in merged:
        if len(top) < 20:
            continue
        centre, a, ext = rect_fit(top[:, :2])
        k = int(np.clip(round(ext[0] * ext[1] / 0.0016), 1, 4)) if max(ext) > 0.055 else 1
        if k > 1:
            lab = kmeans(top[:, :2], k)
            log(f"split blob ext={ext.round(3)} into {k}")
            for j in range(k):
                sub = top[lab == j]
                if len(sub) >= 15:
                    parts.append(sub)
        else:
            parts.append(top)
    out = []
    for top in parts:
        centre, a, ext = rect_fit(top[:, :2])
        out.append({"xy": centre.tolist(), "ztop": float(np.median(top[:, 2])), "yaw": a,
                    "ext": ext.tolist(), "n": int(len(top))})
    log(f"table z {tz:.4f}; containers:", "; ".join(
        f"({o['xy'][0]:.3f},{o['xy'][1]:.3f}) z={o['ztop']:.3f} yaw={np.degrees(o['yaw']):.0f} ext=({o['ext'][0]:.3f},{o['ext'][1]:.3f}) n={o['n']}"
        for o in out))
    return out, tz


def detect_containers(ctx, log, cam="wrist"):
    pkt, rgb, hsv, depth = live_view(ctx, cam)
    K = pkt["camera_intrinsics"][cam]["matrix"]; pose = pkt["camera_poses"][cam]
    out, tz = containers_from_arrays(hsv, depth, K, pose, log)
    return out, tz, pkt


def choose_grasp_yaw(cont, others):
    """Gripper yaw (TCP x-axis) aligned with the box; of the two square options pick more finger clearance.
    Fingers close along TCP y (yaw+90deg)."""
    a0 = wrap90(cont["yaw"])
    a1 = a0 + np.pi / 2 if a0 < 0 else a0 - np.pi / 2
    c = np.array(cont["xy"])
    def clearance(a):
        f = np.array([-np.sin(a), np.cos(a)])
        clear = 1.0
        for o in others:
            q = np.array(o["xy"])
            if np.linalg.norm(q - c) < 0.01:
                continue
            for sgn in (1, -1):
                clear = min(clear, np.linalg.norm(q - (c + sgn * 0.05 * f)))
        return clear
    c0, c1 = clearance(a0), clearance(a1)
    best = (0, a1, c1) if (c0 < 0.05 and c1 > c0 + 0.01) else (0, a0, c0)
    return best[1], best[2]


FAULT = {"high_grasp": int(os.environ.get("VU_FAULT_HIGH_GRASP", "0"))}  # dev-only fault injection


def pick_and_replace(robot, cont, log, others=()):
    x, y = cont["xy"]; ztop = cont["ztop"]
    zg = ztop - GRASP_BELOW_TOP
    if FAULT["high_grasp"] > 0:
        FAULT["high_grasp"] -= 1
        zg = ztop + 0.03
        log("FAULT INJECTION: grasp above container")
    robot.move_to([x, y, ztop + 0.10], grip=-1)
    gy, clear = choose_grasp_yaw(cont, others)
    log(f"grasp yaw {np.degrees(gy):.0f} finger clearance {clear:.3f}")
    rotate_to_yaw(robot, gy)
    robot.move_to([x, y, ztop + 0.04], grip=-1, tol=0.004)
    robot.move_to([x, y, zg], grip=-1, tol=0.003, speed=0.3)
    log("at grasp", robot.state().round(4).tolist())
    robot.set_gripper(1, 10)
    fingers = robot.state()[6:8].sum()
    log("closed fingers", round(float(fingers), 4))
    if fingers < 0.01:
        log("grasp miss")
        robot.set_gripper(-1, 6)
        robot.move_to([x, y, ztop + 0.08], grip=-1)
        return False
    robot.move_to([x, y, LIFT_Z], grip=1, speed=0.4)
    fingers = robot.state()[6:8].sum()
    log("lifted", robot.state().round(4).tolist(), "fingers", round(float(fingers), 4), "complete", robot.complete)
    held = fingers > 0.01
    if robot.ended:
        return held
    # put it back down where it was
    robot.move_to([x, y, zg + 0.02], grip=1, speed=0.4)
    robot.move_to([x, y, zg + 0.004], grip=1, speed=0.2, tol=0.003)
    robot.set_gripper(-1, 8)
    robot.move_to([x, y, ztop + 0.10], grip=-1)
    return held


def merge_clusters(cl, radius=0.04):
    out = []
    for c in sorted(cl, key=lambda c: -c["count"]):
        for o in out:
            if o["color"] == c["color"] and np.linalg.norm(np.array(o["xy"]) - np.array(c["xy"])) < radius:
                o["count"] += c["count"]
                break
        else:
            out.append(dict(c))
    return out


def plan_targets(instr):
    """List of (colour, want_all) in instruction order."""
    s = instr.lower()
    if "then" in s:
        s = s.split("then", 1)[1]
    plan = []
    for m in re.finditer(r"(all containers?|the container|the one|ones?|containers?)[^,.;]*?\b(" + "|".join(COLORS) + r")\b(\s+cubes?)?", s):
        phrase = m.group(0)
        want_all = ("all" in phrase.split()[0]) or (m.group(3) or "").strip() == "cubes"
        plan.append((m.group(2), want_all))
    seen = [c for c, _ in plan]
    for m in re.finditer(r"\b(" + "|".join(COLORS) + r")\b(\s+cubes)?", s):
        if m.group(1) not in seen:
            plan.append((m.group(1), bool(m.group(2))))
            seen.append(m.group(1))
    return plan


def main(ctx):
    log = log_factory()
    t0 = time.time()
    robot = Robot(ctx, log)
    reply = ctx.call("robomme_observe")
    pkt = packet(reply)
    robot.last = pkt
    instr = pkt.get("instruction", "")
    plan = plan_targets(instr)
    log("instruction:", instr); log("plan:", plan)
    seq = pkt.get("frame_sequence") or {}
    mpath = seq.get("manifest_path")
    if not mpath:
        log("no manifest; abort"); return
    colors = list(dict.fromkeys([c for c, _ in plan] + ["red", "green", "blue", "yellow", "orange", "purple"]))
    if "black" in [c for c, _ in plan]:
        colors.append("black")
    try:
        clusters = merge_clusters(cube_clusters(mpath, colors, log=log))
    except Exception:
        log("cube clustering failed", traceback.format_exc()); clusters = []
    log(f"perception time {time.time()-t0:.1f}s")
    conts, tz, _ = detect_containers(ctx, log, "wrist")
    if len(conts) < 2:
        conts2, _, _ = detect_containers(ctx, log, "front")
        if len(conts2) > len(conts):
            conts = conts2
    used = set()
    for colour, want_all in plan:
        if robot.ended:
            break
        cl = [c for c in clusters if c["color"] == colour]
        if not want_all and len(cl) > 1:
            cl = sorted(cl, key=lambda c: -c["count"])[:1]
        if not cl:
            log("no cube seen for", colour); continue
        # assign containers
        jobs = []
        for c in cl:
            cxy = np.array(c["xy"])
            cand = [(np.linalg.norm(np.array(o["xy"]) - cxy), j) for j, o in enumerate(conts) if j not in used]
            if not cand:
                continue
            d, j = min(cand)
            log(f"target {colour}: cube xy {cxy.round(3)} -> container {j} {np.round(conts[j]['xy'], 3)} d={d:.3f}")
            used.add(j); jobs.append(j)
        # nearest-first order
        while jobs and not robot.ended:
            pos = robot.state()[:2]
            jobs.sort(key=lambda j: np.linalg.norm(np.array(conts[j]["xy"]) - pos))
            j = jobs.pop(0)
            ok = False
            for attempt in range(3):
                if robot.ended:
                    break
                try:
                    ok = pick_and_replace(robot, conts[j], log, conts)
                    if ok:
                        break
                    log(f"attempt {attempt} failed; re-detecting")
                    robot.move_to([conts[j]["xy"][0], conts[j]["xy"][1], 0.45], grip=-1)
                    new, _, _ = detect_containers(ctx, log, "wrist")
                    if new:
                        dd = [np.linalg.norm(np.array(o["xy"]) - np.array(conts[j]["xy"])) for o in new]
                        k = int(np.argmin(dd))
                        if dd[k] < 0.05:
                            conts[j] = new[k]
                except Exception:
                    log("attempt error", traceback.format_exc())
                    try:
                        robot.observe()
                    except Exception:
                        pass
            log(f"done {colour} container {j}: ok={ok} complete={robot.complete} ended={robot.ended} "
                f"step={robot.last.get('step_id')} t={time.time()-t0:.0f}s")
    log("final complete", robot.complete, "ended", robot.ended)
