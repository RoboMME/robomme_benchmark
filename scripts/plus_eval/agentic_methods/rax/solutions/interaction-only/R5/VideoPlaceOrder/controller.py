"""VideoPlaceOrder controller: read conditioning-video EE states, find the k-th placement,
pick the instructed cube and place it gently on that target."""
import json, math, os, re, sys, time, traceback
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from perception import parse_reply, world_points, components, purple_mask, cluster_points
from motion import Robot, yaw_of

WS = Path("/workspace")
OUT = WS / "output"

ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7,
            "eighth": 8, "ninth": 9, "tenth": 10, "1st": 1, "2nd": 2, "3rd": 3, "4th": 4, "5th": 5,
            "6th": 6, "7th": 7, "8th": 8, "once": 1, "twice": 2}

LOG = []


def log(*a):
    msg = " ".join(str(x) for x in a)
    LOG.append(msg)
    print(msg, flush=True)
    try:
        OUT.mkdir(parents=True, exist_ok=True)
        with open(OUT / "controller_log.txt", "a") as f:
            f.write(msg + "\n")
    except Exception:
        pass


def color_mask(rgb, color):
    r, g, b = [rgb[..., i].astype(int) for i in range(3)]
    if color == "red":
        return (r > 110) & (g < 60) & (b < 60)
    if color == "green":
        return (g > 100) & (r < 80) & (b < 80)
    if color == "blue":
        return (b > 100) & (r < 60) & (g < 60)
    if color == "yellow":
        return (r > 150) & (g > 150) & (b < 70)
    return (r > 110) & (g < 60) & (b < 60)


def parse_instruction(text):
    t = (text or "").lower()
    k = None
    for w, v in ORDINALS.items():
        if re.search(r"\b" + re.escape(w) + r"\b", t):
            k = v
            break
    last = bool(re.search(r"\blast\b", t))
    m = re.search(r"\b(red|green|blue|yellow)\b\s+cube", t)
    color = m.group(1) if m else "red"
    return k, last, color


def load_manifest(meta):
    seq = meta.get("frame_sequence") or {}
    mp = seq.get("manifest_path")
    if not mp:
        return None
    p = WS / mp
    if not p.exists():
        p = Path(mp)
    return json.loads(p.read_text())


def load_video_states(man):
    frames = [f for f in man.get("frames", []) if f.get("phase") == "video_demo"]
    frames.sort(key=lambda f: f.get("frame_index", 0))
    return [(f.get("frame_index"), f.get("state")) for f in frames if f.get("state")]


def load_frame_obs(frame):
    """Turn a manifest frame into (meta, ims) usable by the detectors."""
    from PIL import Image
    ims = {}
    for f in frame.get("files", []):
        p = WS / f["path"]
        if p.exists():
            ims[(f["camera"], f["kind"])] = np.array(Image.open(p))
    meta = {"camera_intrinsics": frame.get("camera_intrinsics"), "camera_poses": frame.get("camera_poses"),
            "depth": frame.get("depth", {})}
    return meta, ims


def grip_events(states):
    """List of ('grasp'|'release', frame, xyz) for objects held near the table."""
    ev = []
    held = False
    last = None
    for idx, s in states:
        g = s[6] + s[7]
        if 0.012 < g < 0.065:
            if not held and s[2] < 0.08:
                ev.append(("grasp", idx, [s[0], s[1], s[2]]))
            held = True
            last = (idx, s)
        elif g >= 0.065:
            if held and last is not None and last[1][2] < 0.08:
                ls = last[1]
                ev.append(("release", last[0], [ls[0], ls[1], ls[2]]))
            held = False
        else:
            held = False
    return ev


def release_events(states):
    return [{"frame": f, "xyz": p} for k, f, p in grip_events(states) if k == "release"]


def cube_positions(meta, ims, cams=("front", "wrist")):
    out = {}
    for color in ("red", "green", "blue", "yellow"):
        d = cube_top(meta, ims, color, cams=cams)
        if d is not None and d[2] >= 15:
            out[color] = d[0]
    return out


def placements_for(states, init_pos, color):
    """Track which cube is carried in each grasp; return release xyz list for `color`."""
    pos = {c: np.array(p[:2], float) for c, p in init_pos.items()}
    out = []
    cur = None
    for kind, f, p in grip_events(states):
        p2 = np.array(p[:2], float)
        if kind == "grasp":
            cur = None
            if pos:
                c, d = min(((c, np.linalg.norm(q - p2)) for c, q in pos.items()), key=lambda t: t[1])
                cur = c if d < 0.06 else None
            log("video grasp", f, np.round(p2, 3).tolist(), "->", cur)
        else:
            if cur is not None:
                pos[cur] = p2
            if cur == color or cur is None:
                out.append({"frame": f, "xyz": p, "obj": cur})
            cur = None
    return out


def detect_targets(meta, ims):
    pts = []
    for cam in ("wrist", "front"):
        if (cam, "rgb") not in ims or (cam, "depth") not in ims:
            continue
        P, valid = world_points(meta, ims, cam)
        m = purple_mask(ims[(cam, "rgb")][..., :3]) & valid
        for pix in components(m, 5):
            q = P[pix[:, 0], pix[:, 1]]
            q = q[q[:, 2] < 0.03]
            if len(q) >= 3:
                pts.append((np.median(q, axis=0), len(pix), cam))
    # cluster within 4 cm, weight by size; prefer wrist
    clusters = []
    for p, n, cam in pts:
        w = n * (2.0 if cam == "wrist" else 1.0)
        for c in clusters:
            if np.linalg.norm(c["s"][:2] / c["w"] - p[:2]) < 0.04:
                c["s"] += p * w; c["w"] += w
                break
        else:
            clusters.append({"s": p * w, "w": w})
    return [(c["s"] / c["w"]).tolist() for c in clusters if c["w"] > 15]


def cube_top(meta, ims, color, cams=("wrist", "front"), near=None):
    """Return (center_xyz_of_top_face, yaw_mod90, n_points) for the coloured cube."""
    best = None
    for cam in cams:
        if (cam, "rgb") not in ims or (cam, "depth") not in ims:
            continue
        P, valid = world_points(meta, ims, cam)
        m = color_mask(ims[(cam, "rgb")][..., :3], color) & valid
        comps = components(m, 4)
        for pix in comps:
            q = P[pix[:, 0], pix[:, 1]]
            q = q[(q[:, 2] > 0.005) & (q[:, 2] < 0.12)]
            if len(q) < 4:
                continue
            c0 = np.median(q, axis=0)
            if near is not None and np.linalg.norm(c0[:2] - np.asarray(near[:2])) > 0.08:
                continue
            top = q[q[:, 2] > q[:, 2].max() - 0.006]
            score = len(pix) * (3.0 if cam == "wrist" else 1.0)
            if best is None or score > best[3]:
                best = (top, cam, len(pix), score)
    if best is None:
        return None
    top, cam, n, _ = best
    xy = top[:, :2]
    # yaw via min-area bounding box
    bestA, bestyaw, bestc = 1e9, 0.0, xy.mean(0)
    for deg in range(0, 90, 3):
        a = math.radians(deg)
        Rm = np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]])
        r = xy @ Rm.T
        lo, hi = np.percentile(r, 3, axis=0), np.percentile(r, 97, axis=0)
        A = np.prod(hi - lo)
        if A < bestA:
            bestA, bestyaw = A, a
            cr = (lo + hi) / 2
            bestc = cr @ Rm  # back to world
    yaw = bestyaw
    if yaw > math.pi / 4:
        yaw -= math.pi / 2
    z = float(np.median(top[:, 2]))
    return [float(bestc[0]), float(bestc[1]), z], yaw, n, cam


def refine_target(meta, ims, coarse, radius=0.045):
    """Centroid of purple pixels (rings + centre dot) near the coarse target centre, wrist camera."""
    best = None
    for cam in ("wrist", "front"):
        if (cam, "rgb") not in ims or (cam, "depth") not in ims:
            continue
        P, valid = world_points(meta, ims, cam)
        m = purple_mask(ims[(cam, "rgb")][..., :3]) & valid
        q = P[m]
        if len(q) == 0:
            continue
        q = q[(q[:, 2] < 0.03) & (np.linalg.norm(q[:, :2] - np.asarray(coarse[:2]), axis=1) < radius)]
        if len(q) < 30:
            continue
        c = q.mean(axis=0)
        # second pass centred on the first estimate
        q2 = P[m]
        q2 = q2[(q2[:, 2] < 0.03) & (np.linalg.norm(q2[:, :2] - c[:2], axis=1) < radius)]
        if len(q2) >= 30:
            c = q2.mean(axis=0)
        best = (c.tolist(), len(q2), cam)
        break
    return best


def inhand_offset(meta, ims, color, tcp):
    """Cube top-face centre minus TCP (xy) while the cube is held, from the wrist camera."""
    if ("wrist", "rgb") not in ims or ("wrist", "depth") not in ims:
        return None
    P, valid = world_points(meta, ims, "wrist")
    m = color_mask(ims[("wrist", "rgb")][..., :3], color) & valid
    q = P[m]
    if len(q) < 50:
        return None
    q = q[(q[:, 2] > tcp[2] - 0.01) & (q[:, 2] < tcp[2] + 0.04) &
          (np.linalg.norm(q[:, :2] - np.asarray(tcp[:2]), axis=1) < 0.05)]
    if len(q) < 50:
        return None
    top = q[q[:, 2] > q[:, 2].max() - 0.004]
    if len(top) < 30:
        return None
    lo, hi = np.percentile(top[:, :2], 3, axis=0), np.percentile(top[:, :2], 97, axis=0)
    c = (lo + hi) / 2
    return [float(c[0] - tcp[0]), float(c[1] - tcp[1])], len(top), (hi - lo).tolist()


def choose_goal(seq, k, last, rule, s0, s_ret):
    """seq: list of target centres the cube was placed on (chronological)."""
    n = len(seq)
    if n == 0:
        return None, "none"
    if k is None:
        return (seq[-1], "last") if last else (seq[0], "first")
    if rule == "F":
        i = min(max(k, 1), n) - 1
        return seq[i], "F[%d]" % i
    if rule == "F-1":
        hist = [None] + list(seq)  # initial spot (not a target) first
        i = min(max(k, 1), len(hist)) - 1
        if hist[i] is None:
            return seq[0], "F-1 fallback first"
        return hist[i], "F-1 hist[%d]" % i
    if rule == "BR":
        hist = list(seq) + [None]  # final return (not a target) last
        i = len(hist) - min(max(k, 1), len(hist))
        if hist[i] is None:
            return seq[-1], "BR fallback last"
        return hist[i], "BR hist[%d]" % i
    return seq[min(max(k, 1), n) - 1], "default"


def ended(R):
    m = R.meta or {}
    tr = m.get("transition") or {}
    if tr.get("task_complete"):
        R.success = True
    for s in tr.get("steps", []) or []:
        if s.get("task_complete"):
            R.success = True
        if s.get("episode_ended"):
            R.ended = True
    if m.get("episode_ended") or tr.get("episode_ended"):
        R.ended = True
    return R.ended


class RobotX(Robot):
    def step(self, rows):
        out = super().step(rows)
        ended(self)
        return out


def grasp_cube(R, color, home_z=0.16):
    for attempt in range(3):
        if R.ended:
            return False
        meta, ims = R.observe()
        det = cube_top(meta, ims, color)
        if det is None:
            log("cube not found; moving to overview")
            R.move_to([-0.06, 0.0, 0.40], -1, yaw=0.0, max_steps=40)
            meta, ims = R.observe()
            det = cube_top(meta, ims, color)
            if det is None:
                log("cube still not found")
                return False
        c, yaw, n, cam = det
        log("cube coarse", np.round(c, 3).tolist(), "yaw", round(math.degrees(yaw), 1), cam, n)
        # hover above with wrist camera roughly over cube (camera offset ~ +0.045x)
        R.move_to([c[0], c[1], home_z], -1, yaw=yaw, max_steps=60)
        meta, ims = R.observe()
        det2 = cube_top(meta, ims, color, cams=("wrist",), near=c)
        if det2 is not None:
            c, yaw, n, cam = det2
            log("cube fine", np.round(c, 3).tolist(), "yaw", round(math.degrees(yaw), 1), n)
        zc = max(0.012, c[2] - 0.02)  # grasp at cube centre height
        R.move_to([c[0], c[1], zc + 0.04], -1, yaw=yaw, max_steps=40, vmax=0.5)
        R.move_to([c[0], c[1], zc], -1, yaw=yaw, max_steps=30, vmax=0.25)
        R.gripper(1, 10)
        g = R.state()[6] + R.state()[7]
        log("grip width after close", round(g, 4))
        R.move_to([c[0], c[1], 0.12], 1, yaw=yaw, max_steps=40, vmax=0.5)
        g = R.state()[6] + R.state()[7]
        if 0.012 < g < 0.07:
            log("holding, width", round(g, 4))
            return True
        log("grasp failed, width", round(g, 4))
        R.move_to(list(R.state()[:2]) + [0.16], -1, max_steps=20)
    return False


def place(R, tgt, color, z_release=0.018, offset=(0.0, 0.0)):
    yaw = yaw_of(R.state())
    gx, gy = tgt[0] - offset[0], tgt[1] - offset[1]
    R.move_to([gx, gy, 0.12], 1, yaw=yaw, max_steps=60)
    R.move_to([gx, gy, 0.06], 1, yaw=yaw, max_steps=30, vmax=0.4, tol=0.003)
    # slow descent to the height the video released at (cube resting on the target)
    R.move_to([gx, gy, z_release], 1, yaw=yaw, max_steps=50, vmax=0.12, tol=0.002)
    log("pre-release state", np.round(R.state(), 4).tolist(), "tcp goal", round(gx, 4), round(gy, 4))
    R.step([[0, 0, -0.05, 0, 0, 0, 1]] * 3)
    log("pre-open state", np.round(R.state(), 4).tolist())
    R.gripper(-1, 10)
    if not R.ended:
        R.move_to([gx, gy, 0.15], -1, yaw=yaw, max_steps=30)


def main(ctx):
    try:
        _main(ctx)
    except Exception:
        log("controller exception:\n" + traceback.format_exc())


def _main(ctx):
    t0 = time.time()
    R = RobotX(ctx, log=log)
    meta, ims = R.observe()
    instr = meta.get("instruction", "")
    k, last, color = parse_instruction(instr)
    log("instruction:", instr, "-> k", k, "last", last, "color", color)
    try:
        man = load_manifest(meta)
        states = load_video_states(man)
    except Exception as e:
        log("video load failed", repr(e)); man = None; states = []
    init = {}
    try:
        vf = [f for f in man.get("frames", []) if f.get("phase") == "video_demo"]
        if vf:
            m0, i0 = load_frame_obs(sorted(vf, key=lambda f: f.get("frame_index", 0))[0])
            init = cube_positions(m0, i0, cams=("front",))
    except Exception as e:
        log("frame0 cube detection failed", repr(e))
    cur_pos = cube_positions(meta, ims)
    for c, p in cur_pos.items():
        init.setdefault(c, p)
    log("initial cube positions", {c: np.round(p, 3).tolist() for c, p in init.items()})
    ev = placements_for(states, init, color)
    log("release events for", color, [(e["frame"], np.round(e["xyz"], 3).tolist(), e["obj"]) for e in ev])
    targets = detect_targets(meta, ims)
    log("targets", np.round(targets, 3).tolist())
    # map events to targets
    seq = []
    for e in ev:
        if not targets:
            seq.append(e["xyz"]); continue
        d = [np.linalg.norm(np.array(t[:2]) - np.array(e["xyz"][:2])) for t in targets]
        i = int(np.argmin(d))
        if d[i] < 0.045:
            seq.append(targets[i])
        else:
            log("event not on a target (skipped)", np.round(e["xyz"], 3).tolist(), "dist", round(d[i], 3))
    log("placement sequence", np.round(seq, 3).tolist() if seq else [])
    if not seq:
        log("no placements found; fallback to first target")
        seq = targets[:1]
    if not seq:
        return
    rule = load_rule()
    goal, why = choose_goal(seq, k, last, rule, None, None)
    log("rule", rule, "choice", why, "goal", np.round(goal, 3).tolist())
    zs = [e["xyz"][2] for e in ev if e["xyz"][2] < 0.06]
    z_rel = float(np.median(zs)) if zs else 0.019
    z_rel = min(max(z_rel, 0.012), 0.024)
    # close-up of the goal target with the empty gripper (wrist camera sits ~+0.046 x, +0.02 y of TCP)
    try:
        R.move_to([goal[0] - 0.045, goal[1] - 0.02, 0.17], -1, yaw=0.0, max_steps=60, tol=0.006)
        m1, i1 = R.observe()
        ref = refine_target(m1, i1, goal)
        if ref is not None and np.linalg.norm(np.array(ref[0][:2]) - np.array(goal[:2])) < 0.03:
            log("goal refined", np.round(goal, 4).tolist(), "->", np.round(ref[0], 4).tolist(), "n", ref[1])
            goal = [ref[0][0], ref[0][1], goal[2]]
        else:
            log("goal refine rejected", ref)
    except Exception:
        log("refine failed", traceback.format_exc())
    if R.ended:
        return
    if not grasp_cube(R, color):
        log("could not grasp")
        return
    if R.ended:
        return
    offset = (0.0, 0.0)
    try:
        m2, i2 = R.observe()
        off = inhand_offset(m2, i2, color, R.state()[:3])
        log("in-hand offset", off)
        # logged only: the wrist sees part of the held cube's top face, so this estimate is biased (+x)
        # and is not applied.
    except Exception:
        log("offset failed", traceback.format_exc())
    log("release z", round(z_rel, 4), "offset", np.round(offset, 4).tolist())
    place(R, goal, color, z_release=z_rel, offset=offset)
    log("done; ended", R.ended, "success", R.success, "steps", R.meta.get("step_id"), "t", round(time.time() - t0, 1))


def load_rule():
    try:
        return json.loads((Path(__file__).parent / "rule.json").read_text()).get("rule", "F-1")
    except Exception:
        return "F-1"
