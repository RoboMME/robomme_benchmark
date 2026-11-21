# RoboMME-Plus BinFill — v03 (ASPIRE CaP, demo-only: written offline, 0 simulator starts)
# v03 vs v02: API-shape robustness. SegmentationSet is read through seg_list(), which accepts
# either .segmentations[i].(mask, box_xyxy, score) (skills/localize.md) or parallel
# .masks/.boxes/.scores (docs/api-reference.md contract table); point-prompt masks must contain
# the prompted pixel (the other axis order is tried once); a step reply is terminal on
# terminated/truncated/terminal/done or on two consecutive ok=False replies.
# v02 vs v01: the held-colour check examines up to HELD_BLOBS wrist blobs (not only the 3
# largest) and returns every colour whose blob is near the end effector, so a cube whose shaded
# face splits into two hue classes (yellow/orange seen in the recordings) is not rejected.
# v01 vs v00: (a) follow_path can execute the IK-feasible prefix (button press no longer aborts
# when the deepest press keyframe is unreachable); (b) after lifting, the held cube's colour is
# verified in the wrist view by 3-D proximity to the end effector (unbounded blob size), and a
# wrong-colour cube is either credited to another still-needed colour or dropped outside the bin.
#
# Strategy (derived from the released recordings' task STRUCTURE, never their coordinates):
#   1. Parse "put <n> <colour> cube(s) [and ...] into the bin, then press the button" at runtime.
#   2. Localize the static bin and button once (front camera, SAM3 text prompts + colour checks).
#   3. Loop: scan the front RGB for saturated cube colours, lift needed-colour blobs to 3D with
#      SAM3 point prompts, estimate the belt velocity from timestamped re-observations (zero when
#      cubes are static), plan a time-synchronised intercept, refine with the wrist camera, follow
#      the cube while descending/closing/lifting, carry it to the bin and release. Delivered
#      cubes are remembered per colour (cubes inside the bin are not visible).
#   4. When every requested count is delivered, press the button with a closed gripper.
# Everything episode-dependent comes from the goal string and live observations.
import math

# ---------------------------------------------------------------- generic tunables
SCAN_STRIDE = 2            # pixel stride of the colour scan (front)
WRIST_STRIDE = 3
SAT_MIN = 0.75             # cubes are strongly saturated; wood/belt/robot are not
VAL_MIN = 0.08
MIN_BLOB_CELLS = 3
MAX_BLOB_CELLS = 600
MAX_CUBE_SPREAD = 0.09     # m, reject masks that bled into the belt/table
HOVER_DZ = 0.08            # m above cube top
DESCEND_TICKS = 10
CLOSE_TICKS = 12
LIFT_TICKS = 8
LIFT_DZ = 0.10
KEYFRAME_EVERY = 3
LAG_TICKS = 2              # lead the moving target by the controller's tracking lag
MAX_JOINT_STEP = 0.05      # rad per tick for interpolated transits
TRANSIT_DZ = 0.15          # m above bin rim
RELEASE_DZ = 0.03          # m clearance of the cube bottom above the rim
OPEN_TICKS = 6
WAIT_TICKS = 6
VEL_PAIR_MIN_DT = 0.15
HELD_MIN = 0.10            # normalized gripper reading above which a cube is held (0 closed)
HELD_MAX = 0.95
BUTTON_RESERVE_TICKS = 60
MAX_LOOPS = 400
HELD_NEAR = 0.06           # m, wrist blob centroid to end-effector distance for the held cube
HELD_BLOBS = 6             # wrist blobs examined for the held-cube colour check

NUMBER_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
COLOUR_WORDS = {"red": "red", "green": "green", "blue": "blue", "yellow": "yellow",
                "cyan": "cyan", "magenta": "magenta", "purple": "magenta", "pink": "magenta",
                "orange": "orange"}
CUBE_WORDS = ("cube", "cubes", "block", "blocks")
# hue windows in degrees (measured offline on all three recordings; colour classes are generic)
HUE_CLASSES = (("red", 345.0, 360.1), ("red", 0.0, 12.0), ("orange", 30.0, 50.0),
               ("yellow", 50.0, 75.0), ("green", 80.0, 150.0), ("cyan", 155.0, 205.0),
               ("blue", 210.0, 280.0), ("magenta", 280.0, 340.0))

report = {"version": "v00", "events": [], "deliveries": [], "errors": []}
STATE = {"q": None, "g": 1.0, "done": False, "ticks": 0, "terminal_reply": None, "bad_replies": 0}
VEL = {"sx": 0.0, "sy": 0.0, "st": 0.0, "pairs": 0}


def log(kind, info):
    if len(report["events"]) < 300:
        report["events"].append([kind, info])


# ---------------------------------------------------------------- goal parsing
def parse_goal(text):
    clean = ""
    for ch in str(text).lower():
        clean = clean + (ch if ch.isalnum() else " ")
    toks = clean.split()
    need = {}
    for i in range(len(toks)):
        colour = COLOUR_WORDS.get(toks[i])
        if colour is None:
            continue
        if i + 1 >= len(toks) or toks[i + 1] not in CUBE_WORDS:
            continue
        count = None
        if i > 0:
            prev = toks[i - 1]
            if prev in NUMBER_WORDS:
                count = NUMBER_WORDS[prev]
            elif prev.isdigit():
                count = int(prev)
        if count is None:
            count = 1
        need[colour] = need.get(colour, 0) + count
    return need


# ---------------------------------------------------------------- small math helpers
def quantile(values, frac):
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return ordered[int((len(ordered) - 1) * frac)]


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def quat_to_axes(q):
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    xa = (1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y))
    ya = (2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x))
    za = (2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y))
    return xa, ya, za


def axes_to_quat(xa, ya, za):
    m00, m01, m02 = xa[0], ya[0], za[0]
    m10, m11, m12 = xa[1], ya[1], za[1]
    m20, m21, m22 = xa[2], ya[2], za[2]
    tr = m00 + m11 + m22
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        return [0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s]
    if m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2
        return [(m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s]
    if m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2
        return [(m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s]
    s = math.sqrt(1.0 + m22 - m00 - m11) * 2
    return [(m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s]


def top_down_quat(current_q):
    xa, ya, za = quat_to_axes(current_q)
    if za[2] < -0.97:
        return [float(v) for v in current_q]
    hx, hy = xa[0], xa[1]
    norm = math.sqrt(hx * hx + hy * hy)
    if norm < 1e-6:
        hx, hy, norm = 1.0, 0.0, 1.0
    xn = (hx / norm, hy / norm, 0.0)
    zn = (0.0, 0.0, -1.0)
    yn = (zn[1] * xn[2] - zn[2] * xn[1], zn[2] * xn[0] - zn[0] * xn[2], zn[0] * xn[1] - zn[1] * xn[0])
    return axes_to_quat(xn, yn, zn)


# ---------------------------------------------------------------- robot interface
ctx = get_task_context()
GOAL = str(ctx.language)
NEED = parse_goal(GOAL)
report["goal"] = GOAL
report["need"] = dict(NEED)

META = robomme_plus.get_task_metadata()
MAX_STEPS = int(META["max_steps"])
HZ = 20.0
for key in ("control_hz", "control_frequency", "control_freq", "frequency_hz"):
    try:
        HZ = float(META[key])
        break
    except Exception:
        pass
DT = 1.0 / HZ
report["max_steps"] = MAX_STEPS
report["control_hz"] = HZ
try:
    report["conditioning_video_frames"] = len(robomme_plus.get_conditioning_video())
except Exception:
    report["conditioning_video_frames"] = "unavailable"

STATE0 = get_robot_state()
BASE = STATE0.base_frame
EMB = STATE0.embodiment
HOME_Q = [float(v) for v in STATE0.joint_positions["primary"]]
DOWN_Q = top_down_quat(STATE0.end_effector_poses["primary"].quaternion_wxyz)
T0 = float(STATE0.timestamp_s)
STATE["q"] = list(HOME_Q)
STATE["g"] = float(STATE0.gripper_positions["primary"])


def now():
    return float(get_robot_state().timestamp_s)


def ticks_used():
    return int(round((now() - T0) / DT))


def ticks_left():
    return MAX_STEPS - ticks_used()


def reply_flag(reply, name):
    try:
        if name == "terminated":
            return bool(reply.terminated)
        if name == "truncated":
            return bool(reply.truncated)
        if name == "terminal":
            return bool(reply.terminal)
        if name == "done":
            return bool(reply.done)
        if name == "failed":
            return reply.ok is False
    except Exception:
        return False
    return False


def is_terminal(reply):
    if reply is None:
        return False
    term = False
    for name in ("terminated", "truncated", "terminal", "done"):
        term = term or reply_flag(reply, name)
    if reply_flag(reply, "failed"):
        STATE["bad_replies"] = STATE["bad_replies"] + 1
        log("step_not_ok", {"tick": STATE["ticks"]})
    else:
        STATE["bad_replies"] = 0
    return term or STATE["bad_replies"] >= 2


def send(q, g):
    if STATE["done"]:
        return None
    reply = step(RobotAction(arms={"primary": ArmCommand(
        mode="joint_position", target=[float(v) for v in q],
        gripper_position=float(g), embodiment=EMB)}))
    STATE["q"] = [float(v) for v in q]
    STATE["g"] = float(g)
    STATE["ticks"] = STATE["ticks"] + 1
    if is_terminal(reply):
        STATE["done"] = True
        log("terminal", {"tick": ticks_used()})
    return reply


def hold(n, g=None):
    grip = STATE["g"] if g is None else g
    for tick in range(n):
        if STATE["done"]:
            return
        send(STATE["q"], grip)


def interp_to(q_target, g, max_step=MAX_JOINT_STEP, settle=2):
    start = STATE["q"]
    biggest = max(abs(float(a) - float(b)) for a, b in zip(q_target, start))
    n = max(1, int(math.ceil(biggest / max_step)))
    for i in range(1, n + 1):
        if STATE["done"]:
            return n
        frac = float(i) / n
        send([float(a) + (float(b) - float(a)) * frac for a, b in zip(start, q_target)], g)
    hold(settle, g)
    return n + settle


def transit_ticks(q_from, q_to):
    biggest = max(abs(float(a) - float(b)) for a, b in zip(q_from, q_to))
    return max(1, int(math.ceil(biggest / MAX_JOINT_STEP)))


def ik(pos, seed=None):
    pose = Pose([float(pos[0]), float(pos[1]), float(pos[2])], DOWN_Q, BASE)
    try:
        res = solve_ik(pose, seed_joints=seed)
    except Exception as exc:
        log("ik_exception", str(exc)[:120])
        return None
    if not res.ok:
        return None
    return [float(v) for v in res.joint_positions]


def ee_pos():
    p = get_robot_state().end_effector_poses["primary"].position
    return [float(p[0]), float(p[1]), float(p[2])]


def gripper_reading():
    return float(get_robot_state().gripper_positions["primary"])


def follow_path(waypoints, allow_prefix=False):
    """waypoints: list of (pos, gripper) per tick. IK on keyframes, joint-lerp between.
    allow_prefix: on the first unreachable keyframe, still execute the reachable prefix and
    return False (used for pressing, where going as deep as reachable is the right fallback)."""
    keys = []
    seed = STATE["q"]
    last = len(waypoints) - 1
    complete = True
    for i in range(len(waypoints)):
        if i % KEYFRAME_EVERY == KEYFRAME_EVERY - 1 or i == last:
            sol = ik(waypoints[i][0], seed)
            if sol is None:
                if not allow_prefix:
                    return False
                complete = False
                log("path_prefix", {"executed": i, "of": len(waypoints)})
                break
            keys.append((i, sol))
            seed = sol
    prev_i, prev_q = -1, STATE["q"]
    for key_i, key_q in keys:
        span = key_i - prev_i
        for j in range(1, span + 1):
            if STATE["done"]:
                return True
            frac = float(j) / span
            q = [float(a) + (float(b) - float(a)) * frac for a, b in zip(prev_q, key_q)]
            send(q, waypoints[prev_i + j][1])
        prev_i, prev_q = key_i, key_q
    return complete


# ---------------------------------------------------------------- perception
def rgb_class(r, g, b):
    mx = max(r, g, b)
    mn = min(r, g, b)
    if mx < 255.0 * VAL_MIN or mx <= 0:
        return None
    d = float(mx - mn)
    if d / mx < SAT_MIN:
        return None
    if mx == r:
        h = 60.0 * (((g - b) / d) % 6.0)
    elif mx == g:
        h = 60.0 * ((b - r) / d + 2.0)
    else:
        h = 60.0 * ((r - g) / d + 4.0)
    for name, lo, hi in HUE_CLASSES:
        if lo <= h < hi:
            return name
    return None


def colour_blobs(img, stride, wanted, max_cells=MAX_BLOB_CELLS):
    height = len(img)
    width = len(img[0])
    cells = {}
    for y in range(0, height, stride):
        row = img[y]
        for x in range(0, width, stride):
            px = row[x]
            c = rgb_class(int(px[0]), int(px[1]), int(px[2]))
            if c is not None and (wanted is None or c in wanted):
                cells[(y // stride, x // stride)] = c
    seen = set()
    blobs = []
    for start in cells:
        if start in seen:
            continue
        colour = cells[start]
        stack = [start]
        seen.add(start)
        members = []
        while stack:
            cell = stack.pop()
            members.append(cell)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    nb = (cell[0] + dy, cell[1] + dx)
                    if nb in cells and nb not in seen and cells[nb] == colour:
                        seen.add(nb)
                        stack.append(nb)
        if MIN_BLOB_CELLS <= len(members) <= max_cells:
            sy = sum(m[0] for m in members) * stride / float(len(members))
            sx = sum(m[1] for m in members) * stride / float(len(members))
            blobs.append({"colour": colour, "u": sx, "v": sy, "cells": len(members),
                          "width": width, "height": height})
    return blobs


def cloud_stats(points):
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    zs = [float(p[2]) for p in points]
    z_top = quantile(zs, 0.9)
    z_low = quantile(zs, 0.1)
    cut = z_low + 0.6 * (z_top - z_low)
    up_x = [xs[i] for i in range(len(zs)) if zs[i] >= cut]
    up_y = [ys[i] for i in range(len(zs)) if zs[i] >= cut]
    if len(up_x) < 3:
        up_x, up_y = xs, ys
    spread = max(quantile(xs, 0.9) - quantile(xs, 0.1), quantile(ys, 0.9) - quantile(ys, 0.1))
    return {"x": quantile(up_x, 0.5), "y": quantile(up_y, 0.5), "z_top": z_top, "z_low": z_low,
            "spread": spread, "n": len(zs),
            "x_lo": quantile(xs, 0.05), "x_hi": quantile(xs, 0.95),
            "y_lo": quantile(ys, 0.05), "y_hi": quantile(ys, 0.95)}


def as_box(raw):
    try:
        if raw is None:
            return None
        vals = [float(b) for b in raw]
        if len(vals) != 4:
            return None
        return vals
    except Exception:
        return None


def mask_box(mask):
    """Pixel box [x1, y1, x2, y2] of a 2-D mask array (stride 2), or None if not an array."""
    try:
        rows = len(mask)
        cols = len(mask[0])
        xs, ys = [], []
        for y in range(0, rows, 2):
            row = mask[y]
            for x in range(0, cols, 2):
                if row[x]:
                    xs.append(x)
                    ys.append(y)
    except Exception:
        return None
    if not xs:
        return None
    return [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]


def seg_list(seg, limit):
    """Normalize a SegmentationSet into [{"obj", "alt", "box", "score"}] (best first as returned)."""
    out = []
    try:
        if not seg.ok:
            return out
    except Exception:
        return out
    items = None
    try:
        items = list(seg.segmentations)
    except Exception:
        items = None
    if items is not None:
        for c in items[:limit]:
            box, score, alt = None, 0.0, None
            try:
                box = as_box(c.box_xyxy)
            except Exception:
                box = None
            try:
                score = float(c.score)
            except Exception:
                score = 0.0
            try:
                alt = c.mask
            except Exception:
                alt = None
            if box is None and alt is not None:
                box = mask_box(alt)
            out.append({"obj": c, "alt": alt, "box": box, "score": score})
        return out
    try:
        masks = list(seg.masks)
    except Exception:
        return out
    boxes, scores = [], []
    try:
        boxes = list(seg.boxes)
    except Exception:
        boxes = []
    try:
        scores = list(seg.scores)
    except Exception:
        scores = []
    for i in range(min(limit, len(masks))):
        box = as_box(boxes[i]) if i < len(boxes) else None
        score = 0.0
        if i < len(scores):
            try:
                score = float(scores[i])
            except Exception:
                score = 0.0
        if box is None:
            box = mask_box(masks[i])
        out.append({"obj": masks[i], "alt": None, "box": box, "score": score})
    return out


def mask_cloud(cand, camera):
    cloud = None
    for obj in (cand["obj"], cand["alt"]):
        if obj is None:
            continue
        try:
            cloud = mask_to_point_cloud(obj, camera, target_frame=BASE)
            break
        except Exception as exc:
            log("cloud_exception", str(exc)[:120])
            cloud = None
    if cloud is None:
        return None
    try:
        pts = cloud.points
        if len(pts) < 6:
            return None
        return pts
    except Exception:
        return None


def box_contains(cand, u, v, margin):
    box = cand["box"]
    if box is None:
        return False
    x1, y1, x2, y2 = box
    return x1 - margin <= u <= x2 + margin and y1 - margin <= v <= y2 + margin


def blob_to_3d(camera, blob, t):
    u, v = float(blob["u"]), float(blob["v"])
    cands = []
    for prompt_pt in ([u, v], [v, u]):
        try:
            found = seg_list(segment_points(camera, [prompt_pt]), 3)
        except Exception as exc:
            log("segment_points_exception", str(exc)[:120])
            found = []
        # keep masks whose box contains the prompted pixel; an unboxed mask is accepted only
        # for the documented (x, y) order
        cands = [c for c in found if box_contains(c, u, v, 2.0) or (c["box"] is None and prompt_pt[0] == u)]
        if cands or u == v:
            break
    tried_text = False
    while True:
        for cand in cands:
            if tried_text and not box_contains(cand, u, v, 2.0):
                continue
            pts = mask_cloud(cand, camera)
            if pts is None:
                continue
            st = cloud_stats(pts)
            if st["spread"] > MAX_CUBE_SPREAD:
                continue
            st["colour"] = blob["colour"]
            st["t"] = t
            st["camera"] = camera
            st["u"] = u
            st["v"] = v
            return st
        if tried_text:
            return None
        tried_text = True
        cands = []
        try:
            cands = seg_list(segment_text(camera, blob["colour"] + " cube"), 10)
        except Exception:
            cands = []


def observe(camera, stride, wanted):
    obs = get_observation()
    img = obs.cameras[camera].rgb
    t = float(obs.timestamp_s)
    return colour_blobs(img, stride, wanted), t, img


def velocity():
    if VEL["st"] <= 0.0:
        return 0.0, 0.0
    return VEL["sx"] / VEL["st"], VEL["sy"] / VEL["st"]


def update_velocity(prev_dets, new_dets):
    vx, vy = velocity()
    established = VEL["st"] >= 0.4
    used = set()
    for nd in new_dets:
        best = None
        best_d = None
        for k in range(len(prev_dets)):
            pd = prev_dets[k]
            if k in used or pd["colour"] != nd["colour"] or pd["camera"] != nd["camera"]:
                continue
            dt = nd["t"] - pd["t"]
            if dt < VEL_PAIR_MIN_DT or dt > 3.0:
                continue
            px = pd["x"] + vx * dt
            py = pd["y"] + vy * dt
            d = math.sqrt((nd["x"] - px) ** 2 + (nd["y"] - py) ** 2)
            tol = (0.02 + 0.02 * dt) if established else (0.025 + 0.08 * dt)
            if d <= tol and (best_d is None or d < best_d):
                best, best_d = k, d
        if best is not None:
            pd = prev_dets[best]
            dt = nd["t"] - pd["t"]
            mx, my = nd["x"] - pd["x"], nd["y"] - pd["y"]
            if math.sqrt(mx * mx + my * my) / dt < 0.3:
                used.add(best)
                VEL["sx"] = VEL["sx"] + mx
                VEL["sy"] = VEL["sy"] + my
                VEL["st"] = VEL["st"] + dt
                VEL["pairs"] = VEL["pairs"] + 1


def predict(det, t):
    vx, vy = velocity()
    dt = t - det["t"]
    return det["x"] + vx * dt, det["y"] + vy * dt


# ---------------------------------------------------------------- fixtures
def centre_darkness(img, cand):
    if cand["box"] is None:
        return 1.0
    x1, y1, x2, y2 = cand["box"]
    height = len(img)
    width = len(img[0])

    def mean_v(xa, ya, xb, yb):
        total, count = 0.0, 0
        for yy in range(max(0, int(ya)), min(height, int(yb) + 1)):
            for xx in range(max(0, int(xa)), min(width, int(xb) + 1)):
                px = img[yy][xx]
                total = total + max(int(px[0]), int(px[1]), int(px[2]))
                count = count + 1
        return total / count if count else 0.0

    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    hw, hh = (x2 - x1) * 0.18, (y2 - y1) * 0.18
    inner = mean_v(cx - hw, cy - hh, cx + hw, cy + hh)
    outer = mean_v(x1, y1, x2, y2)
    return inner / outer if outer > 0 else 1.0


def find_bin(img):
    for prompt in ("bin", "open box", "box", "square container"):
        try:
            seg = segment_text("front", prompt)
        except Exception:
            continue
        for cand in seg_list(seg, 5):
            if centre_darkness(img, cand) > 0.6:
                continue
            pts = mask_cloud(cand, "front")
            if pts is None:
                continue
            st = cloud_stats(pts)
            if st["spread"] > 0.3:
                continue
            st["cx"] = (st["x_lo"] + st["x_hi"]) / 2.0
            st["cy"] = (st["y_lo"] + st["y_hi"]) / 2.0
            st["rim"] = quantile([float(p[2]) for p in pts], 0.95)
            st["prompt"] = prompt
            st["box"] = cand["box"]
            return st
    return None


def find_button(img, bin_info):
    best = None
    for prompt in ("button", "push button", "round button", "white button"):
        try:
            seg = segment_text("front", prompt)
        except Exception:
            continue
        for cand in seg_list(seg, 5):
            if bin_info is not None and bin_info["box"] is not None and cand["box"] is not None:
                bx = bin_info["box"]
                x1, y1, x2, y2 = cand["box"]
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                if bx[0] <= cx <= bx[2] and bx[1] <= cy <= bx[3]:
                    continue
            if centre_darkness(img, cand) < 0.6:
                continue
            pts = mask_cloud(cand, "front")
            if pts is None:
                continue
            st = cloud_stats(pts)
            if st["spread"] > 0.15 or st["z_top"] - st["z_low"] > 0.06:
                continue
            cx = (st["x_lo"] + st["x_hi"]) / 2.0
            cy = (st["y_lo"] + st["y_hi"]) / 2.0
            if math.sqrt(cx * cx + cy * cy) < 0.12:
                continue   # robot pedestal sits at the base-frame origin
            if bin_info is not None and st["z_top"] > bin_info["rim"] + 0.06:
                continue
            score = cand["score"]
            if best is None or score > best["score"]:
                st["cx"], st["cy"], st["score"], st["prompt"] = cx, cy, score, prompt
                best = st
        if best is not None:
            return best
    return best


# ---------------------------------------------------------------- task logic
def remaining():
    out = {}
    for colour in NEED:
        left = NEED[colour] - COUNTS[colour]
        if left > 0:
            out[colour] = left
    return out


def detect_needed(wanted, prev_dets):
    blobs, t, img = observe("front", SCAN_STRIDE, None)
    needed = [b for b in blobs if b["colour"] in wanted]
    others = [b for b in blobs if b["colour"] not in wanted]
    needed = sorted(needed, key=blob_size, reverse=True)[:4]
    dets = []
    for b in needed:
        d = blob_to_3d("front", b, t)
        if d is not None:
            dets.append(d)
    if VEL["st"] < 1.0:
        for b in sorted(others, key=blob_size, reverse=True)[:2]:
            d = blob_to_3d("front", b, t)
            if d is not None:
                d["aux"] = True
                dets.append(d)
    update_velocity(prev_dets, dets)
    return dets, t, len(blobs)


def blob_size(b):
    return b["cells"]


def plan_intercept(det, t_now):
    cube_h = clamp(det["z_top"] - det["z_low"], 0.02, 0.06)
    hover_z = det["z_top"] + HOVER_DZ
    grasp_z = det["z_top"] - 0.5 * cube_h
    n = int(round(1.0 / DT))
    qh = None
    for it in range(3):
        px, py = predict(det, t_now + (n + LAG_TICKS) * DT)
        qh = ik((px, py, hover_z), STATE["q"])
        if qh is None:
            return None
        n = transit_ticks(STATE["q"], qh) + 1
    follow_ticks = DESCEND_TICKS + CLOSE_TICKS
    gx, gy = predict(det, t_now + (n + follow_ticks + LAG_TICKS) * DT)
    if ik((gx, gy, grasp_z), qh) is None:
        return None
    return {"q_hover": qh, "ticks": n, "hover_z": hover_z, "grasp_z": grasp_z, "cube_h": cube_h,
            "hover_xy": predict(det, t_now + (n + LAG_TICKS) * DT)}


def wrist_refine(det, plan):
    blobs, t, img = observe("wrist", WRIST_STRIDE, set([det["colour"]]))
    blobs = sorted(blobs, key=blob_size, reverse=True)[:2]
    px, py = predict(det, t)
    best = None
    best_d = None
    for b in blobs:
        w = blob_to_3d("wrist", b, t)
        if w is None:
            continue
        d = math.sqrt((w["x"] - px) ** 2 + (w["y"] - py) ** 2)
        if d < 0.05 and (best_d is None or d < best_d):
            best, best_d = w, d
    if best is None:
        return det, None
    refined = {"x": best["x"], "y": best["y"], "t": t, "colour": det["colour"],
               "z_top": best["z_top"], "z_low": best["z_top"] - plan["cube_h"], "camera": "wrist"}
    return refined, best_d


def held_colour():
    """Colours of saturated wrist blobs whose 3-D centroid is near the end effector.
    Returns (nearest colour or None, its distance or None, set of all near colours). The held
    cube fills a large part of the wrist view, so the scene-scan blob-size cap is lifted here."""
    obs = get_observation()
    img = obs.cameras["wrist"].rgb
    t = float(obs.timestamp_s)
    blobs = colour_blobs(img, WRIST_STRIDE, None, len(img) * len(img[0]))
    blobs = sorted(blobs, key=blob_size, reverse=True)[:HELD_BLOBS]
    ee = ee_pos()
    best = None
    best_d = None
    near = set()
    for b in blobs:
        w = blob_to_3d("wrist", b, t)
        if w is None:
            continue
        d = math.sqrt((w["x"] - ee[0]) ** 2 + (w["y"] - ee[1]) ** 2 + (0.5 * (w["z_top"] + w["z_low"]) - ee[2]) ** 2)
        if d < HELD_NEAR:
            near.add(b["colour"])
            if best_d is None or d < best_d:
                best, best_d = b["colour"], d
    return best, best_d, near


def pick(det, plan):
    """Transit to the predicted hover point, refine, follow-descend-close-lift. Returns held flag."""
    interp_to(plan["q_hover"], 1.0, settle=0)
    if STATE["done"]:
        return False
    anchor, shift = wrist_refine(det, plan)
    grasp_z = plan["grasp_z"]
    if shift is not None:
        grasp_z = anchor["z_top"] - 0.5 * plan["cube_h"]
    log("refine", {"shift": shift, "grasp_z": round(grasp_z, 4)})
    t_start = now()
    start = ee_pos()
    waypoints = []
    total = DESCEND_TICKS + CLOSE_TICKS + LIFT_TICKS
    for i in range(1, total + 1):
        t_i = t_start + (i + LAG_TICKS) * DT
        cx, cy = predict(anchor, t_i)
        if i <= DESCEND_TICKS:
            frac = float(i) / DESCEND_TICKS
            z = start[2] + (grasp_z - start[2]) * frac
            g = 1.0
        elif i <= DESCEND_TICKS + CLOSE_TICKS:
            z = grasp_z
            g = 0.0
        else:
            frac = float(i - DESCEND_TICKS - CLOSE_TICKS) / LIFT_TICKS
            z = grasp_z + LIFT_DZ * frac
            g = 0.0
        waypoints.append(((cx, cy, z), g))
    if not follow_path(waypoints):
        log("pick_ik_failed", det["colour"])
        return False
    reading = gripper_reading()
    held = HELD_MIN < reading < HELD_MAX
    seen, dist, near = (None, None, set())
    if held and not STATE["done"]:
        seen, dist, near = held_colour()
    log("grasp_check", {"colour": det["colour"], "reading": round(reading, 3), "held": held,
                        "wrist_colour": seen, "wrist_d": None if dist is None else round(dist, 3),
                        "near": sorted(near)})
    if not held:
        return False, None
    if seen is None or det["colour"] in near:
        return True, det["colour"]      # confirmed, or unverifiable: trust the targeted colour
    return True, seen


def deliver(colour, cube_h):
    pos = ee_pos()
    transit_z = max(BIN["rim"] + TRANSIT_DZ, pos[2])
    q_up = ik((pos[0], pos[1], transit_z), STATE["q"])
    if q_up is not None:
        interp_to(q_up, 0.0, settle=0)
    q_bin = ik((BIN["cx"], BIN["cy"], transit_z), STATE["q"])
    if q_bin is None:
        log("bin_ik_failed", colour)
        return False
    interp_to(q_bin, 0.0, settle=1)
    release_z = BIN["rim"] + 0.5 * cube_h + RELEASE_DZ
    q_rel = ik((BIN["cx"], BIN["cy"], release_z), STATE["q"])
    if q_rel is not None:
        interp_to(q_rel, 0.0, max_step=0.03, settle=2)
    reading = gripper_reading()
    still_held = HELD_MIN < reading < HELD_MAX
    hold(OPEN_TICKS, 1.0)
    interp_to(q_bin, 1.0, settle=0)
    log("release", {"colour": colour, "reading": round(reading, 3), "held_at_release": still_held})
    return still_held


def drop_and_recover():
    pos = ee_pos()
    hold(OPEN_TICKS, 1.0)
    q_up = ik((pos[0], pos[1], pos[2] + 0.06), STATE["q"])
    if q_up is not None:
        interp_to(q_up, 1.0, settle=0)


def press_button():
    if BUTTON is None:
        report["errors"].append("button not localized")
        return False
    bx, by, top = BUTTON["cx"], BUTTON["cy"], BUTTON["z_top"]
    q_hover = ik((bx, by, top + 0.08), STATE["q"])
    if q_hover is None:
        report["errors"].append("button hover IK failed")
        return False
    interp_to(q_hover, 0.0, settle=4)
    waypoints = []
    for i in range(1, 9):
        waypoints.append(((bx, by, top + 0.08 - 0.07 * i / 8.0), 0.0))
    for i in range(1, 21):
        waypoints.append(((bx, by, top + 0.01 - 0.0025 * i), 0.0))
    follow_path(waypoints, True)
    hold(6, 0.0)
    if not STATE["done"]:
        log("button_retry", ticks_used())
        waypoints = [((bx, by, top - 0.04 - 0.002 * i), 0.0) for i in range(1, 6)]
        follow_path(waypoints, True)
        hold(6, 0.0)
    return STATE["done"]


# ---------------------------------------------------------------- main
COUNTS = {}
for colour_name in NEED:
    COUNTS[colour_name] = 0

hold(4, 1.0)
first_obs = get_observation()
FRONT_IMG = first_obs.cameras["front"].rgb
BIN = find_bin(FRONT_IMG)
BUTTON = find_button(FRONT_IMG, BIN)
report["bin"] = None if BIN is None else {"cx": round(BIN["cx"], 4), "cy": round(BIN["cy"], 4),
                                          "rim": round(BIN["rim"], 4), "prompt": BIN["prompt"]}
report["button"] = None if BUTTON is None else {"cx": round(BUTTON["cx"], 4),
                                                "cy": round(BUTTON["cy"], 4),
                                                "top": round(BUTTON["z_top"], 4),
                                                "prompt": BUTTON["prompt"]}

if not NEED:
    report["errors"].append("goal parse found no cube counts")
if BIN is None:
    report["errors"].append("bin not localized")

prev_dets = []
empty_streak = 0
loops = 0
attempts = 0
while BIN is not None and remaining() and not STATE["done"] and loops < MAX_LOOPS:
    loops = loops + 1
    if ticks_left() < BUTTON_RESERVE_TICKS:
        log("budget_stop", ticks_used())
        break
    wanted = remaining()
    dets, t_obs, n_blobs = detect_needed(wanted, prev_dets)
    prev_dets = dets
    targets = [d for d in dets if d["colour"] in wanted and not d.get("aux", False)]
    if not targets:
        empty_streak = empty_streak + 1
        if empty_streak >= 3 and max(abs(a - b) for a, b in zip(STATE["q"], HOME_Q)) > 0.05:
            interp_to(HOME_Q, 1.0, settle=1)
        else:
            hold(WAIT_TICKS, 1.0)
        continue
    empty_streak = 0
    if VEL["st"] < 0.5:
        hold(2 * WAIT_TICKS, 1.0)   # re-look before the first intercept: belt velocity
        continue
    best = None
    for det in targets[:3]:
        plan = plan_intercept(det, t_obs)
        if plan is not None and (best is None or plan["ticks"] < best[1]["ticks"]):
            best = (det, plan)
    if best is None:
        hold(WAIT_TICKS, 1.0)
        continue
    det, plan = best
    attempts = attempts + 1
    vx, vy = velocity()
    log("intercept", {"colour": det["colour"], "ticks": plan["ticks"], "v": [round(vx, 4), round(vy, 4)],
                      "tick": ticks_used()})
    held, held_as = pick(det, plan)
    if STATE["done"]:
        break
    if held and held_as not in remaining():
        log("wrong_colour_held", {"target": det["colour"], "seen": held_as})
        report["errors"].append("dropped wrong-colour cube " + str(held_as))
        held = False
    if held:
        delivered = deliver(held_as, plan["cube_h"])
        if delivered:
            COUNTS[held_as] = COUNTS[held_as] + 1
            report["deliveries"].append([held_as, ticks_used()])
    else:
        drop_and_recover()
    prev_dets = []

report["counts"] = dict(COUNTS)
report["attempts"] = attempts
vx_final, vy_final = velocity()
report["belt_velocity"] = [round(vx_final, 4), round(vy_final, 4), VEL["pairs"]]
if not STATE["done"] and not remaining():
    report["button_pressed"] = press_button()
report["terminal"] = STATE["done"]
report["ticks_used"] = ticks_used()
result = report
