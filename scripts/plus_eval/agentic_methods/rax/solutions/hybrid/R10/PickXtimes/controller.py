"""PickXtimes controller: pick green cube, place on target N times, press button."""
import json, math, os, re, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from perception import parse_reply, backproject, masks, largest_blob, blob_stats, yaw_of_square, hsv, fit_target

LOG = []
OUT = "/workspace/output"

NUM_WORDS = {"one": 1, "once": 1, "two": 2, "twice": 2, "three": 3, "thrice": 3, "four": 4, "five": 5,
             "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}

HOVER_Z = 0.22
CUBE_COLOR = "green"
CARRY_Z = 0.14
SPEED = 0.6          # max action magnitude for translation (x0.05 m per step)


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


class Ended(Exception):
    pass


def parse_count(instr):
    s = instr.lower()
    m = re.search(r"repeating this pick-and-place action (\w+) times", s)
    words = [m.group(1)] if m else []
    words += re.findall(r"(\w+) times", s)
    for w in words:
        if w.isdigit():
            return int(w)
        if w in NUM_WORDS:
            return NUM_WORDS[w]
    for w, n in NUM_WORDS.items():
        if re.search(r"\b%s\b" % w, s):
            return n
    return 1


class Robot:
    def __init__(self, ctx):
        self.ctx = ctx
        self.meta = None
        self.imgs = None
        self.state = None
        self.ended = False
        self.complete = False
        self.step_id = 0
        self.remaining = None

    def observe(self):
        rep = self.ctx.call("robomme_observe")
        self.meta, self.imgs = parse_reply(rep)
        self.state = np.array(self.meta["state"], float)
        self.step_id = self.meta.get("step_id", self.step_id)
        self._limits(self.meta)
        return self.meta

    def _limits(self, meta):
        lim = meta.get("episode_limits") or {}
        if "remaining_episode_steps" in lim:
            self.remaining = lim["remaining_episode_steps"]
        if meta.get("episode_ended"):
            self.ended = True
        tr = meta.get("transition") or {}
        if isinstance(tr, dict):
            if tr.get("task_complete"):
                self.complete = True
            if tr.get("episode_ended"):
                self.ended = True
        for st in (meta.get("steps", []) or []) + ((tr.get("steps") or []) if isinstance(tr, dict) else []):
            if isinstance(st, dict):
                if st.get("task_complete"):
                    self.complete = True
                if st.get("episode_ended"):
                    self.ended = True
        if meta.get("task_complete"):
            self.complete = True

    def step(self, rows):
        if self.ended:
            raise Ended()
        rows = [[float(np.clip(v, -1, 1)) for v in r] for r in rows]
        if self.remaining is not None:
            rows = rows[:max(0, self.remaining)]
            if not rows:
                self.ended = True
                raise Ended()
        rep = self.ctx.call("robomme_step", actions=rows)
        meta, imgs = parse_reply(rep)
        self._limits(meta)
        if "state" in meta:
            self.state = np.array(meta["state"], float)
            self.meta, self.imgs = meta, imgs
        self.step_id = meta.get("step_id", self.step_id)
        if self.ended:
            log("episode ended; complete =", self.complete, "step", self.step_id)
            raise Ended()
        return meta

    # ---- orientation helpers ----
    def yaw(self):
        rv = self.state[3:6]
        th = np.linalg.norm(rv)
        if th < 1e-9:
            R = np.eye(3)
        else:
            k = rv / th
            K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
            R = np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K
        return math.atan2(R[1, 0], R[0, 0])

    def grip(self):
        return float(self.state[6] + self.state[7])

    def move_to(self, target, yaw=None, g=-1.0, tol=0.004, max_steps=120, speed=SPEED, chunk=6):
        target = np.array(target, float)
        used = 0
        while used < max_steps:
            d = target - self.state[:3]
            dyaw = 0.0
            if yaw is not None:
                dyaw = (yaw - self.yaw() + math.pi) % (2 * math.pi) - math.pi
            if np.linalg.norm(d) < tol and abs(dyaw) < 0.03:
                return True
            n = int(min(chunk, max(1, math.ceil(np.linalg.norm(d) / (0.05 * speed)))))
            rows = []
            for i in range(n):
                a = d / 0.05
                mag = np.max(np.abs(a))
                if mag > speed:
                    a = a * speed / mag
                # remaining displacement shrinks each step (controller targets measured+delta)
                rz = float(np.clip(dyaw / 0.5, -0.5, 0.5))
                rows.append([a[0], a[1], a[2], 0.0, 0.0, rz, g])
                d = d - a * 0.05
                dyaw -= rz * 0.5
                if np.linalg.norm(d) < tol * 0.5 and abs(dyaw) < 0.01:
                    break
            self.step(rows)
            used += len(rows)
        log("move_to timeout", target.round(3), "at", self.state[:3].round(3))
        return False

    def hold_track(self, vel, g, n):
        a = np.clip(np.asarray(vel) / 0.05, -0.9, 0.9)
        self.step([[a[0], a[1], 0, 0, 0, 0, g]] * n)

    def hold(self, g, n):
        self.step([[0, 0, 0, 0, 0, 0, g]] * n)


# ---------------- perception ----------------

def purple_points_other(rb, cam):
    """Purple table points from the other camera of the same observation (world xy)."""
    try:
        oc = "front" if cam == "wrist" else "wrist"
        if (oc, "rgb") not in rb.imgs or (oc, "depth") not in rb.imgs:
            return None
        P = backproject(rb.meta, rb.imgs[(oc, "depth")], oc)
        m = masks(rb.imgs[(oc, "rgb")])
        pts = P[m["purple"] & (P[..., 2] < 0.03)]
        return pts[np.all(np.isfinite(pts), 1)][:, :2]
    except Exception as e:
        log("purple_points_other failed", e)
        return None


def find_objects(rb, cam="wrist"):
    meta, imgs = rb.meta, rb.imgs
    rgb = imgs[(cam, "rgb")]
    P = backproject(meta, imgs[(cam, "depth")], cam)
    m = masks(rgb)
    z = P[..., 2]
    out = {}
    g = largest_blob(m.get(CUBE_COLOR, m["green"]) & (z < 0.2))
    st = blob_stats(g, P)
    if st and st["n"] >= 15:
        pts = st["pts"]
        top = np.percentile(pts[:, 2], 95)
        topf = pts[pts[:, 2] > top - 0.006]
        c = np.median(topf, 0)
        out["cube"] = {"xy": c[:2], "top": float(top), "yaw": yaw_of_square(topf[:, :2]) if len(topf) > 15 else 0.0,
                       "n": st["n"]}
    p = largest_blob(m["purple"] & (z < 0.03))
    st = blob_stats(p, P)
    if st and st["n"] >= 10:
        pts = st["pts"]
        rough = np.median(pts[:, :2], 0)
        allp = P[m["purple"] & (z < 0.03)]
        allp = allp[np.all(np.isfinite(allp), 1)][:, :2]
        extra = purple_points_other(rb, cam)
        if extra is not None and len(extra):
            allp = np.concatenate([allp, extra], 0)
        c, q = fit_target(allp, rough)
        out["target"] = {"xy": c, "z": float(np.median(pts[:, 2])), "n": st["n"], "q": q,
                         "rough": rough}
    X = P[..., 0]; Y = P[..., 1]
    elev = (z > 0.006) & (z < 0.06) & (m["sat"] < 0.25) & (X > -0.45) & (X < 0.25) & (np.abs(Y) < 0.45)
    b = largest_blob(elev)
    st = blob_stats(b, P)
    if st and st["n"] >= 30:
        pts = st["pts"]
        zt = np.percentile(pts[:, 2], 97)
        cap = pts[pts[:, 2] > zt - 0.006]
        out["button"] = {"xy": np.median(cap[:, :2], 0), "top": float(zt), "n": st["n"],
                         "base_xy": np.median(pts[:, :2], 0)}
    return out


OVERVIEW = (-0.06, 0.0, 0.50)


def overview(rb):
    """Rise to the start-like overview pose and detect everything (wrist, front fallback)."""
    rb.move_to([rb.state[0], rb.state[1], 0.30], g=-1.0, tol=0.01, speed=0.8)
    rb.move_to(list(OVERVIEW), yaw=0.0, g=-1.0, tol=0.01, speed=0.8)
    rb.hold(-1.0, 2)
    rb.observe()
    o = find_objects(rb)
    of = find_objects(rb, "front")
    for k in ("cube", "target"):
        if k not in o and k in of:
            o[k] = of[k]
            log("overview: using front for", k)
    log("overview:", {k: np.round(v["xy"], 3).tolist() for k, v in o.items()},
        "tq", round(o["target"].get("q", 0), 2) if "target" in o else None)
    return o


def save_frame(rb, name):
    try:
        from PIL import Image
        os.makedirs(OUT, exist_ok=True)
        for cam in ("front", "wrist"):
            if (cam, "rgb") in rb.imgs:
                Image.fromarray(rb.imgs[(cam, "rgb")]).save(f"{OUT}/{name}_{cam}.png")
    except Exception as e:
        log("save_frame failed", e)


def nearest_yaw(cur, cube_yaw):
    best = None
    for k in range(-4, 5):
        y = cube_yaw + k * math.pi / 2
        d = abs((y - cur + math.pi) % (2 * math.pi) - math.pi)
        if best is None or d < best[0]:
            best = (d, y)
    y = best[1]
    return (y + math.pi) % (2 * math.pi) - math.pi


# ---------------- task phases ----------------

def locate_cube(rb, guess_xy, z=HOVER_Z):
    """Hover above guess, observe wrist, refine. Returns cube dict or None."""
    for it in range(3):
        rb.move_to([guess_xy[0], guess_xy[1], z], g=-1.0, tol=0.006)
        rb.observe()
        obj = find_objects(rb)
        if "cube" not in obj:
            obj_f = find_objects(rb, "front")
            if "cube" in obj_f:
                log("cube from front", obj_f["cube"]["xy"].round(3))
                guess_xy = obj_f["cube"]["xy"]
                z = HOVER_Z + 0.08
                continue
            log("cube not visible")
            return None
        c = obj["cube"]
        if np.linalg.norm(c["xy"] - np.array(guess_xy)) < 0.01:
            return c
        guess_xy = c["xy"]
    return c


def pick(rb, cube):
    yaw = nearest_yaw(rb.yaw(), cube["yaw"])
    xy = cube["xy"]
    rb.move_to([xy[0], xy[1], HOVER_Z - 0.08], yaw=yaw, g=-1.0, tol=0.005)
    # refine at lower height
    rb.observe()
    obj = find_objects(rb)
    if "cube" in obj and np.linalg.norm(obj["cube"]["xy"] - xy) < 0.05:
        cube = obj["cube"]; xy = cube["xy"]
        yaw = nearest_yaw(rb.yaw(), cube["yaw"])
        rb.move_to([xy[0], xy[1], HOVER_Z - 0.08], yaw=yaw, g=-1.0, tol=0.004)
    gz = max(0.012, cube["top"] - 0.022)
    rb.move_to([xy[0], xy[1], gz + 0.04], yaw=yaw, g=-1.0, tol=0.004, speed=0.4)
    rb.move_to([xy[0], xy[1], gz], yaw=yaw, g=-1.0, tol=0.003, speed=0.25)
    rb.hold(1.0, 12)
    rb.move_to([xy[0], xy[1], CARRY_Z], g=1.0, tol=0.006, speed=0.4)
    rb.observe()
    gr = rb.grip()
    log("after lift grip=%.4f z=%.3f" % (gr, rb.state[2]))
    return gr > 0.01


def place(rb, target_xy, target_z=0.0):
    rb.move_to([target_xy[0], target_xy[1], CARRY_Z], g=1.0, tol=0.004)
    rb.move_to([target_xy[0], target_xy[1], target_z + 0.028], g=1.0, tol=0.003, speed=0.3)
    rb.hold(-1.0, 8)
    rb.move_to([target_xy[0], target_xy[1], CARRY_Z], g=-1.0, tol=0.006, speed=0.4)


def target_estimate(rb):
    """Target xy from the current observation, front camera blob + merged purple points."""
    rb.observe()
    o = find_objects(rb, "front")
    if "target" not in o:
        o = find_objects(rb, "wrist")
    if "target" in o:
        return o["target"]["xy"]
    return None


def place_tracking(rb, target_xy, target_z=0.008, max_iter=80):
    """Carry the held cube over the (possibly moving) target, follow it, descend, release."""
    place_z = target_z + 0.030
    hist = []
    xy = np.array(target_xy, float)
    vel = np.zeros(2)
    L = 1.0  # estimated fraction of commanded xy displacement actually achieved
    for it in range(max_iter):
        est = target_estimate(rb)
        s_now = rb.step_id
        if est is not None:
            hist.append((s_now, est.copy()))
            xy = est
        else:
            xy = xy + vel * 3
        if len(hist) >= 2:
            (s0, p0) = hist[-3] if len(hist) >= 3 else hist[-2]
            (s1, p1) = hist[-1]
            if s1 > s0 and s1 - s0 <= 12:
                v = (p1 - p0) / (s1 - s0)
                if np.linalg.norm(v) < 0.025:
                    vel = v
        pos = rb.state[:3]
        # target position at the time the next rows execute
        err = (xy + vel * 1.0) - pos[:2]
        e = np.linalg.norm(err)
        z = pos[2]
        if z <= place_z + 0.006 and e < 0.010:
            rb.hold_track(vel, -1.0, 4)
            log("released at err %.4f z %.3f vel/step %s L %.2f it %d" % (e, z, np.round(vel, 4), L, it))
            return True
        if e < 0.015:
            zg = max(place_z, z - 0.03)
        elif e < 0.03:
            zg = max(place_z + 0.02, min(z, CARRY_Z))
        else:
            zg = max(z, 0.10)
        rows = []
        p = pos.copy()
        cmd = np.zeros(2)
        for k in range(3):
            gk = xy + vel * (k + 1)
            d = np.array([gk[0] - p[0], gk[1] - p[1], zg - p[2]])
            a = d / 0.05
            a[:2] = a[:2] / L
            mag = np.max(np.abs(a[:2]))
            if mag > 0.9:
                a[:2] = a[:2] * 0.9 / mag
            a[2] = float(np.clip(a[2], -0.4, 0.4))
            rows.append([a[0], a[1], a[2], 0, 0, 0, 1.0])
            p = p + a * 0.05 * np.array([L, L, 1.0])
            cmd += a[:2] * 0.05
        p_before = pos[:2].copy()
        rb.step(rows)
        if np.linalg.norm(cmd) > 0.01:
            act = rb.state[:2] - p_before
            l_new = float(np.dot(act, cmd) / np.dot(cmd, cmd))
            L = float(np.clip(0.6 * L + 0.4 * l_new, 0.4, 1.0))
    log("place_tracking timeout; releasing at current pose")
    rb.hold(-1.0, 6)
    return False


def press_button(rb, button):
    xy = button["xy"]
    rb.move_to([xy[0], xy[1], HOVER_Z - 0.06], g=1.0, tol=0.006)
    rb.observe()
    obj = find_objects(rb)
    if "button" in obj and np.linalg.norm(obj["button"]["xy"] - xy) < 0.04:
        button = obj["button"]; xy = button["xy"]
        log("button refined", xy.round(3), round(button["top"], 3))
    rb.move_to([xy[0], xy[1], button["top"] + 0.05], g=1.0, tol=0.004)
    for dz in (0.0, -0.01, -0.02, -0.03):
        rb.move_to([xy[0], xy[1], button["top"] - 0.012 + dz], g=1.0, tol=0.003, speed=0.2, max_steps=40)
        rb.hold(1.0, 5)
        rb.move_to([xy[0], xy[1], button["top"] + 0.05], g=1.0, tol=0.006, speed=0.4)
        if rb.ended:
            return
        log("press attempt done dz", dz)
        if rb.complete:
            return


def main(ctx):
    t0 = time.time()
    rb = Robot(ctx)
    try:
        rb.observe()
        instr = rb.meta.get("instruction", "")
        N = parse_count(instr)
        global CUBE_COLOR
        for col in ("green", "red", "blue", "yellow"):
            if re.search(r"\b%s\b" % col, instr.lower()):
                CUBE_COLOR = col
                break
        log("cube colour", CUBE_COLOR)
        log("instruction:", instr, "| N =", N, "| remaining", rb.remaining)
        save_frame(rb, "start")
        objs = find_objects(rb)
        objs_f = find_objects(rb, "front")
        for k in ("cube", "target", "button"):
            if k not in objs and k in objs_f:
                objs[k] = objs_f[k]
                log("using front for", k)
        log("objects:", {k: {kk: (np.round(vv, 3).tolist() if isinstance(vv, np.ndarray) else vv) for kk, vv in v.items()} for k, v in objs.items()})
        target = objs.get("target")
        button = objs.get("button")
        cube_xy = objs["cube"]["xy"] if "cube" in objs else (target["xy"] if target else rb.state[:2])
        done = 0
        attempts = 0
        while done < N and attempts < N + 6:
            attempts += 1
            if attempts > 1:
                o = overview(rb)
                if "target" in o:
                    if target is not None:
                        moved = np.linalg.norm(o["target"]["xy"] - target["xy"])
                        log("target moved by %.3f" % moved)
                        if moved < 0.008 and "cube" in o and np.linalg.norm(o["cube"]["xy"] - o["target"]["xy"]) > 0.025 and done > 0:
                            done -= 1
                            log("placement apparently not registered (target static, cube off it); redo -> done=%d" % done)
                    target = o["target"]
                else:
                    log("WARNING target not found in overview; keeping old")
                if "cube" in o:
                    cube_xy = o["cube"]["xy"]
                if "button" in o and (button is None or np.linalg.norm(o["button"]["xy"] - button["xy"]) < 0.04):
                    button = o["button"]
            if target is None:
                log("no target known; looking again")
                continue
            cube = locate_cube(rb, cube_xy)
            if cube is None:
                continue
            ok = pick(rb, cube)
            if not ok:
                log("grasp failed, retry")
                rb.move_to([rb.state[0], rb.state[1], HOVER_Z], g=-1.0)
                cube_xy = cube["xy"]
                continue
            rb.move_to([target["xy"][0], target["xy"][1], CARRY_Z], g=1.0, tol=0.01)
            place_tracking(rb, target["xy"], target.get("z", 0.008))
            rb.move_to([rb.state[0], rb.state[1], CARRY_Z], g=-1.0, tol=0.008, speed=0.4)
            done += 1
            cube_xy = target["xy"]
            log("placed %d/%d at %s step %s t=%.0fs" % (done, N, np.round(target["xy"], 3), rb.step_id, time.time() - t0))
            if done == 1:
                save_frame(rb, "placed1")
        o = overview(rb)
        if "button" in o and (button is None or np.linalg.norm(o["button"]["xy"] - button["xy"]) < 0.04):
            button = o["button"]
        if button is None:
            log("ERROR no button known")
            return {"complete": False}
        log("pressing button at", np.round(button["xy"], 3))
        rb.observe()
        save_frame(rb, "before_press")
        press_button(rb, button)
        rb.hold(1.0, 10)
        rb.observe()
        save_frame(rb, "end")
        log("finished; complete=", rb.complete, "ended=", rb.ended, "step", rb.step_id)
    except Ended:
        log("ended; complete=", rb.complete)
    finally:
        try:
            os.makedirs(OUT, exist_ok=True)
            open(f"{OUT}/controller_log.txt", "w").write("\n".join(LOG))
        except Exception:
            pass
    return {"complete": rb.complete}
