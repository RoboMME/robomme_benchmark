"""RouteStick controller v1: reproduce the conditioning-video stick route.

Pipeline: read conditioning video manifest -> demo TCP path (from video frame states)
-> detect targets in video frame 0 and live frame (front RGB-D) -> split demo path into
hops between targets -> map each hop into live target frame -> follow path closed-loop.
"""
import json, os, sys, time, traceback
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from geom import load_rgb, load_depth, world_to_pixel, red_count, rot_error
import base64, io
from PIL import Image
from scene import detect_scene
from tool import wrist_tool_lowest

VERSION = "v6"
WS = "/workspace"
ALPHA = 0.40          # measured fraction of commanded delta achieved per step (ep1 probe 0.38-0.40)
WP_SPACING = float(os.environ.get("ROUTESTICK_SPEED", "0.007"))  # m per step along path (demo ~4.4 mm/step)
DWELL = 8             # zero rows held on each target centre (red marker shows ~5-10 steps after touch, ep1)
RED_FRONT = 10; RED_WRIST = 20  # red pixels around projected target = registered visit
Z_SEARCH_STEP = 0.004; Z_SEARCH_MAX = 0.012  # lower draw height if first target never turns red
BUTTON_Z = 0.016
ROT_GAIN = 1.0        # orientation hold: fraction of rotation error commanded per row (rehearsal1 drifted 1.2 rad without it)
MAX_TRANS = 0.6       # per-row translation action clip      # measured button top height (ep1)
CHUNK = 5             # rows per robomme_step call
HOVER_DZ = 0.06       # hover height above draw height before descending
VISIT_R = 0.02        # demo sample within this of a target = visit
MATCH_TOL = 0.01      # live vs video target "same layout" tolerance
MATCH_TOL_FAR = 0.06  # max shift accepted when matching a video target to a live one
LOG = None


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    try:
        with open(os.path.join(WS, "output", "controller_log.txt"), "a") as f:
            f.write(msg + "\n")
    except Exception:
        pass


def meta(res):
    return json.loads(next(b["text"] for b in res["content"] if b.get("type") == "text"))


def frame_scene(frame):
    files = {(x["camera"], x["kind"]): x["path"] for x in frame["files"]}
    rgb = load_rgb(os.path.join(WS, files[("front", "rgb")]))
    d = load_depth(os.path.join(WS, files[("front", "depth")]))
    return detect_scene(rgb, d, frame["camera_intrinsics"]["front"]["matrix"], frame["camera_poses"]["front"])


def tool_off(frame):
    try:
        files = {(x["camera"], x["kind"]): x["path"] for x in frame["files"]}
        d = load_depth(os.path.join(WS, files[("wrist", "depth")]))
        return wrist_tool_lowest(d, frame["camera_intrinsics"]["wrist"]["matrix"], frame["camera_poses"]["wrist"], frame["state"][2])
    except Exception:
        log("tool_off failed", traceback.format_exc())
        return None


def demo_path(frames):
    vid = [f for f in frames if f["phase"] == "video_demo"]
    vid.sort(key=lambda f: f["frame_index"])
    S = np.array([f["state"][:3] for f in vid])
    return S, vid


def split_hops(S, T, draw_z):
    """Return list of visits [(target_idx, sample_idx)] on the drawing part of the path."""
    visits = []
    low = S[:, 2] < draw_z + 0.01
    for k in range(len(S)):
        if not low[k]:
            continue
        dd = np.linalg.norm(T - S[k, :2], axis=1)
        j = int(dd.argmin())
        if dd[j] < VISIT_R:
            if not visits or visits[-1][0] != j:
                visits.append([j, k, dd[j]])
            elif dd[j] < visits[-1][2]:
                visits[-1][1] = k; visits[-1][2] = dd[j]
    return [(v[0], v[1]) for v in visits]


def local_map(P, A, B, A2, B2):
    """map points P (Nx2) from hop frame (A->B) into hop frame (A2->B2)."""
    e = B - A; L = np.linalg.norm(e); e = e / L; n = np.array([-e[1], e[0]])
    u = (P - A) @ e / L; w = (P - A) @ n / L
    e2 = B2 - A2; L2 = np.linalg.norm(e2); e2 = e2 / L2; n2 = np.array([-e2[1], e2[0]])
    return A2 + L2 * (u[:, None] * e2 + w[:, None] * n2)


def build_plan(frames):
    S, vid = demo_path(frames)
    Tv, _ = frame_scene(vid[0])
    live = [f for f in frames if f["phase"] == "execution"][-1]
    Tl, _ = frame_scene(live)
    draw_z = float(np.median(S[S[:, 2] < S[:, 2].min() + 0.01, 2]))
    ov, ol = tool_off(vid[0]), tool_off(live)
    dz = 0.0
    if ov is not None and ol is not None:
        dz = float(np.clip(ol - ov, -0.03, 0.10))
    log("tool lowest-point offset video", ov, "live", ol, "-> draw_z shift", round(-dz, 4))
    draw_z_video = draw_z
    draw_z = draw_z - dz
    log("video targets", np.round(Tv, 3).tolist())
    log("live targets", np.round(Tl, 3).tolist())
    log("draw_z", round(draw_z, 4))
    visits = split_hops(S, Tv, draw_z_video)
    hops = []
    for (a, ka), (b, kb) in zip(visits[:-1], visits[1:]):
        seg = S[ka:kb + 1, :2]
        e = Tv[b] - Tv[a]; n = np.array([-e[1], e[0]]) / np.linalg.norm(e)
        side = float(np.mean((seg - Tv[a]) @ n))
        hops.append((a, b, ka, kb, side))
    log("visits", [v[0] for v in visits])
    log("hops", [(a, b, "L" if s > 0 else "R", round(s, 3)) for a, b, _, _, s in hops])
    # correspondence live <-> video targets
    Tm = Tv.copy()
    if len(Tl) == len(Tv):
        # same number of buttons in a row: correspond by order along the row (layouts shift/rotate, ep2)
        Tm = Tl.copy()
        log("target correspondence by row order")
    else:
        log("WARN live target count", len(Tl), "!= video", len(Tv), "; nearest-match after centroid shift")
        shift = (Tl.mean(0) - Tv.mean(0)) if len(Tl) else np.zeros(2)
    for i in range(len(Tv) if len(Tl) != len(Tv) else 0):
        if len(Tl):
            dd = np.linalg.norm(Tl - (Tv[i] + shift), axis=1)
            j = int(dd.argmin())
            if dd[j] < MATCH_TOL_FAR:
                Tm[i] = Tl[j]
            else:
                Tm[i] = Tv[i] + shift
                log("WARN video target", i, "has no live match; using shifted video position")
    same = bool(np.all(np.linalg.norm(Tm - Tv, axis=1) < MATCH_TOL))
    log("layout_same", same, "max shift", round(float(np.max(np.linalg.norm(Tm - Tv, axis=1))), 4))
    Tl = Tm
    arcs = []
    for a, b, ka, kb, s in hops:
        seg = S[ka:kb + 1, :2].copy()
        seg[0] = Tv[a]; seg[-1] = Tv[b]
        arcs.append((a, b, local_map(seg, Tv[a], Tv[b], Tl[a], Tl[b])))
    return arcs, draw_z, Tl, visits, hops


def resample(path2, spacing):
    seg = np.linalg.norm(np.diff(path2, axis=0), axis=1)
    s = np.r_[0, np.cumsum(seg)]
    n = max(2, int(np.ceil(s[-1] / spacing)) + 1)
    q = np.linspace(0, s[-1], n)
    return np.c_[np.interp(q, s, path2[:, 0]), np.interp(q, s, path2[:, 1])]


def load_frames(first):
    man = None
    fs = first.get("frame_sequence") or {}
    if fs.get("manifest_path"):
        man = os.path.join(WS, fs["manifest_path"])
    if man is None or not os.path.exists(man):
        man = os.environ.get("ROUTESTICK_MANIFEST")
    if man is None or not os.path.exists(man):
        import glob
        c = glob.glob(os.path.join(WS, "runtime/autonomous_controller/runs/*/frames/sequence_000000/manifest.json"))
        c.sort(key=os.path.getmtime)
        man = c[-1]
    log("manifest", man)
    return json.load(open(man))["frames"]


class Robot:
    def __init__(self, ctx):
        self.ctx = ctx
        self.ended = False
        self.complete = False
        res = ctx.call("robomme_observe")
        r = meta(res)
        self.update(r)
        self.first = r
        self.last_res, self.last = res, r
        self.ref_rot = np.array(r["state"][3:6])

    def update(self, r):
        self.state = np.array(r["state"])
        self.step = r.get("step_id")
        lim = r.get("episode_limits") or {}
        self.remaining = lim.get("remaining_episode_steps", 10 ** 6)
        if r.get("episode_ended"):
            self.ended = True
        tr = r.get("transition") or {}
        steps = tr.get("steps", []) if isinstance(tr, dict) else []
        if isinstance(tr, dict):
            if tr.get("task_complete"):
                self.complete = True
            if tr.get("episode_ended"):
                self.ended = True
        for st in steps or []:
            if st.get("task_complete"):
                self.complete = True
            if st.get("episode_ended"):
                self.ended = True
        if r.get("task_complete"):
            self.complete = True

    def send(self, rows):
        if self.ended:
            return
        rows = rows[: max(0, int(self.remaining))]
        if not rows:
            self.ended = True
            return
        rows = [[float(np.clip(v, -1, 1)) for v in row] for row in rows]
        res = self.ctx.call("robomme_step", actions=rows)
        r = meta(res)
        self.update(r)
        self.last_res, self.last = res, r
        return r

    def observe(self):
        res = self.ctx.call("robomme_observe")
        r = meta(res); self.update(r); self.last_res, self.last = res, r
        return r

    def red_at(self, xy):
        """red pixel counts around target xy in front and wrist images of the last reply."""
        r, res = self.last, self.last_res
        out = {}
        for a in r.get("attachments", []):
            if a.get("kind") != "rgb":
                continue
            cam = a["camera"]
            try:
                b = res["content"][a["content_index"]]
                rgb = np.asarray(Image.open(io.BytesIO(base64.b64decode(b["data"]))).convert("RGB")).astype(float)
                uv = world_to_pixel([xy[0], xy[1], BUTTON_Z], r["camera_intrinsics"][cam]["matrix"], r["camera_poses"][cam])
                out[cam] = red_count(rgb, uv)
            except Exception:
                out[cam] = -1
        return out

    def track(self, targets):
        """targets: list of xyz; command rows assuming ALPHA response."""
        p = self.state[:3].copy()
        try:
            err = rot_error(self.ref_rot, self.state[3:6])
        except Exception:
            err = np.zeros(3)
        rows = []
        for t in targets:
            d = (np.asarray(t) - p) / ALPHA
            d = np.clip(d / 0.05, -MAX_TRANS, MAX_TRANS)
            rr = np.clip(ROT_GAIN * err / 0.5, -0.4, 0.4)
            rows.append([d[0], d[1], d[2], rr[0], rr[1], rr[2], -1])
            p = p + ALPHA * d * 0.05
            err = err - ALPHA * rr * 0.5
        return self.send(rows)

    def goto(self, goal, speed=0.008, tol=0.004, max_calls=40):
        goal = np.asarray(goal, float)
        prev, stall = None, 0
        self.stalled = False
        for _ in range(max_calls):
            if self.ended:
                return False
            p = self.state[:3]
            dist = np.linalg.norm(goal - p)
            if dist < tol:
                return True
            if prev is not None and prev - dist < 0.001:
                stall += 1
                if stall >= 2:
                    self.stalled = True
                    log("goto stalled at", np.round(p, 4).tolist(), "goal", np.round(goal, 4).tolist(), "dist", round(float(dist), 4))
                    return False
            else:
                stall = 0
            prev = dist
            n = int(np.clip(np.ceil(dist / speed), 1, 10))
            tg = [p + (goal - p) * (i + 1) / n for i in range(n)]
            self.track(tg)
        return np.linalg.norm(goal - self.state[:3]) < 3 * tol


def visit(rob, xy, z, label, attempts=3):
    """centre on a target, hold, verify the red registration marker; retry up to 3 times."""
    reds = {}
    for attempt in range(attempts):
        rob.goto(np.r_[xy, z], speed=0.004, tol=0.0015, max_calls=6)
        rob.send([[0, 0, 0, 0, 0, 0, -1]] * (DWELL + 3 * attempt))
        if rob.ended:
            break
        if rob.complete:
            return True
        reds = rob.red_at(xy)
        ok = reds.get("front", 0) >= RED_FRONT or reds.get("wrist", 0) >= RED_WRIST
        log("visit", label, "attempt", attempt, "pos", np.round(rob.state[:3], 4).tolist(),
            "err", round(float(np.linalg.norm(rob.state[:2] - xy)), 4), "red", reds, "ok", ok, "step", rob.step)
        if ok:
            return True
    return False


def follow(rob, wp, z):
    k, N, maxerr = 0, len(wp), 0.0
    while k < N - 1 and not rob.ended:
        p = rob.state[:2]
        win = wp[k:min(N, k + 15)]
        j = k + int(np.argmin(np.linalg.norm(win - p, axis=1)))
        maxerr = max(maxerr, float(np.linalg.norm(wp[j] - p)))
        k = j
        tg = [np.r_[wp[min(N - 1, k + i + 1)], z] for i in range(CHUNK)]
        rob.track(tg)
        k = min(N - 1, k + CHUNK)
    return maxerr


def main(ctx):
    global WP_SPACING
    t0 = time.time()
    log("RouteStick controller", VERSION, "speed", WP_SPACING)
    rob = Robot(ctx)
    try:
        frames = load_frames(rob.first)
        arcs, draw_z, Tl, visits, hops = build_plan(frames)
    except Exception:
        log("PLAN FAILED", traceback.format_exc())
        return
    if not arcs:
        log("no hops found"); return
    total = sum(float(np.sum(np.linalg.norm(np.diff(a[2], axis=0), axis=1))) for a in arcs)
    need = total / WP_SPACING + (len(arcs) + 1) * (DWELL + 6) + 80
    if need > 0.85 * rob.remaining:
        WP_SPACING = min(0.011, total / max(1.0, 0.85 * rob.remaining - (len(arcs) + 1) * (DWELL + 6) - 80))
    log("hops", len(arcs), "length", round(total, 3), "remaining", rob.remaining, "est steps", int(need),
        "spacing", round(WP_SPACING, 4), "plan time", round(time.time() - t0, 1))
    try:
        first = Tl[arcs[0][0]]
        rob.goto(np.r_[first, draw_z + 0.15], speed=0.014, tol=0.008)
        ok = rob.goto(np.r_[first, draw_z + HOVER_DZ], speed=0.012, tol=0.005)
        log("hover", ok, np.round(rob.state[:3], 4).tolist(), "rot", np.round(rob.state[3:6], 3).tolist(), "step", rob.step)
        ok = rob.goto(np.r_[first, draw_z], speed=0.006, tol=0.003)
        if not ok and rob.stalled:
            # cannot reach the planned height (reach limit or contact): draw at the reachable height
            draw_z = float(rob.state[2]) + 0.002
            log("descent stalled; draw_z ->", round(draw_z, 4))
        log("descend", ok, np.round(rob.state[:3], 4).tolist(), "rot", np.round(rob.state[3:6], 3).tolist(), "step", rob.step)
        ok = visit(rob, first, draw_z, "T%d" % arcs[0][0], attempts=2)
        lowered = 0.0
        stalled_desc = rob.stalled
        while not ok and not stalled_desc and not rob.ended and not rob.complete and lowered < Z_SEARCH_MAX - 1e-9:
            lowered += Z_SEARCH_STEP
            ok = visit(rob, first, draw_z - lowered, "T%d-z%.3f" % (arcs[0][0], draw_z - lowered), attempts=1)
            if rob.stalled:
                break
        if ok:
            draw_z = draw_z - lowered
        log("first visit ok", ok, "draw_z", round(draw_z, 4))
        for h, (a, b, arc) in enumerate(arcs):
            if rob.ended or rob.complete:
                break
            wp = resample(arc, WP_SPACING)
            e = follow(rob, wp, draw_z)
            log("hop", h, a, "->", b, "maxerr", round(e, 4), "z", round(float(rob.state[2]), 4), "rot", np.round(rob.state[3:6], 3).tolist(), "step", rob.step, "remaining", rob.remaining)
            visit(rob, Tl[b], draw_z, "T%d" % b)
        log("route done step", rob.step, "ended", rob.ended, "complete", rob.complete)
        if not rob.ended and not rob.complete:
            rob.send([[0, 0, 0, 0, 0, 0, -1]] * 10)
        log("final step", rob.step, "ended", rob.ended, "complete", rob.complete, "wall", round(time.time() - t0, 1))
    except Exception:
        log("EXEC ERROR", traceback.format_exc())
