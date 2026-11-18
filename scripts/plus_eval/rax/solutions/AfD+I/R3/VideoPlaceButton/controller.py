"""VideoPlaceButton controller v1.
Perception: from the conditioning video find the button press, the first new cube placement after it,
follow that target through pair swaps, and place the cube on that target's final slot."""
import json, os, sys, time, base64, io, math, traceback
from pathlib import Path
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import perception as P

WS = Path(os.environ.get("VPB_WS_ROOT", "/workspace"))
OUT = Path("/workspace/output"); OUT.mkdir(parents=True, exist_ok=True)
LOG = open(OUT / "vpb_log.txt", "a")

def log(*a):
    s = " ".join(str(x) for x in a)
    print(s); LOG.write(s + "\n"); LOG.flush()

def meta_of(reply):
    for b in reply.get("content", []):
        if b.get("type") == "text":
            try:
                return json.loads(b["text"])
            except Exception:
                pass
    return {}

def image_of(reply, meta, camera, kind="rgb"):
    for att in meta.get("attachments", []):
        if att.get("camera") == camera and att.get("kind") == kind:
            blk = reply["content"][att["content_index"]]
            data = base64.b64decode(blk["data"])
            im = Image.open(io.BytesIO(data))
            return np.asarray(im.convert("RGB")) if kind == "rgb" else np.asarray(im)
    return None

class Robot:
    def __init__(self, ctx):
        self.ctx = ctx; self.meta = None; self.reply = None; self.ended = False; self.success = None
        self.observe()

    def observe(self):
        self.reply = self.ctx.call("robomme_observe"); self.meta = meta_of(self.reply)
        return self.meta

    @property
    def state(self):
        return np.array(self.meta["state"], float)

    def remaining(self):
        lim = self.meta.get("episode_limits") or {}
        return lim.get("remaining_episode_steps", 10 ** 6)

    def step(self, rows):
        if self.ended:
            return False
        rows = [[float(np.clip(v, -1, 1)) for v in r] for r in rows]
        rem = self.remaining()
        if rem is not None and rem <= 0:
            self.ended = True; return False
        rows = rows[:max(1, min(100, rem if rem is not None else 100))]
        reply = self.ctx.call("robomme_step", actions=rows)
        m = meta_of(reply)
        self.reply = reply
        tr = m.get("transition") or {}
        steps = tr.get("steps") or m.get("steps") or []
        if tr.get("task_complete") or any(isinstance(s, dict) and s.get("task_complete") for s in steps):
            self.success = True
        if m.get("episode_ended") or any(isinstance(s, dict) and s.get("episode_ended") for s in steps):
            self.ended = True
            log("episode ended; task_complete=", self.success, "keys", list(m.keys()))
        if "state" in m:
            self.meta = m
        else:
            if not self.ended:
                self.observe()
        return not self.ended

    def move_to(self, target, grip, tol=0.006, vmax=0.35, max_calls=30, chunk=8):
        target = np.asarray(target, float)
        for _ in range(max_calls):
            if self.ended: return False
            pos = self.state[:3]; err = target - pos; d = np.linalg.norm(err)
            if d < tol: return True
            k = int(min(chunk, max(1, math.ceil(d / (vmax * 0.05)))))
            delta = err / (k * 0.05)
            n = np.linalg.norm(delta)
            if n > vmax: delta = delta / n * vmax
            rows = [list(delta) + [0.0, 0.0, 0.0, grip]] * k
            self.step(rows)
        return np.linalg.norm(target - self.state[:3]) < 3 * tol

    def grip(self, g, n=10):
        pos = self.state[:3]
        self.step([[0, 0, 0, 0, 0, 0, g]] * n)

def load_video(meta):
    seq = meta.get("frame_sequence") or {}
    mp = seq.get("manifest_path")
    if not mp:
        return [], [], None
    man = json.loads((WS / mp).read_text())
    frames = sorted(man["frames"], key=lambda f: f["frame_index"])
    imgs, times, exe = [], [], None
    for f in frames:
        files = {(x["camera"], x["kind"]): x["path"] for x in f["files"]}
        if ("front", "rgb") not in files: continue
        if f["phase"] == "video_demo":
            imgs.append(np.asarray(Image.open(WS / files[("front", "rgb")]).convert("RGB")))
            times.append(float(f.get("timestamp_s", len(times) * 0.05)))
        elif f["phase"] == "execution" and exe is None:
            exe = f
    return imgs, times, exe

def p2w(robot, pts, radius=1):
    sel = [{"camera": "front", "u": float(u), "v": float(v)} for u, v in pts]
    r = robot.ctx.call("robomme_pixel_to_position", observation_step_id=int(robot.meta["step_id"]),
                       selections=sel, depth_window_radius=radius)
    m = meta_of(r)
    out = []
    res = m.get("selections") or m.get("positions") or m.get("results") or []
    for x in res:
        if isinstance(x, dict):
            p = x.get("position_m") or x.get("world_position_m") or x.get("position") or x.get("world")
            out.append(None if (p is None or x.get("valid") is False) else np.array(p, float))
        else:
            out.append(np.array(x, float))
    if not out:
        log("p2w unexpected reply", json.dumps(m)[:800])
    return out

def main(ctx):
    try:
        _main(ctx)
    except Exception:
        log("controller exception", traceback.format_exc())

def _main(ctx):
    t0 = time.time()
    robot = Robot(ctx)
    meta = robot.meta
    instr = meta.get("instruction", "")
    log("instruction:", instr, "step", meta.get("step_id"), "limits", meta.get("episode_limits"))
    log("state", meta.get("state"))
    imgs, times, exe = load_video(meta)
    log("video frames", len(imgs), "span", (times[0], times[-1]) if times else None, "load_s", round(time.time() - t0, 1))
    res = None
    try:
        live0 = image_of(robot.reply, meta, "front")
        res = P.analyse_v3(imgs, times, instr, log=log, live=live0) if len(imgs) > 5 else None
    except Exception:
        log("analysis error", traceback.format_exc())
    live = image_of(robot.reply, meta, "front")
    color = P.instruction_color(instr)
    live_t = P.detect_targets(live)
    log("live targets", [(round(x), round(y), n) for x, y, n in live_t])
    if not live_t:
        log("no targets detected live; abort"); return
    if res and res.get("ok"):
        fx, fy = res["final_px"]
    else:
        log("analysis failed; fallback to first live target")
        fx, fy = live_t[0][0], live_t[0][1]
    j = int(np.argmin([math.hypot(x - fx, y - fy) for x, y, _ in live_t]))
    tx, ty = live_t[j][0], live_t[j][1]
    log("target px", (round(tx, 1), round(ty, 1)), "from video px", (round(fx, 1), round(fy, 1)))
    cp = P.cube_pos(live, color)
    log("cube px", cp)
    if cp is None:
        log("cube not found; abort"); return
    # 3D localisation via depth
    pts = p2w(robot, [(tx, ty)], radius=1)
    log("p2w target", [None if p is None else np.round(p, 4).tolist() for p in pts])
    tgt3 = pts[0]
    if tgt3 is None:
        m = robot.meta; dep = image_of(robot.reply, m, "front", "depth").astype(float) * 0.001
        tgt3 = P.backproject(dep, m["camera_intrinsics"]["front"]["matrix"], m["camera_poses"]["front"], [tx], [ty])[0]
    table_z = float(min(tgt3[2], 0.02))
    run_pick_place(robot, color, tgt3, table_z)
    log("done; success flag", robot.success, "elapsed", round(time.time() - t0, 1))

CUBE = 0.04   # cube edge measured live (top z ~0.041 m, x extent ~0.044 m)

def locate_cube(robot, color, camera="front"):
    m = robot.meta
    rgb = image_of(robot.reply, m, camera); dep = image_of(robot.reply, m, camera, "depth")
    if rgb is None or dep is None:
        return None
    dep = dep.astype(float) * float(m.get("depth", {}).get(camera, {}).get("meters_per_unit", 0.001))
    c, info = P.cube_top_center(rgb, dep, m["camera_intrinsics"][camera]["matrix"], m["camera_poses"][camera], color)
    log(f"locate_cube[{camera}]", None if c is None else np.round(c, 4).tolist(), info)
    return c

def run_pick_place(robot, color, tgt3, table_z):
    s = robot.state; log("start state", np.round(s, 4).tolist())
    hover = table_z + 0.14
    robot.move_to([s[0], s[1], max(s[2], hover)], -1)
    grasped = False
    for attempt in range(3):
        if robot.ended: return
        c = locate_cube(robot, color, "front")
        if c is None:
            log("cube not visible in front; trying wrist"); c = locate_cube(robot, color, "wrist")
            if c is None: return
        robot.move_to([c[0], c[1], hover], -1)
        cw = locate_cube(robot, color, "wrist")
        if cw is not None and np.linalg.norm(cw[:2] - c[:2]) < 0.04:
            c = cw
            robot.move_to([c[0], c[1], hover], -1, tol=0.004)
        top = c[2]
        grasp_z = max(table_z + 0.012, top - 0.021)
        robot.move_to([c[0], c[1], grasp_z + 0.04], -1, tol=0.004)
        robot.move_to([c[0], c[1], grasp_z], -1, tol=0.004)
        robot.grip(1.0, 12)
        st = robot.state; w = st[6] + st[7]
        log(f"attempt {attempt} after grasp state", np.round(st, 4).tolist(), "opening", round(w, 4))
        robot.move_to([c[0], c[1], hover], 1.0)
        st = robot.state; w = st[6] + st[7]
        if w > 0.01:
            grasped = True; break
        log("grasp failed (opening %.4f); retry" % w)
        robot.grip(-1.0, 6)
    if not grasped:
        log("giving up grasp"); return
    robot.move_to([tgt3[0], tgt3[1], hover], 1.0, tol=0.004)
    robot.move_to([tgt3[0], tgt3[1], table_z + CUBE + 0.035], 1.0, tol=0.004)
    robot.move_to([tgt3[0], tgt3[1], table_z + CUBE * 0.5 + 0.012], 1.0, tol=0.004)
    robot.grip(-1.0, 10)
    robot.move_to([tgt3[0], tgt3[1], hover], -1.0)
    c = locate_cube(robot, color, "front")
    if c is not None:
        log("placed cube xy err to target", np.round(c[:2] - tgt3[:2], 4).tolist())
    log("final state", np.round(robot.state, 4).tolist())
