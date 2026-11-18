"""VideoPlaceOrder controller v1.
Strategy: read conditioning-video robot states -> release events (placements);
pick k-th placement from instruction; detect targets & cube in live front RGB-D;
pick cube, place on k-th target.
"""
import json, re, time, traceback
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial.transform import Rotation as R

WS = Path("/workspace")
OUT = WS / "output"
LOG = []

def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)

ORD = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
       "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10, "1st": 1, "2nd": 2, "3rd": 3,
       "4th": 4, "5th": 5, "6th": 6, "last": -1, "final": -1}

COLORS = ["red", "green", "blue", "yellow", "purple", "orange", "white", "black", "pink", "cyan"]

def meta(reply):
    for b in reply.get("content", []):
        if b.get("type") == "text":
            try:
                return json.loads(b["text"])
            except Exception:
                pass
    return {}

class Robot:
    def __init__(self, ctx):
        self.ctx = ctx
        self.obs = None
        self.ended = False
        self.complete = None
        self.down = R.from_rotvec([np.pi, 0, 0])

    def observe(self):
        self.obs = meta(self.ctx.call("robomme_observe"))
        return self.obs

    @property
    def state(self):
        return np.array(self.obs["state"], float)

    def remaining(self):
        lim = self.obs.get("episode_limits") or {}
        return lim.get("remaining_episode_steps", 999)

    def step(self, rows):
        if self.ended:
            return None
        rows = [[float(np.clip(x, -1, 1)) for x in r] for r in rows]
        m = meta(self.ctx.call("robomme_step", actions=rows))
        for s in m.get("steps", []) or []:
            if isinstance(s, dict) and s.get("task_complete") is not None:
                self.complete = s.get("task_complete")
        if m.get("episode_ended"):
            self.ended = True
            log("EPISODE ENDED", "task_complete=", self.complete)
        if "state" in m:
            self.obs = m
        else:
            self.observe()
        return m

    def rot_cmd(self, target_rot):
        cur = R.from_rotvec(self.state[3:6])
        err = (target_rot * cur.inv()).as_rotvec()
        return np.clip(err / 0.5, -0.6, 0.6)

    def move_to(self, xyz, grip, tol=0.004, max_steps=80, vmax=0.5, rot=None, chunk=1):
        rot = rot or self.down
        xyz = np.array(xyz, float)
        n = 0
        while n < max_steps and not self.ended:
            p = self.state[:3]
            d = xyz - p
            if np.linalg.norm(d) < tol:
                break
            a = np.clip(d / 0.05, -vmax, vmax)
            # keep proportional for final approach
            r = self.rot_cmd(rot)
            row = list(a) + list(r) + [grip]
            self.step([row] * chunk)
            n += chunk
        return np.linalg.norm(xyz - self.state[:3])

    def hold(self, grip, n):
        rows = []
        for _ in range(n):
            rows.append([0, 0, 0, 0, 0, 0, grip])
        self.step(rows)

# ---------------- perception -----------------

def load_rgb(path):
    return np.array(Image.open(WS / path).convert("RGB")).astype(int)

def load_depth(path):
    return np.array(Image.open(WS / path)).astype(float) * 0.001

def deproject(u, v, d, K, pose):
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    pc = np.array([(u - cx) * d / fx, -(v - cy) * d / fy, -d])
    w, x, y, z = pose["quaternion_wxyz"]
    rot = R.from_quat([x, y, z, w])
    return rot.apply(pc) + np.array(pose["position_m"])

def color_mask(im, color):
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    if color == "red":
        return (r > 100) & (g < 60) & (b < 60)
    if color == "green":
        return (g > 120) & (r < 80) & (b < 80)
    if color == "blue":
        return (b > 120) & (r < 80) & (g < 80)
    if color == "yellow":
        return (r > 150) & (g > 150) & (b < 80)
    return (r > 100) & (g < 60) & (b < 60)

def target_mask(im):
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    return (r > 120) & (b > 150) & (g < 130) & (b - g > 60)

def blobs_world(mask, depth, K, pose, min_px=15, dilate=3):
    m2 = ndimage.binary_dilation(mask, iterations=dilate) if dilate else mask
    lab, n = ndimage.label(m2)
    out = []
    for i in range(1, n + 1):
        sel = (lab == i) & mask
        cnt = int(sel.sum())
        if cnt < min_px:
            continue
        vs, us = np.nonzero(sel)
        pts = []
        for v, u in zip(vs, us):
            d = depth[v, u]
            if d > 0:
                pts.append(deproject(u, v, d, K, pose))
        if not pts:
            continue
        pts = np.array(pts)
        out.append({"px": cnt, "uv": (float(us.mean()), float(vs.mean())),
                    "xyz": np.median(pts, axis=0), "zmax": float(np.percentile(pts[:, 2], 90))})
    return out

def frame_files(frame):
    return {(f["camera"], f["kind"]): f["path"] for f in frame["files"]}

def obs_files(obs):
    # live observation: images are inline; use frame paths if published
    return None

# ---------------- video analysis -----------------

def video_events(frames):
    """Return list of grasp and release events from robot state in video frames."""
    vids = [f for f in frames if f.get("phase") == "video_demo"]
    vids.sort(key=lambda f: f["frame_index"])
    events = []
    holding = False
    run = 0
    for i, f in enumerate(vids):
        s = f["state"]
        w = s[6] + s[7]
        if not holding:
            if 0.01 < w < 0.065:
                run += 1
            else:
                run = 0
            if run >= 3:
                # grasp: gripper partly closed (object between fingers) for 3 frames
                holding = True
                g = vids[i - 2]["state"]
                events.append(("grasp", np.array(g[:3]), vids[i - 2]["frame_index"]))
        elif w > 0.07 or w < 0.004:
            holding = False
            run = 0
            events.append(("release", np.array(s[:3]), f["frame_index"], w))
    return events

def releases_from(events, frames):
    vids = sorted([f for f in frames if f.get("phase") == "video_demo"], key=lambda f: f["frame_index"])
    byidx = {f["frame_index"]: f for f in vids}
    rel = []
    for e in events:
        if e[0] != "release" or e[3] < 0.004:
            continue
        # position a couple frames before opening (pose at release)
        fi = e[2]
        prev = [byidx[j] for j in range(fi - 3, fi + 1) if j in byidx]
        pos = np.mean([np.array(p["state"][:3]) for p in prev], axis=0)
        rel.append({"xyz": pos, "frame": fi})
    return rel

def parse_instruction(text):
    t = text.lower()
    color = next((c for c in COLORS if re.search(r"\b%s\b" % c, t)), "red")
    k = None
    for w, v in ORD.items():
        if re.search(r"\b%s\b" % w, t):
            k = v
            break
    if k is None:
        mm = re.search(r"(\d+)", t)
        k = int(mm.group(1)) if mm else 1
    return color, k

# ---------------- main -----------------

def save_log():
    try:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "controller_log.txt").write_text("\n".join(LOG))
    except Exception:
        pass

def main(ctx):
    t0 = time.time()
    rb = Robot(ctx)
    try:
        run(rb, t0)
    except Exception:
        log("EXC", traceback.format_exc())
    save_log()

def find_targets(img, depth, K, pose):
    tg = blobs_world(target_mask(img), depth, K, pose, min_px=20, dilate=3)
    # keep only table-level blobs
    tg = [t for t in tg if t["xyz"][2] < 0.03]
    return tg

def run(rb, t0):
    obs = rb.observe()
    instr = obs.get("instruction", "")
    color, k = parse_instruction(instr)
    log("instruction:", instr, "| color", color, "k", k)
    seq = obs.get("frame_sequence") or {}
    man = json.loads((WS / seq["manifest_path"]).read_text())
    frames = man["frames"]
    events = video_events(frames)
    for e in events:
        log("event", e[0], np.round(e[1], 3).tolist(), e[2])
    rel = releases_from(events, frames)
    grasps = [e for e in events if e[0] == "grasp"]
    exe = [f for f in frames if f.get("phase") == "execution"]
    exe = exe[-1] if exe else frames[-1]
    ff = frame_files(exe)
    img = load_rgb(ff[("front", "rgb")])
    dep = load_depth(ff[("front", "depth")])
    K = exe["camera_intrinsics"]["front"]["matrix"]
    pose = exe["camera_poses"]["front"]
    targets = find_targets(img, dep, K, pose)
    for t in targets:
        log("target", np.round(t["xyz"], 3).tolist(), t["px"])
    goal = None
    try:
        import pairs as PZ
        A = PZ.analyse(frames, log=log)
        slots = A["slots"]
        plist = []
        for r_ in rel:
            d = np.linalg.norm(slots - r_["xyz"][:2], axis=1)
            sidx = int(np.argmin(d))
            if d[sidx] > 0.05:
                log("release not on target (dropped)", np.round(r_["xyz"][:2], 3).tolist(), round(float(d[sidx]), 3))
                continue
            fi = r_["frame"]
            occ = A["occ"].get(fi)
            if occ is None:
                occ = A["occ"][min(A["occ"].keys(), key=lambda x: abs(x - fi))]
            tid = int(occ[sidx])
            fs = int(np.where(A["final"] == tid)[0][0])
            plist.append({"frame": fi, "slot": sidx, "tid": tid, "final_slot": fs, "xy": slots[fs].copy()})
            log("placement", len(plist), "frame", fi, "slot", sidx, "target", tid, "-> now at slot", fs, np.round(slots[fs], 3).tolist())
        if plist:
            idx = (k - 1) if k > 0 else len(plist) - 1
            idx = max(0, min(idx, len(plist) - 1))
            goal = plist[idx]
    except Exception:
        log("pair analysis failed", traceback.format_exc())
    if goal is None:
        placements = [{"xy": r_["xyz"][:2]} for r_ in rel]
        if not placements:
            log("NO PLACEMENTS FOUND")
            return
        idx = (k - 1) if k > 0 else len(placements) - 1
        goal = placements[max(0, min(idx, len(placements) - 1))]
    # snap to live detected target
    if targets:
        dd = [np.linalg.norm(t["xyz"][:2] - goal["xy"]) for t in targets]
        j = int(np.argmin(dd))
        if dd[j] < 0.03:
            goal["xy"] = targets[j]["xyz"][:2].copy()
    log("goal", np.round(goal["xy"], 3).tolist())

    # locate cube in live front image
    cubes = blobs_world(color_mask(img, color), dep, K, pose, min_px=8, dilate=1)
    cubes = [c for c in cubes if c["xyz"][2] < 0.08]
    cube = None
    if cubes:
        c = max(cubes, key=lambda c: c["px"])
        cube = c["xyz"].copy()
        log("cube front surface", np.round(cube, 3).tolist(), c["px"])
    # expected cube location: last release in video, else last grasp
    exp = rel[-1]["xyz"] if rel else (grasps[0][1] if grasps else None)
    if cube is None and exp is not None:
        cube = np.array(exp, float)
    log("cube estimate", np.round(cube, 3).tolist(), "video end", None if exp is None else np.round(exp, 3).tolist())
    pick(rb, cube, color)
    place(rb, goal["xy"])
    log("done; elapsed", round(time.time() - t0, 1), "complete", rb.complete)

def refine_with_wrist(rb, color, guess):
    """Above the cube, use wrist RGB-D to recentre."""
    o = rb.observe()
    # get wrist image paths? live obs images inline only; use pixel_to_position on mask centroid
    return guess

def current_images(rb):
    """Fetch current images via observe content blocks (base64 PNG)."""
    import base64, io
    rep = rb.ctx.call("robomme_observe")
    m = meta(rep)
    rb.obs = m
    blocks = rep.get("content", [])
    out = {}
    for a in m.get("attachments", []):
        b = blocks[a["content_index"]]
        data = base64.b64decode(b.get("data", ""))
        im = Image.open(io.BytesIO(data))
        if a["kind"] == "rgb":
            out[(a["camera"], "rgb")] = np.array(im.convert("RGB")).astype(int)
        else:
            out[(a["camera"], "depth")] = np.array(im).astype(float) * 0.001
    return m, out

def wrist_locate(rb, color):
    m, ims = current_images(rb)
    if ("wrist", "rgb") not in ims:
        return None
    img, dep = ims[("wrist", "rgb")], ims[("wrist", "depth")]
    K = m["camera_intrinsics"]["wrist"]["matrix"]
    pose = m["camera_poses"]["wrist"]
    bl = blobs_world(color_mask(img, color), dep, K, pose, min_px=15, dilate=1)
    bl = [b for b in bl if b["xyz"][2] < 0.08]
    if not bl:
        return None
    b = max(bl, key=lambda b: b["px"])
    # yaw estimate from mask
    return b["xyz"], b

def pick(rb, cube, color):
    HZ = 0.15
    target = np.array([cube[0] - 0.012, cube[1], HZ])  # front surface -> centre (camera looks -x)
    rb.move_to(target, -1, tol=0.01)
    # refine with wrist view
    for _ in range(2):
        r = wrist_locate(rb, color)
        if r is None:
            log("wrist: cube not seen")
            break
        xyz, b = r
        log("wrist cube", np.round(xyz, 3).tolist(), b["px"])
        target = np.array([xyz[0], xyz[1], HZ])
        rb.move_to(target, -1, tol=0.005)
    top = max(0.02, 0.02)
    rb.move_to([target[0], target[1], 0.06], -1, tol=0.006, vmax=0.4)
    rb.move_to([target[0], target[1], 0.02], -1, tol=0.004, vmax=0.2)
    rb.hold(1, 12)
    w = rb.state[6] + rb.state[7]
    log("grip width after close", round(w, 4))
    rb.move_to([target[0], target[1], HZ], 1, tol=0.01)
    w = rb.state[6] + rb.state[7]
    log("grip width after lift", round(w, 4))
    return w

def place(rb, xy):
    HZ = 0.15
    rb.move_to([xy[0], xy[1], HZ], 1, tol=0.006)
    rb.move_to([xy[0], xy[1], 0.04], 1, tol=0.004, vmax=0.25)
    rb.hold(-1, 10)
    rb.move_to([xy[0], xy[1], HZ], -1, tol=0.01)
    log("placed at", np.round(xy, 3).tolist(), "state", np.round(rb.state[:3], 3).tolist())
