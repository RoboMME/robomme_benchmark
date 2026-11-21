"""VideoPlaceButton controller v1.
Parse conditioning-video robot states -> dips (grasp/release/press);
target = EE xy at first release after first button press;
cube = last release xy (refined by colour segmentation); regrasp with the
EE orientation used when the cube was last released; place at target.
"""
import json, math, time, base64, io, os, re
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R

WS = Path("/workspace")
LOG = []
OUT = WS / "output"


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)
    try:
        OUT.mkdir(parents=True, exist_ok=True)
        with open(OUT / "vpb_log.txt", "a") as f:
            f.write(s + "\n")
    except Exception:
        pass


class Ended(Exception):
    pass


def packet(res):
    for b in res.get("content", []):
        if b.get("type") == "text":
            try:
                return json.loads(b["text"])
            except Exception:
                continue
    return {}


def image_of(res, pkt, camera, kind="rgb"):
    from PIL import Image
    for a in pkt.get("attachments", []):
        if a.get("camera") == camera and a.get("kind") == kind:
            b = res["content"][a["content_index"]]
            return Image.open(io.BytesIO(base64.b64decode(b["data"])))
    return None


class Robot:
    def __init__(self, ctx):
        self.ctx = ctx
        self.pkt = None
        self.res = None
        self.ended = False
        self.complete = False

    def observe(self):
        self.res = self.ctx.call("robomme_observe")
        self.pkt = packet(self.res)
        self._check(self.pkt)
        return self.pkt

    def _check(self, p):
        if p.get("episode_ended"):
            self.ended = True
        if p.get("task_complete"):
            self.complete = True
        tr = p.get("transition") or {}
        for s in tr.get("steps", []) or []:
            if s.get("task_complete"):
                self.complete = True
            if s.get("episode_ended"):
                self.ended = True

    @property
    def state(self):
        return np.array(self.pkt["state"], dtype=float)

    def remaining(self):
        lim = self.pkt.get("episode_limits") or {}
        return lim.get("remaining_episode_steps", 999)

    def step(self, rows):
        if self.ended:
            raise Ended()
        rows = [[float(np.clip(v, -1, 1)) for v in r] for r in rows]
        rem = self.remaining()
        if rem is not None and rem <= 0:
            raise Ended()
        rows = rows[:max(1, int(rem))]
        self.res = self.ctx.call("robomme_step", actions=rows)
        self.pkt = packet(self.res)
        self._check(self.pkt)
        if self.complete:
            log("TASK COMPLETE reported at step", self.pkt.get("step_id"))
        if self.ended:
            raise Ended()
        return self.pkt


def rot_err(cur_rv, tgt_rv):
    """world-frame rotvec taking current to target"""
    return (R.from_rotvec(tgt_rv) * R.from_rotvec(cur_rv).inv()).as_rotvec()


def move_to(rb, pos, rv, grip, tol=0.004, rtol=0.03, max_iter=40, vmax=0.03, chunk=8):
    pos = np.asarray(pos, float); rv = np.asarray(rv, float)
    for it in range(max_iter):
        s = rb.state
        e = pos - s[:3]
        er = rot_err(s[3:6], rv)
        d, dr = np.linalg.norm(e), np.linalg.norm(er)
        if d < tol and dr < rtol:
            return True
        k = max(1, int(math.ceil(max(d / vmax, dr / 0.25))))
        n = min(k, chunk)
        dp = e / k / 0.05
        drr = er / k / 0.5
        row = list(dp) + list(drr) + [grip]
        rb.step([row] * n)
    s = rb.state
    log("move_to not converged", np.round(s[:3], 4), "target", np.round(pos, 4))
    return False


def gripper(rb, grip, n=12):
    rb.step([[0, 0, 0, 0, 0, 0, grip]] * n)
    return rb.state[6] + rb.state[7]


# ---------------------------------------------------------------- video
def load_video(rb):
    seq = rb.pkt.get("frame_sequence") or {}
    mp = seq.get("manifest_path")
    if not mp:
        return []
    m = json.loads((WS / mp).read_text())
    fr = [f for f in m["frames"] if f.get("phase") == "video_demo" and f.get("state")]
    fr.sort(key=lambda f: f["frame_index"])
    return fr


def parse_events(frames, zlow=0.06):
    S = np.array([f["state"] for f in frames], float)
    if len(S) == 0:
        return [], S
    z = S[:, 2]; g = S[:, 6] + S[:, 7]
    def gs(v):
        return "open" if v > 0.065 else ("closed" if v < 0.01 else "hold")
    ev = []
    i = 0; n = len(S)
    while i < n:
        if z[i] < zlow:
            j = i
            while j + 1 < n and z[j + 1] < zlow:
                j += 1
            a = max(0, i - 1); b = min(n - 1, j + 1)
            g0, g1 = gs(g[a]), gs(g[b])
            seg = slice(i, j + 1)
            kmin = i + int(np.argmin(z[seg]))
            if g0 != "hold" and g1 == "hold":
                kind = "grasp"; k = kmin
            elif g0 == "hold" and g1 != "hold":
                kind = "release"
                # first frame where fingers opened
                k = next((t for t in range(i, j + 2) if t < n and g[t] > 0.065 * 0 + g[a] + 0.01), kmin)
            elif g0 == "hold" and g1 == "hold" or g0 == g1:
                kind = "press"; k = kmin
            else:
                kind = "other"; k = kmin
            ev.append(dict(kind=kind, i=i, j=j, k=k, xy=S[kmin, :2].tolist(), zmin=float(z[kmin]),
                           rv=S[kmin, 3:6].tolist(), g0=g0, g1=g1,
                           t=frames[kmin].get("timestamp_s")))
            i = j + 1
        else:
            i += 1
    return ev, S


def plan_from_video(ev):
    presses = [e for e in ev if e["kind"] == "press"]
    rels = [e for e in ev if e["kind"] == "release"]
    target = None
    if presses:
        p0 = presses[0]
        after = [e for e in rels if e["i"] > p0["j"]]
        if after:
            target = after[0]
    if target is None and rels:
        log("WARN no press/release-after-press; fallback to first release")
        target = rels[0]
    last_rel = rels[-1] if rels else None
    return target, last_rel


# ---------------------------------------------------------------- perception
COLORS = {"red": lambda r, g, b: (r > 150) & (g < 60) & (b < 60),
          "blue": lambda r, g, b: (b > 120) & (r < 60) & (g < 60),
          "green": lambda r, g, b: (g > 120) & (r < 80) & (b < 80)}


def find_cube_front(rb, color, near_xy=None, maxd=0.06):
    try:
        img = image_of(rb.res, rb.pkt, "front")
        if img is None:
            return None
        a = np.asarray(img.convert("RGB")).astype(int)
        r, g, b = a[..., 0], a[..., 1], a[..., 2]
        mask = COLORS[color](r, g, b)
        from scipy import ndimage
        lab, nl = ndimage.label(mask)
        sels = []
        comps = []
        for l in range(1, nl + 1):
            ys, xs = np.nonzero(lab == l)
            if len(xs) < 6:
                continue
            comps.append((len(xs), float(xs.mean()), float(ys.mean())))
        if not comps:
            return None
        sels = [{"camera": "front", "u": c[1], "v": c[2]} for c in comps[:30]]
        res = rb.ctx.call("robomme_pixel_to_position", observation_step_id=int(rb.pkt.get("step_id", 0)),
                          selections=sels, depth_window_radius=1)
        p = packet(res)
        pts = []
        for item in p.get("selections", []):
            w = item.get("position_m") if item.get("valid", True) else None
            if w is not None:
                pts.append(np.array(w, float))
        if not pts:
            log("pixel_to_position reply keys", list(p.keys())[:10], str(p)[:300])
            return None
        if near_xy is not None:
            dists = [np.linalg.norm(q[:2] - np.asarray(near_xy)) for q in pts]
            k = int(np.argmin(dists))
            if dists[k] > maxd:
                log("colour cube far from expected", np.round(pts[k], 3), dists[k])
                return None
            return pts[k]
        return pts[0]
    except Exception as ex:
        log("find_cube_front failed", repr(ex))
        return None


# ---------------------------------------------------------------- main
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import track


def wrist_color(path):
    from PIL import Image
    a = np.asarray(Image.open(path).convert("RGB")).astype(int)[128:, 64:192]
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    cnt = {c: int(f(r, g, b).sum()) for c, f in COLORS.items()}
    return max(cnt, key=cnt.get), cnt


def analyse_video(frames, ev, color):
    from PIL import Image
    files = [{(x["camera"], x["kind"]): x["path"] for x in f["files"]} for f in frames]
    src = [f.get("source_frame_index", f["frame_index"]) for f in frames]
    # cube colour for each release (wrist view just before opening)
    for e in ev:
        if e["kind"] == "release":
            k = max(0, e["i"])
            try:
                e["color"], e["cnt"] = wrist_color(WS / files[k][("wrist", "rgb")])
            except Exception as ex:
                e["color"] = None
            e["src"] = src[min(e["k"], len(src) - 1)]
    # targets
    cam = frames[0].get("camera_poses", {}).get("front", {})
    pos = np.array(cam.get("position_m", track.FRONT_POS), float)
    q = cam.get("quaternion_wxyz", track.FRONT_Q)
    Km = frames[0].get("camera_intrinsics", {}).get("front", {}).get("matrix")
    K = (Km[0][0], Km[1][1], Km[0][2], Km[1][2]) if Km else track.FRONT_K
    wd = []
    for fl in files:
        a = np.asarray(Image.open(WS / fl[("front", "rgb")]).convert("RGB"))
        D = track.detect(a)
        wd.append([(track.pix_to_plane(d[0], d[1], 0.0, pos, q, K)[:2], d[2]) for d in D])
    H = track.find_homes_w(wd)
    try:
        # add any large, unoccluded disc in the last video frames that is not yet a home
        extra = []
        for D in wd[-3:]:
            for p, sz in D:
                if sz >= 60 and (len(H) == 0 or np.linalg.norm(H - p, axis=1).min() > 0.03) and \
                        all(np.linalg.norm(p - q) > 0.03 for q in extra):
                    extra.append(p)
        if extra:
            log("WARN adding homes from last frames", np.round(extra, 3).tolist())
            H = np.vstack([H.reshape(-1, 2)] + [np.asarray(extra)])
    except Exception as ex:
        log("home augmentation failed", repr(ex))
    sw = track.fit_swaps(wd, H, src, T=33.0, tol=0.025, min_ratio=0.32)
    return H, sw, src


def main(ctx):
    rb = Robot(ctx)
    t0 = time.time()
    try:
        rb.observe()
        instr = rb.pkt.get("instruction", "")
        log("instruction:", instr)
        m = re.search(r"place the (\w+) cube", instr)
        color = m.group(1) if m else "red"
        if color not in COLORS:
            color = "red"
        frames = load_video(rb)
        ev, S = parse_events(frames)
        H, sw, src = analyse_video(frames, ev, color)
        for e in ev:
            log("event", e["kind"], "frames", e["i"], e["j"], "xy", np.round(e["xy"], 3), "z", round(e["zmin"], 3), e["g0"], e["g1"], e.get("color"), e.get("src"))
        log("homes", np.round(H, 3).tolist())
        for x in sw:
            log("swap", x)
        presses = [e for e in ev if e["kind"] == "press"]
        rels = [e for e in ev if e["kind"] == "release"]
        mine = [e for e in rels if e.get("color") in (color, None)]
        if not mine:
            mine = rels
        p0 = presses[0] if presses else None
        cand = [e for e in mine if p0 is None or e["i"] > p0["j"]]
        txy = None
        for e in cand:
            dh = np.linalg.norm(H - np.array(e["xy"]), axis=1) if len(H) else np.array([9.0])
            h0 = int(np.argmin(dh))
            if dh[h0] < 0.05:
                hf = track.follow3(h0, sw, e["src"], T=33.0)
                txy = H[hf]
                log("placement after press at home", h0, np.round(H[h0], 3), "src", e["src"], "-> final home", hf, np.round(txy, 3))
                break
            log("release not on a home, skipping", np.round(e["xy"], 3), round(float(dh[h0]), 3))
        if txy is None:
            if cand:
                txy = np.array(cand[0]["xy"]); log("WARN fallback target = release xy")
            elif rels:
                txy = np.array(rels[-1]["xy"]); log("WARN fallback target = last release")
            else:
                log("FATAL no target"); return
        last_rel = mine[-1] if mine else None
        s0 = rb.state
        if last_rel is not None:
            cube_xy = np.array(last_rel["xy"]); grasp_rv = np.array(last_rel["rv"])
        else:
            cube_xy = None; grasp_rv = s0[3:6].copy()
        cp = find_cube_front(rb, color, near_xy=cube_xy)
        if cp is not None:
            log("colour cube at", np.round(cp, 4))
            if cube_xy is None:
                cube_xy = cp[:2]
            elif np.linalg.norm(cp[:2] - cube_xy) < 0.03:
                cube_xy = 0.5 * (cube_xy + cp[:2])
        if cube_xy is None:
            log("FATAL no cube location"); return
        log("TARGET xy", np.round(txy, 4), "CUBE xy", np.round(cube_xy, 4), "grasp_rv", np.round(grasp_rv, 3), "analysis s", round(time.time() - t0, 1))
        pick_place(rb, cube_xy, grasp_rv, txy, color)
    except Ended:
        log("episode ended; complete =", rb.complete)
    except Exception as ex:
        import traceback
        log("ERROR in main:", repr(ex)); log(traceback.format_exc())
    log("done; complete =", rb.complete, "elapsed", round(time.time() - t0, 1))


ZH = 0.15
ZG = 0.017
ZP = 0.025


def pick_place(rb, cube_xy, grasp_rv, txy, color="red"):
    OPEN, CLOSE = -1, 1
    cube_xy = np.asarray(cube_xy, float)
    for attempt in range(3):
        move_to(rb, [cube_xy[0], cube_xy[1], ZH], grasp_rv, OPEN, tol=0.006)
        move_to(rb, [cube_xy[0], cube_xy[1], ZH], grasp_rv, OPEN, tol=0.003)
        move_to(rb, [cube_xy[0], cube_xy[1], ZG], grasp_rv, OPEN, tol=0.003, vmax=0.02)
        gw = gripper(rb, CLOSE, 15)
        log("attempt", attempt, "grip width after close", round(gw, 4))
        move_to(rb, [cube_xy[0], cube_xy[1], ZH], grasp_rv, CLOSE, tol=0.01)
        gw = rb.state[6] + rb.state[7]
        log("grip width lifted", round(gw, 4))
        if gw > 0.015:
            break
        # missed: open, look again for the cube near the expected place
        gripper(rb, OPEN, 10)
        rb.observe()
        cp = find_cube_front(rb, color, near_xy=cube_xy, maxd=0.12)
        if cp is not None:
            cube_xy = np.asarray(cp[:2], float)
            log("re-localised cube", np.round(cube_xy, 4))
    move_to(rb, [txy[0], txy[1], ZH], grasp_rv, CLOSE, tol=0.004)
    move_to(rb, [txy[0], txy[1], ZP], grasp_rv, CLOSE, tol=0.003, vmax=0.02)
    gripper(rb, OPEN, 12)
    s = rb.state
    move_to(rb, [s[0], s[1], ZH], grasp_rv, OPEN, tol=0.01)
    rb.step([[0, 0, 0, 0, 0, 0, -1]] * 5)
    rb.step([[0, 0, 0, 0, 0, 0, -1]] * 20)
