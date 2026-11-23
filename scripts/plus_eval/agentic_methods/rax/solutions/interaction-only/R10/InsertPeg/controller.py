"""InsertPeg controller v1: track the expert EE trajectory carried in this episode's
conditioning video (per-frame `state`), with per-step closed-loop feedback."""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R

WS = Path("/workspace")
OUT = WS / "output"
LOG = None

def log(*a):
    global LOG
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    try:
        if LOG is None:
            OUT.mkdir(parents=True, exist_ok=True)
            LOG = open(OUT / "controller_log.txt", "a")
        LOG.write(s + "\n"); LOG.flush()
    except Exception:
        pass

def meta(reply):
    return json.loads(next(b["text"] for b in reply["content"] if b.get("type") == "text"))

def _video_frames(path):
    try:
        man = json.loads(Path(path).read_text())
    except Exception:
        return []
    frames = [f for f in man.get("frames", []) if f.get("phase") == "video_demo" and f.get("state")]
    frames.sort(key=lambda f: f["frame_index"])
    return frames

def load_video(packet):
    cands = []
    if os.environ.get("INSERTPEG_MANIFEST"):
        cands.append(Path(os.environ["INSERTPEG_MANIFEST"]))
    mp = (packet.get("frame_sequence") or {}).get("manifest_path")
    if mp:
        cands.append(WS / mp)
    try:
        found = sorted((WS / "runtime/autonomous_controller/runs").glob("*/frames/sequence_000000/manifest.json"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        cands.extend(found[:3])
    except Exception:
        pass
    for c in cands:
        fr = _video_frames(c)
        if fr:
            log("video manifest:", c)
            return fr
    return []

POS_GAIN = 2.5
ROT_GAIN = 2.0

def make_row(cur, tgt, grip):
    cur = np.asarray(cur, float); tgt = np.asarray(tgt, float)
    dp = (tgt[:3] - cur[:3]) * POS_GAIN / 0.05
    rc = R.from_rotvec(cur[3:6]); rt = R.from_rotvec(tgt[3:6])
    dr = (rt * rc.inv()).as_rotvec() * ROT_GAIN / 0.5
    row = np.concatenate([np.clip(dp, -1, 1), np.clip(dr, -1, 1), [grip]])
    return [float(x) for x in row]

def errs(cur, tgt):
    cur = np.asarray(cur, float); tgt = np.asarray(tgt, float)
    pe = float(np.linalg.norm(tgt[:3] - cur[:3]))
    re = float(np.linalg.norm((R.from_rotvec(tgt[3:6]) * R.from_rotvec(cur[3:6]).inv()).as_rotvec()))
    return pe, re

class Robot:
    def __init__(self, ctx):
        self.ctx = ctx; self.ended = False; self.complete = False; self.state = None; self.remaining = None
    def observe(self):
        m = meta(self.ctx.call("robomme_observe"))
        self.state = m["state"]; self._lim(m)
        return m
    def _lim(self, m):
        lim = m.get("episode_limits") or {}
        self.remaining = lim.get("remaining_episode_steps", self.remaining)
        if m.get("episode_ended"):
            self.ended = True
        for s in (m.get("transition") or {}).get("steps", []) or []:
            if s.get("task_complete"):
                self.complete = True
            if s.get("episode_ended"):
                self.ended = True
        if m.get("task_complete"):
            self.complete = True
    def step(self, row):
        if self.ended:
            return None
        m = meta(self.ctx.call("robomme_step", actions=[row]))
        self.state = m["state"]; self._lim(m)
        return m

def gripper_commands(frames):
    f = np.array([fr["state"][6] + fr["state"][7] for fr in frames])
    n = len(f)
    g = np.full(n, -1.0)
    # command close from the frame where fingers start to drop meaningfully (and while closed)
    for i in range(n):
        nxt = f[min(i + 2, n - 1)]
        if f[i] < 0.07 or nxt < f[i] - 0.004:
            g[i] = 1.0
    return g

def unit(v):
    v = np.asarray(v, float); n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v

def find_preinsert(states, dist=0.05):
    F = states[-1][:3]
    k = len(states) - 1
    while k > 0 and np.linalg.norm(states[k][:3] - F) < dist:
        k -= 1
    return k

def locate_hole(m, F, d, lat, s_lim=0.10, s_min=None):
    """From a wrist observation packet, find the hole opening centre (lateral, z offsets relative to F)."""
    from PIL import Image
    from geom import backproject
    files = {}
    # depth image is an attachment of the reply; use saved artifact path via frame files when available
    dep = m.get("_depth_wrist")
    K = m["camera_intrinsics"]["wrist"]["matrix"]; pose = m["camera_poses"]["wrist"]
    P, valid = backproject(dep, K, pose)
    c = np.asarray(pose["position_m"], float)
    rel = P - F
    s = rel @ d; l = rel @ lat; z = P[..., 2]
    zF = F[2]
    if s_min is None:
        s_min = float((np.asarray(m["state"][:3]) - F) @ d) + 0.09
    win = valid & (np.abs(l) < 0.04) & (np.abs(z - zF) < 0.025) & (s > s_min) & (s < 0.15)
    if win.sum() < 30:
        return None, "few window pts %d" % win.sum()
    sv = s[win]
    hist, edges = np.histogram(sv, bins=np.arange(s_min, 0.15, 0.001))
    j = int(np.argmax(hist)); s_face = 0.5 * (edges[j] + edges[j + 1])
    # refine face as median of points near the mode
    near = sv[np.abs(sv - s_face) < 0.002]
    s_face = float(np.median(near)) if len(near) else s_face
    hole = win & (s > s_face + 0.005)
    nh = int(hole.sum())
    if nh < 8:
        return None, "few hole pts %d s_face %.4f" % (nh, s_face)
    Ph = P[hole]
    rc = (c - F) @ d
    t = (s_face - rc) / ((Ph - c) @ d - 0.0 + 1e-12)
    O = c + t[:, None] * (Ph - c)
    ol = (O - F) @ lat; oz = O[:, 2] - zF
    lc = 0.5 * (np.percentile(ol, 3) + np.percentile(ol, 97))
    zc = 0.5 * (np.percentile(oz, 3) + np.percentile(oz, 97))
    info = dict(s_face=round(s_face, 4), n_hole=nh, l_rng=[round(float(np.percentile(ol, 3)), 4), round(float(np.percentile(ol, 97)), 4)],
                z_rng=[round(float(np.percentile(oz, 3)), 4), round(float(np.percentile(oz, 97)), 4)], n_face=int(len(near)))
    return (float(lc), float(zc)), info

def depth_from_reply(reply, m, camera):
    import base64, io
    from PIL import Image
    for a in m.get("attachments", []):
        if a.get("camera") == camera and a.get("kind") == "depth":
            b = reply["content"][a["content_index"]]
            return np.array(Image.open(io.BytesIO(base64.b64decode(b["data"]))))
    return None

def observe_full(rob):
    reply = rob.ctx.call("robomme_observe")
    m = meta(reply)
    m["_depth_wrist"] = depth_from_reply(reply, m, "wrist")
    rob.state = m["state"]
    return m

def goto(rob, tgt, n, grip, tol=0.0015, rtol=0.01):
    for _ in range(n):
        if rob.ended: return
        pe, re = errs(rob.state, tgt)
        if pe < tol and re < rtol: return
        rob.step(make_row(rob.state, tgt, grip))

def main(ctx):
    t0 = time.time()
    rob = Robot(ctx)
    pkt = rob.observe()
    log("instruction:", pkt.get("instruction"), "state:", pkt["state"])
    frames = load_video(pkt)
    log("video frames with state:", len(frames))
    if not frames:
        log("no video states; abort"); return
    states = [np.array(fr["state"], float) for fr in frames]
    grips = gripper_commands(frames)
    close_idx = int(np.argmax(grips > 0)) if (grips > 0).any() else None
    k_pre = find_preinsert(states, 0.05)
    F = states[-1][:3].copy()
    d = unit(F - states[k_pre][:3]); d[2] = 0.0; d = unit(d)
    lat = unit(np.cross([0, 0, 1.0], d))
    log("close index", close_idx, "preinsert index", k_pre, "F", F.round(4), "d", d.round(3))
    # phase 1: follow the expert trajectory up to the pre-insert frame
    i = 1; wait = 0
    while i <= k_pre and not rob.ended:
        if rob.remaining is not None and rob.remaining <= 2:
            break
        tgt = states[i]; g = grips[i]
        pe, re = errs(rob.state, tgt)
        if close_idx is not None and i == close_idx and (pe > 0.004 or re > 0.05) and wait < 25:
            rob.step(make_row(rob.state, tgt, -1.0)); wait += 1; continue
        if pe > 0.03 and wait < 40:
            rob.step(make_row(rob.state, tgt, g)); wait += 1; continue
        wait = 0
        rob.step(make_row(rob.state, tgt, g))
        if i % 40 == 0:
            log(f"i={i} pe={pe:.4f} re={re:.3f} state={[round(x,4) for x in rob.state]} rem={rob.remaining}")
        i += 1
    if rob.ended:
        log("ended early"); return
    gk = grips[k_pre]
    fin = states[-1].copy()
    sv = [float((st[:3] - F) @ d) for st in states]
    # video view frame: last frame before pre-insert with s <= -0.062
    k_view = k_pre
    for j in range(k_pre, 0, -1):
        if sv[j] <= -0.062:
            k_view = j; break
    log("k_view", k_view, "s", round(sv[k_view], 4))
    shift = np.zeros(3)
    try:
        from PIL import Image
        from holestat import hole_stats
        fr = frames[k_view]
        files = {(x["camera"], x["kind"]): x["path"] for x in fr["files"]}
        vdep = np.array(Image.open(WS / files[("wrist", "depth")]))
        st_v = hole_stats(vdep, fr["camera_intrinsics"]["wrist"]["matrix"], fr["camera_poses"]["wrist"], F, d, lat, fr["state"][:3])
        log("video hole stats", st_v)
        goto(rob, states[k_view], 40, gk, tol=0.0006, rtol=0.005)
        m = observe_full(rob)
        st_o = hole_stats(m["_depth_wrist"], m["camera_intrinsics"]["wrist"]["matrix"], m["camera_poses"]["wrist"], F, d, lat, m["state"][:3])
        log("live hole stats", st_o, "tcp", [round(x, 4) for x in m["state"][:3]])
        if st_v and st_o and "floor" in st_v and "floor" in st_o:
            dl = 0.5 * ((st_o["l_lo"] - st_v["l_lo"]) + (st_o["l_hi"] - st_v["l_hi"]))
            dz = st_o["floor"] - st_v["floor"]
            if abs(dl) < 0.02 and abs(dz) < 0.02:
                shift = dl * lat + np.array([0, 0, dz])
    except Exception as e:
        log("hole stats error", repr(e))
    log("applied shift", shift.round(4))
    sl = float(shift @ lat)
    if abs(sl) >= 0.003:
        ext = [(0.35 * sl, 0), (0.7 * sl, 0), (0, -0.002), (0.35 * sl, -0.002)]
    else:
        ext = [(0.002, 0), (-0.002, 0), (0, -0.002), (0, 0.002)]
    pattern = [(0, 0)] + ext + [(0.0012, 0), (-0.0012, 0), (0, 0.0012), (0, -0.0012)]
    if np.linalg.norm(shift) > 0.003:
        pattern.append((-sl, -float(shift[2])))  # the expert's unshifted line
    t = fin.copy()
    for attempt, (dl, dz) in enumerate(pattern):
        if rob.ended or (rob.remaining is not None and rob.remaining < 45):
            break
        sh = shift + dl * lat + np.array([0, 0, dz])
        back = fin.copy(); back[:3] = F - d * 0.058 + sh
        goto(rob, back, 30, gk, tol=0.0006, rtol=0.005)
        cur = (np.asarray(rob.state[:3]) - F - sh) @ d
        best = cur; stall = 0; lead = 0.004; td = cur
        while not rob.ended and (rob.remaining is None or rob.remaining > 2):
            td = min(td + 0.0015, cur + lead, 0.006)
            t = fin.copy(); t[:3] = F + d * td + sh
            rob.step(make_row(rob.state, t, gk))
            cur = (np.asarray(rob.state[:3]) - F - sh) @ d
            if cur > best + 0.0004:
                best = cur; stall = 0
            else:
                stall += 1
                if stall == 6:
                    lead = 0.012  # push harder once in contact
            if cur >= 0.003 or stall > 14:
                break
        log(f"attempt {attempt} offs=({dl},{dz}) depth {cur:.4f} best {best:.4f} rem={rob.remaining}")
        if best >= -0.012:
            # inserted past the entrance: finish pushing to the expert's final depth
            for _ in range(30):
                if rob.ended: break
                t = fin.copy(); t[:3] = F + d * 0.004 + sh
                rob.step(make_row(rob.state, t, gk))
            break
    for _ in range(10):
        if rob.ended: break
        rob.step(make_row(rob.state, t, gk))
    cur = (np.asarray(rob.state[:3]) - F - shift) @ d
    log(f"done: insertion depth rel final {cur:.4f} state={[round(x,4) for x in rob.state]} ended={rob.ended} complete={rob.complete} t={time.time()-t0:.1f}s")
