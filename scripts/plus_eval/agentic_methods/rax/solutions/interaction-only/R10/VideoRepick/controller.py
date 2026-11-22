"""VideoRepick controller.

1. Read the conditioning video (frame manifest); find the grasp/release from the robot
   state in the video, then track the released object through the shuffle (front-cam
   green blobs -> world xy via depth, K-means with fixed K and velocity prediction).
2. Pick and place that object twice at its current spot (closed loop, verified grasp).
3. Locate the button (gray/white, table height), close gripper and press down until the
   episode ends.
"""
import os, sys, time, traceback
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from robot import Robot  # noqa: E402
from seg import find_button, object_blobs, merge_buttons  # noqa: E402

OUT = Path("/workspace/output")
SAFE_Z = 0.15
GRASP_Z = 0.022
PLACE_Z = 0.025
T0 = time.time()


def make_log():
    try:
        OUT.mkdir(parents=True, exist_ok=True)
        fh = open(OUT / "controller_log.txt", "a")
    except Exception:
        fh = None

    def log(*a):
        s = "[%6.1fs] " % (time.time() - T0) + " ".join(str(x) for x in a)
        print(s, flush=True)
        if fh:
            fh.write(s + "\n"); fh.flush()
    return log


BUTTON = None  # world-frame button box, set once per episode


def cam_args(m, cam):
    return (m["_imgs"][(cam, "rgb")], m["_imgs"][(cam, "depth")], m["camera_intrinsics"][cam], m["camera_poses"][cam])


def button_find(m, cam, near=None):
    b = find_button(*cam_args(m, cam))
    if b is not None and near is not None and np.linalg.norm(b["xy"] - near) > 0.06:
        return None
    return b


def live_blobs(m, cam):
    return object_blobs(*cam_args(m, cam), button=BUTTON)


def wrist_offset(m):
    """camera xy minus TCP xy at the current pose."""
    return np.array(m["camera_poses"]["wrist"]["position_m"][:2]) - np.array(m["state"][:2])


def locate_object(r, est, log, gate=0.06):
    """refine object xy with the wrist camera from above; returns (xy, ok)."""
    m = r.observe()
    off = wrist_offset(m)
    # put the camera roughly above the object (camera looks straight down)
    r.move_to([est[0] - off[0] * 0.5, est[1] - off[1] * 0.5, 0.24], -1, vmax=0.6, max_steps=40)
    m = r.observe()
    best = None
    for cam in ("wrist", "front"):
        bl = live_blobs(m, cam)
        if not bl:
            continue
        d = [np.linalg.norm(b["xy"] - est) for b in bl]
        b = bl[int(np.argmin(d))]
        if min(d) < gate:
            P = b["pts"]
            top = P[P[:, 2] > P[:, 2].max() - 0.008]
            xy = top[:, :2].mean(0) if len(top) > 10 else b["xy"]
            log(f"locate {cam}: blob {b['xy'].round(3)} top {xy.round(3)} n={b['n']} d={min(d):.3f}")
            best = xy
            global NEIGHBORS
            NEIGHBORS = [o["xy"] for o in bl if o is not b]
            break
        else:
            log(f"locate {cam}: nearest blob {min(d):.3f} m away (gate {gate})")
    if best is None:
        return np.asarray(est), False
    return best, True


NEIGHBORS = []
R0 = None


def conflict(xy, axis):
    """penalty for neighbours near the finger sweep along unit `axis` (closing direction)."""
    pen = 0.0
    for n in NEIGHBORS:
        d = np.asarray(n) - xy
        along = abs(d @ axis); perp = abs(d[0] * axis[1] - d[1] * axis[0])
        if along < 0.075 and perp < 0.035:
            pen += (0.075 - along) + (0.035 - perp)
    return pen


def grasp_rot(xy, log):
    from scipy.spatial.transform import Rotation as Rr
    if R0 is None:
        return None
    p0 = conflict(xy, np.array([0.0, 1.0])); p90 = conflict(xy, np.array([1.0, 0.0]))
    yaw = 0.0 if p0 <= p90 else np.pi / 2
    log(f"grasp yaw {np.degrees(yaw):.0f} (conflict y-axis {p0:.3f}, x-axis {p90:.3f})")
    return (Rr.from_euler("z", yaw) * Rr.from_rotvec(R0)).as_rotvec()


def pick(r, xy, log):
    x, y = xy
    rot = grasp_rot(xy, log)
    r.move_to([x, y, SAFE_Z], -1, vmax=0.6, max_steps=40)
    r.move_to([x, y, 0.07], -1, rot=rot, vmax=0.4, max_steps=40)
    r.move_to([x, y, GRASP_Z], -1, rot=rot, vmax=0.25, max_steps=40, tol=0.003)
    r.hold(1, 10)
    r.observe()
    g = r.state()[6] + r.state()[7]
    r.move_to([x, y, SAFE_Z], 1, vmax=0.4, max_steps=40)
    r.observe()
    g2 = r.state()[6] + r.state()[7]
    log(f"pick at {np.round(xy,3)}: opening after close {g:.4f}, after lift {g2:.4f}, z {r.state()[2]:.3f}")
    return g2 > 0.008


def place(r, xy, log):
    x, y = xy
    r.move_to([x, y, 0.07], 1, vmax=0.4, max_steps=30)
    r.move_to([x, y, PLACE_Z], 1, vmax=0.25, max_steps=40, tol=0.003)
    r.hold(-1, 8)
    r.move_to([x, y, SAFE_Z], -1, rot=R0, vmax=0.4, max_steps=40)
    r.observe()
    log(f"placed at {np.round(xy,3)}; state {r.state()[:3].round(3)}")


def press_button(r, bxy, log):
    x, y = bxy
    r.move_to([x, y, SAFE_Z], 1, vmax=0.6, max_steps=40)
    r.move_to([x, y, 0.07], 1, vmax=0.4, max_steps=30)
    for z in [0.05, 0.04, 0.032, 0.025, 0.018, 0.012, 0.006]:
        r.move_to([x, y, z], 1, vmax=0.2, max_steps=12, tol=0.003)
        if r.ended:
            break
    r.observe() if not r.ended else None
    log(f"press attempt at {np.round(bxy,3)} ended={r.ended} complete={r.complete} z={r.state()[2]:.3f}")
    if not r.ended:
        r.move_to([x, y, 0.09], 1, vmax=0.4, max_steps=20)


def analyze(m, log):
    from tracker import analyze_video
    fs = m.get("frame_sequence") or {}
    mp = fs.get("manifest_path")
    if not mp:
        raise RuntimeError("no frame_sequence manifest in initial observation")
    res = analyze_video(mp, log=log, use_wrist=True)
    log(f"tracked target final {res['final_xy'].round(3)} min_nn {res['min_nn']:.3f} K={res['K']}")
    return res


def main(ctx):
    log = make_log()
    r = Robot(ctx, log=log)
    m = r.observe()
    log("instruction:", m.get("instruction"), "step", m.get("step_id"), "limits", m.get("episode_limits"))
    init = m
    global BUTTON, GRASP_Z, PLACE_Z, R0
    R0 = np.array(m["state"][3:6], float)
    btn = button_find(init, "front")
    if init["state"][2] > 0.15:
        btn = merge_buttons(btn, button_find(init, "wrist"))
    BUTTON = btn
    log("button coarse (front):", None if btn is None else (btn["xy"].round(3), btn["ext"].round(3), btn["n"]))
    # ---------- 1. which object
    target = None
    try:
        res = analyze(m, log)
        target = np.asarray(res["final_xy"], float)
        gz = res.get("grasp_z")
        if gz is not None and np.isfinite(gz):
            GRASP_Z = float(np.clip(gz + 0.003, 0.014, 0.03))
            PLACE_Z = max(GRASP_Z + 0.004, 0.025)
        log(f"grasp z {GRASP_Z:.3f} place z {PLACE_Z:.3f}")
    except Exception as e:
        log("video analysis failed:", repr(e)); log(traceback.format_exc())
    live = live_blobs(init, "front")
    log("live front blobs:", [(b["xy"].round(3).tolist(), b["n"]) for b in live])
    if target is None:
        if not live:
            log("no objects visible; abort"); return
        target = live[0]["xy"]
    elif live:
        d = [np.linalg.norm(b["xy"] - target) for b in live]
        log(f"nearest live front blob to target: {min(d):.3f} m")
    # ---------- button coarse from the initial front view

    # ---------- 2. pick & place twice
    obj = np.asarray(target, float)
    done = 0
    attempts = 0
    while done < 2 and attempts < 6 and not r.ended:
        attempts += 1
        if r.remaining() < 150:
            log("step budget low; go to button"); break
        xy, ok = locate_object(r, obj, log, gate=0.06 if attempts <= 2 else 0.09)
        obj = xy
        if pick(r, xy, log):
            place(r, xy, log)
            done += 1
            # object may have shifted slightly; re-locate next time
        else:
            log("grasp failed; retry")
            r.move_to([xy[0], xy[1], SAFE_Z], -1, vmax=0.5, max_steps=20)
    log(f"pick/place cycles done: {done} in {attempts} attempts; step {r.last.get('step_id')}")
    if r.ended:
        log("episode ended early; complete=", r.complete); return

    # ---------- 3. button
    bxy = None if btn is None else btn["xy"]
    m = r.observe()
    if bxy is not None:
        off = wrist_offset(m)
        r.move_to([bxy[0] - off[0], bxy[1] - off[1], 0.22], -1, vmax=0.6, max_steps=40)
        m = r.observe()
        bw = button_find(m, "wrist", near=bxy)
        if bw is not None and np.linalg.norm(bw["xy"] - bxy) < 0.06 and bw["n"] > 200:
            log(f"button wrist refine {bw['xy'].round(3)} ext {bw['ext'].round(3)} n={bw['n']}")
            bxy = (bw["lo"] + bw["hi"]) / 2 if np.all(bw["ext"] > 0.05) else bw["xy"]
        else:
            log("button wrist refine failed:", None if bw is None else (bw["xy"].round(3), bw["n"]))
    else:
        bf = button_find(m, "front") or button_find(m, "wrist")
        if bf is None:
            log("no button found"); return
        bxy = bf["xy"]
    log("button target", np.round(bxy, 3))
    for k, dxy in enumerate([(0, 0), (-0.01, 0), (0.01, 0), (0, 0.01), (0, -0.01)]):
        if r.ended or r.remaining() < 20:
            break
        press_button(r, np.asarray(bxy) + np.asarray(dxy), log)
    log(f"FINISHED ended={r.ended} complete={r.complete} step={r.last.get('step_id')} t={time.time()-T0:.1f}s")
