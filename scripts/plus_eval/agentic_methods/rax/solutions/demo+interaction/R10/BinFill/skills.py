"""Task skills: pick (possibly moving) cube with wrist refinement, drop in bin, press button."""
import numpy as np
from percep import find_cubes, find_bin, find_button

APPROACH_Z = 0.10

def refine_with_wrist(rb, color, guess, max_dist=0.06):
    m, im = rb.observe()
    cs = [c for c in find_cubes(m, im, "wrist", colors=(color,), min_px=20)]
    if not cs:
        return None
    cs.sort(key=lambda c: np.linalg.norm(c["xyz"][:2]-guess[:2]))
    best = cs[0]
    if np.linalg.norm(best["xyz"][:2]-guess[:2]) > max_dist:
        return None
    return best

def pick(rb, color, xyz, t_obs, vel, log=print):
    """Cube seen at xyz at step t_obs moving with vel (m/step). True if held after lift."""
    xyz = np.array(xyz, float); vel = np.array(vel, float); vel[2] = 0
    rb.move_to([xyz[0], xyz[1], APPROACH_Z], grip=-1, vmax=1.0, tol=0.012, vel=vel, t0=t_obs, max_iter=10)
    guess = xyz + vel*(rb.sid()-t_obs)
    r = refine_with_wrist(rb, color, guess)
    if r is None:
        log(f"pick: wrist lost {color} near {guess.round(3)}")
        return False
    xyz = r["xyz"]; t_obs = rb.sid(); ztop = r["ztop"]
    log(f"pick: refine {color} -> {xyz.round(3)} (pred {guess.round(3)}) ztop={ztop:.3f} npx={r['npx']} step={t_obs}")
    gz = max(0.012, ztop - 0.022)
    rb.move_to([xyz[0], xyz[1], gz+0.025], vmax=0.8, tol=0.007, vel=vel, t0=t_obs, max_iter=6, lead=2)
    rb.move_to([xyz[0], xyz[1], max(0.014, gz-0.006)], vmax=0.4, tol=0.006, max_iter=4, vel=vel, t0=t_obs, lead=2)
    log(f"pick: at {rb.state[:3].round(3)} target {(np.array([xyz[0], xyz[1], gz])+vel*(rb.sid()-t_obs)).round(3)} step={rb.sid()}")
    vy = vel/0.05
    rb.grip = 1
    rb.step([[vy[0], vy[1], 0, 0, 0, 0, 1]]*7)
    w = rb.width()
    rb.move_to([rb.state[0], rb.state[1], APPROACH_Z+0.02], vmax=1.0, tol=0.02, max_iter=4)
    w2 = rb.width()
    log(f"pick: width after close {w:.4f}, after lift {w2:.4f} step={rb.sid()}")
    return 0.012 < w2 < 0.06

def drop_in_bin(rb, bin_xyz, log=print):
    rb.move_to([bin_xyz[0], bin_xyz[1], 0.15], vmax=1.0, tol=0.015)
    w = rb.width()
    held = 0.012 < w < 0.06
    try:
        m, im = rb.observe()
        wb = find_bin(m, im, "wrist")
        if wb is not None and wb["npx"] > 150:
            c = (wb["lo"] + wb["hi"]) / 2
            if np.linalg.norm(c[:2] - np.array(bin_xyz[:2])) < 0.04:
                log(f"drop: wrist bin centre {c.round(3)} (front {np.round(bin_xyz[:2],3)})")
                bin_xyz = [c[0], c[1], bin_xyz[2]]
    except Exception as e:
        log("drop: wrist bin err", repr(e))
    rb.move_to([bin_xyz[0], bin_xyz[1], 0.12], vmax=0.6, tol=0.008, max_iter=4)
    rb.gripper(-1, 5)
    log(f"drop: held_before={held} w={w:.4f}")
    return held
