import numpy as np
from perceive import refine

def wrap45(a):
    return (a + np.pi / 4) % (np.pi / 2) - np.pi / 4

def angdiff(a, b):
    return (a - b + np.pi) % (2 * np.pi) - np.pi

def choose_yaw(c, yaw_edge, others, cur_yaw=0.0):
    """Two edge-aligned options (yaw, yaw+90). Fingers close along (sin y, -cos y).
    Prefer finger positions far from neighbouring containers."""
    best = None
    for y in (wrap45(yaw_edge), wrap45(yaw_edge) + np.pi / 2, wrap45(yaw_edge) - np.pi / 2):
        if abs(y) > np.radians(100): continue
        ax = np.array([np.sin(y), -np.cos(y)])
        fingers = [c + 0.045 * ax, c - 0.045 * ax]
        dmin = min([np.linalg.norm(f - o) for f in fingers for o in others] + [1.0])
        score = dmin - 0.01 * abs(angdiff(y, cur_yaw))
        if best is None or score > best[0]: best = (score, y, dmin)
    return best[1], best[2]

def descend(rb, target, yaw, log, max_steps=60):
    """Move down; return True if reached, False if blocked."""
    hist = []
    for k in range(max_steps):
        if rb.ended: return False
        if rb.move_to(target, yaw=yaw, tol=0.004, max_steps=1, speed=0.5):
            return True
        hist.append(rb.pos[2])
        if len(hist) > 8 and hist[-8] - hist[-1] < 0.002 and rb.pos[2] > target[2] + 0.006:
            log(f'  descent blocked at z={rb.pos[2]:.3f} xy={np.round(rb.pos[:2],3)}')
            return False
    return np.linalg.norm(target - rb.pos) < 0.008

def pick_container(rb, xy, others, log=print, z_grasp=0.04, z_lift=0.15, hover=0.15, tries=3):
    """Grasp the container near xy and lift it. Returns dict (held, xy, yaw)."""
    xy = np.asarray(xy, float)
    rb.grip = -1.0
    if rb.state[6] + rb.state[7] < 0.07: rb.set_gripper(-1.0, n=8)
    rb.move_to([xy[0], xy[1], hover], max_steps=60)
    for attempt in range(tries):
        if rb.ended: break
        r = refine(rb.meta, rb.imgs, xy)
        if r is None:
            log(f'  refine failed at {np.round(xy,3)}'); r = dict(xy=xy, yaw=0.0)
        c = np.asarray(r['xy'])
        if np.linalg.norm(c - xy) > 0.04: c = xy
        oth = [o for o in others if np.linalg.norm(np.asarray(o) - c) > 0.03]
        yaw, dmin = choose_yaw(c, r['yaw'], oth, rb.yaw())
        if attempt > 0: yaw = yaw + (np.radians(8) if attempt == 1 else -np.radians(8))
        log(f'pick try {attempt}: track {np.round(xy,3)} refined {np.round(c,3)} yaw {np.degrees(yaw):.0f} finger-clearance {dmin:.3f}')
        rb.move_to([c[0], c[1], hover], yaw=yaw, max_steps=60)
        r2 = refine(rb.meta, rb.imgs, c)
        if r2 is not None and np.linalg.norm(np.asarray(r2['xy']) - c) < 0.02:
            c = np.asarray(r2['xy'])
        rb.move_to([c[0], c[1], 0.11], yaw=yaw, max_steps=40)
        ok = descend(rb, np.array([c[0], c[1], z_grasp]), yaw, log)
        if not ok:
            rb.move_to([c[0], c[1], hover], yaw=yaw, max_steps=40)
            xy = c
            continue
        rb.set_gripper(1.0, n=10)
        w = rb.state[6] + rb.state[7]
        if w < 0.02:
            log(f'  gripper closed on nothing (w={w:.3f}); reopen and retry')
            rb.set_gripper(-1.0, n=8); rb.move_to([c[0], c[1], hover], yaw=yaw, max_steps=40); xy = c
            continue
        rb.move_to([c[0], c[1], z_lift], yaw=yaw, speed=0.6, max_steps=60)
        w2 = rb.state[6] + rb.state[7]
        held = 0.02 < w2 < 0.075
        log(f'  grasp width {w:.3f} after lift {w2:.3f} held={held} step={rb.meta.get("step_id")} ended={rb.ended}')
        if held or rb.ended:
            return dict(xy=c, yaw=yaw, held=held)
        rb.set_gripper(-1.0, n=8); xy = c
    return dict(xy=xy, yaw=0.0, held=False)

def place_back(rb, c, yaw, z_grasp=0.04):
    if rb.ended: return
    rb.move_to([c[0], c[1], z_grasp + 0.006], yaw=yaw, speed=0.5, max_steps=60)
    rb.set_gripper(-1.0, n=8)
    rb.move_to([c[0], c[1], 0.15], yaw=yaw, max_steps=40)
