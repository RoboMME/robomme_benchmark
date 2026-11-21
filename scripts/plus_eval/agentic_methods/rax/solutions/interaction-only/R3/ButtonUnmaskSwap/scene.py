import numpy as np
from percep import find_colored, cluster_xy
from tracker import white_world_points

COLORS = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']

def parse_targets(instr):
    s = instr.lower()
    after = s.split('then', 1)[1] if 'then' in s else s
    found = []
    for c in COLORS + ['black', 'white', 'pink', 'cyan', 'brown', 'gray', 'grey']:
        i = after.find(c + ' cube')
        if i < 0:
            i = after.find(c)
        if i >= 0:
            found.append((i, c))
    return [c for _, c in sorted(found)]

def initial_scene(rgb, depth, K, pose):
    cubes = {}
    for name, comps in find_colored(rgb, depth, K, pose, min_px=8).items():
        comps = [c for c in comps if c[0] is not None and c[0][2] > 0.01 and c[0][0] > -0.40]
        if comps:
            best = max(comps, key=lambda c: c[1])
            cubes[name] = best[0][:2]
    P0 = white_world_points(rgb, depth, K, pose, zmin=0.003)
    buttons, whites = [], []
    for c in cluster_xy(P0, min_pts=10):
        if c['n'] < 30:
            continue
        (buttons if max(c['ext']) > 0.055 else whites).append(c)
    buttons = sorted(buttons, key=lambda c: -max(c['ext']))
    whites += buttons[2:]
    buttons = buttons[:2]
    btn = []
    for b in buttons:
        Q = P0[np.linalg.norm(P0[:, :2] - b['xy'], axis=1) < 0.04]
        top = Q[Q[:, 2] > 0.024]
        xy = top[:, :2].mean(0) if len(top) > 5 else b['xy']
        btn.append(xy)
    return cubes, btn, [w['xy'] for w in whites]


def initial_cubes_fused(m, ims):
    """Coloured cubes from front + wrist RGB-D at step 0, deduplicated in world XY."""
    from percep import color_masks, components
    from geom import backproject
    allc = []
    for cam in ('front', 'wrist'):
        rgb, dep = ims.get((cam, 'rgb')), ims.get((cam, 'depth'))
        if rgb is None or dep is None:
            continue
        K = m['camera_intrinsics'][cam]['matrix']; pose = m['camera_poses'][cam]
        for name, mask in color_masks(rgb).items():
            for comp in components(mask, 6):
                pts = [backproject(x, y, dep[y, x], K, pose) for y, x in comp if dep[y, x] > 0]
                if not pts:
                    continue
                Pp = np.array(pts); Pp = Pp[(Pp[:, 2] > 0.008) & (Pp[:, 2] < 0.08) & (Pp[:, 0] > -0.40)]
                if len(Pp) < 4:
                    continue
                allc.append((name, np.median(Pp[:, :2], axis=0), len(Pp), cam))
    cubes = []
    for name, xy, n, cam in sorted(allc, key=lambda c: -c[2]):
        if any(c[0] == name and np.linalg.norm(c[1] - xy) < 0.025 for c in cubes):
            continue
        cubes.append([name, xy, n, cam])
    return cubes
