import numpy as np
from percep import depth_to_world, load_depth, load_rgb, blobs_from_points, detect_cubes
from tracker import Tracker

TARGET_COLORS = ["red", "green", "blue", "yellow", "orange", "purple", "black", "white", "pink", "cyan", "brown", "gray"]

def parse_order(instruction):
    s = instruction.lower()
    i = s.find("then")
    s2 = s[i:] if i >= 0 else s
    found = []
    for c in TARGET_COLORS:
        k = 0
        while True:
            j = s2.find(c + " cube", k)
            if j < 0: break
            found.append((j, c)); k = j + 1
    found.sort()
    return [c for _, c in found]

def frame_geo(frame, root="/workspace/"):
    fp = {(x['camera'], x['kind']): x['path'] for x in frame['files']}
    d = load_depth(root + fp[('front', 'depth')])
    rgb = load_rgb(root + fp[('front', 'rgb')])
    P = depth_to_world(d, frame['camera_intrinsics']['front']['matrix'], frame['camera_poses']['front'])
    return rgb, P, d > 0

def container_blobs(P, V):
    return [b for b in blobs_from_points(P, V, zhi=0.085, xlim=(-0.45, 0.30)) if b['n'] >= 30]

def top_center(P, V, xy, r=0.045):
    """refine container centre from its top-surface points"""
    m = V & (np.linalg.norm(P[..., :2] - np.array(xy), axis=-1) < r) & (P[..., 2] > 0.064) & (P[..., 2] < 0.085)
    if m.sum() < 10:
        return np.array(xy), 0
    pts = P[m]
    return np.median(pts[:, :2], axis=0), int(m.sum())
