import numpy as np
from geom import depth_to_world

def tip_offset(depth_m, K, pose, tcp, r=0.03, table_z=0.03):
    """Estimate (tcp_z - lowest tool point z) from a front depth image.
    Points within horizontal radius r of the TCP, above the table, below TCP."""
    P = depth_to_world(depth_m, K, pose)
    x, y, z = P[..., 0], P[..., 1], P[..., 2]
    m = (np.hypot(x - tcp[0], y - tcp[1]) < r) & (z > table_z) & (z < tcp[2] + 0.02)
    if m.sum() < 3:
        return None, int(m.sum())
    zs = np.sort(z[m])
    low = float(np.median(zs[:3]))
    return float(tcp[2] - low), int(m.sum())


def wrist_tool_lowest(depth_m, K, pose, tcp_z, near=0.3, min_px=150):
    """Lowest world z of the tool seen by the wrist camera (pixels nearer than `near` m),
    returned relative to the TCP z.  ep1 (normal stick): -0.001; ep2 live (short stick): +0.063."""
    P = depth_to_world(depth_m, K, pose)
    m = (depth_m > 0) & (depth_m < near)
    if m.sum() < min_px:
        return None
    z = np.sort(P[..., 2][m])
    return float(np.median(z[:5]) - tcp_z)
