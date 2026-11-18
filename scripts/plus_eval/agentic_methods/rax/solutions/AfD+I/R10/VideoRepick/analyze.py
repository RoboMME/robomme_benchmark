import numpy as np
from track import object_points, clusters, Tracker

def grasp_events(frames):
    """Find pick (gripper closes low) and place (gripper reopens low) in video states."""
    st = np.array([f['state'] for f in frames])
    w = st[:, 6] + st[:, 7]
    z = st[:, 2]
    pick = place = None
    for i in range(1, len(st)):
        if pick is None and w[i] < 0.06 and z[i] < 0.08:
            pick = i
        if pick is not None and i > pick and w[i] > 0.06 and z[i] < 0.12:
            place = i
            break
    return pick, place, st

def analyze(frames, log=print):
    vid = [f for f in frames if f['phase'] == 'video_demo']
    pick, place, st = grasp_events(vid)
    log(f'pick={pick} place={place}')
    if pick is None:
        return None
    pick_xy = st[pick, :2].copy()
    # find grasp point: gripper position when fully closed (a few frames later)
    if place is None:
        place = pick
    place_xy = st[place, :2].copy()
    # start tracking once EE is high enough after place
    start = place
    while start < len(vid) - 1 and st[start, 2] < 0.12:
        start += 1
    xy, _ = object_points(vid[start])
    cl = clusters(xy)
    cents = np.array([c for c, n, _ in cl])
    log(f'start={start} clusters={np.round(cents,3).tolist()}')
    d = np.linalg.norm(cents - place_xy, axis=1)
    tgt = int(d.argmin())
    log(f'target idx {tgt} dist to place {d[tgt]:.3f}')
    tr = Tracker(cents)
    traj = [tr.c[tgt].copy()]
    for i in range(start + 1, len(vid)):
        xy, _ = object_points(vid[i])
        tr.update(xy)
        traj.append(tr.c[tgt].copy())
    return dict(start_centers=cents, pick=pick, place=place, start=start, place_xy=place_xy, pick_xy=pick_xy,
                final=tr.c[tgt].copy(), all_final=tr.c.copy(), tgt=tgt, traj=np.array(traj))
