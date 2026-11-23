"""Infer the goal target from the conditioning video."""
import numpy as np
from itertools import combinations
from swaps import infer_segment
from geom import Cam
from perception import (load_frames, read_rgb, read_depth, world_points, detect_targets,
                        wrist_color, parse_instruction)

OFF = 0.025     # detection farther than this from every slot => moving
PAIR_TOL = 0.03

def analyze(manifest_path, instruction, log=print, root='/workspace'):
    frames = load_frames(manifest_path, root)
    last = frames[-1]['meta']
    cam = Cam(last['camera_intrinsics']['front'], last['camera_poses']['front'])
    color, nth = parse_instruction(instruction)
    log(f'instruction color={color} nth={nth} frames={len(frames)}')
    dets, states, wrist = [], [], []
    for fr in frames:
        rgb = read_rgb(fr['files'][('front', 'rgb')])
        dp = fr['files'].get(('front', 'depth'))
        P = world_points(cam, read_depth(dp)) if dp else None
        dets.append(detect_targets(rgb, P, cam))
        states.append(np.array(fr['meta']['state'], float))
    vid = [i for i, fr in enumerate(frames) if fr['meta'].get('phase') == 'video_demo']
    # ---- slots: final frame detections (+ frame 0)
    slots = [d[0] for d in dets[-1]]
    for p, n in dets[0]:
        if all(np.linalg.norm(p - s) > OFF for s in slots):
            slots.append(p)
    slots = np.array(slots)
    log('slots ' + str(np.round(slots * 100, 1).tolist()))
    # ---- grasp / release events
    events = []
    holding = None
    for i in range(1, len(frames)):
        g0 = states[i - 1][6] + states[i - 1][7]; g1 = states[i][6] + states[i][7]
        ee = states[i]
        if g0 > 0.07 and g1 <= 0.07 and ee[2] < 0.08:
            # grasp: wrist color over next frames
            cnts = {}
            for j in range(i, min(i + 4, len(frames))):
                wf = frames[j]['files'].get(('wrist', 'rgb'))
                if wf:
                    c, cc = wrist_color(read_rgb(wf))
                    for k, v in cc.items():
                        cnts[k] = cnts.get(k, 0) + v
            c = max(cnts, key=cnts.get) if cnts and max(cnts.values()) > 30 else None
            holding = c
            events.append(('grasp', i, c, ee[:3].copy()))
        elif g0 <= 0.07 and g1 > 0.07:
            pos = states[i - 1][:3]
            events.append(('release', i, holding, pos.copy()))
            holding = None
    for e in events:
        log(f'event {e[0]} f{e[1]} {e[2]} {np.round(e[3], 3).tolist()}')
    # ---- motion segments
    moving = []
    for i, D in enumerate(dets):
        off = [np.min(np.linalg.norm(slots - p, axis=1)) for p, n in D]
        moving.append(bool(off) and max(off) > OFF)
    segs = []
    i = 0
    while i < len(moving):
        if moving[i]:
            j = i
            while j + 1 < len(moving) and moving[j + 1]:
                j += 1
            segs.append((i, j)); i = j + 1
        else:
            i += 1
    # merge fragments of one swap split by a single rest frame (a swap lasts ~1.6 s)
    ts = [float(fr['meta'].get('timestamp_s', i * 0.05)) for i, fr in enumerate(frames)]
    merged = []
    for a, b in segs:
        if merged and a - merged[-1][1] <= 2 and ts[b] - ts[merged[-1][0]] <= 1.9:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    if len(merged) != len(segs):
        log(f'merged segments {len(segs)} -> {len(merged)}')
    segs = merged
    swaps = []
    for a, b in segs:
        a2, b2 = max(0, a - 1), min(len(dets) - 1, b + 1)
        chosen, _ = infer_segment(dets, slots, a2, b2, log=log)
        if chosen:
            log(f'swap seg {a}-{b} chosen={chosen}')
        swaps.append((b, chosen))
    # ---- identity bookkeeping
    ns = len(slots)
    who = list(range(ns))
    timeline = sorted([(e[1], 'ev', e) for e in events] + [(b + 0.5, 'sw', ch) for b, ch in swaps], key=lambda x: x[0])
    placements = []
    for t, kind, x in timeline:
        if kind == 'sw':
            for i1, i2 in x:
                who[i1], who[i2] = who[i2], who[i1]
        else:
            typ, i, c, pos = x
            if typ != 'release' or pos[2] > 0.08:
                continue
            dd = np.linalg.norm(slots - pos[:2], axis=1)
            k = int(np.argmin(dd))
            if dd[k] < 0.04:
                placements.append((i, c, k, who[k]))
                log(f'placement f{i} color={c} slot={k} identity={who[k]}')
    mine = [p for p in placements if p[1] == color] or placements
    if not mine:
        return None, dict(slots=slots, placements=placements)
    idx = nth - 1 if nth and nth > 0 else len(mine) - 1
    idx = min(idx, len(mine) - 1)
    ident = mine[idx][3]
    final_slot = who.index(ident)
    log(f'answer identity={ident} final_slot={final_slot} xy={np.round(slots[final_slot], 4).tolist()}')
    return slots[final_slot], dict(slots=slots, placements=placements, who=who)
