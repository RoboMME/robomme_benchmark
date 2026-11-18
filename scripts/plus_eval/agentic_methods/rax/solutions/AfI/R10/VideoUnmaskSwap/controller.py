"""VideoUnmaskSwap controller (v1).

Pipeline: track containers through the conditioning video (colour prior per container),
keep tracking live at HOME until containers stop drifting, then for each instructed colour
peek under the most likely container (lift aside, look with the wrist camera), and when the
colour matches restore it over the cube and lift it (the pick), then put it back.
"""
import json, os, re, sys, time, traceback
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from motion import Robot, parse, yaw_of_state            # noqa: E402
from perception import (frame_from_reply, detections, init_from_video, live_containers,  # noqa: E402
                        load_frame)
from skills import (observe_frame, choose_aside, grasp_at, carry_to, lower_release,  # noqa: E402
                    look_cube, HOME, SAFE_Z, CARRY_Z)

OUT = '/workspace/output'
COLORS = ['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'black', 'white', 'pink', 'cyan', 'brown', 'gray', 'grey']
PICK_Z = 0.18
PEEK_COST = 215    # steps per target with peek (v2 estimate)
DIRECT_COST = 90   # steps per target with direct pick


class Log:
    def __init__(self):
        os.makedirs(OUT, exist_ok=True)
        self.f = open(os.path.join(OUT, 'controller_log.txt'), 'a')
        self.t0 = time.time()

    def __call__(self, *a):
        s = f'[{time.time() - self.t0:7.1f}] ' + ' '.join(str(x) for x in a)
        print(s, flush=True)
        try:
            self.f.write(s + '\n'); self.f.flush()
        except Exception:
            pass


def parse_instruction(instr):
    """Return list of (color, all_flag) in order."""
    s = instr.lower()
    s = s.split('then', 1)[1] if 'watch' in s and 'then' in s else s
    parts = re.split(r',|followed by|and finally|finally|then', s)
    out = []
    for p in parts:
        found = [(p.find(c), c) for c in COLORS if re.search(r'\b' + c + r'\b', p)]
        if not found:
            continue
        c = min(found)[1]
        allf = bool(re.search(r'\ball\b', p)) or bool(re.search(c + r' cubes', p))
        out.append((c, allf))
    return out


def go_home(rb):
    rb.move_to(HOME, 0.0, -1, max_iter=60, tol=0.015, rtol=0.05, vmax=1.0, gain=1.5)


def wait_static(rb, tr, log, max_steps=260, still=0.003, window=10):
    hist = []
    for i in range(max_steps):
        if rb.ended:
            return
        reply = rb.ctx.call('robomme_step', actions=[[0, 0, 0, 0, 0, 0, -1]])
        rb.steps_used += 1
        f = frame_from_reply(reply)
        rb.last = {k: v for k, v in f.items() if k != 'arrays'}
        rb._lim(rb.last)
        tr.update(detections(f))
        hist.append(tr.pos.copy())
        if len(hist) > window:
            disp = np.max(np.linalg.norm(hist[-1] - hist[-1 - window], axis=1)[~tr.hidden]) if (~tr.hidden).any() else 0
            if i % 10 == 0:
                log(f'wait {i}: max disp over {window} steps {disp:.4f}, hidden {int(tr.hidden.sum())}')
            if disp < still and i >= 20:
                log(f'static after {i + 1} steps')
                return
    log('wait_static: cap reached')


def match_det(tr, k, conts, r=0.05):
    if not conts:
        return None
    d = [np.linalg.norm(c['xy'] - tr.pos[k]) for c in conts]
    j = int(np.argmin(d))
    return conts[j] if d[j] < r else None


def update_from_home(rb, tr, log):
    go_home(rb)
    f = observe_frame(rb)
    conts = live_containers(f, cams=('wrist', 'front'))
    tr.update([(c['xy'], c['n']) for c in conts])
    return f, conts


def main(ctx):
    log = Log()
    rb = Robot(ctx, log)
    meta = parse(ctx.call('robomme_observe'))
    rb.last = meta; rb._lim(meta)
    instr = meta.get('instruction', '')
    targets = parse_instruction(instr)
    log('instruction:', instr)
    log('targets:', targets, 'remaining steps', rb.remaining)
    man = meta['frame_sequence']['manifest_path']
    man = man if os.path.isabs(man) else os.path.join('/workspace', man)
    frames = json.load(open(man))['frames']
    tr, rest, cubes = init_from_video(frames, log=log)
    for f in rest:
        tr.update(detections(f))
    log('after video:', [(k, tr.colors.get(k), tr.pos[k].round(3).tolist(), round(float(tr.reacq_dist[k]), 3)) for k in range(tr.K)])
    cube_count = {}
    for c in cubes or []:
        cube_count[c['color']] = cube_count.get(c['color'], 0) + 1
    log('cube counts:', cube_count)

    wait_static(rb, tr, log)
    log('after wait:', [(k, tr.colors.get(k), tr.pos[k].round(3).tolist(), bool(tr.hidden[k])) for k in range(tr.K)],
        'remaining', rb.remaining)

    known = {}      # track -> colour actually seen under it
    done = set()    # picked tracks
    failed = {}     # track -> grasp failures

    plan = [(color, cube_count.get(color, 1) if allf else 1) for color, allf in targets]
    total_left = sum(n for _, n in plan)
    for color, need in plan:
        got = 0
        tries = 0
        while got < need and not rb.ended and tries < 12:
            tries += 1
            try:
                f, conts = update_from_home(rb, tr, log)
                cands = []
                # colours seen under containers whose prior said `color`: their prior-holders are likely swap partners
                partners = {known[j] for j in known if tr.colors.get(j) == color and known[j] != color}
                for k in range(tr.K):
                    if k in done or failed.get(k, 0) >= 2:
                        continue
                    if k in known:
                        if known[k] != color:
                            continue
                        sc = 0.0
                    elif tr.colors.get(k) == color:
                        sc = 1.0 + float(tr.reacq_dist[k])
                    elif tr.colors.get(k) in partners:
                        sc = 1.5 + float(tr.reacq_dist[k])
                    elif tr.colors.get(k) is None:
                        sc = 3.0
                    else:
                        sc = 5.0 - min(float(tr.reacq_dist[k]), 1.0)
                    if match_det(tr, k, conts) is None:
                        sc += 10
                    cands.append((sc, k))
                if not cands:
                    log(f'no candidates for {color}'); break
                cands.sort()
                sc, k = cands[0]
                c = match_det(tr, k, conts)
                if c is None:
                    log(f'track {k} not visible'); failed[k] = failed.get(k, 0) + 1; continue
                yaw = c['yaw'] if c['yaw'] is not None else 0.0
                log(f'target {color}: track {k} (prior {tr.colors.get(k)}, score {sc:.2f}) at {c["xy"].round(3)} yaw {np.rad2deg(yaw):.1f}, remaining {rb.remaining}')
                rem = rb.remaining if rb.remaining is not None else 10 ** 6
                # peek unless the budget cannot afford peeking every remaining target
                must_direct = rem < 120 or (rem < PEEK_COST * total_left + 40 and sc < 2.0)
                if must_direct:
                    log(f'direct pick (remaining {rem}, targets left {total_left})')
                    res = direct_pick(rb, c, yaw, color, log, put_back=total_left > 1)
                else:
                    res = peek_and_pick(rb, f, c, yaw, color, log, put_back=total_left > 1)
                if res is None:
                    failed[k] = failed.get(k, 0) + 1
                    continue
                seen, newxy = res
                known[k] = seen
                tr.pos[k] = newxy; tr.vel[k] = 0
                if seen == color:
                    done.add(k); got += 1; total_left -= 1
                    log(f'PICKED {color} (track {k}); task_complete={rb.task_complete} ended={rb.ended}')
            except Exception as e:
                log('ERROR', repr(e), traceback.format_exc())
                if rb.ended:
                    break
        if rb.ended:
            break
    log(f'finished: ended={rb.ended} task_complete={rb.task_complete} steps={rb.steps_used} remaining={rb.remaining}')


def direct_pick(rb, c, yaw, color, log, put_back=True):
    xy = np.array(c['xy'])
    if not grasp_at(rb, xy, yaw, log=log):
        s = rb.state(); rb.move_to([s[0], s[1], SAFE_Z], yaw, -1, max_iter=20, tol=0.015, vmax=1.0)
        return None
    s = rb.state()
    rb.move_to([s[0], s[1], PICK_Z], yaw, 1, max_iter=40, tol=0.01, vmax=0.6)
    rb.hold(1, 4)
    if rb.ended or not put_back:
        return color, xy
    rb.move_to([s[0], s[1], 0.05], yaw, 1, max_iter=40, tol=0.005, vmax=0.5)
    rb.hold(-1, 6)
    rb.move_to([s[0], s[1], SAFE_Z], yaw, -1, max_iter=25, tol=0.015, vmax=1.0, gain=1.5)
    return color, xy


def peek_and_pick(rb, f, c, yaw, color, log, put_back=True):
    """Peek under container c; if the cube matches `color`, pick it. Returns (seen_color, final_xy) or None."""
    xy = np.array(c['xy'])
    aside, clr = choose_aside(f, xy, log=log)
    if not grasp_at(rb, xy, yaw, log=log):
        s = rb.state(); rb.move_to([s[0], s[1], SAFE_Z], yaw, -1, max_iter=20, tol=0.01)
        return None
    carry_to(rb, aside)
    lower_release(rb)
    seen, cxy, f2 = look_cube(rb, xy, log=log)
    if rb.ended:
        return seen, xy
    # relocate the moved container near the aside spot
    conts = live_containers(f2, cams=('wrist', 'front'))
    near = [cc for cc in conts if np.linalg.norm(cc['xy'] - aside) < 0.05]
    if near:
        cc = min(near, key=lambda cc: np.linalg.norm(cc['xy'] - aside))
        axy = cc['xy']; ayaw = cc['yaw'] if cc['yaw'] is not None else yaw
    else:
        axy, ayaw = aside, yaw
    dest = cxy if cxy is not None else xy
    if not grasp_at(rb, axy, ayaw, log=log):
        log('regrasp failed; retry from fresh observation')
        go_home(rb)
        f3 = observe_frame(rb)
        conts = live_containers(f3, cams=('wrist', 'front'))
        near = [cc for cc in conts if np.linalg.norm(cc['xy'] - aside) < 0.07]
        if not near:
            return seen, aside
        cc = min(near, key=lambda cc: np.linalg.norm(cc['xy'] - aside))
        if not grasp_at(rb, cc['xy'], cc['yaw'] if cc['yaw'] is not None else yaw, log=log):
            s = rb.state(); rb.move_to([s[0], s[1], SAFE_Z], yaw, -1, max_iter=20, tol=0.01)
            return seen, cc['xy']
    carry_to(rb, dest, z=0.095)
    s = rb.state(); y2 = yaw_of_state(s)
    rb.move_to([s[0], s[1], 0.05], y2, 1, max_iter=25, tol=0.005, vmax=0.4)
    if seen == color:
        log('match -> pick')
        rb.move_to([s[0], s[1], PICK_Z], y2, 1, max_iter=40, tol=0.01, vmax=0.6)
        rb.hold(1, 5)
        if rb.ended or not put_back:
            return seen, dest
        rb.move_to([s[0], s[1], 0.05], y2, 1, max_iter=40, tol=0.005, vmax=0.5)
    rb.hold(-1, 6)
    rb.move_to([s[0], s[1], SAFE_Z], y2, -1, max_iter=25, tol=0.01)
    return seen, np.array(dest)
