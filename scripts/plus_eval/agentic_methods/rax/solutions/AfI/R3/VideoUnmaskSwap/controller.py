"""VideoUnmaskSwap controller (v2: front+wrist fused tracking, robust cube detection).

Pipeline (each phase logs start/end step and wall time):
  P0 parse     : instruction -> ordered target colour list [(colour, all?)]
  P1 video     : read reset frame sequence, detect cubes before containers appear, track containers
                 (front RGB-D, world XY) to the first execution frame -> labelled Tracker
  P2 wait      : hold the arm parked (initial pose, out of the front view) and keep tracking each step
                 until containers stop moving (STILL_STEPS still steps) or WAIT_MAX steps
  P3 pick/place: for each target container in order: pick (approach/descend/close/lift, retries),
                 then place aside at a free spot unless it is the last target
Exits: episode_ended at any point -> return. Exceptions are logged; the loop re-observes and continues.
"""
import json, os, re, sys, time, traceback
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from robot import Robot, EpisodeEnded, log, observe          # noqa: E402
from track import video_track2, tall_obstacles                # noqa: E402
from grasp import Session, pick, place, rise, row_toward, HOVER_Z, obs_views   # noqa: E402

VERSION = 'vus-v2'
WAIT_MAX = 140          # max steps to wait for containers to stop (ep1 stopped ~step 60)
COLOURS = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
WS = Path('/workspace')
OUT = WS / 'output'


def parse_instruction(text):
    t = text.lower()
    if 'then' in t:
        t = t.split('then', 1)[1]
    specs = []
    for m in re.finditer(r'\b(red|orange|yellow|green|blue|purple|pink|magenta|violet)\s+(cubes?)\b', t):
        col = m.group(1)
        col = {'pink': 'purple', 'magenta': 'purple', 'violet': 'purple'}.get(col, col)
        before = t[max(0, m.start() - 40):m.start()]
        plural = m.group(2) == 'cubes' or ' all ' in (' ' + before.split(',')[-1] + ' ')
        specs.append((col, plural))
    return specs


def target_order(specs, labels):
    order = []
    used = set()
    for col, plural in specs:
        idx = [j for j, l in enumerate(labels) if l == col and j not in used]
        if not idx:
            log(f'plan: no container labelled {col}')
            continue
        take = idx if plural else idx[:1]
        order += [(j, col) for j in take]
        used.update(take)
    return order


def load_manifest(meta):
    seq = meta.get('frame_sequence') or {}
    p = seq.get('manifest_path')
    if not p:
        return None
    for _ in range(20):
        f = WS / p
        if f.exists():
            try:
                return json.loads(f.read_text())
            except Exception:
                pass
        time.sleep(0.5)
    return None


def write_summary(summary):
    try:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / 'vus_summary.json').write_text(json.dumps(summary, indent=1, default=str))
    except Exception:
        pass


def main(ctx):
    t_start = time.time()
    summary = {'version': VERSION, 'phases': []}
    def phase(name, **kw):
        summary['phases'].append(dict(name=name, wall=round(time.time() - t_start, 1), **kw))
        log(f'PHASE {name} wall={time.time()-t_start:.1f}s', kw)
        write_summary(summary)

    log(VERSION, 'start')
    rb = None
    try:
        res = ctx.call('robomme_observe')
        meta = json.loads(next(b['text'] for b in res['content'] if b['type'] == 'text'))
        instruction = meta.get('instruction', '')
        specs = parse_instruction(instruction)
        phase('parse', instruction=instruction, specs=specs)
        man = load_manifest(meta)
        if man is None:
            raise RuntimeError('no frame_sequence manifest')
        tr, info = video_track2(WS, man, log=log)
        labels = tr.labels
        order = target_order(specs, labels)
        phase('video', labels=labels, centres=np.round(tr.C, 3).tolist(),
              init_err=round(float(info['init_err']), 4), order=order,
              cubes=[(cc.get('colour'), np.round(cc.get('xy'), 3).tolist(), cc.get('cams')) for cc in info.get('cubes', [])])
        rb = Robot(ctx)
        S = Session(rb, tr, tr.t)
        try:
            S.obstacles = tall_obstacles(obs_views(rb.obs))
            log(f'obstacles: {len(S.obstacles)} tall points')
        except Exception as e:
            log('obstacles failed', repr(e))
    except EpisodeEnded:
        phase('ended_early'); return
    except Exception as e:
        log('FATAL setup', repr(e)); traceback.print_exc()
        phase('setup_failed', err=repr(e))
        return

    try:
        S.track()
        # P2: wait parked until containers stop moving
        n = 0
        while not S.static and n < WAIT_MAX:
            S.step1([0, 0, 0, 0, 0, 0, -1])
            n += 1
            if n % 20 == 0:
                log(f'wait n={n} still_run={S.still_run} C={np.round(tr.C,3).tolist()}')
        phase('wait', steps=n, static=S.static, step=rb.obs.step_id, centres=np.round(tr.C, 3).tolist())

        # P3: picks
        for k, (idx, col) in enumerate(order):
            last = k == len(order) - 1
            for attempt in range(2):
                try:
                    r = pick(S, idx)
                except EpisodeEnded:
                    raise
                except Exception as e:
                    log('pick error', repr(e)); traceback.print_exc()
                    r = {'ok': False, 'why': repr(e)}
                    try:
                        rb.refresh(); S.track()
                        rise(S, HOVER_Z, -1)
                    except EpisodeEnded:
                        raise
                    except Exception:
                        pass
                if r.get('ok'):
                    break
            phase(f'pick_{k}_{col}', idx=idx, result=r, step=rb.obs.step_id)
            if not r.get('ok'):
                continue
            if not last:
                try:
                    pr = place(S)
                except EpisodeEnded:
                    raise
                except Exception as e:
                    log('place error', repr(e)); traceback.print_exc()
                    pr = {'ok': False, 'why': repr(e)}
                phase(f'place_{k}', result=pr, step=rb.obs.step_id)
        # all targets handled but episode still running: hold the last one up
        log('all targets done; holding')
        for _ in range(10):
            rb.step([[0, 0, 0, 0, 0, 0, 1]] * 20)
    except EpisodeEnded as e:
        phase('episode_ended', msg=str(e), step=rb.obs.step_id if rb else None)
        return
    except Exception as e:
        log('FATAL loop', repr(e)); traceback.print_exc()
        phase('loop_failed', err=repr(e))
    finally:
        write_summary(summary)
