"""VideoUnmask controller (RAX / RoboMME-Plus).

Pipeline (transition table)
  S0 observe            -> S1 (always; on failure retry observe x3 then abort)
  S1 calibrate cameras  -> S2 (tool-checked camera conventions; default on failure)
  S2 video analysis     -> S3 (cube clusters per colour in world xy; front + wrist video)
  S3 plan               -> S4 (instruction -> ordered colour groups -> container targets)
  S4 overview           -> S5 (containers at step 0 from wrist/front depth; assign cubes)
  S5 per target: approach -> look -> look close -> align -> descend -> close -> lift
                 -> (not last) put back, open, rise -> next target
                 grasp miss or blocked descent -> relocate, alternate axes (max 4 tries) -> next target
  END when task_complete / episode_ended / plan exhausted (hold last container lifted).
"""
import json
import math
import os
import re
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vu_geom as G  # noqa: E402
import vu_io as IO  # noqa: E402
import vu_percep as PC  # noqa: E402
import vu_reg as REG  # noqa: E402

VERSION = 'vu-v2'

# ---------------------------------------------------------------- tunables
POS_SCALE = 0.05          # m per unit action (protocol)
ROT_SCALE = 0.5           # rad per unit action (protocol)
VMAX_TRAVEL = 0.030       # m per step in free space (60% of the protocol max)
VMAX_NEAR = 0.012         # m per step near contact
RMAX = 0.12               # rad per step
VIEW1_H = 0.26            # TCP height above table for the first look
VIEW2_ABOVE_TOP = 0.13    # TCP height above container top for the close look
PREGRASP_ABOVE_TOP = 0.05
LIFT_H = 0.16             # lift above grasp height
CUBE_MERGE_R = 0.035      # cubes in different containers are >= ~5 cm apart
ASSIGN_R = 0.05           # cube -> container centre association radius
LOCAL_R = 0.05            # container re-identification radius in close views
MIN_OPEN_MARGIN = 0.016   # extra opening per side beyond container width
MAX_FINGER = 0.04
WALL_SOFT_LIMIT = 1050.0   # s since import; hard limit 1200 s

T0 = time.time()
LOGF = None


def log(*a):
    global LOGF
    s = f'[{VERSION} t={time.time() - T0:6.1f}] ' + ' '.join(str(x) for x in a)
    print(s, flush=True)
    try:
        if LOGF is None:
            os.makedirs('/workspace/output', exist_ok=True)
            LOGF = open('/workspace/output/vu_log.txt', 'a')
        LOGF.write(s + '\n')
        LOGF.flush()
    except Exception:
        pass


def jdump(o):
    try:
        return json.dumps(o, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
    except Exception:
        return str(o)


class EpisodeOver(Exception):
    pass


# ================================================================== robot I/O
class Robot:
    def __init__(self, ctx):
        self.ctx = ctx
        self.ended = False
        self.success = False
        self.state = None
        self.step_id = None
        self.remaining = None
        self.last_js = None
        self.last_result = None
        self.calls = 0
        self.grip_cmd = -1.0

    def _update(self, js):
        st = IO.get_state(js)
        if st is not None:
            self.state = st
        sid = IO.get_step_id(js)
        if sid is not None:
            self.step_id = sid
        rem = IO.remaining_steps(js)
        if rem is not None:
            self.remaining = rem
        if IO.episode_ended(js):
            self.ended = True
        if IO.task_complete(js):
            self.success = True

    def observe(self):
        last = None
        for k in range(3):
            try:
                res = self.ctx.call('robomme_observe')
                js = IO.reply_json(res)
                self.last_js, self.last_result = js, res
                self._update(js)
                return res, js
            except Exception as e:
                last = e
                log('observe failed', k, repr(e)[:300])
                time.sleep(0.5)
        raise last

    def step(self, rows):
        if self.ended:
            raise EpisodeOver()
        rows = [[float(max(-1.0, min(1.0, x))) for x in r] for r in rows]
        out_js = None
        for i in range(0, len(rows), 100):
            chunk = rows[i:i + 100]
            try:
                res = self.ctx.call('robomme_step', actions=chunk)
            except Exception as e:
                # never blindly retry motion: physics may have advanced
                log('step error', repr(e)[:400])
                try:
                    self.observe()
                except Exception:
                    pass
                if self.ended:
                    raise EpisodeOver()
                raise
            self.calls += 1
            js = IO.reply_json(res)
            self.last_js, self.last_result = js, res
            self.last_step_js = js
            self._update(js)
            out_js = js
            if self.ended:
                log('episode ended; success flag =', self.success, 'step', self.step_id)
                raise EpisodeOver()
        if IO.get_state(out_js or {}) is None:
            self.observe()
        return out_js

    # --- state helpers
    def pos(self):
        return np.array(self.state[:3], float)

    def R(self):
        return G.rotvec_to_R(self.state[3:6])

    def fingers(self):
        return float(self.state[6] + self.state[7])


# ============================================================ camera calibration
def cam_variants(K, pose):
    """Candidate Camera models for a parsed pose dict."""
    out = []
    if K is None or pose is None:
        return out
    convs = ['opengl', 'opencv']
    if 'T' in pose:
        T = np.asarray(pose['T'], float)
        for name, TT in (('T', T), ('Tinv', G.inv_T(T))):
            for cv in convs:
                out.append((f'{name}/{cv}', G.Camera(K, TT[:3, :3], TT[:3, 3], cv)))
        return out
    q = np.asarray(pose['quat'], float)
    t = np.asarray(pose['pos'], float)
    qs = []
    if pose.get('qorder') == 'xyzw':
        qs.append(('xyzw', np.array([q[3], q[0], q[1], q[2]])))
    else:
        qs.append(('wxyz', q))
        qs.append(('xyzw', np.array([q[3], q[0], q[1], q[2]])))
    for qn, qq in qs:
        R = G.quat_wxyz_to_R(qq)
        for cv in convs:
            out.append((f'{qn}/{cv}', G.Camera(K, R, t, cv)))
            Ti = G.inv_T(G.make_T(R, t))
            out.append((f'{qn}/inv/{cv}', G.Camera(K, Ti[:3, :3], Ti[:3, 3], cv)))
    # documented default first
    pref = 'opengl' if ('gl' in (pose.get('conv') or 'opengl')) else 'opencv'
    out.sort(key=lambda x: (0 if (x[0].startswith('wxyz/') and x[0].endswith(pref) and '/inv/' not in x[0]) else 1))
    return out


def parse_points_reply(js, n):
    """Extract up to n world points (or None) from a pixel_to_position reply, in order."""
    pts = []
    cands = None
    for k, v, parent in IO.walk(js):
        if isinstance(v, list) and len(v) >= 1 and all(isinstance(e, dict) for e in v):
            if any(any(IO.is_num_list(x, 3) for x in e.values()) or 'valid' in e for e in v):
                cands = v
                break
    if cands is None:
        return [None] * n
    for e in cands[:n]:
        p = None
        if e.get('valid') is False:
            pts.append(None)
            continue
        for key in ('position', 'world_position', 'point', 'xyz', 'world', 'position_m', 'world_point'):
            if IO.is_num_list(e.get(key), 3):
                p = np.array(e[key], float)
                break
        if p is None:
            for key, val in e.items():
                if IO.is_num_list(val, 3) and 'pixel' not in key.lower():
                    p = np.array(val, float)
                    break
        if p is None:
            for key, val in e.items():
                if isinstance(val, dict):
                    for kk in ('x', 'X'):
                        if kk in val and 'y' in val and 'z' in val:
                            p = np.array([val['x'], val['y'], val['z']], float)
        pts.append(p)
    while len(pts) < n:
        pts.append(None)
    return pts


def build_cameras(robot, res, js, imgs):
    """Return dict cam -> Camera validated against robomme_pixel_to_position."""
    cams = {}
    info = {}
    for cam in ('front', 'wrist'):
        K = IO.get_intrinsics(js, cam)
        pose = IO.get_pose(js, cam)
        info[cam] = dict(K=None if K is None else K.tolist(), pose=IO.summarize(pose, 300) if pose else None)
        cams[cam] = cam_variants(K, pose)
    log('camera parse', jdump(info)[:1500])
    # tool check
    sels, meta = [], []
    for cam in ('front', 'wrist'):
        d = imgs.get(cam, {}).get('depth')
        if d is None:
            continue
        for (u, v) in [(64, 64), (128, 64), (192, 64), (64, 128), (128, 128), (192, 128),
                       (64, 192), (128, 192), (192, 192), (100, 160), (160, 100), (40, 220)]:
            if np.isfinite(d[v, u]):
                sels.append({'camera': cam, 'u': float(u), 'v': float(v)})
                meta.append((cam, u, v, float(d[v, u])))
    chosen = {}
    tool_pts = []
    if sels and robot.step_id is not None:
        try:
            r = robot.ctx.call('robomme_pixel_to_position', observation_step_id=int(robot.step_id),
                               selections=sels[:32], depth_window_radius=0)
            rjs = IO.reply_json(r)
            log('pixel_to_position reply', IO.summarize(rjs, 900))
            tool_pts = parse_points_reply(rjs, len(sels[:32]))
        except Exception as e:
            log('pixel_to_position failed', repr(e)[:300])
    for cam in ('front', 'wrist'):
        best = None
        for name, C in cams[cam]:
            errs = []
            for (c, u, v, dd), p in zip(meta, tool_pts):
                if c != cam or p is None:
                    continue
                q = C.backproject(np.array([u]), np.array([v]), np.array([dd]))[0]
                errs.append(float(np.linalg.norm(q - p)))
            e = float(np.median(errs)) if errs else None
            if e is not None and (best is None or e < best[0]):
                best = (e, name, C)
        if best is not None:
            log(f'cam {cam}: best variant {best[1]} median err {best[0]:.4f} m')
            if best[0] < 0.03:
                chosen[cam] = best[2]
                continue
        if cams[cam]:
            log(f'cam {cam}: using default variant {cams[cam][0][0]} (no tool validation)')
            chosen[cam] = cams[cam][0][1]
    return chosen, cams


# =================================================================== utilities
def frame_cam_image(frame, cam, kind):
    ff = IO.frame_files(frame)
    if (cam, kind) not in ff:
        return None
    path, mpu = ff[(cam, kind)]
    try:
        return IO.load_depth(path, mpu) if kind == 'depth' else IO.load_rgb(path)
    except Exception as e:
        log('image load failed', path, repr(e)[:200])
        return None


def parse_instruction(text):
    """Ordered list of (colour, many) from the instruction."""
    t = text.lower()
    idx = t.find('pick up')
    if idx >= 0:
        t = t[idx:]
    words = [w for w in PC.COLOR_WORDS]
    pat = re.compile(r'\b(' + '|'.join(words) + r')\b')
    out = []
    prev_end = 0
    for m in pat.finditer(t):
        c = PC.COLOR_ALIASES.get(m.group(1), m.group(1))
        seg = t[prev_end:m.start()]
        after = t[m.end():m.end() + 8]
        many = ('all' in seg.split()) or after.strip().startswith('cubes') or ('every' in seg) or ('both' in seg)
        out.append((c, bool(many)))
        prev_end = m.end()
    # merge repeats of the same colour adjacent
    res = []
    for c, mny in out:
        if res and res[-1][0] == c:
            res[-1] = (c, res[-1][1] or mny)
        else:
            res.append((c, mny))
    return res


# ============================================================= video analysis
def choose_video_wrist(video, cams, T_ee_cam, table_z, overview_xy):
    """Decide how to pose the wrist camera in video frames: ('state', None) | ('static', Camera) | ('none', None)."""
    C = cams.get('wrist')
    if C is None:
        return 'none', None
    last = None
    for fr in reversed(video):
        if (('wrist', 'depth') in IO.frame_files(fr)):
            last = fr
            break
    if last is None:
        return ('state', None) if T_ee_cam is not None else ('none', None)
    depth = frame_cam_image(last, 'wrist', 'depth')
    if depth is None:
        return ('state', None) if T_ee_cam is not None else ('none', None)
    qS = None
    st = last.get('state')
    if IO.is_num_list(st, 8) and T_ee_cam is not None:
        Tc = G.make_T(G.rotvec_to_R(st[3:6]), np.array(st[:3], float)) @ T_ee_cam
        Cs = C.copy_with_pose(Tc[:3, :3], Tc[:3, 3])
        qS = REG.pose_quality(Cs, depth, table_z, overview_xy)
        log('video wrist pose from state: quality', jdump(qS))
        if qS and qS['table_err'] < 0.015 and (qS['n'] == 0 or len(overview_xy) < 2 or
                                               (qS['n_in'] >= max(2, 0.5 * qS['n']) and qS['med'] < 0.02)):
            return 'state', None
    try:
        Cr = REG.estimate_wrist_pose(depth, C, table_z, overview_xy, log=log)
    except Exception as e:
        log('wrist registration failed', repr(e)[:200])
        Cr = None
    if Cr is not None:
        qR = REG.pose_quality(Cr, depth, table_z, overview_xy)
        log('video wrist pose from registration: quality', jdump(qR))
        if qS is None or (qR['n_in'] > qS['n_in'] and qR['table_err'] < 0.015):
            return 'static', Cr
    if qS is not None:
        return 'state', None
    return 'none', None


def analyze_video(frames, cams, T_ee_cam, colors, table_z, overview_xy=()):
    video = [f for f in frames if str(f.get('phase', '')).lower().startswith('video')]
    log('video frames', len(video))
    obs = []    # dict(color, cam, frame, xyz, area)
    if not video:
        return obs
    try:
        wmode, Cstatic = choose_video_wrist(video, cams, T_ee_cam, table_z, np.asarray(overview_xy).reshape(-1, 2))
    except Exception as e:
        log('choose_video_wrist failed', repr(e)[:200], traceback.format_exc()[-800:])
        wmode, Cstatic = 'state', None
    log('video wrist mode', wmode)
    step = 1 if len(video) <= 160 else 2
    t_start = time.time()
    n_wrist_pose_missing = 0
    for i, fr in enumerate(video[::step]):
        if time.time() - t_start > 110:
            log('video analysis time cap reached at', i)
            break
        fidx = fr.get('frame_index', i)
        for cam in ('front', 'wrist'):
            C = cams.get(cam)
            if C is None:
                continue
            rgb = frame_cam_image(fr, cam, 'rgb')
            if rgb is None:
                continue
            blobs = PC.detect_blobs(rgb, colors, min_area=10)
            if not blobs:
                continue
            depth = frame_cam_image(fr, cam, 'depth')
            if depth is None:
                continue
            Cf = C
            if cam == 'wrist' and wmode == 'none':
                continue
            if cam == 'wrist' and wmode == 'static':
                Cf = Cstatic
            elif cam == 'wrist':
                st = fr.get('state')
                if not IO.is_num_list(st, 8) or T_ee_cam is None:
                    n_wrist_pose_missing += 1
                    continue
                T_w_ee = G.make_T(G.rotvec_to_R(st[3:6]), np.array(st[:3], float))
                Tc = T_w_ee @ T_ee_cam
                Cf = C.copy_with_pose(Tc[:3, :3], Tc[:3, 3])
            for b in blobs:
                r = PC.blob_world_point(b, depth, Cf)
                if r is None:
                    continue
                p, dz = r
                if table_z is not None and not (table_z - 0.03 < p[2] < table_z + 0.09):
                    continue
                obs.append(dict(color=b['color'], cam=cam, frame=fidx, xyz=p, area=b['area'],
                                uv=(b['u'], b['v'])))
    log('video blob observations', len(obs), 'wrist pose missing frames', n_wrist_pose_missing,
        f'({time.time() - t_start:.1f}s)')
    return obs


def cluster_obs(obs, radius):
    clusters = []
    for o in sorted(obs, key=lambda o: o['frame']):
        best, bd = None, None
        for c in clusters:
            if c['color'] != o['color']:
                continue
            d = float(np.linalg.norm(np.median(c['xy'], 0) - o['xyz'][:2]))
            if d < radius and (bd is None or d < bd):
                best, bd = c, d
        if best is None:
            clusters.append(dict(color=o['color'], xy=[o['xyz'][:2]], z=[o['xyz'][2]], frames={o['frame']},
                                 cams={o['cam']}, n=1, last=o['frame'], first=o['frame']))
        else:
            best['xy'].append(o['xyz'][:2])
            best['z'].append(o['xyz'][2])
            best['frames'].add(o['frame'])
            best['cams'].add(o['cam'])
            best['n'] += 1
            best['last'] = max(best['last'], o['frame'])
    for c in clusters:
        c['c'] = np.median(np.array(c['xy']), 0)
        c['nf'] = len(c['frames'])
    return clusters


def build_cube_clusters(obs):
    front = [o for o in obs if o['cam'] == 'front']
    wrist = [o for o in obs if o['cam'] == 'wrist']
    cf = [c for c in cluster_obs(front, CUBE_MERGE_R) if c['nf'] >= 2]
    cw = [c for c in cluster_obs(wrist, CUBE_MERGE_R) if c['nf'] >= 2]
    # wrist-chain consistency: offsets between same-colour clusters seen by both cameras
    offs = []
    for a in cw:
        best = None
        for b in cf:
            if a['color'] != b['color']:
                continue
            d = a['c'] - b['c']
            if np.linalg.norm(d) < 0.06 and (best is None or np.linalg.norm(d) < np.linalg.norm(best)):
                best = d
        if best is not None:
            offs.append(best)
    shift = np.zeros(2)
    if offs:
        offs = np.array(offs)
        med = np.median(offs, 0)
        spread = float(np.median(np.linalg.norm(offs - med, axis=1)))
        log('wrist-vs-front cube offsets', jdump(np.round(offs, 4)), 'median', jdump(np.round(med, 4)),
            'spread', round(spread, 4))
        if np.linalg.norm(med) > 0.008 and spread < 0.015:
            shift = med
    obs2 = [dict(o, xyz=np.r_[o['xyz'][:2] - shift, o['xyz'][2]]) if o['cam'] == 'wrist' else o for o in obs]
    cl = [c for c in cluster_obs(obs2, CUBE_MERGE_R) if c['nf'] >= 2]
    for c in cl:
        log(f"cube cluster {c['color']} at {np.round(c['c'], 4).tolist()} frames={c['nf']} "
            f"cams={sorted(c['cams'])} span=[{c['first']},{c['last']}]")
    return cl, shift


# ============================================================ container views
def view_points(imgs, cam_model):
    d = imgs.get('depth')
    if d is None or cam_model is None:
        return None
    return cam_model.backproject_image(d)


def containers_from_view(P, table_z, exclude_near=None):
    comps, lab = PC.segment_objects(P, table_z, min_h=0.015, max_h=0.15, min_px=12)
    # tall structures (pillars, the arm): components touching them are their lower parts
    hz = P[..., 2] - table_z
    tall = np.isfinite(hz) & (hz > 0.16)
    tall_d = PC.dilate(tall, 2) if tall.any() else tall
    res = []
    for c in comps:
        touch = int((c['mask'] & tall_d).sum())
        if touch > 0.2 * c['n']:
            continue
        parts = [c]
        cx, cy, w, h, a = PC.robust_rect(c['pts'][:, :2])
        if max(w, h) > 0.085:
            sp = PC.split_by_top(P, c, table_z)
            if len(sp) >= 2:
                parts = sp
        for p in parts:
            cx, cy, w, h, a = PC.robust_rect(p['pts'][:, :2])
            if not (0.022 < min(w, h) and max(w, h) < 0.12):
                continue
            tb = p['pts'][p['pts'][:, 2] > p['top'] - 0.015]     # top band: side faces bias the centre
            if len(tb) < 5:
                tb = p['pts']
            res.append(dict(cx=float(np.median(tb[:, 0])), cy=float(np.median(tb[:, 1])),
                            rcx=cx, rcy=cy, w=w, h=h, ang=a, top=p['top'], n=p['n'], mask=p['mask'],
                            pts=p['pts']))
    return res


def dedupe_containers(lst, r=0.03):
    out = []
    for c in lst:
        if any(math.hypot(c['cx'] - o['cx'], c['cy'] - o['cy']) < r for o in out):
            continue
        out.append(c)
    return out


# ================================================================== motion
class Motion:
    def __init__(self, robot, wrist_cam):
        self.r = robot
        R0 = robot.R()
        # approach axis = EE column pointing most downward; finger axis = column best
        # aligned with the wrist camera's image-x axis
        self.ai = int(np.argmin(R0[2, :]))
        fi = None
        if wrist_cam is not None:
            xa = wrist_cam.x_axis()
            scores = [abs(float(R0[:, i] @ xa)) if i != self.ai else -1 for i in range(3)]
            fi = int(np.argmax(scores))
            log('finger-axis alignment scores', jdump(np.round(scores, 3)))
        if fi is None or fi == self.ai:
            fi = 1 if self.ai != 1 else 0
        self.fi = fi
        self.ki = 3 - self.ai - self.fi
        f = R0[:, self.fi]
        self.yaw0 = math.atan2(f[1], f[0])
        log(f'EE axes: approach col {self.ai} (z={R0[2, self.ai]:.3f}), finger col {self.fi}, yaw0 {math.degrees(self.yaw0):.1f}')
        # sign of the third column so that det=+1 (keep handedness of the measured frame)
        self.k_sign = 1.0
        self.gripper_linear = None   # (g0 finger at g=0) once measured

    def R_target(self, yaw):
        a = np.array([0, 0, -1.0])
        f = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        R = np.zeros((3, 3))
        R[:, self.ai] = a
        R[:, self.fi] = f
        k = np.cross(a, f) if (self.ai, self.fi) in ((0, 1), (1, 2), (2, 0)) else np.cross(f, a)
        R[:, self.ki] = k
        if np.linalg.det(R) < 0:
            R[:, self.ki] = -k
        return R

    def cur_yaw(self):
        f = self.r.R()[:, self.fi]
        return math.atan2(f[1], f[0])

    def best_yaw_for_axis(self, axis_angle):
        """Finger yaw equivalent (mod pi) to axis_angle closest to the initial yaw."""
        d = G.wrap_pi(axis_angle - self.yaw0)
        if d > math.pi / 2:
            d -= math.pi
        elif d < -math.pi / 2:
            d += math.pi
        return self.yaw0 + d

    def move(self, target, yaw=None, grip=None, vmax=VMAX_TRAVEL, tol=0.005, iters=4, label=''):
        """Closed-loop chunked move of the TCP to target (world), yaw about z."""
        target = np.asarray(target, float)
        grip = self.r.grip_cmd if grip is None else grip
        yaw = self.cur_yaw() if yaw is None else yaw
        # unwrap yaw near the current measurement
        cy = self.cur_yaw()
        yaw = cy + G.wrap_pi(yaw - cy)
        Rt = self.R_target(yaw)
        for it in range(iters):
            p = self.r.pos()
            err = target - p
            dist = float(np.linalg.norm(err))
            rerr = G.R_to_rotvec(Rt @ self.r.R().T)
            rang = float(np.linalg.norm(rerr))
            if dist < tol and rang < 0.03:
                break
            n = int(max(1, math.ceil(dist / vmax), math.ceil(rang / RMAX)))
            n = min(n, 60)
            dp = err / n / POS_SCALE
            dr = rerr / n / ROT_SCALE
            # last iteration: small corrective chunks
            rows = [[dp[0], dp[1], dp[2], dr[0], dr[1], dr[2], grip] for _ in range(n)]
            if it > 0:
                rows = rows + [[0, 0, 0, 0, 0, 0, grip]]
            self.r.step(rows)
        p = self.r.pos()
        e = float(np.linalg.norm(target - p))
        log(f'move[{label}] -> {np.round(target, 3).tolist()} got {np.round(p, 3).tolist()} err {e:.4f} '
            f'yaw {math.degrees(self.cur_yaw()):.1f}/{math.degrees(yaw):.1f} fingers {self.r.fingers():.4f} step {self.r.step_id}')
        return e

    def hold(self, n, grip):
        self.r.grip_cmd = grip
        self.r.step([[0, 0, 0, 0, 0, 0, grip]] * n)

    def grip_value_for_opening(self, opening):
        """Gripper command for a desired total opening (if proportional control was measured)."""
        if self.gripper_linear is None:
            return -1.0
        f0 = self.gripper_linear   # finger sum at g=0, finger sum at g=-1 is 2*MAX
        fo = 2 * MAX_FINGER
        # linear in g between (-1, fo) and (0, f0)
        if fo - f0 < 0.01:
            return -1.0
        g = -(opening - f0) / (fo - f0)
        return float(max(-1.0, min(0.0, g)))


# ================================================================= main task
class Task:
    def __init__(self, ctx):
        self.r = Robot(ctx)
        self.ctx = ctx
        self.cams = {}
        self.table_z = None
        self.motion = None
        self.T_ee_cam = None

    # ---------------------------------------------------------------- images
    def current_view(self, cam='wrist'):
        res, js = self.r.observe()
        imgs = IO.current_images(res, js)
        C = None
        K = IO.get_intrinsics(js, cam)
        pose = IO.get_pose(js, cam)
        base = self.cams.get(cam)
        if base is not None and pose is not None:
            # same convention as the calibrated variant: rebuild from the fresh pose
            for name, Cv in cam_variants(K if K is not None else base.K, pose):
                if Cv.conv == base.conv and name == self.cam_variant_name.get(cam):
                    C = Cv
                    break
            if C is None:
                C = cam_variants(K if K is not None else base.K, pose)[0][1]
        elif base is not None and cam == 'front':
            C = base
        elif base is not None and cam == 'wrist' and self.T_ee_cam is not None and self.r.state is not None:
            T = G.make_T(self.r.R(), self.r.pos()) @ self.T_ee_cam
            C = base.copy_with_pose(T[:3, :3], T[:3, 3])
        out = imgs.get(cam, {})
        if (out.get('depth') is None or C is None) and getattr(self.r, 'last_step_js', None):
            fb = self.view_from_step_manifest(cam)
            if fb is not None:
                out, C = fb
        return out, C, js

    def view_from_step_manifest(self, cam):
        """Fallback: newest execution frame (with depth) of the last motion reply's manifest."""
        try:
            man, _ = IO.load_manifest(self.r.last_step_js)
            frs = [f for f in IO.sorted_frames(man) if str(f.get('phase', '')).lower().startswith('exec')]
            for f in reversed(frs):
                ff = IO.frame_files(f)
                if (cam, 'depth') not in ff:
                    continue
                d = frame_cam_image(f, cam, 'depth')
                rgb = frame_cam_image(f, cam, 'rgb')
                base = self.cams.get(cam)
                if d is None or base is None:
                    return None
                if cam == 'front':
                    C = base
                else:
                    st = f.get('state')
                    if self.T_ee_cam is None or not IO.is_num_list(st, 8):
                        return None
                    T = G.make_T(G.rotvec_to_R(st[3:6]), np.array(st[:3], float)) @ self.T_ee_cam
                    C = base.copy_with_pose(T[:3, :3], T[:3, 3])
                log(f'view fallback: {cam} from step-manifest frame step {f.get("step_id")}')
                return {'depth': d, 'rgb': rgb}, C
        except Exception as e:
            log('view fallback failed', repr(e)[:200])
        return None

    # ----------------------------------------------------------------- run
    def run(self):
        r = self.r
        res, js = r.observe()
        log('observe keys', list(js.keys())[:40])
        log('observe summary', IO.summarize(js, 2500))
        instr = IO.get_instruction(js)
        log('instruction:', instr)
        log('state', jdump(r.state), 'step', r.step_id, 'remaining', r.remaining)
        man, mpath = IO.load_manifest(js)
        frames = IO.sorted_frames(man) if man else []
        log('manifest', mpath, 'frames', len(frames),
            'phases', jdump({p: sum(1 for f in frames if f.get('phase') == p) for p in set(f.get('phase') for f in frames)}))
        if frames:
            log('frame0 summary', IO.summarize({k: v for k, v in frames[0].items()}, 600))
        imgs0 = IO.current_images(res, js)
        log('current images', jdump({c: {k: list(v.shape) for k, v in d.items()} for c, d in imgs0.items()}))
        if (('depth' not in imgs0.get('wrist', {})) or ('depth' not in imgs0.get('front', {}))) and frames:
            ex = [f for f in frames if str(f.get('phase', '')).lower().startswith('exec')]
            if ex:
                for cam in ('front', 'wrist'):
                    for kind in ('rgb', 'depth'):
                        if kind not in imgs0.get(cam, {}):
                            a = frame_cam_image(ex[-1], cam, kind)
                            if a is not None:
                                imgs0.setdefault(cam, {})[kind] = a

        # ---- S1 cameras
        chosen, variants = build_cameras(r, res, js, imgs0)
        self.cams = chosen
        self.cam_variant_name = {}
        for cam in chosen:
            for name, C in variants[cam]:
                if C is chosen[cam]:
                    self.cam_variant_name[cam] = name
        if 'wrist' in chosen and r.state is not None:
            T_w_ee = G.make_T(r.R(), r.pos())
            T_w_c = G.make_T(chosen['wrist'].R, chosen['wrist'].t)
            self.T_ee_cam = G.inv_T(T_w_ee) @ T_w_c
            log('T_ee_cam', jdump(np.round(self.T_ee_cam, 4)))

        # ---- table height from the step-0 views
        zs = []
        for cam in ('front', 'wrist'):
            P = view_points(imgs0.get(cam, {}), chosen.get(cam))
            if P is not None:
                zs.append(P[..., 2].ravel())
        if zs:
            self.table_z = PC.table_height(np.concatenate(zs))
        log('table_z', self.table_z)
        if self.table_z is None:
            self.table_z = 0.0

        # ---- S3 plan colours
        plan = parse_instruction(instr)
        log('plan colours', plan)
        colors = sorted(set(c for c, _ in plan)) or ['red', 'green', 'blue']

        # ---- S4 overview containers at step 0
        conts = []
        for cam in ('wrist', 'front'):
            P = view_points(imgs0.get(cam, {}), chosen.get(cam))
            if P is None:
                continue
            cc = containers_from_view(P, self.table_z)
            log(f'{cam} overview containers', len(cc),
                jdump([[round(c['cx'], 3), round(c['cy'], 3), round(min(c['w'], c['h']), 3), round(c['top'] - self.table_z, 3)] for c in cc]))
            conts += cc
        conts = dedupe_containers(conts)
        log('overview containers total', len(conts))

        # ---- S2 video
        obs = analyze_video(frames, chosen, self.T_ee_cam, colors, self.table_z,
                             [[k['cx'], k['cy']] for k in conts])
        clusters, shift = build_cube_clusters(obs)

        targets = self.assign(plan, clusters, conts)
        log('TARGETS', jdump([(t['color'], np.round(t['xy'], 4).tolist(), t['src']) for t in targets]))
        if not targets:
            log('no targets; ending')
            return

        # ---- S5 manipulate
        self.motion = Motion(r, chosen.get('wrist'))
        self.lifted = []
        for i, t in enumerate(targets):
            last = (i == len(targets) - 1)
            if r.ended:
                break
            log(f'=== target {i + 1}/{len(targets)} {t["color"]} at {np.round(t["xy"], 3).tolist()} last={last}')
            ok = False
            for tries in range(2):
                try:
                    ok = self.pick(t, last)
                except EpisodeOver:
                    raise
                except Exception as e:
                    log('pick exception', repr(e)[:300], traceback.format_exc()[-1500:])
                    self.recover()
                if ok or r.success or r.ended or self.budget_low():
                    break
                # skipping a target fails the ordered task anyway: try it once more
                log(f'target {i + 1} retry')
            log(f'target {i + 1} result ok={ok} success_flag={r.success}')
            if r.success:
                break
        if not r.ended:
            # final hold to let the native evaluator register
            try:
                self.motion.hold(20, self.r.grip_cmd)
            except EpisodeOver:
                pass

    # -------------------------------------------------------------- planning
    def assign(self, plan, clusters, conts):
        targets = []
        used = []
        for color, many in plan:
            cs = [c for c in clusters if c['color'] == color]
            cs.sort(key=lambda c: -c['nf'])
            if not many:
                cs = cs[:1]
            if not cs:
                log(f'WARNING no cube cluster for {color}')
                continue
            group = []
            for c in cs:
                xy = c['c']
                src = 'cube'
                if conts:
                    d = [math.hypot(xy[0] - k['cx'], xy[1] - k['cy']) for k in conts]
                    j = int(np.argmin(d))
                    if d[j] < ASSIGN_R:
                        xy = np.array([conts[j]['cx'], conts[j]['cy']])
                        src = f'container({d[j]:.3f})'
                if any(np.linalg.norm(xy - u) < 0.03 for u in used):
                    log('skip duplicate target', color, np.round(xy, 3).tolist())
                    continue
                used.append(xy)
                group.append(dict(color=color, xy=np.array(xy, float), cube=c['c'], src=src))
            targets += group
        # order within colour groups by proximity chain (cheaper travel)
        out = []
        pos = self.r.pos()[:2] if self.r.state is not None else np.zeros(2)
        i = 0
        while i < len(targets):
            j = i
            while j < len(targets) and targets[j]['color'] == targets[i]['color']:
                j += 1
            grp = targets[i:j]
            while grp:
                k = int(np.argmin([np.linalg.norm(g['xy'] - pos) for g in grp]))
                out.append(grp[k])
                pos = grp[k]['xy']
                grp.pop(k)
            i = j
        return out

    # ---------------------------------------------------------------- picking
    def locate(self, xy_guess, radius=LOCAL_R, label=''):
        """Segment the container nearest xy_guess in the current wrist view (front fallback)."""
        for cam in ('wrist', 'front'):
            imgs, C, js = self.current_view(cam)
            P = view_points(imgs, C)
            if P is None:
                log(f'locate[{label}] no depth for {cam}')
                continue
            # ignore the gripper itself: points near the TCP height or above
            tcp = self.r.pos()
            valid = np.isfinite(P[..., 2]) & (P[..., 2] < tcp[2] - 0.04)
            comps, _ = PC.segment_objects(P, self.table_z, min_h=0.012, max_h=0.15, valid=valid, min_px=20)
            hz = P[..., 2] - self.table_z
            tall = np.isfinite(hz) & (hz > 0.16) & (P[..., 2] < tcp[2] - 0.03)
            tall_d = PC.dilate(tall, 2) if tall.any() else tall
            cand = []
            for c in comps:
                if int((c['mask'] & tall_d).sum()) > 0.2 * c['n']:
                    continue
                parts = [c]
                cx, cy, w, h, a = PC.robust_rect(c['pts'][:, :2])
                if max(w, h) > 0.085:
                    sp = PC.split_by_top(P, c, self.table_z)
                    if len(sp) >= 2:
                        parts = sp
                for p in parts:
                    d = math.hypot(np.median(p['pts'][:, 0]) - xy_guess[0], np.median(p['pts'][:, 1]) - xy_guess[1])
                    cand.append((d, p))
            cand.sort(key=lambda x: x[0])
            if cand and cand[0][0] < radius:
                d, p = cand[0]
                cx, cy, w, h, a = PC.robust_rect(p['pts'][:, :2])
                top = p['top']
                log(f'locate[{label}] {cam}: d={d:.3f} rect c=({cx:.3f},{cy:.3f}) w={w:.3f} h={h:.3f} '
                    f'ang={math.degrees(a):.1f} top={top - self.table_z:.3f} n={p["n"]} comps={len(cand)}')
                if not (0.02 < min(w, h) < 0.11):
                    log(f'locate[{label}] implausible size, trying next camera')
                    continue
                return dict(cx=cx, cy=cy, w=w, h=h, ang=a, top=top, P=P, mask=p['mask'], cam=cam)
            log(f'locate[{label}] {cam}: nothing within {radius} (nearest {cand[0][0] if cand else None})')
        return None

    def budget_low(self):
        rem = self.r.remaining
        if rem is not None and rem < 70:
            return True
        return time.time() - T0 > WALL_SOFT_LIMIT

    def clearance(self, loc, axis_ang, half_w, z_grasp):
        """Free space (m) on each side (-, +) of the container along a squeeze axis, at finger depth."""
        P = loc.get('P')
        if P is None:
            return 0.05, 0.05
        m = np.isfinite(P[..., 2]) & ~loc['mask'] & (P[..., 2] > z_grasp - 0.012) & \
            (P[..., 2] < loc['top'] + 0.03)
        pts = P[m][:, :2] - np.array([loc['cx'], loc['cy']])
        if len(pts) == 0:
            return 0.05, 0.05
        g = np.array([math.cos(axis_ang), math.sin(axis_ang)])
        n = np.array([-g[1], g[0]])
        along = pts @ g
        perp = pts @ n
        out = []
        for sgn in (-1, 1):
            sel = (np.abs(perp) < 0.016) & (sgn * along > half_w + 0.002)
            out.append(float(np.min(sgn * along[sel]) - half_w) if sel.any() else 0.05)
        return out[0], out[1]

    def grasp_plans(self, loc, z_grasp):
        """Ordered grasp plans: (score, ang, width, clr_minus, clr_plus, shift)."""
        need = MIN_OPEN_MARGIN + 0.012     # opening margin + finger thickness
        plans = []
        for ang, width in ((loc['ang'], loc['w']), (loc['ang'] + math.pi / 2, loc['h'])):
            cm, cp = self.clearance(loc, ang, width / 2, z_grasp)
            # centre the fingers between the neighbours (bounded so the jaw still encloses the body)
            shift = float(np.clip((cp - cm) / 2, -0.008, 0.008)) if min(cm, cp) < need else 0.0
            clr = min(cm + shift, cp - shift)
            score = (width > 0.075) * 10 + max(0.0, need - clr) * 100 + width * 5
            plans.append((score, ang, width, cm, cp, shift, clr))
        plans.sort(key=lambda s: s[0])
        return plans

    def pick(self, t, last):
        r, M = self.r, self.motion
        tz = self.table_z
        xy = np.array(t['xy'], float)
        # first look from above
        M.move([xy[0], xy[1], tz + VIEW1_H], yaw=M.yaw0, grip=-1.0, label='view1')
        if M.gripper_linear is None and not getattr(self, '_grip_probe_done', False):
            self._grip_probe_done = True
            try:
                M.hold(6, 0.0)
                f0 = r.fingers()
                M.hold(6, -1.0)
                fo = r.fingers()
                log(f'gripper probe: g=0 -> {f0:.4f}, g=-1 -> {fo:.4f}')
                if 0.01 < f0 < fo - 0.012:
                    M.gripper_linear = f0
            except EpisodeOver:
                raise
            except Exception as e:
                log('gripper probe failed', repr(e)[:200])
        loc = self.locate(xy, label='view1')
        if loc is None:
            loc = dict(cx=xy[0], cy=xy[1], top=tz + 0.06, w=0.05, h=0.05, ang=0.0)
            log('view1 locate failed; using planned xy')
        # close look
        M.move([loc['cx'], loc['cy'], loc['top'] + VIEW2_ABOVE_TOP], yaw=M.yaw0, grip=-1.0, label='view2')
        loc2 = self.locate(np.array([loc['cx'], loc['cy']]), radius=0.035, label='view2')
        if loc2 is not None:
            loc = loc2
        if 'P' not in loc:
            log('no geometry for this target; blind grasp at the planned position')
        n_att = 4
        for attempt in range(n_att):
            if self.budget_low():
                log('budget low; stop attempts')
                return False
            cx, cy = loc['cx'], loc['cy']
            hgt = loc['top'] - tz
            z_grasp = tz + float(np.clip(0.5 * hgt, 0.018, 0.04))
            plans = self.grasp_plans(loc, z_grasp)
            # attempt order: best, other axis, best again (relocated), other again
            score, ang, width, cm, cp, shift, clr = plans[attempt % 2]
            yaw = M.best_yaw_for_axis(ang)
            opening = 2 * MAX_FINGER
            if attempt >= 2 and clr < MIN_OPEN_MARGIN + 0.012:
                # tight on both axes: partial opening if the gripper is proportional
                opening = min(opening, width + 2 * max(0.008, clr - 0.010))
            g_open = M.grip_value_for_opening(opening) if opening < 2 * MAX_FINGER - 0.004 else -1.0
            g = np.array([math.cos(ang), math.sin(ang)])
            gx, gy = cx + shift * g[0], cy + shift * g[1]
            log(f'grasp plan attempt {attempt}: squeeze width {width:.3f} clearance -{cm:.3f}/+{cp:.3f} '
                f'shift {shift:.3f} yaw {math.degrees(yaw):.1f} z_grasp {z_grasp - tz:.3f} '
                f'opening {opening:.3f} g_open {g_open:.2f}')
            M.move([gx, gy, loc['top'] + PREGRASP_ABOVE_TOP], yaw=yaw, grip=g_open, vmax=VMAX_TRAVEL,
                   label='pregrasp')
            M.r.grip_cmd = g_open
            M.move([gx, gy, z_grasp], yaw=yaw, grip=g_open, vmax=VMAX_NEAR, tol=0.004, label='descend')
            pz = r.pos()[2]
            if pz > z_grasp + 0.012:
                # the fingers landed on something: closing here would miss or pinch the rim
                log(f'descent blocked at z {pz - tz:.3f} above table (wanted {z_grasp - tz:.3f}); re-plan')
                M.move([gx, gy, loc['top'] + VIEW2_ABOVE_TOP], yaw=yaw, grip=-1.0, label='retreat')
                loc_n = self.locate(np.array([cx, cy]), radius=0.04, label='relook')
                if loc_n is not None:
                    loc = loc_n
                continue
            M.hold(8, 1.0)
            f = r.fingers()
            log(f'closed: fingers {f:.4f} (container width est {width:.3f})')
            if f < 0.008:
                log('grasp missed (fingers closed)')
                M.hold(5, -1.0)
                M.move([gx, gy, loc['top'] + VIEW2_ABOVE_TOP], yaw=yaw, grip=-1.0, label='retreat')
                loc_n = self.locate(np.array([cx, cy]), radius=0.04, label='relook')
                if loc_n is not None:
                    loc = loc_n
                continue
            # lift
            M.move([gx, gy, z_grasp + LIFT_H], yaw=yaw, grip=1.0, vmax=0.02, tol=0.008, label='lift')
            f2 = r.fingers()
            log(f'lifted: fingers {f2:.4f} success_flag {r.success}')
            if f2 < 0.008:
                log('lost the container during lift')
                M.hold(4, -1.0)
                M.move([gx, gy, tz + VIEW1_H - 0.05], yaw=yaw, grip=-1.0, label='rise-after-drop')
                loc_n = self.locate(np.array([cx, cy]), radius=0.05, label='relook2')
                if loc_n is not None:
                    loc = loc_n
                continue
            M.hold(6, 1.0)
            if last or r.success:
                return True
            # put back
            M.move([gx, gy, z_grasp + 0.006], yaw=yaw, grip=1.0, vmax=0.02, tol=0.005, label='putback')
            M.hold(6, -1.0)
            M.move([gx, gy, loc['top'] + 0.10], yaw=yaw, grip=-1.0, vmax=VMAX_TRAVEL, label='rise')
            return True
        return False

    def recover(self):
        """After an unexpected error: open the gripper and rise."""
        try:
            p = self.r.pos()
            self.motion.hold(4, -1.0)
            self.motion.move([p[0], p[1], self.table_z + VIEW1_H], grip=-1.0, label='recover')
        except EpisodeOver:
            raise
        except Exception as e:
            log('recover failed', repr(e)[:200])


def main(ctx):
    log('controller start', VERSION)
    try:
        task = Task(ctx)
    except Exception as e:
        log('FATAL init', repr(e)[:500], traceback.format_exc()[-2000:])
        return
    try:
        task.run()
    except EpisodeOver:
        log('episode over')
    except Exception as e:
        log('FATAL', repr(e)[:500], traceback.format_exc()[-3000:])
    log('controller end; success_flag', task.r.success, 'ended', task.r.ended, 'step', task.r.step_id,
        'calls', task.r.calls)
    try:
        if LOGF:
            LOGF.flush()
    except Exception:
        pass
