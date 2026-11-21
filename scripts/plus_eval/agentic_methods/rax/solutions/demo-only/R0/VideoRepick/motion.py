"""Closed-loop end-effector motion on the measured state."""
import numpy as np
from io_utils import log, reply_json, get_state, episode_over, remaining_steps, manifest_path, get_step_id, find
from geom import axis_angle_to_R, R_to_axis_angle

POS_SCALE = 0.05   # m per unit action
ROT_SCALE = 0.5    # rad per unit action

class EpisodeOver(Exception):
    pass

class Robot:
    def __init__(self, ctx, packet):
        self.ctx = ctx
        self.packet = packet
        self.state = get_state(packet)
        self.step_id = get_step_id(packet) or 0
        self.R_goal = axis_angle_to_R(self.state[3:6]) if self.state is not None else None
        self.rot_enabled = True
        self.rot_hist = []
        self.grip = -1.0
        self.over = False
        self.complete = False
        self.calls = 0
        self.steps = 0
        self.remaining = remaining_steps(packet)

    # ------------------------------------------------------------ low level
    def send(self, rows):
        if self.over:
            raise EpisodeOver()
        if self.remaining is not None:
            rows = rows[:max(0, self.remaining)]
            if not rows:
                self.over = True
                raise EpisodeOver()
        rows = [[float(np.clip(x, -1, 1)) for x in r] for r in rows]
        try:
            res = self.ctx.call('robomme_step', actions=rows)
        except Exception as e:
            log('robomme_step error:', repr(e)[:300])
            # check status rather than retrying blindly
            try:
                st = reply_json(self.ctx.call('robomme_status'))
                if episode_over(st):
                    self.over = True
                    raise EpisodeOver()
            except EpisodeOver:
                raise
            except Exception:
                pass
            self.refresh()
            return None
        d = reply_json(res)
        self.packet = d
        self.calls += 1; self.steps += len(rows)
        s = get_state(d)
        if s is not None:
            self.state = s
        sid = get_step_id(d)
        if sid is not None:
            self.step_id = sid
        rem = remaining_steps(d)
        if rem is not None:
            self.remaining = rem
        if find(d, 'task_complete') is True:
            self.complete = True
        if episode_over(d):
            self.over = True
            log('episode over reported; task_complete=', find(d, 'task_complete'))
            raise EpisodeOver()
        return d

    def refresh(self):
        try:
            d = reply_json(self.ctx.call('robomme_observe'))
            self.packet = d
            s = get_state(d)
            if s is not None:
                self.state = s
            sid = get_step_id(d)
            if sid is not None:
                self.step_id = sid
            if episode_over(d):
                self.over = True
                raise EpisodeOver()
            return d
        except EpisodeOver:
            raise
        except Exception as e:
            log('observe failed', repr(e)[:200])
            return None

    @property
    def pos(self):
        return self.state[0:3].copy()

    @property
    def fingers(self):
        return float(self.state[6] + self.state[7])

    def rot_correction(self):
        if not self.rot_enabled or self.R_goal is None:
            return np.zeros(3)
        R = axis_angle_to_R(self.state[3:6])
        err = R_to_axis_angle(self.R_goal @ R.T)
        e = float(np.linalg.norm(err))
        self.rot_hist.append(e)
        if len(self.rot_hist) >= 6 and all(self.rot_hist[-i] > self.rot_hist[-i - 1] + 1e-3 for i in range(1, 5)) and e > 0.12:
            log(f'orientation correction diverging (err {e:.3f}); disabling')
            self.rot_enabled = False
            return np.zeros(3)
        if e < 0.015:
            return np.zeros(3)
        return np.clip(err * 0.6 / ROT_SCALE, -0.25, 0.25)

    # ------------------------------------------------------------ primitives
    def hold(self, n, grip=None):
        if grip is not None:
            self.grip = grip
        rows = [[0, 0, 0, 0, 0, 0, self.grip]] * n
        self.send(rows)

    def set_gripper(self, g, n=10):
        self.grip = g
        self.hold(n)

    def move_to(self, goal, vmax=0.025, tol=0.004, max_calls=40, chunk=6, label=''):
        goal = np.asarray(goal, float)
        best = 1e9; stall = 0
        for it in range(max_calls):
            err = goal - self.pos
            dist = float(np.linalg.norm(err))
            if dist < tol:
                return True
            if dist < best - 0.001:
                best = dist; stall = 0
            else:
                stall += 1
                if stall >= 4:
                    log(f'move_to {label}: stalled at dist {dist:.4f}')
                    return False
            n = int(np.clip(np.ceil(dist / vmax), 1, chunk))
            # near the goal use single steps for precision
            if dist < 0.015:
                n = 1
            step = err / n
            rot = self.rot_correction()
            rows = []
            for k in range(n):
                r = list(step / POS_SCALE) + (list(rot) if k == 0 else [0, 0, 0]) + [self.grip]
                rows.append(r)
            self.send(rows)
        log(f'move_to {label}: max calls, dist {np.linalg.norm(goal - self.pos):.4f}')
        return False

    def rotate_to(self, R_goal, tol=0.03, max_calls=25, label='rot'):
        """Rotate in place to R_goal (world frame), keeping position."""
        self.R_goal = np.asarray(R_goal, float)
        p_hold = self.pos
        best = 1e9; stall = 0
        for it in range(max_calls):
            R = axis_angle_to_R(self.state[3:6])
            err = R_to_axis_angle(self.R_goal @ R.T)
            e = float(np.linalg.norm(err))
            if e < tol:
                return True
            if e < best - 0.005:
                best = e; stall = 0
            else:
                stall += 1
                if stall >= 4:
                    log(f'rotate_to {label}: stalled at err {e:.3f}')
                    return False
            rot = np.clip(err * 0.8 / ROT_SCALE, -0.25, 0.25)
            dp = (p_hold - self.pos) / POS_SCALE
            self.send([list(dp) + list(rot) + [self.grip]])
        log(f'rotate_to {label}: max calls, err {e:.3f}')
        return False

    def open_to(self, width, n=8):
        """Open the gripper to about `width` (total finger gap, m). Learns the command->width
        map online (assumes a continuous gripper); if the gripper behaves as binary it is
        flagged and simply opened fully from then on."""
        width = float(np.clip(width, 0.02, 0.08))
        if getattr(self, 'grip_binary', False):
            self.grip = -1.0; self.hold(n); return self.fingers
        cal = getattr(self, 'grip_cal', [])
        g = 1.0 - 2.0 * width / 0.08
        if len(cal) >= 2:
            ws = np.array([b for _, b in cal]); gs = np.array([a for a, _ in cal])
            if np.ptp(ws) > 0.01:
                o = np.argsort(ws); g = float(np.interp(width, ws[o], gs[o]))
        for it in range(2):
            g = float(np.clip(g, -0.9, 0.6))
            self.grip = g
            self.hold(n)
            w = self.fingers
            cal = cal + [(g, w)]; self.grip_cal = cal
            log(f'open_to {width:.3f}: g={g:.2f} -> width {w:.3f}')
            if abs(w - width) < 0.007:
                return w
            if w <= 0.005:
                log('gripper closed at intermediate command; treating as binary'); self.grip_binary = True; break
            g = g + float(np.clip((w - width) / 0.05, -0.4, 0.4))
        if getattr(self, 'grip_binary', False) or self.fingers < width - 0.006:
            self.grip = -1.0; self.hold(n)
        if len(cal) >= 2 and all(w >= 0.072 for _, w in cal[-2:]):
            log('gripper stays fully open at intermediate commands; treating as binary'); self.grip_binary = True
        return self.fingers
