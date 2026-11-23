"""Physical button deadlines and mirrored occluding boards."""

import numpy as np

MOVE_STEPS = 32
POST_BUTTON_MOTION_STEPS = 96


class ButtonBoard:
    """React once to each distinct, accepted button press, in either order."""

    def __init__(self, origin, side, move_steps=MOVE_STEPS):
        self.origin = np.asarray(origin, dtype=float)
        self.side = np.asarray(side, dtype=float)
        self.move_steps = int(move_steps)
        self.position = self.origin.copy()
        self.start_position = self.origin.copy()
        self.destination = self.origin.copy()
        self.started = None
        self.seen = set()
        self.events = []
        self.history = []

    def advance(self, now):
        if self.started is not None:
            fraction = np.clip((now - self.started) / self.move_steps, 0, 1)
            blend = fraction * fraction * (3 - 2 * fraction)
            self.position = (1 - blend) * self.start_position + blend * self.destination
        return self.position.copy()

    def observe(self, now, accepted_presses):
        self.advance(now)
        for name, pressed_at in sorted(
            accepted_presses.items(), key=lambda item: (item[1], item[0])
        ):
            if name in self.seen:
                continue
            self.seen.add(name)
            self.start_position = self.position.copy()
            self.destination = (
                self.side if len(self.seen) == 1 else self.origin
            ).copy()
            self.started = int(now)
            self.events.append(
                {
                    "button": name,
                    "press_control_step": int(pressed_at),
                    "start_control_step": int(now),
                    "arrival_control_step": int(now) + self.move_steps,
                    "from_xyz_m": self.start_position.tolist(),
                    "to_xyz_m": self.destination.tolist(),
                }
            )

    @property
    def active(self):
        return not np.allclose(self.position, self.destination, atol=1e-9, rtol=0)


class ButtonDeadline:
    """Enforce a press deadline independently of whether motion ever stops."""

    PRESS_DEPTH_M = 0.005
    RELEASE_DEPTH_M = 0.002
    WITHDRAWAL_M = 0.03
    RELEASE_STEPS = 2

    def __init__(self, names, start, stop, post_press_steps=POST_BUTTON_MOTION_STEPS,
                 *, fixed_motion_stop=False, continuous_motion=False,
                 repeat_press_allowed=False, motion_stop_step=None):
        self.names, self.start, self.stop = tuple(names), int(start), int(stop)
        self.motion_limit = self.stop if motion_stop_step is None else int(motion_stop_step)
        self.post_press_steps = int(post_press_steps)
        self.fixed_motion_stop = bool(fixed_motion_stop)
        self.continuous_motion = bool(continuous_motion)
        self.repeat_press_allowed = bool(repeat_press_allowed)
        self.presses = {}
        self.depressed = set()
        self._press_tcp = {}
        self._release_started = {}
        self.press_events = []
        self.failed = False
        self.failed_at = None
        self.failure_reason = None
        self.repeated_button = None

    def observe(self, step, depths, moving, *, tcp_position):
        tcp = np.asarray(tcp_position, dtype=float)
        if not self.failed and step >= self.start:
            for name in self.names:
                if self.repeat_press_allowed and name in self.presses:
                    continue
                depth = depths.get(name, 0)
                if name in self.depressed:
                    delta = tcp - self._press_tcp[name]
                    withdrawn = max(delta[2], np.linalg.norm(delta[:2])) >= self.WITHDRAWAL_M
                    # A spring rebound alone does not end a pressing action.
                    # Require a clear lift/retreat and a stable release. Measure
                    # elapsed control steps, not the many physics callbacks.
                    if depth <= self.RELEASE_DEPTH_M and withdrawn:
                        since = self._release_started.setdefault(name, int(step))
                        if step - since >= self.RELEASE_STEPS:
                            self.depressed.discard(name)
                            self._release_started.pop(name, None)
                    else:
                        self._release_started.pop(name, None)
                if depth <= self.PRESS_DEPTH_M or name in self.depressed:
                    continue
                self.depressed.add(name)
                self._press_tcp[name] = tcp.copy()
                if name in self.presses:
                    self.press_events.append(dict(button=name, control_step=int(step), repeated=True))
                    self.failed = True
                    self.failed_at = int(step)
                    self.failure_reason = "repeated_button_press"
                    self.repeated_button = name
                    break
                if moving and step < self.stop:
                    self.presses[name] = int(step)
                    self.press_events.append(dict(button=name, control_step=int(step), repeated=False))
        if (not self.failed
                and step >= self.stop and len(self.presses) != len(self.names)):
            self.failed = True
            self.failed_at = int(step)
            self.failure_reason = "button_deadline_missed"
        return self.failed

    @property
    def motion_stop(self):
        if self.continuous_motion:
            return None
        if not self.fixed_motion_stop and len(self.presses) == len(self.names) and not self.failed:
            return min(self.motion_limit, max(self.presses.values()) + self.post_press_steps)
        return self.motion_limit

    def snapshot(self):
        return {
            "required_buttons": list(self.names),
            "press_control_steps": dict(self.presses),
            "press_events": list(self.press_events),
            "order": "either",
            "repeat_press_allowed": self.repeat_press_allowed,
            "press_depth_threshold_m": self.PRESS_DEPTH_M,
            "release_depth_threshold_m": self.RELEASE_DEPTH_M,
            "repeat_requires_gripper_withdrawal_m": self.WITHDRAWAL_M,
            "release_debounce_control_steps": self.RELEASE_STEPS,
            "motion_start_control_step": self.start,
            "deadline_control_step": self.stop,
            "motion_stop_control_step": self.motion_stop,
            "post_button_motion_steps": self.post_press_steps,
            "fixed_motion_stop": self.fixed_motion_stop,
            "continuous_motion": self.continuous_motion,
            "both_pressed_in_time": len(self.presses) == len(self.names)
            and not self.failed,
            "failed": self.failed,
            "failed_at_control_step": self.failed_at,
            "failure_reason": self.failure_reason,
            "repeated_button": self.repeated_button,
            "clock": "simulator control steps; API waiting excluded",
        }
