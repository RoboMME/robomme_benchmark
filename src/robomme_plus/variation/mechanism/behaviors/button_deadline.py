"""Keep the native press deadline while allowing the post-mask motion phase."""
from robomme_plus.tasks.permanence.utils.board import ButtonDeadline
from robomme_plus.tasks.permanence.utils.tasks import BUTTON_SWAP_MOTION_STOP_STEP


class ShuffleButtonDeadline(ButtonDeadline):
    @property
    def motion_stop(self):
        if self.continuous_motion:
            return None
        if not self.fixed_motion_stop and len(self.presses) == len(self.names) and not self.failed:
            origin = max(self.presses.values())
            shuffle = getattr(self._environment, '_visible_shuffle', None)
            if shuffle is not None:
                covered = shuffle['audit']['covered_step']
                if covered is None:
                    return self.original_motion_limit
                origin = max(origin, covered)
            return min(self.original_motion_limit, origin + self.post_press_steps)
        return self.original_motion_limit


def apply(env, config):
    deadline = env._permanence_deadline
    deadline.__class__ = ShuffleButtonDeadline
    deadline.original_motion_limit = BUTTON_SWAP_MOTION_STOP_STEP
    deadline._environment = env
    config['button_press_deadline'] = dict(exclusive_failure_control_step=deadline.stop,
        last_allowed_press_control_step=deadline.stop - 1)
    env._limit_diagnostics['button_press_deadline'] = config['button_press_deadline']
