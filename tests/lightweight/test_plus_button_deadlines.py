"""Button deadlines fail at the boundary, independently of motion and scoring."""

from types import SimpleNamespace

import numpy as np
import pytest

from robomme_plus.scoring.failures import failure_message
from robomme_plus.tasks.permanence.utils import tasks
from robomme_plus.tasks.permanence.utils.board import ButtonDeadline


def observe(deadline, step, depths=None):
    return deadline.observe(step, depths or {}, True, tcp_position=[0, 0, .03])


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_task_deadlines_do_not_depend_on_difficulty(level):
    assert tasks.task_config("ButtonUnmask", level, 0)["button_deadline_steps"] == 90
    assert tasks.task_config("ButtonUnmaskSwap", level, 0)["button_deadline_steps"] == 180


@pytest.mark.parametrize("press_step", [0, 89, 90, 91])
def test_single_button_deadline_boundary(press_step):
    deadline = ButtonDeadline(("button",), 0, 90, repeat_press_allowed=True)
    assert observe(deadline, press_step, {"button": .01}) == (press_step >= 90)
    assert deadline.presses == ({"button": press_step} if press_step < 90 else {})
    assert observe(deadline, 100, {"button": .01}) == (press_step >= 90)
    if press_step >= 90:
        assert deadline.failure_reason == "button_deadline_missed"
        assert deadline.failed_at == press_step


@pytest.mark.parametrize("continuous", [False, True])
@pytest.mark.parametrize("second_press", [179, 180, 181])
def test_both_swap_variants_require_both_buttons_before_180(continuous, second_press):
    deadline = ButtonDeadline(("left", "right"), 10, 180,
                              continuous_motion=continuous, motion_stop_step=480)
    assert not observe(deadline, 20, {"left": .01})
    assert observe(deadline, second_press, {"right": .01}) == (second_press >= 180)
    assert deadline.snapshot()["deadline_control_step"] == 180
    assert observe(deadline, 1000) == (second_press >= 180)
    if continuous:
        assert deadline.motion_stop is None
    elif second_press < 180:
        assert deadline.motion_stop == second_press + 96
    if second_press >= 180:
        assert deadline.failure_reason == "button_deadline_missed"
        assert deadline.presses == {"left": 20}
        assert deadline.failed_at == second_press


def test_single_button_repeat_does_not_add_an_unrelated_failure_rule():
    deadline = ButtonDeadline(("button",), 0, 90, repeat_press_allowed=True)
    assert not observe(deadline, 60, {"button": .01})
    for step in range(61, 65):
        assert not deadline.observe(step, {}, True, tcp_position=[0, 0, .1])
    assert not observe(deadline, 100, {"button": .01})
    assert deadline.presses == {"button": 60}
    assert deadline.snapshot()["repeat_press_allowed"]


@pytest.mark.parametrize("task,continuous,limit", [
    ("ButtonUnmask", False, 90),
    ("ButtonUnmaskSwap", False, 180),
    ("ButtonUnmaskSwap", True, 180),
])
def test_task_scoring_latches_deadline_failure_without_a_press(task, continuous, limit):
    names = ("button",) if task == "ButtonUnmask" else ("button_left", "button_right")
    deadline = ButtonDeadline(names, 0, limit, continuous_motion=continuous,
                              repeat_press_allowed=task == "ButtonUnmask")
    env = SimpleNamespace(
        elapsed_steps=np.array([limit - 1]),
        agent=SimpleNamespace(tcp=SimpleNamespace(pose=SimpleNamespace(p=np.zeros(3)))),
        _limit_config={"task": task}, _limit_failed=False,
        _permanence_deadline=deadline, _permanence_presses={},
        _permanence_board=None, _permanence_cover_spawned=set(), _limit_diagnostics={},
        task_list=[tasks.base.task("press buttons", lambda: False, None)],
    )
    for name in names:
        setattr(env, name, SimpleNamespace(name=name, get_qpos=lambda: np.zeros((1, 1))))

    assert not tasks.evaluate(env)["fail"].item()
    env.elapsed_steps[0] = limit
    info = tasks.evaluate(env)
    assert info["fail"].item() and not info["success"].item()
    assert env._failure_code == "button_deadline_missed"
    assert failure_message(env) == "Did not press all required buttons before the deadline."
    assert env._limit_diagnostics["button_deadline"]["failed_at_control_step"] == limit

    # Later presses or completion of every subgoal cannot undo a terminal failure.
    env.elapsed_steps[0] += 1
    for name in names:
        getattr(env, name).get_qpos = lambda: np.array([[-.01]])
    env.task_list = []
    info = tasks.evaluate(env)
    assert info["fail"].item() and not info["success"].item()
    assert not deadline.presses
