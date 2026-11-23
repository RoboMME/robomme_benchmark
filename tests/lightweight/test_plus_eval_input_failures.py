"""Invalid actions and failed demos must not silently reach policy execution."""
from types import SimpleNamespace
from unittest.mock import Mock, patch

import gymnasium as gym
import numpy as np
import pytest

from robomme_plus.control.actions import normalize_joint_action
from robomme_plus.control.ee_pose import EndeffectorDemonstrationWrapper
from robomme_plus.control.waypoint import MultiStepDemonstrationWrapper
from robomme_plus.evaluation.demonstration import DemonstrationWrapper
from robomme_plus.evaluation.clock import ExecutionClock
from robomme_plus.evaluation.failure import FailAwareWrapper


@pytest.mark.parametrize('task,size', [('BinFill', 8), ('PatternLock', 7), ('RouteStick', 7)])
@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_joint_actions_reject_nonfinite_values(task, size, bad):
    action = np.zeros(size)
    action[-1] = bad
    with pytest.raises(ValueError, match='finite'):
        normalize_joint_action(SimpleNamespace(task_name=task), action)


@pytest.mark.parametrize('wrapper', [EndeffectorDemonstrationWrapper, MultiStepDemonstrationWrapper])
@pytest.mark.parametrize('index', [0, 3, 6])
@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_pose_actions_reject_nonfinite_values_before_planning(wrapper, index, bad):
    native = gym.Env()
    env = wrapper(native)
    action = np.zeros(7)
    action[index] = bad
    # No robot or planner exists: rejection must happen before accessing either.
    with pytest.raises(ValueError, match='finite'):
        env.step(action)


def demo_harness(first_solve):
    later = Mock()
    native = SimpleNamespace(task_name='VideoRepick', spec=SimpleNamespace(id='test'),
        agent=SimpleNamespace(robot=SimpleNamespace(pose=None)), demonstration_record_traj=False)
    return SimpleNamespace(unwrapped=native,
        task_list=[dict(demonstration=True, name='show target', solve=first_solve),
                   dict(demonstration=True, name='later demo', solve=later)],
        _demo_screw_max_attempts=1, _demo_rrt_max_attempts=3,
        evaluate=Mock(), step=Mock()), later


@pytest.mark.parametrize('raises', [False, True])
def test_failed_demo_is_rejected_before_later_demo_tasks(raises):
    from robomme_plus.control.planner import ScrewPlanFailure
    def failed(_env, _planner):
        if raises:
            raise ScrewPlanFailure('cannot reach demonstration target')
        return -1
    env, later = demo_harness(failed)
    planner = SimpleNamespace(env=env, move_to_pose_with_screw=Mock(), move_to_pose_with_RRTStar=Mock())
    with patch('robomme_plus.control.planner.FailAwarePandaArmMotionPlanningSolver', return_value=planner):
        with pytest.raises(RuntimeError, match='show target'):
            DemonstrationWrapper.get_demonstration_trajectory(env)
    later.assert_not_called()
    assert not env.unwrapped.demonstration_record_traj


@pytest.mark.parametrize('recovered', [False, True])
def test_demo_rrt_recovery_is_allowed_but_exhaustion_is_an_error(recovered):
    from robomme_plus.control.planner import ScrewPlanFailure
    def solve(_env, planner):
        planner.move_to_pose_with_screw(None)
    env, later = demo_harness(solve)
    env.step.return_value = ({'frame': np.zeros((1,))}, 0., False, False, {})
    planner = SimpleNamespace(env=env,
        move_to_pose_with_screw=Mock(side_effect=ScrewPlanFailure('screw failed')),
        move_to_pose_with_RRTStar=Mock(side_effect=(lambda *_: env.step(None)) if recovered else None,
                                      return_value=-1))
    with patch('robomme_plus.control.planner.FailAwarePandaArmMotionPlanningSolver', return_value=planner):
        if recovered:
            batch = DemonstrationWrapper.get_demonstration_trajectory(env)
            assert len(batch[0]['frame']) == 1
            later.assert_called_once()
        else:
            with pytest.raises(RuntimeError, match='show target'):
                DemonstrationWrapper.get_demonstration_trajectory(env)
            later.assert_not_called()
    assert planner.move_to_pose_with_RRTStar.call_count == (1 if recovered else 3)
    assert not env.unwrapped.demonstration_record_traj


def test_nonfinite_joint_action_consumes_error_budget_without_physics():
    class Native(gym.Env):
        task_name = 'BinFill'
        def __init__(self):
            self.execution_clock = ExecutionClock(dict(max_steps=250, oracle_policy_step=10))
        def reset(self, **kwargs):
            self.steps = 0
            self.execution_clock.start()
            return self.observe_current()
        def observe_current(self):
            return {'frame': [self.steps]}, {'status': 'ongoing'}
        def step(self, action):
            normalize_joint_action(self, action)
            self.steps += 1
            self.execution_clock.advance()
            obs, info = self.observe_current()
            return obs, 0., False, False, info
    native = Native()
    env = FailAwareWrapper(native)
    env.reset()
    _, _, terminated, truncated, info = env.step([np.nan] * 8)
    assert not (terminated or truncated)
    assert info['action_error'] and info['exception_type'] == 'ValueError'
    assert info['execution_penalty_steps'] == 100 and native.steps == 0
    env.step(np.zeros(8))
    assert native.steps == 1 and native.execution_clock.steps == 101
