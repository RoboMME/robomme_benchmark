import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

from robomme.env_record_wrapper.EndeffectorDemonstrationWrapper import (
    EndeffectorDemonstrationWrapper,
)
from robomme.env_record_wrapper.RecordWrapper import RobommeRecordWrapper


class _DummyPlannerCore:
    def __init__(self, n_joints=9):
        self.move_group = "panda_hand_tcp"
        self.link_name_2_idx = {self.move_group: 9}
        self.user_joint_names = [f"joint{i}" for i in range(n_joints)]

    def transform_goal_to_wrt_base(self, goal_world):
        return np.asarray(goal_world, dtype=np.float64)

    def IK(self, goal_base, current_qpos):
        return "Success", [np.zeros(9, dtype=np.float64)]


class _DummyRobot:
    def get_qpos(self):
        return torch.zeros((1, 9), dtype=torch.float64)


def _solver_ctor(tag, calls):
    n_joints = 7 if tag == "stick" else 9

    class _DummySolver:
        def __init__(self, *args, **kwargs):
            calls.append((tag, kwargs))
            self.planner = _DummyPlannerCore(n_joints)
            self.robot = _DummyRobot()

    return _DummySolver


class _FakeEnv:
    def __init__(self, env_id):
        self.unwrapped = types.SimpleNamespace(
            spec=types.SimpleNamespace(id=env_id),
            agent=types.SimpleNamespace(robot=types.SimpleNamespace(pose="base_pose")),
        )
        self.last_action = None

    def step(self, action):
        self.last_action = np.asarray(action, dtype=np.float64)
        return "ok"


def _build_ee_wrapper(env_id, action_repr="rpy"):
    wrapper = object.__new__(EndeffectorDemonstrationWrapper)
    wrapper.env = _FakeEnv(env_id)
    wrapper.action_repr = action_repr
    wrapper._ee_pose_planner = None
    return wrapper


def _build_record_wrapper(spec_id, robomme_env=None, include_spec=True):
    env_unwrapped = types.SimpleNamespace(
        agent=types.SimpleNamespace(robot=types.SimpleNamespace(pose="base_pose"))
    )
    if include_spec:
        env_unwrapped.spec = types.SimpleNamespace(id=spec_id)

    wrapper = object.__new__(RobommeRecordWrapper)
    wrapper.env = types.SimpleNamespace(unwrapped=env_unwrapped)
    wrapper.Robomme_env = robomme_env
    return wrapper


class TestStickEnvHelpers(unittest.TestCase):
    def test_is_stick_env_id(self):
        _STICK_IDS = ("PatternLock", "RouteStick")
        self.assertTrue("PatternLock" in _STICK_IDS)
        self.assertTrue("RouteStick" in _STICK_IDS)
        self.assertFalse("MoveCube" in _STICK_IDS)


class TestEndeffectorPlannerSelection(unittest.TestCase):
    def test_stick_env_uses_stick_solver(self):
        calls = []
        with (
            patch(
                "robomme.env_record_wrapper.EndeffectorDemonstrationWrapper.PandaStickMotionPlanningSolver",
                _solver_ctor("stick", calls),
            ),
            patch(
                "robomme.env_record_wrapper.EndeffectorDemonstrationWrapper.PandaArmMotionPlanningSolver",
                _solver_ctor("arm", calls),
            ),
        ):
            wrapper = _build_ee_wrapper("PatternLock", action_repr="rpy")
            result = wrapper.step(np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0], dtype=np.float64))

        self.assertEqual(result, "ok")
        self.assertEqual(calls[0][0], "stick")
        self.assertEqual(wrapper.env.last_action.shape, (7,))
        self.assertEqual(calls[0][1]["joint_vel_limits"], 0.3)

    def test_non_stick_env_uses_arm_solver(self):
        calls = []
        with (
            patch(
                "robomme.env_record_wrapper.EndeffectorDemonstrationWrapper.PandaStickMotionPlanningSolver",
                _solver_ctor("stick", calls),
            ),
            patch(
                "robomme.env_record_wrapper.EndeffectorDemonstrationWrapper.PandaArmMotionPlanningSolver",
                _solver_ctor("arm", calls),
            ),
        ):
            wrapper = _build_ee_wrapper("MoveCube", action_repr="rpy")
            result = wrapper.step(
                np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, -1.0], dtype=np.float64)
            )

        self.assertEqual(result, "ok")
        self.assertEqual(calls[0][0], "arm")
        self.assertEqual(wrapper.env.last_action.shape, (8,))


class TestRecordFkPlannerSelection(unittest.TestCase):
    def test_stick_env_uses_stick_fk_solver(self):
        calls = []
        with (
            patch(
                "robomme.env_record_wrapper.RecordWrapper.PandaStickMotionPlanningSolver",
                _solver_ctor("stick", calls),
            ),
            patch(
                "robomme.env_record_wrapper.RecordWrapper.PandaArmMotionPlanningSolver",
                _solver_ctor("arm", calls),
            ),
        ):
            wrapper = _build_record_wrapper("RouteStick")
            wrapper._init_fk_planner()

        self.assertEqual(calls[0][0], "stick")
        self.assertTrue(wrapper._fk_available)
        self.assertEqual(wrapper._ee_link_idx, 9)
        self.assertEqual(calls[0][1]["joint_vel_limits"], 0.3)

    def test_non_stick_env_uses_arm_fk_solver(self):
        calls = []
        with (
            patch(
                "robomme.env_record_wrapper.RecordWrapper.PandaStickMotionPlanningSolver",
                _solver_ctor("stick", calls),
            ),
            patch(
                "robomme.env_record_wrapper.RecordWrapper.PandaArmMotionPlanningSolver",
                _solver_ctor("arm", calls),
            ),
        ):
            wrapper = _build_record_wrapper("MoveCube")
            wrapper._init_fk_planner()

        self.assertEqual(calls[0][0], "arm")
        self.assertTrue(wrapper._fk_available)
        self.assertEqual(wrapper._ee_link_idx, 9)

    def test_missing_spec_id_falls_back_to_robomme_env(self):
        calls = []
        with (
            patch(
                "robomme.env_record_wrapper.RecordWrapper.PandaStickMotionPlanningSolver",
                _solver_ctor("stick", calls),
            ),
            patch(
                "robomme.env_record_wrapper.RecordWrapper.PandaArmMotionPlanningSolver",
                _solver_ctor("arm", calls),
            ),
        ):
            wrapper = _build_record_wrapper(
                spec_id=None, robomme_env="PatternLock", include_spec=False
            )
            wrapper._init_fk_planner()

        self.assertEqual(calls[0][0], "stick")
        self.assertTrue(wrapper._fk_available)
        self.assertEqual(wrapper._ee_link_idx, 9)


if __name__ == "__main__":
    unittest.main()
