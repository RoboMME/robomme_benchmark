import importlib
import unittest
from unittest.mock import patch

import numpy as np
import torch


_WRAPPER_MODULE = importlib.import_module(
    "robomme.env_record_wrapper.OraclePlannerDemonstrationWrapper"
)
OraclePlannerDemonstrationWrapper = _WRAPPER_MODULE.OraclePlannerDemonstrationWrapper


class _Actor:
    def __init__(self, name):
        self.name = name


class _PlannerEnv:
    def __init__(self):
        self.step_calls = 0

    def step(self, _action):
        self.step_calls += 1
        return (
            {"obs_key": self.step_calls},
            float(self.step_calls),
            False,
            False,
            {"info_key": f"info-{self.step_calls}"},
        )


class _FakePlanner:
    def __init__(self):
        self.env = _PlannerEnv()


class _FakeEnv:
    def __init__(self, seg_raw, seg_id_map=None):
        self._seg_raw = np.asarray(seg_raw, dtype=np.int64)
        self.segmentation_id_map = seg_id_map or {}
        self.evaluate_calls = []
        self.unwrapped = self

    def get_obs(self, unflattened=True):
        del unflattened
        return {
            "sensor_data": {
                "base_camera": {
                    "segmentation": self._seg_raw,
                }
            }
        }

    def evaluate(self, solve_complete_eval=False):
        self.evaluate_calls.append(bool(solve_complete_eval))
        return {"solve_complete_eval": bool(solve_complete_eval)}


def _build_wrapper(seg_raw, seg_id_map=None):
    wrapper = object.__new__(OraclePlannerDemonstrationWrapper)
    wrapper.env = _FakeEnv(seg_raw=seg_raw, seg_id_map=seg_id_map)
    wrapper.env_id = "TestEnv"
    wrapper.gui_render = False
    wrapper.planner = _FakePlanner()
    wrapper.seg_raw = None
    wrapper.available_options = []
    return wrapper


class TestOraclePlannerWrapperStep(unittest.TestCase):
    def _single_option_patch(self, label="pick", available=None, solve=None):
        if solve is None:
            solve = lambda: 0
        return patch.object(
            _WRAPPER_MODULE,
            "get_vqa_options",
            return_value=[{"label": label, "available": available, "solve": solve}],
        )

    def test_non_dict_action_returns_empty_batch(self):
        wrapper = _build_wrapper(seg_raw=np.array([[0, 0], [0, 0]], dtype=np.int64))
        with self._single_option_patch():
            obs, reward, terminated, truncated, info = wrapper.step("invalid")

        self.assertEqual(obs, {})
        self.assertIsInstance(reward, torch.Tensor)
        self.assertEqual(reward.numel(), 0)
        self.assertIsInstance(terminated, torch.Tensor)
        self.assertEqual(terminated.numel(), 0)
        self.assertIsInstance(truncated, torch.Tensor)
        self.assertEqual(truncated.numel(), 0)
        self.assertEqual(info, {})
        self.assertEqual(wrapper.env.evaluate_calls, [])

    def test_missing_or_none_action_returns_empty_batch(self):
        wrapper = _build_wrapper(seg_raw=np.array([[0, 0], [0, 0]], dtype=np.int64))
        with self._single_option_patch():
            for action in ({}, {"action": None}):
                obs, reward, terminated, truncated, info = wrapper.step(action)
                self.assertEqual(obs, {})
                self.assertEqual(reward.numel(), 0)
                self.assertEqual(terminated.numel(), 0)
                self.assertEqual(truncated.numel(), 0)
                self.assertEqual(info, {})

    def test_unmatched_action_returns_empty_batch(self):
        wrapper = _build_wrapper(seg_raw=np.array([[0, 0], [0, 0]], dtype=np.int64))
        with self._single_option_patch(label="pick"):
            obs, reward, terminated, truncated, info = wrapper.step({"action": "place"})

        self.assertEqual(obs, {})
        self.assertEqual(reward.numel(), 0)
        self.assertEqual(terminated.numel(), 0)
        self.assertEqual(truncated.numel(), 0)
        self.assertEqual(info, {})
        self.assertEqual(wrapper.env.evaluate_calls, [])

    def test_point_hit_updates_selected_target_and_executes(self):
        actor = _Actor("cube")
        wrapper = _build_wrapper(
            seg_raw=np.array(
                [
                    [1, 0],
                    [0, 0],
                ],
                dtype=np.int64,
            ),
            seg_id_map={1: actor},
        )
        captured_target = {}

        def _get_options(_env, planner, selected_target, _env_id):
            def _solve():
                captured_target.update(selected_target)
                planner.env.step("run")
                return 0

            return [{"label": "pick", "available": [actor], "solve": _solve}]

        with patch.object(_WRAPPER_MODULE, "get_vqa_options", side_effect=_get_options):
            obs, reward, terminated, truncated, info = wrapper.step(
                {"action": "pick", "point": [0, 0]}
            )

        self.assertEqual(obs["obs_key"], [1])
        self.assertEqual(float(reward), 1.0)
        self.assertFalse(bool(terminated))
        self.assertFalse(bool(truncated))
        self.assertEqual(info["info_key"], "info-1")

        self.assertIs(captured_target["obj"], actor)
        self.assertEqual(captured_target["seg_id"], 1)
        self.assertEqual(captured_target["click_point"], (0, 0))
        self.assertEqual(captured_target["centroid_point"], (0, 0))
        self.assertEqual(wrapper.env.evaluate_calls, [False, True])

    def test_point_miss_sets_only_click_point_when_no_available_candidates(self):
        wrapper = _build_wrapper(seg_raw=np.array([[0, 0], [0, 0]], dtype=np.int64))
        captured_target = {}

        def _get_options(_env, planner, selected_target, _env_id):
            def _solve():
                captured_target.update(selected_target)
                planner.env.step("run")
                return 0

            return [{"label": "pick", "available": [], "solve": _solve}]

        with patch.object(_WRAPPER_MODULE, "get_vqa_options", side_effect=_get_options):
            wrapper.step({"action": "pick", "point": [9, 9]})

        self.assertIsNone(captured_target["obj"])
        self.assertIsNone(captured_target["seg_id"])
        self.assertEqual(captured_target["click_point"], (1, 1))
        self.assertIsNone(captured_target["centroid_point"])

    def test_solve_minus_one_raises_runtime_error(self):
        wrapper = _build_wrapper(seg_raw=np.array([[0, 0], [0, 0]], dtype=np.int64))

        with self._single_option_patch(solve=lambda: -1):
            with self.assertRaises(RuntimeError):
                wrapper.step({"action": "pick"})

        self.assertEqual(wrapper.env.evaluate_calls, [])


if __name__ == "__main__":
    unittest.main()
