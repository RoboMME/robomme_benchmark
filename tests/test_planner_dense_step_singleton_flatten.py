import importlib.util
from pathlib import Path
import unittest

import torch


def _load_planner_dense_step_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "src/robomme/robomme_env/utils/planner_denseStep.py"
    )
    spec = importlib.util.spec_from_file_location("planner_dense_step", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_PLANNER_DENSE_STEP = _load_planner_dense_step_module()


class TestPlannerDenseStepSingletonFlatten(unittest.TestCase):
    def test_to_step_batch_flattens_singleton_list_tensor(self):
        frame = torch.tensor([[1, 2], [3, 4]], dtype=torch.uint8)
        step = (
            {"front_rgb_list": [frame]},
            0.0,
            False,
            False,
            {"simple_subgoal_online": ["pick"]},
        )

        obs_batch, _, _, _, info_batch = _PLANNER_DENSE_STEP.to_step_batch([step])

        self.assertIsInstance(obs_batch["front_rgb_list"][0], torch.Tensor)
        self.assertTrue(torch.equal(obs_batch["front_rgb_list"][0], frame))
        self.assertEqual(info_batch["simple_subgoal_online"][0], "pick")

    def test_to_step_batch_flattens_nested_singleton_lists_recursively(self):
        frame = torch.tensor([[9]], dtype=torch.uint8)
        step = (
            {"front_rgb_list": [[frame]]},
            0.0,
            False,
            False,
            {"simple_subgoal_online": [[["align"]]]},
        )

        obs_batch, _, _, _, info_batch = _PLANNER_DENSE_STEP.to_step_batch([step])

        self.assertIsInstance(obs_batch["front_rgb_list"][0], torch.Tensor)
        self.assertTrue(torch.equal(obs_batch["front_rgb_list"][0], frame))
        self.assertEqual(info_batch["simple_subgoal_online"][0], "align")

    def test_to_step_batch_keeps_non_singleton_lists(self):
        options = ["open", "close"]
        step = (
            {"available_options": options},
            0.0,
            False,
            False,
            {"scores": [0.2, 0.8]},
        )

        obs_batch, _, _, _, info_batch = _PLANNER_DENSE_STEP.to_step_batch([step])

        self.assertIsInstance(obs_batch["available_options"][0], list)
        self.assertEqual(obs_batch["available_options"][0], options)
        self.assertEqual(info_batch["scores"][0], [0.2, 0.8])

    def test_concat_step_batches_flattens_legacy_nested_values(self):
        batch_a = (
            {"front_rgb_list": [[torch.tensor([1], dtype=torch.int64)]]},
            torch.tensor([0.0], dtype=torch.float32),
            torch.tensor([False], dtype=torch.bool),
            torch.tensor([False], dtype=torch.bool),
            {"simple_subgoal_online": [["A"]]},
        )
        batch_b = (
            {"front_rgb_list": [[[torch.tensor([2], dtype=torch.int64)]]]},
            torch.tensor([1.0], dtype=torch.float32),
            torch.tensor([False], dtype=torch.bool),
            torch.tensor([False], dtype=torch.bool),
            {"simple_subgoal_online": [[["B"]]]},
        )

        obs_out, reward_out, _, _, info_out = _PLANNER_DENSE_STEP.concat_step_batches(
            [batch_a, batch_b]
        )

        self.assertEqual(int(reward_out.numel()), 2)
        self.assertIsInstance(obs_out["front_rgb_list"][0], torch.Tensor)
        self.assertIsInstance(obs_out["front_rgb_list"][1], torch.Tensor)
        self.assertTrue(
            torch.equal(obs_out["front_rgb_list"][0], torch.tensor([1], dtype=torch.int64))
        )
        self.assertTrue(
            torch.equal(obs_out["front_rgb_list"][1], torch.tensor([2], dtype=torch.int64))
        )
        self.assertEqual(info_out["simple_subgoal_online"][0], "A")
        self.assertEqual(info_out["simple_subgoal_online"][1], "B")


if __name__ == "__main__":
    unittest.main()
