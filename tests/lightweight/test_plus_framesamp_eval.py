"""Check history continuity and OOD stopping across upstream action chunks."""

import asyncio
import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np



ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("framesamp_ood", ROOT / "scripts/plus_eval/non_agentic_methods/framesamp_modul/eval_plus.py")
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)


class FakeEnvironment:
    def __init__(self, outcome="success"):
        self.actions = []
        self.outcome = outcome
        self.close_calls = 0

    def observation(self, indices):
        return {
            "front_rgb_list": [np.full((4, 4, 3), i, np.uint8) for i in indices],
            "wrist_rgb_list": [np.full((4, 4, 3), i + 100, np.uint8) for i in indices],
            "joint_state_list": [np.arange(7) + i for i in indices],
            "gripper_state_list": [np.array([0.03, 0.99]) for i in indices],
        }

    def reset(self):
        return self.observation(range(4)), {
            "task_goal": ["Put cubes in the bin"], "status": "ongoing",
            "grounded_subgoal_online": "PRIVILEGED", "oracle_policy_step": 1118,
        }

    def step(self, action):
        self.actions.append(action.copy())
        n = len(self.actions)
        if self.outcome == "error" and n == 19:
            return None, 0, True, True, {"status": "error", "error": "simulation failed"}
        ended = n == 19
        return self.observation([n + 3]), 0, ended, False, {
            "execution_control_steps": 1300 + n, "status": self.outcome if ended else "ongoing",
        }

    def close(self):
        self.close_calls += 1


class FrameSampOODTest(unittest.TestCase):
    def run_episode(self, outcome="success", env=None, method="framesamp-modul"):
        env = env or FakeEnvironment(outcome)
        builder = SimpleNamespace(make_env_for_episode=lambda _: env, episode_max_steps=lambda _: 1877)
        client = Mock()
        client.reset.return_value = {"reset_finished": True}
        client.add_buffer.return_value = {"add_buffer_finished": True}
        client.infer.return_value = {"actions": np.arange(32 * 8).reshape(32, 8) / 100}
        upstream = adapter.load_upstream_utils()
        with tempfile.TemporaryDirectory() as tmp, patch("evaluation.VideoRecorder"), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            evaluation = adapter.FrameSampEvaluation(SimpleNamespace())
            evaluation.method = method
            evaluation.client, evaluation.upstream = client, upstream
            result = asyncio.run(evaluation.run_episode(builder, "BinFill", 0, Path(tmp) / "episode"))
            saved = np.load(Path(tmp) / "episode/actions.npy")
        return env, client, result, saved

    def test_history_demo_boundary_and_chunk_order(self):
        env, client, result, saved = self.run_episode()
        self.assertTrue(result["success"])
        self.assertEqual(result["steps"], 1319)  # No upstream 1300-step cap.
        self.assertEqual(result["metrics"]["components"]["policy"]["count"], 2)
        self.assertEqual(env.close_calls, 1)
        client.reset.assert_called_once()
        initial, following = [call.args[0] for call in client.add_buffer.call_args_list]
        np.testing.assert_array_equal(initial["images"][:, 0, 0, 0, 0], range(4))
        np.testing.assert_array_equal(following["images"][:, 0, 0, 0, 0], range(4, 20))
        self.assertEqual(initial["exec_start_idx"], 3)
        self.assertEqual(following["exec_start_idx"], 0)
        self.assertEqual(initial["images"].shape, (4, 1, 4, 4, 3))
        self.assertEqual(initial["state"].dtype, np.float32)
        np.testing.assert_allclose(initial["state"][0], [0, 1, 2, 3, 4, 5, 6, 0.03])
        for call, frame in zip(client.infer.call_args_list, [3, 19]):
            self.assertEqual(set(call.args[0]), {
                "observation/image", "observation/wrist_image", "observation/state", "prompt",
            })
            self.assertEqual(call.args[0]["observation/image"][0, 0, 0], frame)
        expected = client.infer.return_value["actions"][:16]
        np.testing.assert_allclose(saved, np.concatenate([expected, expected[:3]]), rtol=1e-6)

    def test_missing_observation_on_error_is_recorded_and_closed(self):
        env, _, result, saved = self.run_episode("error")
        self.assertFalse(result["success"])
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["steps"], 1318)
        self.assertIn("simulation failed", result["error"])
        self.assertEqual(saved.shape, (19, 8))
        self.assertEqual(env.close_calls, 1)

    def test_long_demo_is_capped_before_upload_with_correct_execution_boundary(self):
        env = FakeEnvironment()
        original = env.observation([0])
        env.reset = lambda: ({key: values * 4510 for key, values in original.items()},
                             {"task_goal": ["Move the cube"], "status": "ongoing"})
        _, client, result, _ = self.run_episode(env=env)
        self.assertTrue(result["success"])
        initial = client.add_buffer.call_args_list[0].args[0]
        self.assertEqual(initial["images"].shape, (801, 1, 4, 4, 3))
        self.assertEqual(initial["state"].shape, (801, 8))
        self.assertEqual(initial["exec_start_idx"], 800)
        self.assertEqual(result["demonstration_preprocessing"]["source_demo_frames"], 4509)
        self.assertEqual(result["metrics"]["video_frames"], 4510 + 19)

    def test_timeout_stays_failure_and_pending_cases_are_counted(self):
        _, _, result, _ = self.run_episode("timeout")
        from evaluation import summarize
        summary = summarize([{"task": "BinFill", "episode": i} for i in range(2)], [result])
        self.assertEqual(summary["counts"]["timeout"], 1)
        self.assertEqual(summary["pending"], 1)
        self.assertEqual(summary["success_rate"], 0)

    def test_finetuned_identity_retains_demo_cap_and_resume_compatibility(self):
        from non_agentic_methods.framesamp_modul.demonstration_sampling import configuration_for_method
        from run_storage import requires_demonstration_resampling
        for method in ("framesamp-modul-plus", "framesamp-modul-ood"):
            with self.subTest(method=method):
                env = FakeEnvironment()
                original = env.observation([0])
                env.reset = lambda: ({key: values * 1201 for key, values in original.items()},
                                     {"task_goal": ["Move the cube"], "status": "ongoing"})
                _, client, result, _ = self.run_episode(env=env, method=method)
                initial = client.add_buffer.call_args_list[0].args[0]
                self.assertEqual(initial["images"].shape[0], 801)
                self.assertEqual(initial["exec_start_idx"], 800)
                self.assertEqual(result["method"], method)
                self.assertEqual(configuration_for_method(method),
                                 configuration_for_method("framesamp-modul"))
                self.assertFalse(requires_demonstration_resampling(Path("unused"), {}, result))
                result["demonstration_preprocessing"]["input_demo_frames"] = 1200
                self.assertTrue(requires_demonstration_resampling(Path("unused"), {}, result))


    def test_checkpoint_requires_matching_verbatim_history_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "79999"
            (checkpoint / "params").mkdir(parents=True)
            (checkpoint / "assets").mkdir()
            config = checkpoint.parent / "history_config.txt"
            for content in ("symbolic.yaml", adapter.HISTORY_CONFIG + "\n"):
                config.write_text(content)
                with self.assertRaises(ValueError):
                    adapter.check_checkpoint(checkpoint)
            config.write_text(adapter.HISTORY_CONFIG)
            self.assertEqual(adapter.check_checkpoint(checkpoint), checkpoint)


if __name__ == "__main__":
    unittest.main()
