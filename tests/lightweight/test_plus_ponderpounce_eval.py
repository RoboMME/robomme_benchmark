"""Check the real upstream scheduler and observation mapper with fake model fires."""

import asyncio
import contextlib
import importlib.metadata
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("ponderpounce_ood", ROOT / "scripts/plus_eval/non_agentic_methods/ponderpounce/eval_plus.py")
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)
try:
    HAS_DEPS = tuple(map(int, importlib.metadata.version("transformers").split(".")[:2])) >= (5, 13)
    HAS_DEPS = HAS_DEPS and importlib.util.find_spec("vla_eval") is not None
except importlib.metadata.PackageNotFoundError:
    HAS_DEPS = False


class FakeEnvironment:
    def __init__(self, outcome="success"):
        self.outcome = outcome
        self.actions = []
        self.close_calls = 0

    def observation(self, indices):
        return {
            "front_rgb_list": [np.full((4, 4, 3), i, np.uint8) for i in indices],
            "wrist_rgb_list": [np.full((4, 4, 3), i + 100, np.uint8) for i in indices],
            "joint_state_list": [np.arange(7) + i for i in indices],
            "gripper_state_list": [np.array([0.04, 0.99]) for i in indices],
        }

    def reset(self):
        return self.observation(range(4)), {
            "task_goal": ["Put cubes in the bin"], "status": "ongoing",
            "grounded_subgoal_online": "PRIVILEGED", "oracle_policy_step": 1118,
        }

    def step(self, action):
        self.actions.append(action.copy())
        n = len(self.actions)
        if n == 47 and self.outcome == "error":
            return None, 0, True, True, {"status": "error", "error_message": "simulation failed"}
        ended = n == 47
        return self.observation([n + 3]), 0, ended, False, {
            "status": self.outcome if ended else "ongoing", "execution_control_steps": 1300 + n,
        }

    def close(self):
        self.close_calls += 1


@unittest.skipUnless(HAS_DEPS, "Run in the PonderPounce environment")
class PonderPounceOODTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from ponderpounce.eval import robomme_server
        from vla_eval.benchmarks.robomme.benchmark import RoboMMEBenchmark
        import torch
        cls.upstream, cls.mapper, cls.torch = robomme_server, RoboMMEBenchmark, torch

    def make_server(self):
        up = self.upstream
        server = object.__new__(adapter.recorded_server_class(SimpleNamespace(task="BinFill")))
        server._episodes, server._episode_counts = {}, {}
        server._seed = 0
        server._dt_ns = 50_000_000
        server._s1_period_ns = server._s2_period_ns = 1_000_000_000
        server._s2_delay_ns = 300_000_000
        server._wait_first_cognition = True
        server._norm_stats = server._s1_tokenizer = server._s2_processor = server._s1_transform = None
        server._subgoal_grounded = True
        server._s1_num_camera_slots = 2
        server._s2 = SimpleNamespace(num_cognition_tokens=1)
        server._adapter = up.ObsAdapter(("agentview", "wrist"), 8, -1, None, 1, 20)
        server.ponder_ticks, server.pounce_ticks, server.observations = [], [], []

        def fire_s2(ep, obs, now):
            server.ponder_ticks.append(ep.tick)
            ep.n_s2_fires += 1
            ep.cognitions.append(up.Cognition(self.torch.zeros(1, 1), now, now, now + 300_000_000, "model subgoal", "test"))
            ep.active_subgoal = "model subgoal"

        def fire_s1(ep, obs, now):
            self.assertIsNotNone(server._visible_cognition(ep, now))
            server.pounce_ticks.append(ep.tick)
            ep.n_s1_fires += 1
            ep.chunk = up.Chunk(self.torch.arange(160, dtype=self.torch.float32).reshape(20, 8), None)

        server._fire_s2, server._fire_s1 = fire_s2, fire_s1
        original = server.on_observation

        async def observe(obs, ctx):
            server.observations.append(obs)
            await original(obs, ctx)

        server.on_observation = observe
        return server

    def run_episode(self, outcome="success"):
        server = self.make_server()
        env = FakeEnvironment(outcome)
        builder = SimpleNamespace(make_env_for_episode=lambda _: env, episode_max_steps=lambda _: 1877)
        with tempfile.TemporaryDirectory() as tmp, patch("evaluation.VideoRecorder"), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            folder = Path(tmp) / "episode"
            evaluation = adapter.PonderPounceEvaluation(SimpleNamespace())
            evaluation.server, evaluation.mapper_class = server, self.mapper
            result = asyncio.run(evaluation.run_episode(builder, "BinFill", 0, folder))
            saved = np.load(folder / "actions.npy")
            self.assertEqual(json.loads((folder / "result.json").read_text()), result)
        return server, env, result, saved

    def test_corrected_hold_preserves_cognition_delay_and_twenty_step_schedule(self):
        server, env, result, saved = self.run_episode()
        self.assertTrue(result["success"])
        self.assertEqual(result["steps"], 1347)
        self.assertEqual(server.ponder_ticks, [1, 21, 41])
        self.assertEqual(server.pounce_ticks, [7, 27, 47])
        self.assertEqual(env.close_calls, 1)
        self.assertEqual(server._episodes, {})
        self.assertEqual(saved.shape, (47, 8))
        for i in range(6):
            np.testing.assert_allclose(saved[i], list(np.arange(7) + i + 3) + [1.0])
        np.testing.assert_array_equal(saved[6:26], np.arange(160).reshape(20, 8))
        np.testing.assert_array_equal(saved[26:46], saved[6:26])
        np.testing.assert_array_equal(saved[46], saved[6])

    def test_hold_converts_metres_without_changing_arm_targets(self):
        from ponderpounce.normalize import NormStats

        server = self.make_server()
        server._norm_stats = NormStats(
            q01=np.array([0] * 7 + [-1], dtype=np.float32),
            q99=np.ones(8, dtype=np.float32),
            delta_action_mask=[True] * 7 + [False],
        )
        lower, upper = adapter.panda_gripper_limits()
        for position, command in ((0.04, 1.0), (0.0, -0.6), (0.015, 0.0)):
            with self.subTest(position=position):
                state = np.r_[np.arange(7, dtype=np.float32), position]
                action = server._hold(state)[0]
                np.testing.assert_allclose(action[:7], state[:7])
                self.assertAlmostEqual(float(action[7]), command, places=6)
                target = lower + (float(action[7]) + 1) * (upper - lower) / 2
                self.assertAlmostEqual(target, position, places=7)

    def test_learned_gripper_predictions_are_not_converted_again(self):
        from ponderpounce.normalize import NormStats

        server = self.make_server()
        server._norm_stats = NormStats(
            q01=np.array([0] * 7 + [-1], dtype=np.float32),
            q99=np.ones(8, dtype=np.float32),
            delta_action_mask=[True] * 7 + [False],
        )
        grippers = [0.7003865, -0.9121144, 0.5542777]
        chunk = self.torch.tensor([[0] * 7 + [g] for g in grippers])
        episode = SimpleNamespace(chunk=self.upstream.Chunk(chunk, np.zeros(8, dtype=np.float32)))
        actions = [server._dispense(episode, {})["actions"][0, -1] for _ in grippers]
        np.testing.assert_allclose(actions, grippers, atol=1e-7)

    def test_stick_tasks_keep_their_unused_gripper_channel(self):
        for task in ("PatternLock", "RouteStick"):
            with self.subTest(task=task):
                server = object.__new__(adapter.recorded_server_class(SimpleNamespace(task=task)))
                server._norm_stats = None
                state = np.r_[np.arange(7, dtype=np.float32), 0.0]
                np.testing.assert_array_equal(server._hold(state)[0], state)

    def test_demo_is_sent_once_and_privileged_fields_are_excluded(self):
        server, _, _, _ = self.run_episode()
        first = server.observations[0]
        self.assertEqual([f[0, 0, 0] for f in first["video_history"]], [0, 1, 2])
        self.assertEqual(first["images"]["agentview"][0, 0, 0], 3)
        self.assertEqual(first["images"]["wrist"][0, 0, 0], 103)
        self.assertEqual(first["states"].dtype, np.float32)
        np.testing.assert_allclose(first["states"], [3, 4, 5, 6, 7, 8, 9, 0.04])
        for obs in server.observations[1:]:
            self.assertEqual(set(obs), {"images", "states", "task_description"})
        self.assertEqual(set(first), {
            "images", "states", "task_description", "video_history", "wrist_video_history", "episode_restart",
        })

    def test_environment_error_closes_both_session_and_environment(self):
        server, env, result, _ = self.run_episode("error")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["steps"], 1346)
        self.assertIn("simulation failed", result["error"])
        self.assertEqual(env.close_calls, 1)
        self.assertEqual(server._episodes, {})

    def test_timeout_and_pending_cases_remain_failures(self):
        _, _, result, _ = self.run_episode("timeout")
        from evaluation import summarize
        summary = summarize([{"task": "BinFill", "episode": i} for i in range(2)], [result])
        self.assertEqual(summary["counts"]["timeout"], 1)
        self.assertEqual(summary["pending"], 1)
        self.assertEqual(summary["success_rate"], 0)


if __name__ == "__main__":
    unittest.main()
