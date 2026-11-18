"""Exercise Astra policy contracts through the shared runner, without GPUs or API calls."""

import asyncio
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from robomme_plus import BenchmarkEnvBuilder


ROOT = Path(__file__).resolve().parents[2]
ADAPTER = ROOT / "scripts/plus_eval/agentic_methods/astra/three_tier"
CHAMP = ADAPTER / "Astra-on-RoboMME/examples/champ"


@unittest.skipUnless((CHAMP / "runner.py").exists(), "Initialize the Astra submodule")
class AstraOODTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path_patch = patch.object(sys, "path", [str(CHAMP), *sys.path])
        cls.path_patch.start()
        cls.addClassCleanup(cls.path_patch.stop)
        spec = importlib.util.spec_from_file_location("astra_ood_eval", ADAPTER / "eval_plus.py")
        cls.adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.adapter)
        import runner
        cls.runner = runner

    def test_default_selection_uses_ood_metadata(self):
        builders, document = self.adapter.select_cases(BenchmarkEnvBuilder)
        self.assertEqual(document["benchmark"], "robomme_plus")
        self.assertEqual(len(builders), 16)
        self.assertEqual(len(document["cases"]), 800)
        for task, builder in builders.items():
            self.assertEqual(builder.action_space, "joint_angle")
            ids = [case["episode"] for case in document["cases"] if case["task"] == task]
            self.assertEqual(ids, list(range(builder.get_episode_num())))

    def test_invalid_selection_is_rejected(self):
        for tasks, episodes in [
            (["unknown"], [0]), (["BinFill", "BinFill"], [0]),
            (["BinFill"], [0, 0]), (["BinFill"], [-1]), (["BinFill"], [50]),
        ]:
            with self.subTest(tasks=tasks, episodes=episodes), self.assertRaises(ValueError):
                self.adapter.select_cases(BenchmarkEnvBuilder, tasks, episodes)

    def run_rollout(self, budget, terminal_step=None, status="timeout", error=False):
        class Env:
            steps = 0
            closed = False

            def observation(self, demo=False):
                # Tiny frames keep this CPU test cheap; production front() checks 256x256.
                frame = np.full((4, 4, 3), self.steps % 256, dtype=np.uint8)
                return {
                    "front_rgb_list": [frame.copy(), frame] if demo else [frame],
                    "wrist_rgb_list": [frame, frame] if demo else [frame], "joint_state_list": [np.zeros(7)],
                    "gripper_state_list": [np.zeros(2)],
                }

            def reset(self):
                return self.observation(demo=True), {
                    "task_goal": ["Remember the demonstration"],
                    "grounded_subgoal_online": "PRIVILEGED", "oracle_policy_step": 123,
                }

            def step(self, action):
                self.steps += 1
                if error:
                    return None, 0, True, False, {"status": "error", "error_message": "sim failure"}
                ended = self.steps == (terminal_step or budget)
                return self.observation(), 0, ended and status != "timeout", ended, {
                    "status": status if ended else "ongoing",
                }

            def close(self):
                self.closed = True

        env = Env()
        builder = SimpleNamespace(
            episode_max_steps=lambda _: budget, make_env_for_episode=lambda _: env,
            resolve_episode=lambda _: (321, "hard"),
        )
        client = SimpleNamespace(reset=Mock(return_value={}), infer=Mock(return_value={"actions": np.zeros((16, 8))}))
        monitor = SimpleNamespace(config=None, engine=SimpleNamespace(infer=Mock(return_value=[
            SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="false"))])
        ])))
        planner = SimpleNamespace(predict=Mock(return_value=("predicted subgoal", "mock", 0.0)))
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace()
            evaluation = self.adapter.ThreeTierAstraEvaluation(args)
            evaluation.client, evaluation.monitor = client, monitor
            evaluation.infer_request, evaluation.responder = SimpleNamespace, None
            original_start = evaluation.start_episode
            def start(obs, info):
                original_start(obs, info)
                evaluation.planner = planner
            evaluation.start_episode = start
            with patch("evaluation.VideoRecorder"), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                result = asyncio.run(evaluation.run_episode(builder, "VideoRepick", 0, Path(tmp) / "episode"))
            self.assertTrue(env.closed)
            client.reset.assert_called_once()
            self.assertEqual(result["steps"], 0 if error else env.steps)
            self.assertEqual(result["max_steps"], budget)
            self.assertEqual(result["benchmark"], "robomme_plus")
            self.assertEqual(len(planner.predict.call_args.args[3]), 1)
            self.assertNotIn("PRIVILEGED", str(planner.predict.call_args))
            for call in client.infer.call_args_list:
                self.assertEqual(call.args[0]["grounded_subgoal"], "predicted subgoal")
                self.assertNotIn("oracle_policy_step", call.args[0])
            if not error and env.steps > 16:
                prompts = list((Path(tmp) / "episode/calls").glob("*_system1_5/prompt.txt"))
                self.assertTrue(prompts)
                self.assertIn("predicted subgoal", prompts[0].read_text())
            return result

    def test_plus_budgets_replace_1300_and_stop_inside_chunks(self):
        for budget in (19, 1313):
            with self.subTest(budget=budget):
                result = self.run_rollout(budget)
                self.assertEqual(result["steps"], budget)
                self.assertEqual(result["status"], "timeout")

    def test_environment_termination_wins_over_budget(self):
        for status in ("success", "fail"):
            with self.subTest(status=status):
                result = self.run_rollout(99, terminal_step=5, status=status)
                self.assertEqual(result["steps"], 5)
                self.assertEqual(result["status"], status)

    def test_simulator_error_is_recorded_and_environment_closes(self):
        result = self.run_rollout(99, error=True)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["steps"], 0)

    def test_summary_keeps_errors_and_unfinished_cases_in_denominator(self):
        from evaluation import summarize
        _, document = self.adapter.select_cases(BenchmarkEnvBuilder, ["BinFill"], [0, 1, 2])
        rows = [{"task": "BinFill", "episode": i, "status": status, "success": status == "success"}
                for i, status in enumerate(("success", "error"))]
        report = summarize(document["cases"], rows, "3-tier-astra")
        self.assertFalse(report["complete"])
        self.assertEqual(report["counts"]["error"], 1)
        self.assertEqual(report["pending"], 1)
        self.assertEqual(report["success_rate"], 1 / 3)
        self.assertEqual(report["per_task"]["BinFill"]["success_rate"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
