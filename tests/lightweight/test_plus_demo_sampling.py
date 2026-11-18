"""Long demos retain temporal coverage and aligned observations before inference."""
import asyncio
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/plus_eval"))
from non_agentic_methods.framesamp_modul.demonstration_sampling import configuration_for_method, prepare_demonstration
from evaluation import Evaluation
from run_storage import select_pending


def observation(count):
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    return {
        "front_rgb_list": [image] * count,
        "wrist_rgb_list": [image] * count,
        "joint_state_list": list(range(count)),
        "eef_state_list": list(range(count)),
        "front_camera_extrinsic_list": list(range(count)),
        "static_calibration": "unchanged",
    }


class DemonstrationSamplingTest(unittest.TestCase):
    def test_short_and_exactly_forty_second_demos_are_unchanged(self):
        for count in (0, 1, 10, 800, 801):
            with self.subTest(frames=count):
                obs = observation(count)
                selected, report = prepare_demonstration(obs, method="framesamp-modul")
                self.assertIs(selected, obs)
                self.assertFalse(report["applied"])
                self.assertEqual(report["input_demo_frames"], max(0, count - 1))

    def test_uniform_coverage_and_synchronized_state_camera_columns(self):
        for count in (802, 4510, 20001):
            with self.subTest(frames=count):
                obs = observation(count)
                sampled, report = prepare_demonstration(obs, method="framesamp-modul")
                indices = report["source_indices"]
                self.assertEqual(len(indices), 801)
                self.assertEqual(indices[0], 0)
                self.assertEqual(indices[-2:], [count - 2, count - 1])
                self.assertEqual(len(set(indices)), len(indices))
                spacing = np.diff(indices[:-1])
                self.assertLessEqual(int(spacing.max() - spacing.min()), 1)
                for key in ("joint_state_list", "eef_state_list", "front_camera_extrinsic_list"):
                    self.assertEqual(sampled[key], indices)
                for key in ("front_rgb_list", "wrist_rgb_list"):
                    self.assertEqual(len(sampled[key]), 801)
                    self.assertIs(sampled[key][-1], obs[key][-1])
                self.assertEqual(len(obs["front_rgb_list"]), count)
                self.assertEqual(sampled["static_calibration"], "unchanged")
                self.assertEqual(report["input_video_seconds"], 40)

    def test_misaligned_observation_column_is_rejected(self):
        obs = observation(900)
        obs["eef_state_list"].pop()
        with self.assertRaisesRegex(ValueError, "eef_state_list"):
            prepare_demonstration(obs, method="framesamp-modul")

    def test_other_methods_keep_long_demonstrations_unchanged(self):
        for method in ("embodied-astra", "simplememvla", "3-tier-astra", "ponderpounce", "fixture"):
            with self.subTest(method=method):
                obs = observation(4510)
                selected, report = prepare_demonstration(obs, method=method)
                self.assertIs(selected, obs)
                self.assertIsNone(report)
                self.assertEqual(configuration_for_method(method), {"sampling": "none"})

    def test_shared_runner_caps_only_demo_and_preserves_full_recording(self):
        class Environment:
            calls = 0
            def reset(self):
                return observation(4510), {"task_goal": ["goal"]}
            def step(self, action):
                self.calls += 1
                done = self.calls == 2
                return observation(1 if done else 900), 0, done, False, {
                    "status": "success" if done else "ongoing", "execution_control_steps": 901 if done else 900}
            def close(self):
                pass
        class Policy(Evaluation):
            method = "framesamp-modul"
            def start_episode(self, obs, info):
                self.initial = obs
                self.seen = []
            def act(self, obs, info):
                self.seen.append(len(obs["front_rgb_list"]))
                return np.zeros(8)
        policy = Policy(SimpleNamespace())
        builder = SimpleNamespace(make_env_for_episode=lambda *a, **kw: Environment(),
                                  episode_max_steps=lambda ep: 1500)
        with tempfile.TemporaryDirectory() as tmp, patch("evaluation.VideoRecorder"):
            directory = Path(tmp) / "episode"
            result = asyncio.run(policy.run_episode(builder, "MoveCube", 0, directory))
            saved = json.loads((directory / "demonstration_sampling.json").read_text())
        self.assertTrue(result["success"])
        self.assertEqual(policy.seen, [801, 900])
        self.assertEqual(policy.initial["joint_state_list"][-2:], [4508, 4509])
        self.assertEqual(result["metrics"]["video_frames"], 4510 + 900 + 1)
        self.assertEqual(result["demonstration_preprocessing"], saved)
        for method in ("embodied-astra", "simplememvla"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as tmp, patch("evaluation.VideoRecorder"):
                policy.method = method
                directory = Path(tmp) / "episode"
                result = asyncio.run(policy.run_episode(builder, "MoveCube", 0, directory))
                self.assertTrue(result["success"])
                self.assertEqual(policy.seen, [4510, 900])
                self.assertNotIn("demonstration_preprocessing", result)
                self.assertFalse((directory / "demonstration_sampling.json").exists())

    def test_resume_replaces_old_long_demo_outcomes_once(self):
        cases = [{"task": "MoveCube", "episode": i} for i in range(3)]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for case, demo_count in zip(cases, (800, 4509, 4509)):
                directory = root / f"MoveCube_ep{case['episode']}"
                directory.mkdir()
                result = {**case, "status": "success", "method": "framesamp-modul"}
                if case["episode"] == 2:
                    result["demonstration_preprocessing"] = {"source_demo_frames": demo_count, "input_demo_frames": 800}
                (directory / "result.json").write_text(json.dumps(result))
                (directory / "frames.jsonl").write_text(json.dumps({"phase": "demonstration", "batch_size": demo_count + 1}) + "\n")
            with patch("run_storage.valid_outcome", return_value=True):
                self.assertEqual(select_pending(root, cases, "resume"), [cases[1]])

    def test_resume_other_methods_replaces_only_mistakenly_capped_results(self):
        cases = [{"task": "MoveCube", "episode": i} for i in range(4)]
        for method in ("embodied-astra", "simplememvla"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / "config.json").write_text(json.dumps({"method": method}))
                for case in cases:
                    directory = root / f"MoveCube_ep{case['episode']}"
                    directory.mkdir()
                    result = {**case, "status": "success"}
                    if case["episode"] == 1:
                        result["demonstration_preprocessing"] = {"source_demo_frames": 4509, "input_demo_frames": 800}
                    elif case["episode"] == 2:
                        result["demonstration_preprocessing"] = {"source_demo_frames": 500, "input_demo_frames": 500, "applied": False}
                    elif case["episode"] == 3:
                        result["demonstration_preprocessing"] = {"source_demo_frames": 4509, "input_demo_frames": 800, "applied": True}
                    (directory / "result.json").write_text(json.dumps(result))
                    (directory / "frames.jsonl").write_text(json.dumps({"phase": "demonstration", "batch_size": 4510}) + "\n")
                with patch("run_storage.valid_outcome", return_value=True):
                    self.assertEqual(select_pending(root, cases, "resume"), [cases[1], cases[3]])


if __name__ == "__main__":
    unittest.main()
