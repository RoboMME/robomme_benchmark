"""Exercise the SimpleMemVLA history/decision interface through the shared runner."""

import asyncio
import contextlib
import importlib.util
import importlib.metadata
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("simplememvla_ood", ROOT / "scripts/plus_eval/non_agentic_methods/simplememvla/eval_plus.py")
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
        if self.outcome == "reset_error":
            raise RuntimeError("reset failed")
        return self.observation(range(4)), {
            "task_goal": ["Put cubes in the bin"], "status": "ongoing",
            "oracle_policy_step": 1118, "grounded_subgoal_online": "PRIVILEGED",
        }

    def step(self, action):
        self.actions.append(action.copy())
        n = len(self.actions)
        if self.outcome == "error" and n == 19:
            return None, 0, True, True, {"status": "error", "error_message": "simulation failed"}
        ended = n == 19
        return self.observation([n + 3]), 0, ended, False, {
            "execution_control_steps": 1300 + n, "status": self.outcome if ended else "ongoing",
        }

    def close(self):
        self.close_calls += 1


class FakeBuffer:
    image_keys = ["observation.images.front", "observation.images.wrist"]
    native_fps = 20
    history_image_keys = [image_keys[0]]

    def reset(self):
        self.frames = []
        self._buffers = {k: [] for k in self.image_keys}
        self._frames_seen = {k: 0 for k in self.image_keys}

    def observe(self, frame):
        self.frames.append(frame)
        for k, v in frame.items():
            self._buffers[k].append(v)
            self._frames_seen[k] += 1

    def _prepare_inputs(self, instruction):
        return {"frames": self.frames.copy(), "instruction": instruction,
                "input_ids": np.array([[1, 2]]), "video_patch_keys": [("front", 0, len(self.frames)-1)]}


class FakePolicy:
    device = SimpleNamespace(type="cpu")
    processor = SimpleNamespace(tokenizer=SimpleNamespace(decode=lambda *args, **kw: "fixture prompt"))

    def __init__(self):
        self.calls = []
        self.chunk = np.arange(30 * 8).reshape(30, 8) / 100

    def generate_batch(self, processed, states):
        self.calls.append((processed, states))
        return [(self.chunk.copy(), "predicted subtask")]


try:
    HAS_MODEL_DEPS = tuple(map(int, importlib.metadata.version("transformers").split(".")[:2])) >= (5, 11)
except importlib.metadata.PackageNotFoundError:
    HAS_MODEL_DEPS = False


@unittest.skipUnless(HAS_MODEL_DEPS, "Run in the SimpleMemVLA environment (Transformers >= 5.11)")
class SimpleMemVLAOODTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from robomme_sim import eval_success
        cls.upstream = eval_success

    def run_episode(self, outcome="success"):
        env = FakeEnvironment(outcome)
        builder = SimpleNamespace(make_env_for_episode=lambda _: env, episode_max_steps=lambda _: 1877)
        policy = FakePolicy()
        with tempfile.TemporaryDirectory() as tmp, \
                patch("evaluation.VideoRecorder"), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            folder = Path(tmp) / "episode"
            from robomme_sim.robomme_env import encode_frames, encode_states
            evaluation = adapter.SimpleMemEvaluation(SimpleNamespace())
            evaluation.policy, evaluation.buffer_factory, evaluation.normalize_state = policy, FakeBuffer, None
            evaluation.encode_frames, evaluation.encode_states = encode_frames, encode_states
            result = asyncio.run(evaluation.run_episode(builder, "BinFill", 0, folder))
            if result["status"] != "error":
                annotations = [json.loads(line) for line in (folder / "annotations.jsonl").read_text().splitlines()]
                self.assertTrue(all(row["subgoal"] == "predicted subtask" for row in annotations))
                for call in (folder / "calls").glob("*_policy/input.json"):
                    prepared = folder / json.loads(call.read_text())["processed"]["path"]
                    self.assertEqual(json.loads(prepared.read_text())["instruction"], "Put cubes in the bin")
            self.assertEqual(json.loads((folder / "result.json").read_text()), result)
            saved = np.load(folder / "actions.npy")
        return env, policy, result, saved

    def test_original_loop_sees_demo_history_and_stops_mid_chunk(self):
        env, policy, result, saved = self.run_episode()
        self.assertTrue(result["success"])
        self.assertEqual(result["steps"], 1319)
        self.assertEqual(result["metrics"]["components"]["policy"]["count"], 2)
        self.assertEqual(env.close_calls, 1)
        for (processed, states), total in zip(policy.calls, [4, 20]):
            self.assertEqual(states, [None])
            self.assertEqual(processed[0]["instruction"], "Put cubes in the bin")
            self.assertEqual(set(processed[0]), {"frames", "instruction", "input_ids", "video_patch_keys"})
            frames = processed[0]["frames"]
            self.assertEqual(len(frames), total)
            self.assertEqual(set(frames[0]), set(FakeBuffer.image_keys))
            self.assertEqual([f["observation.images.front"][0, 0, 0] for f in frames], list(range(total)))
        np.testing.assert_array_equal(saved, np.concatenate([policy.chunk[:16], policy.chunk[:3]]))
        self.assertEqual(env.actions[0].dtype, np.float64)

    def test_state_packing_matches_joint_action_layout(self):
        from robomme_sim.robomme_env import encode_states
        state = encode_states(FakeEnvironment().observation([2]))[0]
        self.assertEqual(state.dtype, np.float32)
        np.testing.assert_allclose(state, [2, 3, 4, 5, 6, 7, 8, 0.03])

    def test_environment_errors_are_saved_and_closed(self):
        for outcome, steps in [("reset_error", 0), ("error", 1318)]:
            with self.subTest(outcome=outcome):
                env, _, result, _ = self.run_episode(outcome)
                self.assertEqual(result["status"], "error")
                self.assertFalse(result["success"])
                self.assertEqual(result["steps"], steps)
                self.assertEqual(env.close_calls, 1)

    def test_timeout_is_failure_and_pending_not_excluded(self):
        _, _, result, _ = self.run_episode("timeout")
        from evaluation import summarize
        summary = summarize([{"task": "BinFill", "episode": i} for i in range(2)], [result])
        self.assertEqual(summary["counts"]["timeout"], 1)
        self.assertEqual(summary["pending"], 1)
        self.assertEqual(summary["success_rate"], 0)


if __name__ == "__main__":
    unittest.main()
