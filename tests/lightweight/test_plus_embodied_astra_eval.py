"""Use the original waypoint policy functions with fake API responses and OOD observations."""

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


ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "scripts/plus_eval/agentic_methods/astra/embodied"
HARNESS = FOLDER / "embodied-astra-robomme/harness/astra_waypoint_eval_v3.py"


def load_adapter():
    spec = importlib.util.spec_from_file_location("embodied_astra_ood", FOLDER / "eval_plus.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeEnvironment:
    def __init__(self, outcome="success", noops=False):
        self.outcome, self.noops = outcome, noops
        self.actions = []
        self.close_calls = 0

    def observation(self, count=1):
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        return {
            "front_rgb_list": [frame] * count,
            "wrist_rgb_list": [frame] * count,
            "eef_state_list": [np.arange(6) / 10] * count,
            "gripper_state_list": [np.array([0.04, 0.04])] * count,
            "front_camera_extrinsic_list": [np.eye(3, 4)] * count,
            "wrist_camera_extrinsic_list": [np.eye(3, 4)] * count,
        }

    def information(self, steps, status="ongoing"):
        return {
            "task_goal": ["Put the cubes in the bin"], "execution_control_steps": steps,
            "front_camera_intrinsic": np.eye(3), "wrist_camera_intrinsic": np.eye(3),
            "status": status, "max_steps": 1877, "oracle_policy_step": 1118,
            "grounded_subgoal_online": "PRIVILEGED_SUBGOAL",
        }

    def reset(self):
        return self.observation(4), self.information(0)

    def step(self, action):
        self.actions.append(action.copy())
        if self.noops:
            return self.observation(0), 0, False, False, self.information(0)
        # A waypoint can cross 1300 control steps without ending the OOD episode.
        end = len(self.actions) == 2
        steps = 1877 if self.outcome == "timeout" and end else 1400 + len(self.actions)
        status = self.outcome if end else "ongoing"
        return self.observation(4), 0, end and status == "success", end and status == "timeout", self.information(steps, status)

    def close(self):
        self.close_calls += 1


def response(index, waypoint=None):
    answer = {"waypoint": waypoint if waypoint is not None else [0.1, 0.2, 0.3, -3.14, 0, 0], "gripper": -1}
    return SimpleNamespace(
        id=f"response-{index}", output_text=json.dumps(answer),
        model_dump=lambda **_: {
            "model": "fake-astra", "usage": {"input_tokens": 10, "output_tokens": 5},
        },
    )


@unittest.skipUnless(HARNESS.exists() and importlib.util.find_spec("openai"), "Requires the embodied Astra submodule and openai")
class EmbodiedAstraOODTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = load_adapter()

    def setUp(self):
        self.harness = self.adapter.load_harness()
        self.harness.PROMPT_VARIANT = "lr_note"

    def run_harness(self, run_dir, env, replies):
        client = Mock()
        client.responses.create.side_effect = replies
        evaluation = self.adapter.EmbodiedAstraEvaluation(SimpleNamespace(model="fake-astra"))
        evaluation.client, evaluation.harness = client, self.harness
        factory = Mock(return_value=env)
        builder = SimpleNamespace(make_env_for_episode=factory, episode_max_steps=lambda _: 1877)
        with patch("evaluation.VideoRecorder"), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = asyncio.run(evaluation.run_episode(builder, "BinFill", 0, run_dir / "BinFill_ep0"))
        return result, client, factory

    def test_retry_message_reaches_astra_and_preserves_conversation(self):
        class RetryingEnvironment(FakeEnvironment):
            def step(self, action):
                self.actions.append(action.copy())
                if len(self.actions) == 1:
                    return self.observation(), 0, False, False, {
                        **self.information(0), "trial_reset": True, "trial_index": 2,
                        "max_steps": 941,
                        "execution_total_control_steps": 1877,
                        "message": "You have one more opportunity to try again.",
                    }
                return self.observation(), 0, True, False, {
                    **self.information(10, "success"), "trial_index": 2,
                    "max_steps": 941,
                    "execution_total_control_steps": 1887,
                }
        env = RetryingEnvironment()
        self.harness.MAX_MODEL_CALLS = 1  # One call allowed in each physical trial.
        with tempfile.TemporaryDirectory() as tmp:
            result, client, _ = self.run_harness(Path(tmp), env, [response(0), response(1)])
        self.assertTrue(result["success"])
        self.assertEqual(result["steps"], 1887)
        follow = client.responses.create.call_args_list[1].kwargs
        self.assertEqual(follow["previous_response_id"], "response-0")
        self.assertIn("one more opportunity", json.dumps(follow["input"]))
        self.assertIn("restarted at zero", json.dumps(follow["input"]))
        self.assertIn("after 1877 control steps.", client.responses.create.call_args_list[0].kwargs["instructions"])
        self.assertIn("after 941 control steps.", follow["instructions"])
        self.assertNotIn("after 1877 control steps.", follow["instructions"])

    def test_long_demos_have_exactly_750_images_and_keep_endpoints(self):
        for count in (2246, 4495, 4509, 9000, 20000):
            with self.subTest(frames=count):
                obs = FakeEnvironment().observation(count)
                obs["front_rgb_list"] = list(range(count))
                obs["wrist_rgb_list"][-1] = "current-wrist"
                with patch.object(self.harness, "image_item", side_effect=lambda x: {"type": "input_image", "image_url": x}):
                    content = self.adapter.observation_content(
                        self.harness, obs, FakeEnvironment().information(0), "Video", first=True)
                images = [x["image_url"] for x in content if x["type"] == "input_image"]
                self.assertEqual(len(images), 750)
                self.assertEqual(images[0], 0)
                self.assertEqual(images[-2:], [count - 1, "current-wrist"])
                self.assertEqual(images[:-1], sorted(set(images[:-1])))
                spacing = np.diff(images[:-1])
                self.assertLessEqual(int(spacing.max() - spacing.min()), 1)
                self.assertIn("Robot state and calibration", content[-1]["text"])

    def test_short_observation_matches_original_harness(self):
        for count in (1, 10, 2242, 2245):
            with self.subTest(frames=count):
                obs, info = FakeEnvironment().observation(count), FakeEnvironment().information(0)
                self.assertEqual(self.adapter.observation_content(self.harness, obs, info, "Video", first=True),
                                 self.harness.observation(obs, info, "Video", first=True))

    def test_execution_observations_keep_original_sampling(self):
        obs, info = FakeEnvironment().observation(2246), FakeEnvironment().information(100)
        content = self.adapter.observation_content(self.harness, obs, info, "Motion", first=False)
        self.assertEqual(content, self.harness.observation(obs, info, "Motion", first=False))
        self.assertEqual(sum(x['type'] == 'input_image' for x in content), 751)
        obs = FakeEnvironment().observation(20000)
        content = self.adapter.observation_content(self.harness, obs, info, "Motion", first=False)
        self.assertLessEqual(sum(x['type'] == 'input_image' for x in content), 1500)

    def test_initial_request_is_capped_before_api_and_sampling_is_recorded(self):
        env = FakeEnvironment()
        obs = env.observation(4510)
        env.reset = lambda: (obs, env.information(0))
        with tempfile.TemporaryDirectory() as tmp:
            result, client, _ = self.run_harness(Path(tmp), env, [response(0), response(1)])
            saved = json.loads((Path(tmp) / 'BinFill_ep0/astra_demonstration_sampling.json').read_text())
        self.assertTrue(result['success'])
        counts = [sum(x['type'] == 'input_image' for x in c.kwargs['input'][0]['content'])
                  for c in client.responses.create.call_args_list]
        self.assertEqual(counts, [750, 3])
        self.assertEqual(result['astra_demonstration_sampling'], saved)
        self.assertEqual(saved['source_images'], 1130)
        self.assertEqual(saved['input_images'], 750)
        self.assertEqual(saved['source_observation_frames'], 4510)
        self.assertTrue(saved['applied'])
        self.assertEqual(len(obs['front_rgb_list']), 4510)
        self.assertEqual(result['metrics']['video_frames'], 4518)
        self.assertNotIn('demonstration_preprocessing', result)

    def test_total_image_budget_stops_episode_before_sending_over_limit_request(self):
        for motion_counts, expected_calls, submitted, projected in [
                ([2245, 1], 2, 1500, 1502),  # Exactly 1,500 is permitted.
                ([2248], 1, 750, 1501)]:
            class LongMotionEnvironment(FakeEnvironment):
                def reset(self):
                    return self.observation(2245), self.information(0)

                def step(self, action):
                    self.actions.append(action.copy())
                    return (self.observation(motion_counts[len(self.actions) - 1]),
                            0, False, False, self.information(len(self.actions)))

                def get_wrapper_attr(self, name):
                    if name == 'try_execution_retry':
                        return retry
                    raise AttributeError(name)

            retry = Mock(side_effect=AssertionError('Image limit must end the whole episode'))
            with self.subTest(motion_counts=motion_counts), tempfile.TemporaryDirectory() as tmp:
                env = LongMotionEnvironment()
                result, client, _ = self.run_harness(Path(tmp), env, [response(i) for i in range(expected_calls)])
                saved = json.loads((Path(tmp) / 'BinFill_ep0/astra_image_budget.json').read_text())
                self.assertEqual(result['astra_image_budget'], saved)
            self.assertEqual(result['status'], 'fail')
            self.assertFalse(result['success'])
            self.assertTrue(result['artifacts_complete'])
            self.assertNotIn('error', result)
            self.assertEqual(result['end_reason'], 'limit:1500_total_images')
            self.assertEqual(client.responses.create.call_count, expected_calls)
            self.assertEqual(len(env.actions), expected_calls)
            self.assertEqual(env.close_calls, 1)
            retry.assert_not_called()
            self.assertEqual(saved['submitted_images'], submitted)
            self.assertEqual(saved['projected_total_images'], projected)
            self.assertEqual(saved['stop_source'], 'preflight')
            self.assertTrue(saved['stopped'])

    def test_total_image_budget_persists_across_physical_trial_reset(self):
        class RetryingEnvironment(FakeEnvironment):
            def reset(self):
                return self.observation(2245), self.information(0)

            def step(self, action):
                self.actions.append(action.copy())
                return self.observation(2248), 0, False, False, {
                    **self.information(0), 'trial_reset': True, 'trial_index': 2,
                    'execution_total_control_steps': 10, 'message': 'Try again.',
                }

        self.harness.MAX_MODEL_CALLS = 1
        with tempfile.TemporaryDirectory() as tmp:
            env = RetryingEnvironment()
            result, client, _ = self.run_harness(Path(tmp), env, [response(0)])
        self.assertEqual(result['end_reason'], 'limit:1500_total_images')
        self.assertEqual(result['astra_image_budget']['submitted_images'], 750)
        self.assertEqual(result['astra_image_budget']['projected_total_images'], 1501)
        self.assertEqual(client.responses.create.call_count, 1)
        self.assertEqual(len(env.actions), 1)

    def test_server_image_limit_is_failure_while_other_bad_requests_remain_errors(self):
        import httpx
        from openai import BadRequestError
        for code, expected in [('max_images_per_request', 'fail'), ('invalid_request', 'error')]:
            error = BadRequestError('fixture API rejection',
                response=httpx.Response(400, request=httpx.Request('POST', 'https://api.invalid/responses')),
                body={'error': {'type': code, 'message': 'fixture API rejection'}})
            with self.subTest(code=code), tempfile.TemporaryDirectory() as tmp:
                env = FakeEnvironment()
                result, client, _ = self.run_harness(Path(tmp), env, [response(0), error])
            self.assertEqual(result['status'], expected)
            self.assertFalse(result['success'])
            self.assertEqual(client.responses.create.call_count, 2)
            self.assertEqual(len(env.actions), 1)
            self.assertEqual(env.close_calls, 1)
            if expected == 'fail':
                self.assertEqual(result['end_reason'], 'limit:1500_total_images')
                self.assertEqual(result['astra_image_budget']['stop_source'], 'api')
            else:
                self.assertEqual(result['error_type'], 'BadRequestError')

    def test_resume_replaces_only_astra_requests_above_750_images(self):
        from run_storage import select_pending
        from agentic_methods.astra.image_sampling import image_count
        cases = [{'task': 'MoveCube', 'episode': i} for i in range(5)]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for case, count in zip(cases, (749, 750, 751, 1130, 750)):
                directory = root / f"MoveCube_ep{case['episode']}"
                directory.mkdir()
                result = {**case, 'method': 'embodied-astra', 'status': 'success'}
                if case['episode'] == 4:
                    result['astra_demonstration_sampling'] = {'source_images': 1130, 'input_images': 750}
                (directory / 'result.json').write_text(json.dumps(result))
                call = directory / 'calls/000001_astra_api'
                call.mkdir(parents=True)
                payload = {'input': [{'role': 'user', 'content': [{'type': 'input_image', 'image_url': 'fixture'}] * count}]}
                self.assertEqual(image_count(payload), count)
                (call / 'input.json').write_text(json.dumps(payload))
            with patch('run_storage.valid_outcome', return_value=True):
                self.assertEqual(select_pending(root, cases, 'resume'), cases[2:4])

    def test_calibration_clock_prompts_and_unchanged_actions(self):
        env = FakeEnvironment()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result, client, factory = self.run_harness(root, env, [response(0), response(1)])
            self.assertTrue(result["success"])
            self.assertEqual(result["metrics"]["components"]["astra"]["count"], 2)
            self.assertEqual(result["steps"], 1402)
            self.assertEqual(env.close_calls, 1)
            np.testing.assert_array_equal(env.actions[0], [0.1, 0.2, 0.3, -3.14, 0, 0, -1])
            self.assertEqual(factory.call_args.kwargs, {
                "include_front_camera_extrinsic": True, "include_front_camera_intrinsic": True,
                "include_wrist_camera_extrinsic": True, "include_wrist_camera_intrinsic": True,
            })
            first, follow = [call.kwargs for call in client.responses.create.call_args_list]
            self.assertNotIn("previous_response_id", first)
            self.assertEqual(follow["previous_response_id"], "response-0")
            state = json.loads(follow["input"][0]["content"][-1]["text"].split(": ", 1)[1])
            self.assertEqual(state["elapsed_control_steps"], 1401)
            self.assertIn("wrist_camera_K", state)
            for private in ("PRIVILEGED_SUBGOAL", "oracle_policy_step", "remaining_steps", "max_steps"):
                self.assertNotIn(private, json.dumps(follow["input"]))
            published = json.loads((HARNESS.parents[1] / "results/e2e_v3_lrnote_test_800/config.json").read_text())
            expected_prompt = published["system_prompts"]["BinFill"].replace(
                "after 1300 control steps.", "after 1877 control steps.")
            self.assertEqual(first["instructions"], expected_prompt)
            self.assertEqual(follow["instructions"], expected_prompt)
            self.assertEqual(result["metrics"]["video_frames"], 12)
            self.assertEqual(result["metrics"]["api"]["attempts"], 2)
            self.assertIsNone(result["metrics"]["api"]["estimated_cost_usd"])
            prompts = sorted((root / "BinFill_ep0/calls").glob("*_astra_api/prompt.txt"))
            self.assertEqual(len(prompts), 2)
            self.assertIn("previous_response_id=response-0", prompts[-1].read_text())

    def test_environment_timeout_and_noop_limit_remain_failures(self):
        for outcome, noops, status in [("timeout", False, "timeout"), ("success", True, "fail")]:
            with self.subTest(status=status), tempfile.TemporaryDirectory() as tmp:
                env = FakeEnvironment(outcome, noops)
                result, _, _ = self.run_harness(Path(tmp), env, [response(i) for i in range(3)])
                self.assertEqual(result["status"], status)
                self.assertFalse(result["success"])
                self.assertEqual(env.close_calls, 1)
                if noops:
                    self.assertEqual(result["metrics"]["components"]["astra"]["count"], 3)
                    self.assertEqual(result["end_reason"], "limit:3_consecutive_noops")
                else:
                    self.assertEqual(result["steps"], 1877)

    def test_api_exception_is_recorded_and_environment_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = FakeEnvironment()
            result, _, _ = self.run_harness(Path(tmp), env, [RuntimeError("fake API failure")])
            self.assertEqual(result["status"], "error")
            self.assertFalse(result["success"])
            self.assertEqual(env.close_calls, 1)
            self.assertEqual(result["metrics"]["api"]["attempts"], 1)
            self.assertEqual(result["metrics"]["api"]["successful_requests"], 0)

    def test_retries_are_visible_and_resend_the_same_input(self):
        self.harness.RETRYABLE = (RuntimeError,)
        with tempfile.TemporaryDirectory() as tmp, patch.object(self.adapter.time, "sleep"):
            result, client, _ = self.run_harness(Path(tmp), FakeEnvironment(),
                [RuntimeError("rate limit fixture"), response(0), response(1)])
            self.assertTrue(result["success"])
            self.assertEqual(result["metrics"]["api"]["attempts"], 3)
            self.assertEqual(result["metrics"]["api"]["successful_requests"], 2)
            self.assertEqual(result["metrics"]["latency"]["retry_wait"]["count"], 1)
            self.assertEqual(client.responses.create.call_args_list[0], client.responses.create.call_args_list[1])
            self.assertEqual(result['astra_image_budget']['submitted_images'], 6)


if __name__ == "__main__":
    unittest.main()
