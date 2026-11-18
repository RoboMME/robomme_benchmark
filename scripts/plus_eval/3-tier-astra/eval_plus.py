#!/usr/bin/env python3
"""Three-tier Astra inference with the shared RoboMME-Plus evaluation protocol."""
import asyncio
from collections import deque
from email.utils import parsedate_to_datetime
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import shutil
import time
import urllib.error
import urllib.request
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ASTRA_ROOT = HERE / "Astra-on-RoboMME"
sys.path[:0] = [str(HERE.parent), str(ROOT / "src"), str(ASTRA_ROOT / "examples/champ"),
               str(ASTRA_ROOT / "packages/openpi-client/src")]
from evaluation import Evaluation, EpisodeStop, PolicyStop, common_parser, write_json, select_cases
from privacy import sanitize_text
from astra_image_sampling import MAX_DEMONSTRATION_IMAGES, MAX_OBSERVATION_IMAGES, image_count
from three_tier_images import demonstration_indices, is_image_limit_error, request_image_budget


def recorded_responder(owner, key):
    from api_client import ResponsesClient

    class RecordedResponsesClient(ResponsesClient):
        def __call__(self, out):
            count = len(json.loads((Path(out) / 'request.json').read_text())['images'])
            if count > MAX_OBSERVATION_IMAGES:
                owner.stop_for_image_limit('local', count)
            try:
                return super().__call__(out)
            finally:
                # The upstream bridge catches transport exceptions; propagate an
                # image-budget stop as a scored failure rather than an API error.
                if owner.image_budget['stopped']:
                    raise EpisodeStop(f'limit:{MAX_OBSERVATION_IMAGES}_request_images')

        def wait(self, seconds, component):
            if seconds <= 0:
                return
            started = time.perf_counter()
            time.sleep(seconds)
            owner.recorder.event(component, "retry_wait", time.perf_counter() - started)

        def _send(self, request, out):
            payload = json.loads(request.data)
            count = image_count(payload['input'])
            if count > MAX_OBSERVATION_IMAGES:
                owner.stop_for_image_limit('local', count)
            owner.image_budget['submitted_requests'] += 1
            owner.image_budget['submitted_images'] += count
            owner.image_budget['largest_request_images'] = max(
                count, owner.image_budget['largest_request_images'])
            prompt = (out / "prompt.txt").read_text()
            for attempt in range(9):
                self.wait(max(0, self.next_request_at - time.monotonic()), "astra_rate_limit_wait")
                self.next_request_at = time.monotonic() + 20
                def send():
                    with urllib.request.urlopen(request, timeout=180) as response:
                        return json.loads(response.read()), response.headers.get("x-request-id")
                try:
                    return owner.recorder.call("astra_api", send, inputs=payload, prompt=prompt,
                        kind="api", model=payload["model"],
                        output_transform=lambda output: {**output[0], "transport_request_id": output[1]})
                except urllib.error.HTTPError as error:
                    body = error.read().decode("utf-8", errors="replace")
                    try:
                        detail = json.loads(body).get("error", {})
                    except ValueError:
                        detail = {}
                    code = detail.get("code") if isinstance(detail, dict) else None
                    kind = detail.get("type") if isinstance(detail, dict) else None
                    headers = {k: v for k, v in error.headers.items() if k.lower() in ("retry-after", "x-request-id") or k.lower().startswith("x-ratelimit-")}
                    write_json(out / f"http_error_{attempt:02d}.json", {"status": error.code, "body": self.scrub(body), "headers": headers})
                    if is_image_limit_error(detail):
                        owner.stop_for_image_limit('api', count)
                    retryable = error.code == 429 and (code in ("rate_limit_exceeded", "slow_down") or kind == "rate_limit_error")
                    if not retryable or attempt == 8:
                        raise RuntimeError(self.scrub(f"HTTP {error.code}: {body}")) from None
                    delay = min(120, 10 * 2 ** attempt)
                    retry_after = error.headers.get("Retry-After")
                    if retry_after:
                        try:
                            delay = max(delay, float(retry_after))
                        except ValueError:
                            try:
                                delay = max(delay, parsedate_to_datetime(retry_after).timestamp() - time.time())
                            except (TypeError, ValueError, OverflowError):
                                pass
                    self.wait(delay + random.uniform(0, 3), "astra_retry")
    return RecordedResponsesClient(key)


class ThreeTierAstraEvaluation(Evaluation):
    method = "3-tier-astra"
    api_applicable = True
    source_dependencies = [HERE / 'three_tier_images.py', HERE.parent / 'astra_image_sampling.py']

    def configuration(self):
        return {"model": "gpt-6-astra", "reasoning_effort": "medium", "execute_horizon": 16,
                "max_planner_calls": 24, "monitor_contract": "v6_bool_notime",
                "api_max_attempts": 9, "api_min_interval_seconds": 20,
                "server_metadata": self.client.get_server_metadata(),
                "stopcube_debounce_steps": 32, "history_stride": 3,
                "planner_image_budget": {"max_initial_images": MAX_DEMONSTRATION_IMAGES,
                    "max_request_images": MAX_OBSERVATION_IMAGES, "frames_per_sheet": 16,
                    "demo_sampling": "uniform_with_endpoints", "execution_history": "unchanged",
                    "scope": "independent_planner_request", "on_limit": "stop_episode_as_failure"}}

    def setup(self):
        for key, value in {"USE_HF": "1", "IMAGE_MAX_TOKEN_NUM": "128", "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}.items():
            os.environ.setdefault(key, value)
        from openpi_client.websocket_client_policy import MMEVLAWebsocketClientPolicy
        from release_utils import validate_checkpoints
        from runner import Monitor
        from swift.llm import InferRequest
        validate_checkpoints(self.args.vla_checkpoint, self.args.monitor_adapter)
        self.client = MMEVLAWebsocketClientPolicy(self.args.host, self.args.port)
        self.monitor = Monitor(str(self.args.monitor_base), str(self.args.monitor_adapter))
        self.infer_request = InferRequest
        self.responder = recorded_responder(self, os.environ["OPENAI_API_KEY"])

    def start_episode(self, obs, info):
        from core import Planner, Trigger, VIDEO_TASKS, memory_indices
        self.image_budget = request_image_budget()
        self.frames = [np.asarray(obs["front_rgb_list"][-1], dtype=np.uint8).copy()]
        demo = obs['front_rgb_list'][:-1] if self.task in VIDEO_TASKS else []
        indices, self.demo_sampling = demonstration_indices(len(demo),
            initial_memory_frames=len(memory_indices(self.task, 1)))
        self.demo = [np.asarray(demo[i], dtype=np.uint8).copy() for i in indices]
        write_json(self.recorder.directory / 'three_tier_demo_sampling.json', self.demo_sampling)
        self.recorder.call("policy_reset", self.client.reset, kind="transport")
        # Upstream reads its own request/response files during inference. Keep that
        # exchange private and unchanged; publish sanitized copies after each call.
        self.native_spool = tempfile.TemporaryDirectory(prefix="plus-astra-")
        self.native_exported = {}
        self.planner = Planner(self.native_spool.name, responder=self.responder)
        self.memory = set(memory_indices(self.task, 1))
        self.memory_cutoff = None
        self.completed, self.issued = [], []
        self.trigger = Trigger()
        self.subgoal = None
        self.command_start = self.planner_calls = 0
        self.continue_last = False
        self.actions = deque()

    def observe(self, obs, info):
        self.frames.append(np.asarray(obs["front_rgb_list"][-1], dtype=np.uint8).copy())

    def planner_call(self, function, *args):
        try:
            return function(*args)
        finally:
            started = time.perf_counter()
            try:
                source = Path(self.native_spool.name)
                for file in source.rglob("*"):
                    if not file.is_file() or file.is_symlink():
                        continue
                    stamp = (file.stat().st_mtime_ns, file.stat().st_size)
                    if self.native_exported.get(file) == stamp:
                        continue
                    destination = self.recorder.directory / "native/astra" / file.relative_to(source)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if file.suffix in (".json", ".txt"):
                        destination.write_text(sanitize_text(file.read_text(), file.suffix))
                    elif file.suffix in (".png", ".jpg", ".jpeg"):
                        shutil.copyfile(file, destination)
                    else:
                        raise ValueError("Unexpected Astra native artifact format")
                    self.native_exported[file] = stamp
            finally:
                self.recorder.disk_seconds += time.perf_counter() - started

    def end_episode(self, result):
        if getattr(self, 'image_budget', None) is not None:
            result['astra_image_budget'] = self.image_budget
            write_json(self.recorder.directory / 'astra_image_budget.json', self.image_budget)
        if getattr(self, "native_spool", None) is not None:
            self.native_spool.cleanup()
            self.native_spool = None

    def stop_for_image_limit(self, source, count):
        self.image_budget.update(stopped=True, stop_source=source, next_request_images=count)
        raise EpisodeStop(f'limit:{MAX_OBSERVATION_IMAGES}_request_images')

    def monitor_prediction(self, wrist):
        from input_contract import from_observations, parse_answer
        sample, ids = from_observations(self.task, self.goal, self.subgoal, self.frames, self.command_start, wrist)
        # Preserve upstream's file-backed InferRequest image path contract.
        native = {**sample, "images": [str(self.recorder.directory / self.recorder.artifact(image)["path"]) for image in sample["images"]]}
        prompt = "\n\n".join(f"{m['role']}:\n{m['content']}" for m in sample["messages"])
        text = self.recorder.call("system1_5", lambda: self.monitor.engine.infer(
            [self.infer_request(**native)], request_config=self.monitor.config)[0].choices[0].message.content,
            inputs={**sample, "frame_ids": ids, "command_start": self.command_start}, prompt=prompt)
        return parse_answer(text), ids

    def act(self, obs, info):
        from core import memory_indices, freeze_memory_at_press
        from runner import pack_state
        if self.actions:
            return self.actions.popleft()
        t = len(self.frames) - 1
        wrist = np.asarray(obs["wrist_rgb_list"][-1], dtype=np.uint8)
        signal = False
        if self.subgoal is not None and not self.continue_last:
            pred, ids = self.monitor_prediction(wrist)
            signal = self.trigger.update(pred, t, self.task)
            if signal and self.task == "ButtonUnmaskSwap" and self.subgoal.lower().startswith("press the second button at "):
                approved, rid, _ = self.recorder.call("astra_review", lambda: self.planner_call(self.planner.review_second_button,
                    self.goal, self.frames, wrist, self.subgoal, self.command_start, self.completed, self.issued, self.episode, t),
                    inputs={"subgoal": self.subgoal, "command_start": self.command_start, "native_directory": "native/astra"})
                signal = approved
                self.recorder.annotation(kind="second_button_review", approved=approved, request_id=rid)
            self.memory_cutoff = freeze_memory_at_press(self.task, self.subgoal, signal, t, self.memory_cutoff)
            self.memory = set(memory_indices(self.task, len(self.frames), self.memory_cutoff))
            self.recorder.annotation(kind="system1_5", subgoal=self.subgoal, prediction=pred, frame_ids=ids,
                replan=signal, command_start=self.command_start, memory_ids=sorted(self.memory), memory_cutoff=self.memory_cutoff)
        if self.subgoal is None or signal:
            if self.planner_calls >= 24:
                raise PolicyStop("limit:24_planner_calls")
            if self.subgoal is not None and self.task != "StopCube":
                self.completed.append(self.subgoal)
            next_subgoal, rid, _ = self.recorder.call("astra", lambda: self.planner_call(self.planner.predict,
                self.task, self.goal, self.frames, self.demo, self.memory, self.completed, self.issued, self.episode, t),
                inputs={"task_goal": self.goal, "completed": self.completed, "issued": self.issued,
                        "memory_ids": sorted(self.memory), "native_directory": "native/astra"})
            self.planner_calls += 1
            if next_subgoal is None:
                if self.subgoal is None:
                    raise PolicyStop("model_answer:no_initial_subgoal")
                self.continue_last = True
            else:
                self.subgoal, self.command_start = next_subgoal, t
                self.issued.append({"t": t, "subgoal": self.subgoal})
            self.recorder.annotation(kind="astra", subgoal=next_subgoal, active_subgoal=self.subgoal,
                continue_last=self.continue_last, request_id=rid, native_directory=f"native/astra/{rid}")
        inputs = {"observation/image": self.frames[-1], "observation/wrist_image": wrist,
                  "observation/state": pack_state(obs), "prompt": self.goal,
                  "grounded_subgoal": self.subgoal, "simple_subgoal": self.subgoal}
        prediction = self.recorder.call("pi05", lambda: self.client.infer(inputs), inputs=inputs,
                                       prompt=f"Task: {self.goal}\nSubgoal: {self.subgoal}")
        chunk = np.asarray(prediction["actions"])
        if chunk.ndim != 2 or chunk.shape[1] != 8 or len(chunk) < 16 or not np.isfinite(chunk).all():
            raise ValueError("Invalid VLA action chunk.")
        self.actions.extend(chunk[:16])
        self.recorder.annotation(kind="pi05", action_chunk=chunk[:16], execute_horizon=16, subgoal=self.subgoal)
        return self.actions.popleft()


def main():
    parser = common_parser(__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18762)
    parser.add_argument("--model-seed", type=int, default=42, help="Recorded VLA server seed; start the server with the same seed.")
    for option in ("vla-checkpoint", "monitor-base", "monitor-adapter"):
        parser.add_argument(f"--{option}", type=Path, default=os.environ.get(option.upper().replace("-", "_")))
    args = parser.parse_args()
    for option in ("vla_checkpoint", "monitor_base", "monitor_adapter"):
        value = getattr(args, option)
        if value is None or not value.is_dir():
            parser.error(f"Set --{option.replace('_', '-')} or ${option.upper()} to a local directory.")
    if not os.environ.get("OPENAI_API_KEY"):
        parser.error("Set OPENAI_API_KEY for the Astra planner.")
    asyncio.run(ThreeTierAstraEvaluation(args).run())


if __name__ == "__main__":
    main()
