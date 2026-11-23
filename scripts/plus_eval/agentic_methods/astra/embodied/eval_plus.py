#!/usr/bin/env python3
"""Original Astra waypoint v3 policy with unified RoboMME-Plus recording."""
import asyncio
import importlib.util
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
SCRIPTS = next(parent for parent in HERE.parents if (parent / "benchmark_io").is_dir())
ROOT = SCRIPTS.parents[1]
HARNESS = HERE / "embodied-astra-robomme/harness/astra_waypoint_eval_v3.py"
sys.path[:0] = [str(SCRIPTS), str(ROOT / "src")]
from evaluation import Evaluation, EpisodeStop, PolicyStop, common_parser, write_json
from agentic_methods.astra.image_sampling import (CONFIGURATION as DEMONSTRATION_SAMPLING,
                                 MAX_DEMONSTRATION_IMAGES, MAX_OBSERVATION_IMAGES,
                                 image_count, observation_indices)


def observation_content(harness, obs, info, label, first=False, max_demo_images=MAX_DEMONSTRATION_IMAGES):
    """Keep both endpoints and current wrist while bounding long benchmark videos."""
    count = len(obs["front_rgb_list"])
    indices, stride, report = observation_indices(count, harness.FRAME_STRIDE, first=first,
                                                max_demo_images=max_demo_images)
    capped_demo = report is not None and report["applied"]
    if stride == harness.FRAME_STRIDE and not capped_demo:
        return harness.observation(obs, info, label, first=first)
    sampling = (f"Uniformly sampled across the full demonstration to {report['input_images']} "
                "total input images, including the current wrist view:" if capped_demo else
                f"Sampling every {stride} control steps to fit the {MAX_OBSERVATION_IMAGES}-image request limit:")
    content = [harness.text_item(
        f"{label}: front-camera frames {indices} of {count} (last = current). "
        + sampling)]
    content.extend(harness.image_item(obs["front_rgb_list"][i]) for i in indices)
    content.extend([harness.text_item("Current wrist camera:"),
                    harness.image_item(obs["wrist_rgb_list"][-1]),
                    harness.state_text(obs, info, first)])
    return content


def load_harness():
    spec = importlib.util.spec_from_file_location("embodied_astra_waypoint_v3", HARNESS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Parameterize the benchmark prompt without changing the upstream harness file.
    module.SYSTEM = module.SYSTEM.replace("after 1300 control steps.",
                                          "after EPISODE_MAX_STEPS control steps.")
    return module


def clock_info(info):
    return {**info, "elapsed_steps": info.get("execution_control_steps", 0)}


def is_image_limit_error(error):
    body = getattr(error, "body", None)
    if not isinstance(body, dict):
        return False
    detail = body.get("error", body)
    return isinstance(detail, dict) and "max_images_per_request" in (detail.get("type"), detail.get("code"))


class EmbodiedAstraEvaluation(Evaluation):
    method = "embodied-astra"
    action_space = "waypoint"
    api_applicable = True
    observation_options = {f"include_{camera}_camera_{field}": True
                           for camera in ("front", "wrist") for field in ("intrinsic", "extrinsic")}

    def configuration(self):
        return {"harness": "astra_waypoint_eval_v3", "max_model_calls": self.harness.MAX_MODEL_CALLS,
                "max_consecutive_noops": self.harness.MAX_CONSECUTIVE_NOOPS,
                "max_observation_images": MAX_OBSERVATION_IMAGES,
                "demonstration_image_sampling": DEMONSTRATION_SAMPLING,
                "conversation_image_limit": {"max_images": MAX_OBSERVATION_IMAGES,
                                             "on_limit": "fail_episode"},
                "api_max_attempts": 6, "sdk_retries": 0, "original_prompts_preserved": False,
                "system_prompt_max_steps": "environment_info.max_steps"}

    def setup(self):
        from openai import OpenAI
        self.harness = load_harness()
        self.harness.FRAME_STRIDE = self.args.frame_stride
        self.harness.PROMPT_VARIANT = self.args.prompt_variant
        # Keep the harness's six bounded attempts; disable invisible SDK retries
        # so every actual request and its latency is recorded.
        self.client = OpenAI(max_retries=0)

    def demonstration_image_limit(self):
        return MAX_DEMONSTRATION_IMAGES

    def system_prompt(self, info):
        return self.harness.system_prompt(self.task).replace("EPISODE_MAX_STEPS", str(info["max_steps"]))

    def initial_content(self, obs, info):
        return [self.harness.text_item(f"Task goal: {self.goal}")] + observation_content(
            self.harness, obs, info, "Video so far", first=True,
            max_demo_images=self.demonstration_image_limit())

    def start_episode(self, obs, info):
        self.latest_obs, self.latest_info = obs, clock_info(info)
        self.response = None
        self.calls = self.noops = 0
        # previous_response_id retains all earlier input images, across trials.
        self.image_budget = {"max_total_images": MAX_OBSERVATION_IMAGES,
                             "submitted_images": 0, "stopped": False}
        self.had_frames = True
        _, _, self.demo_sampling = observation_indices(
            len(obs["front_rgb_list"]), self.harness.FRAME_STRIDE, first=True,
            max_demo_images=self.demonstration_image_limit())
        self.demo_sampling_directory = self.recorder.directory
        write_json(self.recorder.directory / "astra_demonstration_sampling.json", self.demo_sampling)

    def end_episode(self, result):
        # A reset/build error may precede start_episode; never reuse another
        # episode's report when that happens.
        if getattr(self, "demo_sampling_directory", None) == self.recorder.directory:
            result["astra_demonstration_sampling"] = self.demo_sampling
            result["astra_image_budget"] = self.image_budget
            write_json(self.recorder.directory / "astra_image_budget.json", self.image_budget)

    def stop_for_image_limit(self, source, request_images):
        self.image_budget.update(stopped=True, stop_source=source,
                                 next_request_images=request_images,
                                 projected_total_images=self.image_budget["submitted_images"] + request_images)
        raise EpisodeStop(f"limit:{MAX_OBSERVATION_IMAGES}_total_images")

    def observe(self, obs, info):
        if info.get("trial_reset"):
            # Retain the conversation so the policy can learn from the miss;
            # give the fresh physical trial the same policy call/no-op caps.
            self.calls = self.noops = 0
        self.had_frames = bool(len(obs.get("front_rgb_list", [])))
        if self.had_frames:
            self.noops = 0
            self.latest_obs, self.latest_info = obs, clock_info(info)
        else:
            self.noops += 1

    def act(self, obs, info):
        h = self.harness
        if self.noops >= h.MAX_CONSECUTIVE_NOOPS:
            raise PolicyStop(f"limit:{h.MAX_CONSECUTIVE_NOOPS}_consecutive_noops")
        if self.calls >= h.MAX_MODEL_CALLS:
            raise PolicyStop(f"limit:{h.MAX_MODEL_CALLS}_model_calls")
        obs, info = self.latest_obs, self.latest_info
        if self.response is None:
            content = self.initial_content(obs, info)
        elif self.had_frames:
            content = observation_content(h, obs, info, "Frames recorded during that motion")
        else:
            content = [h.text_item(h.NOOP_TEXT), h.text_item("Current front camera:"), h.image_item(obs["front_rgb_list"][-1]),
                       h.text_item("Current wrist camera:"), h.image_item(obs["wrist_rgb_list"][-1]), h.state_text(obs, info)]
        if info.get("trial_reset"):
            content.insert(0, h.text_item(info["message"] + " The control-step count has restarted at zero."))
        request = {"model": self.args.model, "instructions": self.system_prompt(info),
                   "input": [{"role": "user", "content": content}]}
        if self.response is not None:
            request["previous_response_id"] = self.response.id
        request_images = image_count(content)
        total_images = self.image_budget["submitted_images"] + request_images
        if total_images > MAX_OBSERVATION_IMAGES:
            self.stop_for_image_limit("preflight", request_images)
        prompt = request["instructions"] + "\n\n" + "\n".join(p.get("text", "[image: see input.json]") for p in content)
        if self.response is not None:
            prompt = f"Continues previous_response_id={self.response.id}\n\n" + prompt

        def ask():
            for attempt in range(6):
                try:
                    return self.recorder.call("astra_api", lambda: self.client.responses.create(**request),
                        inputs=request, prompt=prompt, kind="api", model=self.args.model,
                        output_transform=lambda r: r.model_dump(mode="json"))
                except h.openai.BadRequestError as error:
                    if is_image_limit_error(error):
                        self.stop_for_image_limit("api", request_images)
                    raise
                except h.RETRYABLE:
                    if attempt == 5:
                        raise
                    t = time.perf_counter()
                    time.sleep(2 ** attempt * 5)
                    self.recorder.event("astra_retry", "retry_wait", time.perf_counter() - t, attempt=attempt)
        self.response = self.recorder.call("astra", ask, inputs=request, prompt=prompt,
            output_transform=lambda r: r.model_dump(mode="json"))
        self.image_budget["submitted_images"] = total_images
        self.calls += 1
        answer = h.parse_answer(self.response.output_text)
        action, kind = h.to_action(answer)
        self.recorder.annotation(answer=answer, raw_output=self.response.output_text, action=action, invalid_kind=kind)
        if action is None:
            raise PolicyStop(f"model_answer:{kind}")
        return action


def main():
    parser = common_parser(__doc__)
    parser.add_argument("--model", default="gpt-6-astra")
    parser.add_argument("--frame-stride", type=int, default=3)
    parser.add_argument("--prompt-variant", choices=("base", "lr_note"), default="base")
    args = parser.parse_args()
    if args.frame_stride < 1:
        parser.error("--frame-stride must be positive.")
    if not HARNESS.is_file():
        parser.error("Initialize the embodied-astra-robomme submodule.")
    asyncio.run(EmbodiedAstraEvaluation(args).run())


if __name__ == "__main__":
    main()
