#!/usr/bin/env python3
"""Original PonderPounce callbacks with unified RoboMME-Plus recording."""
import asyncio
from dataclasses import fields
import json
import os
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
SCRIPTS = next(parent for parent in HERE.parents if (parent / "benchmark_io").is_dir())
ROOT = SCRIPTS.parents[1]
POLICY_REPO = HERE / "PonderPounce"
sys.path[:0] = [str(HERE), str(SCRIPTS), str(ROOT / "src"), str(POLICY_REPO)]
from evaluation import Evaluation, common_parser
from privacy import write_text


class ActionContext:
    def __init__(self, session_id):
        self.session_id = session_id
        self.action = None

    async def send_action(self, response):
        if self.action is not None:
            raise RuntimeError("Server sent multiple actions for one observation.")
        action = np.asarray(response["actions"])
        if action.shape != (1, 8) or not np.isfinite(action).all():
            raise ValueError(f"Expected one finite 8D joint action, got {action.shape}.")
        self.action = action[0].copy()


def panda_gripper_limits():
    """Read the physical limits behind RoboMME's normalized Panda gripper."""
    from mani_skill.agents.robots import Panda

    config = object.__new__(Panda)._controller_configs["pd_joint_pos"]["gripper"]
    lower, upper = float(config.lower), float(config.upper)
    if config.use_delta or not config.normalize_action or upper <= lower:
        raise RuntimeError("PonderPounce hold requires a normalized absolute Panda gripper controller")
    return lower, upper


def recorded_server_class(owner):
    from ponderpounce.eval.robomme_server import PonderPounceRoboMMEServer
    from context_window import WindowedS2Context

    gripper_lower, gripper_upper = panda_gripper_limits()

    class TracedContext(WindowedS2Context):
        """Record appends, rollbacks and bounded-context rebuilds."""
        def __init__(self, **kwargs):
            self.trace = []
            self.last_result = None
            super().__init__(**kwargs)

        def _append(self, segment, *, total_queries):
            entry = {"operation": "append", "context_length_before": self._context_len,
                     "input_ids": segment.input_ids.tolist(),
                     "mm_token_type_ids": segment.mm_token_type_ids.tolist(),
                     "cognition_mask": segment.cognition_mask.tolist(),
                     "s2_query_ids": segment.s2_query_ids.tolist(),
                     "image_grid_thw": segment.image_grid_thw.tolist() if segment.image_grid_thw is not None else None}
            self.trace.append(entry)
            return super()._append(segment, total_queries=total_queries)

        def _restore(self, snapshot):
            self.trace.append({"operation": "restore", "context_length": snapshot.context_len})
            return super()._restore(snapshot)

        def on_truncation(self, event):
            self.trace.append(dict(event))

        def fire(self, obs_pils):
            self.last_result = super().fire(obs_pils)
            return self.last_result

    class RecordedServer(PonderPounceRoboMMEServer):
        initial_hold_gripper_limits = (gripper_lower, gripper_upper)

        def _hold(self, raw_state):
            action = super()._hold(raw_state)
            # Proprio reports finger positions in metres; the gripper action
            # uses [-1, 1]. Correct only the startup hold. Learned chunks already
            # contain normalized gripper commands and bypass this method.
            if (raw_state is not None and action.shape[-1] >= 8
                    and np.asarray(raw_state).size >= 8
                    and getattr(owner, "task", None) not in ("PatternLock", "RouteStick")):
                position = np.asarray(raw_state).reshape(-1)[7]
                action[0, 7] = np.clip(
                    2 * (position - gripper_lower) / (gripper_upper - gripper_lower) - 1,
                    -1, 1,
                )
            return action

        def _fire_s2(self, ep, obs, now):
            rec = owner.recorder
            initial = ep.s2_context is None
            if not initial:
                ep.s2_context.trace = []
            inputs = {"task_goal": ep.session.task_text, "simulation_time_ns": now,
                      "images": self._adapter.camera_pils(obs),
                      "demonstration_images": ep.session.demo_images if initial else [],
                      "previous_ponder_call": owner.previous_ponder_call,
                      "current_obs_only": bool(getattr(self._s2, "current_obs_only", False))}

            def fire():
                if ep.s2_context is None:
                    ep.s2_context = TracedContext(session=ep.session, system2=self._s2, device=self._device,
                        dtype=self._dtype, max_new_tokens=self._s2_max_new_tokens,
                        context_tokens=getattr(owner.args, "ponder_context_tokens", None))
                super(RecordedServer, self)._fire_s2(ep, obs, now)
                output = ep.s2_context.last_result
                return {"result": {field.name: getattr(output, field.name) for field in fields(output)},
                        "append_trace": ep.s2_context.trace,
                        "context_length": ep.s2_context._context_len,
                        "context_truncation_count": ep.s2_context.truncation_count,
                        "active_subgoal": ep.active_subgoal}
            output = rec.call("ponder", fire, inputs=inputs)
            owner.previous_ponder_call = rec.last_call_id
            # This is an append-only sequence, including generated tokens and
            # branch rollbacks. Prior call IDs identify the cached context.
            tokenizer = ep.session.s2_processor.tokenizer
            prompt = [f"Task: {ep.session.task_text}",
                      f"Previous context call: {inputs['previous_ponder_call']}",
                      "Exact sequence operations (input images are in input.json; generated tokens also appear here):"]
            for segment in output["append_trace"]:
                if segment["operation"] == "restore":
                    prompt.append(f"[restore context length {segment['context_length']}]")
                elif segment["operation"] == "truncate":
                    prompt.append("[truncate context]\n" + json.dumps(segment))
                else:
                    prompt.append(f"[append at {segment['context_length_before']}; image grids={segment['image_grid_thw']}]\n"
                                  + tokenizer.decode(segment["input_ids"], skip_special_tokens=False))
            write_text(rec.directory / "calls" / rec.last_call_id / "prompt.txt", "\n\n".join(prompt))
            result = output["result"]
            rec.annotation(kind="ponder", subgoal=result["subgoal_text"], active_subgoal=ep.active_subgoal,
                           reasoning=result["reasoning_text"], gate_score=result["gate_score"],
                           transition_kind=result["kind"], ready_at_ns=ep.cognitions[-1].ready_at_ns)

        def _fire_s1(self, ep, obs, now):
            # Capture exactly the arguments the original scheduler passes to Pi0.5,
            # including cognition, its age, the deterministic noise, and prompt IDs.
            original = self._s1.predict_action
            def predict(**kwargs):
                ids = kwargs.get("symbolic_prompt_ids")
                prompt = (self._s1_tokenizer.decode(ids[0].tolist(), skip_special_tokens=False)
                          if ids is not None else "No symbolic prompt; conditioning is in input.json.")
                return owner.recorder.call("pounce", lambda: original(**kwargs), inputs=kwargs, prompt=prompt)
            self._s1.predict_action = predict
            try:
                super()._fire_s1(ep, obs, now)
            finally:
                self._s1.predict_action = original
            owner.recorder.annotation(kind="pounce", active_subgoal=ep.active_subgoal,
                                      simulation_time_ns=now, action_chunk=ep.chunk.actions)
    return RecordedServer


class PonderPounceEvaluation(Evaluation):
    method = "ponderpounce"
    source_dependencies = (HERE / "context_window.py",)

    def configuration(self):
        return {"compute_dtype": str(self.server._dtype), "control_hz": 20,
                "ponder_device": str(self.server._device),
                "pounce_device": str(self.server._s1._device),
                "s2_period_ns": self.server._s2_period_ns, "s1_period_ns": self.server._s1_period_ns,
                "s2_delay_ns": self.server._s2_delay_ns, "wait_first_cognition": self.server._wait_first_cognition,
                "s2_max_new_tokens": self.server._s2_max_new_tokens,
                "initial_hold_gripper": "panda_position_to_normalized_action_v1",
                "initial_hold_gripper_limits_m": self.server.initial_hold_gripper_limits,
                "context_policy": "task_demo_recent_complete_queries_v1",
                "context_token_cap": (getattr(self.args, "ponder_context_tokens", None)
                                      or int(self.server._s2.effective_context_cap)),
                "session_id_pattern": "ood_{task}_ep{episode}"}

    def setup(self):
        from vla_eval.benchmarks.robomme.benchmark import RoboMMEBenchmark
        self.mapper_class = RoboMMEBenchmark
        ponder_device = getattr(self.args, 'ponder_device', 'cuda:0')
        pounce_device = getattr(self.args, 'pounce_device', ponder_device)
        self.server = recorded_server_class(self)(str(self.args.checkpoint),
                                                  seed=self.args.model_seed, device=ponder_device)
        if pounce_device != ponder_device:
            from ponderpounce.inference.local import LocalSystem1
            # LocalSystem1 places Pi0.5 and all input tensors on its own device;
            # Ponder's context/cache stay on the server's original device.
            self.server._s1 = LocalSystem1(self.server._model, device=pounce_device,
                                          dtype=self.server._dtype, num_inference_steps=10)
            import torch
            torch.cuda.empty_cache()

    async def start_episode(self, obs, info):
        self.ctx = ActionContext(f"ood_{self.task}_ep{self.episode}")
        self.previous_ponder_call = None
        self.mapper = self.mapper_class(tasks=[self.task], action_space="joint_angle", send_subgoal=False)
        self.mapper._task_description = self.goal
        self.mapper._video_frames = list(obs["front_rgb_list"][:-1])
        self.mapper._wrist_video_frames = list(obs["wrist_rgb_list"][:-1])
        await self.server.on_episode_start({"task": {"task_description": self.goal}}, self.ctx)

    async def act(self, obs, info):
        self.ctx.action = None
        await self.server.on_observation(self.mapper.make_obs(obs, {"name": self.task}), self.ctx)
        if self.ctx.action is None:
            raise RuntimeError("Policy callback returned without an action.")
        return self.ctx.action

    async def end_episode(self, result):
        if hasattr(self, "ctx"):
            await self.server.on_episode_end({"success": result["success"]}, self.ctx)


def main():
    parser = common_parser(__doc__)
    parser.add_argument("--checkpoint", type=Path, default=os.environ.get("PONDERPOUNCE_CHECKPOINT"))
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument('--ponder-device', default='cuda:0')
    parser.add_argument('--pounce-device', default='cuda:0')
    parser.add_argument('--ponder-context-tokens', type=int,
                        help='Optional smaller Ponder context cap; default: checkpoint/cache limit.')
    args = parser.parse_args()
    if args.checkpoint is None:
        parser.error("Supply --checkpoint or PONDERPOUNCE_CHECKPOINT.")
    args.checkpoint = args.checkpoint.expanduser().resolve()
    for name in ("config.json", "model.safetensors", "norm_stats.json"):
        if not (args.checkpoint / name).is_file():
            parser.error(f"Missing checkpoint file: {args.checkpoint / name}")
    asyncio.run(PonderPounceEvaluation(args).run())


if __name__ == "__main__":
    main()
