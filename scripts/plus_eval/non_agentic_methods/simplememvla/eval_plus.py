#!/usr/bin/env python3
"""SimpleMemVLA inference using the shared RoboMME-Plus evaluation protocol."""
import asyncio
from collections import deque
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np

HERE = Path(__file__).resolve().parent
SCRIPTS = next(parent for parent in HERE.parents if (parent / "benchmark_io").is_dir())
ROOT = SCRIPTS.parents[1]
POLICY_REPO = HERE / "SimpleMemVLA"
sys.path[:0] = [str(SCRIPTS), str(ROOT / "src"), str(POLICY_REPO)]
from evaluation import Evaluation, common_parser


class SimpleMemEvaluation(Evaluation):
    method = "simplememvla"

    def configuration(self):
        return {"seed": 0, "compute_dtype": "bfloat16", "num_denoising_steps": 10,
                "eval_temperature": 1.0, "max_subtask_tokens": 64, "execute_horizon": 16,
                "pipelined": False, "group_size": 1}

    def setup(self):
        from robomme_sim import eval_success as upstream
        from robomme_sim.robomme_env import encode_frames, encode_states
        import torch
        torch.manual_seed(0)
        np.random.seed(0)
        self.torch = torch
        self.encode_frames, self.encode_states = encode_frames, encode_states
        model_args = SimpleNamespace(
            pretrained_checkpoint=str(self.args.checkpoint), compute_dtype="bfloat16",
            attn_implementation=self.args.attn_implementation, pipelined=False,
            num_denoising_steps=10, eval_temperature=1.0, max_subtask_tokens=64,
        )
        self.policy, self.buffer_factory, self.normalize_state = upstream.build_policy(model_args)

    def start_episode(self, obs, info):
        self.buffer = self.buffer_factory()
        self.buffer.reset()
        self.cam_map = {k.split(".")[-1]: k for k in self.buffer.image_keys}
        self.actions = deque()
        self.observe(obs, info)

    def observe(self, obs, info):
        for frame in self.encode_frames(obs):
            self.buffer.observe({self.cam_map.get(k, k): np.asarray(v, dtype=np.uint8) for k, v in frame.items()})
        self.state = np.asarray(self.encode_states(obs)[-1], dtype=np.float32)

    def act(self, obs, info):
        if not self.actions:
            processed = self.recorder.call("prepare_inputs", lambda: self.buffer._prepare_inputs(self.goal),
                inputs={"task_goal": self.goal}, kind="preprocess",
                output_transform=lambda p: {k: v for k, v in p.items() if k != "policy"})
            processed_record = f"calls/{self.recorder.last_call_id}/output.json"
            state = None
            if self.normalize_state is not None:
                state = self.normalize_state.normalize(self.torch.from_numpy(self.state)).unsqueeze(0).to(self.policy.device)
            prompt = self.policy.processor.tokenizer.decode(processed["input_ids"][0], skip_special_tokens=False)
            # Patch keys refer to the same frame indices as the upstream history
            # sampler. Save source images as well as its preprocessed tensors.
            selected = {}
            for tag, ga, gb in processed["video_patch_keys"]:
                camera = next(k for k in self.buffer.history_image_keys if k.split(".")[-1] == tag)
                frames = self.buffer._buffers[camera]
                base = self.buffer._frames_seen[camera] - len(frames)
                for index in (ga, gb):
                    selected[f"{tag}_{index}"] = np.asarray(frames[index - base])
            inputs = {"normalized_state": state, "history_images": selected,
                      "current_wrist": obs["wrist_rgb_list"][-1],
                      # The preceding call already saved these exact tensors.
                      # Reference them instead of copying GPU buffers twice.
                      "processed": {"path": processed_record}}
            def generate():
                if self.policy.device.type == "cuda":
                    self.torch.cuda.synchronize(self.policy.device)
                output = self.policy.generate_batch([processed], [state])
                if self.policy.device.type == "cuda":
                    self.torch.cuda.synchronize(self.policy.device)
                return output
            decisions = self.recorder.call("policy", generate, inputs=inputs, prompt=prompt,
                output_transform=lambda d: {"actions": d[0][0], "subgoal": d[0][1]})
            actions, subgoal = decisions[0]
            chunk = np.asarray(actions[:16], dtype=np.float64)
            if chunk.shape != (16, 8) or not np.isfinite(chunk).all():
                raise ValueError(f"Expected 16 finite 8D actions, got {chunk.shape}.")
            self.recorder.annotation(subgoal=subgoal, action_chunk=chunk, execute_horizon=16)
            self.actions.extend(chunk)
        return self.actions.popleft()


def main():
    parser = common_parser(__doc__)
    parser.add_argument("--checkpoint", type=Path, default=os.environ.get("SIMPLEMEMVLA_CHECKPOINT"))
    parser.add_argument("--attn-implementation", choices=("flash_attention_2", "sdpa"), default="flash_attention_2")
    args = parser.parse_args()
    if args.checkpoint is None:
        parser.error("Supply --checkpoint or SIMPLEMEMVLA_CHECKPOINT.")
    args.checkpoint = args.checkpoint.expanduser().resolve()
    for name in ("config.json", "stats.json", "processor_config.json", "tokenizer.json"):
        if not (args.checkpoint / name).is_file():
            parser.error(f"Missing checkpoint file: {args.checkpoint / name}")
    if not any((args.checkpoint / n).is_file() for n in ("model.safetensors", "model.safetensors.index.json")):
        parser.error("Checkpoint has no safetensors weights.")
    asyncio.run(SimpleMemEvaluation(args).run())


if __name__ == "__main__":
    main()
