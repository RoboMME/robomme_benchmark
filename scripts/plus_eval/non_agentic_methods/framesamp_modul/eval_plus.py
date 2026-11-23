#!/usr/bin/env python3
"""FrameSamp+Modul inference using the shared RoboMME-Plus evaluation protocol."""
import asyncio
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
SCRIPTS = next(parent for parent in HERE.parents if (parent / "benchmark_io").is_dir())
ROOT = SCRIPTS.parents[1]
POLICY_REPO = HERE / "robomme_policy_learning"
HISTORY_CONFIG = "perceptual-framesamp-modul.yaml"
EXEC_HORIZON = 16
sys.path[:0] = [str(SCRIPTS), str(ROOT / "src"), str(POLICY_REPO / "packages/openpi-client/src")]
from evaluation import Evaluation, PolicyUnavailable, common_parser


def load_upstream_utils():
    path = POLICY_REPO / "examples/robomme/utils.py"
    spec = importlib.util.spec_from_file_location("robomme_vla_eval_utils", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_checkpoint(checkpoint):
    checkpoint = checkpoint.expanduser().resolve()
    for name in ("params", "assets"):
        if not (checkpoint / name).is_dir():
            raise ValueError(f"Missing checkpoint directory: {checkpoint / name}")
    config = checkpoint.parent / "history_config.txt"
    # Upstream reads this file verbatim, including any trailing newline.
    if not config.is_file() or config.read_text() != HISTORY_CONFIG:
        raise ValueError(f"{config} must contain exactly {HISTORY_CONFIG!r}, without a newline.")
    return checkpoint


def unpack_observation(obs, index=-1):
    # Same state packing as upstream examples/robomme/env_runner.py.
    state = np.concatenate([
        obs["joint_state_list"][index], obs["gripper_state_list"][index][:1],
    ], dtype=np.float32)
    return obs["front_rgb_list"][index], obs["wrist_rgb_list"][index], state


class FrameSampEvaluation(Evaluation):
    method = "framesamp-modul"

    def configuration(self):
        return {"history_config": HISTORY_CONFIG, "execute_horizon": EXEC_HORIZON,
                "server_metadata": self.client.get_server_metadata(),
                "memory_buffer_sha256": hashlib.sha256((POLICY_REPO / "src/mme_vla_suite/shared/mem_buffer.py").read_bytes()).hexdigest(),
                "server_configuration_source": "operator-supplied checkpoint and model-seed arguments"}

    def setup(self):
        from openpi_client.websocket_client_policy import MMEVLAWebsocketClientPolicy
        from openpi_client import msgpack_numpy
        class CheckedClient(MMEVLAWebsocketClientPolicy):
            def request(self, payload):
                self._ws.send(self._packer.pack(payload))
                reply = self._ws.recv()
                if isinstance(reply, str):
                    raise PolicyUnavailable(reply)
                return msgpack_numpy.unpackb(reply)
            def reset(self):
                return self.request({'reset': True})
            def add_buffer(self, buffer):
                return self.request(buffer)
        self.upstream = load_upstream_utils()
        self.client = CheckedClient(self.args.host, self.args.port)

    def policy_call(self, component, callback, **kwargs):
        try:
            return self.recorder.call(component, callback, **kwargs)
        except Exception as exc:
            raise PolicyUnavailable(f'{component}: {exc}') from exc

    def start_episode(self, obs, info):
        reply = self.policy_call("policy_reset", self.client.reset, kind="transport")
        if not reply.get("reset_finished", False):
            raise PolicyUnavailable("Policy server did not acknowledge reset.")
        self.state = self.upstream.EpisodeState()
        self.observe(obs, info)
        self.state.exec_start_idx = len(self.state.image_buffer) - 1

    def observe(self, obs, info):
        for index in range(len(obs["front_rgb_list"])):
            self.state.add_observation(*unpack_observation(obs, index))

    def act(self, obs, info):
        state = self.state
        image, wrist, robot = state.get_current_obs()
        if not state.action_plan:
            history = self.upstream.pack_buffer(state.image_buffer, state.state_buffer, state.exec_start_idx)
            reply = self.policy_call("history_upload", lambda: self.client.add_buffer(history), inputs=history, kind="transport")
            if not reply.get("add_buffer_finished", False):
                raise PolicyUnavailable("Policy server did not acknowledge history update.")
            inputs = {"observation/image": image, "observation/wrist_image": wrist,
                      "observation/state": robot, "prompt": self.goal}
            prediction = self.policy_call("policy", lambda: self.client.infer(inputs), inputs=inputs, prompt=self.goal)
            chunk = np.asarray(prediction["actions"])[:EXEC_HORIZON]
            if chunk.shape != (EXEC_HORIZON, 8) or not np.isfinite(chunk).all():
                raise ValueError(f"Expected 16 finite 8D actions, got {chunk.shape}.")
            self.recorder.annotation(action_chunk=chunk, execute_horizon=EXEC_HORIZON)
            state.action_plan.extend(chunk)
            state.clear_buffers()
        return state.action_plan.popleft()


def main():
    parser = common_parser(__doc__)
    parser.add_argument("--method-name", choices=("framesamp-modul", "framesamp-modul-plus", "framesamp-modul-ood"),
                        default="framesamp-modul", help="Result identity; all names use the same inference protocol.")
    parser.add_argument("--checkpoint", type=Path, default=os.environ.get("VLA_CHECKPOINT"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18763)
    parser.add_argument("--model-seed", type=int, default=7, help="Recorded server seed; start the server with this same seed.")
    args = parser.parse_args()
    if args.checkpoint is None:
        parser.error("Supply --checkpoint or VLA_CHECKPOINT.")
    args.checkpoint = check_checkpoint(args.checkpoint)
    method = vars(args).pop("method_name")
    evaluation = FrameSampEvaluation(args)
    evaluation.method = method
    asyncio.run(evaluation.run())


if __name__ == "__main__":
    main()
