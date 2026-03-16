"""Minimal RoboMME custom-env CPU sanity check.

This bypasses BenchmarkEnvBuilder/make_env_for_episode and instantiates a
RoboMME custom environment class directly.
"""

from __future__ import annotations

import argparse
import faulthandler
import os
import sys
import warnings
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def configure_cpu_only_runtime() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["NVIDIA_VISIBLE_DEVICES"] = "void"
    os.environ.pop("NVIDIA_DRIVER_CAPABILITIES", None)
    os.environ.pop("SAPIEN_RENDER_DEVICE", None)
    os.environ.pop("MUJOCO_GL", None)
    if "VK_ICD_FILENAMES" not in os.environ:
        lvp_icd = Path("/usr/share/vulkan/icd.d/lvp_icd.x86_64.json")
        if lvp_icd.exists():
            os.environ["VK_ICD_FILENAMES"] = str(lvp_icd)


configure_cpu_only_runtime()
faulthandler.enable(all_threads=True)

warnings.filterwarnings(
    "ignore",
    message=r"CUDA reports that you have .* fork_rng",
    category=UserWarning,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-class", default="StopCube")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import robomme.robomme_env as robomme_env

    env_cls = getattr(robomme_env, args.env_class)
    env = None

    print(f"instantiate start: env_class={args.env_class}", flush=True)
    try:
        env = env_cls(
            obs_mode="rgb+depth+segmentation",
            control_mode="pd_joint_pos",
            render_mode="rgb_array",
            reward_mode="dense",
            sim_backend="physx_cpu",
            render_backend="sapien_cpu",
            seed=args.seed,
        )
        print("instantiate ok", flush=True)

        obs, info = env.reset(seed=args.seed)
        print(f"reset ok: obs keys={list(obs.keys())}", flush=True)
        print(f"reset info keys={list(info.keys())}", flush=True)

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        rgb = obs["sensor_data"]["base_camera"]["rgb"]
        depth = obs["sensor_data"]["base_camera"]["depth"]

        print("step ok", flush=True)
        print(f"reward={reward}", flush=True)
        print(f"terminated={terminated}", flush=True)
        print(f"truncated={truncated}", flush=True)
        print(f"rgb shape={tuple(rgb.shape)} dtype={rgb.dtype}", flush=True)
        print(f"depth shape={tuple(depth.shape)} dtype={depth.dtype}", flush=True)
        print(f"step info keys={list(info.keys())}", flush=True)
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
