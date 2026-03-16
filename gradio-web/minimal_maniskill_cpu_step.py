"""Minimal ManiSkill CPU sim/render sanity check.

This uses an official ManiSkill environment instead of RoboMME wrappers so the
execution path stays as small as possible.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import gymnasium as gym
import mani_skill.envs  # noqa: F401 - registers ManiSkill environments


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import robomme  # noqa: F401,E402 - applies ManiSkill PCI render-backend patch


os.environ.setdefault("ROBOMME_RENDER_BACKEND", "pci:0")


warnings.filterwarnings(
    "ignore",
    message=r"CUDA reports that you have .* fork_rng",
    category=UserWarning,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="PickCube-v1")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = gym.make(
        args.env_id,
        obs_mode="rgbd",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        sim_backend="physx_cpu",
        render_backend=os.environ["ROBOMME_RENDER_BACKEND"],
    )

    try:
        obs, info = env.reset(seed=args.seed)
        print(f"reset ok: env_id={args.env_id}")
        print(f"obs keys: {list(obs.keys())}")
        print(f"info keys: {list(info.keys())}")

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        rgb = obs["sensor_data"]["base_camera"]["rgb"]
        depth = obs["sensor_data"]["base_camera"]["depth"]

        print("step ok")
        print(f"reward={reward}")
        print(f"terminated={terminated}")
        print(f"truncated={truncated}")
        print(f"rgb shape={tuple(rgb.shape)} dtype={rgb.dtype}")
        print(f"depth shape={tuple(depth.shape)} dtype={depth.dtype}")
        print(f"info keys after step: {list(info.keys())}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
