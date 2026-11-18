"""Replay RoboMME-Plus training actions in the simulator and save front/wrist MP4 videos."""

import argparse
import json
from pathlib import Path
import sys

import h5py
import imageio.v2 as imageio
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from robomme_plus import BenchmarkEnvBuilder
from robomme_plus.evaluation.metadata import replay_identity


def write_frames(writer, obs, demonstration=False):
    def numpy(image):
        return image.detach().cpu().numpy() if hasattr(image, "detach") else image

    pairs = zip(obs["front_rgb_list"], obs["wrist_rgb_list"], strict=True)
    for index, (front, wrist) in enumerate(pairs):
        frame = np.concatenate([numpy(front), numpy(wrist)], axis=1).astype(np.uint8)
        if demonstration and index < len(obs["front_rgb_list"]) - 1:
            frame[:3] = frame[-3:] = (255, 0, 0)
            frame[:, :3] = frame[:, -3:] = (255, 0, 0)
        writer.append_data(frame)


def replay(task, index, episode, output_dir, action_space):
    builder = BenchmarkEnvBuilder(task, dataset="train", action_space=action_space)
    metadata = builder.get_episode_metadata(index)
    recorded = json.loads(np.asarray(episode["setup/episode_metadata"][()]).item())
    fps = int(episode["setup/control_freq"][()])
    if (replay_identity(task, recorded) != replay_identity(task, metadata)
            or fps != metadata["control_freq"]):
        raise ValueError(f"{task} episode {index}: H5 and packaged training metadata differ")

    path = output_dir / task / f"episode_{index}_{action_space}.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)
    steps = sorted((key for key in episode if key.startswith("timestep_")),
                   key=lambda key: int(key.split("_")[1]))
    env = builder.make_env_for_episode(index)
    try:
        with imageio.get_writer(path, fps=fps, codec="libx264", quality=7) as writer:
            obs, info = env.reset()
            write_frames(writer, obs, demonstration=True)
            for key in steps:
                step = episode[key]
                if bool(step["info/is_video_demo"][()]):
                    continue  # env.reset() already performed the demonstration.
                field = "eef_action" if action_space == "ee_pose" else "joint_action"
                action = np.asarray(step[f"action/{field}"][()], dtype=np.float64)
                obs, _, terminated, truncated, info = env.step(action)
                if obs:
                    write_frames(writer, obs)
                if terminated or truncated:
                    break
    finally:
        env.close()
    status = info.get("status", "ongoing")
    print(f"{task} episode {index} ({action_space}): {status} -> {path}", flush=True)
    if info.get("failure_reason"):
        print(info["failure_reason"], flush=True)
    if info.get("error_message"):
        print(info["error_message"], flush=True)
    return status == "success"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5-data-dir", type=Path, required=True,
                        help="Directory containing record_dataset_<Task>.h5 files")
    parser.add_argument("--tasks", nargs="+", choices=BenchmarkEnvBuilder.get_task_list())
    parser.add_argument("--episodes", nargs="+", type=int, help="Episode indices within each task")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/plus_dataset_replay"))
    parser.add_argument("--action-space", choices=("both", "joint_angle", "ee_pose"),
                        default="both", help="Replay both saved action formats by default")
    args = parser.parse_args()
    files = ([args.h5_data_dir / f"record_dataset_{task}.h5" for task in args.tasks]
             if args.tasks else sorted(args.h5_data_dir.glob("record_dataset_*.h5")))
    if not files:
        parser.error(f"No training H5 files found in {args.h5_data_dir}")
    passed = total = 0
    for path in files:
        task = path.stem.removeprefix("record_dataset_")
        with h5py.File(path, "r") as dataset:
            indices = args.episodes if args.episodes is not None else sorted(
                int(key.removeprefix("episode_")) for key in dataset if key.startswith("episode_"))
            for index in indices:
                if f"episode_{index}" not in dataset:
                    parser.error(f"{path.name} has no episode {index}")
                spaces = ("joint_angle", "ee_pose") if args.action_space == "both" else (args.action_space,)
                for space in spaces:
                    passed += replay(task, index, dataset[f"episode_{index}"], args.output_dir, space)
                    total += 1
    print(f"{passed}/{total} replays succeeded.")
    return 0 if total and passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
