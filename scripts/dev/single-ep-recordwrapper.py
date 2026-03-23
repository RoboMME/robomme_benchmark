import argparse
import os
from pathlib import Path
from typing import Optional, Set

import gymnasium as gym
import numpy as np
import torch

from robomme.env_record_wrapper import RobommeRecordWrapper, FailsafeTimeout
from robomme.robomme_env import *
from robomme.robomme_env.utils.SceneGenerationError import SceneGenerationError
from robomme.robomme_env.utils.planner_fail_safe import (
    FailAwarePandaArmMotionPlanningSolver,
    FailAwarePandaStickMotionPlanningSolver,
    ScrewPlanFailure,
)

DEFAULT_ENVS = [
    "PickXtimes",
    "StopCube",
    "SwingXtimes",
    "BinFill",
    "VideoUnmaskSwap",
    "VideoUnmask",
    "ButtonUnmaskSwap",
    "ButtonUnmask",
    "VideoRepick",
    "VideoPlaceButton",
    "VideoPlaceOrder",
    "PickHighlight",
    "InsertPeg",
    "MoveCube",
    "PatternLock",
    "RouteStick",
]
VALID_ENVS: Set[str] = set(DEFAULT_ENVS)
VALID_DIFFICULTIES: Set[str] = {"easy", "medium", "hard"}
DATASET_SCREW_MAX_ATTEMPTS = 3
DATASET_RRT_MAX_ATTEMPTS = 3


def _latest_recorded_mp4(
    output_root: Path, env_id: str, episode: int, seed: int
) -> Optional[Path]:
    """Resolve the mp4 written by RobommeRecordWrapper for this run (newest match by mtime)."""
    videos_dir = output_root / "videos"
    if not videos_dir.is_dir():
        return None
    tag = f"{env_id}_ep{episode}_seed{seed}"
    candidates = [p for p in videos_dir.glob("*.mp4") if tag in p.name]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _tensor_to_bool(value) -> bool:
    if value is None:
        return False
    if isinstance(value, torch.Tensor):
        return bool(value.detach().cpu().bool().item())
    if isinstance(value, np.ndarray):
        return bool(np.any(value))
    return bool(value)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a single Robomme episode and record video only."
    )
    parser.add_argument(
        "--env",
        "-e",
        default="ButtonUnmaskSwap",
        choices=sorted(VALID_ENVS),
        help="Environment ID to run (default: ButtonUnmaskSwap).",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=1,
        help="Episode index to run (default: 0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Environment seed to use (default: 0).",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        default="hard",
        choices=sorted(VALID_DIFFICULTIES),
        help="Episode difficulty (default: easy).",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        choices=[0, 1],
        help="GPU id to expose via CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/replay_videos"),
        help="Directory used as video output root.",
    )
    return parser


def _run_episode(
    env_id: str,
    episode: int,
    seed: int,
    difficulty: str,
    output_dir: Path,
) -> bool:
    print(
        f"--- Running env={env_id} episode={episode} seed={seed} difficulty={difficulty} ---"
    )

    env: Optional[gym.Env] = None
    episode_successful = False

    try:
        env_kwargs = dict(
            obs_mode="rgb+depth+segmentation",
            control_mode="pd_joint_pos",
            render_mode="rgb_array",
            reward_mode="dense",
            seed=seed,
            difficulty=difficulty,
        )

        if episode <= 5:
            env_kwargs["robomme_failure_recovery"] = True
            if episode <= 2:
                env_kwargs["robomme_failure_recovery_mode"] = "z"
            else:
                env_kwargs["robomme_failure_recovery_mode"] = "xy"

        env = gym.make(env_id, **env_kwargs)
        env = RobommeRecordWrapper(
            env,
            dataset=str(output_dir),
            env_id=env_id,
            episode=episode,
            seed=seed,
            save_video=True,
            record_hdf5=False,
        )

        env.reset()

        if env_id in {"PatternLock", "RouteStick"}:
            planner = FailAwarePandaStickMotionPlanningSolver(
                env,
                debug=False,
                vis=False,
                base_pose=env.unwrapped.agent.robot.pose,
                visualize_target_grasp_pose=False,
                print_env_info=False,
                joint_vel_limits=0.3,
            )
        else:
            planner = FailAwarePandaArmMotionPlanningSolver(
                env,
                debug=False,
                vis=False,
                base_pose=env.unwrapped.agent.robot.pose,
                visualize_target_grasp_pose=False,
                print_env_info=False,
            )

        original_move_to_pose_with_screw = planner.move_to_pose_with_screw
        original_move_to_pose_with_rrt = planner.move_to_pose_with_RRTStar

        def _move_to_pose_with_screw_then_rrt_retry(*args, **kwargs):
            for attempt in range(1, DATASET_SCREW_MAX_ATTEMPTS + 1):
                try:
                    result = original_move_to_pose_with_screw(*args, **kwargs)
                except ScrewPlanFailure as exc:
                    print(
                        f"[Replay] screw planning failed "
                        f"(attempt {attempt}/{DATASET_SCREW_MAX_ATTEMPTS}): {exc}"
                    )
                    continue

                if isinstance(result, int) and result == -1:
                    print(
                        f"[Replay] screw planning returned -1 "
                        f"(attempt {attempt}/{DATASET_SCREW_MAX_ATTEMPTS})"
                    )
                    continue

                return result

            print(
                "[Replay] screw planning exhausted; "
                f"fallback to RRT* (max {DATASET_RRT_MAX_ATTEMPTS} attempts)"
            )

            for attempt in range(1, DATASET_RRT_MAX_ATTEMPTS + 1):
                try:
                    result = original_move_to_pose_with_rrt(*args, **kwargs)
                except Exception as exc:
                    print(
                        f"[Replay] RRT* planning failed "
                        f"(attempt {attempt}/{DATASET_RRT_MAX_ATTEMPTS}): {exc}"
                    )
                    continue

                if isinstance(result, int) and result == -1:
                    print(
                        f"[Replay] RRT* planning returned -1 "
                        f"(attempt {attempt}/{DATASET_RRT_MAX_ATTEMPTS})"
                    )
                    continue

                return result

            print("[Replay] screw->RRT* planning exhausted; return -1")
            return -1

        planner.move_to_pose_with_screw = _move_to_pose_with_screw_then_rrt_retry

        env.unwrapped.evaluate()
        tasks = list(getattr(env.unwrapped, "task_list", []) or [])
        print(f"{env_id}: Task list has {len(tasks)} tasks")

        for idx, task_entry in enumerate(tasks):
            task_name = task_entry.get("name", f"Task {idx}")
            print(f"Executing task {idx + 1}/{len(tasks)}: {task_name}")

            solve_callable = task_entry.get("solve")
            if not callable(solve_callable):
                raise ValueError(f"Task '{task_name}' must supply a callable 'solve'.")

            env.unwrapped.evaluate(solve_complete_eval=True)
            screw_failed = False
            try:
                solve_result = solve_callable(env, planner)
                if isinstance(solve_result, int) and solve_result == -1:
                    screw_failed = True
                    print(f"Screw->RRT* planning exhausted during '{task_name}'")
                    env.unwrapped.failureflag = torch.tensor([True])
                    env.unwrapped.successflag = torch.tensor([False])
                    env.unwrapped.current_task_failure = True
            except ScrewPlanFailure as exc:
                screw_failed = True
                print(f"Screw plan failure during '{task_name}': {exc}")
                env.unwrapped.failureflag = torch.tensor([True])
                env.unwrapped.successflag = torch.tensor([False])
                env.unwrapped.current_task_failure = True
            except FailsafeTimeout as exc:
                print(f"Failsafe: {exc}")
                break

            evaluation = env.unwrapped.evaluate(solve_complete_eval=True)
            fail_flag = evaluation.get("fail", False)
            success_flag = evaluation.get("success", False)

            if _tensor_to_bool(success_flag):
                print("All tasks completed successfully.")
                episode_successful = True
                break

            if screw_failed or _tensor_to_bool(fail_flag):
                print("Encountered failure condition; stopping task sequence.")
                break
        else:
            evaluation = env.unwrapped.evaluate(solve_complete_eval=True)
            episode_successful = _tensor_to_bool(evaluation.get("success", False))

        episode_successful = episode_successful or _tensor_to_bool(
            getattr(env, "episode_success", False)
        )
    except SceneGenerationError as exc:
        print(
            f"Scene generation failed for env {env_id}, episode {episode}, seed {seed}: {exc}"
        )
        episode_successful = False
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as close_exc:
                print(
                    f"Warning: Exception during env.close() for episode {episode}, "
                    f"seed {seed}: {close_exc}"
                )

    status_text = "SUCCESS" if episode_successful else "FAILED"
    print(
        f"--- Finished env={env_id} episode={episode} seed={seed} "
        f"difficulty={difficulty} [{status_text}] ---"
    )
    return episode_successful


def main() -> None:
    args = _build_parser().parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Environment: {args.env}")
    print(f"Episode: {args.episode}")
    print(f"Seed: {args.seed}")
    print(f"Difficulty: {args.difficulty}")
    print(f"GPU: {args.gpu}")
    print(f"Video output root: {output_dir}")

    success = _run_episode(
        env_id=args.env,
        episode=args.episode,
        seed=args.seed,
        difficulty=args.difficulty,
        output_dir=output_dir,
    )

    mp4_path = _latest_recorded_mp4(output_dir, args.env, args.episode, args.seed)
    if mp4_path is not None:
        print(f"Final MP4: {mp4_path.resolve()}")
    else:
        print(
            f"No MP4 matched under {output_dir / 'videos'} "
            f"(expected filename fragment '{args.env}_ep{args.episode}_seed{args.seed}')."
        )

    if success:
        print("Replay finished successfully.")
    else:
        print("Replay finished with failure status.")


if __name__ == "__main__":
    main()
