# /// script
# dependencies = [
#   "robomme",
# ]
# [tool.uv.sources]
# robomme = { path = "../..", editable = true }
# ///
"""Batch-record Robomme episodes with optional parallel processes.

Defaults: 10 workers, 20 episodes, episode indices 1..20, seeds 0..19, difficulty hard.
Parallel runs share one output directory; video filenames include env/episode/seed.

Run (from repository root, recommended)::

    uv sync
    uv run python scripts/dev/multi-ep-recordwrapper.py

Multiple environments run **in order** (finish all jobs for env A, then B). Example::

    uv run python scripts/dev/multi-ep-recordwrapper.py -e PickXtimes,StopCube --total-episodes 1 --episode-start 1 --seed-start 0

Or let uv resolve the editable package via this file's inline metadata::

    uv run scripts/dev/multi-ep-recordwrapper.py

GPU: many workers on a single GPU can exhaust VRAM or run slowly. Pass ``--gpus 0,1,...``
to round-robin jobs across devices (job ``i`` uses ``gpus[i % len(gpus)]``).
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing.context import BaseContext
from pathlib import Path
from typing import List, Optional, Set, Tuple

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

# Job tuple: env_id, episode, seed, difficulty, output_dir_str, cuda_visible_devices
RunJob = Tuple[str, int, int, str, str, str]


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


def _parse_gpu_list(gpus_arg: Optional[str], gpu_fallback: int) -> List[str]:
    if gpus_arg is not None and gpus_arg.strip():
        parts = [p.strip() for p in gpus_arg.split(",") if p.strip()]
        if not parts:
            raise ValueError("--gpus must list at least one GPU id.")
        return parts
    return [str(gpu_fallback)]


def _parse_env_list(raw: str) -> List[str]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise SystemExit("--env must list at least one environment ID.")
    invalid = [p for p in parts if p not in VALID_ENVS]
    if invalid:
        raise SystemExit(
            f"Invalid environment ID(s): {invalid}. "
            f"Valid options: {', '.join(sorted(VALID_ENVS))}"
        )
    return parts


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Robomme episode(s) and record video using a process pool (--workers). "
            "Episode indices and seeds come from --episode-start and --seed-start."
        )
    )
    parser.add_argument(
        "--env",
        "-e",
        default="ButtonUnmaskSwap,VideoUnmaskSwap,VideoRepick",
        type=str,
        help=(
            "Environment ID(s), comma-separated for multiple. "
            "Envs run in order (each env fully before the next). "
            "Default: ButtonUnmaskSwap."
        ),
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        default="hard",
        choices=sorted(VALID_DIFFICULTIES),
        help="Episode difficulty (default: hard).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=32,
        help="Max parallel worker processes (default: 20).",
    )
    parser.add_argument(
        "--total-episodes",
        type=int,
        default=64,
        help="Number of episodes to run per environment (default: 40). Uses --episode-start and --seed-start.",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=0,
        help="Seed for the first run; run i uses seed-start + i (default: 0).",
    )
    parser.add_argument(
        "--episode-start",
        type=int,
        default=0,
        help="Episode index for the first run; run i uses episode-start + i (default: 1).",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        choices=[0, 1],
        help="GPU id when --gpus is not set (default: 0).",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help=(
            "Comma-separated GPU ids for round-robin across jobs (e.g. 0,1). "
            "If omitted, every job uses --gpu."
        ),
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


def _run_episode_worker(job: RunJob) -> Tuple[int, int, bool]:
    """Subprocess entry: set CUDA device, ensure output dir, run one episode."""
    env_id, episode, seed, difficulty, output_dir_str, cuda_visible = job
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible
    output_dir = Path(output_dir_str).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ok = _run_episode(env_id, episode, seed, difficulty, output_dir)
    return episode, seed, ok


def _spawn_context() -> BaseContext:
    import multiprocessing as mp

    return mp.get_context("spawn")


def main() -> None:
    args = _build_parser().parse_args()
    env_ids = _parse_env_list(args.env)
    gpu_list = _parse_gpu_list(args.gpus, args.gpu)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Environment(s) ({len(env_ids)}): {', '.join(env_ids)}")
    print(f"Difficulty: {args.difficulty}")
    print(f"Video output root: {output_dir}")

    if args.total_episodes < 1:
        raise SystemExit("--total-episodes must be >= 1")
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")

    if args.workers > 1 and len(gpu_list) == 1:
        print(
            "Warning: multiple workers on a single GPU (--gpu) may cause VRAM pressure "
            "or slowdown; consider --gpus 0,1,... to spread jobs across devices."
        )

    out_str = str(output_dir)
    grand_ok = 0
    grand_total = 0

    for env_idx, env_id in enumerate(env_ids):
        print(
            f"\n=== Environment {env_idx + 1}/{len(env_ids)}: {env_id} ===\n"
        )

        jobs: List[RunJob] = []
        for i in range(args.total_episodes):
            episode = args.episode_start + i
            seed = args.seed_start + i
            cuda_vis = gpu_list[i % len(gpu_list)]
            jobs.append(
                (env_id, episode, seed, args.difficulty, out_str, cuda_vis)
            )

        print(
            f"Batch: {args.total_episodes} episodes, episode {args.episode_start}.."
            f"{args.episode_start + args.total_episodes - 1}, "
            f"seed {args.seed_start}..{args.seed_start + args.total_episodes - 1}, "
            f"{args.workers} workers, GPUs (round-robin): {','.join(gpu_list)}"
        )

        ctx = _spawn_context()
        results: List[Tuple[int, int, bool]] = []
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as executor:
            future_map = {
                executor.submit(_run_episode_worker, job): job for job in jobs
            }
            for fut in as_completed(future_map):
                results.append(fut.result())

        results.sort(key=lambda t: (t[0], t[1]))
        ok_count = sum(1 for _, _, ok in results if ok)
        failed = [(ep, sd) for ep, sd, ok in results if not ok]
        grand_ok += ok_count
        grand_total += len(results)

        print(
            f"Batch finished [{env_id}]: {ok_count}/{len(results)} succeeded, "
            f"{len(results) - ok_count} failed."
        )
        if failed:
            print(f"Failed [{env_id}] (episode, seed): {failed}")

    if len(env_ids) > 1:
        print(
            f"\nAll batches finished: {grand_ok}/{grand_total} episode runs succeeded "
            f"across {len(env_ids)} environment(s)."
        )


if __name__ == "__main__":
    main()
