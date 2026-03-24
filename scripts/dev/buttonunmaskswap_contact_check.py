# /// script
# dependencies = [
#   "robomme",
# ]
# [tool.uv.sources]
# robomme = { path = "../..", editable = true }
# ///
"""Batch-run ButtonUnmaskSwap and record swap-window bin contact summaries to JSONL."""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing.context import BaseContext
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch

from robomme.env_record_wrapper import FailsafeTimeout, RobommeRecordWrapper
from robomme.robomme_env import *
from robomme.robomme_env.utils.SceneGenerationError import SceneGenerationError
from robomme.robomme_env.utils.planner_fail_safe import (
    FailAwarePandaArmMotionPlanningSolver,
    FailAwarePandaStickMotionPlanningSolver,
    ScrewPlanFailure,
)

ENV_ID = "ButtonUnmaskSwap"
VALID_DIFFICULTIES = {"easy", "medium", "hard"}
DATASET_SCREW_MAX_ATTEMPTS = 3
DATASET_RRT_MAX_ATTEMPTS = 3

RunJob = Tuple[int, int, str, str, str]


def _latest_recorded_mp4(output_root: Path, episode: int, seed: int) -> Optional[Path]:
    videos_dir = output_root / "videos"
    if not videos_dir.is_dir():
        return None
    tag = f"{ENV_ID}_ep{episode}_seed{seed}"
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run ButtonUnmaskSwap episode(s), detect swap-window bin contacts, "
            "and append one JSONL record per episode."
        )
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
        default=5,
        help="Max parallel worker processes (default: 32).",
    )
    parser.add_argument(
        "--total-episodes",
        type=int,
        default=5,
        help="Number of episodes to run (default: 64).",
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
        default=1,
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
        help="Comma-separated GPU ids for round-robin across jobs (e.g. 0,1).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/replay_videos"),
        help="Directory used as video output root.",
    )
    parser.add_argument(
        "--jsonl-path",
        type=Path,
        default=None,
        help=(
            "JSONL output path. Defaults to "
            "OUTPUT_DIR/buttonunmaskswap_contact_results.jsonl."
        ),
    )
    return parser


def _append_jsonl_record(jsonl_path: Path, record: Dict) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _build_episode_record(
    episode: int,
    seed: int,
    difficulty: str,
    episode_success: bool,
    contact_summary: Dict,
    video_path: Optional[Path],
) -> Dict:
    return {
        "env": ENV_ID,
        "episode": int(episode),
        "seed": int(seed),
        "difficulty": difficulty,
        "episode_success": bool(episode_success),
        "swap_contact_detected": bool(contact_summary.get("swap_contact_detected", False)),
        "first_contact_step": contact_summary.get("first_contact_step"),
        "contact_pairs": list(contact_summary.get("contact_pairs", [])),
        "max_force_norm": float(contact_summary.get("max_force_norm", 0.0)),
        "max_force_pair": contact_summary.get("max_force_pair"),
        "max_force_step": contact_summary.get("max_force_step"),
        "pair_max_force": {
            str(pair_name): float(force_norm)
            for pair_name, force_norm in contact_summary.get("pair_max_force", {}).items()
        },
        "video_path": str(video_path.resolve()) if video_path is not None else None,
    }


def _run_episode(
    episode: int,
    seed: int,
    difficulty: str,
    output_dir: Path,
) -> Dict:
    print(
        f"--- Running env={ENV_ID} episode={episode} seed={seed} difficulty={difficulty} ---"
    )

    env: Optional[gym.Env] = None
    episode_successful = False
    contact_summary = {
        "swap_contact_detected": False,
        "first_contact_step": None,
        "contact_pairs": [],
        "max_force_norm": 0.0,
        "max_force_pair": None,
        "max_force_step": None,
        "pair_max_force": {},
    }

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

        env = gym.make(ENV_ID, **env_kwargs)
        env = RobommeRecordWrapper(
            env,
            dataset=str(output_dir),
            env_id=ENV_ID,
            episode=episode,
            seed=seed,
            save_video=True,
            record_hdf5=False,
        )
        env.unwrapped.swap_contact_log_context = {
            "env": ENV_ID,
            "episode": episode,
            "seed": seed,
        }
        env.reset()

        if ENV_ID in {"PatternLock", "RouteStick"}:
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
        print(f"{ENV_ID}: Task list has {len(tasks)} tasks")

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
        contact_summary = env.unwrapped.get_swap_contact_summary()
    except SceneGenerationError as exc:
        print(
            f"Scene generation failed for env {ENV_ID}, episode {episode}, seed {seed}: {exc}"
        )
        episode_successful = False
    finally:
        if env is not None:
            try:
                if hasattr(env.unwrapped, "get_swap_contact_summary"):
                    contact_summary = env.unwrapped.get_swap_contact_summary()
            except Exception as summary_exc:
                print(
                    f"Warning: Exception while collecting swap contact summary for "
                    f"episode {episode}, seed {seed}: {summary_exc}"
                )
            try:
                env.close()
            except Exception as close_exc:
                print(
                    f"Warning: Exception during env.close() for episode {episode}, "
                    f"seed {seed}: {close_exc}"
                )

    video_path = _latest_recorded_mp4(output_dir, episode, seed)
    record = _build_episode_record(
        episode=episode,
        seed=seed,
        difficulty=difficulty,
        episode_success=episode_successful,
        contact_summary=contact_summary,
        video_path=video_path,
    )

    status_text = "SUCCESS" if episode_successful else "FAILED"
    print(
        f"--- Finished env={ENV_ID} episode={episode} seed={seed} "
        f"difficulty={difficulty} [{status_text}] "
        f"swap_contact_detected={record['swap_contact_detected']} ---"
    )
    return record


def _run_episode_worker(job: RunJob) -> Dict:
    episode, seed, difficulty, output_dir_str, cuda_visible = job
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible
    output_dir = Path(output_dir_str).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return _run_episode(episode, seed, difficulty, output_dir)


def _spawn_context() -> BaseContext:
    import multiprocessing as mp

    return mp.get_context("spawn")


def main() -> None:
    args = _build_parser().parse_args()
    gpu_list = _parse_gpu_list(args.gpus, args.gpu)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = (
        args.jsonl_path.resolve()
        if args.jsonl_path is not None
        else (output_dir / "buttonunmaskswap_contact_results.jsonl").resolve()
    )

    print(f"Environment: {ENV_ID}")
    print(f"Difficulty: {args.difficulty}")
    print(f"Video output root: {output_dir}")
    print(f"JSONL output: {jsonl_path}")

    if args.total_episodes < 1:
        raise SystemExit("--total-episodes must be >= 1")
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")

    if args.workers > 1 and len(gpu_list) == 1:
        print(
            "Warning: multiple workers on a single GPU (--gpu) may cause VRAM pressure "
            "or slowdown; consider --gpus 0,1,... to spread jobs across devices."
        )

    jobs: List[RunJob] = []
    for i in range(args.total_episodes):
        episode = args.episode_start + i
        seed = args.seed_start + i
        cuda_vis = gpu_list[i % len(gpu_list)]
        jobs.append((episode, seed, args.difficulty, str(output_dir), cuda_vis))

    print(
        f"Batch: {args.total_episodes} episodes, episode {args.episode_start}.."
        f"{args.episode_start + args.total_episodes - 1}, "
        f"seed {args.seed_start}..{args.seed_start + args.total_episodes - 1}, "
        f"{args.workers} workers, GPUs (round-robin): {','.join(gpu_list)}"
    )

    results: List[Dict] = []
    if args.workers == 1:
        for job in jobs:
            record = _run_episode_worker(job)
            _append_jsonl_record(jsonl_path, record)
            results.append(record)
    else:
        ctx = _spawn_context()
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as executor:
            future_map = {
                executor.submit(_run_episode_worker, job): job for job in jobs
            }
            for future in as_completed(future_map):
                record = future.result()
                _append_jsonl_record(jsonl_path, record)
                results.append(record)

    results.sort(key=lambda record: (record["episode"], record["seed"]))
    ok_count = sum(1 for record in results if record["episode_success"])
    collision_count = sum(1 for record in results if record["swap_contact_detected"])

    print(
        f"Batch finished: {ok_count}/{len(results)} succeeded, "
        f"{collision_count}/{len(results)} swap-contact episodes detected."
    )


if __name__ == "__main__":
    main()
