import argparse
import json
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


def _to_python_int(value) -> int:
    if value is None:
        return 0
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return 0
        value = value.detach().cpu().reshape(-1)[0].item()
    elif isinstance(value, np.ndarray):
        if value.size == 0:
            return 0
        value = np.asarray(value).reshape(-1)[0].item()
    return int(value)


def _actor_position_xyz(actor) -> list[float]:
    if actor is None:
        return [0.0, 0.0, 0.0]

    pose = actor.pose if hasattr(actor, "pose") else actor.get_pose()
    position = pose.p
    if isinstance(position, torch.Tensor):
        position = position.detach().cpu().numpy()

    position_np = np.asarray(position, dtype=np.float64).reshape(-1)
    if position_np.size < 3:
        padded = np.zeros(3, dtype=np.float64)
        padded[: position_np.size] = position_np
        position_np = padded
    return [float(position_np[0]), float(position_np[1]), float(position_np[2])]


def _snapshot_json_path(output_root: Path, env_id: str, episode: int, seed: int) -> Path:
    return (
        output_root
        / "snapshots"
        / f"{env_id}_ep{episode}_seed{seed}_after_drop.json"
    )


def _button_unmask_swap_inspect_this_timestep() -> int:
    """Timestep at which to capture the after-drop snapshot for ButtonUnmaskSwap."""
    return 33



def _collect_button_unmask_swap_snapshot(
    base_env,
    env_id: str,
    episode: int,
    seed: int,
    difficulty: str,
    capture_elapsed_steps: int,
) -> dict:
    spawned_bins = list(getattr(base_env, "spawned_bins", []) or [])
    cube_bin_pairs = list(getattr(base_env, "cube_bin_pairs", []) or [])
    color_names = list(getattr(base_env, "color_names", []) or [])
    bin_to_color = dict(getattr(base_env, "bin_to_color", {}) or {})
    inspect_this_timestep = _button_unmask_swap_inspect_this_timestep()
    bin_index_by_id = {id(bin_actor): idx for idx, bin_actor in enumerate(spawned_bins)}
    bins_with_cubes = set()
    cubes = []

    for pair_idx, pair in enumerate(cube_bin_pairs):
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            continue

        cube_actor, bin_actor = pair
        paired_bin_index = bin_index_by_id.get(id(bin_actor))
        if paired_bin_index is not None:
            bins_with_cubes.add(paired_bin_index)

        cube_color = None
        if paired_bin_index is not None:
            cube_color = bin_to_color.get(paired_bin_index)
        if cube_color is None and pair_idx < len(color_names):
            cube_color = color_names[pair_idx]

        cubes.append(
            {
                "name": getattr(cube_actor, "name", None),
                "color": cube_color,
                "position_xyz": _actor_position_xyz(cube_actor),
                "paired_bin_index": (
                    int(paired_bin_index) if paired_bin_index is not None else None
                ),
                "paired_bin_name": getattr(bin_actor, "name", None),
            }
        )

    bins = []
    for idx, bin_actor in enumerate(spawned_bins):
        bins.append(
            {
                "index": idx,
                "name": getattr(bin_actor, "name", None),
                "position_xyz": _actor_position_xyz(bin_actor),
                "has_cube_under_bin": idx in bins_with_cubes,
            }
        )

    return {
        "env_id": env_id,
        "episode": int(episode),
        "seed": int(seed),
        "difficulty": difficulty,
        "inspect_this_timestep": inspect_this_timestep,
        "capture_elapsed_steps": int(capture_elapsed_steps),
        "capture_phase": "after_drop",
        "cubes": cubes,
        "bins": bins,
    }


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


def _build_env_kwargs(episode: int, seed: int, difficulty: str) -> dict:
    """Construct gymnasium env kwargs, including episode-based failure recovery."""
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
    return env_kwargs


def _create_env(
    env_id: str, env_kwargs: dict, output_dir: Path, episode: int, seed: int
) -> gym.Env:
    """Create the gymnasium env and wrap it with RobommeRecordWrapper."""
    env = gym.make(env_id, **env_kwargs)
    return RobommeRecordWrapper(
        env,
        dataset=str(output_dir),
        env_id=env_id,
        episode=episode,
        seed=seed,
        save_video=True,
    )


def _install_snapshot_instrumentation(
    env: gym.Env,
    env_id: str,
    episode: int,
    seed: int,
    difficulty: str,
    output_dir: Path,
) -> dict:
    """Monkey-patch env.step to capture after-drop snapshot for ButtonUnmaskSwap.

    Returns a mutable state dict with keys ``snapshot_written`` and
    ``snapshot_json_path`` that the instrumented closure updates in place.
    For non-ButtonUnmaskSwap envs this is a no-op.
    """
    state: dict = {"snapshot_written": False, "snapshot_json_path": None}
    if env_id != "ButtonUnmaskSwap":
        return state

    original_step = env.step

    def instrumented_step(action):
        step_result = original_step(action)
        if state["snapshot_written"]:
            return step_result

        base_env = env.unwrapped
        elapsed_steps = _to_python_int(getattr(base_env, "elapsed_steps", 0))
        if elapsed_steps < _button_unmask_swap_inspect_this_timestep():
            return step_result

        snapshot_payload = _collect_button_unmask_swap_snapshot(
            base_env=base_env,
            env_id=env_id,
            episode=episode,
            seed=seed,
            difficulty=difficulty,
            capture_elapsed_steps=elapsed_steps,
        )
        path = _snapshot_json_path(
            output_root=output_dir, env_id=env_id, episode=episode, seed=seed
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(snapshot_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        state["snapshot_written"] = True
        state["snapshot_json_path"] = path
        print(f"After-drop snapshot JSON: {path.resolve()}")
        return step_result

    env.step = instrumented_step  # type: ignore[method-assign]
    return state


def _create_planner(env: gym.Env, env_id: str):
    """Create the appropriate motion planner (stick vs arm) based on env_id."""
    if env_id in {"PatternLock", "RouteStick"}:
        return FailAwarePandaStickMotionPlanningSolver(
            env,
            debug=False,
            vis=False,
            base_pose=env.unwrapped.agent.robot.pose,
            visualize_target_grasp_pose=False,
            print_env_info=False,
            joint_vel_limits=0.3,
        )
    return FailAwarePandaArmMotionPlanningSolver(
        env,
        debug=False,
        vis=False,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=False,
        print_env_info=False,
    )


def _wrap_planner_with_screw_then_rrt_retry(planner) -> None:
    """Monkey-patch planner.move_to_pose_with_screw with retry + RRT* fallback."""
    original_screw = planner.move_to_pose_with_screw
    original_rrt = planner.move_to_pose_with_RRTStar

    def _retry(*args, **kwargs):
        for attempt in range(1, DATASET_SCREW_MAX_ATTEMPTS + 1):
            try:
                result = original_screw(*args, **kwargs)
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
                result = original_rrt(*args, **kwargs)
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

    planner.move_to_pose_with_screw = _retry


def _execute_task_list(env: gym.Env, planner, env_id: str) -> bool:
    """Run the evaluate-solve-check loop over all tasks. Returns True on success."""
    env.unwrapped.evaluate()
    tasks = list(getattr(env.unwrapped, "task_list", []) or [])
    print(f"{env_id}: Task list has {len(tasks)} tasks")

    episode_successful = False

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

    return episode_successful or _tensor_to_bool(
        getattr(env, "episode_success", False)
    )


def _close_env(env: Optional[gym.Env], episode: int, seed: int) -> None:
    """Safely close the environment, logging any exception."""
    if env is None:
        return
    try:
        env.close()
    except Exception as close_exc:
        print(
            f"Warning: Exception during env.close() for episode {episode}, "
            f"seed {seed}: {close_exc}"
        )


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
    snapshot_state: dict = {"snapshot_written": False, "snapshot_json_path": None}

    try:
        env_kwargs = _build_env_kwargs(episode, seed, difficulty)
        env = _create_env(env_id, env_kwargs, output_dir, episode, seed)
        snapshot_state = _install_snapshot_instrumentation(
            env, env_id, episode, seed, difficulty, output_dir
        )
        env.reset()
        planner = _create_planner(env, env_id)
        _wrap_planner_with_screw_then_rrt_retry(planner)
        episode_successful = _execute_task_list(env, planner, env_id)
    except SceneGenerationError as exc:
        print(
            f"Scene generation failed for env {env_id}, episode {episode}, seed {seed}: {exc}"
        )
        episode_successful = False
    finally:
        _close_env(env, episode, seed)

    status_text = "SUCCESS" if episode_successful else "FAILED"
    print(
        f"--- Finished env={env_id} episode={episode} seed={seed} "
        f"difficulty={difficulty} [{status_text}] ---"
    )
    if env_id == "ButtonUnmaskSwap" and not snapshot_state["snapshot_written"]:
        print(
            "Warning: after-drop snapshot JSON was not captured before the episode ended."
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
