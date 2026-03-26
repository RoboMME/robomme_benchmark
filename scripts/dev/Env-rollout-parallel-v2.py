"""并行回放 Robomme env 并可抓取 unmask after-drop 快照。

这个脚本的职责有三件事：
1. 按给定环境、关卡编号和随机种子创建 Robomme 环境，并在同一 env_id 内并行执行多局任务。
2. 使用 `RobommeRecordWrapper` 录制回放视频，方便后续人工检查。
3. 对支持的 unmask env，在固定时间步额外抓取一次“drop 之后”的场景快照，
   把箱子、方块及其对应关系导出为 JSON，便于离线比对状态是否正确。

整体执行链路是：
参数解析 -> 创建环境 -> 给 `env.step` 打补丁以便抓快照 -> 创建规划器并加重试逻辑
-> 逐任务调用 solve -> 关闭环境 -> 打印最终视频路径。

seed 规则采用 legacy 布局：
base_seed = 1_000_000 + env_code * 10000 + episode * 100
seed = base_seed + attempt

Episode 级失败时各产物（由 `RobommeRecordWrapper.close()` 与 `snapshot.install_snapshot_for_step` 决定）：
- MP4：`save_video=True` 时失败分支仍会尽量落盘失败回放视频（wrapper 内 `success=False`）。
- HDF5：仅在 `RobommeRecordWrapper.episode_success` 为真时写入轨迹；失败时不写 buffer，并删除本 episode
  对应 HDF5 group；`.h5` 文件可能仍存在但通常不含有效 episode 数据。若失败原因是 bin collision，
  同样会在 close 前被强制压成失败，从而进入 `seed+1` 重试。
- after-drop JSON：在达到 `scripts/dev/snapshot.py` 中配置的抓取步（默认第 33 步）时即写入，与后续任务是否
  成功无关；若在该步之前 rollout 已结束，则可能不产生 JSON（`_run_episode` 会打印相应 warning）。
"""

import argparse
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Set

import gymnasium as gym
import numpy as np
import snapshot as snapshot_utils
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
ENV_ID_TO_CODE = {name: idx + 1 for idx, name in enumerate(DEFAULT_ENVS)}
SEED_OFFSET = 1_500_000
VALID_ENVS: Set[str] = set(DEFAULT_ENVS)
VALID_DIFFICULTIES: Set[str] = {"easy", "medium", "hard"}
DATASET_SCREW_MAX_ATTEMPTS = 3
DATASET_RRT_MAX_ATTEMPTS = 3
MAX_SEED_ATTEMPTS = 100


def _latest_recorded_mp4(
    output_root: Path, env_id: str, episode: int, seed: int
) -> Optional[Path]:
    """返回当前回放对应的最新 mp4 文件。

    `RobommeRecordWrapper` 在落盘时可能会产出多个 mp4，同一组参数也可能因为重复运行
    留下历史文件。这里按修改时间选择最新的那一个，避免拿到旧视频。
    """
    videos_dir = output_root / "videos"
    if not videos_dir.is_dir():
        return None
    tag = f"{env_id}_ep{episode}_seed{seed}"
    candidates = [p for p in videos_dir.glob("*.mp4") if tag in p.name]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _tensor_to_bool(value) -> bool:
    """把 Tensor / ndarray / Python 标量统一转换成布尔值。"""
    if value is None:
        return False
    if isinstance(value, torch.Tensor):
        return bool(value.detach().cpu().bool().item())
    if isinstance(value, np.ndarray):
        return bool(np.any(value))
    return bool(value)


def _base_seed_for_episode(env_id: str, episode: int) -> int:
    """按 legacy 规则计算某个 env/episode 的基础 seed。"""
    if env_id not in ENV_ID_TO_CODE:
        raise ValueError(f"Environment {env_id} missing from ENV_ID_TO_CODE mapping")
    env_code = ENV_ID_TO_CODE[env_id]
    return SEED_OFFSET + env_code * 10000 + episode * 100


def _build_parser() -> argparse.ArgumentParser:
    """定义命令行参数。"""
    parser = argparse.ArgumentParser(
        description=(
            "Run one or more Robomme episodes in parallel and record video. "
            "Seeds use the legacy env/episode/attempt layout."
        )
    )
    parser.add_argument(
        "--env",
        "-e",
        nargs="+",
        default=["VideoUnmaskSwap",],
        choices=sorted(VALID_ENVS),
        metavar="ENV",
        help=(
            "One or more environment IDs to run in order (default: ButtonUnmaskSwap). "
            "Each env runs the same episode range; seeds are derived from "
            "env_id/episode/attempt."
        ),
    )
    parser.add_argument(
        "--episode-number",
        type=int,
        default=20,
        metavar="N",
        help=(
            "How many consecutive episodes to run starting from index 0: "
            "episodes 0 .. N-1 (e.g. N=5 runs episodes 0,1,2,3,4). Default: 5."
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
    parser.add_argument(
        "--max-workers",
        type=int,
        default=20,
        help=(
            "Maximum number of worker processes used to parallelize episodes within "
            "the same env_id. Default: auto=min(os.cpu_count(), episode count)."
        ),
    )
    return parser


def _build_env_kwargs(episode: int, seed: int, difficulty: str) -> dict:
    """构造环境参数，并按 episode 编号启用不同级别的失败恢复。"""
    env_kwargs = dict(
        obs_mode="rgb+depth+segmentation",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        reward_mode="dense",
        seed=seed,
        difficulty=difficulty,
    )
    if episode <= 5:
        # 早期 episode 更容易触发规划失败，这里开启恢复逻辑。
        # episode 1-2 只做 z 方向恢复，3-5 再放宽到 xy 平面恢复。
        env_kwargs["robomme_failure_recovery"] = True
        if episode <= 2:
            env_kwargs["robomme_failure_recovery_mode"] = "z"
        else:
            env_kwargs["robomme_failure_recovery_mode"] = "xy"
    return env_kwargs


def _create_env(
    env_id: str, env_kwargs: dict, output_dir: Path, episode: int, seed: int
) -> gym.Env:
    """创建 gym 环境，并包上录屏 wrapper。"""
    env = gym.make(env_id, **env_kwargs)
    return RobommeRecordWrapper(
        env,
        dataset=str(output_dir),
        env_id=env_id,
        episode=episode,
        seed=seed,
        save_video=True,
    )


def _create_planner(env: gym.Env, env_id: str):
    """按任务类型选择 stick 规划器或机械臂规划器。"""
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
    """给 screw 规划加“多次重试 + RRT* 兜底”逻辑。"""
    original_screw = planner.move_to_pose_with_screw
    original_rrt = planner.move_to_pose_with_RRTStar

    def _retry(*args, **kwargs):
        # 第一层优先走 screw planner，因为它通常更快，也更符合原始策略。
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

        # screw planner 连续失败后，再切到 RRT* 做较重但更鲁棒的兜底搜索。
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

        # 两层规划都失败时返回 -1，让上层任务循环统一按失败分支处理。
        print("[Replay] screw->RRT* planning exhausted; return -1")
        return -1

    planner.move_to_pose_with_screw = _retry


def _execute_task_list(env: gym.Env, planner, env_id: str) -> bool:
    """按任务列表执行 `evaluate -> solve -> evaluate` 主循环。

    返回值表示整局 episode 是否成功。这个函数会同时关注：
    - solve 过程是否显式返回 `-1`。
    - 规划过程中是否抛出 `ScrewPlanFailure` / `FailsafeTimeout`。
    - 环境 `evaluate()` 给出的 success / fail 标志。
    """
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

        # 每个 task 开始前都重新做一次完整评估，让环境内部状态保持同步。
        env.unwrapped.evaluate(solve_complete_eval=True)
        screw_failed = False
        try:
            solve_result = solve_callable(env, planner)
            if isinstance(solve_result, int) and solve_result == -1:
                screw_failed = True
                print(f"Screw->RRT* planning exhausted during '{task_name}'")
                # 手动补齐环境里的失败标志，确保后面的 `evaluate()` 能观察到失败状态。
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
            # failsafe 触发通常意味着继续执行已经没有意义，直接终止当前 episode。
            print(f"Failsafe: {exc}")
            break

        evaluation = env.unwrapped.evaluate(solve_complete_eval=True)
        fail_flag = evaluation.get("fail", False)
        success_flag = evaluation.get("success", False)

        if _tensor_to_bool(success_flag):
            # 某些环境可能在中途 task 就宣告整局成功，不必继续跑剩余任务。
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
    """安全关闭环境，避免清理阶段的异常吞掉真正的执行结果。"""
    if env is None:
        return
    try:
        env.close()
    except Exception as close_exc:
        print(
            f"Warning: Exception during env.close() for episode {episode}, "
            f"seed {seed}: {close_exc}"
        )


def _mark_episode_failed(env: Optional[gym.Env], reason: str) -> None:
    """在 close 前把 wrapper 和底层 env 的状态统一压成失败。"""
    if env is None:
        return

    if hasattr(env, "episode_success"):
        env.episode_success = False

    base_env = getattr(env, "unwrapped", None)
    if base_env is None:
        return

    base_env.failureflag = torch.tensor([True])
    base_env.successflag = torch.tensor([False])
    base_env.current_task_failure = True
    print(f"[Replay] Episode failure forced before close: {reason}")


def _run_episode(
    env_id: str,
    episode: int,
    seed: int,
    difficulty: str,
    output_dir: Path,
) -> bool:
    """执行单个 episode，并返回是否成功。"""
    print(
        f"--- Running env={env_id} episode={episode} seed={seed} difficulty={difficulty} ---"
    )

    env: Optional[gym.Env] = None
    episode_successful = False
    snapshot_state: dict = {
        "snapshot_enabled": False,
        "snapshot_written": False,
        "snapshot_json_path": None,
    }

    try:
        env_kwargs = _build_env_kwargs(episode, seed, difficulty)
        env = _create_env(env_id, env_kwargs, output_dir, episode, seed)
        # 先装快照补丁，再 reset；这样从 episode 的第一个 step 开始就受监控。
        snapshot_state = snapshot_utils.install_snapshot_for_step(
            env, env_id, episode, seed, difficulty, output_dir
        )
        env.reset()
        planner = _create_planner(env, env_id)
        _wrap_planner_with_screw_then_rrt_retry(planner)
        episode_successful = _execute_task_list(env, planner, env_id)
        if snapshot_state.get("collision_detected"):
            print(
                f"[Replay] env={env_id} episode={episode} seed={seed} "
                "forcing episode failure because bin collision was detected."
            )
            _mark_episode_failed(
                env,
                reason=(
                    f"env={env_id} episode={episode} seed={seed} "
                    "reason=bin_collision"
                ),
            )
            episode_successful = False
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
    if snapshot_state.get("snapshot_enabled") and not snapshot_state["snapshot_written"]:
        # 快照缺失本身不一定说明 episode 失败，但通常意味着流程比预期更早结束了。
        print(
            "Warning: after-drop snapshot JSON was not captured before the episode ended."
        )
    return episode_successful


def _run_episode_with_retry(
    env_id: str,
    episode: int,
    difficulty: str,
    output_dir: Path,
) -> tuple[bool, int]:
    """按 legacy seed 规则执行单个 episode，并在失败时最多换 seed 重试 100 次。"""
    base_seed = _base_seed_for_episode(env_id, episode)
    print(
        f"[Retry] env={env_id} episode={episode} "
        f"base_seed={base_seed} max_attempts={MAX_SEED_ATTEMPTS}"
    )

    last_seed = base_seed
    for attempt in range(MAX_SEED_ATTEMPTS):
        seed = base_seed + attempt
        last_seed = seed
        print(
            f"[Retry] env={env_id} episode={episode} "
            f"attempt={attempt + 1}/{MAX_SEED_ATTEMPTS} seed={seed}"
        )
        try:
            success = _run_episode(
                env_id=env_id,
                episode=episode,
                seed=seed,
                difficulty=difficulty,
                output_dir=output_dir,
            )
        except Exception as exc:
            print(
                f"[Retry] env={env_id} episode={episode} seed={seed} "
                f"raised {type(exc).__name__}: {exc}"
            )
            success = False

        if success:
            print(
                f"[Retry] env={env_id} episode={episode} "
                f"succeeded with seed={seed} on attempt {attempt + 1}/{MAX_SEED_ATTEMPTS}"
            )
            return True, seed

    print(
        f"[Retry] env={env_id} episode={episode} exhausted "
        f"{MAX_SEED_ATTEMPTS} attempts; last_seed={last_seed}"
    )
    return False, last_seed


def _resolve_max_workers(requested: Optional[int], n_episodes: int) -> int:
    """计算实际启用的 worker 数量。"""
    if requested is not None and requested < 1:
        raise SystemExit("--max-workers must be at least 1.")
    cpu_count = os.cpu_count() or 1
    max_allowed = min(cpu_count, n_episodes)
    if requested is None:
        return max_allowed
    return min(requested, max_allowed)


def _print_episode_artifacts(
    output_dir: Path, env_id: str, episode: int, run_seed: int
) -> None:
    """打印当前 episode 对应的视频产物路径。"""
    mp4_path = _latest_recorded_mp4(output_dir, env_id, episode, run_seed)
    if mp4_path is not None:
        print(
            f"Final MP4 ({env_id} episode {episode}, seed={run_seed}): "
            f"{mp4_path.resolve()}"
        )
    else:
        print(
            f"No MP4 matched under {output_dir / 'videos'} "
            f"(expected filename fragment '{env_id}_ep{episode}_seed{run_seed}')."
        )


def _run_env_episodes(
    env_id: str,
    episode_numbers: list[int],
    difficulty: str,
    output_dir: Path,
    max_workers: int,
) -> list[bool]:
    """在同一个 env_id 内并行执行多个 episode。"""
    env_successes: list[tuple[int, bool]] = []

    if max_workers == 1:
        for ep in episode_numbers:
            success, used_seed = _run_episode_with_retry(
                env_id=env_id,
                episode=ep,
                difficulty=difficulty,
                output_dir=output_dir,
            )
            env_successes.append((ep, success))
            _print_episode_artifacts(output_dir, env_id, ep, used_seed)
        return [success for _, success in sorted(env_successes)]

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        futures = {
            executor.submit(
                _run_episode_with_retry,
                env_id,
                ep,
                difficulty,
                output_dir,
            ): ep
            for ep in episode_numbers
        }

        for future in as_completed(futures):
            ep = futures[future]
            try:
                success, used_seed = future.result()
            except Exception as exc:
                raise RuntimeError(
                    f"Worker crashed for env={env_id} episode={ep}"
                ) from exc
            env_successes.append((ep, success))
            print(
                f"[Parent] Completed env={env_id} episode={ep} seed={used_seed} "
                f"success={success}"
            )
            _print_episode_artifacts(output_dir, env_id, ep, used_seed)

    return [success for _, success in sorted(env_successes)]


def main() -> None:
    """脚本入口：解析参数、执行回放、打印结果路径。"""
    args = _build_parser().parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    n_episodes = args.episode_number
    if n_episodes < 1:
        raise SystemExit("--episode-number must be at least 1 (run episodes 0..N-1).")
    if n_episodes > 100:
        raise SystemExit(
            "--episode-number must be at most 100; the legacy seed layout only "
            "reserves 100 episode slots per environment."
        )
    episode_numbers = list(range(0, n_episodes))
    print(f"Environments (in order): {args.env}")
    print(
        f"Episode number N={n_episodes} → running episode indices {episode_numbers} "
        f"(0 .. {n_episodes - 1})"
    )
    print(
        "Seed policy: "
        "base_seed = 1000000 + env_code * 10000 + episode * 100; "
        f"each episode retries up to {MAX_SEED_ATTEMPTS} seeds."
    )
    print(f"Difficulty: {args.difficulty}")
    print(f"GPU: {args.gpu}")
    worker_count = _resolve_max_workers(args.max_workers, len(episode_numbers))
    print(f"Max workers per env: {worker_count}")
    print(f"Video output root: {output_dir}")

    successes: list[bool] = []
    for env_id in args.env:
        print(f"\n========== env={env_id} ==========")
        print(
            f"Dispatching {len(episode_numbers)} episodes for env={env_id} "
            f"with up to {worker_count} worker process(es)."
        )
        successes.extend(
            _run_env_episodes(
                env_id=env_id,
                episode_numbers=episode_numbers,
                difficulty=args.difficulty,
                output_dir=output_dir,
                max_workers=worker_count,
            )
        )

    all_success = all(successes)
    if all_success:
        print("Replay finished successfully (all episodes).")
    else:
        print("Replay finished with failure status (one or more episodes failed).")


if __name__ == "__main__":
    main()
