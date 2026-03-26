"""单次回放 ButtonUnmaskInspect 脚本。

这个脚本的职责有三件事：
1. 按给定环境、关卡编号和随机种子创建 Robomme 环境并执行一局任务。
2. 使用 `RobommeRecordWrapper` 录制回放视频，方便后续人工检查。
3. 在 `ButtonUnmaskSwap` 的固定时间步额外抓取一次“drop 之后”的场景快照，
   把箱子、方块及其对应关系导出为 JSON，便于离线比对状态是否正确。

整体执行链路是：
参数解析 -> 创建环境 -> 给 `env.step` 打补丁以便抓快照 -> 创建规划器并加重试逻辑
-> 逐任务调用 solve -> 关闭环境 -> 打印最终视频路径。
"""

import argparse
import os
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
VALID_ENVS: Set[str] = set(DEFAULT_ENVS)
VALID_DIFFICULTIES: Set[str] = {"easy", "medium", "hard"}
DATASET_SCREW_MAX_ATTEMPTS = 3
DATASET_RRT_MAX_ATTEMPTS = 3


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


def _build_parser() -> argparse.ArgumentParser:
    """定义命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Run a Robomme episode and record video."
    )
    parser.add_argument(
        "--env",
        "-e",
        default="ButtonUnmaskSwap",
        choices=sorted(VALID_ENVS),
        help="Environment ID to run (default: ButtonUnmaskSwap).",
    )
    parser.add_argument(
        "--episode-number",
        type=int,
        default=3,
        metavar="N",
        help="Episode index to run (default: 3).",
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
    snapshot_state: dict = {"snapshot_written": False, "snapshot_json_path": None}

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
        # 快照缺失本身不一定说明 episode 失败，但通常意味着流程比预期更早结束了。
        print(
            "Warning: after-drop snapshot JSON was not captured before the episode ended."
        )
    return episode_successful


def main() -> None:
    """脚本入口：解析参数、执行回放、打印结果路径。"""
    args = _build_parser().parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ep = args.episode_number
    print(f"Environment: {args.env}")
    print(f"Episode number: {ep}")
    print(f"Seed: {args.seed}")
    print(f"Difficulty: {args.difficulty}")
    print(f"GPU: {args.gpu}")
    print(f"Video output root: {output_dir}")

    success = _run_episode(
        env_id=args.env,
        episode=ep,
        seed=args.seed,
        difficulty=args.difficulty,
        output_dir=output_dir,
    )

    # 视频文件由 wrapper 异步落盘；这里在 episode 结束后再去定位最终产物。
    mp4_path = _latest_recorded_mp4(output_dir, args.env, ep, args.seed)
    if mp4_path is not None:
        print(f"Final MP4: {mp4_path.resolve()}")
    else:
        print(
            f"No MP4 matched under {output_dir / 'videos'} "
            f"(expected filename fragment '{args.env}_ep{ep}_seed{args.seed}')."
        )

    if success:
        print("Replay finished successfully.")
    else:
        print("Replay finished with failure status.")


if __name__ == "__main__":
    main()
