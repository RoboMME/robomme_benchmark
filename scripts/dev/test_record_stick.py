"""
test_record_stick.py
====================
验证 Stick 环境（PatternLock）和非 Stick 环境（PickXtimes）
在 RecordWrapper 录制 HDF5 时，以下四处维度对齐是否正确：

1. gripper_state  : Stick → [0.0, 0.0]；非 Stick → shape==(2,)
2. joint_action   : Stick → shape==(8,) 且 [-1] == -1.0；非 Stick → shape==(8,)
3. eef_action     : Stick → shape==(7,) 且 [-1] == -1.0；非 Stick → shape==(7,)
4. waypoint_action: Stick → [-1] == -1.0；非 Stick → ±1.0

测试方法：参照 generate-dataset-control-seed-readJson-advanceV3.py，
对每个测试用例使用 FailAware Planner + screw→RRT* 重试 patch 跑一个完整 episode
（带种子重试），然后打开生成的 HDF5 文件逐项断言。

运行方式（需要 display / headless GPU）：
    cd /data/hongzefu/robomme_benchmark
    uv run python scripts/dev/test_record_stick.py
"""

import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
import gymnasium as gym

# ── 确保 robomme 包可被找到 ──────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]  # robomme_benchmark/
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from robomme.env_record_wrapper import RobommeRecordWrapper, FailsafeTimeout
from robomme.robomme_env import *  # 注册所有自定义环境
from robomme.robomme_env.utils.SceneGenerationError import SceneGenerationError
from robomme.robomme_env.utils.planner_fail_safe import (
    FailAwarePandaArmMotionPlanningSolver,
    FailAwarePandaStickMotionPlanningSolver,
    ScrewPlanFailure,
)

# ── 参照 V3 脚本的重试参数 ──────────────────────────────────────────────────
DATASET_SCREW_MAX_ATTEMPTS = 3
DATASET_RRT_MAX_ATTEMPTS = 3
MAX_SEED_ATTEMPTS = 30  # 种子重试上限

# ────────────────────────────────────────────────────────────────────────────
# 辅助函数
# ────────────────────────────────────────────────────────────────────────────

def _tensor_to_bool(value) -> bool:
    if value is None:
        return False
    if isinstance(value, torch.Tensor):
        return bool(value.detach().cpu().bool().item())
    if isinstance(value, np.ndarray):
        return bool(np.any(value))
    return bool(value)


def _patch_planner_screw_to_rrt(planner):
    """
    参照 V3 脚本：将 planner.move_to_pose_with_screw 替换为
    先重试 screw（最多 DATASET_SCREW_MAX_ATTEMPTS 次），
    失败后 fallback 到 RRT* 的复合版本。
    """
    original_screw = planner.move_to_pose_with_screw
    original_rrt = planner.move_to_pose_with_RRTStar

    def _move_screw_then_rrt(*args, **kwargs):
        for attempt in range(1, DATASET_SCREW_MAX_ATTEMPTS + 1):
            try:
                result = original_screw(*args, **kwargs)
            except ScrewPlanFailure as exc:
                print(f"[Planner] screw attempt {attempt}/{DATASET_SCREW_MAX_ATTEMPTS} failed: {exc}")
                continue
            if isinstance(result, int) and result == -1:
                print(f"[Planner] screw attempt {attempt}/{DATASET_SCREW_MAX_ATTEMPTS} returned -1")
                continue
            return result

        print(f"[Planner] screw exhausted; fallback to RRT* (max {DATASET_RRT_MAX_ATTEMPTS})")
        for attempt in range(1, DATASET_RRT_MAX_ATTEMPTS + 1):
            try:
                result = original_rrt(*args, **kwargs)
            except Exception as exc:
                print(f"[Planner] RRT* attempt {attempt}/{DATASET_RRT_MAX_ATTEMPTS} failed: {exc}")
                continue
            if isinstance(result, int) and result == -1:
                print(f"[Planner] RRT* attempt {attempt}/{DATASET_RRT_MAX_ATTEMPTS} returned -1")
                continue
            return result

        print("[Planner] screw→RRT* exhausted; return -1")
        return -1

    planner.move_to_pose_with_screw = _move_screw_then_rrt


def _run_one_episode(
    env_id: str,
    episode: int,
    seed: int,
    difficulty: Optional[str],
    output_dir: Path,
) -> bool:
    """
    参照 V3 脚本的 _run_episode_attempt 跑单个 episode。
    返回是否成功（成功时 HDF5 已由 close() 落盘）。
    """
    env_kwargs = dict(
        obs_mode="rgb+depth+segmentation",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        reward_mode="dense",
        seed=seed,
        difficulty=difficulty,
    )
    # V3 对前几个 episode 开启 failure recovery
    if episode <= 5:
        env_kwargs["robomme_failure_recovery"] = True
        env_kwargs["robomme_failure_recovery_mode"] = "z" if episode <= 2 else "xy"

    env = gym.make(env_id, **env_kwargs)
    # save_video=True：_video_should_record() 同时控制 HDF5 数据录制分支
    env = RobommeRecordWrapper(
        env,
        dataset=str(output_dir),
        env_id=env_id,
        episode=episode,
        seed=seed,
        save_video=True,
    )

    episode_successful = False
    try:
        env.reset()

        is_stick = env_id in ("PatternLock", "RouteStick")
        if is_stick:
            planner = FailAwarePandaStickMotionPlanningSolver(
                env,
                debug=False, vis=False,
                base_pose=env.unwrapped.agent.robot.pose,
                visualize_target_grasp_pose=False,
                print_env_info=False,
                joint_vel_limits=0.3,
            )
        else:
            planner = FailAwarePandaArmMotionPlanningSolver(
                env,
                debug=False, vis=False,
                base_pose=env.unwrapped.agent.robot.pose,
                visualize_target_grasp_pose=False,
                print_env_info=False,
            )

        # V3 的关键改进：patch screw 为 screw→RRT* fallback
        _patch_planner_screw_to_rrt(planner)

        env.unwrapped.evaluate()
        tasks = list(getattr(env.unwrapped, "task_list", []) or [])
        print(f"  [{env_id}] task_list 共 {len(tasks)} 个子任务，seed={seed}")

        for idx, task_entry in enumerate(tasks):
            task_name = task_entry.get("name", f"Task {idx}")
            print(f"  [{env_id}] 执行子任务 {idx + 1}/{len(tasks)}: {task_name}")
            solve_callable = task_entry.get("solve")
            if not callable(solve_callable):
                raise ValueError(f"Task '{task_name}' 没有合法的 solve 函数")

            env.unwrapped.evaluate(solve_complete_eval=True)
            screw_failed = False
            try:
                solve_result = solve_callable(env, planner)
                if isinstance(solve_result, int) and solve_result == -1:
                    screw_failed = True
                    print(f"  [{env_id}] screw→RRT* planning exhausted during '{task_name}'")
                    env.unwrapped.failureflag = torch.tensor([True])
                    env.unwrapped.successflag = torch.tensor([False])
                    env.unwrapped.current_task_failure = True
            except ScrewPlanFailure as exc:
                screw_failed = True
                print(f"  [{env_id}] ScrewPlanFailure: {exc}")
                env.unwrapped.failureflag = torch.tensor([True])
                env.unwrapped.successflag = torch.tensor([False])
                env.unwrapped.current_task_failure = True
            except FailsafeTimeout as exc:
                print(f"  [{env_id}] FailsafeTimeout: {exc}")
                break

            evaluation = env.unwrapped.evaluate(solve_complete_eval=True)
            fail_flag = evaluation.get("fail", False)
            success_flag = evaluation.get("success", False)

            if _tensor_to_bool(success_flag):
                print(f"  [{env_id}] Episode 成功！")
                episode_successful = True
                break
            if screw_failed or _tensor_to_bool(fail_flag):
                print(f"  [{env_id}] 遇到失败条件，停止。")
                break
        else:
            evaluation = env.unwrapped.evaluate(solve_complete_eval=True)
            episode_successful = _tensor_to_bool(evaluation.get("success", False))

        episode_successful = episode_successful or _tensor_to_bool(
            getattr(env, "episode_success", False)
        )

    except SceneGenerationError as exc:
        print(f"  [{env_id}] SceneGenerationError: {exc}")
        episode_successful = False
    finally:
        try:
            env.close()
        except Exception as exc:
            print(f"  [{env_id}] close() 异常（忽略）: {exc}")

    return episode_successful


def _run_episode_with_retry(
    env_id: str,
    episode: int,
    base_seed: int,
    difficulty: Optional[str],
    output_dir: Path,
) -> Path:
    """
    带种子重试的 episode 运行（参照 V3 脚本 run_env_dataset 中的重试逻辑）：
    失败时累加 seed 继续尝试，直到成功或耗尽 MAX_SEED_ATTEMPTS。
    成功后返回 HDF5 文件路径。
    """
    for attempt in range(MAX_SEED_ATTEMPTS):
        seed = base_seed + attempt
        print(f"\n  [{env_id}] 种子尝试 {attempt + 1}/{MAX_SEED_ATTEMPTS}, seed={seed}")
        try:
            success = _run_one_episode(env_id, episode, seed, difficulty, output_dir)
        except Exception as exc:
            print(f"  [{env_id}] 异常: {exc}，尝试下一个种子")
            continue

        if success:
            h5_path = output_dir / "hdf5_files" / f"{env_id}_ep{episode}_seed{seed}.h5"
            if not h5_path.exists():
                raise FileNotFoundError(f"期望的 HDF5 文件不存在: {h5_path}")
            return h5_path
        print(f"  [{env_id}] seed={seed} 未成功，尝试下一个种子...")

    raise RuntimeError(
        f"[{env_id}] 连续 {MAX_SEED_ATTEMPTS} 个种子均未成功，放弃验证"
    )


# ────────────────────────────────────────────────────────────────────────────
# 断言函数
# ────────────────────────────────────────────────────────────────────────────

def _verify_stick(h5_path: Path, env_id: str):
    """验证 Stick 环境 HDF5 数据断言。"""
    print(f"\n  [验证 Stick] 打开 {h5_path.name}")
    with h5py.File(h5_path, "r") as f:
        episode_keys = [k for k in f.keys() if k.startswith("episode_")]
        assert len(episode_keys) > 0, "HDF5 文件中没有 episode 组"
        ep_grp = f[episode_keys[0]]
        ts_keys = [k for k in ep_grp.keys() if k.startswith("timestep_")]
        assert len(ts_keys) > 0, "episode 组中没有 timestep"

        for ts_key in ts_keys:
            ts = ep_grp[ts_key]

            # 1. gripper_state → [0.0, 0.0]
            gs = np.array(ts["obs"]["gripper_state"])
            assert gs.shape == (2,), \
                f"[{env_id}/{ts_key}] gripper_state shape={gs.shape} 期望 (2,)"
            assert np.allclose(gs, 0.0), \
                f"[{env_id}/{ts_key}] gripper_state={gs} 期望 [0.0, 0.0]"

            # 2. joint_action → 8维，最末位 == -1.0
            ja = np.array(ts["action"]["joint_action"]).flatten()
            assert ja.shape == (8,), \
                f"[{env_id}/{ts_key}] joint_action shape={ja.shape} 期望 (8,)"
            assert float(ja[-1]) == -1.0, \
                f"[{env_id}/{ts_key}] joint_action[-1]={ja[-1]} 期望 -1.0"

            # 3. eef_action → 7维，最末位 == -1.0
            ea = np.array(ts["action"]["eef_action"]).flatten()
            assert ea.shape == (7,), \
                f"[{env_id}/{ts_key}] eef_action shape={ea.shape} 期望 (7,)"
            assert float(ea[-1]) == -1.0, \
                f"[{env_id}/{ts_key}] eef_action[-1]={ea[-1]} 期望 -1.0"

            # 4. waypoint_action → 7维，最末位 == -1.0
            wa = np.array(ts["action"]["waypoint_action"]).flatten()
            assert wa.shape == (7,), \
                f"[{env_id}/{ts_key}] waypoint_action shape={wa.shape} 期望 (7,)"
            assert float(wa[-1]) == -1.0, \
                f"[{env_id}/{ts_key}] waypoint_action[-1]={wa[-1]} 期望 -1.0"

    print(f"  [验证 Stick ✓] {env_id} 所有断言通过，共 {len(ts_keys)} 个 timestep")


def _verify_non_stick(h5_path: Path, env_id: str):
    """验证非 Stick 环境 HDF5 数据断言（原有逻辑未被破坏）。"""
    print(f"\n  [验证 非Stick] 打开 {h5_path.name}")
    with h5py.File(h5_path, "r") as f:
        episode_keys = [k for k in f.keys() if k.startswith("episode_")]
        assert len(episode_keys) > 0, "HDF5 文件中没有 episode 组"
        ep_grp = f[episode_keys[0]]
        ts_keys = [k for k in ep_grp.keys() if k.startswith("timestep_")]
        assert len(ts_keys) > 0, "episode 组中没有 timestep"

        for ts_key in ts_keys:
            ts = ep_grp[ts_key]

            # 1. gripper_state shape == (2,)
            gs = np.array(ts["obs"]["gripper_state"])
            assert gs.shape == (2,), \
                f"[{env_id}/{ts_key}] gripper_state shape={gs.shape} 期望 (2,)"

            # 2. joint_action → 8维
            ja = np.array(ts["action"]["joint_action"]).flatten()
            assert ja.shape == (8,), \
                f"[{env_id}/{ts_key}] joint_action shape={ja.shape} 期望 (8,)"

            # 3. eef_action → 7维
            ea = np.array(ts["action"]["eef_action"]).flatten()
            assert ea.shape == (7,), \
                f"[{env_id}/{ts_key}] eef_action shape={ea.shape} 期望 (7,)"

            # 4. waypoint_action → 7维，last in {-1.0, 1.0}
            wa = np.array(ts["action"]["waypoint_action"]).flatten()
            assert wa.shape == (7,), \
                f"[{env_id}/{ts_key}] waypoint_action shape={wa.shape} 期望 (7,)"
            assert float(wa[-1]) in (-1.0, 1.0), \
                f"[{env_id}/{ts_key}] waypoint_action[-1]={wa[-1]} 应为 ±1.0"

    print(f"  [验证 非Stick ✓] {env_id} 所有断言通过，共 {len(ts_keys)} 个 timestep")


# ────────────────────────────────────────────────────────────────────────────
# 测试用例配置
# ────────────────────────────────────────────────────────────────────────────

# (env_id, is_stick, episode, base_seed, difficulty)
# base_seed 与 V3 脚本中 SOURCE_METADATA_ROOT 对应的 seed 无关，
# 这里直接使用 generate_dataset.py 的 SEED_OFFSET 规则
TEST_CASES = [
    ("PatternLock", True,  0, 510001, "easy"),
    ("PickXtimes",  False, 0, 504101, "easy"),
]


def main():
    all_pass = True
    results = []

    for env_id, is_stick, episode, base_seed, difficulty in TEST_CASES:
        print(f"\n{'='*60}")
        print(f"测试用例: {env_id}  (is_stick={is_stick}, ep={episode}, base_seed={base_seed})")
        print(f"{'='*60}")

        with tempfile.TemporaryDirectory(prefix=f"test_record_{env_id}_") as tmpdir:
            output_dir = Path(tmpdir)
            try:
                h5_path = _run_episode_with_retry(
                    env_id, episode, base_seed, difficulty, output_dir
                )
                if is_stick:
                    _verify_stick(h5_path, env_id)
                else:
                    _verify_non_stick(h5_path, env_id)
                results.append((env_id, "PASS", None))
            except AssertionError as exc:
                results.append((env_id, "FAIL", str(exc)))
                all_pass = False
                print(f"\n  [断言失败] {exc}")
                traceback.print_exc()
            except Exception as exc:
                results.append((env_id, "ERROR", str(exc)))
                all_pass = False
                print(f"\n  [错误] {exc}")
                traceback.print_exc()

    # ── 汇总输出 ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("测试结果汇总")
    print(f"{'='*60}")
    for env_id, status, msg in results:
        marker = "✓" if status == "PASS" else "✗"
        suffix = f"  ({msg})" if msg else ""
        print(f"  {marker} [{status}] {env_id}{suffix}")

    if all_pass:
        print("\n✓ ALL ASSERTIONS PASSED")
        sys.exit(0)
    else:
        print("\n✗ SOME ASSERTIONS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
