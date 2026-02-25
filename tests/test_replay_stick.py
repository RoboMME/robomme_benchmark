"""
test_replay_stick.py
====================
验证 Stick 环境（PatternLock）和非 Stick 环境（PickXtimes）
在被 EpisodeDatasetResolver（dataset_replay.py 读取的方式）解析和重放时，
各类维度和 state 是否按预期对齐。
与 test_record_stick.py 类似，我们会先短跑一两个完整的 episode，确保本地有一个正确的 HDF5 文件。
然后用 BenchmarkEnvBuilder 结合 EpisodeDatasetResolver 进行重放读取并针对 obs 断言。

1. gripper_state(读出的 eef_state_list 和 obs): Stick -> [0.0, 0.0]; 非 Stick -> shape(2,)
2. action (eef / joint_action 从 resolver 中读取): 末端维度对齐
3. 等等

运行方式（需要 display / headless GPU）：
    cd /data/hongzefu/robomme_benchmark
    uv run python tests/test_replay_stick.py
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
_PROJECT_ROOT = Path(__file__).resolve().parents[1]  # robomme_benchmark/ when under tests/
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from robomme.env_record_wrapper import RobommeRecordWrapper, FailsafeTimeout, BenchmarkEnvBuilder, EpisodeDatasetResolver
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
# 辅助录制和 Planner 补丁函数
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
    original_screw = planner.move_to_pose_with_screw
    original_rrt = planner.move_to_pose_with_RRTStar

    def _move_screw_then_rrt(*args, **kwargs):
        for attempt in range(1, DATASET_SCREW_MAX_ATTEMPTS + 1):
            try:
                result = original_screw(*args, **kwargs)
            except ScrewPlanFailure as exc:
                continue
            if isinstance(result, int) and result == -1:
                continue
            return result
        for attempt in range(1, DATASET_RRT_MAX_ATTEMPTS + 1):
            try:
                result = original_rrt(*args, **kwargs)
            except Exception as exc:
                continue
            if isinstance(result, int) and result == -1:
                continue
            return result
        return -1
    planner.move_to_pose_with_screw = _move_screw_then_rrt

def _run_one_record_episode(
    env_id: str,
    episode: int,
    seed: int,
    difficulty: Optional[str],
    output_dir: Path,
) -> bool:
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
        env_kwargs["robomme_failure_recovery_mode"] = "z" if episode <= 2 else "xy"

    env = gym.make(env_id, **env_kwargs)
    env = RobommeRecordWrapper(
        env,
        dataset=str(output_dir),
        env_id=env_id,
        episode=episode,
        seed=seed,
        save_video=False,
    )

    episode_successful = False
    try:
        env.reset()
        is_stick = env_id in ("PatternLock", "RouteStick")
        if is_stick:
            planner = FailAwarePandaStickMotionPlanningSolver(
                env, debug=False, vis=False,
                base_pose=env.unwrapped.agent.robot.pose,
                visualize_target_grasp_pose=False, print_env_info=False, joint_vel_limits=0.3,
            )
        else:
            planner = FailAwarePandaArmMotionPlanningSolver(
                env, debug=False, vis=False,
                base_pose=env.unwrapped.agent.robot.pose,
                visualize_target_grasp_pose=False, print_env_info=False,
            )

        _patch_planner_screw_to_rrt(planner)

        tasks = list(getattr(env.unwrapped, "task_list", []) or [])
        for idx, task_entry in enumerate(tasks):
            solve_callable = task_entry.get("solve")
            env.unwrapped.evaluate(solve_complete_eval=True)
            screw_failed = False
            try:
                solve_result = solve_callable(env, planner)
                if isinstance(solve_result, int) and solve_result == -1:
                    screw_failed = True
                    env.unwrapped.failureflag = torch.tensor([True])
                    env.unwrapped.successflag = torch.tensor([False])
                    env.unwrapped.current_task_failure = True
            except ScrewPlanFailure as exc:
                screw_failed = True
                env.unwrapped.failureflag = torch.tensor([True])
                env.unwrapped.successflag = torch.tensor([False])
                env.unwrapped.current_task_failure = True
            except FailsafeTimeout as exc:
                break

            evaluation = env.unwrapped.evaluate(solve_complete_eval=True)
            fail_flag = evaluation.get("fail", False)
            success_flag = evaluation.get("success", False)

            if _tensor_to_bool(success_flag):
                episode_successful = True
                break
            if screw_failed or _tensor_to_bool(fail_flag):
                break
        else:
            evaluation = env.unwrapped.evaluate(solve_complete_eval=True)
            episode_successful = _tensor_to_bool(evaluation.get("success", False))

        episode_successful = episode_successful or _tensor_to_bool(getattr(env, "episode_success", False))

    except SceneGenerationError as exc:
        episode_successful = False
    finally:
        try:
            env.close()
        except:
            pass

    return episode_successful


def _run_record_episode_with_retry(
    env_id: str,
    episode: int,
    base_seed: int,
    difficulty: Optional[str],
    output_dir: Path,
) -> Path:
    for attempt in range(MAX_SEED_ATTEMPTS):
        seed = base_seed + attempt
        try:
            success = _run_one_record_episode(env_id, episode, seed, difficulty, output_dir)
        except Exception as exc:
            continue

        if success:
            h5_path = output_dir / "hdf5_files" / f"{env_id}_ep{episode}_seed{seed}.h5"
            if not h5_path.exists():
                raise FileNotFoundError(f"Missing expected HDF5: {h5_path}")
            return h5_path
    raise RuntimeError(f"[{env_id}] Failed to generate successful record in {MAX_SEED_ATTEMPTS} attempts.")

# ────────────────────────────────────────────────────────────────────────────
# 断言函数（重放测试阶段）
# ────────────────────────────────────────────────────────────────────────────
def _verify_replay(env_id: str, dataset_dir: Path, h5_path: Path, is_stick: bool):
    """验证从 Builder 和 Resolver 读取的状态是否符合预期模型维数规则"""
    
    # 我们知道我们刚录制的是 ep 0
    replay_episode = 0
    # 注意，在 Dataset Resolver 中，我们会扫描 H5 文件所在夹，由于我们在重试时可能会导致后缀 Seed 变化
    # 解析依然能自动找到以 dataset_dir 传参对应的第一个 HDF5 文件
    
    ACTION_SPACE = "joint_angle"
    print(f"\n  [启动 Replay 验证] ACTION_SPACE: {ACTION_SPACE}, env: {env_id}")
    env_builder = BenchmarkEnvBuilder(
        env_id=env_id,
        dataset="test",  # 并不真实用 dataset json 扫描，只是作为 placeholder
        action_space=ACTION_SPACE,
        gui_render=False,
    )
    
    # 绕过 resolver json 的 episode 限缩，直接通过 dataset resolver 本地文件搜索。
    env = env_builder.make_env_for_episode(replay_episode, max_steps=1000)
    
    # EpisodeDatasetResolver expects the file to be named `record_dataset_{env_id}.h5` 
    # and placed directly in `dataset_directory`. Let's copy the mapped file there.
    import shutil
    resolver_expected_path = dataset_dir / f"record_dataset_{env_id}.h5"
    shutil.copy2(h5_path, resolver_expected_path)

    # 创建解析器（需传到 dataset_dir/hdf5_files 的上一层，即 dataset_dir）
    try:
        dataset_resolver = EpisodeDatasetResolver(
            env_id=env_id,
            episode=replay_episode,
            dataset_directory=str(dataset_dir),
        )
    except Exception as e:
        env.close()
        raise e
        
    try:
        obs, info = env.reset()
        
        step_id = 0
        while True:
            # ======= 获取并验证 Dataset中的action =======
            action = dataset_resolver.get_step(ACTION_SPACE, step_id)
            if action is None:
                break
                
            eef_action = dataset_resolver.get_step("eef_action", step_id)
            waypoint_action = dataset_resolver.get_step("waypoint_action", step_id)
            joint_action = dataset_resolver.get_step("joint_action", step_id)
                
            if is_stick:
                assert float(joint_action[-1]) == -1.0, f"[{env_id}] joint_action[-1]={joint_action[-1]} expected -1.0"
                if eef_action is not None:
                     assert float(eef_action[-1]) == -1.0, f"[{env_id}] eef_action[-1]={eef_action[-1]} expected -1.0"
                if waypoint_action is not None and len(waypoint_action) > 0:
                     assert float(waypoint_action[-1]) == -1.0, f"[{env_id}] waypoint_action[-1]={waypoint_action[-1]} expected -1.0"
            else:
                if waypoint_action is not None and len(waypoint_action) >0:
                    assert float(waypoint_action[-1]) in (-1.0, 1.0), f"[{env_id}] expected ±1 for waypoint_action"

            # ======= 执行 step =======
            obs, reward, terminated, truncated, info = env.step(action)
            
            # ======= 断言 DemonstrationWrapper 返回的 obs 状态 =======
            gripper_state = obs["gripper_state_list"]
            
            if is_stick:
                assert len(gripper_state) == 2, f"[{env_id}] gripper_state shape expected 2"
                assert np.allclose(gripper_state, 0.0), f"[{env_id}] gripper_state expected [0.0, 0.0] but got {gripper_state}"
            else:
                assert len(gripper_state) == 2, f"[{env_id}] gripper_state shape expected 2"

            step_id += 1
            if truncated.item() or terminated.item():
                break
                
    finally:
        env.close()

    print(f"  [{env_id} - Replay 验证 ✓] 所有断言通过，共重放 {step_id} 个 timestep")

# ────────────────────────────────────────────────────────────────────────────
# 测试流程控制
# ────────────────────────────────────────────────────────────────────────────
TEST_CASES = [
    ("PatternLock", True,  0, 510001, "easy"),
    ("PickXtimes",  False, 0, 504101, "easy"),
]

def main():
    all_pass = True
    results = []

    for env_id, is_stick, episode, base_seed, difficulty in TEST_CASES:
        print(f"\n{'='*60}")
        print(f"测试用例: {env_id}  (is_stick={is_stick}, ep={episode})")
        print(f"{'='*60}")

        with tempfile.TemporaryDirectory(prefix=f"test_replay_{env_id}_") as tmpdir:
            output_dir = Path(tmpdir)
            try:
                # 1.先生成成功的数据集文件 HDF5
                print(f"  [1. 录制数据阶段]")
                h5_path = _run_record_episode_with_retry(
                    env_id, episode, base_seed, difficulty, output_dir
                )
                print(f"  => 录制完成：{h5_path}")
                
                # 2.重新加载并验证重放
                print(f"  [2. Replay解析阶段]")
                _verify_replay(env_id, output_dir, h5_path, is_stick=is_stick)
                
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
