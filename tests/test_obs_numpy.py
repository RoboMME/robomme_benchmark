# -*- coding: utf-8 -*-
"""
test_obs_numpy.py
===================
集成测试：直接调用真实环境 + /data/hongzefu/data_0225 数据集，
测试 DemonstrationWrapper._augment_obs_and_info 内部原生的类型转换
在四种 ActionSpace 下对 obs / info 字段的输出类型和形状是否正确。

覆盖的 ActionSpace：
    joint_angle / ee_pose / waypoint / multi_choice

断言内容：
  1. 返回的 dtype 符合规范 (如 uint8, int16, float32, float64 等)
  2. info 非 Tensor 字段类型符合预期

运行方式（必须用 uv）：
    cd /data/hongzefu/robomme_benchmark
    uv run python tests/test_obs_numpy.py
"""
from __future__ import annotations

import copy
import sys
import traceback
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np

# 确保 src 路径可被找到
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from robomme.robomme_env import *  # noqa: F401,F403  注册所有自定义环境
from robomme.robomme_env.utils import *  # noqa: F401,F403
from robomme.env_record_wrapper import BenchmarkEnvBuilder, EpisodeDatasetResolver

# ──────────────────────────────────────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────────────────────────────────────
DATASET_ROOT = "/data/hongzefu/data_0225"
TEST_ENV_ID = "VideoUnmaskSwap"
TEST_EPISODE = 0
MAX_STEPS_PER_ACTION_SPACE = 3   # 每种 ActionSpace 最多验证的 step 数
MAX_STEPS_ENV = 1000

ActionSpaceType = Literal["joint_angle", "ee_pose", "waypoint", "multi_choice"]

# ──────────────────────────────────────────────────────────────────────────────
# 断言辅助
# ──────────────────────────────────────────────────────────────────────────────

def _assert_ndarray(val: Any, dtype: np.dtype, tag: str) -> None:
    assert isinstance(val, np.ndarray), (
        f"[{tag}] expected ndarray, got {type(val).__name__}"
    )
    assert val.dtype == dtype, (
        f"[{tag}] expected dtype={dtype}, got {val.dtype}"
    )


def _assert_ndarray_loose(val: Any, tag: str) -> None:
    """只断言是 ndarray，不检查具体 dtype。"""
    assert isinstance(val, np.ndarray), (
        f"[{tag}] expected ndarray, got {type(val).__name__}"
    )

# ──────────────────────────────────────────────────────────────────────────────
# 核心断言：原生输出类型正确
# ──────────────────────────────────────────────────────────────────────────────

def assert_obs(obs: dict, tag: str) -> None:
    """断言 obs 输出 dtype 正确且 shape 与预期一致。"""
    n = len(obs.get("front_rgb_list", []))
    assert n > 0, f"[{tag}] obs front_rgb_list is empty"

    for i in range(n):
        pfx = f"{tag}[{i}]"

        # ── RGB → uint8 ───────────────────────────────────────────────────
        for key, dtype in (("front_rgb_list", np.uint8), ("wrist_rgb_list", np.uint8)):
            _assert_ndarray(obs[key][i], dtype, f"{pfx} {key}")

        # ── Depth → int16 ─────────────────────────────────────────────────
        for key, dtype in (("front_depth_list", np.int16), ("wrist_depth_list", np.int16)):
            _assert_ndarray(obs[key][i], dtype, f"{pfx} {key}")

        # ── end_effector_pose_raw → dict 内各键 float32，且 shape 无 batch 维 ──
        eef_raw = obs["end_effector_pose_raw"][i]
        assert isinstance(eef_raw, dict), f"[{pfx} end_effector_pose_raw] expected dict"
        for key in ("pose", "quat", "rpy"):
            assert key in eef_raw, f"[{pfx} end_effector_pose_raw] missing key '{key}'"
            _assert_ndarray(eef_raw[key], np.float32, f"{pfx} end_effector_pose_raw['{key}']")
        assert eef_raw["pose"].shape == (3,), (
            f"[{pfx} end_effector_pose_raw['pose']] expected (3,), got {eef_raw['pose'].shape}"
        )
        assert eef_raw["quat"].shape == (4,), (
            f"[{pfx} end_effector_pose_raw['quat']] expected (4,), got {eef_raw['quat'].shape}"
        )
        assert eef_raw["rpy"].shape == (3,), (
            f"[{pfx} end_effector_pose_raw['rpy']] expected (3,), got {eef_raw['rpy'].shape}"
        )

        # ── eef_state_list → float64, shape (6,) ─────────────────────────
        eef_state = obs["eef_state_list"][i]
        _assert_ndarray(eef_state, np.float64, f"{pfx} eef_state_list")
        assert eef_state.shape == (6,), (
            f"[{pfx} eef_state_list] expected shape (6,), got {eef_state.shape}"
        )

        # ── joint_state_list → ndarray（shape 不变）───────────────────────
        _assert_ndarray_loose(obs["joint_state_list"][i], f"{pfx} joint_state_list")

        # ── gripper_state_list → ndarray（shape 不变）─────────────────────
        _assert_ndarray_loose(obs["gripper_state_list"][i], f"{pfx} gripper_state_list")

        # ── camera extrinsics → float32, shape (3,4) ───────────────────────
        for key in ("front_camera_extrinsic_list", "wrist_camera_extrinsic_list"):
            _assert_ndarray(obs[key][i], np.float32, f"{pfx} {key}")
            assert obs[key][i].shape == (3, 4), (
                f"[{pfx} {key}] expected (3, 4), got {obs[key][i].shape}"
            )


def assert_info(info: dict, tag: str) -> None:
    """断言 info 输出字段 dtype 等属实。"""
    for key in ("front_camera_intrinsic", "wrist_camera_intrinsic"):
        assert key in info, f"[{tag}] info missing key '{key}'"
        _assert_ndarray(info[key], np.float32, f"{tag} info['{key}']")
        assert info[key].shape == (3, 3), (
            f"[{tag} info['{key}']] expected (3, 3), got {info[key].shape}"
        )

    # 非 Tensor 字段类型不变
    task_goal = info.get("task_goal")
    assert isinstance(task_goal, (str, list, type(None))), (
        f"[{tag}] info['task_goal'] unexpected type {type(task_goal)}"
    )
    status = info.get("status")
    assert isinstance(status, (str, type(None))), (
        f"[{tag}] info['status'] unexpected type {type(status)}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# 单次 ActionSpace 的完整 episode 测试
# ──────────────────────────────────────────────────────────────────────────────

def _parse_oracle_command(choice_action: Optional[Any]) -> Optional[dict]:
    """与 dataset_replay—printType.py 中保持一致的 oracle 命令解析。"""
    if not isinstance(choice_action, dict):
        return None
    label = choice_action.get("label")
    if not isinstance(label, str) or not label:
        return None
    return choice_action


def run_one_action_space(action_space: ActionSpaceType) -> None:
    print(f"\n{'='*60}")
    print(f"[TEST] ActionSpace = {action_space}")
    print(f"{'='*60}")

    # multi_choice 使用 OraclePlannerDemonstrationWrapper，
    # BenchmarkEnvBuilder 直接使用统一后的 action_space 命名。

    env_builder = BenchmarkEnvBuilder(
        env_id=TEST_ENV_ID,
        dataset="train",
        action_space=action_space,
        gui_render=False,
    )
    env = env_builder.make_env_for_episode(TEST_EPISODE, max_steps=MAX_STEPS_ENV)

    dataset_resolver = EpisodeDatasetResolver(
        env_id=TEST_ENV_ID,
        episode=TEST_EPISODE,
        dataset_directory=DATASET_ROOT,
    )

    # ── RESET ──────────────────────────────────────────────────────────────
    obs, info = env.reset()

    reset_tag = f"{TEST_ENV_ID} ep{TEST_EPISODE} RESET [{action_space}]"
    assert_obs(obs, reset_tag)
    assert_info(info, reset_tag)
    print(f"  RESET 断言通过  (obs list len={len(obs['front_rgb_list'])}, dtype ✓)")

    # ── STEP LOOP ──────────────────────────────────────────────────────────
    step = 0
    while step < MAX_STEPS_PER_ACTION_SPACE:
        replay_key = action_space
        action = dataset_resolver.get_step(replay_key, step)
        if action_space == "multi_choice":
            action = _parse_oracle_command(action)
        if action is None:
            print(f"  step {step}: action=None（数据集结束），跳出")
            break

        obs, reward, terminated, truncated, info = env.step(action)

        step_tag = f"{TEST_ENV_ID} ep{TEST_EPISODE} STEP{step} [{action_space}]"
        assert_obs(obs, step_tag)
        assert_info(info, step_tag)
        print(f"  STEP {step} 断言通过  (obs list len={len(obs['front_rgb_list'])}, dtype ✓)")

        terminated_flag = bool(terminated.item())
        truncated_flag = bool(truncated.item())
        step += 1
        if terminated_flag or truncated_flag:
            print(f"  terminated={terminated_flag} truncated={truncated_flag}，提前退出")
            break

    env.close()
    print(f"  [{action_space}] ✓ 全部断言通过 (共 {step} 个 step)")


# ──────────────────────────────────────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────────────────────────────────────

ACTION_SPACES: list[ActionSpaceType] = [
    "joint_angle",
    "ee_pose",
    "waypoint",
    "multi_choice",
]


def main() -> None:
    all_pass = True
    results: list[tuple[str, str, str]] = []

    for action_space in ACTION_SPACES:
        try:
            run_one_action_space(action_space)
            results.append((action_space, "PASS", ""))
        except AssertionError as exc:
            results.append((action_space, "FAIL", str(exc)))
            all_pass = False
            print(f"\n  [断言失败] {exc}")
            traceback.print_exc()
        except Exception as exc:
            results.append((action_space, "ERROR", str(exc)))
            all_pass = False
            print(f"\n  [异常] {exc}")
            traceback.print_exc()

    print(f"\n{'='*60}")
    print("测试结果汇总")
    print(f"{'='*60}")
    for action_space, status, msg in results:
        marker = "✓" if status == "PASS" else "✗"
        suffix = f"  ({msg})" if msg else ""
        print(f"  {marker} [{status}] {action_space}{suffix}")

    if all_pass:
        print("\n✓ ALL ASSERTIONS PASSED")
        sys.exit(0)
    else:
        print("\n✗ SOME ASSERTIONS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
