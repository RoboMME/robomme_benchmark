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
    uv run python tests/dataset/test_replay_stick.py
"""

from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import pytest

from tests._shared.dataset_generation import DatasetCase, DatasetFactoryCache
from tests._shared.repo_paths import find_repo_root

pytestmark = pytest.mark.dataset

# ── 确保 robomme 包可被找到 ──────────────────────────────────────────────────
_PROJECT_ROOT = find_repo_root(__file__)
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from robomme.env_record_wrapper import BenchmarkEnvBuilder, EpisodeDatasetResolver  # noqa: E402
from robomme.robomme_env import *  # noqa: F401,F403,E402  注册所有自定义环境


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
    env = env_builder.make_env_for_episode(
        replay_episode,
        max_steps=1000,
        include_maniskill_obs=True,
        include_front_depth=True,
        include_wrist_depth=True,
        include_front_camera_extrinsic=True,
        include_wrist_camera_extrinsic=True,
        include_available_multi_choices=True,
        include_front_camera_intrinsic=True,
        include_wrist_camera_intrinsic=True,
    )

    # 统一数据生成 fixture 已提前准备好 record_dataset_{env_id}.h5 命名。

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

            eef_action = dataset_resolver.get_step("ee_pose", step_id)
            waypoint_action = dataset_resolver.get_step("waypoint", step_id)
            joint_action = dataset_resolver.get_step("joint_angle", step_id)

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
            gripper_state_list = obs["gripper_state_list"]

            if is_stick:
                for gs in gripper_state_list:
                    assert gs.shape == (2,), f"[{env_id}] gripper_state shape expected (2,)"
                    assert np.allclose(gs, 0.0), f"[{env_id}] gripper_state expected [0.0, 0.0] but got {gs}"
            else:
                for gs in gripper_state_list:
                    assert gs.shape == (2,), f"[{env_id}] gripper_state shape expected (2,)"

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


def _make_case(env_id: str, episode: int, base_seed: int, difficulty: str | None) -> DatasetCase:
    return DatasetCase(
        env_id=env_id,
        episode=episode,
        base_seed=base_seed,
        difficulty=difficulty,
        save_video=True,
        mode_tag="stick_record_replay",
    )


@pytest.mark.parametrize("env_id,is_stick,episode,base_seed,difficulty", TEST_CASES)
def test_replay_stick_case(
    env_id: str,
    is_stick: bool,
    episode: int,
    base_seed: int,
    difficulty: str | None,
    dataset_factory,
):
    generated = dataset_factory(_make_case(env_id, episode, base_seed, difficulty))
    _verify_replay(
        env_id=env_id,
        dataset_dir=generated.resolver_dataset_dir,
        h5_path=generated.raw_h5_path,
        is_stick=is_stick,
    )


def main():
    all_pass = True
    results = []

    with tempfile.TemporaryDirectory(prefix="test_replay_shared_cache_") as tmpdir:
        cache = DatasetFactoryCache(Path(tmpdir))
        for env_id, is_stick, episode, base_seed, difficulty in TEST_CASES:
            print(f"\n{'='*60}")
            print(f"测试用例: {env_id}  (is_stick={is_stick}, ep={episode})")
            print(f"{'='*60}")
            try:
                generated = cache.get(_make_case(env_id, episode, base_seed, difficulty))
                print(f"  => 录制完成：{generated.raw_h5_path}")
                print(f"  [2. Replay解析阶段]")
                _verify_replay(
                    env_id=env_id,
                    dataset_dir=generated.resolver_dataset_dir,
                    h5_path=generated.raw_h5_path,
                    is_stick=is_stick,
                )
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
