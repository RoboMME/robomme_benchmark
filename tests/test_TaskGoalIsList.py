"""
test_TaskGoalIsList.py

直接创建真实 Gymnasium 环境（包裹 DemonstrationWrapper），
调用 env.reset()，验证 info["task_goal"] 是 list 且非空。

全部 16 个 env 均覆盖。

运行：
    uv run python -m pytest tests/test_TaskGoalIsList.py -v -s
"""

import gymnasium as gym
import pytest

from robomme.env_record_wrapper.DemonstrationWrapper import DemonstrationWrapper

# ── 全部 16 个 env_id ──────────────────────────────────────────────────────────
ALL_ENV_IDS = [
    "BinFill",
    "PickXtimes",
    "SwingXtimes",
    "StopCube",
    "VideoUnmask",
    "VideoUnmaskSwap",
    "ButtonUnmask",
    "ButtonUnmaskSwap",
    "PickHighlight",
    "VideoRepick",
    "VideoPlaceButton",
    "VideoPlaceOrder",
    "MoveCube",
    "InsertPeg",
    "PatternLock",
    "RouteStick",
]


def _make_env(env_id: str):
    """创建并返回包裹了 DemonstrationWrapper 的真实环境。"""
    env = gym.make(
        env_id,
        obs_mode="rgb+depth+segmentation",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        reward_mode="dense",
    )
    env = DemonstrationWrapper(
        env,
        max_steps_without_demonstration=10002,  # 足够大，避免测试期间截断
        gui_render=False,
    )
    return env


@pytest.mark.parametrize("env_id", ALL_ENV_IDS)
def test_task_goal_is_list(env_id: str):
    """
    对每个 env：
    1. 创建真实环境（含 DemonstrationWrapper）
    2. 调用 reset()
    3. 断言 info["task_goal"] 是 list
    4. 断言 list 非空（至少 1 个元素）
    5. 断言每个元素都是 str
    """
    env = _make_env(env_id)
    try:
        _, info = env.reset()
    finally:
        env.close()

    task_goal = info["task_goal"]
    print(f"\n[{env_id}] task_goal = {task_goal!r}")

    assert isinstance(task_goal, list), (
        f"[{env_id}] info['task_goal'] 应为 list，实际为 {type(task_goal).__name__!r}: {task_goal!r}"
    )
    assert len(task_goal) >= 1, (
        f"[{env_id}] info['task_goal'] 不应为空 list"
    )
    for i, item in enumerate(task_goal):
        assert isinstance(item, str), (
            f"[{env_id}] task_goal[{i}] 应为 str，实际为 {type(item).__name__!r}: {item!r}"
        )
