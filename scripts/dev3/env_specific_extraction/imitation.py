"""Imitation 套件 reset 钩子（no-op）+ 套件特有 planner 选择。

适用 env：MoveCube / InsertPeg / PatternLock / RouteStick。

为什么文件存在：
- 与其他 3 个 suite 保持 enrich_visible_payload 对外接口对称（reset 阶段
  imitation 没有要写到 visible_objects.json 的额外字段，函数为 no-op）。
- PatternLock / RouteStick 这两个 stick 末端执行器 env 需要专属的运动规划器
  (`FailAwarePandaStickMotionPlanningSolver` + joint_vel_limits=0.3)；该
  套件特有的 planner 选择从主脚本 `_create_planner` 抽出来，落到这里。
- default arm planner（`FailAwarePandaArmMotionPlanningSolver`）不属于
  imitation 套件特有逻辑，仍由主脚本在 `select_planner` 返回 None 时实例化。

本模块只 import 自 robomme.* 包（allowed），不会 import scripts/dev*、
scripts/dev3/ 下其他模块、或 scripts/dev2-snapshot-object/。
"""
from __future__ import annotations

from typing import Any, Optional

IMITATION_ENV_IDS: frozenset[str] = frozenset(
    {"MoveCube", "InsertPeg", "PatternLock", "RouteStick"}
)
STICK_ENV_IDS: frozenset[str] = frozenset({"PatternLock", "RouteStick"})


def enrich_visible_payload(payload: dict, env: Any, env_id: str) -> None:
    """Imitation 套件 reset 阶段无 env-specific 字段写入；保持同签名 no-op。

    非 imitation env_id 与 imitation env_id 都是 no-op。
    """
    if env_id not in IMITATION_ENV_IDS:
        return
    # Imitation 套件 reset 阶段没有需要写到 visible_objects.json 的额外字段。
    return


def select_planner(env: Any, env_id: str) -> Optional[Any]:
    """对 PatternLock / RouteStick 返回 FailAwarePandaStickMotionPlanningSolver
    实例（joint_vel_limits=0.3）；其他 env_id 一律返回 None，由调用方自行实例化
    默认 arm planner。

    返回 None 不代表错误，而是表示"本 suite 不接管此 env 的 planner 选择"，
    主脚本 `_create_planner` 收到 None 后会落到 default arm planner。
    """
    if env_id not in STICK_ENV_IDS:
        return None
    # 延迟 import：避免在不需要 stick planner 的子进程里加载 motion planning
    # 模块（含 sapien 物理引擎，启动开销不小）。
    from robomme.robomme_env.utils.planner_fail_safe import (
        FailAwarePandaStickMotionPlanningSolver,
    )

    return FailAwarePandaStickMotionPlanningSolver(
        env,
        debug=False,
        vis=False,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=False,
        print_env_info=False,
        joint_vel_limits=0.3,  # 比标准臂更保守，避免 stick 震荡
    )


__all__ = [
    "IMITATION_ENV_IDS",
    "STICK_ENV_IDS",
    "enrich_visible_payload",
    "select_planner",
]
