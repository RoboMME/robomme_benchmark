"""Imitation 套件 reset 钩子 + 套件特有 planner 选择。

适用 env：MoveCube / InsertPeg / PatternLock / RouteStick。

为什么文件存在：
- 与其他 3 个 suite 保持 enrich_visible_payload 对外接口对称。InsertPeg 与
  MoveCube 在 reset 时分别在 env 内部用独立 seed-offset torch.Generator 选择
  (direction, obj_flag) / (way)；本钩子负责把这些选择 + 相关 actor xy 写到
  ``visible_objects.json`` 顶层，给 inspect 端做 distribution / scatter 可视化。
  PatternLock / RouteStick 没有此类选择，对它们而言钩子是 no-op。
- PatternLock / RouteStick 这两个 stick 末端执行器 env 需要专属的运动规划器
  (``FailAwarePandaStickMotionPlanningSolver`` + joint_vel_limits=0.3)；该
  套件特有的 planner 选择从主脚本 ``_create_planner`` 抽出来，落到这里。
- default arm planner（``FailAwarePandaArmMotionPlanningSolver``）不属于
  imitation 套件特有逻辑，仍由主脚本在 ``select_planner`` 返回 None 时实例化。

数据写入策略：与 reference 套件一致，imitation 不写独立 sidecar 文件，所有
字段都原地塞进 visible_objects.json 的顶层 ——
- InsertPeg → 'insertpeg_choice'
- MoveCube  → 'movecube_choice'
inspect 端通过同一份 visible_objects.json 即可读到这些字段，无需走独立
sidecar 路径。

本模块只 import 自 robomme.* / 标准库 / 第三方包（allowed），不会 import
scripts/dev*、scripts/dev3/ 下其他模块、或 scripts/dev2-snapshot-object/。
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

IMITATION_ENV_IDS: frozenset[str] = frozenset(
    {"MoveCube", "InsertPeg", "PatternLock", "RouteStick"}
)
STICK_ENV_IDS: frozenset[str] = frozenset({"PatternLock", "RouteStick"})

# visible_objects.json 顶层字段名
INSERTPEG_CHOICE_KEY = "insertpeg_choice"
MOVECUBE_CHOICE_KEY = "movecube_choice"

# 标签：与 InsertPeg / MoveCube env 内部的 raw 值映射对齐
INSERTPEG_DIRECTION_LABELS = {-1: "left", 1: "right"}
# obj_flag = -1 时 grasp head / insert tail；obj_flag = +1 时 grasp tail / insert head
# 用 insert_end 表达「选中那一端插入孔」，而非 grasp_end，避免 head/tail 二义。
INSERTPEG_INSERT_END_LABELS = {-1: "tail", 1: "head"}
MOVECUBE_WAYS = ("peg_push", "gripper_push", "grasp_putdown")


# ---------------------------------------------------------------------------
# 内部工具（与 permanence.py / reference.py 同名 helper 等价；4 个 suite 模块
# 之间不互相 import，故各自重复实现 —— 等出现第 3 个 suite 用到时再抽到
# _rollout_common.py）。
# ---------------------------------------------------------------------------


def _to_jsonable(value: Any) -> Any:
    """递归把 numpy / torch 张量转成 JSON 可序列化的 Python 原生类型。"""
    if value is None:
        return None
    if isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
        return value.detach().cpu().numpy().tolist()
    return value


def _actor_xy(actor: Any) -> list[float]:
    """从 actor.pose.p 提取 (x, y)。pose.p 形状通常为 [B, 3] 或 [3]。"""
    if actor is None:
        raise ValueError("cannot read pose from None actor")
    pose = getattr(actor, "pose", None)
    if pose is None:
        raise ValueError(f"actor {actor!r} missing .pose attribute")
    p = pose.p
    if hasattr(p, "detach"):
        p = p.detach()
    if hasattr(p, "cpu"):
        p = p.cpu()
    arr = np.asarray(p)
    if arr.ndim == 2:
        arr = arr[0]
    if arr.ndim != 1 or arr.shape[0] < 2:
        raise ValueError(f"unexpected pose.p shape {arr.shape} for actor {actor!r}")
    return [float(arr[0]), float(arr[1])]


# ---------------------------------------------------------------------------
# InsertPeg / MoveCube 的 choice 提取
# ---------------------------------------------------------------------------


def _extract_insertpeg_choice(base: Any) -> dict:
    """从 InsertPeg env.unwrapped 提取 (direction, obj_flag) + 关键 actor xy。

    direction ∈ {-1, +1}（-1=left, +1=right）
    obj_flag  ∈ {-1, +1}（-1=insert_tail, +1=insert_head）
    """
    direction = getattr(base, "direction", None)
    obj_flag = getattr(base, "obj_flag", None)
    if direction is None:
        raise ValueError("InsertPeg.direction is None after reset")
    if obj_flag is None:
        raise ValueError("InsertPeg.obj_flag is None after reset")
    direction = int(direction)
    obj_flag = int(obj_flag)
    if direction not in (-1, 1):
        raise ValueError(
            f"InsertPeg.direction must be -1 or +1, got {direction}"
        )
    if obj_flag not in (-1, 1):
        raise ValueError(
            f"InsertPeg.obj_flag must be -1 or +1, got {obj_flag}"
        )

    peg_head = getattr(base, "peg_head", None)
    peg_tail = getattr(base, "peg_tail", None)
    box = getattr(base, "box", None)
    if peg_head is None:
        raise ValueError("InsertPeg.peg_head is None after reset")
    if peg_tail is None:
        raise ValueError("InsertPeg.peg_tail is None after reset")
    if box is None:
        raise ValueError("InsertPeg.box is None after reset")

    peg_head_xy = _actor_xy(peg_head)
    peg_tail_xy = _actor_xy(peg_tail)
    box_xy = _actor_xy(box)

    # insert_end_xy = the end of the peg that ends up inserted into the hole.
    # obj_flag = -1 → insert_target = peg_tail；obj_flag = +1 → insert_target = peg_head.
    insert_end_xy = peg_tail_xy if obj_flag == -1 else peg_head_xy

    return {
        "direction": direction,
        "direction_label": INSERTPEG_DIRECTION_LABELS[direction],
        "obj_flag": obj_flag,
        "insert_end_label": INSERTPEG_INSERT_END_LABELS[obj_flag],
        "peg_head_xy": peg_head_xy,
        "peg_tail_xy": peg_tail_xy,
        "insert_end_xy": insert_end_xy,
        "box_xy": box_xy,
    }


def _extract_movecube_choice(base: Any) -> dict:
    """从 MoveCube env.unwrapped 提取 way ∈ MOVECUBE_WAYS。"""
    way = getattr(base, "way", None)
    if way is None:
        raise ValueError("MoveCube.way is None after reset")
    way = str(way)
    if way not in MOVECUBE_WAYS:
        raise ValueError(
            f"MoveCube.way must be one of {MOVECUBE_WAYS}, got {way!r}"
        )
    return {"way": way}


# ---------------------------------------------------------------------------
# 公开接口
# ---------------------------------------------------------------------------


def enrich_visible_payload(payload: dict, env: Any, env_id: str) -> None:
    """Imitation 套件 reset 阶段写入 InsertPeg / MoveCube 的 choice 顶层字段。

    InsertPeg → payload['insertpeg_choice'] = (direction, obj_flag, ...xy字段)
    MoveCube  → payload['movecube_choice']  = {way}
    PatternLock / RouteStick / 非 imitation env_id → no-op
    """
    if env_id not in IMITATION_ENV_IDS:
        return
    base = getattr(env, "unwrapped", env)
    if env_id == "InsertPeg":
        payload[INSERTPEG_CHOICE_KEY] = _to_jsonable(_extract_insertpeg_choice(base))
        return
    if env_id == "MoveCube":
        payload[MOVECUBE_CHOICE_KEY] = _to_jsonable(_extract_movecube_choice(base))
        return
    # PatternLock / RouteStick: imitation 套件内但无 reset-时刻 choice 字段。
    return


def select_planner(env: Any, env_id: str) -> Optional[Any]:
    """对 PatternLock / RouteStick 返回 FailAwarePandaStickMotionPlanningSolver
    实例（joint_vel_limits=0.3）；其他 env_id 一律返回 None，由调用方自行实例化
    默认 arm planner。

    返回 None 不代表错误，而是表示"本 suite 不接管此 env 的 planner 选择"，
    主脚本 ``_create_planner`` 收到 None 后会落到 default arm planner。
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
    "INSERTPEG_CHOICE_KEY",
    "MOVECUBE_CHOICE_KEY",
    "INSERTPEG_DIRECTION_LABELS",
    "INSERTPEG_INSERT_END_LABELS",
    "MOVECUBE_WAYS",
    "enrich_visible_payload",
    "select_planner",
]
