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

import math
import re
from typing import Any, Optional

import numpy as np

IMITATION_ENV_IDS: frozenset[str] = frozenset(
    {"MoveCube", "InsertPeg", "PatternLock", "RouteStick"}
)
STICK_ENV_IDS: frozenset[str] = frozenset({"PatternLock", "RouteStick"})

# visible_objects.json 顶层字段名
INSERTPEG_CHOICE_KEY = "insertpeg_choice"
MOVECUBE_CHOICE_KEY = "movecube_choice"
PATTERNLOCK_WALK_KEY = "patternlock_walk_path"
ROUTESTICK_WALK_KEY = "routestick_walk_path"

# 标签：与 InsertPeg / MoveCube env 内部的 raw 值映射对齐
INSERTPEG_DIRECTION_LABELS = {-1: "left", 1: "right"}
# obj_flag = -1 时 grasp head / insert tail；obj_flag = +1 时 grasp tail / insert head
# 用 insert_end 表达「选中那一端插入孔」，而非 grasp_end，避免 head/tail 二义。
INSERTPEG_INSERT_END_LABELS = {-1: "tail", 1: "head"}
MOVECUBE_WAYS = ("peg_push", "gripper_push", "grasp_putdown")

# PatternLock 8-direction labels —— 与 robomme_env/utils/subgoal_evaluate_func.py
# 中 direction(curr, prev) 默认 8 方向语义一致：forward = +x, left = +y。
PATTERNLOCK_DIRECTION_LABELS = (
    "forward",
    "backward",
    "left",
    "right",
    "forward-left",
    "forward-right",
    "backward-left",
    "backward-right",
)

# RouteStick 4 组合 = stick_side ("left"/"right") + "+" + swing_direction
# ("clockwise"/"counterclockwise")。task_name 模板见 RouteStick.py:418, 454。
ROUTESTICK_DIRECTION_COMBOS = (
    "left+clockwise",
    "left+counterclockwise",
    "right+clockwise",
    "right+counterclockwise",
)

# PatternLock.py:222 / RouteStick.py:244 actor 命名模板 "target_<idx>".
_TARGET_NAME_RE = re.compile(r"^target_(\d+)$")


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


def _parse_target_index(actor: Any) -> int:
    """从 actor.name (例如 'target_5') 解析索引。reset 后命名固定，必须能解析；
    解析失败一律 raise（缺位 = 上游 env 重命名/破坏 schema，必须暴露）。"""
    name = getattr(actor, "name", "")
    match = _TARGET_NAME_RE.match(str(name))
    if match is None:
        raise ValueError(
            f"actor name {name!r} does not match 'target_<idx>' pattern"
        )
    return int(match.group(1))


# 8 方向单位向量。顺序与 PATTERNLOCK_DIRECTION_LABELS 严格对齐，便于按 argmax
# 直接 index 取标签。语义：forward = +x, left = +y（与 subgoal_evaluate_func.py
# direction() 默认 8 方向一致）。
_DIAG = 1.0 / math.sqrt(2.0)
_PATTERNLOCK_DIRECTION_VECTORS = (
    (1.0, 0.0),               # forward
    (-1.0, 0.0),              # backward
    (0.0, 1.0),               # left
    (0.0, -1.0),              # right
    (_DIAG, _DIAG),           # forward-left
    (_DIAG, -_DIAG),          # forward-right
    (-_DIAG, _DIAG),          # backward-left
    (-_DIAG, -_DIAG),         # backward-right
)


def _compute_relative_direction_8(
    curr_xy: list[float], prev_xy: list[float]
) -> str:
    """与 robomme_env/utils/subgoal_evaluate_func.py:direction() 默认 8 方向语义
    完全一致：dx,dy 归一化后与 8 单位向量逐个 dot，取 max。

    PatternLock reset 阶段的两个相邻 grid 按钮 xy 不可能完全重合，norm < 1e-8
    属于异常路径，直接 raise。
    """
    dx = float(curr_xy[0]) - float(prev_xy[0])
    dy = float(curr_xy[1]) - float(prev_xy[1])
    norm = math.hypot(dx, dy)
    if norm < 1e-8:
        raise ValueError(
            "two adjacent grid buttons coincide in xy plane; cannot compute "
            "relative direction"
        )
    dx /= norm
    dy /= norm
    best_idx = max(
        range(len(_PATTERNLOCK_DIRECTION_VECTORS)),
        key=lambda i: dx * _PATTERNLOCK_DIRECTION_VECTORS[i][0]
        + dy * _PATTERNLOCK_DIRECTION_VECTORS[i][1],
    )
    return PATTERNLOCK_DIRECTION_LABELS[best_idx]


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
# PatternLock / RouteStick 的 walk-path 提取
# ---------------------------------------------------------------------------


def _extract_patternlock_walk(base: Any) -> dict:
    """从 PatternLock env.unwrapped 提取 reset 阶段的游走路径。

    依赖 env 属性（PatternLock.py 中已 set 完）：
      - selected_buttons         (line 307)：N 个 actor，N >= 2
      - buttons_grid             (line 240)：grid_dim ** 2 个 actor
      - difficulty + configs     (line 85-89, 143-152)：grid_dim ∈ {3, 4, 5}

    relative_directions 用 _compute_relative_direction_8 计算，与 env 内部使用
    的 direction(curr, prev) 默认 8 方向语义一致。
    """
    selected = getattr(base, "selected_buttons", None)
    grid = getattr(base, "buttons_grid", None)
    difficulty = getattr(base, "difficulty", None)
    configs = getattr(base, "configs", None)
    if not selected:
        raise ValueError("PatternLock.selected_buttons is empty after reset")
    if not grid:
        raise ValueError("PatternLock.buttons_grid is empty after reset")
    if difficulty not in ("easy", "medium", "hard"):
        raise ValueError(
            f"PatternLock.difficulty={difficulty!r} not in (easy/medium/hard)"
        )
    if not isinstance(configs, dict) or difficulty not in configs:
        raise ValueError(
            f"PatternLock.configs missing entry for difficulty {difficulty!r}"
        )
    grid_dim = int(configs[difficulty]["grid"])

    path_indices = [_parse_target_index(actor) for actor in selected]
    path_names = [str(getattr(actor, "name", "")) for actor in selected]
    path_xy = [_actor_xy(actor) for actor in selected]
    all_xy = [_actor_xy(actor) for actor in grid]

    rels: list[str] = []
    for prev, curr in zip(path_xy[:-1], path_xy[1:]):
        rels.append(_compute_relative_direction_8(curr, prev))

    return {
        "grid_rows": grid_dim,
        "grid_cols": grid_dim,
        "path_button_indices": path_indices,
        "path_button_names": path_names,
        "path_xy": path_xy,
        "all_button_xy": all_xy,
        "relative_directions": rels,
    }


def _extract_routestick_walk(base: Any) -> dict:
    """从 RouteStick env.unwrapped 提取 reset 阶段的游走路径。

    依赖 env 属性（RouteStick.py 中已 set 完）：
      - selected_buttons         (line 371)：N 个 actor，N >= 2
      - buttons_grid             (line 265)：9 个 actor（1×9 line）
      - swing_directions         (line 392-399)：长度 = N-1，∈ {clockwise, counterclockwise}
      - target_cube_indices      (line 275)：list[int]，固定 [1, 3, 5, 7]
      - cubes_on_targets         (line 344)：4 个 stick actors

    stick_side 复刻 RouteStick._stick_side(curr, prev) 的逻辑：
      "left" if curr.y > prev.y else "right"
    （RouteStick.py:373-389 中 _stick_side 是 nested function，本模块独立实现以
    保持 imitation.py 不依赖 env 内部命名细节。）
    """
    selected = getattr(base, "selected_buttons", None)
    grid = getattr(base, "buttons_grid", None)
    swing_dirs = getattr(base, "swing_directions", None)
    stick_actors = getattr(base, "cubes_on_targets", None)
    stick_indices = getattr(base, "target_cube_indices", None)
    if not selected:
        raise ValueError("RouteStick.selected_buttons is empty after reset")
    if not grid:
        raise ValueError("RouteStick.buttons_grid is empty after reset")
    if swing_dirs is None:
        raise ValueError("RouteStick.swing_directions is None after reset")
    if len(swing_dirs) != len(selected) - 1:
        raise ValueError(
            f"RouteStick.swing_directions length mismatch: got "
            f"{len(swing_dirs)}, expected {len(selected) - 1}"
        )
    if not stick_actors or not stick_indices:
        raise ValueError(
            "RouteStick.cubes_on_targets / target_cube_indices empty"
        )
    if len(stick_actors) != len(stick_indices):
        raise ValueError(
            f"RouteStick stick actor count mismatch: "
            f"len(cubes_on_targets)={len(stick_actors)} vs "
            f"len(target_cube_indices)={len(stick_indices)}"
        )

    path_indices = [_parse_target_index(actor) for actor in selected]
    path_names = [str(getattr(actor, "name", "")) for actor in selected]
    path_xy = [_actor_xy(actor) for actor in selected]
    all_xy = [_actor_xy(actor) for actor in grid]
    stick_xy = [_actor_xy(actor) for actor in stick_actors]

    sides: list[str] = []
    combos: list[str] = []
    swing_normalized: list[str] = []
    for i, raw_swing in enumerate(swing_dirs):
        sw = str(raw_swing).lower()
        if sw not in ("clockwise", "counterclockwise"):
            raise ValueError(
                f"RouteStick.swing_directions[{i}]={raw_swing!r} "
                f"not in (clockwise, counterclockwise)"
            )
        prev_y = path_xy[i][1]
        curr_y = path_xy[i + 1][1]
        side = "left" if curr_y > prev_y else "right"
        sides.append(side)
        combos.append(f"{side}+{sw}")
        swing_normalized.append(sw)

    return {
        "path_button_indices": path_indices,
        "path_button_names": path_names,
        "path_xy": path_xy,
        "all_button_xy": all_xy,
        "stick_indices": [int(i) for i in stick_indices],
        "stick_xy": stick_xy,
        "swing_directions": swing_normalized,
        "stick_sides": sides,
        "relative_directions": combos,
    }


# ---------------------------------------------------------------------------
# 公开接口
# ---------------------------------------------------------------------------


def enrich_visible_payload(payload: dict, env: Any, env_id: str) -> None:
    """Imitation 套件 reset 阶段写入 4 个 env 的顶层字段。

    InsertPeg   → payload['insertpeg_choice']    = (direction, obj_flag, ...xy字段)
    MoveCube    → payload['movecube_choice']     = {way}
    PatternLock → payload['patternlock_walk_path'] = {grid + path nodes + 8 方向 rels}
    RouteStick  → payload['routestick_walk_path'] = {path nodes + sticks + 4 组合 rels}
    非 imitation env_id → no-op
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
    if env_id == "PatternLock":
        payload[PATTERNLOCK_WALK_KEY] = _to_jsonable(_extract_patternlock_walk(base))
        return
    if env_id == "RouteStick":
        payload[ROUTESTICK_WALK_KEY] = _to_jsonable(_extract_routestick_walk(base))
        return
    # IMITATION_ENV_IDS = 4 元组，前面分支已穷尽；落到这里说明集合内部不一致。
    raise AssertionError(f"unhandled imitation env_id {env_id!r}")


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
    "PATTERNLOCK_WALK_KEY",
    "ROUTESTICK_WALK_KEY",
    "INSERTPEG_DIRECTION_LABELS",
    "INSERTPEG_INSERT_END_LABELS",
    "MOVECUBE_WAYS",
    "PATTERNLOCK_DIRECTION_LABELS",
    "ROUTESTICK_DIRECTION_COMBOS",
    "enrich_visible_payload",
    "select_planner",
]
