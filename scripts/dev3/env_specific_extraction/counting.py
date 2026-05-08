"""Counting 套件 reset 时刻 env-specific 钩子（当前为 no-op）。

适用 env：BinFill / PickXtimes / SwingXtimes / StopCube。

为什么文件存在：
- 与 permanence.py / reference.py / imitation.py 保持对称的 enrich_visible_payload
  入口，让 Env-rollout-parallel-segmentation.py 能用同一种盲调形态调用 4 个 suite。
- counting 套件目前在 reset 阶段没有需要写到 visible_objects.json 顶层的
  额外字段（target / board / cubes 等已经体现在通用 'objects' 列表里），
  所以 enrich 函数对 counting env 是显式 no-op。

如果未来 counting 套件需要 reset 时刻的额外元数据（例如 target count），
这是它的入口位置；不要再回到主脚本里加 if env_id == ... 分支。
"""
from __future__ import annotations

from typing import Any

COUNTING_ENV_IDS: frozenset[str] = frozenset(
    {"BinFill", "PickXtimes", "SwingXtimes", "StopCube"}
)


def enrich_visible_payload(payload: dict, env: Any, env_id: str) -> None:
    """Counting 套件 reset 阶段无需向 visible_objects.json 写 env-specific
    字段；保留同签名空函数与其他 3 个 suite 对称。

    非 counting env_id 与 counting env_id 都是 no-op。
    """
    if env_id not in COUNTING_ENV_IDS:
        return
    # Counting 套件本身也无需 enrich：明确 no-op，无副作用。
    return


__all__ = [
    "COUNTING_ENV_IDS",
    "enrich_visible_payload",
]
