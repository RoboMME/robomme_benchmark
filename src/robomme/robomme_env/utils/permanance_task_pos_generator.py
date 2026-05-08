"""Permanence 任务的位置生成 utility.

四个 permanence task env（VideoUnmask / VideoUnmaskSwap / ButtonUnmask /
ButtonUnmaskSwap）都通过 `permanance_task_pos_generator` 拿位置，env 内部仅
负责把返回的 (x, y) 与索引数据变成 actor。

生成机制使用 torch.Generator + manual_seed，与原 permanence 任务保持同一
RNG stream 风格（KNUTH_HASH 派生独立 stream）。

两个距离类超参数严格对齐 permanence 任务：
- MIN_CENTER_DIST = 0.135：对应 VideoUnmask/ButtonUnmask 的 Poisson disk 间隔
  r_sep = (bin_half_size + min_gap) * 2 = (0.0275 + 0.04) * 2，配合 panda
  cube_half_size=0.02。
- BUTTON_MIN_CENTER_DIST = 0.15：对应 ButtonUnmaskSwap 中两按钮最坏间隔——
  名义 y-spacing 0.2 减去左右各自的 randomize_range 0.025 = 0.15。

每个 episode 流程：
1. 在 [-XY_HALF, XY_HALF]^2 上均匀采样 n_bins 个 (x, y)，最小中心距
   MIN_CENTER_DIST（rejection sampling）；落入 button strip
   [BUTTON_X_MIN, BUTTON_X_MAX] x [BUTTON_Y_MIN, BUTTON_Y_MAX] 的样本被拒绝，
   保证 cube/bin 不与 button 位置重叠。
2. 随机洗牌 (red, green, blue, "no_cube" * (n_bins - 3)) 多重集，给每个 bin
   分配颜色（torch.randperm）。
3. 用独立 pickup_generator（seed * KNUTH_HASH + 1）从 3 个有色 bin 中无放回挑出
   n_pickups 个作为目标。
4. 用独立 swap_generator（seed * KNUTH_HASH + 2）循环 n_swaps 次抽取 (i, j) 对
   作为 swap 对象。
5. 仅当 n_buttons > 0：用独立 button_generator（seed * KNUTH_HASH + 3）在
   button strip 上均匀采样 n_buttons 个位置。
"""

from __future__ import annotations

import torch

# === 间隔/范围超参数（permanance_task_pos_generator 不接受这些；固定模块级值）===
XY_HALF = 0.2                  # bin 生成 xy 范围 [-XY_HALF, XY_HALF]^2
MIN_CENTER_DIST = 0.135        # bin 之间的最小中心距
BUTTON_X_MIN = -0.25           # button strip x 下界
BUTTON_X_MAX = -0.15           # button strip x 上界
BUTTON_Y_MIN = -0.2            # button strip y 下界
BUTTON_Y_MAX = 0.2             # button strip y 上界
BUTTON_MIN_CENTER_DIST = 0.15  # button 之间的最小中心距

# === 调色板（与 demo 共享）===
COLORS: tuple[str, ...] = ("red", "green", "blue", "no_cube")


def sample_bin_positions(
    generator: torch.Generator,
    n_bins: int,
    max_attempts: int = 2000,
) -> list[tuple[float, float]]:
    """在 [-XY_HALF, XY_HALF]^2 上 rejection-sample 出 n_bins 个 (x, y).

    位置必须满足两个约束：互相之间最小中心距 MIN_CENTER_DIST；不落入
    button strip [BUTTON_X_MIN, BUTTON_X_MAX] x [BUTTON_Y_MIN, BUTTON_Y_MAX]。
    超过 max_attempts 仍无法填满则抛 RuntimeError。
    """
    pts: list[tuple[float, float]] = []
    min_dist_sq = MIN_CENTER_DIST * MIN_CENTER_DIST
    for _ in range(max_attempts):
        if len(pts) == n_bins:
            return pts
        u = torch.rand(2, generator=generator)
        cx = float((u[0].item() - 0.5) * 2.0 * XY_HALF)
        cy = float((u[1].item() - 0.5) * 2.0 * XY_HALF)
        if (BUTTON_X_MIN <= cx <= BUTTON_X_MAX
                and BUTTON_Y_MIN <= cy <= BUTTON_Y_MAX):
            continue
        if all((cx - x) ** 2 + (cy - y) ** 2 >= min_dist_sq for x, y in pts):
            pts.append((cx, cy))
    raise RuntimeError(
        f"failed to place {n_bins} bins after {max_attempts} attempts "
        f"(xy_half={XY_HALF}, min_dist={MIN_CENTER_DIST})"
    )


def sample_button_positions(
    generator: torch.Generator,
    n_buttons: int,
    max_attempts: int = 2000,
) -> list[tuple[float, float]]:
    """在 button strip 矩形内 rejection-sample 出 n_buttons 个 (x, y).

    最小中心距 BUTTON_MIN_CENTER_DIST。超过 max_attempts 抛 RuntimeError。
    """
    pts: list[tuple[float, float]] = []
    min_dist_sq = BUTTON_MIN_CENTER_DIST * BUTTON_MIN_CENTER_DIST
    x_span = BUTTON_X_MAX - BUTTON_X_MIN
    y_span = BUTTON_Y_MAX - BUTTON_Y_MIN
    for _ in range(max_attempts):
        if len(pts) == n_buttons:
            return pts
        u = torch.rand(2, generator=generator)
        cx = float(BUTTON_X_MIN + u[0].item() * x_span)
        cy = float(BUTTON_Y_MIN + u[1].item() * y_span)
        if all((cx - x) ** 2 + (cy - y) ** 2 >= min_dist_sq for x, y in pts):
            pts.append((cx, cy))
    raise RuntimeError(
        f"failed to place {n_buttons} buttons after {max_attempts} attempts "
        f"(x_range=[{BUTTON_X_MIN}, {BUTTON_X_MAX}], "
        f"y_range=[{BUTTON_Y_MIN}, {BUTTON_Y_MAX}], "
        f"min_dist={BUTTON_MIN_CENTER_DIST})"
    )


def permanance_task_pos_generator(
    n_bins: int,
    n_swaps: int,
    n_buttons: int,
    n_pickups: int,
    seed: int,
) -> dict:
    """生成单个 episode 的 permanence 任务位置.

    入参：
        n_bins:    bin 数量（≥ N_COLORED_CUBES = 3）
        n_swaps:   生成的 swap 对数量
        n_buttons: button 数量；为 0 则不采样 button
        n_pickups: 从有色 bin 中挑选的 pickup 数量
        seed:      episode 种子

    成功返回 dict：
        {
            "bin_positions":   list[(x, y)]，长度 n_bins
            "bin_colors":      list[str]，每个 bin 的 cube 颜色（含 "no_cube"）
            "pickup_map":      dict[bin_index -> pickup_order(0,1,...)]
            "swap_pairs":      list[(i, j)]，长度 n_swaps，按 i<j 排序
            "button_positions": list[(x, y)]，长度 n_buttons
        }

    rejection sampling 失败返回 {"fail": "<error message>"}（无其他字段）；
    调用方用 `if "fail" in result:` 判定。
    """
    # 任务设计内部常量（不暴露到模块层）
    N_COLORED_CUBES = 3       # red / green / blue（与 COLORS 调色板对齐）
    KNUTH_HASH = 2654435761   # RNG stream offset 共用（Knuth multiplicative hash）

    generator = torch.Generator()
    generator.manual_seed(seed)
    pickup_generator = torch.Generator()
    pickup_generator.manual_seed(seed * KNUTH_HASH + 1)
    swap_generator = torch.Generator()
    swap_generator.manual_seed(seed * KNUTH_HASH + 2)

    try:
        positions = sample_bin_positions(generator, n_bins)
    except RuntimeError as exc:
        return {"fail": str(exc)}

    color_pool = list(COLORS[:N_COLORED_CUBES]) + ["no_cube"] * (n_bins - N_COLORED_CUBES)
    shuffle_indices = torch.randperm(n_bins, generator=generator).tolist()
    bin_colors = [color_pool[i] for i in shuffle_indices]

    target_indices = [i for i, c in enumerate(bin_colors) if c != "no_cube"]
    sel = torch.randperm(len(target_indices), generator=pickup_generator)[:n_pickups].tolist()
    chosen_bin_indices = [target_indices[k] for k in sel]
    pickup_map = {chosen_bin_indices[k]: k for k in range(n_pickups)}

    swap_pairs: list[tuple[int, int]] = []
    for _ in range(n_swaps):
        idx = torch.randperm(n_bins, generator=swap_generator)[:2].tolist()
        a, b = sorted((idx[0], idx[1]))
        swap_pairs.append((a, b))

    button_positions: list[tuple[float, float]] = []
    if n_buttons > 0:
        button_generator = torch.Generator()
        button_generator.manual_seed(seed * KNUTH_HASH + 3)
        try:
            button_positions = sample_button_positions(button_generator, n_buttons)
        except RuntimeError as exc:
            return {"fail": str(exc)}

    return {
        "bin_positions": positions,
        "bin_colors": bin_colors,
        "pickup_map": pickup_map,
        "swap_pairs": swap_pairs,
        "button_positions": button_positions,
    }


__all__ = [
    "permanance_task_pos_generator",
    "sample_bin_positions",
    "sample_button_positions",
    "XY_HALF",
    "MIN_CENTER_DIST",
    "BUTTON_X_MIN",
    "BUTTON_X_MAX",
    "BUTTON_Y_MIN",
    "BUTTON_Y_MAX",
    "BUTTON_MIN_CENTER_DIST",
    "COLORS",
]
