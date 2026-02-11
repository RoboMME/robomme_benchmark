import torch
import numpy as np  # 用于绘图和数据结构
import matplotlib.pyplot as plt


def generate_dynamic_walk(indices, steps=50, start_idx=None, allow_backtracking=True,
                          generator=None, plot=False):
    """
    生成随机游走轨迹 (支持 PyTorch Generator)。

    Args:
        indices (list): 离散的位置点 (如 [0, 2, 4...])
        steps (int): 总步数
        start_idx (int): 起始索引，None 则随机
        allow_backtracking (bool): 是否允许立即回头
        generator (torch.Generator): 用于控制随机种子的 PyTorch 生成器
        plot (bool): 是否绘图
    """

    # 1. 初始化
    if start_idx is None:
        # [修改点] 使用 torch 生成随机起点
        # randint 返回的是 tensor，需要 .item() 转为 python int
        start_idx = torch.randint(0, len(indices), (1,), generator=generator).item()

    start_val = indices[start_idx]
    print(f"   -> 随机起点索引: {start_idx} (对应值: {start_val})")

    history_idxs = [start_idx]

    # 2. 生成循环
    for _ in range(steps):
        current_idx = history_idxs[-1]
        prev_idx = history_idxs[-2] if len(history_idxs) > 1 else None

        # 找出所有物理上可达的邻居
        neighbors = []
        if current_idx > 0:
            neighbors.append(current_idx - 1)
        if current_idx < len(indices) - 1:
            neighbors.append(current_idx + 1)

        # --- 核心逻辑：回溯过滤 ---
        candidates = []
        if allow_backtracking:
            candidates = neighbors
        else:
            # 不允许回头
            if prev_idx is not None:
                filtered = [n for n in neighbors if n != prev_idx]
                candidates = filtered if filtered else neighbors  # 如果没路了必须回头
            else:
                candidates = neighbors

        # [修改点] 使用 torch 从 candidates 中随机选择
        # 原理: 随机生成一个 0 到 len(candidates)-1 的索引
        rand_choice_idx = torch.randint(0, len(candidates), (1,), generator=generator).item()
        next_idx = candidates[rand_choice_idx]

        history_idxs.append(next_idx)

    # 映射回真实值
    path_values = [indices[i] for i in history_idxs]

    # 3. 可视化
    if plot:
        plt.figure(figsize=(12, 5))
        time_axis = range(len(path_values))
        color = '#1f77b4' if allow_backtracking else '#ff7f0e'
        mode_str = "With Backtracking" if allow_backtracking else "No Backtracking (Inertia)"

        plt.step(time_axis, path_values, where='post', marker='o', markersize=5,
                 linestyle='-', color=color, alpha=0.8, linewidth=2)
        plt.yticks(indices)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.title(f'Random Walk (Torch Seeded): {mode_str}\nStart Value: {start_val}', fontsize=12)
        plt.xlabel('Time Step')
        plt.ylabel('Button Value')

        # 标起点终点
        plt.scatter(0, path_values[0], c='green', s=150, label='Start', zorder=5, edgecolors='white')
        plt.scatter(steps, path_values[-1], c='red', marker='X', s=150, label='End', zorder=5, edgecolors='white')
        plt.legend()
        plt.tight_layout()
        plt.show()

    return path_values


# # --- 对比测试 (带种子) ---
# button_indices = [0, 2, 4, 6, 8]

# # 创建一个 Generator 并设定种子 (保证结果可复现)
# seed = 42
# rng = torch.Generator()
# rng.manual_seed(seed)

# print(f"--- 测试开始 (Seed: {seed}) ---")

# # 1. 开启回溯
# print("方案1: 允许回溯")
# traj_1 = generate_dynamic_walk(button_indices, steps=30, allow_backtracking=True, generator=rng)

# # 2. 关闭回溯 (注意：因为共用同一个 generator，这里的随机数序列是接续上一次调用的)
# print("\n方案2: 禁止回溯 (惯性模式)")
# traj_2 = generate_dynamic_walk(button_indices, steps=30, allow_backtracking=False, generator=rng)