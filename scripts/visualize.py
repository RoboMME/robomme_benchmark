import numpy as np
import matplotlib.pyplot as plt
import torch

def uniform_2d(gen, low_x, high_x, low_y, high_y, device):
    """用给定的 torch.Generator 采样均匀分布的 (x, y)。"""
    u = torch.rand(2, generator=gen, device=device)
    x = low_x + (high_x - low_x) * u[0]
    y = low_y + (high_y - low_y) * u[1]
    return torch.stack([x, y])

def generate_positions_rejection_torch(seed=42, num_cubes=15,
                                       region_center=(-0.1, 0.0),
                                       region_half_size=0.3,
                                       min_gap=0.02,
                                       max_trials=100000,
                                       device="cpu"):
    """
    使用 PyTorch 的随机数和拒绝采样生成 2D 位置（shape: [num_cubes, 2]）
    - seed:        随机种子（每次调用都可复现）
    - device:      "cpu" 或 "cuda"（如果你想在 GPU 上跑随机）
    """
    # 为当前函数创建独立的随机发生器（不污染全局状态）
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))

    cx, cy = float(region_center[0]), float(region_center[1])
    h = float(region_half_size)
    low_x, high_x = cx - h, cx + h
    low_y, high_y = cy - h, cy + h
    min_gap = float(min_gap)

    positions = []  # 存 torch.tensor([x, y], device=device)
    trials = 0

    while len(positions) < num_cubes and trials < max_trials:
        trials += 1
        pos = uniform_2d(gen, low_x, high_x, low_y, high_y, device)

        # 与已有点的最小距离检查（拒绝采样）
        if len(positions) == 0:
            positions.append(pos)
            continue

        # 将已有点堆叠成 [N,2]，计算新点到所有点的欧氏距离并取最小值
        stack = torch.stack(positions, dim=0)  # [N, 2]
        dists = torch.linalg.norm(stack - pos, dim=1)
        if torch.all(dists > min_gap):
            positions.append(pos)

    if len(positions) < num_cubes:
        raise RuntimeError(f"无法在 {max_trials} 次尝试中采样到足够的点（得到 {len(positions)} 个）")

    # 返回到 CPU + numpy，便于 matplotlib 画图
    return torch.stack(positions, dim=0).detach().cpu().numpy()


if __name__ == "__main__":
    # 区域边界（用于画方框）
    region_center = np.array([-0.1, 0.0], dtype=float)
    region_half_size = 0.3
    square = np.array([
        region_center + [-region_half_size, -region_half_size],
        region_center + [ region_half_size, -region_half_size],
        region_center + [ region_half_size,  region_half_size],
        region_center + [-region_half_size,  region_half_size],
        region_center + [-region_half_size, -region_half_size],
    ], dtype=float)

    # 画 2x5 子图，展示 seed=1..10
    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    for idx, seed in enumerate(range(1, 11)):
        ax = axes[idx // 5, idx % 5]
        # 如果你想用 GPU 采样，把 device="cuda"（需有可用 CUDA）
        positions = generate_positions_rejection_torch(
            seed=seed, num_cubes=15,
            region_center=(-0.1, 0.0),
            region_half_size=0.3,
            min_gap=0.02,
            device="cpu"
        )

        ax.scatter(positions[:, 0], positions[:, 1], s=50)
        ax.plot(square[:, 0], square[:, 1], linestyle="--")
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(f"seed={seed}")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle("Random Cube Positions with Rejection Sampling (torch, seed=1~10)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    # 如需保存：
    # plt.savefig("all_seeds_torch.png", dpi=300)
