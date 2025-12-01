import json
from collections import Counter, defaultdict

import numpy as np
import matplotlib.pyplot as plt

# ===== 1. 把你那段 JSON 粘到这里 =====
# 直接把 [ { ... }, { ... }, ... ] 整段替换掉下面这个空列表
# data = [
#     # 在这里粘贴你发给我的那整段 JSON 列表
# ]

#如果你是从文件里读，可以用：
with open("/data/hongzefu/historybench-v3.9-fixtask/historybench/HistoryBench_env/target_selection.json", "r") as f:
    data = json.load(f)

# ===== 2. 统计 (which_in_subset, num_targets_to_pick) 频数 =====
pair_counts = Counter()
nums_for_group = defaultdict(Counter)  # nums_for_group[num][which] = count

for d in data:
    which = int(d["which_in_subset"])
    num = int(d["num_targets_to_pick"])
    pair_counts[(which, num)] += 1
    nums_for_group[num][which] += 1

# 所有取值的排序（x 轴：num_targets_to_pick；y 轴：which_in_subset）
all_nums = sorted(nums_for_group.keys())                # 列
all_which = sorted({d["which_in_subset"] for d in data})  # 行

# ===== 3. 计算条件概率 P(which_in_subset | num_targets_to_pick) =====
prob_matrix = np.zeros((len(all_which), len(all_nums)), dtype=float)

for j, num in enumerate(all_nums):
    which_counter = nums_for_group[num]
    total = sum(which_counter.values())
    if total == 0:
        continue
    for i, which in enumerate(all_which):
        prob_matrix[i, j] = which_counter[which] / total

print("all_nums (x 轴):", all_nums)
print("all_which (y 轴):", all_which)
print("条件概率矩阵 P(which_in_subset | num_targets_to_pick):")
print(prob_matrix)

# ===== 4. 画条件概率 Heatmap =====
fig, ax = plt.subplots(figsize=(6, 5))

# 用 imshow 画热力图
im = ax.imshow(prob_matrix, aspect="auto")

# 坐标轴标签
ax.set_xticks(np.arange(len(all_nums)))
ax.set_xticklabels(all_nums)
ax.set_yticks(np.arange(len(all_which)))
ax.set_yticklabels(all_which)

ax.set_xlabel("num_targets_to_pick")
ax.set_ylabel("which_in_subset")
ax.set_title("P(which_in_subset | num_targets_to_pick)")

# 每个格子写上数值（保留两位小数）
for i in range(len(all_which)):
    for j in range(len(all_nums)):
        value = prob_matrix[i, j]
        ax.text(
            j,
            i,
            f"{value:.2f}",
            ha="center",
            va="center"
        )

# 颜色条
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Probability")

plt.tight_layout()
plt.show()
