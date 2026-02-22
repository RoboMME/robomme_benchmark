import torch
import numpy as np  # Used for plotting and data structures
import matplotlib.pyplot as plt

from ...logging_utils import logger


def generate_dynamic_walk(indices, steps=50, start_idx=None, allow_backtracking=True,
                          generator=None, plot=False):
    """
    Generate random walk trajectory (supports PyTorch Generator).

    Args:
        indices (list): Discrete position points (e.g., [0, 2, 4...])
        steps (int): Total steps
        start_idx (int): Start index, random if None
        allow_backtracking (bool): Whether to allow immediate backtracking
        generator (torch.Generator): PyTorch generator for controlling random seed
        plot (bool): Whether to plot
    """

    # 1. Initialization
    if start_idx is None:
        # [Modification] Use torch to generate random start point
        # randint returns tensor, need .item() to convert to python int
        start_idx = torch.randint(0, len(indices), (1,), generator=generator).item()

    start_val = indices[start_idx]
    logger.debug(f"   -> Random start index: {start_idx} (Value: {start_val})")

    history_idxs = [start_idx]

    # 2. Generation loop
    for _ in range(steps):
        current_idx = history_idxs[-1]
        prev_idx = history_idxs[-2] if len(history_idxs) > 1 else None

        # Find all physically reachable neighbors
        neighbors = []
        if current_idx > 0:
            neighbors.append(current_idx - 1)
        if current_idx < len(indices) - 1:
            neighbors.append(current_idx + 1)

        # --- Core logic: Backtracking filter ---
        candidates = []
        if allow_backtracking:
            candidates = neighbors
        else:
            # Backtracking not allowed
            if prev_idx is not None:
                filtered = [n for n in neighbors if n != prev_idx]
                candidates = filtered if filtered else neighbors  # Must backtrack if no way forward
            else:
                candidates = neighbors

        # [Modification] Use torch to select randomly from candidates
        # Principle: Randomly generate an index from 0 to len(candidates)-1
        rand_choice_idx = torch.randint(0, len(candidates), (1,), generator=generator).item()
        next_idx = candidates[rand_choice_idx]

        history_idxs.append(next_idx)

    # Map back to real values
    path_values = [indices[i] for i in history_idxs]

    # 3. Visualization
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

        # Mark start and end points
        plt.scatter(0, path_values[0], c='green', s=150, label='Start', zorder=5, edgecolors='white')
        plt.scatter(steps, path_values[-1], c='red', marker='X', s=150, label='End', zorder=5, edgecolors='white')
        plt.legend()
        plt.tight_layout()
        plt.show()

    return path_values


# # --- Comparison test (with seed) ---
# button_indices = [0, 2, 4, 6, 8]

# # Create a Generator and set seed (guarantee reproducible results)
# seed = 42
# rng = torch.Generator()
# rng.manual_seed(seed)

# print(f"--- Test Start (Seed: {seed}) ---")

# # 1. Enable backtracking
# print("Scheme 1: Allow backtracking")
# traj_1 = generate_dynamic_walk(button_indices, steps=30, allow_backtracking=True, generator=rng)

# # 2. Disable backtracking (Note: using same generator, random sequence continues from last call)
# print("\nScheme 2: Forbid backtracking (Inertia mode)")
# traj_2 = generate_dynamic_walk(button_indices, steps=30, allow_backtracking=False, generator=rng)