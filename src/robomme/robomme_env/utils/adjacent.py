import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import numpy as np
import torch

from ...logging_utils import logger

def grid_adjacency(R=3, C=3, diagonals=False, by_index=True):
    """
    R, C: rows and columns
    diagonals: False=4-connectivity; True=8-connectivity
    by_index: True returns with index (0..R*C-1) as key; False returns with coordinate (r,c) as key
    """
    # 4-connectivity directions
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    # 8-connectivity adds four diagonal directions
    if diagonals:
        dirs += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    adj = {}
    for r in range(R):
        for c in range(C):
            nbrs = []
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if 0 <= nr < R and 0 <= nc < C:
                    nbrs.append((nr, nc))

            key = r * C + c if by_index else (r, c)
            if by_index:
                adj[key] = [nr * C + nc for (nr, nc) in nbrs]
            else:
                adj[key] = nbrs
    return adj


def dfs_path(adj, start, target, generator=None, blocked_nodes=None):
    """
    Execute DFS to find path from start to target
    Returns visit_order, path and edges used during search

    Args:
        adj: adjacency list
        start: start node
        target: target node
        generator: torch.Generator for random neighbor selection
        blocked_nodes: list of node indices to avoid (cannot pass through these nodes)

    Returns:
        visit_order: list of all nodes visited in order
        path: final path from start to target
        edges_used: edges traversed during search
    """
    visited = set()
    visit_order = []  # Track all nodes visited in order
    path = []
    edges_used = []
    found = False

    # Convert blocked_nodes to set for O(1) lookup
    if blocked_nodes is None:
        blocked_nodes = set()
    else:
        blocked_nodes = set(blocked_nodes)

    def dfs_helper(node, current_path):
        nonlocal found
        if found:
            return

        visited.add(node)
        visit_order.append(node)  # Record visit order
        current_path.append(node)

        # Found target node
        if node == target:
            path.extend(current_path)
            found = True
            return

        # Get neighbors and optionally shuffle them
        neighbors = adj[node].copy()
        if generator is not None:
            # Use torch to randomly permute neighbors
            neighbors_tensor = torch.tensor(neighbors)
            perm = torch.randperm(len(neighbors), generator=generator)
            neighbors = neighbors_tensor[perm].tolist()

        # Continue searching
        for neighbor in neighbors:
            # Skip blocked nodes, already visited nodes
            if neighbor not in visited and neighbor not in blocked_nodes and not found:
                edges_used.append((node, neighbor))
                dfs_helper(neighbor, current_path)
                if found:
                    return
                current_path.pop()

    dfs_helper(start, [])
    return visit_order, path, edges_used


def index_to_coord(idx, C):
    """Convert index to coordinate (row, col)"""
    return idx // C, idx % C


def find_path_0_to_8(start, target, R=3, C=3, diagonals=False, generator=None, blocked_nodes=None):
    """Find path from start to target using DFS

    Args:
        start: start node
        target: target node
        R: number of rows
        C: number of columns
        diagonals: whether to use diagonal connections
        generator: torch.Generator for random neighbor selection
        blocked_nodes: list of node indices to avoid (cannot pass through these nodes)

    Returns:
        path: list of nodes in the final path from start to target (first return value)
        visit_order: list of all nodes visited in order during DFS
        edges_used: edges traversed during search
        adj: adjacency list
    """
    # Generate adjacency list
    adj = grid_adjacency(R, C, diagonals=diagonals, by_index=True)

    # Execute DFS to find path
    visit_order, path, edges_used = dfs_path(adj, start, target, generator=generator, blocked_nodes=blocked_nodes)

    if not path:
        logger.debug(f"❌ Cannot find path from {start} to {target}!")
        return None, None, None, None

    # Only print the path
    logger.debug(f"Path: {' → '.join(map(str, path))}")

    return path, visit_order, edges_used, adj


def run_path_generation(run_id, generator, segments, R=3, C=7, backtrack_enable=True):
    """
    Run a single path generation with segments

    Args:
        run_id: Run identifier for printing
        generator: torch.Generator for reproducible randomness
        segments: List of segment tuples, each as (start, end, blocked_nodes)
                  e.g., [(7, 9, [8, 3, 10, 17]), (9, 11, [1, 8, 15]), ...]
        R: Number of rows in grid
        C: Number of columns in grid
        backtrack_enable: Whether random backtracking passes are allowed

    Returns:
        combined_path: Complete path through all waypoints
        combined_edges: All edges used in the path
        adj: Adjacency list of the grid
        start_node: Starting node (first segment's start)
        end_node: Ending node (last segment's end)
        Returns (None, None, None, None, None) if failed
    """
    # Extract waypoints for display
    waypoints = [segments[0][0]]  # Start with first segment's start
    for seg in segments:
        waypoints.append(seg[1])  # Add each segment's end

    logger.debug(f"\n{'=' * 60}")
    logger.debug(f"Run {run_id}: Finding path through waypoints {' → '.join(map(str, waypoints))}")
    logger.debug(f"{'=' * 60}")

    all_paths = []
    all_edges = []
    adj = None

    for seg_idx, (seg_start, seg_target, seg_blocked) in enumerate(segments):

        # Randomly decide whether to use backtracking for this segment
        # Use torch generator for reproducible randomness
        do_backtrack = backtrack_enable and (torch.rand(1, generator=generator).item() > 0.5)

        # Print segment header
        logger.debug(f"\nSegment {seg_idx + 1}: {seg_start} → {seg_target}")

        # Forward path: seg_start → seg_target (always executed)
        logger.debug(f"  Path 1 (Forward {seg_start}→{seg_target}):")
        path_forward, visit_order_fwd, edges_fwd, adj = find_path_0_to_8(
            start=seg_start, target=seg_target, R=R, C=C, diagonals=False,
            generator=generator, blocked_nodes=seg_blocked
        )

        if not path_forward:
            logger.debug(f"\n❌ Failed to find forward path for segment {seg_idx + 1}, skipping this run")
            return None, None, None, None, None

        if do_backtrack:
            # Backward path: seg_target → seg_start (backtracking)
            logger.debug(f"  Path 2 (Backward {seg_target}→{seg_start}):")
            path_backward, visit_order_bwd, edges_bwd, adj = find_path_0_to_8(
                start=seg_target, target=seg_start, R=R, C=C, diagonals=False,
                generator=generator, blocked_nodes=seg_blocked
            )

            if not path_backward:
                logger.debug(f"\n❌ Failed to find backward path for segment {seg_idx + 1}, skipping this run")
                return None, None, None, None, None

            # Forward path again: seg_start → seg_target
            logger.debug(f"  Path 3 (Forward {seg_start}→{seg_target}):")
            path_forward2, visit_order_fwd2, edges_fwd2, adj = find_path_0_to_8(
                start=seg_start, target=seg_target, R=R, C=C, diagonals=False,
                generator=generator, blocked_nodes=seg_blocked
            )

            if not path_forward2:
                logger.debug(f"\n❌ Failed to find second forward path for segment {seg_idx + 1}, skipping this run")
                return None, None, None, None, None

            # Combine: forward + backward + forward (removing duplicate nodes at connection points)
            seg_combined = path_forward + path_backward[1:] + path_forward2[1:]
            seg_edges = edges_fwd + edges_bwd + edges_fwd2

            logger.debug(f"  → Segment generated 3 paths (with backtracking)")
        else:
            # No backtracking: just use forward path
            seg_combined = path_forward
            seg_edges = edges_fwd

            logger.debug(f"  → Segment generated 1 path (no backtracking)")

        all_paths.append(seg_combined)
        all_edges.extend(seg_edges)

    # Combine all paths (remove duplicate waypoint nodes between segments)
    combined_path = all_paths[0]
    for path_seg in all_paths[1:]:
        combined_path = combined_path + path_seg[1:]  # Skip first element (waypoint)

    combined_edges = all_edges

    logger.debug(f"\n✅ Final combined path: {' → '.join(map(str, combined_path))}")

    # Return data separately (not as tuple)
    return combined_path, combined_edges, adj, waypoints[0], waypoints[-1]


def visualize_single_path(path, edges_used, adj, start, target, R, C, blocked_nodes, run_id, segment_info):
    """
    Visualize a single DFS path

    Args:
        path: list of nodes in the path
        edges_used: list of edges used in the path
        adj: adjacency dictionary
        start: start node
        target: target node
        R: number of rows
        C: number of columns
        blocked_nodes: set of blocked node indices
        run_id: run identifier for file naming
        segment_info: string describing the segment selection (e.g., "segments[0:2]")
    """
    if not path:
        logger.debug("❌ No path to visualize!")
        return

    # Convert blocked_nodes to set for O(1) lookup
    if blocked_nodes is None:
        blocked_nodes = set()
    else:
        blocked_nodes = set(blocked_nodes)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    ax.set_xlim(-0.5, C - 0.5)
    ax.set_ylim(-0.5, R - 0.5)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_title(f'Run {run_id}: {segment_info}\nPath: {start} → {target}',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    ax.grid(True, alpha=0.3)

    # Draw all possible edges (adjacency relationships) - very light
    for node, neighbors in adj.items():
        r1, c1 = index_to_coord(node, C)
        for neighbor in neighbors:
            r2, c2 = index_to_coord(neighbor, C)
            if node < neighbor:
                ax.plot([c1, c2], [r1, r2], 'gray', alpha=0.15, linewidth=1, zorder=1)

    # Draw path edges in blue
    color = 'blue'
    for i in range(len(path) - 1):
        node = path[i]
        neighbor = path[i + 1]
        r1, c1 = index_to_coord(node, C)
        r2, c2 = index_to_coord(neighbor, C)

        ax.plot([c1, c2], [r1, r2], color=color, linewidth=3, alpha=0.7, zorder=3)

        # Add arrows
        dx, dy = c2 - c1, r2 - r1
        ax.arrow(c1, r1, dx * 0.6, dy * 0.6,
                head_width=0.12, head_length=0.08,
                fc=color, ec=color, alpha=0.7, zorder=3)

    # Draw all nodes
    for idx in range(R * C):
        r, c = index_to_coord(idx, C)

        # Check if node is blocked
        if idx in blocked_nodes:
            node_color = 'dimgray'
        else:
            if idx == start:
                node_color = 'lightgreen'
            elif idx == target:
                node_color = 'lightcoral'
            elif idx in path:
                node_color = 'lightyellow'
            else:
                node_color = 'lightgray'

        circle = plt.Circle((c, r), 0.35, color=node_color, ec='black', linewidth=2, zorder=2)
        ax.add_patch(circle)
        ax.text(c, r, str(idx), ha='center', va='center',
                fontsize=16, fontweight='bold',
                color='white' if idx in blocked_nodes else 'black', zorder=10)

        # Add X mark for blocked nodes
        if idx in blocked_nodes:
            ax.plot([c - 0.2, c + 0.2], [r - 0.2, r + 0.2], 'r', linewidth=3, zorder=11)
            ax.plot([c - 0.2, c + 0.2], [r + 0.2, r - 0.2], 'r', linewidth=3, zorder=11)

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen',
               markersize=12, label='Start Node', markeredgecolor='black', markeredgewidth=2),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcoral',
               markersize=12, label='Target Node', markeredgecolor='black', markeredgewidth=2),
    ]

    # Add blocked node indicator to legend if there are blocked nodes
    if blocked_nodes:
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor='dimgray',
                   markersize=12, label='Blocked Node', markeredgecolor='black', markeredgewidth=2)
        )

    ax.legend(handles=legend_elements, loc='upper left', fontsize=10, bbox_to_anchor=(1.02, 1))

    plt.tight_layout()

    # Save with run-specific filename
    output_path = f'/Users/fuhongze/Desktop/robotic/verl/9grid/grid_path_run_{run_id}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.debug(f"✅ Visualization saved to: {output_path}")
    plt.close(fig)  # Close figure to free memory


if __name__ == "__main__":
    print("=" * 60)
    print("3x3 Grid: Multiple DFS Path Findings with Randomization")
    print("=" * 60)

    # Configuration
    num_runs = 10  # You can change this number

    # # Define segments as (start, end, blocked_nodes) tuples
    segments = [
        (7, 9, [8, 3, 10, 17]),           # Segment 1: 7 → 9
        (9, 11, [1, 8, 15, 10, 5, 12, 19]), # Segment 2: 9 → 11
        (11, 13, [3, 10, 17, 12])           # Segment 3: 11 → 13
    ]

    segment_2 = [
        (13, 11, [3, 10, 17, 12]),  # Segment 3: 11 → 13
        (11, 9, [1, 8, 15, 10, 5, 12, 19]),  # Segment 2: 9 → 11
        (9, 7, [8, 3, 10, 17]),           # Segment 1: 7 → 9
    ]


    R, C = 3, 7

    # Run multiple times and visualize each separately
    for i in range(num_runs):
        # Create generator with specific seed for this run
        generator = torch.Generator()
        generator.manual_seed(i)

        # Randomly choose between segments and segment_2
        segment_choice = torch.randint(0, 2, (1,), generator=generator).item()
        chosen_segments = segments if segment_choice == 0 else segment_2
        segment_name = "segments" if segment_choice == 0 else "segment_2"

        # Randomly select a continuous slice from chosen segments
        # Possible slices: [0:1], [1:2], [2:3], [0:2], [1:3], [0:3]
        num_segments = len(chosen_segments)
        start_idx = torch.randint(0, num_segments, (1,), generator=generator).item()
        end_idx = torch.randint(start_idx + 1, num_segments + 1, (1,), generator=generator).item()

        selected_segments = chosen_segments[start_idx:end_idx]
        segment_info = f"{segment_name}[{start_idx}:{end_idx}]"

        # Print selection
        print(f"\n🎲 Selected: {segment_info}")
        print(f"   Segments to use:")
        for idx, (seg_start, seg_target, seg_blocked) in enumerate(selected_segments, start=start_idx):
            print(f"     [{idx}] {seg_start} → {seg_target}, blocked: {seg_blocked}")

        # Collect all blocked nodes for visualization
        all_blocked_nodes = set()
        for _, _, blocked in selected_segments:
            all_blocked_nodes.update(blocked)

        # Run path generation
        combined_path, combined_edges, adj, start_node, end_node = run_path_generation(
            run_id=i+1,
            generator=generator,
            segments=selected_segments,
            R=R,
            C=C
        )

        # Visualize this run if generation was successful
        if combined_path is not None:
            print(f"\n{'=' * 60}")
            print(f"Visualizing Run {i+1}")
            print(f"{'=' * 60}")
            visualize_single_path(
                path=combined_path,
                edges_used=combined_edges,
                adj=adj,
                start=start_node,
                target=end_node,
                R=R,
                C=C,
                blocked_nodes=all_blocked_nodes,
                run_id=i+1,
                segment_info=segment_info
            )
            
