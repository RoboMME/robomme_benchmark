from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _normalize_pair(idx1: int, idx2: int) -> tuple[int, int]:
    return (idx1, idx2) if idx1 < idx2 else (idx2, idx1)


def _candidate_position(candidate: Any) -> np.ndarray | None:
    if candidate is None:
        return None

    try:
        pose = candidate.pose if hasattr(candidate, "pose") else candidate.get_pose()
        position = pose.p
    except Exception:
        return None

    if isinstance(position, torch.Tensor):
        position = position.detach().cpu().numpy()

    try:
        position_array = np.asarray(position, dtype=np.float32).reshape(-1)
    except Exception:
        return None

    if position_array.size < 2:
        return None

    if not np.all(np.isfinite(position_array[:2])):
        return None

    if position_array.size < 3:
        padded = np.zeros(3, dtype=np.float32)
        padded[: position_array.size] = position_array
        return padded

    if not np.all(np.isfinite(position_array[:3])):
        return None

    return position_array[:3]


def extract_candidate_positions(candidates) -> list[np.ndarray | None]:
    return [_candidate_position(candidate) for candidate in candidates]


def _build_nearest_neighbors(
    positions: list[np.ndarray | None], neighbor_count: int
) -> dict[int, list[tuple[int, float]]]:
    valid_indices = [idx for idx, position in enumerate(positions) if position is not None]
    neighbor_map: dict[int, list[tuple[int, float]]] = {}

    for idx in valid_indices:
        origin = positions[idx]
        assert origin is not None
        distances: list[tuple[int, float]] = []
        for other_idx in valid_indices:
            if other_idx == idx:
                continue
            other = positions[other_idx]
            assert other is not None
            distance = float(np.linalg.norm(origin[:2] - other[:2]))
            distances.append((other_idx, distance))
        distances.sort(key=lambda item: (item[1], item[0]))
        neighbor_map[idx] = distances[:neighbor_count]

    return neighbor_map


def _resolve_generator(generator: torch.Generator | None) -> torch.Generator:
    if generator is not None:
        return generator

    fallback_generator = torch.Generator()
    fallback_generator.manual_seed(0)
    return fallback_generator


def select_dynamic_swap_pair(
    candidates,
    generator: torch.Generator | None,
    previous_pair: tuple[int, int] | None = None,
    neighbor_count: int = 2,
) -> dict[str, Any] | None:
    positions = extract_candidate_positions(candidates)
    neighbor_map = _build_nearest_neighbors(positions, max(1, int(neighbor_count)))
    valid_indices = [idx for idx, neighbors in neighbor_map.items() if neighbors]
    if not valid_indices:
        return None

    torch_generator = _resolve_generator(generator)
    anchor_idx = valid_indices[
        int(torch.randint(0, len(valid_indices), (1,), generator=torch_generator).item())
    ]
    anchor_neighbors = neighbor_map[anchor_idx]
    partner_pos = int(
        torch.randint(0, len(anchor_neighbors), (1,), generator=torch_generator).item()
    )
    partner_idx, partner_distance = anchor_neighbors[partner_pos]
    pair_key = _normalize_pair(anchor_idx, partner_idx)
    if previous_pair is None or pair_key != previous_pair:
        return {
            "idx1": anchor_idx,
            "idx2": partner_idx,
            "distance": partner_distance,
            "pair_key": pair_key,
        }

    fallback_candidates: list[tuple[float, int, int]] = []
    for idx, neighbors in neighbor_map.items():
        for other_idx, distance in neighbors:
            normalized_pair = _normalize_pair(idx, other_idx)
            if normalized_pair == previous_pair:
                continue
            fallback_candidates.append((distance, normalized_pair[0], normalized_pair[1]))

    if fallback_candidates:
        fallback_candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        distance, idx1, idx2 = fallback_candidates[0]
        return {
            "idx1": idx1,
            "idx2": idx2,
            "distance": distance,
            "pair_key": (idx1, idx2),
        }

    return {
        "idx1": anchor_idx,
        "idx2": partner_idx,
        "distance": partner_distance,
        "pair_key": pair_key,
    }


__all__ = ["extract_candidate_positions", "select_dynamic_swap_pair"]
