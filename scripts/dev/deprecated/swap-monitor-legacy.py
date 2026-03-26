from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ....logging_utils import logger


DEFAULT_SWAP_CONTACT_FORCE_EPS = 1e-6


@dataclass
class SwapContactState:
    force_eps: float = DEFAULT_SWAP_CONTACT_FORCE_EPS
    swap_contact_detected: bool = False
    first_contact_step: Optional[int] = None
    contact_pairs: List[str] = field(default_factory=list)
    max_force_norm: float = 0.0
    max_force_pair: Optional[str] = None
    max_force_step: Optional[int] = None
    pair_max_force: Dict[str, float] = field(default_factory=dict)
    printed_pairs: set[str] = field(default_factory=set)


def new_swap_contact_state(force_eps: float = DEFAULT_SWAP_CONTACT_FORCE_EPS) -> SwapContactState:
    return SwapContactState(force_eps=float(force_eps))


def reset_swap_contact_state(state: SwapContactState) -> None:
    state.swap_contact_detected = False
    state.first_contact_step = None
    state.contact_pairs.clear()
    state.max_force_norm = 0.0
    state.max_force_pair = None
    state.max_force_step = None
    state.pair_max_force.clear()
    state.printed_pairs.clear()


def get_swap_contact_summary(state: Optional[SwapContactState]) -> Dict[str, Any]:
    if state is None:
        state = new_swap_contact_state()
    return {
        "swap_contact_detected": bool(state.swap_contact_detected),
        "first_contact_step": state.first_contact_step,
        "contact_pairs": list(state.contact_pairs),
        "max_force_norm": float(state.max_force_norm),
        "max_force_pair": state.max_force_pair,
        "max_force_step": state.max_force_step,
        "pair_max_force": {
            pair_name: float(force_norm)
            for pair_name, force_norm in sorted(state.pair_max_force.items())
        },
    }


def _is_monitoring_active(
    swap_schedule: Optional[Sequence[Tuple[Any, Any, Any, Any]]], timestep: int
) -> bool:
    for _, _, start_step, end_step in swap_schedule or []:
        if int(start_step) <= int(timestep) <= int(end_step):
            return True
    return False


def _default_pair_name(actors: Sequence[Any], idx_a: int, idx_b: int) -> str:
    actor_a = actors[idx_a]
    actor_b = actors[idx_b]
    name_a = getattr(actor_a, "name", f"actor_{idx_a}")
    name_b = getattr(actor_b, "name", f"actor_{idx_b}")
    return f"{name_a}<->{name_b}"


def _resolve_log_context(log_context: Optional[Dict[str, Any]], default_env_name: str):
    context = log_context or {}
    env_name = context.get("env") or default_env_name
    episode = context.get("episode")
    seed = context.get("seed")
    return env_name, episode, seed


def _force_norm(force_tensor: Any) -> Optional[float]:
    if isinstance(force_tensor, torch.Tensor):
        if force_tensor.numel() == 0:
            return None
        force_vector = force_tensor.reshape(-1, 3)[0].detach().cpu().to(torch.float64)
        return float(torch.linalg.vector_norm(force_vector).item())

    force_array = np.asarray(force_tensor, dtype=np.float64)
    if force_array.size == 0:
        return None
    return float(np.linalg.norm(force_array.reshape(-1, 3)[0]))


def detect_swap_contacts(
    *,
    scene: Any,
    actors: Optional[Sequence[Any]],
    swap_schedule: Optional[Sequence[Tuple[Any, Any, Any, Any]]],
    timestep: int,
    state: SwapContactState,
    log_context: Optional[Dict[str, Any]] = None,
    pair_name_fn: Optional[Callable[[Sequence[Any], int, int], str]] = None,
    default_env_name: str = "SwapEnv",
) -> None:
    if not _is_monitoring_active(swap_schedule, timestep):
        return

    if scene is None or actors is None or len(actors) < 2:
        return

    name_fn = pair_name_fn or _default_pair_name
    env_name, episode, seed = _resolve_log_context(log_context, default_env_name)

    for idx_a, idx_b in combinations(range(len(actors)), 2):
        actor_a = actors[idx_a]
        actor_b = actors[idx_b]
        pair_name = name_fn(actors, idx_a, idx_b)

        try:
            pair_force = scene.get_pairwise_contact_forces(actor_a, actor_b)
        except Exception as exc:
            logger.debug(
                "Failed to query swap contact force for %s at step %s: %s",
                pair_name,
                timestep,
                exc,
            )
            continue

        force_norm = _force_norm(pair_force)
        if force_norm is None or force_norm <= state.force_eps:
            continue

        previous_pair_max = state.pair_max_force.get(pair_name, 0.0)
        if force_norm > previous_pair_max:
            state.pair_max_force[pair_name] = force_norm

        if not state.swap_contact_detected:
            state.swap_contact_detected = True
            state.first_contact_step = int(timestep)

        if pair_name not in state.contact_pairs:
            state.contact_pairs.append(pair_name)

        if force_norm > state.max_force_norm:
            state.max_force_norm = force_norm
            state.max_force_pair = pair_name
            state.max_force_step = int(timestep)

        if pair_name in state.printed_pairs:
            continue

        print(
            "[SwapContact] "
            f"env={env_name} episode={episode} seed={seed} "
            f"step={int(timestep)} pair={pair_name} force={force_norm:.6f}"
        )
        state.printed_pairs.add(pair_name)