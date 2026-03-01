#!/usr/bin/env python3
"""Compare how v3 and v4 replay pipelines read multi_choice actions.

v3 source:
- EpisodeDatasetResolver.get_step("multi_choice", step)

v4-noresolver source:
- scripts.dataset_replay._build_action_sequence(..., "multi_choice")
- then _parse_oracle_command() in replay loop
"""

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _load_episode_dataset_resolver_cls():
    resolver_path = SRC_ROOT / "robomme" / "env_record_wrapper" / "episode_dataset_resolver.py"
    spec = importlib.util.spec_from_file_location(
        "episode_dataset_resolver_direct",
        resolver_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load resolver module from {resolver_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    resolver_cls = getattr(module, "EpisodeDatasetResolver", None)
    if resolver_cls is None:
        raise RuntimeError(f"EpisodeDatasetResolver not found in {resolver_path}")
    return resolver_cls


EpisodeDatasetResolver = _load_episode_dataset_resolver_cls()

DEFAULT_ENV_ID = "PatternLock"
DEFAULT_DATASET_ROOT = "/data/hongzefu/data_0226-test"


def _parse_oracle_command_v4(choice_action: Optional[Any]) -> Optional[dict[str, Any]]:
    """Exact validation logic used in evaluate_dataset_replay-parallelv4-noresolver.py."""
    if not isinstance(choice_action, dict):
        return None
    choice = choice_action.get("choice")
    if not isinstance(choice, str) or not choice.strip():
        return None
    point = choice_action.get("point")
    if not isinstance(point, (list, tuple, np.ndarray)) or len(point) != 2:
        return None
    return choice_action


def _is_video_demo_v4(ts: h5py.Group) -> bool:
    info = ts.get("info")
    if info is None or "is_video_demo" not in info:
        return False
    return bool(np.reshape(np.asarray(info["is_video_demo"][()]), -1)[0])


def _is_subgoal_boundary_v4(ts: h5py.Group) -> bool:
    info = ts.get("info")
    if info is None or "is_subgoal_boundary" not in info:
        return False
    return bool(np.reshape(np.asarray(info["is_subgoal_boundary"][()]), -1)[0])


def _decode_h5_str_v4(raw: Any) -> str:
    if isinstance(raw, np.ndarray):
        raw = raw.flatten()[0]
    if isinstance(raw, (bytes, np.bytes_)):
        raw = raw.decode("utf-8")
    return raw


def _build_multi_choice_sequence_v4(episode_data: h5py.Group) -> list[Any]:
    """
    Re-implementation of dataset_replay._build_action_sequence(..., \"multi_choice\")
    without importing cv2/imageio/torch dependencies.
    """
    timestep_keys = sorted(
        (k for k in episode_data.keys() if k.startswith("timestep_")),
        key=lambda k: int(k.split("_")[1]),
    )

    out: list[Any] = []
    for key in timestep_keys:
        ts = episode_data[key]
        if _is_video_demo_v4(ts):
            continue

        action_grp = ts.get("action")
        if action_grp is None:
            continue
        if not _is_subgoal_boundary_v4(ts):
            continue
        if "choice_action" not in action_grp:
            continue

        raw = _decode_h5_str_v4(action_grp["choice_action"][()])
        try:
            out.append(json.loads(raw))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
    return out


def _resolve_h5_path(env_id: str, dataset_root: Optional[str], h5_path: Optional[str]) -> Path:
    if h5_path:
        return Path(h5_path)
    if not dataset_root:
        raise ValueError("Either --h5_path or --dataset_root must be provided")
    return Path(dataset_root) / f"record_dataset_{env_id}.h5"


def _episode_indices(data: h5py.File) -> list[int]:
    return sorted(
        int(m.group(1))
        for key in data.keys()
        for m in [re.match(r"episode_(\d+)$", key)]
        if m
    )


def _parse_episode_filter(raw: Optional[str], all_eps: list[int]) -> list[int]:
    if not raw:
        return all_eps

    selected: set[int] = set()
    for token in [x.strip() for x in raw.split(",") if x.strip()]:
        if "-" in token:
            lo_s, hi_s = token.split("-", 1)
            lo = int(lo_s)
            hi = int(hi_s)
            if lo > hi:
                lo, hi = hi, lo
            selected.update(range(lo, hi + 1))
        else:
            selected.add(int(token))

    return [ep for ep in all_eps if ep in selected]


def _canonical_command(cmd: Any) -> str:
    """Stable string form for diffing and readable output."""
    try:
        return json.dumps(cmd, ensure_ascii=False, sort_keys=True)
    except TypeError:
        if isinstance(cmd, dict):
            safe = {
                str(k): (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in cmd.items()
            }
            return json.dumps(safe, ensure_ascii=False, sort_keys=True)
        return repr(cmd)


def _read_v4_commands(episode_group: h5py.Group) -> tuple[list[Any], list[dict[str, Any]], int]:
    raw_list = _build_multi_choice_sequence_v4(episode_group)
    parsed_list: list[dict[str, Any]] = []
    skipped = 0

    for item in raw_list:
        parsed = _parse_oracle_command_v4(item)
        if parsed is None:
            skipped += 1
            continue
        parsed_list.append(parsed)

    return raw_list, parsed_list, skipped


def _read_v3_commands(env_id: str, episode: int, dataset_ref: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with EpisodeDatasetResolver(
        env_id=env_id,
        episode=episode,
        dataset_directory=dataset_ref,
    ) as resolver:
        step = 0
        while True:
            cmd = resolver.get_step("multi_choice", step)
            if cmd is None:
                break
            if isinstance(cmd, dict):
                out.append(cmd)
            step += 1
    return out


def compare_episode(
    env_id: str,
    episode: int,
    episode_group: h5py.Group,
    dataset_ref: str,
    max_show: int,
) -> None:
    v4_raw, v4_effective, v4_skipped = _read_v4_commands(episode_group)
    v3_resolver = _read_v3_commands(env_id=env_id, episode=episode, dataset_ref=dataset_ref)

    print(f"\n=== episode_{episode} ===")
    print(
        "counts: "
        f"v4_raw={len(v4_raw)}, "
        f"v4_effective={len(v4_effective)} (skipped_by_parse={v4_skipped}), "
        f"v3_resolver={len(v3_resolver)}"
    )

    v4_effective_c = [_canonical_command(x) for x in v4_effective]
    v3_c = [_canonical_command(x) for x in v3_resolver]

    if v4_effective_c == v3_c:
        print("effective sequence compare: SAME")
    else:
        print("effective sequence compare: DIFFERENT")
        max_len = max(len(v4_effective_c), len(v3_c))
        shown = 0
        for idx in range(max_len):
            left = v4_effective_c[idx] if idx < len(v4_effective_c) else "<MISSING>"
            right = v3_c[idx] if idx < len(v3_c) else "<MISSING>"
            if left == right:
                continue
            print(f"  idx={idx}")
            print(f"    v4_effective: {left}")
            print(f"    v3_resolver : {right}")
            shown += 1
            if shown >= max_show:
                remaining = max_len - idx - 1
                if remaining > 0:
                    print(f"  ... more differences omitted ({remaining} remaining positions)")
                break

    print(f"sample v4_raw (first {max_show}):")
    for i, item in enumerate(v4_raw[:max_show]):
        print(f"  [{i}] {_canonical_command(item)}")

    print(f"sample v4_effective (first {max_show}):")
    for i, item in enumerate(v4_effective[:max_show]):
        print(f"  [{i}] {_canonical_command(item)}")

    print(f"sample v3_resolver (first {max_show}):")
    for i, item in enumerate(v3_resolver[:max_show]):
        print(f"  [{i}] {_canonical_command(item)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare multi_choice read results between "
            "evaluate_dataset_replay-parallelv3 and parallelv4-noresolver."
        )
    )
    parser.add_argument(
        "--env_id",
        type=str,
        default=DEFAULT_ENV_ID,
        help=f"Task/env id. Default: {DEFAULT_ENV_ID}",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=DEFAULT_DATASET_ROOT,
        help=(
            "Directory that contains record_dataset_<env_id>.h5. "
            f"Default: {DEFAULT_DATASET_ROOT}"
        ),
    )
    parser.add_argument(
        "--h5_path",
        type=str,
        default=None,
        help="Direct path to .h5 file (overrides --dataset_root)",
    )
    parser.add_argument(
        "--episodes",
        type=str,
        default=0,
        help="Episode filter, e.g. '0,3,8-10'. Default: all episodes in h5",
    )
    parser.add_argument(
        "--max_show",
        type=int,
        default=50,
        help="Max number of diff/sample rows per episode",
    )
    args = parser.parse_args()

    h5_file = _resolve_h5_path(args.env_id, args.dataset_root, args.h5_path)
    if not h5_file.exists():
        raise FileNotFoundError(f"h5 file not found: {h5_file}")

    dataset_ref = str(h5_file) if h5_file.suffix == ".h5" else str(h5_file.parent)

    print(f"env_id={args.env_id}")
    print(f"h5={h5_file}")

    with h5py.File(h5_file, "r") as data:
        all_eps = _episode_indices(data)
        selected_eps = _parse_episode_filter(args.episodes, all_eps)

        if not selected_eps:
            print("No episodes selected.")
            return

        print(f"episodes={selected_eps}")
        for ep in selected_eps:
            key = f"episode_{ep}"
            if key not in data:
                print(f"\n=== episode_{ep} ===")
                print("missing in h5, skip")
                continue
            compare_episode(
                env_id=args.env_id,
                episode=ep,
                episode_group=data[key],
                dataset_ref=dataset_ref,
                max_show=args.max_show,
            )


if __name__ == "__main__":
    main()
