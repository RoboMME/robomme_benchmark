#!/usr/bin/env python3
"""Compare metadata json files between two dataset directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

RecordKey = Tuple[int, int, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare metadata between two directories and verify env_id/records "
            "in directory A correspond to directory B."
        )
    )
    parser.add_argument(
        "--dir-a",
        type=Path,
        default=Path("/data/hongzefu/dataset_generate"),
        help="Directory A (defaults to /data/hongzefu/dataset_generate)",
    )
    parser.add_argument(
        "--dir-b",
        type=Path,
        default=Path("/data/hongzefu/data_1206"),
        help="Directory B (defaults to /data/hongzefu/data_1206)",
    )
    parser.add_argument(
        "--diff-limit",
        type=int,
        default=10,
        help="Max number of sample record differences to print for each env_id.",
    )
    parser.add_argument(
        "--first-n-from-b",
        type=int,
        default=50,
        help="Check whether A contains the first N records from B for each shared env_id.",
    )
    return parser.parse_args()


def _normalize_record(record: dict) -> RecordKey:
    episode = int(record["episode"])
    seed = int(record["seed"])
    difficulty = str(record.get("difficulty", "")).strip().lower()
    return (episode, seed, difficulty)


def load_metadata_map(root_dir: Path) -> Dict[str, List[RecordKey]]:
    metadata_map: Dict[str, List[RecordKey]] = {}
    for metadata_path in sorted(root_dir.glob("record_dataset_*_metadata.json")):
        with metadata_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        file_env_id = metadata_path.name.replace("record_dataset_", "").replace("_metadata.json", "")
        env_id = str(payload.get("env_id") or file_env_id)

        records_raw = payload.get("records", [])
        if not isinstance(records_raw, list):
            raise ValueError(f"Invalid records list in: {metadata_path}")

        records: List[RecordKey] = []
        for item in records_raw:
            if not isinstance(item, dict):
                continue
            if "episode" not in item or "seed" not in item:
                continue
            records.append(_normalize_record(item))

        # Deduplicate while keeping deterministic order.
        dedup_records = sorted(set(records), key=lambda x: (x[0], x[1], x[2]))
        metadata_map[env_id] = dedup_records

    return metadata_map


def format_records(records: Iterable[RecordKey], limit: int) -> List[str]:
    lines: List[str] = []
    for episode, seed, difficulty in sorted(records)[:limit]:
        lines.append(f"(episode={episode}, seed={seed}, difficulty='{difficulty}')")
    return lines


def main() -> int:
    args = parse_args()

    dir_a = args.dir_a
    dir_b = args.dir_b

    if not dir_a.exists():
        raise FileNotFoundError(f"Directory A not found: {dir_a}")
    if not dir_b.exists():
        raise FileNotFoundError(f"Directory B not found: {dir_b}")

    map_a = load_metadata_map(dir_a)
    map_b = load_metadata_map(dir_b)

    envs_a = set(map_a)
    envs_b = set(map_b)

    missing_in_b = sorted(envs_a - envs_b)
    extra_in_b = sorted(envs_b - envs_a)
    shared_envs = sorted(envs_a & envs_b)

    print(f"Directory A: {dir_a}")
    print(f"Directory B: {dir_b}")
    print(f"A env_id count: {len(envs_a)}")
    print(f"B env_id count: {len(envs_b)}")

    print("\n[env_id check]")
    if missing_in_b:
        print("Missing in B (exists in A but not in B):")
        for env_id in missing_in_b:
            print(f"  - {env_id}")
    else:
        print("All env_id in A exist in B. OK")

    print("\n[extra env_id in B]")
    if extra_in_b:
        for env_id in extra_in_b:
            print(f"  - {env_id}")
    else:
        print("No extra env_id in B.")

    print("\n[record comparison for shared env_id]")
    all_subset_ok = True

    for env_id in shared_envs:
        rec_a_set = set(map_a[env_id])
        rec_b_set = set(map_b[env_id])

        only_a = rec_a_set - rec_b_set
        only_b = rec_b_set - rec_a_set
        subset_ok = len(only_a) == 0
        all_subset_ok = all_subset_ok and subset_ok

        print(
            f"- {env_id}: A={len(rec_a_set)} | B={len(rec_b_set)} | "
            f"A-B={len(only_a)} | B-A={len(only_b)} | subset(A in B)={subset_ok}"
        )

        if only_a:
            print("  sample A-B:")
            for line in format_records(only_a, args.diff_limit):
                print(f"    {line}")

        if only_b:
            print("  sample B-A:")
            for line in format_records(only_b, args.diff_limit):
                print(f"    {line}")

    print(f"\n[first {args.first_n_from_b} records from B -> missing in A]")
    first_n_all_ok = True
    for env_id in shared_envs:
        rec_a_set = set(map_a[env_id])
        rec_b_first_n = map_b[env_id][: args.first_n_from_b]
        missing_first_n = [record for record in rec_b_first_n if record not in rec_a_set]

        if missing_first_n:
            first_n_all_ok = False
            print(
                f"- {env_id}: missing {len(missing_first_n)} / {len(rec_b_first_n)} "
                f"from first {args.first_n_from_b}"
            )
            for line in format_records(missing_first_n, args.diff_limit):
                print(f"    {line}")
        else:
            print(f"- {env_id}: OK (all first {len(rec_b_first_n)} records from B exist in A)")

    ok = (len(missing_in_b) == 0) and all_subset_ok and first_n_all_ok
    print("\n[final]")
    if ok:
        print("PASS: env_id and A records all correspond to B, including first-N coverage check.")
        return 0

    print("FAIL: found missing env_id/records mismatch, or missing first-N records from B in A.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
