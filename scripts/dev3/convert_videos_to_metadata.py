"""
Convert video filenames in videos-success/ to val-format metadata JSON files.

Filename pattern:
  {TASK}_ep{N}_seed{M}_(FailRecoverZ|FailRecoverXY_)?{difficulty}_{...}.mp4

Episodes may not be continuous (e.g. VideoRepick missing ep2, ep3).
This script renumbers them 0, 1, 2, ... (backfills gaps) and emits one
record_dataset_{TASK}_metadata.json per task.
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

DIFFICULTY_QUOTA = {"easy": 10, "medium": 20, "hard": 20}

FILENAME_RE = re.compile(
    r"^(?P<task>[^_]+)_ep(?P<episode>\d+)_seed(?P<seed>\d+)"
    r"_(?:FailRecover\w+_)?(?P<difficulty>easy|medium|hard)_"
)

REPO_ROOT = Path(__file__).resolve().parents[2]
#DEFAULT_VIDEOS_DIR = REPO_ROOT / "runs/replay_videos/videos-success"
DEFAULT_VIDEOS_DIR="/data/hongzefu/robomme_benchmark_cvpr2026-heldoutSeed/videos-success/selected"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runs/replay_videos/heldout_metadata"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--videos-dir",
        type=Path,
        default=DEFAULT_VIDEOS_DIR,
        help="Directory containing success replay .mp4 files",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write record_dataset_*.json files",
    )
    return p.parse_args()


def parse_video_files(videos_dir: Path) -> dict[str, list[dict]]:
    """Return {task: [{"original_episode": N, "seed": M, "difficulty": "..."}]}."""
    task_records: dict[str, list[dict]] = defaultdict(list)
    unmatched = []

    for mp4 in sorted(videos_dir.glob("*.mp4")):
        m = FILENAME_RE.match(mp4.name)
        if not m:
            unmatched.append(mp4.name)
            continue
        task_records[m.group("task")].append(
            {
                "original_episode": int(m.group("episode")),
                "seed": int(m.group("seed")),
                "difficulty": m.group("difficulty"),
            }
        )

    if unmatched:
        print(f"[WARN] {len(unmatched)} file(s) did not match pattern:", file=sys.stderr)
        for name in unmatched:
            print(f"  {name}", file=sys.stderr)

    return task_records


def build_metadata(task: str, raw_records: list[dict]) -> tuple[dict, list[dict]]:
    """Per-difficulty sort + truncate, then assign global episode indices.

    Each difficulty group is sorted by original_episode independently.
    Episodes are truncated to DIFFICULTY_QUOTA before merging.
    Final episode indices are 0..N-1 across easy→medium→hard order.

    Returns (metadata, mapping) where mapping is a list of
    {"new_episode", "original_episode", "seed", "difficulty"} rows in the
    same order as metadata["records"], for printing the new→original
    episode correspondence.
    """
    by_difficulty: dict[str, list[dict]] = defaultdict(list)
    for r in raw_records:
        by_difficulty[r["difficulty"]].append(r)

    records = []
    mapping: list[dict] = []
    for difficulty in ("easy", "medium", "hard"):
        quota = DIFFICULTY_QUOTA[difficulty]
        group = sorted(by_difficulty.get(difficulty, []), key=lambda r: r["original_episode"])
        if len(group) < quota:
            print(
                f"[WARN] {task}/{difficulty}: only {len(group)} episodes available (quota {quota})",
                file=sys.stderr,
            )
        for r in group[:quota]:
            new_ep = len(records)
            records.append(
                {
                    "task": task,
                    "episode": new_ep,
                    "seed": r["seed"],
                    "difficulty": r["difficulty"],
                }
            )
            mapping.append(
                {
                    "new_episode": new_ep,
                    "original_episode": r["original_episode"],
                    "seed": r["seed"],
                    "difficulty": r["difficulty"],
                }
            )

    metadata = {
        "env_id": task,
        "record_count": len(records),
        "records": records,
    }
    return metadata, mapping


def print_mapping(task: str, mapping: list[dict]) -> None:
    """Print the new_episode ← original_episode correspondence as a table."""
    header = f"  {'new_ep':>6}  {'orig_ep':>7}  {'seed':>10}  difficulty"
    print(f"  --- {task}: new_episode ← video filename ep ---")
    print(header)
    print(f"  {'-' * 6}  {'-' * 7}  {'-' * 10}  {'-' * 10}")
    for row in mapping:
        print(
            f"  {row['new_episode']:>6}  {row['original_episode']:>7}  "
            f"{row['seed']:>10}  {row['difficulty']}"
        )


def main():
    args = parse_args()

    if not args.videos_dir.is_dir():
        sys.exit(f"[ERROR] videos-dir does not exist: {args.videos_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    task_records = parse_video_files(args.videos_dir)
    if not task_records:
        sys.exit("[ERROR] No matching .mp4 files found.")

    for task in sorted(task_records):
        raw = task_records[task]
        metadata, mapping = build_metadata(task, raw)

        out_path = args.output_dir / f"record_dataset_{task}_metadata.json"
        out_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

        print(f"[OK] {task}: {len(mapping)} episodes → {out_path.name}")
        print_mapping(task, mapping)
        print()

    print(f"Wrote {len(task_records)} metadata file(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
