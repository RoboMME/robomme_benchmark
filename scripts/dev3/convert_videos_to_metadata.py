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

FILENAME_RE = re.compile(
    r"^(?P<task>[^_]+)_ep(?P<episode>\d+)_seed(?P<seed>\d+)"
    r"_(?:FailRecover\w+_)?(?P<difficulty>easy|medium|hard)_"
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VIDEOS_DIR = REPO_ROOT / "runs/replay_videos/videos-success"
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


def build_metadata(task: str, raw_records: list[dict]) -> dict:
    """Sort by original_episode, renumber 0‥N-1, return val-format dict."""
    sorted_records = sorted(raw_records, key=lambda r: r["original_episode"])

    records = []
    for new_ep, r in enumerate(sorted_records):
        records.append(
            {
                "task": task,
                "episode": new_ep,
                "seed": r["seed"],
                "difficulty": r["difficulty"],
            }
        )

    return {
        "env_id": task,
        "record_count": len(records),
        "records": records,
    }


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
        metadata = build_metadata(task, raw)

        out_path = args.output_dir / f"record_dataset_{task}_metadata.json"
        out_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

        orig_eps = sorted(r["original_episode"] for r in raw)
        new_eps = list(range(len(raw)))
        gap = orig_eps != new_eps
        gap_note = f"  (gaps filled: {orig_eps} → {new_eps})" if gap else ""
        print(f"[OK] {task}: {len(raw)} episodes → {out_path.name}{gap_note}")

    print(f"\nWrote {len(task_records)} metadata file(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
