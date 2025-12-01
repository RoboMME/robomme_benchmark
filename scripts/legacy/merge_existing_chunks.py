#!/usr/bin/env python3
"""
Script to merge existing chunk directories into final HDF5 files.
This is for cleaning up datasets that were already generated but not merged.
"""

import os
import sys
import shutil
from pathlib import Path
from typing import List
import h5py
import argparse


def merge_chunk_directories(
    env_id: str,
    dataset_dir: Path,
    final_dataset_path: Path,
    cleanup: bool = True,
) -> None:
    """
    Merge all chunk directories for a specific environment into a single HDF5 file.

    Args:
        env_id: Environment ID (e.g., "BinFill_hard")
        dataset_dir: Directory containing chunk directories
        final_dataset_path: Path to the final merged HDF5 file
        cleanup: Whether to remove chunk directories after merging
    """
    # Find all chunk directories for this environment
    chunk_pattern = f"record_dataset_{env_id}_chunk_*_hdf5_files"
    chunk_dirs = sorted(dataset_dir.glob(chunk_pattern))

    if not chunk_dirs:
        print(f"No chunk directories found for {env_id}")
        return

    print(f"\n{'='*80}")
    print(f"Merging {len(chunk_dirs)} chunk directories for {env_id}")
    print(f"Output: {final_dataset_path}")
    print(f"{'='*80}\n")

    final_dataset_path.parent.mkdir(parents=True, exist_ok=True)

    total_episodes = 0

    with h5py.File(final_dataset_path, "a") as final_file:
        for chunk_dir in chunk_dirs:
            print(f"Processing: {chunk_dir.name}")

            # Get all h5 files in the chunk directory
            h5_files = sorted(chunk_dir.glob("*.h5"))

            if not h5_files:
                print(f"  Warning: No h5 files found in {chunk_dir}")
                continue

            print(f"  Found {len(h5_files)} episode files")

            # Merge each episode file
            for h5_file in h5_files:
                print(f"    - Merging {h5_file.name}")
                try:
                    with h5py.File(h5_file, "r") as episode_file:
                        for env_group_name, src_env_group in episode_file.items():
                            if env_group_name not in final_file:
                                final_file.copy(src_env_group, env_group_name)
                                print(f"      Created group: {env_group_name}")
                                continue

                            dest_env_group = final_file[env_group_name]
                            if not isinstance(dest_env_group, h5py.Group):
                                print(f"      Warning: {env_group_name} is not a group, skipping...")
                                continue

                            for episode_name in src_env_group.keys():
                                if episode_name in dest_env_group:
                                    print(f"      Warning: Episode {episode_name} already exists, overwriting...")
                                    del dest_env_group[episode_name]
                                src_env_group.copy(episode_name, dest_env_group, name=episode_name)
                                total_episodes += 1
                except Exception as e:
                    print(f"      Error processing {h5_file.name}: {e}")
                    continue

            # Clean up the chunk directory if requested
            if cleanup:
                try:
                    shutil.rmtree(chunk_dir)
                    print(f"  ✓ Cleaned up: {chunk_dir.name}")
                except Exception as e:
                    print(f"  Warning: Failed to remove {chunk_dir}: {e}")

    print(f"\n✓ Successfully merged {total_episodes} episodes into {final_dataset_path}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Merge existing chunk directories into final HDF5 files"
    )
    parser.add_argument(
        "--env",
        "-e",
        type=str,
        required=True,
        help="Environment ID to merge (e.g., BinFill_hard)",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Directory containing chunk directories (default: parent directory)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output path for merged file (default: dataset-dir/record_dataset_<env>.h5)",
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Keep chunk directories after merging (default: remove them)",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List all environments with chunk directories",
    )

    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()

    # List mode: show all environments with chunks
    if args.list:
        print(f"\nSearching for chunk directories in: {dataset_dir}\n")
        chunk_dirs = sorted(dataset_dir.glob("record_dataset_*_chunk_*_hdf5_files"))

        if not chunk_dirs:
            print("No chunk directories found.")
            return

        # Extract unique environment IDs
        env_ids = set()
        for chunk_dir in chunk_dirs:
            # Extract env_id from pattern: record_dataset_{env_id}_chunk_{num}_hdf5_files
            parts = chunk_dir.name.replace("record_dataset_", "").replace("_hdf5_files", "")
            env_id = "_".join(parts.split("_")[:-2])  # Remove "_chunk_{num}"
            env_ids.add(env_id)

        print(f"Found chunk directories for {len(env_ids)} environments:\n")
        for env_id in sorted(env_ids):
            env_chunks = list(dataset_dir.glob(f"record_dataset_{env_id}_chunk_*_hdf5_files"))
            print(f"  - {env_id}: {len(env_chunks)} chunks")

        print()
        return

    # Merge mode
    env_id = args.env
    output_path = args.output or dataset_dir / f"record_dataset_{env_id}.h5"

    merge_chunk_directories(
        env_id=env_id,
        dataset_dir=dataset_dir,
        final_dataset_path=output_path,
        cleanup=not args.no_cleanup,
    )


if __name__ == "__main__":
    main()
